"""Fixture-backed "backend": policyholder lookup, claims, guidance, consent, email outbox.

Everything the agent is allowed to say about a claim comes from here. The LLM never
reads these files directly; the SOP layer decides which slice of this data is
visible in each phase.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

FIXTURES_DIR = Path(os.environ.get("FIXTURES_DIR", Path(__file__).resolve().parent.parent / "fixtures"))


def _load(name: str) -> Any:
    with open(FIXTURES_DIR / name, encoding="utf-8") as f:
        return json.load(f)


def today() -> date:
    """Business date. DEMO_TODAY lets a tester pin the date (e.g. to be inside an appeal window)."""
    override = os.environ.get("DEMO_TODAY")
    if override:
        return date.fromisoformat(override)
    # The business date, not the server's: a container runs in UTC, the claims desk does not.
    return datetime.now(ZoneInfo(os.environ.get("BUSINESS_TZ", "America/Los_Angeles"))).date()


@dataclass
class Repository:
    policyholders: list[dict] = field(default_factory=lambda: _load("policyholders.json"))
    claims: list[dict] = field(default_factory=lambda: _load("claims.json"))
    representatives: list[dict] = field(default_factory=lambda: _load("representatives.json"))
    guideline: dict = field(default_factory=lambda: _load("required_document_guideline.json"))
    consent_scenarios: dict = field(default_factory=lambda: _load("consent_scenarios.json"))
    claim_schema: dict = field(default_factory=lambda: _load("claim_schema.json"))

    # ---- policyholders -------------------------------------------------
    def policyholder(self, party_id: str) -> dict | None:
        return next((p for p in self.policyholders if p["party_id"] == party_id), None)

    # ---- claims ----------------------------------------------------------
    def claims_for(self, party_id: str) -> list[dict]:
        rows = [c for c in self.claims if c["party_id"] == party_id]
        return sorted(rows, key=lambda c: c["created_at"], reverse=True)

    def claim(self, party_id: str, case_id: str) -> dict | None:
        # Scoped by party: a verified caller can never pull another member's claim.
        return next((c for c in self.claims_for(party_id) if c["case_id"].upper() == case_id.upper()), None)

    # ---- representatives -----------------------------------------------
    def authorized_rep(self, party_id: str, rep_name: str | None) -> dict | None:
        if not rep_name:
            return None
        from .identity import norm_name

        for r in self.representatives:
            if r["buyer_party_id"] == party_id and norm_name(r["rep_name"]) == norm_name(rep_name):
                return r
        return None

    # ---- guidance --------------------------------------------------------
    def _doc_key(self, table: dict, doc: str) -> str | None:
        """Claims say 'pathology report'; guidance is keyed 'original pathology report'."""
        d = doc.lower().strip()
        if d in table:
            return d
        for k in table:
            if k == "default":
                continue
            if d in k or k in d:
                return k
        return None

    def claim_knowledge(self, claim: dict) -> dict:
        """Everything the agent may use to answer questions about one claim."""
        g = self.guideline
        docs = claim.get("documents_needed") or []
        doc_guidance, doc_alternatives = {}, {}
        for d in docs:
            k = self._doc_key(g["document_guidance"], d)
            if k:
                doc_guidance[d] = g["document_guidance"][k]["en"]
            k2 = self._doc_key(g["document_alternative_guidance"], d)
            doc_alternatives[d] = g["document_alternative_guidance"][k2 or "default"]["en"]

        settings = g["claim_followup_settings"]
        fill = {
            "case_id": claim["case_id"],
            "documents": _join(docs),
            "average_processing_time_after_submission": settings["average_processing_time_after_submission"]["en"],
        }
        deadline = claim.get("appeal_deadline")
        deadline_info = None
        if deadline:
            d = date.fromisoformat(deadline)
            days = (d - today()).days
            deadline_info = {
                "appeal_deadline": deadline,
                "today": today().isoformat(),
                "deadline_passed": days < 0,
                "days_remaining": max(days, 0),
            }
        deadline_passed = bool(deadline_info and deadline_info["deadline_passed"])

        followups = []
        for item in g["claim_followup_guidance"]:
            if item.get("requires_documents") and not docs:
                continue
            # "submit within a week" / "review restarts" guidance assumes the appeal window is open.
            if deadline_passed and item["topic"] in ("submission_timing", "processing_time_after_submission"):
                continue
            followups.append({
                "topic": item["topic"],
                "applies_to_intents": item.get("intent_hints", []),
                "text": item["en"].format(**fill),
            })
        after_deadline_rule = None
        if deadline_passed:
            after_deadline_rule = (f"The formal appeal window for {claim['case_id']} closed on {deadline}. Whether the "
                                   "missing documents can still be reviewed is something a claims representative has "
                                   "to confirm; do not promise a new review or quote submission deadlines.")

        record = {k: v for k, v in claim.items() if k != "party_id"}
        return {
            "claim": record,
            "amount_field_meanings": {k: v["description"] for k, v in self.claim_schema["field_descriptions"].items()},
            "appeal": deadline_info,
            "submission_guidance": {
                "general": g["default_guidance"]["en"],
                "case_type": g["case_type_guidance"].get(claim["case_type"], {}).get("en"),
                "per_document": doc_guidance,
                "if_document_unavailable": doc_alternatives,
            },
            "followup_guidance": followups,
            "after_deadline_rule": after_deadline_rule,
            "processing_time_after_submission": None if deadline_passed else fill["average_processing_time_after_submission"],
            "human_review_policy": settings["human_review_after_document_alternatives_exhausted"]["en"],
            "fallback_when_no_rule": g["claim_followup_fallback"]["en"],
        }

    # ---- consent (simulated) ---------------------------------------------
    def consent_sequence(self, scenario: str) -> list[str]:
        return self.consent_scenarios.get(scenario, self.consent_scenarios["default"])["status_sequence"]


def _join(items: list[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


def claim_brief(c: dict) -> dict:
    """The minimal description used to let a verified caller pick between claims."""
    return {
        "case_id": c["case_id"],
        "case_type": c["case_type"],
        "created_at": c["created_at"],
        "status": c["status"],
        "summary": c["summary"],
    }


class Outbox:
    """Simulated email delivery. Writes to outbox/ so a reviewer can inspect what was 'sent'."""

    def __init__(self, directory: Path | None = None):
        self.dir = directory or Path(os.environ.get("OUTBOX_DIR", Path(__file__).resolve().parent.parent / "outbox"))

    def send(self, to: str, subject: str, body: str, session_id: str) -> dict:
        msg = {
            "to": to,
            "subject": subject,
            "body": body,
            "session_id": session_id,
            "sent_at": datetime.now(ZoneInfo(os.environ.get("BUSINESS_TZ", "America/Los_Angeles"))).isoformat(timespec="seconds"),
            "transport": "simulated (written to outbox/)",
        }
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            path = self.dir / f"{session_id}-{datetime.now().strftime('%H%M%S')}.json"
            path.write_text(json.dumps(msg, indent=2), encoding="utf-8")
            msg["file"] = path.name
        except OSError:
            msg["file"] = None
        return msg


REPO = Repository()
