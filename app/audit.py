"""Append-only audit log: one JSON line per turn, with what the harness decided and why.

No caller text and no PII is written, only decisions: phase path, SOP steps, tools, guard
outcome, emotion/scope labels, latency and token usage. Disable with AUDIT_LOG=0.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger("audit")
_LOCK = threading.Lock()


def _path() -> Path:
    return Path(os.environ.get("AUDIT_DIR", Path(__file__).resolve().parent.parent / "logs")) / "audit.jsonl"


def audit(s, plan, sig, render_info: dict, usage: list, total_ms: int, degraded: bool) -> None:
    if os.environ.get("AUDIT_LOG", "1") == "0":
        return
    row = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "session": s.id,
        "turn": s.turn_count,
        "phase_path": plan.phase_path,
        "steps": plan.steps,
        "tools": [a["tool"] for a in plan.actions],
        "verified": s.verified,
        "caller_role": s.caller_role,
        "case_id": s.case_id,
        "scope": sig.scope,
        "emotion": sig.emotion,
        "understood_by": sig.source,
        "writer": render_info.get("writer"),
        "guard": render_info.get("guard") or [],
        "rewritten": "first_draft_rejected" in render_info,
        "degraded": degraded,
        "ms": total_ms,
        "tokens": sum((u.get("input_tokens") or 0) + (u.get("output_tokens") or 0) for u in usage),
    }
    try:
        p = _path()
        p.parent.mkdir(parents=True, exist_ok=True)
        with _LOCK, open(p, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
    except OSError as e:
        log.warning("audit log write failed: %s", e)
