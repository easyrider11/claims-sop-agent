"""Identity normalization and the VERIFY_ID gate. Pure functions, no LLM.

Rule (from the SOP): a caller is verified when at least 3 of the 5 PII fields
(full name, DOB, phone, email, SSN/national-ID last 4) match ONE policyholder record
and none of the fields they gave contradict that record. Policy number helps locate
the record but does not count toward the 3.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

PII_FIELDS = ("full_name", "dob", "phone", "email", "ssn_last4")
REQUIRED_MATCHES = 3
MAX_FAILED_ATTEMPTS = 3

FIELD_LABELS = {
    "full_name": "full name",
    "dob": "date of birth",
    "phone": "phone number",
    "email": "email address",
    "ssn_last4": "last 4 digits of your SSN (or national ID)",
    "policy_number": "policy number",
}


def norm_name(s: str | None) -> str:
    if not s:
        return ""
    s = re.sub(r"[^a-z\s'-]", " ", s.lower())
    return " ".join(s.split())


def norm_phone(s: str | None) -> str:
    if not s:
        return ""
    digits = re.sub(r"\D", "", s)
    if len(digits) == 10:
        digits = "1" + digits
    return digits


def norm_email(s: str | None) -> str:
    return (s or "").strip().lower()


def norm_last4(s: str | None) -> str:
    digits = re.sub(r"\D", "", s or "")
    return digits[-4:] if len(digits) >= 4 else ""


def norm_policy(s: str | None) -> str:
    return re.sub(r"[^A-Z0-9]", "", (s or "").upper())


def name_keys(s: str | None) -> set[str]:
    """Order-insensitive name keys. "Ya Wen Li", "Li Yawen", "Yawen Li" and "Li Ya Wen" all share one key,
    so a surname-first or differently spaced spelling still matches the record."""
    from itertools import permutations

    tokens = norm_name(s).split()
    if not tokens:
        return set()
    return {"".join(p) for p in permutations(tokens)} if len(tokens) <= 4 else {"".join(tokens)}


def names_match(given: str, on_file: str) -> bool:
    return bool(name_keys(given) & name_keys(on_file))


def _record_values(rec: dict) -> dict[str, set[str]]:
    return {
        "full_name": name_keys(rec["name"]) | {k for a in rec.get("name_aliases", []) for k in name_keys(a)},
        "dob": {rec["dob"]},
        "phone": {norm_phone(rec["phone"])} | {norm_phone(a) for a in rec.get("phone_aliases", [])},
        "email": {norm_email(rec["email"])} | {norm_email(a) for a in rec.get("email_aliases", [])},
        "ssn_last4": {rec["id_last4"]},
        "policy_number": {norm_policy(rec["policy_number"])},
    }


NORMALIZERS = {
    "full_name": lambda s: "".join(norm_name(s).split()),   # compared via name_keys, this is for change detection
    "dob": lambda s: (s or "").strip(),
    "phone": norm_phone,
    "email": norm_email,
    "ssn_last4": norm_last4,
    "policy_number": norm_policy,
}


@dataclass
class VerifyResult:
    verified: bool
    party_id: str | None = None
    provided: list[str] = field(default_factory=list)      # PII fields the caller has given
    matched: list[str] = field(default_factory=list)       # for the best candidate
    mismatched: list[str] = field(default_factory=list)    # for the best candidate (never shown to caller)
    still_needed: int = REQUIRED_MATCHES
    reason: str = ""


def evaluate(collected: dict[str, str | None], policyholders: list[dict]) -> VerifyResult:
    given = {k: NORMALIZERS[k](v) for k, v in collected.items() if v and k in NORMALIZERS}
    given = {k: v for k, v in given.items() if v}
    provided_pii = [k for k in PII_FIELDS if k in given]

    best = None
    def hit(k: str, vals: dict) -> bool:
        return bool(name_keys(collected["full_name"]) & vals[k]) if k == "full_name" else given[k] in vals[k]

    for rec in policyholders:
        vals = _record_values(rec)
        matched = [k for k in given if hit(k, vals)]
        mismatched = [k for k in given if not hit(k, vals)]
        pii_matched = [k for k in matched if k in PII_FIELDS]
        score = (len(pii_matched), -len(mismatched))
        if best is None or score > best[0]:
            best = (score, rec, pii_matched, mismatched)

    if best is None:
        return VerifyResult(False, provided=provided_pii, reason="no records")

    _, rec, pii_matched, mismatched = best
    if len(pii_matched) >= REQUIRED_MATCHES and not mismatched:
        return VerifyResult(True, rec["party_id"], provided_pii, pii_matched, [], 0, "verified")

    # With fewer than 3 fields we keep collecting and say nothing about matches,
    # so the caller can't probe the records one field at a time.
    if len(provided_pii) < REQUIRED_MATCHES:
        return VerifyResult(False, None, provided_pii, pii_matched, mismatched,
                            REQUIRED_MATCHES - len(provided_pii), "need_more")

    # still_needed must not depend on how many of the supplied values matched: that count would let a
    # caller confirm a DOB or last-4 one field at a time ("I just need 1 more detail" == 2 of 3 right).
    return VerifyResult(False, None, provided_pii, pii_matched, mismatched,
                        max(REQUIRED_MATCHES - len(provided_pii), 0), "mismatch")


def mask_email(e: str) -> str:
    local, _, domain = e.partition("@")
    return f"{local[0]}{'*' * max(len(local) - 1, 3)}@{domain}" if local else e


def mask_value(field_name: str, v: str) -> str:
    if field_name == "email":
        return mask_email(v)
    if field_name in ("phone", "ssn_last4"):
        d = re.sub(r"\D", "", v)
        return "•" * max(len(d) - 2, 2) + d[-2:]
    if field_name == "dob":
        return v[:4] + "-••-••"
    if field_name == "full_name":
        return " ".join(p[0] + "•" * (len(p) - 1) for p in v.split())
    return v
