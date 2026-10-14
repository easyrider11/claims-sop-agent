"""Output guard: a deterministic check on every reply before the caller sees it.

The main protection is structural (claim data is never in the prompt before verification).
This is the second layer: if a reply still contains something it shouldn't, the writer gets one
rewrite and then the template reply is used; the event is logged either way.

Checks:
  0. Always: the reply may not claim an action the harness didn't take (verified / email sent /
     transferred), must be non-empty and reasonably short, and must not leak instruction vocabulary.
  1. Before verification: no claim IDs, denial reasons, required-document names, claim amounts, full
     dates, or on-file PII. Caller-typed claim facts are NOT whitelisted: agreeing with "it was denied for
     the pathology report, right?" confirms data just as much as volunteering it.
  2. After verification: every amount, date and bare number in the reply must come from the facts
     handed to the writer or from the caller's own words; claim IDs must belong to the caller; no other
     member's data; no unmasked PII.

What it cannot do: judge prose. "You could try a new review" is not a number, so it passes. That is why
post-deadline wording is fixed in the facts and templates, not here.
"""
from __future__ import annotations

import json
import re
from datetime import date

from .data import Repository
from .state import Session

MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september",
          "october", "november", "december"]
MONTH_RE = r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
MONEY_RE = re.compile(r"\$\s?(\d[\d,]*(?:\.\d{1,2})?)|\b(\d[\d,]*\.\d{2})\b|\b(\d[\d,]*)\s*(?:dollars|usd|bucks)\b|\busd\s*(\d[\d,]*)", re.I)
ISO_RE = re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b")
US_RE = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b")
MDY_RE = re.compile(r"\b" + MONTH_RE + r"\.?\s+(\d{1,2})(?:st|nd|rd|th)?(?:,?\s+(\d{4}))?\b", re.I)
DMY_RE = re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?" + MONTH_RE + r"\.?(?:,?\s+(\d{4}))?\b", re.I)
CASE_RE = re.compile(r"\bCL-?\s?\d{3,6}\b", re.I)
BARE_NUM_RE = re.compile(r"(?<![\d$])(?<!\d[.,/-])\b(\d{3,6})\b(?![.,/-]?\d)")
VOCAB_LEAK = re.compile(r"\b(FACTS|STEPS|masked)\b")
MAX_REPLY_CHARS = 1600


def _amounts(text: str) -> set[float]:
    out = set()
    for m in MONEY_RE.finditer(text):
        raw = next(g for g in m.groups() if g).replace(",", "")
        try:
            out.add(round(float(raw), 2))
        except ValueError:
            pass
    return out


def _dates(text: str) -> set[tuple[int | None, int, int]]:
    """Dates as (year-or-None, month, day), so 'March 18' can be matched against 2026-03-18 in the facts."""
    out: set[tuple[int | None, int, int]] = set()
    for y, mo, d in ISO_RE.findall(text):
        out.add((int(y), int(mo), int(d)))
    for mo, d, y in US_RE.findall(text):
        out.add((int(y), int(mo), int(d)))
    for mon, d, y in MDY_RE.findall(text):
        out.add((int(y) if y else None, MONTHS.index(next(m for m in MONTHS if m.startswith(mon.lower()[:3]))) + 1, int(d)))
    for d, mon, y in DMY_RE.findall(text):
        out.add((int(y) if y else None, MONTHS.index(next(m for m in MONTHS if m.startswith(mon.lower()[:3]))) + 1, int(d)))
    valid = set()
    for y, mo, d in out:
        try:
            date(y or 2000, mo, d)
            valid.add((y, mo, d))
        except ValueError:
            pass
    return valid


def _date_grounded(d: tuple[int | None, int, int], allowed: set[tuple[int | None, int, int]]) -> bool:
    y, mo, day = d
    return any(am == mo and ad == day and (y is None or ay is None or ay == y) for ay, am, ad in allowed)


def _fact_amounts(facts_text: str) -> set[float]:
    out = _amounts(facts_text)
    for m in re.finditer(r'"(\d+\.\d{2})"', facts_text):   # amounts are stored as "1450.00"
        out.add(round(float(m.group(1)), 2))
    return out


def _bare_numbers(text: str) -> set[str]:
    """3-6 digit integers that are not part of a date, an id, a phone number or a dollar amount."""
    scrubbed = ISO_RE.sub(" ", text)
    scrubbed = US_RE.sub(" ", scrubbed)
    scrubbed = re.sub(r"\b(?:CL|POL)-?\s?\d+\b", " ", scrubbed, flags=re.I)
    scrubbed = re.sub(r"\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}", " ", scrubbed)
    scrubbed = MONEY_RE.sub(" ", scrubbed)
    return {m.group(1) for m in BARE_NUM_RE.finditer(scrubbed) if not (1900 <= int(m.group(1)) <= 2100)}


CLAIMS_VERIFIED = re.compile(
    r"you(?:'re| are) (?:now |all |fully |successfully )?verified|verification (?:is |was )?(?:complete|successful)|"
    r"i(?:'ve| have) verified you|identity (?:is |has been )?(?:confirmed|verified)")
CLAIMS_EMAIL_SENT = re.compile(
    r"(?:summary|email)[^.?!]{0,40}(?:has been sent|was sent|is on its way|went out)|"
    r"i(?:'ve| have)? (?:just )?(?:sent|emailed)[^.?!]{0,40}(?:summary|email)")
CLAIMS_TRANSFER = re.compile(r"transferring you|i(?:'ve| have) transferred you|connecting you (?:now|with)")
CONDITIONAL = re.compile(r"\b(?:once|after|until|when|as soon as|before|if|can|could|would|will|to get)\b[^.?!]{0,25}$")


def _asserts(pattern: re.Pattern, low: str) -> bool:
    """True if the reply states the action as done ('you're verified'), not as a condition
    or offer ('once you're verified', 'I can get you transferring')."""
    return any(not CONDITIONAL.search(low[max(0, m.start() - 40):m.start()]) for m in pattern.finditer(low))


def _action_claims(low: str, session: Session) -> list[str]:
    """The writer must not claim an action the harness did not take."""
    out = []
    if _asserts(CLAIMS_VERIFIED, low) and not session.verified:
        out.append("says the caller is verified, but they are not")
    if _asserts(CLAIMS_EMAIL_SENT, low) and session.email_sent is None:
        out.append("says an email was sent, but none was")
    if _asserts(CLAIMS_TRANSFER, low) and session.phase.value != "HANDOFF":
        out.append("says the caller is being transferred, but no transfer happened")
    return out


def _pii_values(p: dict) -> list[str]:
    phone = re.sub(r"\D", "", p["phone"])[-10:]
    return [p["email"].lower(), p["dob"], p["id_last4"], phone, f"{phone[:3]}-{phone[3:6]}-{phone[6:]}",
            f"({phone[:3]}) {phone[3:6]}-{phone[6:]}"] + [a.lower() for a in p.get("email_aliases", [])]


def check(reply: str, session: Session, facts: dict, repo: Repository) -> list[str]:
    problems: list[str] = []
    low = reply.lower()
    digits_only = re.sub(r"\D", "", reply)
    said = " ".join(m["content"] for m in session.transcript if m["role"] == "user")
    said_low = said.lower()
    if not reply.strip():
        return ["empty reply"]
    if len(reply) > MAX_REPLY_CHARS:
        problems.append(f"reply too long ({len(reply)} chars)")
    if VOCAB_LEAK.search(reply):
        problems.append("instruction vocabulary (FACTS/STEPS/masked) in the reply")
    problems += _action_claims(low, session)

    if not session.verified:
        if CASE_RE.search(reply):
            # even a caller-supplied ID is not echoed: it would sound like confirming the claim exists
            problems.append("a claim id before verification")
        for c in repo.claims:
            reason = (c.get("denial_reason") or "").lower()
            if reason and reason[:40] in low:
                problems.append("a denial reason before verification")
            for d in c.get("documents_needed") or []:
                if d.lower() in low:
                    problems.append("a required document name before verification")
        if _amounts(reply) or _bare_numbers(reply) - _bare_numbers(said):
            problems.append("a money amount or claim number before verification")
        if _dates(reply) - _dates(said):
            problems.append("a date before verification")
        for p in repo.policyholders:
            for v in _pii_values(p):
                if v and (v in low or (v.isdigit() and len(v) >= 10 and v in digits_only)) and v not in said_low:
                    problems.append("on-file personal data before verification")
        return sorted(set(problems))

    facts_text = json.dumps(facts, default=str)
    bad_amounts = _amounts(reply) - _fact_amounts(facts_text) - _amounts(said)
    if bad_amounts:
        problems.append(f"an amount not in the facts: {sorted(bad_amounts)}")
    allowed_dates = _dates(facts_text) | _dates(said)
    bad_dates = [d for d in _dates(reply) if not _date_grounded(d, allowed_dates)]
    if bad_dates:
        problems.append(f"a date not in the facts: {sorted(str(d) for d in bad_dates)}")
    bad_nums = _bare_numbers(reply) - _bare_numbers(facts_text) - _bare_numbers(said)
    if bad_nums:
        problems.append(f"a number not in the facts: {sorted(bad_nums)}")
    own = {c["case_id"].upper() for c in repo.claims_for(session.verified_party_id)}
    for m in CASE_RE.finditer(reply):
        cid = "CL-" + re.sub(r"\D", "", m.group(0))
        if cid not in own:
            problems.append(f"claim id {cid} is not the caller's")
    for c in repo.claims:
        if c["case_id"].upper() in own:
            continue
        reason = (c.get("denial_reason") or "").lower()
        if reason and reason[:40] in low and reason[:40] not in facts_text.lower():
            problems.append("another member's denial reason")
    holder = repo.policyholder(session.verified_party_id)
    for p in repo.policyholders:
        for v in _pii_values(p):
            if v and (v in low or (v.isdigit() and len(v) >= 10 and v in digits_only)) and v not in said_low:
                problems.append("unmasked personal data in reply" if holder and p["party_id"] == holder["party_id"]
                                else "another member's personal data")
    return sorted(set(problems))
