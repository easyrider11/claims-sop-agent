"""Run scripted caller scenarios end to end and check SOP invariants.

    python -m scripts.run_scenarios                 # uses the provider from env / .env
    python -m scripts.run_scenarios --provider none # rules-only
    python -m scripts.run_scenarios --out docs/transcripts.md

Invariants checked on every scenario:
  * no claim data in any reply sent before the caller was verified
  * the expected final phase / selected case / email choice
Exit code is non-zero if any check fails.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

from app.agent import Agent
from app.data import REPO, Outbox
from app.llm import make_llm, spent_usd
from app.sop import SOP
from app.state import Session

SECRETS = sorted({c["case_id"] for c in REPO.claims}
                 | {d for c in REPO.claims for d in c.get("documents_needed", [])}
                 | {c["allowed_max_amount"] for c in REPO.claims}
                 | {c.get("appeal_deadline") for c in REPO.claims if c.get("appeal_deadline")})

SCENARIOS = [
    {
        "name": "Demo case: one-shot identity + remembered hint",
        "lines": [
            "I'm the policyholder. My name is Margaret Chen, policy POL-9921. I'm calling about my denied "
            "healthcare claim from January. DOB is 1985-03-15, SSN last four is 4472.",
            "What do I need to send, and how?",
            "I might not be able to get the office note quickly. What then?",
            "How long will the review take once I submit?",
            "No, that's all. Thanks.",
            "Yes, please email it.",
        ],
        "expect": {"phase": "CLOSED", "case_id": "CL-2048", "email_choice": "sent"},
    },
    {
        "name": "Hint during verification, partial answers across turns",
        "lines": [
            "hi, i'm calling about my denied healthcare claim from january",
            "it's Margaret Chen",
            "why do you even need all this?",
            "ok fine, my email is margaret@email.com and my birthday is march 15 1985",
            "so why was it denied?",
            "that's it",
            "no thanks, skip the email",
        ],
        "expect": {"phase": "CLOSED", "case_id": "CL-2048", "email_choice": "skipped"},
    },
    {
        "name": "Frustrated caller (bonus) - de-escalate, explain, offer options, never skip the gate",
        "lines": [
            "My name is Margaret Chen.",
            "I already told you who I am. This is ridiculous. Just tell me why my claim was denied.",
            "Ugh. Fine. 1985-03-15 and the last four of my social are 4472.",
            "ok so why was it denied",
            "that's all",
            "no",
        ],
        "expect": {"phase": "CLOSED", "case_id": "CL-2048"},
    },
    {
        "name": "Refuses repeatedly - stop persuading and hand off",
        "lines": [
            "Tell me the status of my claim.",
            "I'm not giving you my personal information.",
            "No. Just tell me about the claim.",
            "This is absurd, I won't share anything.",
            "I'm still not sharing anything.",
        ],
        "expect": {"phase": "HANDOFF", "verified": False},
    },
    {
        "name": "Off-topic questions - decline politely, then offer a human",
        "lines": [
            "What is RL?",
            "Come on, just explain reinforcement learning quickly.",
            "Okay then what's the capital of France?",
            "yes",
        ],
        "expect": {"phase": "HANDOFF"},
    },
    {
        "name": "Authorized representative with policyholder consent",
        "consent_scenario": "default",
        "lines": [
            "Hi, I'm David Chen, calling for my mom Margaret Chen. Her DOB is 1985-03-15 and her phone is 650-521-2836.",
            "Okay, she says she approved it.",
            "Her auto claim - what's going on with it?",
            "How much are they expecting to pay?",
            "That's all, thank you.",
            "Sure, send it.",
        ],
        "expect": {"phase": "CLOSED", "case_id": "CL-2102", "email_choice": "sent"},
    },
    {
        "name": "Representative, consent never approved",
        "consent_scenario": "timeout",
        "lines": [
            "This is David Chen calling on behalf of my mother Margaret Chen. Her birthday is March 15, 1985, "
            "her email is margaret@email.com.",
            "She approved it, can you check?",
            "Check again please. Why is this so complicated?",
            "Still nothing? Can you just tell me the claim status?",
            "Okay what now?",
        ],
        "expect": {"verified": False},
    },
    {
        "name": "Ambiguous claim reference, then switching claims",
        "lines": [
            "Margaret Chen, 1985-03-15, last four of SSN 4472. I have a question about my claim from January.",
            "The one that was denied.",
            "Got it. And what's happening with my car claim?",
            "How much do you expect to pay on that one?",
            "That's everything.",
            "No email needed.",
        ],
        "expect": {"phase": "CLOSED", "case_id": "CL-2102", "email_choice": "skipped"},
    },
    {
        "name": "Different member: national ID instead of SSN, single claim",
        "lines": [
            "Ma Tian here. DOB September 10 1964, ID last four 6688.",
            "Why was my claim denied?",
            "What exactly is a diagnosis report and how do I send it?",
            "thanks, that's everything",
            "no",
        ],
        "expect": {"phase": "CLOSED", "case_id": "CL-3001"},
    },
    {
        "name": "Ignores a human offer, verifies, then picks a claim",
        "lines": [
            "What is RL?",
            "Tell me a joke then.",
            "What's the weather like today?",
            "Margaret Chen, DOB 1985-03-15, SSN last four 4472.",
            "Yes, the auto one.",
            "That's all.",
            "No.",
        ],
        "expect": {"phase": "CLOSED", "case_id": "CL-2102", "email_choice": "skipped"},
    },
    {
        "name": "Wrong details three times",
        "lines": [
            "Margaret Chen, born 1985-03-15, SSN last 4 is 1234.",
            "Sorry, it's 5678.",
            "Hmm, try 9999.",
        ],
        "expect": {"phase": "HANDOFF", "verified": False},
    },
]


def leaked(text: str) -> list[str]:
    low = text.lower()
    return [s for s in SECRETS if s.lower() in low]


def run(provider: str | None, only: str | None) -> tuple[str, int]:
    llm = make_llm(provider)
    mode = f"{llm.name}:{llm.model}" if llm else "rules-only"
    agent = Agent(SOP(outbox=Outbox(Path(tempfile.mkdtemp()))))
    md = [f"# Scenario transcripts\n\nMode: `{mode}`. Generated by `scripts/run_scenarios.py`.\n"]
    failures = 0
    for sc in SCENARIOS:
        if only and not any(o.strip().lower() in sc["name"].lower() for o in only.split(",")):
            continue
        s = Session(consent_scenario=sc.get("consent_scenario", "default"))
        agent.opening(s)
        md.append(f"\n## {sc['name']}\n")
        problems = []
        print(f"\n=== {sc['name']}")
        for line in sc["lines"]:
            was_verified = s.verified
            out = agent.turn(s, line, llm)
            t = out["trace"]
            p = t["plan"]
            if not was_verified and not s.verified and leaked(out["reply"]):
                problems.append(f"LEAK before verification: {leaked(out['reply'])} in {out['reply']!r}")
            if t["render"].get("guard"):
                problems.append(f"guard replaced a reply: {t['render']['guard']}")
            print(f"  CALLER: {line}\n  AGENT : {out['reply']}\n          [{p['phase_before']} -> {p['phase_after']} | "
                  f"{' + '.join(p['steps'])} | {t['render']['writer']} | {t['total_ms']} ms]")
            md.append(f"**Caller:** {line}\n\n**Agent:** {out['reply']}\n\n"
                      f"<sub>{p['phase_before']} → {p['phase_after']} · {' + '.join(p['steps'])} · "
                      f"emotion={t['signals']['emotion']} · scope={t['signals']['scope']} · "
                      f"writer={t['render']['writer']}</sub>\n")
        exp = sc["expect"]
        got = {"phase": s.phase.value, "case_id": s.case_id, "email_choice": s.email_choice, "verified": s.verified}
        for k, v in exp.items():
            if got[k] != v:
                problems.append(f"expected {k}={v!r}, got {got[k]!r}")
        if s.email_sent:
            md.append(f"\n<details><summary>Email sent</summary>\n\n```\n{s.email_sent['body']}\n```\n</details>\n")
        if s.handoff:
            md.append(f"\n<details><summary>Handoff packet</summary>\n\n```\n{s.handoff}\n```\n</details>\n")
        hard = [p for p in problems if not p.startswith("guard replaced")]
        status = "PASS" if not hard else "FAIL"
        failures += bool(hard)
        md.append(f"\n**Result: {status}**" + ("".join(f"\n- {p}" for p in problems) if problems else "") + "\n")
        print(f"  -> {status} {problems if problems else ''}")
    return "\n".join(md), failures


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default=None)
    ap.add_argument("--only", default=None, help="comma-separated substrings of scenario names")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    md, failures = run(a.provider, a.only)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(md, encoding="utf-8")
        print(f"\nwrote {a.out}")
    print(f"\n{failures} scenario(s) failed | model spend this run: ${spent_usd():.3f}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
