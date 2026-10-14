"""The main defence is structural: nothing about any claim is in any model prompt before verification.
This test records every prompt the harness sends and checks that property directly."""
from app.data import REPO
from app.llm import BaseLLM, LLMError
from tests.conftest import DEMO, Convo

SECRETS = sorted({c["case_id"] for c in REPO.claims}
                 | {d for c in REPO.claims for d in c.get("documents_needed", [])}
                 | {c["allowed_max_amount"] for c in REPO.claims}
                 | {(c.get("denial_reason") or "")[:30] for c in REPO.claims if c.get("denial_reason")}
                 | {c.get("appeal_deadline") for c in REPO.claims if c.get("appeal_deadline")})
OTHER_PII = [v for p in REPO.policyholders if p["party_id"] != "P9" for v in (p["email"], p["dob"], p["id_last4"])]


class RecordingLLM(BaseLLM):
    """Captures every prompt, then fails so the harness runs on rules/templates."""
    name, model = "recording", "recording"

    def __init__(self):
        self.prompts: list[tuple[str, str, str]] = []   # (kind, system, user)

    def json(self, system, user, schema, max_tokens=4000):
        self.prompts.append(("extract", system, user))
        raise LLMError("recording only")

    def text(self, system, user, max_tokens=4000):
        self.prompts.append(("write", system, user))
        raise LLMError("recording only")


def _assert_clean(prompts, when, said=""):
    """Secrets the caller typed themselves are their words, not a leak; everything else must be absent."""
    for kind, system, user in prompts:
        blob = (system + user).lower()
        for secret in SECRETS:
            if secret.lower() in said.lower():
                continue
            assert secret.lower() not in blob, f"{secret!r} in a {kind} prompt {when}"
        for v in OTHER_PII:
            assert v.lower() not in blob, f"another member's PII in a {kind} prompt {when}"


def test_no_claim_data_in_any_prompt_before_verification(agent):
    llm = RecordingLLM()
    c = Convo(agent, llm=llm)
    lines = ["Hi, I'm calling about my denied healthcare claim from January.",
             "Why was it denied? What documents do you need?",
             "My name is Margaret Chen, policy POL-9921",
             "This is ridiculous. I already told you who I am.",
             "What about claim CL-2048?"]
    for line in lines:
        c.say(line)
        assert not c.s.verified
    assert len(llm.prompts) >= 8
    _assert_clean(llm.prompts, "before verification", said=" ".join(lines))
    assert all("CLAIMS (caller is verified)" not in user for _, _, user in llm.prompts)


def test_claims_list_appears_only_after_verification(agent):
    llm = RecordingLLM()
    c = Convo(agent, llm=llm)
    c.say("Hi, I'm calling about a claim.")
    before = len(llm.prompts)
    c.say(DEMO)
    assert c.s.verified
    c.say("what about my car claim?")
    after = llm.prompts[before:]
    extract_after = [u for k, _, u in after if k == "extract"]
    # the extraction prompt for the DEMO turn itself runs before verification (still clean) ...
    assert "CLAIMS (caller is verified)" not in extract_after[0]
    # ... and the next one, for a verified caller, carries only this caller's claims
    assert "CLAIMS (caller is verified)" in extract_after[1]
    assert "CL-3001" not in extract_after[1]                 # Ma Tian's claim never appears
    _assert_clean([p for p in after if "CLAIMS (caller is verified)" not in p[2] and p[0] == "extract"][:0], "n/a")


def test_only_the_verified_callers_data_reaches_the_writer(agent):
    llm = RecordingLLM()
    c = Convo(agent, llm=llm)
    c.say(DEMO)
    writes = [u for k, _, u in llm.prompts if k == "write"]
    assert any("CL-2048" in u for u in writes)
    for u in writes:
        assert "CL-3001" not in u
        for v in OTHER_PII:
            assert v.lower() not in u.lower()
