"""Bugs found in the pre-submission review. Each test reproduces the original failure."""
from app.llm import BaseLLM, LLMError
from app.nlu import is_bare_answer, rules_extract
from tests.conftest import DEMO, Convo, assert_no_claim_leak

ID = "Margaret Chen, DOB 1985-03-15, SSN last four 4472"


def test_stale_human_offer_does_not_fire_after_verification(convo):
    c = convo()
    for q in ["What is RL?", "Tell me a joke", "What's the weather like today?"]:
        c.say(q)                                    # third one offers a human
    c.say(ID)                                       # caller ignores the offer and verifies
    c.say("Yes, the auto one")
    assert c.phase == "PROCESS_CASE" and c.s.case_id == "CL-2102"


def test_yes_with_content_is_not_an_acceptance(convo):
    c = convo()
    c.say("Tell me my claim status, I'm not giving you my info")
    c.say("No. Why do you need that?")
    c.say("I refuse, this is ridiculous.")          # offers a human
    c.say("ok the dental one")
    assert c.phase != "HANDOFF"


class CountingLLM(BaseLLM):
    name, model = "count", "count"

    def __init__(self):
        self.calls = 0

    def json(self, *a, **k):
        self.calls += 1
        raise LLMError("no")

    def text(self, *a, **k):
        raise LLMError("no")


def test_shortcut_only_for_bare_answers(agent):
    assert is_bare_answer("yes please") and is_bare_answer("No thanks.") and is_bare_answer("nope")
    assert not is_bare_answer("No, what about payment?") and not is_bare_answer("yes, the auto one")
    llm = CountingLLM()
    c = Convo(agent, llm=llm)
    c.say(DEMO)
    before = llm.calls
    c.say("No, what about payment?")
    assert llm.calls == before + 1                  # the model was asked
    c.say("no")
    assert llm.calls == before + 1                  # bare answer: model skipped


def test_dont_is_not_no():
    assert rules_extract("Don't have the office note yet, what can I do?").affirm == "none"
    assert rules_extract("don't send it").affirm == "no"


def test_consent_timeout_closes_after_human_declined(convo):
    c = convo(consent_scenario="timeout")
    c.say("This is David Chen calling on behalf of my mother Margaret Chen. Her birthday is March 15, 1985, "
          "her email is margaret@email.com.")
    for _ in range(5):
        c.say("Check again please")
    assert c.s.consent_status == "timeout"
    out = c.say("no")                               # declines the human too -> close, don't loop
    assert c.phase == "CLOSED" and "CONSENT_CLOSED" in out["trace"]["plan"]["steps"]
    assert not c.s.verified


def test_representative_not_told_policyholder_name_from_record(convo):
    c = convo()
    out = c.say("Hi, I'm David Chen, calling for my mom. Her DOB is 1985-03-15, phone 650-521-2836, "
                "email margaret@email.com.")
    assert "Margaret" not in out["reply"]
    assert_no_claim_leak(out["reply"])


def test_summary_has_no_duplicate_lines(convo):
    c = convo()
    c.say(DEMO)
    c.say("I don't have the office note yet.")
    c.say("Still don't have the office note, honestly.")
    c.say("that's all")
    c.say("yes")
    body = c.s.email_sent["body"]
    assert body.count("If the office note is not available") == 1
