"""The output guard catches a misbehaving model, whatever the prompt said."""
from app.llm import BaseLLM, LLMError
from tests.conftest import DEMO, assert_no_claim_leak


class FakeLLM(BaseLLM):
    """Extraction fails (so rules are used); the writer returns whatever we script."""
    name, model = "fake", "fake"

    def __init__(self, replies):
        self.replies = list(replies)

    def json(self, *a, **k):
        raise LLMError("no extraction in this fake")

    def text(self, *a, **k):
        # repeat the last scripted reply if the harness asks for a rewrite
        return self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]


def test_leak_before_verification_is_replaced(agent):
    from tests.conftest import Convo
    c = Convo(agent, llm=FakeLLM(["Your claim CL-2048 was denied for a missing pathology report."]))
    out = c.say("Why was my claim denied? I'm Margaret Chen.")
    assert out["trace"]["render"]["guard"]
    assert_no_claim_leak(out["reply"])
    assert c.s.events[-1]["kind"] == "guard_blocked"


def test_ungrounded_amount_after_verification_is_replaced(agent):
    from tests.conftest import Convo
    c = Convo(agent, llm=FakeLLM(["Verified! You'll get $999.00 back by 2026-11-01."]))
    out = c.say(DEMO)
    problems = out["trace"]["render"]["guard"]
    assert any("amount" in p for p in problems) and any("date" in p for p in problems)
    assert "999" not in out["reply"]


def test_grounded_llm_reply_passes(agent):
    from tests.conftest import Convo
    good = ("Thanks, you're verified. I found claim CL-2048, your healthcare claim from January 12, 2026. "
            "It was denied because the pathology report and office note were missing. The allowed amount was $1,450.")
    c = Convo(agent, llm=FakeLLM([good]))
    out = c.say(DEMO)
    assert out["trace"]["render"]["guard"] == []
    assert out["reply"] == good


def test_other_members_claim_id_blocked(agent):
    from tests.conftest import Convo
    c = Convo(agent, llm=FakeLLM(["You also have claim CL-3001 open."]))
    out = c.say(DEMO)
    assert any("CL-3001" in p for p in out["trace"]["render"]["guard"])


def test_rewrite_fixes_bad_draft(agent):
    from tests.conftest import Convo
    good = "Thanks, you're verified. I found claim CL-2048, the denied healthcare claim from January you mentioned."
    c = Convo(agent, llm=FakeLLM(["Verified! You'll get $999.00 back.", good]))
    out = c.say(DEMO)
    assert out["reply"] == good
    assert out["trace"]["render"]["writer"].endswith("(rewritten)")
    assert out["trace"]["render"]["first_draft_rejected"]["problems"]


def test_false_action_claims_blocked(agent):
    from tests.conftest import Convo
    c = Convo(agent, llm=FakeLLM(["Great, you're verified now! What's your question?"]))
    out = c.say("Hi, my name is Margaret Chen")
    assert any("verified" in p for p in out["trace"]["render"]["guard"])
    c2 = Convo(agent, llm=FakeLLM(["I'm transferring you to an agent now."]))
    out = c2.say("My name is Margaret Chen")
    assert any("transfer" in p for p in out["trace"]["render"]["guard"])


# ---------------------------------------------------------------- formats the guard must read
def _session(verified=True, said=""):
    from app.state import Session
    s = Session()
    if verified:
        s.verified_party_id = s.matched_party_id = "P9"
    s.transcript = [{"role": "user", "content": said}]
    return s


FACTS = {"claim_knowledge": {"claim": {"case_id": "CL-2048", "created_at": "2026-01-12", "allowed_max_amount": "1450.00",
                                       "appeal_deadline": "2026-03-18"}}}


def test_guard_reads_spelled_out_dates_and_amounts():
    from app.guard import check
    from app.data import REPO
    s = _session()
    assert check("The appeal deadline was March 18, 2026 and the allowed amount was $1,450.", s, FACTS, REPO) == []
    assert check("The deadline was March 18.", s, FACTS, REPO) == []                 # year-less, grounded by month/day
    assert check("The deadline was 18 March 2026.", s, FACTS, REPO) == []
    assert any("date" in p for p in check("You have until April 30th to appeal.", s, FACTS, REPO))
    assert any("amount" in p for p in check("You will get 999 dollars back.", s, FACTS, REPO))
    assert any("number" in p for p in check("Your claim number is 7777.", s, FACTS, REPO))


def test_guard_rejects_instruction_vocabulary_and_other_members_data():
    from app.guard import check
    from app.data import REPO
    s = _session()
    assert any("vocabulary" in p for p in check("I've noted it in the FACTS.", s, FACTS, REPO))
    assert any("masked" in p.lower() or "vocabulary" in p for p in check("Sent to the masked address.", s, FACTS, REPO))
    assert any("another member" in p for p in check(
        "It was denied because the submitted materials did not include the treating provider diagnosis report.", s, FACTS, REPO))
    assert any("another member" in p for p in check("Try matian@example.com", s, FACTS, REPO))


def test_guard_does_not_whitelist_caller_typed_claim_facts_before_verification():
    from app.guard import check
    from app.data import REPO
    s = _session(verified=False, said="Was it denied for the pathology report? Was it $1,450?")
    probs = check("Yes, that's right: it was denied for the pathology report, $1,450.", s, {}, REPO)
    assert any("document" in p for p in probs) and any("amount" in p for p in probs)
    assert check("I can't discuss any claim until you're verified. Could you share your date of birth?", s, {}, REPO) == []
