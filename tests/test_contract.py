import pytest

from app.sop import SOP, SOPViolation, TOOLS, Plan
from app.state import Phase, Session
from tests.conftest import DEMO


def test_claim_data_needs_verification():
    sop, s = SOP(), Session()
    with pytest.raises(SOPViolation):
        sop._party(s)


def test_illegal_transition_raises():
    sop, s = SOP(), Session()
    with pytest.raises(SOPViolation):
        sop._goto(s, Plan("VERIFY_ID"), Phase.POST_PROCESS)


def test_unverified_cannot_enter_resolve():
    sop, s = SOP(), Session()
    with pytest.raises(SOPViolation, match="verified"):
        sop._goto(s, Plan("VERIFY_ID"), Phase.RESOLVE_INTENT)


def test_tool_outside_its_phase_raises():
    sop, s = SOP(), Session()
    assert "send_email" not in TOOLS[Phase.VERIFY_ID]
    with pytest.raises(SOPViolation):
        sop._tool(s, Plan("VERIFY_ID"), "send_email")


def test_every_turn_respects_contract(convo):
    """Run full conversations; every recorded tool was legal in the phase it ran in."""
    c = convo()
    for line in [DEMO, "what do I need to send?", "what about my car claim?", "that's all", "yes"]:
        out = c.say(line)
        for a in out["trace"]["plan"]["actions"]:
            assert a["tool"] in TOOLS[Phase(a["phase"])]


def test_harness_bug_fails_closed(agent, monkeypatch):
    s = Session()
    agent.opening(s)

    def boom(*a, **k):
        raise SOPViolation("simulated bug")

    monkeypatch.setattr(agent.sop, "step", boom)
    out = agent.turn(s, DEMO, None)
    assert s.phase == Phase.HANDOFF and "human representative" in out["reply"]
    assert "CL-2048" not in out["reply"]


def test_contract_snapshot():
    """The SOP as data. A change here is a design change and must be deliberate."""
    from app.sop import contract
    c = contract()
    assert c["transitions"] == {
        "VERIFY_ID": ["CLOSED", "HANDOFF", "RESOLVE_INTENT"],
        "RESOLVE_INTENT": ["HANDOFF", "POST_PROCESS", "PROCESS_CASE"],
        "PROCESS_CASE": ["HANDOFF", "POST_PROCESS", "RESOLVE_INTENT"],
        "POST_PROCESS": ["CLOSED", "HANDOFF", "PROCESS_CASE", "RESOLVE_INTENT"],
        "CLOSED": [], "HANDOFF": [],
    }
    assert c["tools"] == {
        "VERIFY_ID": ["check_consent", "request_consent", "transfer_to_human", "verify_identity"],
        "RESOLVE_INTENT": ["get_claim", "list_claims", "transfer_to_human"],
        "PROCESS_CASE": ["claim_knowledge", "transfer_to_human"],
        "POST_PROCESS": ["send_email", "transfer_to_human"],
        "CLOSED": [], "HANDOFF": [],
    }


def test_demo_conversation_turn_by_turn(convo):
    """Not vacuous: each turn's steps, phase path and tools are pinned."""
    c = convo()
    o = c.say(DEMO)
    assert o["trace"]["plan"]["steps"] == ["VERIFIED", "CASE_RESOLVED", "ANSWER"]
    assert o["trace"]["plan"]["phase_path"] == ["VERIFY_ID", "RESOLVE_INTENT", "PROCESS_CASE"]
    assert [a["tool"] for a in o["trace"]["plan"]["actions"]] == ["verify_identity", "list_claims", "get_claim", "claim_knowledge"]
    o = c.say("what do I need to send?")
    assert o["trace"]["plan"]["steps"] == ["ANSWER"] and o["trace"]["plan"]["phase_path"] == ["PROCESS_CASE"]
    o = c.say("that's all")
    assert o["trace"]["plan"]["steps"] == ["OFFER_EMAIL"] and o["trace"]["plan"]["phase_path"] == ["PROCESS_CASE", "POST_PROCESS"]
    o = c.say("yes")
    assert o["trace"]["plan"]["steps"] == ["EMAIL_SENT"] and [a["tool"] for a in o["trace"]["plan"]["actions"]] == ["send_email"]
    assert c.phase == "CLOSED"
