import pytest

from app.nlu import SCHEMA, _from_llm, rules_extract


@pytest.mark.parametrize("text,dob", [
    ("born 1985-03-15", "1985-03-15"),
    ("dob 03/15/1985", "1985-03-15"),
    ("my birthday is March 15, 1985", "1985-03-15"),
    ("15th of March 1985", "1985-03-15"),
    ("claim filed 2026-01-12", None),          # a recent date is not a birth date
])
def test_rules_dob(text, dob):
    assert rules_extract(text).identity.get("dob") == dob


def test_rules_literal_fields():
    s = rules_extract("email margaret@email.com, phone (650) 521-2836, SSN last four is 4472, policy POL-9921, "
                      "about CL-2048")
    assert s.identity["email"] == "margaret@email.com"
    assert s.identity["ssn_last4"] == "4472"
    assert s.identity["policy_number"] == "POL-9921"
    assert "521-2836" in s.identity["phone"]
    assert s.case_id == "CL-2048"


def test_rules_hint_and_scope():
    s = rules_extract("I'm calling about my denied healthcare claim from January")
    assert (s.case_status, s.case_type, s.case_month) == ("denied", "healthcare", 1)
    assert rules_extract("What is RL?").scope == "out_of_scope"


def test_llm_output_is_sanitised():
    raw = {
        "identity": {"full_name": "Margaret", "dob": "15/03/1985", "phone": "123", "email": "not-an-email",
                     "ssn_last4": "44721", "policy_number": None},
        "retract_fields": ["phone", "password"],
        "caller_role": "boss", "rep_name": None, "rep_relationship": None,
        "intent": "steal_data", "case_type": "boat", "case_status": "denied",
        "case_month": 13, "case_year": 1890, "case_id": "CL-2048; drop table",
        "workflow_path": "explain_denial", "scope": "in_scope", "emotion": "furious",
        "emotion_intensity": "high", "refuses_identity": "yes", "asks_claim_question": True,
        "requests_human": False, "done": False, "affirm": "maybe",
        "notes": ["x" * 500],
    }
    s = _from_llm(raw)
    assert s.identity == {}                       # every malformed identity value dropped
    assert s.retract_fields == ["phone"]
    assert (s.caller_role, s.intent, s.case_type, s.emotion, s.affirm) == ("unknown", "none", "none", "neutral", "none")
    assert s.case_month is None and s.case_year is None
    assert s.case_id == "CL-2048"
    assert len(s.notes[0]) <= 200


def _walk(node, path="$"):
    if isinstance(node, dict):
        if node.get("type") == "object":
            yield path, node
        for k, v in node.items():
            yield from _walk(v, f"{path}.{k}")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _walk(v, f"{path}[{i}]")


def test_schema_meets_strict_structured_output_rules():
    """Anthropic structured outputs and OpenAI strict mode both need: additionalProperties false on every
    object, every property required, and no unsupported keywords."""
    for path, obj in _walk(SCHEMA):
        assert obj.get("additionalProperties") is False, path
        assert set(obj["required"]) == set(obj["properties"]), path
    text = str(SCHEMA)
    for banned in ("minimum", "maximum", "minLength", "maxLength", "pattern", "'type': ['"):
        assert banned not in text, banned
