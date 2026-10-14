from fastapi.testclient import TestClient

from app.server import app

client = TestClient(app)
RULES = {"X-LLM-Provider": "none"}


def _new():
    r = client.post("/api/session", json={"consent_scenario": "default"}, headers=RULES)
    assert r.status_code == 200
    return r.json()


def test_session_and_chat_roundtrip():
    d = _new()
    assert d["mode"].startswith("rules-only") and d["state"]["phase"] == "VERIFY_ID"
    r = client.post("/api/chat", json={"session_id": d["session_id"], "message": "My name is Margaret Chen"})
    body = r.json()
    assert r.status_code == 200 and body["state"]["phase"] == "VERIFY_ID"
    assert body["trace"]["plan"]["steps"]


def test_identity_is_masked_in_state():
    d = _new()
    r = client.post("/api/chat", json={"session_id": d["session_id"],
                                       "message": "Margaret Chen, DOB 1985-03-15, SSN last four 4472"})
    ident = r.json()["state"]["identity"]
    assert "4472" not in str(ident) and "1985-03-15" not in str(ident)


def test_unknown_session_and_validation():
    assert client.post("/api/chat", json={"session_id": "nope", "message": "hi"}).status_code == 404
    d = _new()
    assert client.post("/api/chat", json={"session_id": d["session_id"], "message": ""}).status_code == 422
    assert client.post("/api/chat", json={"session_id": d["session_id"], "message": "x" * 2001}).status_code == 422


def test_bad_provider_rejected():
    r = client.post("/api/session", json={}, headers={"X-LLM-Provider": "llama-local"})
    assert r.status_code == 400


def test_sop_contract_and_health():
    c = client.get("/api/sop").json()
    assert c["transitions"]["VERIFY_ID"] == ["CLOSED", "HANDOFF", "RESOLVE_INTENT"]
    assert "send_email" in c["tools"]["POST_PROCESS"] and "send_email" not in c["tools"]["VERIFY_ID"]
    assert client.get("/api/health").json()["ok"] is True


def test_state_hides_internal_ids_and_full_email():
    d = _new()
    sid = d["session_id"]
    for m in ["I'm the policyholder. My name is Margaret Chen. Denied healthcare claim from January. "
              "DOB 1985-03-15, SSN last four 4472.", "that's all", "yes"]:
        r = client.post("/api/chat", json={"session_id": sid, "message": m})
    st = r.json()["state"]
    assert st["phase"] == "CLOSED"
    assert "margaret@email.com" not in str(st)
    assert "P9" not in str(st)
    assert "matched_party_id" not in st and "verified_party_id" not in st


def test_bad_provider_leaves_no_orphan_session():
    from app.server import SESSIONS
    n = len(SESSIONS)
    client.post("/api/session", json={}, headers={"X-LLM-Provider": "llama-local"})
    assert len(SESSIONS) == n


def test_provider_without_key_is_a_json_400(monkeypatch):
    from app import server
    for v in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "SILICONFLOW_API_KEY"):
        monkeypatch.delenv(v, raising=False)
    server._LLM_CACHE.clear()
    for provider in ("openai", "anthropic", "siliconflow"):
        r = client.post("/api/session", json={}, headers={"X-LLM-Provider": provider})
        assert r.status_code == 400, provider
        assert "key" in r.json()["detail"].lower()
    server._LLM_CACHE.clear()


def test_pasted_key_wins_over_env_provider_none(monkeypatch):
    from app.llm import make_llm
    monkeypatch.setenv("LLM_PROVIDER", "none")
    assert make_llm(None, None) is None
    assert make_llm(None, "sk-ant-abc").name == "anthropic"
    assert make_llm(None, "sk-proj-abc").name == "openai"


def test_session_endpoint_does_not_expose_the_transcript():
    d = _new()
    client.post("/api/chat", json={"session_id": d["session_id"], "message": "Margaret Chen, DOB 1985-03-15"})
    r = client.get(f"/api/session/{d['session_id']}").json()
    assert "transcript" not in r and "1985-03-15" not in str(r)


def test_events_carry_no_party_id():
    d = _new()
    r = client.post("/api/chat", json={"session_id": d["session_id"],
                                       "message": "Margaret Chen, DOB 1985-03-15, SSN last four 4472"})
    assert "P9" not in str(r.json()["state"])
