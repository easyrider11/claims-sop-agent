"""Adapter behaviour with the SDK clients stubbed out (no network)."""
from types import SimpleNamespace

import anthropic
import httpx
import openai
import pytest

from app import llm as llm_mod
from app.llm import USAGE, AnthropicLLM, LLMError, OpenAILLM

REQ = httpx.Request("POST", "http://test")


def _bad(cls, msg):
    return cls(msg, response=httpx.Response(400, request=REQ), body=None)


def _oa_resp(content, finish="stop"):
    return SimpleNamespace(choices=[SimpleNamespace(finish_reason=finish, message=SimpleNamespace(content=content))],
                           usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20))


def _an_resp(text, stop="end_turn"):
    return SimpleNamespace(stop_reason=stop, content=[SimpleNamespace(type="text", text=text)],
                           usage=SimpleNamespace(input_tokens=100, output_tokens=20))


@pytest.fixture(autouse=True)
def _fresh_spend(monkeypatch):
    monkeypatch.setitem(llm_mod._SPENT, "usd", 0.0)
    monkeypatch.delenv("MAX_SPEND_USD", raising=False)


def test_openai_steps_down_json_modes_on_400():
    llm = OpenAILLM(api_key="x", base_url="http://compat.example/v1", model="some-model")
    seen = []

    def create(**kw):
        seen.append(kw.get("response_format", {}).get("type") if kw.get("response_format") else None)
        if len(seen) == 1:
            raise _bad(openai.BadRequestError, "json_schema not supported")
        return _oa_resp('{"ok": true}')

    llm.client.chat.completions.create = create
    assert llm.json("sys", "user", {"type": "object"}) == {"ok": True}
    assert seen == ["json_schema", "json_object"]
    assert llm.json_modes[0] == "json_object"            # remembered for the next call


def test_openai_compatible_uses_max_tokens_and_records_usage():
    llm = OpenAILLM(api_key="x", base_url="http://compat.example/v1", model="gpt-4.1-mini")
    captured = {}

    def create(**kw):
        captured.update(kw)
        return _oa_resp("hello")

    llm.client.chat.completions.create = create
    usage = []
    tok = USAGE.set(usage)
    try:
        assert llm.text("sys", "user", max_tokens=50) == "hello"
    finally:
        USAGE.reset(tok)
    assert captured["max_tokens"] == 50 and "max_completion_tokens" not in captured
    assert usage[0]["input_tokens"] == 100 and usage[0]["usd"] > 0


def test_openai_truncated_output_is_an_error():
    llm = OpenAILLM(api_key="x", model="gpt-4.1-mini")
    llm.client.chat.completions.create = lambda **kw: _oa_resp('{"a": ', finish="length")
    with pytest.raises(LLMError, match="max_tokens"):
        llm.json("sys", "user", {"type": "object"})


def test_anthropic_retries_without_fallbacks_when_rejected():
    llm = AnthropicLLM(api_key="x")
    calls = []

    def create(**kw):
        calls.append(kw)
        if len(calls) == 1:
            raise _bad(anthropic.BadRequestError, "fallbacks: not permitted for this account")
        return _an_resp('{"ok": 1}')

    llm.client.beta.messages.create = create
    assert llm.json("sys", "user", {"type": "object"}) == {"ok": 1}
    assert "fallbacks" in calls[0] and "fallbacks" not in calls[1]
    assert calls[1]["output_config"]["format"]["type"] == "json_schema"
    assert calls[1]["output_config"]["effort"] == "low"


def test_anthropic_refusal_is_an_error():
    llm = AnthropicLLM(api_key="x")
    llm.client.beta.messages.create = lambda **kw: _an_resp("", stop="refusal")
    with pytest.raises(LLMError, match="declined"):
        llm.text("sys", "user")


def test_spend_cap_stops_calls(monkeypatch):
    llm = OpenAILLM(api_key="x", model="gpt-4.1-mini")
    llm.client.chat.completions.create = lambda **kw: _oa_resp("hi")
    monkeypatch.setenv("MAX_SPEND_USD", "0.00001")
    llm.text("sys", "user")                               # this call pushes spend over the cap
    with pytest.raises(LLMError, match="spend cap"):
        llm.text("sys", "user")
