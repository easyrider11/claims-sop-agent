"""Thin LLM adapters. The harness only needs two calls: structured JSON and short text.

Providers:
  anthropic  - Claude via the official SDK (default when ANTHROPIC_API_KEY or a UI key is given)
  openai     - OpenAI via the official SDK
  none       - no model; the harness runs on its rule-based NLU and reply templates
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from contextvars import ContextVar

log = logging.getLogger("llm")

# Per-turn token accounting: the agent sets a list, adapters append one entry per call.
USAGE: ContextVar[list | None] = ContextVar("llm_usage", default=None)


# USD per 1M tokens (input, output). Unknown models are priced high on purpose.
PRICES = {
    "gpt-4.1-nano": (0.10, 0.40), "gpt-4o-mini": (0.15, 0.60), "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1": (2.00, 8.00),
    "gpt-5.4-nano": (0.20, 1.25), "gpt-5.4-mini": (0.75, 4.50),  # estimates
    "claude-haiku-5-5": (0.10, 0.50), "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
}
_SPENT = {"usd": 0.0}
_SPEND_LOCK = threading.Lock()


def spent_usd() -> float:
    return _SPENT["usd"]


def _check_budget() -> None:
    """Optional hard cap for this process (MAX_SPEND_USD). Over the cap, calls fail and the
    harness falls back to rules/templates instead of spending more."""
    cap = os.environ.get("MAX_SPEND_USD")
    if cap and _SPENT["usd"] >= float(cap):
        raise LLMError(f"spend cap reached (${_SPENT['usd']:.3f} >= ${float(cap):.2f})")


def _record(kind: str, model: str, inp: int | None, out: int | None, ms: int) -> None:
    pin, pout = PRICES.get(model, (10.0, 40.0))
    cost = ((inp or 0) * pin + (out or 0) * pout) / 1e6
    with _SPEND_LOCK:
        _SPENT["usd"] += cost
    acc = USAGE.get()
    if acc is not None:
        acc.append({"call": kind, "model": model, "input_tokens": inp, "output_tokens": out, "ms": ms,
                    "usd": round(cost, 5)})


class LLMError(RuntimeError):
    pass


class BaseLLM:
    name = "none"
    model = ""

    def json(self, system: str, user: str, schema: dict, max_tokens: int = 4000) -> dict:
        raise NotImplementedError

    def text(self, system: str, user: str, max_tokens: int = 4000) -> str:
        raise NotImplementedError


def _loose_json(text: str) -> dict:
    text = text.strip()
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise LLMError(f"no JSON object in model output: {text[:200]!r}")
    return json.loads(m.group(0))


class AnthropicLLM(BaseLLM):
    name = "anthropic"
    FALLBACK_BETA = "server-side-fallback-2026-07-01"

    def __init__(self, api_key: str | None = None, model: str | None = None):
        import anthropic

        self._anthropic = anthropic
        self.client = anthropic.Anthropic(api_key=api_key, timeout=30.0, max_retries=1) if api_key \
            else anthropic.Anthropic(timeout=30.0, max_retries=1)
        if not (getattr(self.client, "api_key", None) or getattr(self.client, "auth_token", None)):
            # The SDK only complains at request time (with a bare TypeError); fail at setup instead.
            raise ValueError("Anthropic selected but no API key: set ANTHROPIC_API_KEY or paste a key")
        self.model = model or os.environ.get("ANTHROPIC_MODEL", "claude-opus-5-5")
        self.effort: str | None = os.environ.get("ANTHROPIC_EFFORT", "low")
        self.use_fallbacks = os.environ.get("ANTHROPIC_FALLBACKS", "1") != "0"
        self.use_format = True

    @staticmethod
    def _error_text(e) -> str:
        body = getattr(e, "body", None)
        if isinstance(body, dict):
            err = body.get("error") or {}
            return str(err.get("message") or body).lower() if isinstance(err, dict) else str(body).lower()
        return str(e).lower()

    def _create(self, system: str, user: str, max_tokens: int, fmt: dict | None):
        output_config: dict = {}
        if self.effort:
            output_config["effort"] = self.effort
        if fmt is not None:
            output_config["format"] = fmt
        kwargs = dict(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        if output_config:
            kwargs["output_config"] = output_config
        if self.use_fallbacks:
            # On a safety decline the API re-runs the request on Anthropic's recommended fallback model.
            kwargs.update(betas=[self.FALLBACK_BETA], fallbacks="default")
        _check_budget()
        t0 = time.perf_counter()
        try:
            resp = self.client.beta.messages.create(**kwargs)
        except self._anthropic.BadRequestError as e:
            # Step down one feature at a time, matching on the parameter the API names.
            msg = self._error_text(e)
            if self.use_fallbacks and "fallback" in msg:
                log.warning("fallbacks rejected, retrying without: %s", e)
                self.use_fallbacks = False
                return self._create(system, user, max_tokens, fmt)
            if fmt is not None and ("output_config.format" in msg or "json_schema" in msg or "schema" in msg):
                log.warning("structured output rejected, retrying with prompt-only JSON: %s", e)
                self.use_format = False
                return self._create(system, user, max_tokens, None)
            if self.effort and "effort" in msg:
                log.warning("effort rejected by %s, retrying without: %s", self.model, e)
                self.effort = None
                return self._create(system, user, max_tokens, fmt)
            raise LLMError(f"anthropic 400: {e}") from e
        except self._anthropic.APIStatusError as e:
            raise LLMError(f"anthropic {e.status_code}: {e}") from e
        except self._anthropic.APIConnectionError as e:
            raise LLMError(f"anthropic connection error: {e}") from e
        except LLMError:
            raise
        except Exception as e:  # SDK-level surprises (e.g. credential errors raised lazily) must not 500 a turn
            raise LLMError(f"anthropic error: {type(e).__name__}: {e}") from e
        u = getattr(resp, "usage", None)
        _record("json" if fmt is not None else "text", self.model, getattr(u, "input_tokens", None),
                getattr(u, "output_tokens", None), int((time.perf_counter() - t0) * 1000))
        if resp.stop_reason == "refusal":
            raise LLMError("model declined the request")
        if resp.stop_reason == "max_tokens":
            raise LLMError("response hit max_tokens")
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        if not text.strip():
            raise LLMError(f"empty response (stop_reason={resp.stop_reason})")
        return text

    def json(self, system, user, schema, max_tokens=4000):
        fmt = {"type": "json_schema", "schema": schema} if self.use_format else None
        if fmt is None:
            user = user + "\n\nRespond with a single JSON object only."
        return _loose_json(self._create(system, user, max_tokens, fmt))

    def text(self, system, user, max_tokens=4000):
        return self._create(system, user, max_tokens, None).strip()


class OpenAILLM(BaseLLM):
    """OpenAI, or any OpenAI-compatible endpoint (e.g. SiliconFlow) via base_url."""
    name = "openai"

    def __init__(self, api_key: str | None = None, model: str | None = None, base_url: str | None = None,
                 name: str = "openai", timeout: float = 30.0):
        import openai

        self._openai = openai
        self.name = name
        kwargs = dict(timeout=timeout, max_retries=1)
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        try:
            self.client = openai.OpenAI(**kwargs)
        except openai.OpenAIError as e:   # no key anywhere
            raise ValueError(f"{name} selected but no API key: set {'SILICONFLOW' if name == 'siliconflow' else 'OPENAI'}"
                             f"_API_KEY or paste a key ({e})") from e
        self._modes_lock = threading.Lock()
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")
        # Compatible endpoints differ in what they accept; degrade one step at a time on a 400.
        self.json_modes = ["json_schema", "json_object", "prompt"]
        self.token_param = "max_completion_tokens" if not base_url else "max_tokens"
        extra = os.environ.get("OPENAI_EXTRA_BODY")
        self.extra_body = json.loads(extra) if extra else None
        # reasoning models (gpt-5*, o*) take an effort knob; low keeps chat latency down
        self.reasoning_effort = os.environ.get("OPENAI_REASONING_EFFORT", "low") \
            if re.match(r"(gpt-5|o\d)", self.model) else None

    def _chat(self, system, user, max_tokens, response_format=None):
        kwargs = dict(model=self.model, messages=[{"role": "system", "content": system},
                                                  {"role": "user", "content": user}])
        kwargs[self.token_param] = max_tokens
        if response_format:
            kwargs["response_format"] = response_format
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        _check_budget()
        t0 = time.perf_counter()
        try:
            resp = self.client.chat.completions.create(**kwargs)
        except self._openai.BadRequestError as e:
            raise _BadRequest(str(e)) from e
        except self._openai.APIStatusError as e:
            raise LLMError(f"{self.name} {e.status_code}: {e}") from e
        except self._openai.APIConnectionError as e:
            raise LLMError(f"{self.name} connection error: {e}") from e
        u = getattr(resp, "usage", None)
        _record("json" if response_format else "text", self.model, getattr(u, "prompt_tokens", None),
                getattr(u, "completion_tokens", None), int((time.perf_counter() - t0) * 1000))
        choice = resp.choices[0]
        if choice.finish_reason == "length":
            raise LLMError("response hit max_tokens")
        text = choice.message.content or ""
        if not text.strip():
            raise LLMError("empty response")
        return text

    def json(self, system, user, schema, max_tokens=4000):
        while True:
            mode = self.json_modes[0]
            if mode == "json_schema":
                rf = {"type": "json_schema", "json_schema": {"name": "result", "schema": schema, "strict": True}}
                prompt = user
            elif mode == "json_object":
                rf = {"type": "json_object"}
                prompt = f"{user}\n\nRespond with one JSON object matching this JSON schema:\n{json.dumps(schema)}"
            else:
                rf = None
                prompt = (f"{user}\n\nRespond with ONLY one JSON object (no prose, no code fences) matching this "
                          f"JSON schema:\n{json.dumps(schema)}")
            try:
                return _loose_json(self._chat(system, prompt, max_tokens, rf))
            except _BadRequest as e:
                with self._modes_lock:
                    if self.json_modes[0] != mode:      # another thread already stepped down
                        continue
                    if len(self.json_modes) == 1:
                        raise LLMError(f"{self.name} 400: {e}") from e
                    log.warning("%s rejected %s, falling back: %s", self.name, mode, str(e)[:200])
                    self.json_modes.pop(0)
            except json.JSONDecodeError as e:
                raise LLMError(f"invalid JSON from model: {e}") from e

    def text(self, system, user, max_tokens=4000):
        try:
            return self._chat(system, user, max_tokens).strip()
        except _BadRequest as e:
            raise LLMError(f"{self.name} 400: {e}") from e


class _BadRequest(LLMError):
    pass


SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"


def make_llm(provider: str | None = None, api_key: str | None = None) -> BaseLLM | None:
    """Pick a provider: an explicit provider wins; otherwise a pasted key is sniffed by prefix (it must
    not be silently ignored because the server's env says LLM_PROVIDER=none); otherwise the env.
    Returns None for rules-only mode. Raises ValueError when the chosen provider has no usable key."""
    if not provider and api_key:
        provider = "anthropic" if api_key.startswith("sk-ant-") else "openai"
    provider = (provider or os.environ.get("LLM_PROVIDER") or "auto").lower()
    if provider == "none":
        return None
    if provider == "auto":
        if api_key:
            provider = "anthropic" if api_key.startswith("sk-ant-") else "openai"
        elif os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
            provider = "anthropic"
        elif os.environ.get("OPENAI_API_KEY"):
            provider = "openai"
        elif os.environ.get("SILICONFLOW_API_KEY"):
            provider = "siliconflow"
        else:
            return None
    if provider == "anthropic":
        return AnthropicLLM(api_key=api_key)
    if provider == "openai":
        return OpenAILLM(api_key=api_key, base_url=os.environ.get("OPENAI_BASE_URL") or None)
    if provider == "siliconflow":
        return OpenAILLM(api_key=api_key or os.environ.get("SILICONFLOW_API_KEY"),
                         base_url=os.environ.get("SILICONFLOW_BASE_URL", SILICONFLOW_BASE_URL),
                         model=os.environ.get("SILICONFLOW_MODEL", "deepseek-ai/DeepSeek-V3"),
                         name="siliconflow", timeout=60.0)
    raise ValueError(f"unknown provider {provider}")


class Timer:
    def __init__(self):
        self.t0 = time.perf_counter()

    def ms(self) -> int:
        return int((time.perf_counter() - self.t0) * 1000)
