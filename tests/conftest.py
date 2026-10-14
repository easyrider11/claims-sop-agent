import os

import pytest

from app.agent import Agent
from app.data import Outbox
from app.sop import SOP
from app.state import Session


@pytest.fixture(autouse=True)
def _pin_env(monkeypatch, tmp_path):
    monkeypatch.setenv("DEMO_TODAY", "2026-10-13")
    monkeypatch.setenv("AUDIT_DIR", str(tmp_path / "logs"))


@pytest.fixture
def agent(tmp_path):
    return Agent(SOP(outbox=Outbox(tmp_path / "outbox")))


class Convo:
    """Drive a conversation and keep every reply/trace for assertions."""

    def __init__(self, agent, llm=None, **session_kwargs):
        self.agent, self.llm = agent, llm
        self.s = Session(**session_kwargs)
        self.agent.opening(self.s)
        self.turns = []

    def say(self, text):
        out = self.agent.turn(self.s, text, self.llm)
        self.turns.append(out)
        return out

    @property
    def phase(self):
        return self.s.phase.value


@pytest.fixture
def convo(agent):
    return lambda **kw: Convo(agent, **kw)


DEMO = ("I'm the policyholder. My name is Margaret Chen, policy POL-9921. I'm calling about my denied "
        "healthcare claim from January. DOB is 1985-03-15, SSN last four is 4472.")
CLAIM_SECRETS = ["CL-2048", "pathology", "office note", "1450", "2026-03-18", "CL-2102", "CL-2011"]


def assert_no_claim_leak(text):
    low = text.lower()
    for secret in CLAIM_SECRETS:
        assert secret.lower() not in low, f"leaked {secret!r}: {text}"


os.environ.setdefault("LLM_PROVIDER", "none")
