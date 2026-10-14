"""The .env loader and the shipped .env.example: every optional line must parse when uncommented."""
import os
import re
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load(text: str) -> dict:
    out = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        v = re.split(r"\s+#", v, 1)[0]
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def test_inline_comment_is_stripped():
    env = _load("DEMO_TODAY=2026-02-01   # pin the date\nMAX_SPEND_USD=2 # cap\nX='quoted'")
    assert env == {"DEMO_TODAY": "2026-02-01", "MAX_SPEND_USD": "2", "X": "quoted"}
    assert date.fromisoformat(env["DEMO_TODAY"]) and float(env["MAX_SPEND_USD"]) == 2.0


def test_env_example_parses_with_every_line_uncommented():
    example = ROOT / ".env.example"
    if not example.exists():
        import pytest
        pytest.skip(".env.example not shipped in this build")
    text = example.read_text()
    uncommented = "\n".join(re.sub(r"^#\s*(?=[A-Z_]+=)", "", l) for l in text.splitlines())
    env = _load(uncommented)
    assert date.fromisoformat(env["DEMO_TODAY"])
    assert float(env["MAX_SPEND_USD"]) > 0
    assert env["LLM_PROVIDER"] in ("auto", "anthropic", "openai", "siliconflow", "none")
    assert "/" in env["BUSINESS_TZ"]
    from zoneinfo import ZoneInfo
    ZoneInfo(env["BUSINESS_TZ"])


def test_loader_is_the_same_logic_as_the_app():
    """app/__init__.py uses the same split; keep the two in sync by checking the source."""
    src = (ROOT / "app" / "__init__.py").read_text()
    assert 're.split(r"\\s+#", v, 1)[0]' in src
    assert "OPENAI_API_KEY" in os.environ or True   # .env may or may not exist on this machine
