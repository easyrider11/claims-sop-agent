"""FastAPI app: chat API + the static test UI."""
from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import identity as idm
from .agent import Agent
from .data import REPO, today
from .llm import make_llm
from .sop import contract
from .state import Session

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
STATIC = Path(__file__).resolve().parent.parent / "static"

app = FastAPI(title="SOP-guided insurance claims agent")
agent = Agent()
SESSION_TTL_S = int(os.environ.get("SESSION_TTL_S", 2 * 3600))
MAX_SESSIONS = int(os.environ.get("MAX_SESSIONS", 1000))
SESSIONS: dict[str, Session] = {}
LAST_SEEN: dict[str, float] = {}
SESSION_LLM: dict[str, object] = {}
LOCKS: dict[str, threading.Lock] = {}
_LLM_CACHE: dict[str, object] = {}


class NewSession(BaseModel):
    consent_scenario: str = "default"


class ChatIn(BaseModel):
    session_id: str
    message: str = Field(min_length=1, max_length=2000)


def _gc() -> None:
    """Drop idle sessions (and the API key reference they hold); cap the total."""
    now = time.time()
    stale = [sid for sid, t in LAST_SEEN.items() if now - t > SESSION_TTL_S]
    if len(LAST_SEEN) - len(stale) >= MAX_SESSIONS:
        stale += sorted(set(LAST_SEEN) - set(stale), key=LAST_SEEN.get)[: len(LAST_SEEN) - len(stale) - MAX_SESSIONS + 1]
    for sid in stale:
        for d in (SESSIONS, SESSION_LLM, LOCKS, LAST_SEEN):
            d.pop(sid, None)


def _llm_for(provider: str | None, key: str | None):
    """Provider/key come from request headers (UI) or env, and are never logged. Env-configured clients
    are shared; a client for a key typed into the UI lives only in that session (dropped by _gc)."""
    if key:
        return make_llm(provider or None, key)
    if provider not in _LLM_CACHE:
        _LLM_CACHE[provider] = make_llm(provider or None, None)
    return _LLM_CACHE[provider]


def public_state(s: Session) -> dict:
    d = s.to_public()
    # never ship raw PII back to the browser beyond what the caller typed; mask it for the inspector
    d["identity"] = {k: idm.mask_value(k, v) for k, v in s.identity.items() if v}
    d["claims_unlocked"] = s.verified
    for internal in ("transcript", "matched_party_id", "verified_party_id", "identity_snapshot_last_checked",
                     "new_identity_fields", "flags"):
        d.pop(internal, None)
    if d.get("email_sent"):
        d["email_sent"] = {**d["email_sent"], "to": idm.mask_email(d["email_sent"]["to"])}
    if d.get("handoff"):
        d["handoff"] = {k: v for k, v in d["handoff"].items() if k != "party_id"}
    if s.verified:
        holder = REPO.policyholder(s.verified_party_id)
        d["verified_as"] = holder["name"]
    return d


@app.exception_handler(Exception)
async def _unhandled(request: Request, exc: Exception):
    logging.getLogger("server").exception("unhandled error on %s", request.url.path)
    return JSONResponse({"detail": f"internal error: {type(exc).__name__}"}, status_code=500)


def _llm_or_400(provider: str | None, key: str | None):
    try:
        return _llm_for(provider, key)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(400, f"could not set up the {provider or 'default'} model: {type(e).__name__}: {e}")


@app.get("/api/config")
def config():
    try:
        llm = _llm_for(None, None)
        err = None
    except Exception as e:
        llm, err = None, f"{type(e).__name__}: {e}"
    return {
        "env_provider": getattr(llm, "name", None) if llm else None,
        "env_model": getattr(llm, "model", None) if llm else None,
        "env_error": err,
        "today": today().isoformat(),
        "consent_scenarios": list(REPO.consent_scenarios.keys()),
    }


@app.post("/api/session")
def new_session(body: NewSession, x_llm_provider: str | None = Header(None), x_llm_key: str | None = Header(None)):
    _gc()
    llm = _llm_or_400(x_llm_provider, x_llm_key)
    s = Session(consent_scenario=body.consent_scenario if body.consent_scenario in REPO.consent_scenarios else "default")
    SESSIONS[s.id] = s
    SESSION_LLM[s.id] = llm
    LAST_SEEN[s.id] = time.time()
    LOCKS[s.id] = threading.Lock()
    opening = agent.opening(s)
    return {"session_id": s.id, "reply": opening, "state": public_state(s),
            "mode": f"{llm.name}:{llm.model}" if llm else "rules-only (no model)"}


@app.post("/api/chat")
def chat(body: ChatIn):
    s = SESSIONS.get(body.session_id)
    if not s:
        raise HTTPException(404, "unknown session; start a new one")
    LAST_SEEN[s.id] = time.time()
    with LOCKS.setdefault(s.id, threading.Lock()):
        out = agent.turn(s, body.message, SESSION_LLM.get(s.id))
    out["state"] = public_state(s)
    return out


@app.get("/api/session/{sid}")
def get_session(sid: str):
    """Masked state only. The transcript holds whatever the caller typed (DOB, last-4) and stays server-side."""
    s = SESSIONS.get(sid)
    if not s:
        raise HTTPException(404, "unknown session")
    return {"state": public_state(s), "turns": s.turn_count}


@app.get("/api/sop")
def sop_contract():
    """The SOP as data: phases, legal transitions, tools per phase, freedom per phase, gates."""
    return contract()


@app.get("/api/health")
def health():
    return {"ok": True, "sessions": len(SESSIONS)}


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
