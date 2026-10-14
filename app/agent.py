"""One conversational turn: understand -> SOP step -> write reply -> guard -> record."""
from __future__ import annotations

import logging

from . import nlu, responder
from .audit import audit
from .data import claim_brief
from .llm import USAGE, BaseLLM, Timer
from .sop import SOP, SOPViolation
from .state import Phase, Session

log = logging.getLogger("agent")

FAIL_CLOSED_REPLY = ("Sorry, something went wrong on my side. I'm connecting you with a human representative "
                     "who can help.")


class Agent:
    def __init__(self, sop: SOP | None = None):
        self.sop = sop or SOP()
        self.repo = self.sop.repo

    def opening(self, s: Session) -> str:
        msg = ("Hi, you've reached claims support. I can help with questions about your insurance claims. "
               "To get started, I'll need to verify your identity - what can I help you with today?")
        s.transcript.append({"role": "assistant", "content": msg})
        return msg

    def turn(self, s: Session, text: str, llm: BaseLLM | None) -> dict:
        t = Timer()
        usage: list = []
        token = USAGE.set(usage)
        try:
            return self._turn(s, text, llm, t, usage)
        finally:
            USAGE.reset(token)

    def _turn(self, s: Session, text: str, llm: BaseLLM | None, t: Timer, usage: list) -> dict:
        text = text.strip()[:2000]
        s.transcript.append({"role": "user", "content": text})

        rule_sig = nlu.rules_extract(text, s.phase.value)
        llm_sig, nlu_error = None, None
        # Cheap path: a bare yes/no answering a question we just asked needs no model call.
        trivial = bool(s.pending_offer) and nlu.is_bare_answer(text)
        if trivial:
            rule_sig.source = "rules (short answer, model skipped)"
        if llm is not None and not trivial:
            briefs = [claim_brief(c) for c in self.repo.claims_for(s.verified_party_id)] if s.verified else None
            try:
                llm_sig = nlu.llm_extract(llm, s.phase.value, s.transcript[:-1], text, briefs, s.pending_offer)
            except Exception as e:  # any model/parse failure degrades to rules, never to a 500
                nlu_error = f"{type(e).__name__}: {e}"
                log.warning("LLM extraction failed, using rules: %s", nlu_error)
        sig = nlu.merge(llm_sig, rule_sig)
        nlu_ms = t.ms()
        mode = "rules-only (no model)" if llm is None else f"{llm.name}:{llm.model}"

        try:
            plan = self.sop.step(s, sig, text)
            reply, render_info = responder.render(plan, s, llm, repo=self.repo)
        except Exception as e:
            # A harness bug must fail closed: no model call, no data, hand the caller to a person.
            kind = "sop_violation" if isinstance(e, SOPViolation) else "internal_error"
            log.exception("%s in session %s", kind, s.id)
            s.log(kind, error=f"{type(e).__name__}: {e}")
            s.phase = Phase.HANDOFF
            s.pending_offer = None
            s.handoff = {"reason": "internal error", "verified": s.verified, "caller_role": s.caller_role,
                         "case_id": s.case_id}
            s.transcript.append({"role": "assistant", "content": FAIL_CLOSED_REPLY})
            plan = _FailClosedPlan()
            render_info = {"writer": "fixed (fail-closed)", "guard": []}
            audit(s, plan, sig, render_info, usage, t.ms(), True)
            return {"reply": FAIL_CLOSED_REPLY,
                    "trace": self._trace(mode, True, sig, nlu_error, plan, render_info, usage, nlu_ms, t.ms(),
                                         error=f"{type(e).__name__}: {e}")}

        s.transcript.append({"role": "assistant", "content": reply})
        # "degraded" means a configured model was bypassed; rules-only mode is the mode, not a fallback.
        degraded = llm is not None and (nlu_error is not None or bool(render_info.get("llm_error")))
        audit(s, plan, sig, render_info, usage, t.ms(), degraded)
        return {"reply": reply,
                "trace": self._trace(mode, degraded, sig, nlu_error, plan, render_info, usage, nlu_ms, t.ms())}

    @staticmethod
    def _trace(mode, degraded, sig, nlu_error, plan, render_info, usage, nlu_ms, total_ms, error=None) -> dict:
        return {
            "mode": mode,
            "degraded": bool(degraded),
            "error": error,
            "signals": sig.to_dict(),
            "nlu_error": nlu_error,
            "plan": {
                "phase_before": plan.phase_before,
                "phase_after": plan.phase_after,
                "steps": plan.steps,
                "freedom": plan.freedom,
                "actions": plan.actions,
                "phase_path": plan.phase_path,
                "fact_keys": sorted(plan.facts.keys()),
            },
            "render": render_info,
            "usage": usage,
            "nlu_ms": nlu_ms,
            "total_ms": total_ms,
        }


class _FailClosedPlan:
    """What the trace and audit log see when a turn fails closed."""
    phase_before = "?"
    phase_after = "HANDOFF"
    steps = ["FAIL_CLOSED"]
    freedom = "strict"
    actions: list = []
    phase_path = ["HANDOFF"]
    facts: dict = {}
