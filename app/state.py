"""Session state: the SOP position plus cross-phase memory."""
from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum


class Phase(str, Enum):
    VERIFY_ID = "VERIFY_ID"
    RESOLVE_INTENT = "RESOLVE_INTENT"
    PROCESS_CASE = "PROCESS_CASE"
    POST_PROCESS = "POST_PROCESS"
    CLOSED = "CLOSED"        # conversation finished normally
    HANDOFF = "HANDOFF"      # transferred to a human representative


@dataclass
class Memory:
    """Things the caller said that matter later, captured whenever they are said.

    Example: 'I'm calling about my denied healthcare claim from January' arrives during
    VERIFY_ID. It is stored here and only *used* once the caller is verified.
    """
    intent: str | None = None                  # status_inquiry, denial_question, ...
    case_type: str | None = None               # healthcare | dental | auto
    case_status: str | None = None             # denied | open | closed
    case_month: int | None = None              # 1-12
    case_year: int | None = None
    case_id: str | None = None                 # e.g. CL-2048 if the caller quoted it
    notes: list[str] = field(default_factory=list)   # free-form facts ("doesn't have the office note yet")
    pending_question: str | None = None        # the caller's question we deferred until after verification


@dataclass
class Session:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    phase: Phase = Phase.VERIFY_ID

    # VERIFY_ID
    identity: dict[str, str | None] = field(default_factory=dict)   # latest value per field
    identity_snapshot_last_checked: dict | None = None
    new_identity_fields: list[str] = field(default_factory=list)   # fields that changed this turn
    failed_verifications: int = 0
    verified_party_id: str | None = None         # set only when ALL gates pass (identity + rep consent)
    matched_party_id: str | None = None          # identity matched, but a representative still needs consent
    caller_role: str = "policyholder"            # policyholder | representative
    rep_name: str | None = None
    rep_relationship: str | None = None
    rep_authorized: bool | None = None
    rep_attempts: int = 0
    consent_scenario: str = "default"
    consent_status: str | None = None            # None | pending | approved | timeout
    consent_polls: int = 0

    # RESOLVE_INTENT / PROCESS_CASE
    memory: Memory = field(default_factory=Memory)
    case_id: str | None = None
    candidate_case_ids: list[str] = field(default_factory=list)
    topics_discussed: list[str] = field(default_factory=list)
    topics_by_case: dict[str, list[str]] = field(default_factory=dict)   # case_id -> paths, in order discussed

    # POST_PROCESS
    email_choice: str | None = None              # sent | skipped
    email_sent: dict | None = None

    # conversation control
    pending_offer: str | None = None             # human_transfer | email_summary
    off_topic_strikes: int = 0
    persuasion_attempts: int = 0                 # SOP explanations given to a resisting caller
    last_emotion: str = "neutral"
    handoff: dict | None = None

    transcript: list[dict] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)

    def log(self, kind: str, **data) -> None:
        self.events.append({"turn": self.turn_count, "kind": kind, **data})

    def events_flag(self, name: str) -> bool:
        return name in self.flags

    def events_flag_set(self, name: str) -> None:
        if name not in self.flags:
            self.flags.append(name)

    @property
    def turn_count(self) -> int:
        return sum(1 for m in self.transcript if m["role"] == "user")

    @property
    def verified(self) -> bool:
        return self.verified_party_id is not None

    def to_public(self) -> dict:
        d = asdict(self)
        d["phase"] = self.phase.value
        d.pop("identity_snapshot_last_checked", None)
        return d
