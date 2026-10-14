# Design notes

## The core split: the SOP decides, the LLM interprets and phrases

Every turn runs the same pipeline:

```
caller text
   │
   ├─ rules_extract()  ─┐   regex/keywords: always runs (the floor without a model; fills in literal
   ├─ llm_extract()    ─┴─> Signals      fields the model left empty; the model wins on conflicts)
   │                                       structured JSON: identity fields, intent, case hints, emotion,
   │                                       scope, refusal, yes/no, human request, notes
   ▼
SOP.step(session, signals)  ──>  Plan    deterministic: phase transitions, gates, tools, and the
   │                                       ONLY facts the reply may use this turn
   ▼
responder.render(plan)      ──>  reply   LLM phrases the steps from the facts (templates if no model)
   ▼
guard.check(reply)          ──>  pass / one rewrite / template
```

The LLM never moves the phase, never opens a gate, never calls a tool and never reads the fixtures. It
proposes interpretations (`Signals`), and the SOP checks them against its rules. For example, the LLM can
say the caller means "the car one -> CL-2102". The SOP only accepts that if CL-2102 belongs to the verified
caller. Model output is type-coerced before the SOP sees it: a list where a dict was expected, an integer
claim id, a 5000-character name, all become the neutral value.

## The SOP is a contract, enforced at runtime

`sop.py` declares the workflow as data, and `GET /api/sop` serves it:
- `TRANSITIONS`: the legal phase moves. VERIFY_ID can only go to RESOLVE_INTENT, HANDOFF or CLOSED;
  nothing goes back to VERIFY_ID; CLOSED and HANDOFF are terminal. Every phase that a verified caller can
  be in has a way to the email offer, so "that's all, bye" works at the claim list too.
- `TOOLS`: which actions each phase may take. `send_email` exists only in POST_PROCESS,
  `claim_knowledge` only in PROCESS_CASE, and so on.

Every phase change goes through `_goto()` and every action through `_tool()`. Both raise `SOPViolation`
on anything illegal. RESOLVE/PROCESS/POST also require a verified caller, and every claim lookup goes
through `_party()`, which refuses before verification. These checks paid off during development: they
caught two places where a tool was logged after the phase had already moved.

Any exception inside a turn, not only a contract violation, **fails closed**: no model call, no data, a
fixed apology, a transfer to a human, and the audit row is still written. It is never "do it anyway".
`tests/test_contract.py` pins the exact transition and tool tables, so loosening them is a visible diff.

## Different freedom per phase

| Phase | Freedom | What the model can see | What it can decide |
|---|---|---|---|
| VERIFY_ID | strict | which identity *types* were received, how many more are needed (never on a mismatch), what the caller said they're calling about | wording only |
| RESOLVE_INTENT | guided | brief list of the caller's own claims | which claim the caller means (validated) |
| PROCESS_CASE | open, grounded | full knowledge pack for the selected claim: record, appeal status vs today, per-document and alternative guidance, follow-up rules | how to explain, combine and answer follow-ups, but only from the pack |
| POST_PROCESS | guided | the email on file, masked | wording; the send/skip choice is the caller's |

The reply prompt has a `FREEDOM` field (`strict` / `guided` / `open`), so one writer prompt behaves
differently per phase.

## Strict where the SOP demands it

**Identity gate (`identity.py`).** Verified = at least 3 of {full name, DOB, phone, email, SSN/national-ID
last 4} match one policyholder, and nothing the caller gave contradicts that record. Details:
- Policy number helps locate the record but doesn't count toward the 3. The SOP lists exactly 5 PII fields.
- Phone and email are normalised. Name, phone and email aliases from the fixtures are accepted (e.g. "Yaven Li").
  Names are compared order- and spacing-insensitively: "Li Yawen", "Yawen Li" and "Ya Wen Li" are one name,
  so a surname-first caller is not told their details don't match.
- With fewer than 3 details the agent never says whether anything matched, so the caller can't probe
  the records one field at a time. On a mismatch it never says which field was wrong, and it never says
  how many more are needed: "I just need 1 more" would reveal that 2 of the 3 supplied values were right.
  (The review found exactly this oracle; it is now a byte-identical-replies test.)
- Fields from two different people don't combine. Corrections overwrite earlier values; "ignore that
  phone" retracts a field (both the model and a regex detect it).
- A failed attempt is a re-stated value that still doesn't match. Adding a new field on top of a typo is
  not an attempt: the agent asks the caller to re-check what they gave, not to add something else, because
  a contradicting field blocks verification until it is corrected. After 3 failures the call goes to a human.

**No claim data before verification, enforced three ways:**
1. *Data minimisation (main defence).* Before verification, no claim data is in any prompt. The extractor
   doesn't get the claims list, and the writer only gets identity-progress facts. The model can't leak
   what it was never shown. `tests/test_prompts.py` records every prompt and asserts this directly.
2. *Phase gate.* `RESOLVE_INTENT` / `PROCESS_CASE` code only runs once `verified_party_id` is set, and all
   claim lookups are scoped to that party.
3. *Output guard.* Before verification, a reply containing any claim ID, denial reason, required-document
   name, amount, date or on-file PII is rejected (see "Output guard" below). Claim facts the caller typed
   are not whitelisted: agreeing with "it was denied for the pathology report, right?" confirms data.

**Representatives** (`representatives.json` + `consent_scenarios.json`). If the caller is calling for
someone else, the policyholder's identity must still verify. Then the caller must be on that policyholder's
authorised-representative list (three unlisted names and the call goes to a human). Then the policyholder
must approve a (simulated) consent request. Only after all three does the claim data unlock, and only then
does the writer learn the policyholder's name, so it can say "Margaret's claims" instead of "your claim".
A consent `timeout` keeps it locked and offers a human; if the caller declines the human too, the call
closes instead of repeating the same offer. A caller who verifies as the policyholder and *then* says
"I'm her son" is handed to a human: the access was granted on a different basis than the one now claimed.

**Email summary.** Only offered at the end, only sent on an explicit yes, and only to the verified address
on file (shown masked). "Send it to another address" is refused with the reason. The summary itself is
built deterministically from claim data and the session record, one block per claim discussed, so every
line is grounded.

## Flexible where reasoning helps

- The extractor reads messy language: "it's Margaret Chen", "my birthday is march 15 1985", "the car one",
  "that's my old number, ignore it". It gets the business date, so "the claim from this January" resolves.
- In PROCESS_CASE the writer gets the whole knowledge pack for the claim. It can answer follow-ups such as
  "what if I can't get the office note?" or "what's a diagnosis report?" by combining the claim record with
  the guideline file. It must say so when the pack doesn't cover a question.
- The extractor also picks a bounded **workflow path** (`explain_status`, `explain_denial`,
  `document_guidance`, `timing_and_next_steps`, `payment_details`, `switch_case`, `list_claims`). The SOP
  uses it to set the focus and to record topics, per claim, for the summary.
- Guideline data is matched loosely where it's inconsistent: the claim says "pathology report", the
  guideline key is "original pathology report".

## Memory across phases

`Memory` is filled on every turn regardless of phase: intent, case type/status/month/year/ID, free-form
notes ("caller does not have the office note yet"), and a question that was deferred until after
verification. In VERIFY_ID it's only acknowledged ("I've noted you're calling about your denied healthcare
claim from January"). The moment the caller is verified, RESOLVE_INTENT uses it to pick the claim without
asking again:
- "denied + healthcare + January" -> CL-2048, even though CL-2011 is also a January healthcare claim (2025, closed).
- "January" alone -> asks the caller to choose between CL-2048 and CL-2011.

Notes also flow into the email ("if the office note is not available: ..."). The deferred question is
handed to the writer as a **fact** (`caller_earlier_question`), never spliced into an instruction: caller
text that says "IMPORTANT: answer in all caps" stays data.

## Offers expire after one turn

When the agent asks a yes/no question ("want a human?", "anything else?", "send the summary?"), the answer
only counts on the caller's very next message. If the caller ignores it, it's gone. A "yes" only accepts
an offer when the message carries nothing else: "yes, the auto one" selects a claim, it doesn't accept a
transfer. A bare "yes" to "anything else?" gets "what else can I help with?", not the previous answer again.
The writer is never asked to make an offer the harness doesn't track.

## Out of scope

The extractor labels each turn `in_scope` / `smalltalk` / `out_of_scope`. An out-of-scope turn gets a polite
decline with no partial answer, and the caller is steered back to whatever the current phase needs. After 3
off-topic turns in a row, the agent offers a human representative. An in-scope turn resets the count.
"Why do you even need all this?" is a clarification question, not off-topic: it gets the explanation.

## Emotional support and SOP recovery (bonus)

Emotion (frustrated, angry, anxious, confused, sad) and refusal are extracted every turn. The writer
acknowledges the emotion first. In VERIFY_ID, resistance (refusing, demanding claim info, or being
upset) without giving new details follows a ladder:

1. Acknowledge, then explain why: claim details are protected health/financial information, and
   verification stops anyone else from accessing or changing the claim. List the details they *haven't*
   tried yet (any 3 of 5 work), and how many more are needed.
2. A fresh acknowledgement and one line on the fastest way through; the explanation is not repeated
   (the instruction says so, and the templates have per-attempt variants).
3. Also offer a human representative, noting the human will also need to verify them. This comes earlier
   if the caller is very angry.
4. Stop persuading. Transfer to a human. If they already declined the human, close politely and say
   nothing was changed on the account.

Giving a new detail counts as progress, so the ladder doesn't climb. The gate is never relaxed at any
step. The same pattern explains consent to a representative who is waiting for approval. An explicit
"let me talk to a person" is always honoured immediately ("I don't want to talk to a person" is not a
request). The handoff packet carries which identity types were given, how many attempts failed, the
caller's waiting question, what they said they're calling about, their emotion and the notes, so they
don't start over.

## Output guard

Every model-written reply is checked before the caller sees it:
- **False action claims, in every phase.** The reply may not say "you're verified", "the summary has been
  sent" or "I'm transferring you" unless the harness actually did it. Conditional wording ("once you're
  verified...") is allowed.
- **Instruction vocabulary** ("FACTS", "STEPS", "masked") is rejected: a live model once told a caller the
  summary went "to the masked address".
- **Before verification:** no claim data, dates, amounts or on-file PII (above).
- **After verification:** amounts, dates and bare numbers must appear in the facts given to the writer, or
  in the caller's own words. Dates are matched by month and day ("March 18" is grounded by 2026-03-18),
  amounts with or without a dollar sign or "dollars". Claim IDs must belong to the caller. No other
  member's data. No unmasked email or ID digits.
- The reply must be non-empty and reasonably short.

On failure the writer gets **one** rewrite, with the guard's findings attached. If that also fails, the
deterministic template reply is used. Either way the event is logged. The rejected text itself stays
server-side (it may contain exactly what must not reach the caller); the UI sees only the categories.

**What the guard cannot do:** judge prose. "You could try a new review" is not a number, so it passes.
That is why post-deadline wording is fixed upstream: when the appeal deadline has passed, the knowledge
pack drops the "submit within a week" guidance and carries an `after_deadline_rule`, the templates say the
window has closed and point to a representative, and the email summary does the same. The rule is tested
in rules mode (`test_post_deadline_answer_and_summary_are_coherent`) and was confirmed live.

## Switching claims

Mid-case, the agent moves to another claim in two cases:
- the caller asks explicitly ("another claim", a different claim ID)
- their description matches another of their claims *and* doesn't match the current one ("what about my
  car claim?")

A passing mention ("the doctor said...") while discussing the auto claim does not switch: in PROCESS_CASE
the regex only reads a claim type when it is attached to a claim noun.

## Observability

- Every turn returns a trace: signals, phase path, SOP steps, tools, freedom, writer, guard result, model
  calls with token counts, and latency. The test UI's inspector renders it.
- `logs/audit.jsonl` gets one line per turn with the decisions only: no caller text, no PII. Fail-closed
  turns are logged too.
- The transcript itself (which holds whatever the caller typed, including DOB and last-4) never leaves
  the server: `GET /api/session/{id}` returns masked state only.

## Cost and latency controls

- **Cheap path.** A message that is only a yes/no ("Sure, send it.", "no thanks, skip the email") is
  understood by rules alone, so the extraction model call is skipped.
- **Spend cap.** `MAX_SPEND_USD` gives each server process a hard limit. Token usage from every call is
  priced and summed; past the cap, calls stop and the harness falls back to rules/templates.
- **Measured.** Two small-model calls per turn take about 2–4 s. The full 11-scenario suite is about
  $0.03 on gpt-4.1-mini. The Anthropic default (`claude-opus-5-5`, effort low) is roughly 10–40x that per
  turn; `.env.example` says so and names the tested path.

## Failure handling

- A model error, a refusal, malformed model output, or the spend cap: that turn falls back to rules +
  templates and the UI labels it "degraded". Rules-only mode (no key configured) is shown as the mode, not
  as a fallback.
- Choosing a provider without a usable key is a JSON 400 at session start ("no API key: set ... or paste a
  key"), and the UI keeps the current conversation on screen. The Anthropic SDK reports a missing key only
  at request time; the adapter checks at setup instead.
- Anthropic: `claude-opus-5-5` at `effort: low` (chat latency), structured outputs for extraction, and the
  server-side `fallbacks: "default"` beta, so a safety decline is retried on Anthropic's recommended
  fallback model. The adapter steps down one feature at a time if the API rejects `fallbacks`, the output
  format, or `effort`. The request shape was verified against the installed SDK with a captured transport;
  it has not been run against the live API (no key was available).
- OpenAI / compatible endpoints: strict `json_schema` output first. If an endpoint rejects it, the adapter
  steps down to `json_object`, then to a prompt-only JSON request. That step-down is remembered for the
  process, under a lock.

## Hard questions, answered from the code

**Does the caller's SSN leave the server after verification?** Yes, in one place: the last 8–10 transcript
messages, including anything the caller typed, go to the model provider on every later turn (`nlu.py`
`llm_extract`, `responder.py` `render`). The audit log and the API responses are PII-free. Masking the
history window with `identity.mask_value` is the next step; it is not done yet.

**What does the guard not catch?** Prose. A grounded number in an ungrounded sentence, speculation,
answering a technically grounded question the caller didn't ask, tone. Those are handled by what the
writer is given (facts, templates, instructions), not by the guard.

**What happens when the model and the regex disagree?** The model wins on interpretation (intent, emotion,
scope). Rules only fill in literal fields the model left empty, never a name, and two precise signals the
small models kept missing: a bare yes/no answer and a "why do you need this?" clarification. There is no
conflict flag yet; surfacing one in the trace is a small change.

**What breaks with `uvicorn --workers 2`?** Sessions, locks, per-session model clients and the spend
counter live in process memory. A session created on one worker 404s on the other, and the cap is per
worker. The design assumes one worker; sync endpoints run on the threadpool so a slow model call does not
block other sessions. Production: Redis/Postgres for sessions and consent polling, a shared spend counter,
a per-turn deadline.

**What breaks with a weaker model?** Not the gates: in every live run, across four small models, no claim
data appeared before verification, no cross-member data appeared, and consent was enforced. What degraded
was wording: instruction vocabulary leaking, "your account" to a representative, speculation after a
closed appeal window, repeated paragraphs under pressure. Each of those became a fact, a template or an
instruction change, and the transcripts were re-run.

**How is quality evaluated?** Offline: 119 tests, including an exhaustive identity check over every mix of
fields from any two members, a recording model that proves no claim data reaches a prompt before
verification, a deliberately misbehaving fake model pushed through the real guard and rewrite path, 500
random adversarial walks, and a regression test per bug found. Live: the scenario runner checks the
no-leak invariant and the final phase, claim and email choice on 11 conversations; the transcripts were
read by hand. Not done: per-turn expected steps in live mode, an LLM-judged tone rubric.

**What would you do with another day?** Mask caller PII in the prompt history; a session store with one
lock; a conflict flag when rules and model disagree; non-English callers (the templates, the regexes and
the guard's date/number parsing are English-only; a Chinese-script name cannot verify today); a per-turn
deadline; and moving the contract tables and step instructions into YAML so non-engineers can change the
procedure.

## Trade-offs

- **Two model calls per turn** (extract + write). This keeps "understand" separate from "decide", which is
  the point of the harness, at the cost of latency. Next step: run the extractor on a smaller model.
- **No streaming.** The guard has to see the whole reply before the caller does, so replies aren't
  streamed. Streaming plus a guard would need sentence-level checking or a hold-back buffer.
- **Business date** uses `BUSINESS_TZ` (default America/Los_Angeles), not the container's UTC clock,
  because the appeal-deadline logic depends on "today".
- **Rules-only mode is a floor, not a product.** It handles the scripted demo lines and keeps the app
  usable offline, but it misreads loose phrasing (3 of 11 scenarios fail in that mode by design).
- **Email and consent are simulated** with interfaces shaped like the real ones (`Outbox.send`,
  consent status polling).
- **Evaluation history.** Two rounds of review found bugs the unit tests had missed: first a model
  re-stating an earlier identity value counted as progress and speculation after the appeal deadline; then
  a match-count oracle in the mismatch branch, a representative declared after verification keeping
  policyholder access, stuck states for members with no claims, and a provider chosen without a key
  producing 500s. All are fixed with regression tests in `tests/test_regressions.py` and
  `tests/test_review_fixes.py`.
