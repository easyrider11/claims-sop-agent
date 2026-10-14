const $ = (id) => document.getElementById(id);
const PHASES = ["VERIFY_ID", "RESOLVE_INTENT", "PROCESS_CASE", "POST_PROCESS", "CLOSED"];
const FIELD_LABEL = { full_name: "Full name", dob: "Date of birth", phone: "Phone", email: "Email",
  ssn_last4: "SSN / ID last 4", policy_number: "Policy # (locator only)" };
const MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
const CHIPS = [
  ["Demo case", "I'm the policyholder. My name is Margaret Chen, policy POL-9921. I'm calling about my denied healthcare claim from January. DOB is 1985-03-15, SSN last four is 4472."],
  ["Frustrated caller", "I already told you who I am. This is ridiculous. Just tell me why my claim was denied."],
  ["Partial ID", "My name is Margaret Chen and I'm worried about my claim from January."],
  ["Off-topic", "What is RL?"],
  ["Representative", "Hi, this is David Chen calling on behalf of my mother Margaret Chen. Her date of birth is March 15, 1985 and her phone is 650-521-2836."],
  ["Docs question", "What exactly do I need to send, and what if I can't get the office note?"],
  ["Wrong claim", "Actually, what about my car claim?"],
  ["Done", "No, that's everything, thanks."],
  ["Ask for human", "Can I just talk to a real person?"],
];

let sessionId = null;
let SOP = null;   // the contract from /api/sop
let busy = false;

function headers() {
  const h = { "Content-Type": "application/json" };
  const p = $("provider").value, k = $("apikey").value.trim();
  if (p) h["X-LLM-Provider"] = p;
  if (k) h["X-LLM-Key"] = k;
  return h;
}

function addMsg(role, text, meta) {
  const div = document.createElement("div");
  div.className = `msg ${role}`;
  div.textContent = text;
  if (meta) { const m = document.createElement("span"); m.className = "meta"; m.textContent = meta; div.appendChild(m); }
  $("log").appendChild(div);
  $("log").scrollTop = $("log").scrollHeight;
  return div;
}

function kv(el, rows) {
  el.innerHTML = "";
  for (const [k, v, cls] of rows) {
    const li = document.createElement("li");
    const a = document.createElement("span"); a.className = "k"; a.textContent = k;
    const b = document.createElement("span"); b.className = "v " + (cls || ""); b.textContent = v ?? "—";
    li.append(a, b); el.appendChild(li);
  }
}

function render(state, trace) {
  // phases
  const ph = $("phases"); ph.innerHTML = "";
  const cur = state.phase;
  const curIdx = PHASES.indexOf(cur);
  for (const p of [...PHASES, "HANDOFF"]) {
    const li = document.createElement("li");
    li.textContent = p;
    if (p === "HANDOFF") li.classList.add("handoff");
    if (p === cur) li.classList.add("now");
    else if (curIdx > PHASES.indexOf(p) && PHASES.indexOf(p) >= 0) li.classList.add("done");
    ph.appendChild(li);
  }
  if (SOP && SOP.tools[cur]) {
    const tools = SOP.tools[cur];
    $("phaseTools").textContent = `Allowed tools now: ${tools.length ? tools.join(", ") : "none"} · freedom: ${SOP.freedom[cur] || "—"}`;
  }
  $("phaseNote").textContent = state.claims_unlocked
    ? `Claim data unlocked for ${state.verified_as}${state.caller_role === "representative" ? ` (via representative ${state.rep_name})` : ""}.`
    : "Claim data locked: nothing about any claim is sent to the model yet.";

  // identity
  const idRows = Object.keys(FIELD_LABEL).map((f) => [FIELD_LABEL[f], state.identity[f] || "—", state.identity[f] ? "ok" : ""]);
  idRows.push(["Failed attempts", `${state.failed_verifications} / 3`, state.failed_verifications ? "warn" : ""]);
  idRows.push(["Caller role", state.caller_role + (state.rep_name ? ` (${state.rep_name})` : "")]);
  if (state.caller_role === "representative") {
    idRows.push(["Rep authorized", state.rep_authorized === null ? "—" : String(state.rep_authorized), state.rep_authorized ? "ok" : state.rep_authorized === false ? "bad" : ""]);
    idRows.push(["Policyholder consent", state.consent_status || "—", state.consent_status === "approved" ? "ok" : state.consent_status ? "warn" : ""]);
  }
  kv($("identity"), idRows);
  const n = ["full_name", "dob", "phone", "email", "ssn_last4"].filter((f) => state.identity[f]).length;
  $("idCount").textContent = state.claims_unlocked ? "✓ verified"
    : `${n} given · 3 matching needed`;
  $("idNote").textContent = "Values are masked here. Mismatches are never revealed to the caller.";

  // memory
  const m = state.memory;
  kv($("memory"), [
    ["Intent", m.intent],
    ["Case type", m.case_type], ["Case status", m.case_status],
    ["Case month", m.case_month ? MONTHS[m.case_month - 1] + (m.case_year ? " " + m.case_year : "") : null],
    ["Case id", m.case_id],
    ["Deferred question", m.pending_question],
    ["Notes", (m.notes || []).join("; ") || null],
  ]);

  kv($("case"), [
    ["Selected", state.case_id],
    ["Candidates", (state.candidate_case_ids || []).join(", ") || null],
    ["Topics", (state.topics_discussed || []).join(", ") || null],
    ["Off-topic strikes", String(state.off_topic_strikes)],
    ["Persuasion attempts", String(state.persuasion_attempts)],
    ["Waiting for answer to", state.pending_offer],
  ]);

  if (trace) {
    const s = trace.signals, p = trace.plan;
    kv($("turn"), [
      ["Understood by", s.source],
      ["Scope", s.scope, s.scope === "out_of_scope" ? "warn" : ""],
      ["Emotion", `${s.emotion} (${s.emotion_intensity})`, s.emotion !== "neutral" ? "warn" : ""],
      ["Intent / path", `${s.intent} / ${s.workflow_path}`],
      ["Phase", `${p.phase_before} → ${p.phase_after}`],
      ["SOP steps", p.steps.join(", ")],
      ["Freedom", p.freedom],
      ["Tools", p.actions.map((a) => a.tool).join(", ") || null],
      ["Reply by", trace.render.writer, trace.render.guard?.length ? "bad" : ""],
      ["Guard", trace.render.guard?.length ? trace.render.guard.join("; ") : "pass", trace.render.guard?.length ? "bad" : "ok"],
      ["Phase path", (p.phase_path || []).join(" → ")],
      ["Latency", `${trace.total_ms} ms` + (trace.usage?.length ? ` · ${trace.usage.length} model calls · ${trace.usage.reduce((n, u) => n + (u.input_tokens || 0) + (u.output_tokens || 0), 0)} tokens` : "")],
      ["Model error", trace.nlu_error || trace.render.llm_error || trace.error || null, "bad"],
    ]);
    $("trace").textContent = JSON.stringify(trace, null, 2);
  }

  $("emailCard").classList.toggle("hidden", !state.email_sent);
  if (state.email_sent) $("email").textContent = `To: ${state.email_sent.to}\nSubject: ${state.email_sent.subject}\n\n${state.email_sent.body}`;
  $("handoffCard").classList.toggle("hidden", !state.handoff);
  if (state.handoff) $("handoff").textContent = JSON.stringify(state.handoff, null, 2);
  $("events").textContent = state.events.map((e) => JSON.stringify(e)).join("\n");
}

function setMode(mode, degraded) {
  const el = $("mode");
  const rules = mode.startsWith("rules");
  el.textContent = rules ? "rules-only (no model configured)" : mode + (degraded ? " · degraded turn (model error, fallback used)" : "");
  el.classList.toggle("degraded", !!degraded);
}

// Read a JSON body safely: a 4xx/5xx may be JSON {detail}, a list of validation errors, or plain text.
async function readError(r) {
  let d = null;
  try { d = await r.json(); } catch (_) { try { d = { detail: await r.text() }; } catch (__) { d = {}; } }
  const det = d && d.detail;
  if (Array.isArray(det)) return det.map((x) => x.msg || JSON.stringify(x)).join("; ");
  return det || `HTTP ${r.status}`;
}

function setBusy(on) {
  busy = on;
  $("send").disabled = on;
  for (const el of $("settings").querySelectorAll("button, select, input")) el.disabled = on;
}

async function newSession() {
  if (busy) return;
  setBusy(true);
  try {
    const r = await fetch("/api/session", { method: "POST", headers: headers(),
      body: JSON.stringify({ consent_scenario: $("scenario").value }) });
    if (!r.ok) {
      addMsg("agent", "Could not start a conversation: " + await readError(r));
      return;
    }
    const d = await r.json();
    $("log").innerHTML = "";          // only now: a failed attempt keeps the old conversation on screen
    sessionId = d.session_id;
    setMode(d.mode);
    addMsg("agent", d.reply);
    render(d.state, null);
  } catch (e) {
    addMsg("agent", "Could not reach the server: " + e);
  } finally {
    setBusy(false); $("msg").focus();
  }
}

// Returns true when the message was accepted for sending.
async function send(text) {
  if (busy || !text.trim() || !sessionId) return false;
  const sid = sessionId;              // a reply for an old session must not land in a new one
  setBusy(true);
  addMsg("user", text);
  const typing = addMsg("agent typing", "…");
  try {
    const r = await fetch("/api/chat", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sid, message: text }) });
    typing.remove();
    if (sid !== sessionId) return true;
    if (!r.ok) { addMsg("agent", "Error: " + await readError(r)); return true; }
    const d = await r.json();
    const err = d.trace.nlu_error || d.trace.render.llm_error || d.trace.error;
    const note = err ? ` · ⚠ model error, fallback used: ${String(err).slice(0, 120)}`
      : d.trace.render.guard?.length ? " · guard replaced the model reply"
      : d.trace.render.first_draft_rejected ? " · first draft rejected by guard, rewritten" : "";
    addMsg("agent", d.reply, `${d.trace.plan.phase_after} · ${d.trace.plan.steps.join(" + ")}${note}`);
    setMode(d.trace.mode, d.trace.degraded);
    render(d.state, d.trace);
  } catch (e) {
    typing.remove(); addMsg("agent", "Could not reach the server: " + e);
  } finally {
    setBusy(false); $("msg").focus();
  }
  return true;
}

$("composer").addEventListener("submit", async (e) => {
  e.preventDefault();
  const t = $("msg").value;
  if (busy) return;                   // keep the text; it is sent when the current turn is done
  $("msg").value = "";
  if (!(await send(t))) $("msg").value = t;
});
$("msg").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("composer").requestSubmit(); }
});
$("settings").addEventListener("submit", (e) => { e.preventDefault(); newSession(); });

for (const [label, text] of CHIPS) {
  const b = document.createElement("button");
  b.type = "button"; b.className = "chip"; b.textContent = label; b.title = text;
  b.addEventListener("click", () => { $("msg").value = text; $("msg").focus(); });
  $("chips").appendChild(b);
}

(async () => {
  try {
    const c = await (await fetch("/api/config")).json();
    SOP = await (await fetch("/api/sop")).json();
    for (const s of c.consent_scenarios) {
      const o = document.createElement("option"); o.value = s; o.textContent = s; $("scenario").appendChild(o);
    }
    if (c.env_error) addMsg("agent", "Server-side model setup failed (" + c.env_error + "). Pick a model and paste a key above, or choose rules-only.");
  } catch (e) {
    addMsg("agent", "Could not load server configuration: " + e);
  }
  newSession();
})();
