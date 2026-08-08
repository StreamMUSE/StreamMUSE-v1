"use strict";

const monitor = {
  snapshot: null,
  session: null,
  events: [],
  sortKey: "total_score",
  sortDirection: -1,
  refreshTimer: null,
};

const element = (id) => document.getElementById(id);

function setText(id, value, fallback = "--") {
  const node = element(id);
  if (node) node.textContent = value === null || value === undefined || value === "" ? fallback : String(value);
}

function number(value) {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function percentage(numerator, denominator) {
  return denominator > 0 ? `${((numerator / denominator) * 100).toFixed(1)}%` : "0.0%";
}

function objectValues(value) {
  if (Array.isArray(value)) return value;
  if (value && typeof value === "object") return Object.values(value);
  return [];
}

function payload(event) {
  return event && event.payload && typeof event.payload === "object" ? event.payload : {};
}

function eventType(event) {
  return event ? event.event_type || event.type || "unknown" : "unknown";
}

function latestEvent(type, bar = null) {
  for (let index = monitor.events.length - 1; index >= 0; index -= 1) {
    const event = monitor.events[index];
    if (eventType(event) === type && (bar === null || event.bar === bar)) return event;
  }
  return null;
}

function currentView(snapshot) {
  const tick = number(snapshot.current_tick ?? (snapshot.current && snapshot.current.tick));
  const segment = snapshot.current_segment || snapshot.current || {};
  const bar = number(segment.bar ?? snapshot.current_bar) ?? (tick === null ? null : Math.floor(tick / 16));
  return { tick, bar, segment };
}

function currentDecision(snapshot, bar) {
  const frozen = objectValues(snapshot.frozen_bars).find((item) => item && item.bar === bar);
  const frozenEvent = latestEvent("bar_frozen", bar);
  const replacedEvent = latestEvent("bar_replaced", bar);
  const reservedEvent = latestEvent("bar_reserved", bar);
  return frozen ||
    (frozenEvent && payload(frozenEvent)) ||
    (replacedEvent && payload(replacedEvent)) ||
    (reservedEvent && payload(reservedEvent)) ||
    {};
}

function currentFlow(snapshot, bar, decision) {
  if (decision.flow) return decision.flow;
  const request = snapshot.pending_request || {};
  if (request.flow) return request.flow;
  for (const type of ["bar_planning_started", "bar_reserved", "bar_frozen"]) {
    const event = latestEvent(type, bar);
    if (payload(event).flow) return payload(event).flow;
  }
  return {};
}

function latestBatchEvent(requestId = null) {
  for (let index = monitor.events.length - 1; index >= 0; index -= 1) {
    const event = monitor.events[index];
    const eventRequestId = event && (event.request_id || payload(event).request_id);
    if (eventType(event) === "candidate_batch_received" && (requestId === null || eventRequestId === requestId)) {
      return event;
    }
  }
  return null;
}

function latestBatch(snapshot = monitor.snapshot || {}, requestId = null) {
  const snapshotBatch = snapshot.latest_batch || null;
  if (snapshotBatch && (requestId === null || snapshotBatch.request_id === requestId)) return snapshotBatch;
  return payload(latestBatchEvent(requestId));
}

function renderHeader(snapshot, current) {
  const metadata = (monitor.session && monitor.session.metadata) || snapshot.session_metadata || {};
  setText("session-id", (monitor.session && monitor.session.session_id) || snapshot.session_id);
  setText("generator", metadata.generator);
  setText("model", metadata.model);
  setText("session-state", snapshot.stopped ? "stopped" : snapshot.session_id ? "running" : "waiting");
  setText("tempo", number(metadata.tempo_bpm) === null ? null : `${metadata.tempo_bpm} BPM`);
  setText("tick", current.tick);
  setText("bar", current.bar);
  setText("topic", current.segment.topic);
  setText("template", current.segment.template_id);
  setText("generator-state", snapshot.pending_request ? "planning" : "ready");
}

function renderDecision(snapshot, current, decision) {
  const batch = latestBatch(snapshot, decision.request_id || null);
  const fallback = decision.fallback === true;
  setText("selected-line", decision.text, "Waiting for the first frozen bar.");
  setText("source", decision.source);
  setText("total-score", number(decision.total_score) === null ? null : number(decision.total_score).toFixed(3));
  setText("generation-latency", number(batch.latency_ms) === null ? null : `${number(batch.latency_ms).toFixed(1)} ms`);
  setText("deadline-slack", number(batch.deadline_slack_ms) === null ? null : `${number(batch.deadline_slack_ms).toFixed(1)} ms`);
  setText("fallback-reason", decision.fallback_reason, "none");

  const badge = element("source-badge");
  badge.dataset.fallback = decision.fallback === true ? "true" : decision.fallback === false ? "false" : "unknown";
  badge.textContent = fallback ? `FALLBACK · ${decision.fallback_reason || "unspecified"}` : decision.source ? "GENERATED" : "UNRESOLVED";

  const flow = currentFlow(snapshot, current.bar, decision);
  setText("template", flow.template_id || current.segment.template_id);
  const beats = number(flow.beats_per_bar) || 4;
  const ticksPerBeat = number(flow.ticks_per_beat) || 4;
  setText("flow-meter", `${beats} beats / ${beats * ticksPerBeat} ticks`);
  renderFlow(snapshot, current, flow, decision);
}

function renderFlow(snapshot, current, flow, decision) {
  const track = element("flow-track");
  track.replaceChildren();
  const totalTicks = (number(flow.ticks_per_beat) || 4) * (number(flow.beats_per_bar) || 4);
  const slots = Array.isArray(flow.slots) ? flow.slots : [];
  const scheduled = Array.isArray(decision.scheduled_syllables) ? decision.scheduled_syllables : [];
  const emitted = objectValues(snapshot.emitted_syllables).filter((item) => item && item.bar === current.bar);
  const activeTick = current.tick === null ? null : current.tick % totalTicks;

  for (let tick = 0; tick < totalTicks; tick += 1) {
    const slot = slots.find((item) => item && item.tick_in_bar === tick);
    const scheduledSyllable = scheduled.find((item) => item && (item.tick_in_bar === tick || item.tick === (current.bar * totalTicks) + tick));
    const emittedSyllable = emitted.find((item) => item && item.tick % totalTicks === tick);
    const cell = document.createElement("div");
    cell.className = "flow-cell";
    if (slot) cell.classList.add("has-slot");
    if (tick === activeTick) cell.classList.add("active");

    const index = document.createElement("span");
    index.className = "tick-index";
    index.textContent = String(tick + 1);
    cell.append(index);

    const label = document.createElement("span");
    label.className = "flow-label";
    const syllable = (emittedSyllable && (emittedSyllable.label || emittedSyllable.word)) ||
      (scheduledSyllable && (scheduledSyllable.label || scheduledSyllable.word));
    label.textContent = syllable || (slot ? "slot" : "·");
    if (!slot) label.classList.add("empty");
    cell.append(label);

    if (slot) {
      const meter = document.createElement("span");
      meter.className = "stress-meter";
      const fill = document.createElement("i");
      const stress = Math.max(0, Math.min(1, number(slot.target_stress) || 0));
      fill.style.height = `${Math.max(8, stress * 100)}%`;
      meter.append(fill);
      cell.append(meter);
      cell.title = `Tick ${tick + 1}; stress ${stress.toFixed(2)}${slot.rhyme_group ? `; rhyme ${slot.rhyme_group}` : ""}`;
    }
    track.append(cell);
  }
}

function promptText(prompt) {
  if (!Array.isArray(prompt) || prompt.length === 0) return "No prompt received.";
  return prompt.map((message) => {
    if (!message || typeof message !== "object") return String(message);
    const role = String(message.role || "message").toUpperCase();
    const content = typeof message.content === "string" ? message.content : JSON.stringify(message.content, null, 2);
    return `[${role}]\n${content}`;
  }).join("\n\n");
}

function renderContext(snapshot) {
  const planning = snapshot.latest_request || snapshot.pending_request || payload(latestEvent("bar_planning_started"));
  const requestId = planning.request_id || null;
  const batchEvent = latestBatchEvent(requestId);
  const batch = latestBatch(snapshot, requestId);
  setText("request-id", requestId || batch.request_id || (batchEvent && batchEvent.request_id), "no request");
  setText("request-bar", planning.bar ?? batch.bar ?? (batchEvent && batchEvent.bar));
  setText("required-syllables", planning.required_syllables);
  setText("requested-candidates", planning.candidate_count ?? batch.candidate_count);
  setText("request-seed", planning.seed);
  setText("prompt-tokens", batch.prompt_tokens);
  setText("completion-tokens", batch.completion_tokens);
  setText("batch-warning", batch.warning, "none");
  setText("prompt", promptText(batch.prompt), "No prompt received.");
}

function candidates(snapshot) {
  return objectValues(snapshot.candidates).map((candidate) => candidate || {});
}

function componentText(candidate) {
  return objectValues(candidate.components).map((component) => {
    const name = component.name || component.component || "score";
    const value = number(component.value ?? component.score);
    return `${name}=${value === null ? "--" : value.toFixed(3)}`;
  }).join(" · ");
}

function fallbackWords(candidate) {
  if (Array.isArray(candidate.oov_words)) return candidate.oov_words.join(", ");
  return objectValues(candidate.word_analysis_sources)
    .filter((item) => item && !String(item.source || "").startsWith("cmudict"))
    .map((item) => item.word)
    .filter(Boolean)
    .join(", ");
}

function candidateSortValue(candidate, key) {
  if (key === "selected" || key === "valid") return candidate[key] === true ? 1 : 0;
  if (key === "total_score" || key === "syllable_count") return number(candidate[key]) ?? -Infinity;
  return String(candidate[key] || "").toLowerCase();
}

function renderCandidates(snapshot) {
  const values = candidates(snapshot);
  values.sort((left, right) => {
    const a = candidateSortValue(left, monitor.sortKey);
    const b = candidateSortValue(right, monitor.sortKey);
    if (a < b) return -1 * monitor.sortDirection;
    if (a > b) return 1 * monitor.sortDirection;
    return 0;
  });
  setText("candidate-count", `${values.length} candidate${values.length === 1 ? "" : "s"}`);
  const body = element("candidate-rows");
  body.replaceChildren();

  for (const candidate of values) {
    const row = document.createElement("tr");
    if (candidate.selected === true) row.classList.add("selected-row");
    if (candidate.valid === false) row.classList.add("invalid-row");
    const pick = document.createElement("span");
    pick.className = `tag ${candidate.selected === true ? "selected" : candidate.valid === false ? "rejected" : ""}`;
    pick.textContent = candidate.selected === true ? "SELECTED" : candidate.valid === false ? "REJECTED" : "--";
    appendCell(row, pick);
    appendCell(row, candidate.text || "--");
    appendCell(row, candidate.syllable_count ?? (Array.isArray(candidate.syllables) ? candidate.syllables.length : null) ?? (candidate.analysis && candidate.analysis.syllables && candidate.analysis.syllables.length) ?? "--");
    appendCell(row, candidate.valid === true ? "yes" : candidate.valid === false ? "no" : "--");
    const total = number(candidate.total_score);
    appendCell(row, total === null ? "--" : total.toFixed(3));
    appendCell(row, componentText(candidate) || "--");
    appendCell(row, objectValues(candidate.rejection_reasons).join(", ") || "none");
    appendCell(row, fallbackWords(candidate) || "none");
    body.append(row);
  }
}

function appendCell(row, content) {
  const cell = document.createElement("td");
  if (content instanceof Node) cell.append(content);
  else cell.textContent = String(content);
  row.append(cell);
}

function renderMetrics(snapshot) {
  const research = snapshot.research_metrics || {};
  const metrics = research.metrics || {};
  const candidateValidity = metrics.candidate_validity || {};
  setText("candidate-yield", `${candidateValidity.numerator ?? 0} / ${candidateValidity.denominator ?? 0}`);
  setText("fallback-rate", metricPercentage(metrics.fallback));
  setText("deadline-rate", metricPercentage(metrics.deadline_miss));
  setText("pronunciation-rate", metricPercentage(metrics.pronunciation_fallback));
  const p95 = number(research.latencies && research.latencies.emission_jitter_ms && research.latencies.emission_jitter_ms.p95);
  setText("jitter-p95", p95 === null ? null : `${p95.toFixed(2)} ms`);
}

function metricPercentage(metric) {
  return metric && number(metric.rate) !== null ? `${(metric.rate * 100).toFixed(1)}%` : "--";
}

function renderHistory(snapshot) {
  const body = element("history-rows");
  body.replaceChildren();
  const bars = objectValues(snapshot.frozen_bars).sort((a, b) => (number(b.bar) || 0) - (number(a.bar) || 0));
  for (const bar of bars) {
    const row = document.createElement("tr");
    appendCell(row, bar.bar ?? "--");
    appendCell(row, bar.text || "--");
    appendCell(row, bar.source || "--");
    appendCell(row, bar.fallback === true ? `FALLBACK · ${bar.fallback_reason || "unspecified"}` : "generated");
    body.append(row);
  }
}

function eventDetail(event) {
  const data = payload(event);
  for (const key of ["label", "text", "fallback_reason", "error_message", "candidate_id", "topic"]) {
    if (data[key] !== null && data[key] !== undefined) return `${key}=${String(data[key])}`;
  }
  return event.request_id ? `request=${event.request_id}` : "";
}

function renderEvents() {
  setText("event-count", `${monitor.events.length} events`);
  const consoleNode = element("event-console");
  consoleNode.replaceChildren();
  for (const event of monitor.events.slice(-120).reverse()) {
    const row = document.createElement("li");
    const sequence = document.createElement("span");
    sequence.className = "sequence";
    sequence.textContent = `#${event.sequence ?? "--"}`;
    const type = document.createElement("span");
    type.className = "event-type";
    type.textContent = eventType(event);
    const position = document.createElement("span");
    position.className = "event-position";
    position.textContent = `B${event.bar ?? "-"} T${event.tick ?? "-"}`;
    const detail = document.createElement("span");
    detail.className = "event-detail";
    detail.textContent = eventDetail(event);
    row.append(sequence, type, position, detail);
    consoleNode.append(row);
  }
}

function render() {
  const snapshot = monitor.snapshot || {};
  const current = currentView(snapshot);
  const decision = currentDecision(snapshot, current.bar);
  renderHeader(snapshot, current);
  renderDecision(snapshot, current, decision);
  renderContext(snapshot);
  renderCandidates(snapshot);
  renderMetrics(snapshot);
  renderHistory(snapshot);
  renderEvents();
}

async function refreshSnapshot() {
  monitor.refreshTimer = null;
  try {
    const response = await fetch("/api/state", { cache: "no-store" });
    if (response.ok) monitor.snapshot = await response.json();
  } catch (_) {
    return;
  }
  render();
}

function scheduleSnapshotRefresh() {
  if (monitor.refreshTimer === null) monitor.refreshTimer = window.setTimeout(refreshSnapshot, 25);
}

function connectionState(state, label) {
  element("connection").dataset.state = state;
  setText("connection-label", label);
}

function connect() {
  connectionState("connecting", "Connecting");
  const protocol = location.protocol === "https:" ? "wss" : "ws";
  const socket = new WebSocket(`${protocol}://${location.host}/ws`);
  socket.onopen = () => connectionState("live", "Live");
  socket.onmessage = ({ data }) => {
    const message = JSON.parse(data);
    if (message.type === "snapshot") {
      monitor.snapshot = message.payload;
      monitor.events = Array.isArray(message.payload.recent_events) ? message.payload.recent_events.slice(-240) : [];
    }
    else if (message.type === "event") {
      const sequence = message.payload && message.payload.sequence;
      const duplicate = sequence !== null && sequence !== undefined &&
        monitor.events.some((event) => event && event.sequence === sequence);
      if (!duplicate) monitor.events.push(message.payload);
      if (monitor.events.length > 240) monitor.events.splice(0, monitor.events.length - 240);
      scheduleSnapshotRefresh();
    }
    render();
  };
  socket.onclose = () => {
    connectionState("offline", "Reconnecting");
    window.setTimeout(connect, 1000);
  };
  socket.onerror = () => socket.close();
}

async function loadSession() {
  try {
    const response = await fetch("/api/session", { cache: "no-store" });
    if (response.ok) monitor.session = await response.json();
  } catch (_) {
    monitor.session = null;
  }
  render();
}

for (const button of document.querySelectorAll("[data-sort]")) {
  button.addEventListener("click", () => {
    const nextKey = button.dataset.sort;
    if (monitor.sortKey === nextKey) monitor.sortDirection *= -1;
    else {
      monitor.sortKey = nextKey;
      monitor.sortDirection = nextKey === "text" ? 1 : -1;
    }
    render();
  });
}

loadSession();
connect();
