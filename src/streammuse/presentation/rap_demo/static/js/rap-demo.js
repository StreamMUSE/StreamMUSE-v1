"use strict";

const monitor = {
  snapshot: null,
  session: null,
  events: [],
  sortKey: "total_score",
  sortDirection: -1,
  followLive: true,
  refreshTimer: null,
  refreshInFlight: false,
  refreshPending: false,
  reconnectDelay: 1000,
  reconnectTimer: null,
  socket: null,
  closing: false,
  announcedSequence: null,
  renderedEventSequences: [],
};

const element = (id) => document.getElementById(id);

function setText(id, value, fallback = "--") {
  const node = element(id);
  if (node) node.textContent = value === null || value === undefined || value === "" ? fallback : String(value);
}

function createNode(tag, className = "", text = null) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== null && text !== undefined) node.textContent = String(text);
  return node;
}

function number(value) {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function integer(value) {
  return Number.isInteger(value) ? value : null;
}

function fixed(value, digits = 3, fallback = "--") {
  const parsed = number(value);
  return parsed === null ? fallback : parsed.toFixed(digits);
}

function milliseconds(value, digits = 1) {
  const parsed = number(value);
  return parsed === null ? "--" : `${parsed.toFixed(digits)} ms`;
}

function displayBar(value) {
  const parsed = integer(value);
  return parsed === null ? "--" : String(parsed + 1);
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

function barViews(snapshot) {
  const bars = new Map();
  const merge = (bar) => {
    if (!bar || integer(bar.bar) === null) return;
    bars.set(bar.bar, { ...(bars.get(bar.bar) || {}), ...bar });
  };

  for (const bar of objectValues(snapshot.bars)) merge(bar);
  for (const bar of objectValues(snapshot.frozen_bars)) merge({ ...bar, frozen: true });
  for (const event of monitor.events) {
    const type = eventType(event);
    if (!["bar_reserved", "bar_planning_started", "bar_replaced", "bar_frozen", "fallback_activated"].includes(type)) {
      continue;
    }
    const previous = bars.get(event.bar) || {};
    const previousSequence = number(previous.sequence) ?? -Infinity;
    const eventSequence = number(event.sequence) ?? Infinity;
    if (eventSequence < previousSequence) continue;
    const frozen = type === "bar_frozen" ? true : previous.frozen === true;
    merge({ ...previous, sequence: event.sequence, bar: event.bar, request_id: event.request_id || previous.request_id, ...payload(event), frozen });
  }
  return Array.from(bars.values()).sort((left, right) => left.bar - right.bar);
}

function currentView(snapshot) {
  const tickEvent = latestEvent("tick");
  const eventPlayback = tickEvent ? { ...tickEvent, ...payload(tickEvent) } : {};
  const projectedPlayback = snapshot.current_playback || {};
  const playback = (number(eventPlayback.sequence) ?? -Infinity) > (number(projectedPlayback.sequence) ?? -Infinity)
    ? eventPlayback
    : projectedPlayback;
  const tick = number(playback.tick ?? snapshot.current_tick ?? (snapshot.current && snapshot.current.tick));
  const projectedSegment = snapshot.current_segment || snapshot.current || {};
  const bar = integer(playback.bar ?? projectedSegment.bar ?? snapshot.current_bar) ?? (tick === null ? null : Math.floor(tick / 16));
  const liveBar = barViews(snapshot).find((item) => item.bar === bar) || {};
  const segment = { ...projectedSegment, ...liveBar };
  return {
    tick,
    bar,
    segment,
    beat: integer(playback.beat),
    tickInBeat: integer(playback.tick_in_beat),
  };
}

function currentDecision(snapshot, bar) {
  return barViews(snapshot).find((item) => item.bar === bar) || {};
}

function currentFlow(snapshot, bar, decision) {
  if (decision.flow) return decision.flow;
  const request = snapshot.pending_request || snapshot.latest_request || {};
  if (request.bar === bar && request.flow) return request.flow;
  for (const type of ["bar_frozen", "bar_replaced", "bar_planning_started", "bar_reserved"]) {
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
  const event = latestBatchEvent(requestId);
  const eventBatch = event ? { ...event, ...payload(event) } : null;
  const projected = snapshotBatch && (requestId === null || snapshotBatch.request_id === requestId) ? snapshotBatch : null;
  if (!eventBatch) return projected || {};
  if (!projected) return eventBatch;
  return (number(eventBatch.sequence) ?? -Infinity) > (number(projected.sequence) ?? -Infinity) ? eventBatch : projected;
}

function planningRequest(snapshot) {
  const projected = snapshot.pending_request || snapshot.latest_request || null;
  const event = latestEvent("bar_planning_started");
  const live = event ? { ...event, ...payload(event) } : null;
  if (!live) return projected || {};
  if (!projected) return live;
  return (number(live.sequence) ?? -Infinity) > (number(projected.sequence) ?? -Infinity) ? live : projected;
}

function generationFailure(snapshot, requestId) {
  const projectedEvent = snapshot.last_error;
  const projected = projectedEvent && eventType(projectedEvent) === "generation_failed" &&
    (requestId === null || projectedEvent.request_id === requestId)
    ? { ...projectedEvent, ...payload(projectedEvent) }
    : null;
  const event = latestEvent("generation_failed");
  const live = event && (requestId === null || event.request_id === requestId) ? { ...event, ...payload(event) } : null;
  if (!live) return projected;
  if (!projected) return live;
  return (number(live.sequence) ?? -Infinity) > (number(projected.sequence) ?? -Infinity) ? live : projected;
}

function renderHeader(snapshot, current) {
  const metadata = (monitor.session && monitor.session.metadata) || snapshot.session_metadata || {};
  const maxBars = integer(metadata.max_bars);
  setText("session-id", (monitor.session && monitor.session.session_id) || snapshot.session_id);
  setText("generator", metadata.generator);
  setText("model", metadata.model);
  setText("session-state", snapshot.stopped ? "stopped" : snapshot.session_id ? "running" : "waiting");
  setText("tempo", number(metadata.tempo_bpm) === null ? null : `${metadata.tempo_bpm} BPM`);
  setText("tick", current.tick);
  setText("bar", displayBar(current.bar));
  setText("max-bars", maxBars === 0 ? "continuous (0)" : maxBars);
  setText("topic", current.segment.topic);
  setText("template", current.segment.template_id);
}

function renderDecision(snapshot, current, decision) {
  const batch = latestBatch(snapshot, decision.request_id || null);
  const fallback = decision.fallback === true;
  setText("performance-bar", displayBar(current.bar));
  setText("selected-line", decision.text, "Waiting for the first frozen bar.");
  setText("source", decision.source);
  setText("total-score", number(decision.total_score) === null ? null : fixed(decision.total_score));
  setText("generation-latency", milliseconds(batch.latency_ms));
  setText("deadline-slack", milliseconds(batch.decision_deadline_slack_ms ?? batch.deadline_slack_ms));
  setText("fallback-reason", decision.fallback_reason, "none");

  const badge = element("source-badge");
  const sourceState = fallback ? "fallback" : decision.source ? "generated" : "unknown";
  badge.dataset.source = sourceState;
  badge.textContent = fallback ? `FALLBACK · ${decision.fallback_reason || "unspecified"}` : decision.source ? "GENERATED" : "UNRESOLVED";

  const flow = currentFlow(snapshot, current.bar, decision);
  setText("template", flow.template_id || current.segment.template_id);
  const beats = number(flow.beats_per_bar) || 4;
  const ticksPerBeat = number(flow.ticks_per_beat) || 4;
  setText("flow-meter", `${beats} beats / ${beats * ticksPerBeat} ticks · ${objectValues(flow.slots).length} syllable slots`);
  renderCurrentSyllable(snapshot, current);
  renderFlow(snapshot, current, flow, decision);
  renderFlowFacts(flow);
}

function renderCurrentSyllable(snapshot, current) {
  const projected = snapshot.current_syllable;
  const event = latestEvent("syllable_emitted", current.bar);
  const syllable = projected && projected.bar === current.bar ? projected : event ? { ...event, ...payload(event) } : null;
  setText("current-syllable", syllable && (syllable.label || syllable.word));
  if (!syllable) {
    setText("current-syllable-detail", "beat -- · stress -- · observation delay --");
    return;
  }
  const beat = integer(syllable.beat ?? current.beat);
  const subdivision = integer(syllable.subdivision ?? current.tickInBeat);
  const stress = integer(syllable.stress);
  const observationDelay = number(syllable.observation_delay_ms) === null
    ? "--"
    : milliseconds(syllable.observation_delay_ms, 2);
  setText(
    "current-syllable-detail",
    `beat ${beat === null ? "--" : beat + 1}.${subdivision === null ? "--" : subdivision + 1} · stress ${stress ?? "--"} · observation delay ${observationDelay}`,
  );
}

function renderFlow(snapshot, current, flow, decision) {
  const track = element("flow-track");
  track.replaceChildren();
  track.setAttribute("role", "list");
  const ticksPerBeat = integer(flow.ticks_per_beat) || 4;
  const beatsPerBar = integer(flow.beats_per_bar) || 4;
  const totalTicks = ticksPerBeat * beatsPerBar;
  track.style.setProperty("--flow-slot-count", String(totalTicks));
  const slots = objectValues(flow.slots);
  const slotAtTick = new Map(slots.map((slot) => [slot.tick_in_bar, slot]));
  const scheduled = objectValues(decision.scheduled_syllables);
  const scheduledAtTick = new Map(scheduled.map((item) => [item.tick_in_bar ?? (item.tick % totalTicks), item]));
  const emitted = objectValues(snapshot.emitted_syllables).filter((item) => item && item.bar === current.bar);
  const emittedAtTick = new Map(emitted.map((item) => [item.tick % totalTicks, item]));
  const activeTick = current.tick === null ? null : current.tick % totalTicks;
  let activeCell = null;

  for (let tick = 0; tick < totalTicks; tick += 1) {
    const slot = slotAtTick.get(tick);
    const scheduledSyllable = scheduledAtTick.get(tick);
    const emittedSyllable = emittedAtTick.get(tick);
    const syllable = (emittedSyllable && (emittedSyllable.label || emittedSyllable.word)) ||
      (scheduledSyllable && (scheduledSyllable.label || scheduledSyllable.word));
    const stress = slot ? Math.max(0, Math.min(1, number(slot.target_stress) || 0)) : 0;
    const boundary = slot ? number(slot.boundary_strength) || 0 : 0;
    const cell = createNode("div", "flow-cell");
    cell.setAttribute("role", "listitem");
    if (slot) cell.classList.add("has-slot");
    if ((tick + 1) % ticksPerBeat === 0 && tick < totalTicks - 1) cell.classList.add("beat-boundary");
    if (boundary > 0) cell.classList.add("boundary");
    if (tick === activeTick) {
      cell.classList.add("active");
      activeCell = cell;
    }

    const index = createNode("span", "tick-index", `T${String(tick + 1).padStart(2, "0")}`);
    index.append(createNode("span", "beat-index", `B${Math.floor(tick / ticksPerBeat) + 1}`));
    const label = createNode("span", "slot-label", syllable || (slot ? "slot" : "·"));
    if (!slot) label.classList.add("is-rest");
    const meter = createNode("span", "stress-meter");
    const fill = createNode("i", "stress-fill");
    fill.style.height = slot ? `${Math.max(8, stress * 100)}%` : "0";
    meter.append(fill);
    cell.append(index, label, meter);
    cell.setAttribute(
      "aria-label",
      `Tick ${tick + 1}, beat ${Math.floor(tick / ticksPerBeat) + 1}, ${syllable || (slot ? "unfilled syllable slot" : "rest")}, target stress ${slot ? stress.toFixed(2) : "none"}, duration ${slot ? slot.duration_ticks : "none"}, boundary ${boundary}, rhyme ${slot && slot.rhyme_group ? slot.rhyme_group : "none"}${tick === activeTick ? ", currently playing" : ""}`,
    );
    track.append(cell);
  }

  if (monitor.followLive && activeCell) {
    const rail = track.parentElement;
    const target = activeCell.offsetLeft - (rail.clientWidth - activeCell.offsetWidth) / 2;
    rail.scrollLeft = Math.max(0, target);
  }
}

function compactArray(values, digits = null) {
  return `[${values.map((value) => {
    if (value === null || value === undefined || value === "") return "–";
    return digits === null || number(value) === null ? String(value) : number(value).toFixed(digits);
  }).join(", ")}]`;
}

function appendFact(list, label, value, fallback = "--") {
  const row = createNode("div");
  row.append(createNode("dt", "", label), createNode("dd", "", value === null || value === undefined || value === "" ? fallback : value));
  list.append(row);
}

function renderFlowFacts(flow) {
  const slots = objectValues(flow.slots);
  const exact = element("flow-exact");
  exact.replaceChildren();
  appendFact(exact, "Ticks", compactArray(slots.map((slot) => slot.tick_in_bar)));
  appendFact(exact, "Durations", compactArray(slots.map((slot) => slot.duration_ticks)));
  appendFact(exact, "Stress", compactArray(slots.map((slot) => slot.target_stress), 2));
  appendFact(exact, "Boundaries", compactArray(slots.map((slot) => slot.boundary_strength)));
  appendFact(exact, "Rhyme groups", compactArray(slots.map((slot) => slot.rhyme_group)));

  const provenance = flow.provenance && typeof flow.provenance === "object" ? flow.provenance : {};
  const source = element("flow-provenance");
  source.replaceChildren();
  appendFact(source, "ID", flow.template_id);
  appendFact(source, "Name", flow.name);
  appendFact(source, "Meter", `${flow.beats_per_bar || 4} beats × ${flow.ticks_per_beat || 4} ticks`);
  appendFact(source, "Kind", provenance.kind);
  appendFact(source, "Source", provenance.source);
  appendFact(source, "Source hash", provenance.source_hash);
  appendFact(source, "Quantization error", number(provenance.quantization_error_ticks) === null ? null : `${provenance.quantization_error_ticks} ticks`);
}

function promptText(prompt) {
  if (!Array.isArray(prompt) || prompt.length === 0) return "No LLM prompt was used for this generator round.";
  return prompt.map((message) => {
    if (!message || typeof message !== "object") return String(message);
    const role = String(message.role || "message").toUpperCase();
    const content = typeof message.content === "string" ? message.content : JSON.stringify(message.content, null, 2);
    return `[${role}]\n${content}`;
  }).join("\n\n");
}

function renderContext(snapshot) {
  const planning = planningRequest(snapshot);
  const requestId = planning.request_id || null;
  const batchEvent = latestBatchEvent(requestId);
  const batch = latestBatch(snapshot, requestId);
  const failure = generationFailure(snapshot, requestId);
  setText("request-id", requestId || batch.request_id || (batchEvent && batchEvent.request_id), "no request");
  setText("request-bar", displayBar(planning.bar ?? batch.bar ?? (batchEvent && batchEvent.bar)));
  setText("required-syllables", planning.required_syllables);
  setText("requested-candidates", planning.candidate_count ?? batch.candidate_count);
  setText("request-seed", planning.seed);
  setText("prompt", promptText(batch.prompt), "No prompt received.");

  const context_lines = Array.isArray(planning.context_lines) ? planning.context_lines : [];
  const list = element("context-lines");
  list.replaceChildren();
  if (context_lines.length === 0) list.append(createNode("li", "", "No previous frozen lines."));
  else context_lines.forEach((line) => list.append(createNode("li", "", line)));

  renderResponse(batch, batchEvent, failure);
}

function renderResponse(batch, batchEvent, failure) {
  const raw_response = batch.raw_response;
  const source = batch.source || "--";
  const hasBatch = Boolean(batch.request_id || batchEvent || Object.keys(batch).length > 0);
  const errorType = batch.error_type || (failure && failure.error_type);
  const errorMessage = batch.error_message || (failure && failure.error_message);
  const error = errorType || errorMessage;
  const late = batch.decision_late === true || batch.late === true;
  const status = error ? "ERROR" : !hasBatch ? "WAITING" : late ? "LATE" : "ON TIME";
  const rawFallback = source === "phrase_bank"
    ? "(Phrase-bank mode generated candidates directly; no raw LLM response.)"
    : error ? `No raw response received: ${[errorType, errorMessage].filter(Boolean).join(": ")}` : "No raw response received.";
  setText("raw-response", raw_response, rawFallback);
  setText("prompt-tokens", batch.prompt_tokens);
  setText("completion-tokens", batch.completion_tokens);
  setText("batch-warning", batch.warning, "none");

  const facts = element("response-status");
  facts.replaceChildren();
  appendFact(facts, "Source", source);
  appendFact(facts, "Status", status);
  appendFact(facts, "Returned", batch.candidate_count);
  appendFact(facts, "Latency", milliseconds(batch.latency_ms));
  appendFact(facts, "Response slack", milliseconds(batch.deadline_slack_ms));
  appendFact(facts, "Decision slack", milliseconds(batch.decision_deadline_slack_ms));
  appendFact(facts, "Tokens", `${batch.prompt_tokens ?? "--"} prompt / ${batch.completion_tokens ?? "--"} completion`);
  appendFact(facts, "Warning", batch.warning, "none");
  appendFact(facts, "Error", [errorType, errorMessage].filter(Boolean).join(": "), "none");
}

function candidates(snapshot) {
  const requestId = planningRequest(snapshot).request_id || null;
  const values = new Map();
  for (const candidate of objectValues(snapshot.candidates)) {
    if (!candidate || (requestId !== null && candidate.request_id !== requestId)) continue;
    const key = candidate.candidate_id || `snapshot-${candidate.sequence}`;
    values.set(key, candidate);
  }
  for (const event of monitor.events) {
    if (eventType(event) !== "candidate_evaluated" || (requestId !== null && event.request_id !== requestId)) continue;
    const candidate = { ...event, ...payload(event) };
    const key = candidate.candidate_id || `event-${candidate.sequence}`;
    const previous = values.get(key);
    if (!previous || (number(candidate.sequence) ?? Infinity) >= (number(previous.sequence) ?? -Infinity)) values.set(key, candidate);
  }
  return Array.from(values.values());
}

function componentText(candidate) {
  return objectValues(candidate.components).map((component) => {
    const name = component.name || component.component || "score";
    const value = number(component.value ?? component.score);
    const weight = number(component.weight);
    const contribution = number(component.contribution);
    return `${name}: ${fixed(value)} × ${fixed(component.weight)} = ${fixed(component.contribution)}`;
  }).join("\n");
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
  if (key === "total_score" || key === "syllable_count") {
    const observed = key === "syllable_count" && Array.isArray(candidate.syllables) ? candidate.syllables.length : candidate[key];
    return number(observed) ?? -Infinity;
  }
  return String(candidate[key] || "").toLowerCase();
}

function appendCell(row, content, label, className = "") {
  const cell = createNode("td", className);
  cell.dataset.label = label;
  if (content instanceof Node) cell.append(content);
  else cell.textContent = String(content);
  row.append(cell);
}

function renderCandidates(snapshot) {
  const values = candidates(snapshot);
  const planning = planningRequest(snapshot);
  const required = integer(planning.required_syllables);
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

  if (values.length === 0) {
    const row = createNode("tr");
    const cell = createNode("td", "empty-row", "Waiting for candidate evaluations.");
    cell.colSpan = 9;
    row.append(cell);
    body.append(row);
  }

  for (const candidate of values) {
    const row = createNode("tr", "candidate-record");
    row.dataset.selected = candidate.selected === true ? "true" : "false";
    row.dataset.valid = candidate.valid === false ? "false" : "true";
    const gate = candidate.selected === true ? "SELECTED" : candidate.valid === true ? "VALID" : candidate.valid === false ? "REJECT" : "--";
    const observed = integer(candidate.syllable_count) ?? (Array.isArray(candidate.syllables) ? candidate.syllables.length : null);
    appendCell(row, candidate.selected === true ? "●" : "○", "Pick");
    appendCell(row, candidate.candidate_id || "--", "Candidate ID");
    appendCell(row, candidate.text || "--", "Raw text");
    appendCell(row, `${observed ?? "--"} / ${required ?? "--"}`, "Syllables");
    appendCell(row, gate, "Gate", candidate.valid === false ? "candidate-invalid" : "candidate-valid");
    appendCell(row, fixed(candidate.total_score), "Total");
    appendCell(row, componentText(candidate) || "--", "Components");
    appendCell(row, objectValues(candidate.rejection_reasons).join(", ") || "none", "Rejection reasons", "candidate-reason");
    appendCell(row, fallbackWords(candidate) || "none", "OOV / fallback");
    body.append(row);
  }
  updateSortHeaders();
}

function updateSortHeaders() {
  for (const button of document.querySelectorAll("[data-sort]")) {
    const active = button.dataset.sort === monitor.sortKey;
    const direction = monitor.sortDirection === 1 ? "ascending" : "descending";
    if (active) {
      button.dataset.direction = direction;
      button.setAttribute("aria-label", `${button.textContent}, sorted ${direction}`);
      button.parentElement.setAttribute("aria-sort", direction);
    } else {
      delete button.dataset.direction;
      button.setAttribute("aria-label", `${button.textContent}, sort column`);
      button.parentElement.removeAttribute("aria-sort");
    }
  }
  const menu = element("candidate-sort");
  const menuValue = `${monitor.sortKey}:${monitor.sortDirection}`;
  if (menu && Array.from(menu.options).some((option) => option.value === menuValue)) menu.value = menuValue;
}

function renderSelectedScore(snapshot, decision) {
  const values = candidates(snapshot);
  const selected = values.find((candidate) => candidate.selected === true) ||
    values.find((candidate) => candidate.candidate_id && candidate.candidate_id === decision.candidate_id) ||
    values.find((candidate) => candidate.text && candidate.text === decision.text);
  const list = element("selected-score");
  list.replaceChildren();
  if (!selected) {
    appendFact(list, "Selection", decision.fallback === true ? "fallback path" : "waiting");
    return;
  }
  appendFact(list, "Total", fixed(selected.total_score));
  for (const component of objectValues(selected.components)) {
    appendFact(
      list,
      component.name || "score",
      `${fixed(component.value)} × ${fixed(component.weight)} = ${fixed(component.contribution)}`,
    );
  }
}

function metricPercentage(metric) {
  return metric && number(metric.rate) !== null ? `${(metric.rate * 100).toFixed(1)}%` : "--";
}

function renderMetrics(snapshot) {
  const research = snapshot.research_metrics || {};
  const metrics = research.metrics || {};
  const candidateValidity = metrics.candidate_validity || {};
  const latencies = research.latencies || {};
  const generation = latencies.generation_latency_ms || {};
  const observationDelay = latencies.syllable_observation_delay_ms || {};
  setText("candidate-yield", `${candidateValidity.numerator ?? 0} / ${candidateValidity.denominator ?? 0}`);
  setText("fallback-rate", metricPercentage(metrics.fallback));
  setText("deadline-rate", metricPercentage(metrics.deadline_miss));
  setText("generator-error-rate", metricPercentage(metrics.generator_error));
  setText("pronunciation-rate", metricPercentage(metrics.pronunciation_fallback));
  setText("generation-p50", number(generation.p50) === null ? null : milliseconds(generation.p50));
  setText("generation-p95", number(generation.p95) === null ? null : milliseconds(generation.p95));
  setText("observation-p95", number(observationDelay.p95) === null ? null : milliseconds(observationDelay.p95, 2));
}

function renderAudio(snapshot) {
  const audio = snapshot.audio && typeof snapshot.audio === "object" ? snapshot.audio : {};
  const latencies = snapshot.latencies || {};
  const renderLatency = latencies.bar_render_latency_ms || {};
  const commitSlack = latencies.audio_commit_slack_ms || {};
  const state = audio.state || "disabled";
  setText("audio-state", state);
  setText("audio-queue-depth", audio.queue_depth ?? 0);
  setText("audio-buffered-seconds", number(audio.buffered_seconds) === null ? null : `${number(audio.buffered_seconds).toFixed(1)} s`, "0.0 s");
  setText("audio-device", audio.device);
  setText("audio-recording", audio.recording_path);
  setText("audio-render-latency", number(renderLatency.p95) === null ? null : milliseconds(renderLatency.p95));
  setText("audio-commit-slack", number(commitSlack.p50) === null ? null : milliseconds(commitSlack.p50));
  setText("audio-underruns", audio.underruns ?? 0);

  const body = element("audio-warning-rows");
  body.replaceChildren();
  const warnings = objectValues(snapshot.audio_warnings).slice(-128).reverse();
  if (warnings.length === 0) {
    const row = createNode("tr");
    const cell = createNode("td", "empty-row", "No audio warnings.");
    cell.colSpan = 6;
    row.append(cell);
    body.append(row);
  }
  for (const warning of warnings) {
    const row = createNode("tr");
    const pressure = [warning.available_ms, warning.rendered_ms, warning.compression_ratio]
      .some((value) => number(value) !== null)
      ? `${milliseconds(warning.available_ms)} / ${milliseconds(warning.rendered_ms)} @ ${fixed(warning.compression_ratio, 2)}`
      : "--";
    appendCell(row, `B${displayBar(warning.bar)} / ${warning.slot_index ?? "--"}`, "Bar / slot");
    appendCell(row, warning.type || warning.code || "warning", "Type");
    appendCell(row, warning.word || "--", "Word");
    appendCell(row, warning.source || "--", "Source");
    appendCell(row, pressure, "Duration pressure");
    appendCell(row, warning.action || warning.message || "--", "Action");
    body.append(row);
  }

  const start = element("start-runtime");
  const stop = element("stop-runtime");
  const reset = element("reset-runtime");
  start.disabled = state !== "stopped";
  stop.disabled = !["priming", "running"].includes(state);
  reset.disabled = state !== "stopped";
}

function renderQueueAndHealth(snapshot, current, decision) {
  const bars = barViews(snapshot);
  const next = bars.find((bar) => integer(bar.bar) !== null && (current.bar === null || bar.bar > current.bar) && bar.frozen !== true);
  if (next) {
    const safety = next.fallback === true ? "fallback armed" : next.source ? "generated" : "reserved";
    const score = number(next.total_score) === null ? "" : ` · score ${fixed(next.total_score)}`;
    setText("next-bar", `Bar ${displayBar(next.bar)} · ${safety}${score} · ${next.text || "awaiting text"}`);
  } else {
    setText("next-bar", snapshot.stopped ? "session complete" : "awaiting reservation");
  }

  const lastError = snapshot.last_error || latestEvent("presentation_error") || latestEvent("generation_failed");
  const errorData = lastError ? payload(lastError).error_type ? payload(lastError) : lastError.payload || lastError : {};
  const batch = latestBatch(snapshot);
  const errorSequence = lastError && number(lastError.sequence);
  const batchSequence = number(batch.sequence);
  const activeError = errorSequence !== null && (batchSequence === null || errorSequence > batchSequence);
  const activeRequest = planningRequest(snapshot);
  const activeBatch = latestBatch(snapshot, activeRequest.request_id || null);
  const modelPending = Boolean(activeRequest.request_id && !activeBatch.request_id);
  let health = "READY";
  if (snapshot.stopped) health = "STOPPED";
  else if (activeError) health = "ERROR";
  else if (modelPending) health = "MODEL PENDING";
  setText("model-health", health);
  setText("generator-state", health.toLowerCase());

  const message = [errorData.error_type, errorData.error_message].filter(Boolean).join(": ");
  setText("last-error", message, "none");
  element("last-error").classList.toggle("has-error", Boolean(message));
}

function renderHistory(snapshot) {
  const body = element("history-rows");
  body.replaceChildren();
  const bars = objectValues(snapshot.frozen_bars).sort((left, right) => (number(right.bar) || 0) - (number(left.bar) || 0));
  if (bars.length === 0) {
    const row = createNode("tr");
    const cell = createNode("td", "empty-row", "No bars have frozen yet.");
    cell.colSpan = 5;
    row.append(cell);
    body.append(row);
  }
  for (const bar of bars) {
    const row = createNode("tr");
    appendCell(row, displayBar(bar.bar), "Bar");
    appendCell(row, bar.text || "--", "Line");
    appendCell(row, bar.source || "--", "Source");
    appendCell(row, fixed(bar.total_score), "Score");
    appendCell(row, bar.fallback === true ? `FALLBACK · ${bar.fallback_reason || "unspecified"}` : "generated", "Decision");
    body.append(row);
  }
}

function quoted(value) {
  if (value === null || value === undefined) return "--";
  const text = String(value).replace(/[\r\n\t]+/g, " ");
  return text.length > 150 ? `${text.slice(0, 147)}...` : text;
}

function eventDetail(event) {
  const data = payload(event);
  const type = eventType(event);
  const fields = [];
  const add = (label, value) => {
    if (value !== null && value !== undefined && value !== "") fields.push(`${label}=${quoted(value)}`);
  };
  if (event.request_id) add("request", event.request_id);
  if (type === "session_started") {
    add("tempo", data.tempo_bpm); add("max_bars", data.max_bars); add("generator", data.generator);
  } else if (type === "bar_reserved") {
    add("source", data.source); add("topic", data.topic); add("fallback", data.fallback); add("reason", data.fallback_reason); add("text", data.text);
  } else if (type === "bar_planning_started") {
    add("topic", data.topic); add("syllables", data.required_syllables); add("candidates", data.candidate_count); add("seed", data.seed);
  } else if (type === "candidate_batch_received") {
    add("source", data.source); add("count", data.candidate_count); add("latency_ms", data.latency_ms); add("late", data.late); add("tokens", `${data.prompt_tokens ?? "--"}/${data.completion_tokens ?? "--"}`); add("warning", data.warning); add("error", data.error_type);
  } else if (type === "candidate_evaluated") {
    add("gate", data.selected === true ? "SELECTED" : data.valid === true ? "PASS" : "REJECT"); add("id", data.candidate_id); add("syllables", Array.isArray(data.syllables) ? data.syllables.length : data.syllable_count); add("score", fixed(data.total_score)); add("text", data.text); add("reasons", objectValues(data.rejection_reasons).join(",")); add("oov", objectValues(data.oov_words).join(","));
  } else if (type === "bar_replaced") {
    add("source", data.source); add("candidate", data.candidate_id); add("score", fixed(data.total_score)); add("text", data.text);
  } else if (type === "bar_frozen") {
    add("source", data.source); add("fallback", data.fallback); add("reason", data.fallback_reason); add("text", data.text);
  } else if (type === "fallback_activated") {
    add("reason", data.fallback_reason);
  } else if (type === "generation_failed" || type === "presentation_error") {
    add("type", data.error_type); add("message", data.error_message); add("late", data.late); add("sink", data.sink);
  } else if (type === "tick") {
    add("beat", integer(data.beat) === null ? null : data.beat + 1); add("subdivision", integer(data.tick_in_beat) === null ? null : data.tick_in_beat + 1); add("absolute", event.tick);
  } else if (type === "syllable_emitted") {
    add("label", data.label); add("stress", data.stress); add("observation_delay_ms", number(data.observation_delay_ms) === null ? null : fixed(data.observation_delay_ms));
  } else {
    for (const key of Object.keys(data).slice(0, 5)) add(key, data[key]);
  }
  return fields.join(" · ");
}

function createEventRow(event) {
  const row = createNode("li");
  row.dataset.sequence = String(event.sequence ?? "--");
  row.append(
    createNode("span", "event-sequence", `#${event.sequence ?? "--"}`),
    createNode("span", "event-type", eventType(event)),
    createNode("span", "event-position", `B${displayBar(event.bar)} T${event.tick ?? "-"}`),
    createNode("span", "event-detail", eventDetail(event)),
  );
  return row;
}

function renderEvents() {
  setText("event-count", `${monitor.events.length} events`);
  const consoleNode = element("event-console");
  const previousScrollTop = consoleNode.scrollTop;
  const events = monitor.events.slice(-120);
  const targetSequences = events.map((event) => String(event.sequence ?? "--"));

  if (events.length === 0) {
    if (monitor.renderedEventSequences.length !== 0) consoleNode.replaceChildren();
    if (consoleNode.children.length === 0) consoleNode.append(createNode("li", "empty-console", "Waiting for canonical events."));
    monitor.renderedEventSequences = [];
  } else {
    const targetSet = new Set(targetSequences);
    for (const child of Array.from(consoleNode.children)) {
      if (!targetSet.has(child.dataset.sequence)) child.remove();
    }
    const rendered = Array.from(consoleNode.children).map((child) => child.dataset.sequence);
    const prefixMatches = rendered.every((sequence, index) => sequence === targetSequences[index]);
    if (!prefixMatches) {
      consoleNode.replaceChildren(...events.map(createEventRow));
    } else {
      for (const event of events.slice(rendered.length)) consoleNode.append(createEventRow(event));
    }
    monitor.renderedEventSequences = targetSequences;
  }

  const announcedTypes = new Set([
    "session_started", "session_stopped", "bar_reserved", "bar_planning_started",
    "candidate_batch_received", "bar_replaced", "bar_frozen", "fallback_activated",
    "generation_failed", "presentation_error",
  ]);
  const latest = events.slice().reverse().find((event) => announcedTypes.has(eventType(event)));
  if (
    monitor.followLive && latest && latest.sequence !== monitor.announcedSequence &&
    announcedTypes.has(eventType(latest))
  ) {
    setText("event-announcer", `${eventType(latest).replaceAll("_", " ")}, bar ${displayBar(latest.bar)}`);
    monitor.announcedSequence = latest.sequence;
  }
  if (monitor.followLive) consoleNode.scrollTop = consoleNode.scrollHeight;
  else consoleNode.scrollTop = previousScrollTop;
}

function render() {
  const snapshot = monitor.snapshot || {};
  const current = currentView(snapshot);
  const decision = currentDecision(snapshot, current.bar);
  renderHeader(snapshot, current);
  renderDecision(snapshot, current, decision);
  renderQueueAndHealth(snapshot, current, decision);
  renderContext(snapshot);
  renderCandidates(snapshot);
  renderSelectedScore(snapshot, decision);
  renderMetrics(snapshot);
  renderAudio(snapshot);
  renderHistory(snapshot);
  renderEvents();
}

async function refreshSnapshot() {
  monitor.refreshTimer = null;
  if (monitor.refreshInFlight) {
    monitor.refreshPending = true;
    return;
  }
  monitor.refreshInFlight = true;
  try {
    const response = await fetch("/api/state", { cache: "no-store" });
    if (response.ok) {
      monitor.snapshot = await response.json();
      window.StreamMuseRapState.applyRemoteSnapshot(monitor.snapshot);
    }
  } catch (_) {
    connectionState("retrying", "Retrying");
  } finally {
    monitor.refreshInFlight = false;
    render();
    if (monitor.refreshPending) {
      monitor.refreshPending = false;
      scheduleSnapshotRefresh();
    }
  }
}

function scheduleSnapshotRefresh() {
  if (monitor.refreshTimer === null) monitor.refreshTimer = window.setTimeout(refreshSnapshot, 25);
}

function connectionState(state, label) {
  element("connection").dataset.state = state;
  setText("connection-label", label);
}

function connect() {
  if (monitor.closing) return;
  connectionState("connecting", "Connecting");
  const protocol = location.protocol === "https:" ? "wss" : "ws";
  const socket = new WebSocket(`${protocol}://${location.host}/ws`);
  monitor.socket = socket;
  socket.onopen = () => {
    monitor.reconnectDelay = 1000;
    connectionState("live", "Live");
  };
  socket.onmessage = ({ data }) => {
    let message;
    try {
      message = JSON.parse(data);
    } catch (_) {
      connectionState("retrying", "Bad data");
      return;
    }
    if (message.type === "snapshot" && message.payload) {
      monitor.snapshot = message.payload;
      window.StreamMuseRapState.applyRemoteSnapshot(message.payload);
      monitor.events = Array.isArray(message.payload.recent_events) ? message.payload.recent_events.slice(-240) : [];
    } else if (message.type === "event" && message.payload) {
      monitor.events = window.StreamMuseRapState.reduceEventCache(monitor.events, message.payload, 240);
      scheduleSnapshotRefresh();
    }
    render();
  };
  socket.onclose = () => {
    monitor.socket = null;
    if (monitor.closing) {
      connectionState("stopped", "Stopped");
      return;
    }
    connectionState("offline", "Reconnecting");
    window.clearTimeout(monitor.reconnectTimer);
    monitor.reconnectTimer = window.setTimeout(connect, monitor.reconnectDelay);
    monitor.reconnectDelay = Math.min(10000, monitor.reconnectDelay * 1.6);
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

async function sendControl(action) {
  try {
    const response = await fetch(`/api/control/${action}`, { method: "POST" });
    const body = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(body.detail || `Control request failed (${response.status})`);
    if (monitor.snapshot && monitor.snapshot.audio && body.state) monitor.snapshot.audio.state = body.state;
    setText("last-error", "none");
    element("last-error").classList.remove("has-error");
    render();
    scheduleSnapshotRefresh();
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    setText("last-error", message);
    element("last-error").classList.add("has-error");
  }
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

element("candidate-sort").addEventListener("change", (event) => {
  const [key, direction] = event.currentTarget.value.split(":");
  monitor.sortKey = key;
  monitor.sortDirection = Number(direction);
  render();
});

element("follow-live").addEventListener("change", (event) => {
  monitor.followLive = event.currentTarget.checked;
  if (monitor.followLive) renderEvents();
});

element("start-runtime").addEventListener("click", () => sendControl("start"));
element("stop-runtime").addEventListener("click", () => sendControl("stop"));
element("reset-runtime").addEventListener("click", () => sendControl("reset"));

window.addEventListener("beforeunload", () => {
  monitor.closing = true;
  window.clearTimeout(monitor.reconnectTimer);
  if (monitor.socket) monitor.socket.close();
});

updateSortHeaders();
refreshSnapshot();
loadSession();
connect();
