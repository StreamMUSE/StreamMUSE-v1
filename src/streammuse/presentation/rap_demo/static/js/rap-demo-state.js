(function attachRapState(root) {
  "use strict";

  const CHUNK_EVENTS = new Set([
    "chunk_request_submitted",
    "chunk_remote_completed",
    "chunk_remote_rejected",
    "chunk_committed",
    "chunk_fallback_activated",
  ]);
  const TIMING_KEYS = [
    "generation", "evaluation", "moss", "aligner", "r3",
    "package", "transfer", "mac", "total",
  ];
  const MAX_TEXT = 512;
  const MAX_SUMMARY = 4096;
  const MAX_REFERENCE = 1024;
  const MAX_ITEMS = 8;
  const MAX_NUMBER_MAGNITUDE = 1_000_000_000_000;
  const MAX_EVENT_BYTES = 24000;
  let latestRemoteChunk = null;
  let latestRemoteIdentity = null;

  function finite(value) {
    return typeof value === "number" && Number.isFinite(value)
      && Math.abs(value) <= MAX_NUMBER_MAGNITUDE ? value : null;
  }

  function nonnegativeInteger(value) {
    return Number.isSafeInteger(value) && value >= 0 ? value : null;
  }

  function normalizedNumber(value) {
    const parsed = finite(value);
    return parsed !== null && parsed >= 0 && parsed <= 1 ? parsed : null;
  }

  function nonnegativeNumber(value) {
    const parsed = finite(value);
    return parsed !== null && parsed >= 0 ? parsed : null;
  }

  function boundedText(value, limit = MAX_TEXT) {
    return typeof value === "string" ? value.slice(0, limit) : null;
  }

  function boundedStrings(value, limit = MAX_ITEMS) {
    if (!Array.isArray(value)) return [];
    const result = [];
    for (const item of value.slice(0, limit)) {
      if (typeof item === "string") result.push(item.slice(0, MAX_TEXT));
    }
    return result;
  }

  function boundedEntries(value, limit = MAX_ITEMS) {
    if (!value || typeof value !== "object" || Array.isArray(value)) return [];
    const result = [];
    for (const key in value) {
      if (!Object.prototype.hasOwnProperty.call(value, key)) continue;
      result.push([key, value[key]]);
      if (result.length >= limit) break;
    }
    return result;
  }

  function boundedStringMap(value) {
    if (!value || typeof value !== "object" || Array.isArray(value)) return {};
    const result = {};
    for (const [key, item] of boundedEntries(value)) {
      if (typeof key === "string" && typeof item === "string") {
        result[key.slice(0, MAX_TEXT)] = item.slice(0, MAX_REFERENCE);
      }
    }
    return result;
  }

  function boundedNumberMap(value, limit = MAX_ITEMS) {
    if (!value || typeof value !== "object" || Array.isArray(value)) return {};
    const result = {};
    for (const [key, item] of boundedEntries(value, limit)) {
      const parsed = finite(item);
      if (parsed !== null) result[String(key).slice(0, MAX_TEXT)] = parsed;
    }
    return result;
  }

  function boundedIntegerMap(value) {
    if (!value || typeof value !== "object" || Array.isArray(value)) return {};
    const result = {};
    for (const [key, item] of boundedEntries(value)) {
      const parsed = nonnegativeInteger(item);
      if (parsed !== null) result[String(key).slice(0, MAX_TEXT)] = parsed;
    }
    return result;
  }

  function sanitizeFlow(value) {
    const source = value && typeof value === "object" && !Array.isArray(value) ? value : {};
    const slots = Array.isArray(source.slots) ? source.slots.slice(0, 32).flatMap((slot) => {
      if (!slot || typeof slot !== "object" || Array.isArray(slot)) return [];
      return [{
        tick_in_bar: nonnegativeInteger(slot.tick_in_bar),
        target_stress: normalizedNumber(slot.target_stress),
        duration_ticks: nonnegativeNumber(slot.duration_ticks),
        boundary_strength: nonnegativeNumber(slot.boundary_strength),
        rhyme_group: boundedText(slot.rhyme_group),
      }];
    }) : [];
    const derivedSchedule = slots.map((slot) => (
      slot.tick_in_bar === null || slot.target_stress === null
        ? `t${slot.tick_in_bar ?? "-"}@?`
        : `t${slot.tick_in_bar}@${slot.target_stress.toFixed(2)}`
    )).join(", ");
    const result = {
      template_id: boundedText(source.template_id),
      name: boundedText(source.name),
      ticks_per_beat: nonnegativeInteger(source.ticks_per_beat),
      beats_per_bar: nonnegativeInteger(source.beats_per_bar),
      slots,
      slot_stress_schedule: boundedText(source.slot_stress_schedule, MAX_SUMMARY) ?? derivedSchedule,
      selected_syllable_schedule: boundedText(source.selected_syllable_schedule, MAX_SUMMARY),
    };
    return result;
  }

  function sanitizeScore(value) {
    const source = value && typeof value === "object" && !Array.isArray(value) ? value : {};
    const result = {
      bar: nonnegativeInteger(source.bar),
      total: finite(source.total),
      component_scores: boundedNumberMap(source.component_scores, 16),
    };
    return result;
  }

  function sanitizeChunkPayload(value) {
    const source = value && typeof value === "object" && !Array.isArray(value) ? value : {};
    const counts = source.candidate_counts && typeof source.candidate_counts === "object"
      ? source.candidate_counts : {};
    const timings = {};
    const sourceTimings = source.stage_timings_ms && typeof source.stage_timings_ms === "object"
      ? source.stage_timings_ms : {};
    for (const key of TIMING_KEYS) {
      const parsed = nonnegativeNumber(sourceTimings[key]);
      if (parsed !== null) timings[key] = parsed;
    }
    const sourceAlignment = source.alignment && typeof source.alignment === "object"
      ? source.alignment : {};
    const result = {
      state: boundedText(source.state),
      renderer_decision: boundedText(source.renderer_decision),
      coordinator_epoch: nonnegativeInteger(source.coordinator_epoch),
      chunk_index: nonnegativeInteger(source.chunk_index),
      bars: Array.isArray(source.bars)
        ? source.bars.slice(0, 2).map(nonnegativeInteger).filter((item) => item !== null)
        : [],
      selected_lines: boundedStrings(source.selected_lines, 2),
      flows: Array.isArray(source.flows) ? source.flows.slice(0, 2).map(sanitizeFlow) : [],
      candidate_counts: {
        requested: nonnegativeInteger(counts.requested),
        parseable: nonnegativeInteger(counts.parseable),
        valid: nonnegativeInteger(counts.valid),
        selectable: nonnegativeInteger(counts.selectable),
      },
      selected_scores: Array.isArray(source.selected_scores)
        ? source.selected_scores.slice(0, 2).map(sanitizeScore) : [],
      prompt_summary: boundedText(source.prompt_summary, MAX_SUMMARY),
      context_lines: boundedStrings(source.context_lines, 4),
      stage_timings_ms: timings,
      request_budget_ms: nonnegativeNumber(source.request_budget_ms),
      elapsed_ms: nonnegativeNumber(source.elapsed_ms),
      deadline_slack_ms: finite(source.deadline_slack_ms),
      alignment: {
        method: boundedText(sourceAlignment.method),
        confidence: normalizedNumber(sourceAlignment.confidence),
        fallback_counts: boundedIntegerMap(sourceAlignment.fallback_counts),
      },
      stretch_warnings: boundedStrings(source.stretch_warnings),
      warnings: boundedStrings(source.warnings),
      hashes: boundedStringMap(source.hashes),
      artifact_refs: boundedStringMap(source.artifact_refs),
      transfer_bytes: nonnegativeInteger(source.transfer_bytes),
      failure_reason: boundedText(source.failure_reason),
    };
    return enforceChunkBytes(result);
  }

  function encodedBytes(value) {
    try {
      return new TextEncoder().encode(JSON.stringify(value)).length;
    } catch (_) {
      return MAX_EVENT_BYTES + 1;
    }
  }

  function enforceChunkBytes(value) {
    if (encodedBytes(value) <= MAX_EVENT_BYTES) return value;
    const compact = {
      ...value,
      selected_lines: value.selected_lines.map((item) => item.slice(0, 256)),
      flows: value.flows.map((flow) => ({
        ...flow,
        slots: flow.slots.slice(0, 8),
        slot_stress_schedule: boundedText(flow.slot_stress_schedule, 1024),
        selected_syllable_schedule: boundedText(flow.selected_syllable_schedule, 1024),
      })),
      selected_scores: value.selected_scores.map((score) => ({
        ...score,
        component_scores: Object.fromEntries(boundedEntries(score.component_scores, 4)),
      })),
      prompt_summary: boundedText(value.prompt_summary, 1024),
      context_lines: value.context_lines.slice(0, 2).map((item) => item.slice(0, 256)),
      stretch_warnings: value.stretch_warnings.slice(0, 4).map((item) => item.slice(0, 256)),
      warnings: value.warnings.slice(0, 4).map((item) => item.slice(0, 256)),
      hashes: Object.fromEntries(boundedEntries(value.hashes, 4)),
      artifact_refs: Object.fromEntries(boundedEntries(value.artifact_refs, 4)),
    };
    if (encodedBytes(compact) <= MAX_EVENT_BYTES) return compact;
    return {
      ...compact,
      flows: [],
      selected_scores: [],
      context_lines: [],
      stretch_warnings: [],
      warnings: [],
      hashes: {},
      artifact_refs: {},
    };
  }

  function projectRemoteChunk(previous, event) {
    if (!event || typeof event !== "object") return previous;
    const identity = eventIdentity(event);
    if (event.event_type === "session_started") {
      const previousIdentity = projectionIdentity(previous);
      return previousIdentity && identity.session_id !== previousIdentity.session_id ? null : previous;
    }
    if (event.event_type === "session_reset") return null;
    if (!CHUNK_EVENTS.has(event.event_type)) return previous;
    if (previous && compareIdentity(identity, projectionIdentity(previous)) <= 0) {
      return previous;
    }
    return {
      session_id: identity.session_id,
      sequence: identity.sequence,
      coordinator_epoch: identity.coordinator_epoch,
      event_type: event.event_type,
      bar: nonnegativeInteger(event.bar),
      tick: nonnegativeInteger(event.tick),
      request_id: boundedText(event.request_id),
      ...sanitizeChunkPayload(event.payload),
    };
  }

  function sanitizeChunkEvent(event) {
    return {
      session_id: boundedText(event.session_id),
      sequence: nonnegativeInteger(event.sequence),
      event_type: event.event_type,
      utc_time: boundedText(event.utc_time),
      monotonic_ns: nonnegativeInteger(event.monotonic_ns),
      bar: nonnegativeInteger(event.bar),
      tick: nonnegativeInteger(event.tick),
      request_id: boundedText(event.request_id),
      payload: sanitizeChunkPayload(event.payload),
    };
  }

  function eventIdentity(event) {
    const payload = event && event.payload && typeof event.payload === "object"
      ? event.payload : {};
    return {
      session_id: boundedText(event && event.session_id),
      coordinator_epoch: nonnegativeInteger(
        event && event.coordinator_epoch !== undefined
          ? event.coordinator_epoch : payload.coordinator_epoch,
      ),
      sequence: nonnegativeInteger(event && event.sequence),
    };
  }

  function projectionIdentity(value) {
    if (!value || typeof value !== "object") return null;
    return {
      session_id: boundedText(value.session_id),
      coordinator_epoch: nonnegativeInteger(value.coordinator_epoch),
      sequence: nonnegativeInteger(value.sequence),
    };
  }

  function compareIdentity(next, previous) {
    if (!previous) return 1;
    if (next.session_id !== null && previous.session_id !== null
        && next.session_id !== previous.session_id) return 1;
    if (next.session_id === null && previous.session_id !== null) return -1;
    if (next.coordinator_epoch !== null && previous.coordinator_epoch !== null) {
      if (next.coordinator_epoch > previous.coordinator_epoch) return 1;
      if (next.coordinator_epoch < previous.coordinator_epoch) return -1;
    } else if (next.coordinator_epoch === null && previous.coordinator_epoch !== null) {
      return -1;
    }
    if (next.sequence === null) return -1;
    if (previous.sequence === null) return 1;
    return next.sequence > previous.sequence ? 1 : next.sequence === previous.sequence ? 0 : -1;
  }

  function updateRemoteProjection(event) {
    const identity = eventIdentity(event);
    const sessionChanged = latestRemoteIdentity
      && identity.session_id !== null
      && latestRemoteIdentity.session_id !== null
      && identity.session_id !== latestRemoteIdentity.session_id;
    if (event.event_type === "session_started" && sessionChanged) {
      latestRemoteChunk = null;
      latestRemoteIdentity = identity;
    } else if (event.event_type === "session_reset") {
      if (compareIdentity(identity, latestRemoteIdentity) > 0 || sessionChanged) {
        latestRemoteChunk = null;
        latestRemoteIdentity = identity;
      }
    } else if (CHUNK_EVENTS.has(event.event_type)
        && compareIdentity(identity, latestRemoteIdentity) > 0) {
      latestRemoteChunk = projectRemoteChunk(null, event);
      latestRemoteIdentity = identity;
    }
    if (root.document) renderRemoteChunk(latestRemoteChunk, root.document);
  }

  function reduceEventCache(events, event, limit) {
    const maximum = Number.isInteger(limit) && limit > 0 ? limit : 240;
    const prior = Array.isArray(events) ? events : [];
    if (!event || typeof event !== "object") return prior.slice(-maximum);

    const identity = eventIdentity(event);
    const priorSession = prior.length > 0 && prior[prior.length - 1]
      ? boundedText(prior[prior.length - 1].session_id) : null;
    const sessionChanged = priorSession !== null && identity.session_id !== null
      && priorSession !== identity.session_id;
    const next = event.event_type === "session_reset" || sessionChanged ? [] : prior.slice();
    const safeEvent = CHUNK_EVENTS.has(event.event_type) ? sanitizeChunkEvent(event) : event;
    const safeIdentity = eventIdentity(safeEvent);
    const duplicate = safeIdentity.sequence !== null && next.some((item) => {
      if (!item) return false;
      const itemIdentity = eventIdentity(item);
      return itemIdentity.session_id === safeIdentity.session_id
        && itemIdentity.coordinator_epoch === safeIdentity.coordinator_epoch
        && itemIdentity.sequence === safeIdentity.sequence;
    });
    if (!duplicate) next.push(safeEvent);
    updateRemoteProjection(safeEvent);
    return next.slice(-maximum);
  }

  function display(value, fallback = "--") {
    return value === null || value === undefined || value === "" ? fallback : String(value);
  }

  function displayBar(value) {
    return Number.isInteger(value) ? String(value + 1) : "--";
  }

  function fixed(value, digits = 1) {
    return finite(value) === null ? "--" : value.toFixed(digits);
  }

  function pairs(value, digits = null) {
    if (!value || typeof value !== "object" || Array.isArray(value)) return "none";
    const entries = Object.entries(value);
    if (entries.length === 0) return "none";
    return entries.map(([key, item]) => (
      digits === null || finite(item) === null ? `${key}=${display(item)}` : `${key}=${item.toFixed(digits)}`
    )).join(" ");
  }

  function setText(document, id, value, fallback = "--") {
    const node = document.getElementById(id);
    if (node) node.textContent = display(value, fallback);
  }

  function setList(document, id, values) {
    const node = document.getElementById(id);
    if (!node) return;
    const rows = values.length > 0 ? values : ["--"];
    node.replaceChildren(...rows.map((value) => {
      const item = document.createElement("li");
      item.textContent = value;
      return item;
    }));
  }

  function rendererName(value) {
    return ["prevalidated_fallback", "espeak", "local_espeak_fallback"].includes(value)
      ? "local eSpeak fallback" : display(value);
  }

  function renderRemoteChunk(value, document = root.document) {
    if (!document) return;
    const panel = document.getElementById("remote-chunk-panel");
    if (!panel) return;
    if (!value) {
      panel.hidden = true;
      return;
    }
    const chunk = { ...value, ...sanitizeChunkPayload(value) };
    panel.hidden = false;
    setText(document, "remote-request-id", value.request_id, "no request");
    setText(document, "remote-renderer", rendererName(chunk.renderer_decision));
    setText(document, "remote-state", `${display(chunk.state)} | ${display(value.event_type)}`);
    setText(document, "remote-chunk-index", chunk.chunk_index);
    setText(document, "remote-bars", chunk.bars.map(displayBar).join(" + "));
    setList(document, "remote-lines", chunk.selected_lines.map((line, index) => (
      `Bar ${displayBar(chunk.bars[index])}: ${line}`
    )));
    setList(document, "remote-flows", chunk.flows.map((flow) => (
      `${display(flow.template_id)} | targets ${display(flow.slot_stress_schedule)}`
        + (flow.selected_syllable_schedule
          ? ` | selected ${flow.selected_syllable_schedule}` : "")
    )));
    const counts = chunk.candidate_counts;
    setText(document, "remote-candidate-counts", [
      counts.requested, counts.parseable, counts.valid, counts.selectable,
    ].map((count) => display(count)).join(" / "));
    setList(document, "remote-scores", chunk.selected_scores.map((score) => (
      `Bar ${displayBar(score.bar)} total=${fixed(score.total, 3)} ${pairs(score.component_scores, 3)}`
    )));
    setText(document, "remote-prompt-summary", chunk.prompt_summary);
    setList(document, "remote-context-lines", chunk.context_lines);
    setText(document, "remote-stage-timings", Object.entries(chunk.stage_timings_ms)
      .map(([key, item]) => `${key}=${fixed(item, 1)}`).join(" "));
    setText(document, "remote-budget", `${fixed(chunk.request_budget_ms, 1)} ms`);
    setText(document, "remote-elapsed", `${fixed(chunk.elapsed_ms, 1)} ms`);
    setText(document, "remote-deadline-slack", `${fixed(chunk.deadline_slack_ms, 1)} ms`);
    setText(
      document,
      "remote-alignment",
      `${display(chunk.alignment.method)} confidence=${fixed(chunk.alignment.confidence, 3)} `
        + `fallbacks=${pairs(chunk.alignment.fallback_counts)}`,
    );
    setList(document, "remote-stretch-warnings", chunk.stretch_warnings);
    setList(document, "remote-warnings", chunk.warnings);
    setText(document, "remote-hashes", pairs(chunk.hashes));
    setText(document, "remote-artifacts", pairs(chunk.artifact_refs));
    setText(document, "remote-transfer-bytes", `${display(chunk.transfer_bytes)} bytes`);
    setText(document, "remote-failure-reason", chunk.failure_reason, "none");
  }

  function applyRemoteSnapshot(snapshot) {
    if (!snapshot || typeof snapshot !== "object") return latestRemoteChunk;
    const snapshotIdentity = {
      session_id: boundedText(snapshot.session_id),
      coordinator_epoch: nonnegativeInteger(snapshot.coordinator_epoch),
      sequence: nonnegativeInteger(snapshot.last_sequence),
    };
    if (snapshot.remote_chunk === null || snapshot.remote_chunk === undefined) {
      latestRemoteChunk = null;
      latestRemoteIdentity = snapshotIdentity;
      if (root.document) renderRemoteChunk(null, root.document);
      return null;
    }
    const source = snapshot.remote_chunk;
    if (!source || typeof source !== "object" || Array.isArray(source)) {
      return latestRemoteChunk;
    }
    const identity = {
      session_id: boundedText(source.session_id) ?? snapshotIdentity.session_id,
      coordinator_epoch: nonnegativeInteger(source.coordinator_epoch)
        ?? snapshotIdentity.coordinator_epoch,
      sequence: nonnegativeInteger(source.sequence) ?? snapshotIdentity.sequence,
    };
    if (compareIdentity(identity, latestRemoteIdentity) <= 0) return latestRemoteChunk;
    latestRemoteChunk = {
      session_id: identity.session_id,
      coordinator_epoch: identity.coordinator_epoch,
      sequence: identity.sequence,
      event_type: boundedText(source.event_type),
      bar: nonnegativeInteger(source.bar),
      tick: nonnegativeInteger(source.tick),
      request_id: boundedText(source.request_id),
      ...sanitizeChunkPayload(source),
    };
    latestRemoteIdentity = identity;
    if (root.document) renderRemoteChunk(latestRemoteChunk, root.document);
    return latestRemoteChunk;
  }

  async function refreshRemoteChunk() {
    try {
      const response = await root.fetch("/api/state", { cache: "no-store" });
      if (!response.ok) return;
      const snapshot = await response.json();
      applyRemoteSnapshot(snapshot);
    } catch (_) {
      // The primary monitor owns connection status and retry messaging.
    }
  }

  function startRemoteChunkProjection() {
    refreshRemoteChunk();
    root.addEventListener("online", refreshRemoteChunk);
    root.document.addEventListener("visibilitychange", () => {
      if (root.document.visibilityState !== "hidden") refreshRemoteChunk();
    });
  }

  root.StreamMuseRapState = Object.freeze({
    applyRemoteSnapshot,
    currentRemoteChunk: () => latestRemoteChunk,
    projectRemoteChunk,
    reduceEventCache,
    renderRemoteChunk,
    sanitizeChunkPayload,
  });

  if (root.document && typeof root.fetch === "function") {
    if (root.document.readyState === "loading") {
      root.document.addEventListener("DOMContentLoaded", startRemoteChunkProjection, { once: true });
    } else {
      startRemoteChunkProjection();
    }
  }
}(typeof window === "undefined" ? globalThis : window));
