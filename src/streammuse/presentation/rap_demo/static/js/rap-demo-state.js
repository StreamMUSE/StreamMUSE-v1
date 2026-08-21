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
  let latestRemoteChunk = null;

  function finite(value) {
    return typeof value === "number" && Number.isFinite(value) ? value : null;
  }

  function nonnegativeInteger(value) {
    return Number.isInteger(value) && value >= 0 ? value : null;
  }

  function boundedText(value, limit = MAX_TEXT) {
    return typeof value === "string" ? value.slice(0, limit) : null;
  }

  function boundedStrings(value, limit = MAX_ITEMS) {
    if (!Array.isArray(value)) return [];
    return value.filter((item) => typeof item === "string").slice(0, limit)
      .map((item) => item.slice(0, MAX_TEXT));
  }

  function boundedStringMap(value) {
    if (!value || typeof value !== "object" || Array.isArray(value)) return {};
    const result = {};
    for (const [key, item] of Object.entries(value).slice(0, MAX_ITEMS)) {
      if (typeof key === "string" && typeof item === "string") {
        result[key.slice(0, MAX_TEXT)] = item.slice(0, MAX_REFERENCE);
      }
    }
    return result;
  }

  function boundedNumberMap(value, limit = MAX_ITEMS) {
    if (!value || typeof value !== "object" || Array.isArray(value)) return {};
    const result = {};
    for (const [key, item] of Object.entries(value).slice(0, limit)) {
      const parsed = finite(item);
      if (parsed !== null) result[String(key).slice(0, MAX_TEXT)] = parsed;
    }
    return result;
  }

  function boundedIntegerMap(value) {
    if (!value || typeof value !== "object" || Array.isArray(value)) return {};
    const result = {};
    for (const [key, item] of Object.entries(value).slice(0, MAX_ITEMS)) {
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
        target_stress: finite(slot.target_stress),
        duration_ticks: finite(slot.duration_ticks),
        boundary_strength: finite(slot.boundary_strength),
        rhyme_group: boundedText(slot.rhyme_group),
      }];
    }) : [];
    const derivedSchedule = slots.map((slot) => (
      slot.tick_in_bar === null || slot.target_stress === null
        ? `t${slot.tick_in_bar ?? "-"}@?`
        : `t${slot.tick_in_bar}@${slot.target_stress.toFixed(2)}`
    )).join(", ");
    return {
      template_id: boundedText(source.template_id),
      name: boundedText(source.name),
      ticks_per_beat: nonnegativeInteger(source.ticks_per_beat),
      beats_per_bar: nonnegativeInteger(source.beats_per_bar),
      slots,
      slot_stress_schedule: boundedText(source.slot_stress_schedule, MAX_SUMMARY) ?? derivedSchedule,
    };
  }

  function sanitizeScore(value) {
    const source = value && typeof value === "object" && !Array.isArray(value) ? value : {};
    return {
      bar: nonnegativeInteger(source.bar),
      total: finite(source.total),
      component_scores: boundedNumberMap(source.component_scores, 16),
    };
  }

  function sanitizeChunkPayload(value) {
    const source = value && typeof value === "object" && !Array.isArray(value) ? value : {};
    const counts = source.candidate_counts && typeof source.candidate_counts === "object"
      ? source.candidate_counts : {};
    const timings = {};
    const sourceTimings = source.stage_timings_ms && typeof source.stage_timings_ms === "object"
      ? source.stage_timings_ms : {};
    for (const key of TIMING_KEYS) {
      const parsed = finite(sourceTimings[key]);
      if (parsed !== null) timings[key] = parsed;
    }
    const sourceAlignment = source.alignment && typeof source.alignment === "object"
      ? source.alignment : {};
    return {
      state: boundedText(source.state),
      renderer_decision: boundedText(source.renderer_decision),
      chunk_index: nonnegativeInteger(source.chunk_index),
      bars: Array.isArray(source.bars)
        ? source.bars.map(nonnegativeInteger).filter((item) => item !== null).slice(0, 2)
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
      request_budget_ms: finite(source.request_budget_ms),
      elapsed_ms: finite(source.elapsed_ms),
      deadline_slack_ms: finite(source.deadline_slack_ms),
      alignment: {
        method: boundedText(sourceAlignment.method),
        confidence: finite(sourceAlignment.confidence),
        fallback_counts: boundedIntegerMap(sourceAlignment.fallback_counts),
      },
      stretch_warnings: boundedStrings(source.stretch_warnings),
      warnings: boundedStrings(source.warnings),
      hashes: boundedStringMap(source.hashes),
      artifact_refs: boundedStringMap(source.artifact_refs),
      transfer_bytes: nonnegativeInteger(source.transfer_bytes),
      failure_reason: boundedText(source.failure_reason),
    };
  }

  function projectRemoteChunk(previous, event) {
    if (!event || typeof event !== "object") return previous;
    if (event.event_type === "session_reset") return null;
    if (!CHUNK_EVENTS.has(event.event_type)) return previous;
    const sequence = finite(event.sequence);
    if (previous && finite(previous.sequence) !== null && sequence !== null && previous.sequence > sequence) {
      return previous;
    }
    return {
      sequence,
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
      sequence: finite(event.sequence),
      event_type: event.event_type,
      utc_time: boundedText(event.utc_time),
      monotonic_ns: finite(event.monotonic_ns),
      bar: nonnegativeInteger(event.bar),
      tick: nonnegativeInteger(event.tick),
      request_id: boundedText(event.request_id),
      payload: sanitizeChunkPayload(event.payload),
    };
  }

  function updateRemoteProjection(event) {
    latestRemoteChunk = projectRemoteChunk(latestRemoteChunk, event);
    if (root.document) renderRemoteChunk(latestRemoteChunk, root.document);
  }

  function reduceEventCache(events, event, limit) {
    const maximum = Number.isInteger(limit) && limit > 0 ? limit : 240;
    const prior = Array.isArray(events) ? events : [];
    if (!event || typeof event !== "object") return prior.slice(-maximum);

    const next = event.event_type === "session_reset" ? [] : prior.slice();
    const safeEvent = CHUNK_EVENTS.has(event.event_type) ? sanitizeChunkEvent(event) : event;
    const sequence = safeEvent.sequence;
    const duplicate = sequence !== null && sequence !== undefined
      && next.some((item) => item && item.sequence === sequence);
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
      `${display(flow.template_id)} | ${display(flow.slot_stress_schedule)}`
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

  async function refreshRemoteChunk() {
    try {
      const response = await root.fetch("/api/state", { cache: "no-store" });
      if (!response.ok) return;
      const snapshot = await response.json();
      const projected = snapshot && snapshot.remote_chunk ? snapshot.remote_chunk : null;
      if (
        projected === null || latestRemoteChunk === null
        || finite(projected.sequence) === null || finite(latestRemoteChunk.sequence) === null
        || projected.sequence >= latestRemoteChunk.sequence
      ) {
        latestRemoteChunk = projected;
        renderRemoteChunk(latestRemoteChunk, root.document);
      }
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
