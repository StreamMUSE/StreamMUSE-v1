(function attachRapFlow(root) {
  "use strict";

  function timingEvidence(flow, decision, playback) {
    const diagnostic = playback.timing_evidence || decision.timing_evidence || {};
    const ticks = (flow.beats_per_bar || 4) * (flow.ticks_per_beat || 4);
    const valid = (value) => Number.isFinite(value) && value >= 0 && value < ticks;
    const offsets = diagnostic.timing?.tick_offsets;
    function sampleTick(sample) {
      if (!Number.isFinite(sample) || sample < 0) return null;
      if (Array.isArray(offsets) && offsets.length === ticks + 1 && offsets.every(Number.isFinite)) {
        for (let tick = 0; tick < ticks; tick += 1) {
          if (sample >= offsets[tick] && sample < offsets[tick + 1]) return tick + (sample - offsets[tick]) / (offsets[tick + 1] - offsets[tick]);
        }
        return null;
      }
      return Number.isFinite(diagnostic.frame_count) && diagnostic.frame_count > 0 ? sample / diagnostic.frame_count * ticks : null;
    }
    const estimates = (diagnostic.rendered_onset_estimates || []).slice(0, 64);
    const estimated = estimates.map((item) => sampleTick(item.estimated_sample)).filter(valid);
    const requested = (diagnostic.requested_onsets || []).slice(0, 64).map((item) => sampleTick(item.sample)).filter(valid);
    const confidence = estimates.map((item) => item.confidence).filter((value) => Number.isFinite(value) && value >= 0 && value <= 1);
    const cursor = sampleTick(diagnostic.playback_cursor?.sample);
    return {
      requested: requested.length ? requested : (flow.slots || []).map((slot) => slot.tick_in_bar).filter(valid),
      estimated: estimated.length ? estimated : null,
      estimate_status: estimated.length ? diagnostic.rendered_estimate_status || "available" : "unavailable",
      cursor: Number.isFinite(playback.current_tick) && playback.current_tick >= 0 ? playback.current_tick % ticks : valid(cursor) ? cursor : null,
      acoustic: null,
      confidence: confidence.length ? Math.min(...confidence) : null,
      method: estimates.find((item) => Number.isFinite(item.estimated_sample))?.method || null,
      drift_ms: estimates.map((item) => item.drift_ms).filter(Number.isFinite),
      unavailable_reason: estimates.find((item) => item.unavailable_reason)?.unavailable_reason || null,
      template_id: decision.template_id || flow.template_id || null,
      timeline_revision: diagnostic.timing?.revision ?? null,
      bpm: diagnostic.timing?.bpm ?? null,
      mix: diagnostic.mix || null,
    };
  }

  function renderStress(track, flow, document = root.document) {
    if (!track) return;
    const slots = flow.slots || [], ticks = (flow.beats_per_bar || 4) * (flow.ticks_per_beat || 4);
    const key = JSON.stringify([flow.template_id, slots]);
    if (track.dataset.flowKey === key) return;
    track.dataset.flowKey = key;
    track.replaceChildren();
    track.style.setProperty("--stress-ticks", ticks);
    track.setAttribute("aria-label", `${flow.name || "Captured rhythm"}: requested stress`);
    for (let tick = 0; tick < ticks; tick += 1) {
      const slot = slots.find((item) => item.tick_in_bar === tick);
      const stress = slot ? Math.max(0, Math.min(1, Number(slot.target_stress) || 0)) : null;
      const level = stress === null ? "rest" : stress < 0.34 ? "weak" : stress < 0.67 ? "medium" : "strong";
      const cell = document.createElement("span");
      cell.className = "stress-pulse";
      cell.dataset.stress = level;
      cell.setAttribute("aria-label", `Tick ${tick + 1}: ${level}${stress === null ? "" : ` stress ${stress.toFixed(2)}`}`);
      cell.title = cell["aria-label"] || `Tick ${tick + 1}: ${level} ${stress ?? ""}`;
      const fill = document.createElement("i");
      const label = document.createElement("b");
      label.textContent = level === "rest" ? "-" : level[0].toUpperCase();
      const index = document.createElement("small");
      index.textContent = String(tick + 1);
      cell.append(fill, label, index);
      track.append(cell);
    }
  }

  function renderTiming(document, flow, decision, playback) {
    const evidence = timingEvidence(flow, decision, playback);
    const ticks = (flow.beats_per_bar || 4) * (flow.ticks_per_beat || 4);
    for (const [id, points] of [["timing-requested", evidence.requested], ["timing-estimated", evidence.estimated], ["timing-cursor", evidence.cursor === null ? null : [evidence.cursor]]]) {
      const track = document.getElementById(id);
      if (!track) continue;
      track.replaceChildren();
      if (!points?.length) { track.textContent = "Unavailable"; continue; }
      for (const value of points) {
        const marker = document.createElement("i");
        marker.style.left = `${value / ticks * 100}%`;
        marker.title = `Tick ${value.toFixed(2)} (zero-based)`;
        track.append(marker);
      }
      track.setAttribute("aria-label", `${id.replace("timing-", "")}: ticks ${points.map((value) => value.toFixed(2)).join(", ")}`);
    }
    const detail = document.getElementById("timing-estimate-detail");
    if (detail) detail.textContent = evidence.estimated ? [evidence.estimate_status === "partial" ? "Partial estimate" : "Rendered estimate", evidence.method, evidence.confidence === null ? "confidence unavailable" : `min confidence ${evidence.confidence.toFixed(2)}`, evidence.drift_ms.length ? `drift ${Math.min(...evidence.drift_ms).toFixed(1)} to ${Math.max(...evidence.drift_ms).toFixed(1)} ms` : "drift unavailable", evidence.unavailable_reason].filter(Boolean).join(" / ") : `Rendered alignment estimate unavailable${evidence.unavailable_reason ? `: ${evidence.unavailable_reason}` : ""}`;
    const identity = document.getElementById("played-identity");
    if (identity) identity.textContent = decision.frozen ? `Played: ${evidence.template_id || "--"} / topic ${decision.topic || "--"} / ${evidence.bpm ?? "--"} BPM / timeline r${evidence.timeline_revision ?? "--"}${evidence.mix ? ` / mix r${evidence.mix.revision}: drums ${evidence.mix.drum_gain}, vocals ${evidence.mix.vocal_gain}` : " / mix unavailable"}` : "Played revision unavailable: no committed bar";
  }

  root.StreamMuseRapFlow = { timingEvidence, renderStress, renderTiming };
})(typeof window !== "undefined" ? window : globalThis);
