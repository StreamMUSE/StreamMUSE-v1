(function attachRapInteraction(root) {
  "use strict";

  function captureAction(event, focused, phase, enabled) {
    const target = event.target;
    if (!enabled || !["Space", "Enter"].includes(event.code) || event.repeat || event.isComposing || event.defaultPrevented
      || event.ctrlKey || event.altKey || event.metaKey || event.shiftKey || focused !== target
      || !target || /^(INPUT|TEXTAREA|SELECT|BUTTON)$/.test(target.tagName) || target.isContentEditable
      || target.closest?.('[contenteditable="true"], [role="textbox"]')) return null;
    if (phase === "DRUMS_IDLE") return "arm";
    if (phase === "CAPTURING") return "tap";
    return null;
  }

  function calibrateClock(sent, received, serverSeconds) {
    if (![sent, received, serverSeconds].every(Number.isFinite) || received < sent || serverSeconds < 0) return null;
    return { offset_seconds: serverSeconds - (sent + received) / 2000, uncertainty_ms: (received - sent) / 2, sampled_at_ms: received };
  }

  function tapPayload(clientTime, clock) {
    if (!clock || !Number.isFinite(clientTime) || clientTime < 0 || clientTime - clock.sampled_at_ms > 10000 || clock.uncertainty_ms > 250) return null;
    return { client_time_ms: clientTime, server_time_seconds: clientTime / 1000 + clock.offset_seconds, uncertainty_ms: clock.uncertainty_ms };
  }

  function eventTime(event, now = performance.now()) {
    return Number.isFinite(event.timeStamp) && event.timeStamp >= 0 && event.timeStamp <= now ? event.timeStamp : now;
  }

  function phaseAt(info, clock, clientTime) {
    const current = String(info?.phase || "DRUMS_IDLE").toUpperCase();
    if (!["ARMED", "CAPTURING"].includes(current)) return current;
    const tap = tapPayload(clientTime, clock);
    if (!tap || !Number.isFinite(info.capture_start_seconds) || !Number.isFinite(info.capture_end_seconds)) return current;
    if (tap.server_time_seconds >= info.capture_end_seconds) return "WAIT_REPLAY";
    if (tap.server_time_seconds >= info.capture_start_seconds) return "CAPTURING";
    return "ARMED";
  }

  function mount(document) {
    const get = (id) => document.getElementById(id);
    const pad = get("capture-pad");
    if (!pad) return null;
    const state = { info: null, settings: null, connected: false, loading: false, arming: false, pendingTaps: 0, clock: null, timer: null, version: 0, error: null, errorKind: null, epoch: null };
    const phase = (time = performance.now()) => phaseAt(state.info, state.clock, time);
    const enabled = () => state.connected && state.settings?.values?.mode === "alternating" && state.settings.state === "running" && state.info?.supported !== false && state.info !== null;
    const set = (id, value) => { const node = get(id); if (node && node.textContent !== String(value)) node.textContent = value; };
    function render() {
      const info = state.info || {}, current = phase();
      const active = enabled();
      get("interaction-band").hidden = state.settings?.values?.mode !== "alternating";
      pad.setAttribute("aria-disabled", String(!active || state.arming || !["DRUMS_IDLE", "CAPTURING"].includes(current)));
      pad.dataset.phase = current.toLowerCase();
      set("capture-action", state.arming ? "Arming..." : current === "CAPTURING" ? "Capture rhythm" : current === "DRUMS_IDLE" ? "Arm next bar" : current === "ARMED" ? "Capture armed" : "Turn in progress");
      set("interaction-phase", current.replaceAll("_", " "));
      const start = Number.isInteger(info.capture_bar) ? info.capture_bar * 16 : null;
      set("capture-countdown", current === "ARMED" && Number.isFinite(start) && Number.isFinite(info.current_tick) ? `${Math.max(0, start - info.current_tick).toFixed(0)} ticks to capture` : current === "CAPTURING" ? "Recording one full bar" : "");
      set("interaction-turn", info.turn_id ?? "--");
      set("interaction-replays", info.replay_count ?? 0);
      set("interaction-response", `${info.delivered_response_bars ?? 0} / ${info.response_bars ?? state.settings?.values?.response_bars ?? "--"}`);
      set("interaction-wait", Number.isFinite(info.wait_elapsed_seconds) ? `${info.wait_elapsed_seconds.toFixed(1)} s` : "--");
      set("interaction-clock", state.clock ? `+/- ${state.clock.uncertainty_ms.toFixed(1)} ms` : "Uncalibrated");
      set("capture-pending", state.pendingTaps ? `${state.pendingTaps} tap${state.pendingTaps === 1 ? "" : "s"} sending` : "");
      const error = state.error || info.last_error || info.generation_error;
      get("interaction-status").dataset.state = error ? "error" : "idle";
      set("interaction-status", error || (!state.connected ? "Waiting for connection." : state.settings?.state !== "running" ? "Playback stopped." : info.message || ""));
      const template = info.flow;
      const slots = template?.slots || (info.captured_ticks || []).map((tick) => ({ tick_in_bar: tick, target_stress: 0.5 }));
      if (root.StreamMuseRapFlow) root.StreamMuseRapFlow.renderStress(get("capture-grid"), { slots, beats_per_bar: 4, ticks_per_beat: 4 }, document);
      set("capture-template", template?.template_id || `${slots.length} captured onsets`);
    }
    function schedule() {
      root.clearTimeout(state.timer);
      if (state.connected && state.settings?.values?.mode === "alternating") state.timer = root.setTimeout(load, document.hidden ? 2000 : enabled() ? 250 : 1000);
    }
    async function load() {
      if (state.loading || state.arming || !state.connected || state.settings?.values?.mode !== "alternating") return;
      state.loading = true;
      const version = state.version, sent = performance.now();
      try {
        const response = await fetch("/api/control/interaction", { cache: "no-store", signal: AbortSignal.timeout(5000) });
        const body = await response.json(), received = performance.now();
        if (!response.ok) throw new Error(typeof body.detail === "string" ? body.detail : `Interaction unavailable (${response.status}).`);
        if (version !== state.version || !state.connected) return;
        const sample = calibrateClock(sent, received, body.monotonic_seconds);
        if (sample && (!state.clock || received - state.clock.sampled_at_ms > 5000 || sample.uncertainty_ms <= state.clock.uncertainty_ms)) state.clock = sample;
        if (state.errorKind === "discovery") { state.error = null; state.errorKind = null; }
        state.info = body; render();
      } catch (error) { if (version === state.version) { state.error = error.message; state.errorKind = "discovery"; render(); } }
      finally { state.loading = false; schedule(); }
    }
    async function act(action, time) {
      if (!enabled() || state.arming) return;
      if (action === "tap" && state.pendingTaps >= 16) {
        state.error = "Tap not sent: previous taps are still sending."; state.errorKind = "tap"; render(); return;
      }
      const payload = action === "arm" ? {} : tapPayload(time, state.clock);
      if (!payload) { state.error = "Tap not sent: clock calibration unavailable or stale."; state.errorKind = "tap"; render(); load(); return; }
      if (action === "arm") state.arming = true;
      else state.pendingTaps += 1;
      state.error = null;
      const version = action === "arm" ? ++state.version : state.version;
      render();
      try {
        const response = await fetch(`/api/control/interaction/${action}`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload), signal: AbortSignal.timeout(5000) });
        const body = await response.json();
        if (!response.ok) throw new Error(typeof body.detail === "string" ? body.detail : `Capture rejected (${response.status}).`);
        if (version !== state.version) return;
        if (body.accepted === false) throw new Error(`Tap rejected: ${body.reason || "not accepted"}.`);
        // Polling owns the phase after taps, so late tap acknowledgements cannot rewind a turn.
        if (action === "arm" && (body.phase || body.state)) state.info = { ...state.info, ...body };
      } catch (error) { if (version === state.version) { state.error = error.message; state.errorKind = "tap"; } }
      finally { if (action === "arm") state.arming = false; else state.pendingTaps = Math.max(0, state.pendingTaps - 1); render(); schedule(); }
    }
    pad.addEventListener("keydown", (event) => {
      if (event.target !== pad || document.activeElement !== pad) return;
      const time = eventTime(event);
      const action = captureAction(event, document.activeElement, phase(time), enabled());
      if (["Space", "Enter"].includes(event.code) && !event.ctrlKey && !event.altKey && !event.metaKey) event.preventDefault();
      if (action) act(action, time);
    });
    pad.addEventListener("click", (event) => {
      const time = eventTime(event);
      if (phase(time) === "DRUMS_IDLE") act("arm", time);
      else if (phase(time) === "CAPTURING") act("tap", time);
    });
    function reset() { state.version += 1; state.info = null; state.clock = null; state.error = null; load(); render(); }
    render();
    return {
      load,
      settings(info) {
        const previous = state.settings;
        state.settings = { ...info, values: { ...info.values } };
        if (previous?.state === "running" && info.state !== "running") { state.version += 1; state.clock = null; }
        if (previous?.values?.mode !== info.values?.mode) { state.version += 1; state.info = null; state.clock = null; load(); }
        render();
        if (previous?.state !== info.state || previous?.values?.mode !== info.values?.mode) schedule();
      },
      connection(connected) {
        const changed = connected !== state.connected;
        state.connected = connected;
        if (!connected) { state.version += 1; state.clock = null; root.clearTimeout(state.timer); }
        else if (changed) load();
        render();
      },
      update(snapshot) {
        if (!snapshot.session_id) return;
        const epoch = `${snapshot.session_id}:${snapshot.coordinator_epoch ?? 0}`;
        if (state.epoch !== null && state.epoch !== epoch) reset();
        state.epoch = epoch;
      },
      reset,
    };
  }

  root.StreamMuseRapInteraction = { captureAction, calibrateClock, tapPayload, eventTime, phaseAt, mount };
})(typeof window !== "undefined" ? window : globalThis);
