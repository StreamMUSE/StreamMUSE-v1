(function attachRapSettings(root) {
  "use strict";

  const groupFields = {
    local: ["drum_gain", "vocal_gain", "chord_gain", "chord_loop", "alignment_policy"],
    generation: ["tempo_bpm"],
    session: ["mode", "response_bars", "output_device"],
  };
  const fields = Object.values(groupFields).flat();
  const stoppedOnly = new Set(groupFields.session);
  const gains = new Set(["drum_gain", "vocal_gain", "chord_gain"]);
  const applicationFor = (field) => stoppedOnly.has(field) ? "stopped-only"
    : gains.has(field) ? "immediate" : field === "alignment_policy" ? "next-uncommitted-bar" : "bar-boundary";
  const applicationLabels = { immediate: "Immediate", "bar-boundary": "Next eligible bar", "next-uncommitted-bar": "Next eligible bar" };
  const stringFields = new Set(["mode", "output_device", "chord_loop", "alignment_policy"]);
  // Two capability probes can take 10 s each around the 10 s tempo ACK wait.
  const submitTimeoutMs = 35000;

  function requestFailure(error, submitting, group) {
    const timedOut = ["TimeoutError", "AbortError"].includes(error.name);
    if (submitting) return `Settings response ${timedOut ? `timed out after ${submitTimeoutMs / 1000} s` : "was not received"}; outcome is unknown. Check Requested values and Played evidence before retrying.${group === "generation" ? " If tempo requires reset: Stop, wait for stopped, Reset, then Start." : ""}`;
    return timedOut ? "Settings refresh timed out; checking again shortly." : error.message;
  }

  function rejectionMessage(error) {
    const message = error.message || "Settings rejected.";
    return /tempo/i.test(message) && /reset|uncertain|synchronized/i.test(message)
      ? `${message} Stop, wait for stopped, Reset, then Start before changing tempo again.` : message;
  }

  function makePayload(values, previous = {}) {
    const next = {};
    for (const field of fields) {
      if (!(field in values)) continue;
      let value = values[field];
      if (field === "mode") {
        if (!["continuous", "alternating"].includes(value)) throw new Error("Choose continuous or alternating mode.");
      } else if (field === "output_device") {
        if (value !== null && typeof value !== "string") throw new Error("Choose an available output device.");
      } else if (field === "alignment_policy") {
        if (!["gentle_sparse_r3", "all_onsets_r3", "all_onsets_stress_r3"].includes(value)) throw new Error("Choose an alignment policy.");
      } else if (field === "chord_loop") {
        if (!["off", "minor_cycle", "major_cycle"].includes(value)) throw new Error("Choose a chord loop.");
      } else {
        if (value === "" || value === null || typeof value === "boolean") throw new Error(`${field} needs a numeric value.`);
        value = Number(value);
        const [min, max] = field === "tempo_bpm" ? [30, 180] : field === "response_bars" ? [1, 8] : [0, 2];
        if (!Number.isFinite(value) || value < min || value > max || (field === "response_bars" && !Number.isInteger(value))) {
          throw new Error(`${field.replaceAll("_", " ")} must be ${field === "response_bars" ? "an integer " : ""}between ${min} and ${max}.`);
        }
      }
      if (value !== previous[field]) next[field] = value;
    }
    return next;
  }

  function fieldEnabled(field, info, connected) {
    if (!connected || info?.supported === false || info?.capabilities?.[field] !== true) return false;
    const state = info.state;
    if (!["stopped", "running"].includes(state)) return false;
    const application = info.application?.[field] || applicationFor(field);
    return application !== "stopped-only" || state !== "running";
  }

  function fieldHint(field, info, connected, busy) {
    if (!connected) return "Connecting";
    if (!info) return "Checking availability";
    if (info.supported === false || info.capabilities?.[field] !== true) return "Unavailable for this backend";
    if (info.state === "priming") return "Wait for playback";
    if (["stopping", "stop_requested"].includes(info.state)) return "Wait for stop";
    if (!["running", "stopped"].includes(info.state)) return "Playback unavailable";
    const application = info.application?.[field] || applicationFor(field);
    if (application === "stopped-only" && info.state !== "stopped") return "Stop to change";
    if (busy && field === "tempo_bpm") return "Waiting for server";
    if (busy && stoppedOnly.has(field)) return "Applying setup";
    if (application === "stopped-only") return "Ready while stopped";
    if (field === "tempo_bpm") return "Server-coordinated";
    return applicationLabels[application] || "Unavailable for this backend";
  }

  function playedEvidence(snapshot) {
    const audio = snapshot.audio || {}, current = audio.timing_evidence;
    const currentBar = audio.current_bar;
    const hasPlayhead = Number.isInteger(currentBar) && currentBar >= 0;
    const evidence = [];
    if (Number.isInteger(current?.bar) && current.bar >= 0
      && (!Object.hasOwn(audio, "current_bar") || (hasPlayhead && current.bar <= currentBar))) evidence.push(current);
    // Frozen means committed, not played. Only the audio playhead admits history.
    if (hasPlayhead) for (const bar of Object.values(snapshot.frozen_bars || {})) {
      if (!Number.isInteger(bar.bar) || bar.bar < 0 || bar.bar > currentBar || bar.frozen === false) continue;
      if (bar.session_id !== undefined && bar.session_id !== snapshot.session_id) continue;
      if (bar.coordinator_epoch !== undefined && bar.coordinator_epoch !== snapshot.coordinator_epoch) continue;
      if (bar.timing_evidence?.bar === bar.bar) evidence.push(bar.timing_evidence);
    }
    return evidence.sort((left, right) => left.bar - right.bar);
  }

  function settingsStatus(accepted, played, audioOutput) {
    const live = (accepted.changed_fields || Object.keys(accepted.values || {})).filter((field) => !stoppedOnly.has(field));
    if (!live.length) return { state: "saved", text: "Saved for next playback." };
    const values = { tempo_bpm: played?.timing?.bpm, chord_loop: played?.mix?.chord_loop,
      alignment_policy: played?.warp?.policy ?? played?.warp_diagnostics?.policy };
    const atBoundary = Number.isInteger(played?.bar) && Number.isInteger(accepted.effective_from_bar) && played.bar >= accepted.effective_from_bar;
    const immediate = (field) => (accepted.application?.[field] || applicationFor(field)) === "immediate";
    const mix = audioOutput?.live_mix;
    if (live.every((field) => {
      if (gains.has(field) && immediate(field)) {
        return mix?.supported === true && mix.supported_gains?.includes(field)
          && Number.isInteger(mix.applied_revision) && mix.applied_revision === mix.revision
          && mix.applied_revision >= (accepted.live_mix?.revision ?? 0)
          && mix.applied?.[field] !== undefined && mix.applied[field] === accepted.values?.[field];
      }
      const observed = gains.has(field) ? played?.mix?.[field] : values[field];
      return atBoundary && observed !== undefined && observed === accepted.values?.[field];
    })) {
      const bar = live.every((field) => gains.has(field) && immediate(field)) ? mix.bar : played.bar;
      return Number.isInteger(bar) ? { state: "applied", text: `Applied in bar ${bar + 1}.`, bar } : { state: "applied", text: "Applied to live output." };
    }
    if (live.every((field) => gains.has(field) && immediate(field))) {
      const missing = live.find((field) => Array.isArray(mix?.supported_gains) && !mix.supported_gains.includes(field));
      return { state: "pending", text: missing ? `Saved; awaiting ${missing.replace("_gain", "")} playback.` : "Accepted locally; awaiting audio output." };
    }
    return { state: "pending", text: Number.isInteger(accepted.effective_from_bar) ? `Scheduled for bar ${accepted.effective_from_bar + 1}. Awaiting playback.` : "Accepted; awaiting playback." };
  }

  function mount(document, { onChange = () => {} } = {}) {
    const get = (id) => document.getElementById(id);
    if (!get("settings-form")) return null;
    const control = (field) => get(`setting-${field.replaceAll("_", "-")}`);
    const groups = Object.fromEntries(Object.keys(groupFields).map((name) => [name, {
      dirty: new Set(), request: null, serial: 0, error: null, errorKind: null, accepted: null, evidence: new Map(), confirmed: new Map(), uncertain: null, timer: null, autoReady: false,
    }]));
    const state = { info: null, connected: false, loading: null, error: null, timer: null, version: 0, epoch: null, sessionId: null, played: [], outputState: null, runtimeState: null };
    const statusId = (name) => name === "generation" ? "settings-status" : `settings-${name}-status`;
    const groupFor = (field) => Object.keys(groupFields).find((name) => groupFields[name].includes(field));
    const set = (id, value) => { const node = get(id); if (node && node.textContent !== String(value)) node.textContent = value; };
    function read(field) {
      if (field === "mode") return get("setting-mode-alternating").checked ? "alternating" : "continuous";
      if (field === "output_device") return control(field).value || null;
      return control(field).value;
    }
    function fill(field, value) {
      if (field === "mode") {
        get("setting-mode-continuous").checked = value === "continuous";
        get("setting-mode-alternating").checked = value === "alternating";
      } else control(field).value = value ?? "";
    }
    const currentValue = (field) => stringFields.has(field) ? read(field) : Number(read(field));
    function clearMatching(group, values) {
      for (const [field, value] of Object.entries(values)) if (currentValue(field) === value) group.dirty.delete(field);
    }
    function observedStatus(group) {
      const statuses = [...group.evidence].map(([field, ack]) => {
        const identity = JSON.stringify([ack.values?.[field], ack.effective_from_bar, ack.revision, ack.mix_revision, ack.live_mix?.revision, ack.coordinator_epoch]);
        if (group.confirmed.get(field)?.identity !== identity) group.confirmed.delete(field);
        if (!group.confirmed.has(field) && !group.request?.fields.includes(field)) {
          const observed = [null, ...state.played].map((played) => settingsStatus(ack, played, state.info?.audio_output)).find((status) => status.state === "applied");
          if (observed) group.confirmed.set(field, { identity, status: observed });
        }
        return { field, status: group.confirmed.get(field)?.status || settingsStatus(ack, null, state.info?.audio_output) };
      });
      const pending = statuses.find(({ field, status }) => !gains.has(field) && status.state === "pending")
        || statuses.find(({ status }) => status.state === "pending");
      if (pending) {
        if (state.info?.state === "stopped") return { state: "saved", text: "Saved for next playback." };
        return pending.status;
      }
      return statuses.map(({ status }) => status).filter((status) => status.state === "applied").sort((left, right) => (right.bar ?? -1) - (left.bar ?? -1))[0]
        || statuses[0]?.status || null;
    }
    function render() {
      for (const field of fields) {
        const group = groups[groupFor(field)];
        const enabled = fieldEnabled(field, state.info, state.connected) && (groupFor(field) === "local" || !group.request);
        if (field === "mode") {
          get("setting-mode-continuous").disabled = get("setting-mode-alternating").disabled = !enabled;
        } else control(field).disabled = !enabled;
        const hint = fieldHint(field, state.info, state.connected, Boolean(group.request));
        set(`setting-${field.replaceAll("_", "-")}-hint`, hint);
        for (const node of field === "mode" ? [get("setting-mode-continuous"), get("setting-mode-alternating")] : [control(field)]) node.title = hint;
      }
      for (const field of ["drum_gain", "vocal_gain", "chord_gain"]) set(`setting-${field.replaceAll("_", "-")}-value`, `${Number(read(field) || 0).toFixed(2)}x`);
      for (const [name, group] of Object.entries(groups)) {
        const status = observedStatus(group);
        let message = !state.info ? "Loading settings..." : state.info.supported === false ? "Settings unavailable for this runtime." : "No pending settings change.";
        let statusState = status?.state || "idle";
        if (status) message = status.text;
        if (group.dirty.size) { message = name === "local" ? "Local changes pending." : "Unsaved settings."; statusState = "pending"; }
        if (group.request) { message = name === "generation" ? "Waiting for server..." : name === "local" ? "Applying locally..." : "Applying setup..."; statusState = "pending"; }
        if (!state.connected) { message = "Connecting..."; statusState = "idle"; }
        if (group.error || state.error) { message = group.error || state.error; statusState = "error"; }
        get(statusId(name)).dataset.state = statusState;
        set(statusId(name), message);
        get(statusId(name)).title = group.accepted ? `Accepted revision ${group.accepted.revision ?? group.accepted.mix_revision ?? "--"}` : "";
        get(`settings-${name}-group`)?.setAttribute("aria-busy", String(Boolean(group.request)));
      }
      for (const [name, id, label] of [["generation", "settings-apply", "Apply tempo"], ["session", "settings-session-apply", "Apply setup"], ["local", "settings-local-retry", "Retry local changes"]]) {
        const group = groups[name];
        const button = get(id);
        button.disabled = Boolean(group.request || group.uncertain) || !state.connected || ![...group.dirty].some((field) => field !== "output_device" && fieldEnabled(field, state.info, true));
        if (name === "local") button.hidden = !group.error || Boolean(group.uncertain);
        button.title = group.uncertain ? "Checking the previous request outcome" : !state.connected ? "Connecting" : group.request ? (name === "generation" ? "Waiting for server" : "Applying changes") : name === "session" && state.info?.state !== "stopped" ? "Stop to change" : button.disabled ? "No changes to apply" : label;
        set(id, group.request ? (name === "generation" ? "Waiting..." : "Applying...") : label);
      }
      const values = state.info?.values;
      set("settings-current", values ? `Requested: ${values.tempo_bpm} BPM / ${values.mode} / ${values.response_bars} response bars` : "");
      const output = state.info?.audio_output;
      const checkingOutput = output && state.info?.state === "running" && state.outputState !== "running";
      const audioWarning = output?.error || (checkingOutput ? "Checking audio output..." : state.info?.state === "running" && output?.audible === false ? `No audible output: ${output.backend || "software"} playback clock only.` : "");
      set("settings-audio-status", [audioWarning, output?.recovery_requires_reset === true ? "Stop, wait for stopped, Reset, then Start to recover audio output." : ""].filter(Boolean).join(" "));
    }
    function applyInfo(info, names = Object.keys(groups)) {
      // POST snapshots can arrive out of order across groups. Merge only their owner.
      const merged = { ...state.info, supported: info.supported ?? state.info?.supported,
        state: state.runtimeState || info.state || state.info?.state, values: { ...state.info?.values },
        capabilities: { ...state.info?.capabilities }, application: { ...state.info?.application }, groups: { ...state.info?.groups } };
      for (const name of names) {
        const group = groups[name];
        const metadata = info.groups?.[name];
        for (const field of groupFields[name]) {
          if (field in (info.values || {})) merged.values[field] = info.values[field];
          if (info.capabilities) merged.capabilities[field] = info.capabilities[field] === true;
          if (field in (info.application || {})) merged.application[field] = info.application[field];
        }
        if (metadata) merged.groups[name] = metadata;
        const changed = (metadata?.changed_fields || (!info.groups ? info.changed_fields : null) || []).filter((field) => groupFields[name].includes(field));
        if (changed.length) group.accepted = { ...(metadata || info), application: { ...merged.application }, changed_fields: changed, values: { ...merged.values } };
        if (group.uncertain && Object.entries(group.uncertain).every(([field, value]) => merged.values[field] === value)) {
          group.accepted = { ...(metadata || info), application: { ...merged.application }, changed_fields: Object.keys(group.uncertain), values: { ...merged.values } };
          clearMatching(group, group.uncertain);
          group.uncertain = null;
          if (group.errorKind === "unknown") { group.error = null; group.errorKind = null; }
        }
        // A newer gain ACK must not hide an alignment change still awaiting playback.
        for (const field of group.accepted?.changed_fields || []) group.evidence.set(field, { ...group.accepted, changed_fields: [field] });
      }
      if (names.includes("session")) {
        merged.devices = info.devices ?? merged.devices;
        merged.device_error = info.device_error ?? null;
      }
      if (names.includes("local") && info.audio_output) { merged.audio_output = info.audio_output; state.outputState = info.state; }
      state.info = merged;
      info = merged;
      const device = control("output_device");
      const selected = groups.session.dirty.has("output_device") ? device.value : info.values?.output_device;
      const selectedMatches = (info.devices || []).filter((item) => item.id === selected || item.name === selected);
      const selectedDevice = selectedMatches.length === 1 ? selectedMatches[0] : null;
      device.replaceChildren();
      const defaultOption = document.createElement("option");
      defaultOption.value = ""; defaultOption.textContent = "System default"; device.append(defaultOption);
      for (const item of info.devices || []) {
        const option = document.createElement("option");
        option.value = item.id || item.name;
        option.textContent = item.label || [item.host_api, item.name].filter(Boolean).join(" / ");
        option.disabled = item.available === false;
        if (option.disabled) option.textContent += ` (unavailable: ${item.error || "unsupported output"})`;
        option.title = item.error || option.textContent;
        device.append(option);
      }
      if (selected && !selectedDevice) {
        const missing = document.createElement("option");
        missing.value = selected; missing.textContent = `${selected} (unavailable)`; missing.disabled = true; device.append(missing);
      }
      device.value = selectedDevice?.id || selected || "";
      set("settings-device-error", info.device_error || (selectedDevice?.available === false ? selectedDevice.error || "Selected output device unavailable." : selected && !selectedDevice ? "Configured output device unavailable or ambiguous." : ""));
      for (const field of fields) if (field !== "output_device" && !groups[groupFor(field)].dirty.has(field) && field in info.values) fill(field, info.values[field]);
      onChange(info); render();
      if (groups.local.autoReady && !groups.local.uncertain && !groups.local.error && !groups.local.request) submit("local");
    }
    function schedule() {
      root.clearTimeout(state.timer);
      if (state.connected) state.timer = root.setTimeout(load, document.hidden ? 10000 : 5000);
    }
    async function load() {
      if (state.loading || !state.connected) return;
      const loading = {};
      state.loading = loading;
      const version = state.version;
      // A pending POST may finish before this poll returns, leaving its serial unchanged.
      const serials = Object.fromEntries(Object.entries(groups)
        .filter(([, group]) => !group.request)
        .map(([name, group]) => [name, group.serial]));
      try {
        const response = await fetch("/api/control/settings", { cache: "no-store", signal: AbortSignal.timeout(5000) });
        const body = await response.json();
        if (!response.ok) throw new Error(typeof body.detail === "string" ? body.detail : `Settings unavailable (${response.status}).`);
        if (version === state.version && state.connected) {
          state.error = null;
          applyInfo(body, Object.keys(groups).filter((name) => !groups[name].request && groups[name].serial === serials[name]));
        }
      } catch (error) {
        if (version === state.version) { state.error = requestFailure(error, false); render(); }
      }
      finally { if (state.loading === loading) { state.loading = null; schedule(); } }
    }
    async function submit(name, selectedFields = groupFields[name]) {
      const group = groups[name];
      if (group.request || group.uncertain || !state.connected) return;
      if (name === "local" && group.dirty.has("alignment_policy")) selectedFields = ["alignment_policy"];
      let payload;
      try {
        const values = Object.fromEntries(selectedFields.filter((field) => group.dirty.has(field) && fieldEnabled(field, state.info, state.connected)).map((field) => [field, read(field)]));
        payload = makePayload(values, state.info?.values);
        if (!Object.keys(payload).length) { for (const field of Object.keys(values)) group.dirty.delete(field); render(); return; }
      } catch (error) { group.error = error.message; group.errorKind = "submit"; render(); return; }
      const request = { fields: Object.keys(payload), rebuild: state.info?.state === "stopped" && (name === "generation" || (name === "session" && !Object.hasOwn(payload, "output_device"))) };
      for (const field of request.fields) group.confirmed.delete(field);
      group.request = request; group.serial += 1; group.autoReady = false; group.error = null; group.errorKind = null;
      root.clearTimeout(group.timer); group.timer = null;
      let explicitlyRejected = false;
      render();
      try {
        const response = await fetch(`/api/control/settings/${name}`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload), signal: AbortSignal.timeout(submitTimeoutMs) });
        explicitlyRejected = !response.ok;
        const body = await response.json();
        if (!response.ok) throw new Error(typeof body.detail === "string" ? body.detail : `Settings rejected (${response.status}).`);
        if (group.request !== request) return;
        clearMatching(group, payload);
        const ack = body.ack || body.groups?.[name] || body;
        applyInfo({ ...body, groups: { ...body.groups, [name]: { ...ack, changed_fields: ack.changed_fields ?? Object.keys(payload) } } }, [name]);
      } catch (error) {
        if (group.request === request) {
          group.error = explicitlyRejected ? rejectionMessage(error) : requestFailure(error, true, name);
          group.errorKind = explicitlyRejected ? "submit" : "unknown";
          group.uncertain = explicitlyRejected ? null : payload;
        }
      }
      finally {
        if (group.request === request) {
          group.request = null;
          if (name === "local" && group.dirty.size && !group.timer && !group.error) submit("local");
          render(); schedule();
        }
      }
    }
    for (const field of fields) {
      const nodes = field === "mode" ? [get("setting-mode-continuous"), get("setting-mode-alternating")] : [control(field)];
      for (const node of nodes) node.addEventListener(stringFields.has(field) ? "change" : "input", () => {
        if (!fieldEnabled(field, state.info, state.connected)) return;
        const name = groupFor(field), group = groups[name];
        group.dirty.add(field);
        if (!group.uncertain) { group.error = null; group.errorKind = null; }
        render();
        if (name === "local") {
          root.clearTimeout(group.timer); group.timer = null;
          if (gains.has(field)) group.timer = root.setTimeout(() => { group.timer = null; group.autoReady = true; submit("local"); }, 150);
          else { group.autoReady = true; submit("local"); }
        } else if (field === "output_device") submit("session", ["output_device"]);
      });
    }
    get("settings-form").addEventListener("submit", (event) => { event.preventDefault(); return submit("generation"); });
    get("settings-session-form").addEventListener("submit", (event) => { event.preventDefault(); return submit("session", ["mode", "response_bars"]); });
    get("settings-local-retry").addEventListener("click", () => submit("local"));
    function invalidate(preserveRebuilds = false, clearEdits = true) {
      state.version += 1; state.loading = null; state.error = null;
      if (clearEdits) state.played = [];
      if (state.info) state.info.audio_output = null;
      for (const [name, group] of Object.entries(groups)) {
        if (clearEdits) group.confirmed.clear();
        if (preserveRebuilds && group.request?.rebuild) continue;
        root.clearTimeout(group.timer);
        group.timer = null; group.request = null; group.serial += 1; group.uncertain = null; group.autoReady = false;
        if (clearEdits) { group.accepted = null; group.evidence.clear(); group.dirty.clear(); group.error = null; group.errorKind = null; }
      }
    }
    render();
    return {
      load,
      connection(connected) {
        const changed = state.connected !== connected;
        state.connected = connected;
        if (!connected) { root.clearTimeout(state.timer); invalidate(false, false); }
        else if (changed) load();
        render();
      },
      update(snapshot) {
        if (snapshot.session_id) {
          const epoch = `${snapshot.session_id}:${snapshot.coordinator_epoch ?? 0}`;
          if (state.epoch !== null && epoch !== state.epoch) {
            // Stopped tempo/setup may emit their new epoch before the POST reply.
            invalidate(snapshot.session_id === state.sessionId);
            load();
          }
          state.epoch = epoch;
          state.sessionId = snapshot.session_id;
        }
        state.played = playedEvidence(snapshot);
        if (state.info && snapshot.audio_output) { state.info.audio_output = snapshot.audio_output; state.outputState = snapshot.audio?.state || state.info.state; }
        state.runtimeState = snapshot.audio?.state || state.runtimeState;
        if (state.info && snapshot.audio?.state) {
          if (state.info.state !== snapshot.audio.state) state.outputState = null;
          state.info.state = snapshot.audio.state; onChange(state.info); render();
        }
        render();
      },
      reset() { invalidate(); load(); render(); },
    };
  }

  root.StreamMuseRapSettings = { makePayload, fieldEnabled, settingsStatus, mount };
})(typeof window !== "undefined" ? window : globalThis);
