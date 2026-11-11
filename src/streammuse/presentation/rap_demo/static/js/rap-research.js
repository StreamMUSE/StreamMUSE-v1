(function (root) {
  "use strict";

  const MAX_SECONDS = 15;
  const LOG_LIMIT = 200;

  function encodeWav(samples, sampleRate) {
    if (!Number.isInteger(sampleRate) || sampleRate < 8000 || sampleRate > 48000) {
      throw new RangeError("Unsupported PCM sample rate");
    }
    const length = Math.min(samples.length, sampleRate * MAX_SECONDS);
    const buffer = new ArrayBuffer(44 + length * 2);
    const view = new DataView(buffer);
    const ascii = (offset, value) => {
      for (let i = 0; i < value.length; i += 1) view.setUint8(offset + i, value.charCodeAt(i));
    };
    ascii(0, "RIFF");
    view.setUint32(4, 36 + length * 2, true);
    ascii(8, "WAVE");
    ascii(12, "fmt ");
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    ascii(36, "data");
    view.setUint32(40, length * 2, true);
    for (let i = 0; i < length; i += 1) {
      const sample = Number.isFinite(samples[i]) ? Math.max(-1, Math.min(1, samples[i])) : 0;
      view.setInt16(44 + i * 2, Math.round(sample * (sample < 0 ? 32768 : 32767)), true);
    }
    return buffer;
  }

  function reduceEvents(previous, incoming, limit = LOG_LIMIT) {
    if (limit <= 0) return [];
    const key = (event) => {
      const run = event.run_id ?? event.session_id ?? "";
      const id = event.event_id ?? event.id ?? event.sequence;
      return id == null ? JSON.stringify(event) : `${run}:${id}`;
    };
    const entries = new Map();
    for (const event of [...previous, ...incoming]) {
      if (event && typeof event === "object") entries.set(key(event), event);
    }
    return Array.from(entries.values()).slice(-Math.max(0, limit));
  }

  function storyBudget(snapshot, text, eventId) {
    const events = snapshot?.events ?? [];
    const limits = snapshot?.limits ?? {};
    const maxBytes = limits.total_text_utf8_bytes ?? 512;
    const maxEvents = limits.events ?? 32;
    const maxChars = limits.text_chars ?? 512;
    const encoder = new TextEncoder();
    const usedBytes = events.reduce((sum, event) => sum + encoder.encode(event.text).length, 0);
    const draft = text.trim();
    const draftBytes = encoder.encode(draft).length;
    const existing = eventId == null ? null : events.find((event) => event.event_id === eventId);
    let reason = "";
    if (!draft) reason = "Enter a story event.";
    else if (existing && existing.text !== draft) reason = "Event ID already belongs to a different direction.";
    else if (!existing && events.length >= maxEvents) reason = `Story limit reached: ${maxEvents} events.`;
    else if (Array.from(draft).length > maxChars) reason = `Story event exceeds ${maxChars} characters.`;
    else if (!existing && usedBytes + draftBytes > maxBytes) reason = `Story exceeds its ${maxBytes}-byte UTF-8 budget (${Math.max(0, maxBytes - usedBytes)} bytes remaining).`;
    return { usedBytes, draftBytes, maxBytes, maxEvents, eventCount: events.length, canSubmit: !reason, reason };
  }

  function reduceCapture(state, action) {
    if (action.token != null && action.token !== state.token) return state;
    if (action.type === "press" && ["idle", "error"].includes(state.phase)) {
      return { phase: "permission", token: state.token + 1 };
    }
    if (action.type === "cancel" || (action.type === "release" && state.phase === "permission")) {
      return { phase: "idle", token: state.token + 1 };
    }
    if (action.type === "ready" && state.phase === "permission") return { ...state, phase: "recording" };
    if (action.type === "release" && state.phase === "recording") return { ...state, phase: "transcribing" };
    if (action.type === "done") return { ...state, phase: "idle" };
    if (action.type === "error") return { ...state, phase: "error" };
    return state;
  }

  // Tokens prevent a late permission grant from restarting a cancelled capture.
  function createRecorder(options) {
    const env = options.env || root;
    let state = { phase: "idle", token: 0 };
    let resources = null;
    let chunks = [];
    let sampleCount = 0;
    let timer = null;
    let startedAt = 0;
    let rate = 0;
    const emit = (action) => { state = reduceCapture(state, action); options.onState(state); };
    const dispose = (owned) => {
      if (!owned) return;
      owned.stream?.getTracks().forEach((track) => track.stop());
      if (owned.processor) owned.processor.onaudioprocess = null;
      for (const node of [owned.source, owned.processor, owned.gain]) {
        try { node?.disconnect(); } catch (_) { /* Already disconnected. */ }
      }
      if (owned.context && owned.context.state !== "closed") void owned.context.close().catch(() => {});
    };
    const cleanup = () => {
      clearTimeout(timer);
      timer = null;
      dispose(resources);
      resources = null;
    };
    const cancel = () => {
      cleanup();
      chunks = [];
      sampleCount = 0;
      emit({ type: "cancel" });
    };
    const release = () => {
      if (state.phase === "permission") { cancel(); return; }
      if (state.phase !== "recording") return;
      cleanup();
      const samples = new Float32Array(sampleCount);
      let offset = 0;
      for (const chunk of chunks) { samples.set(chunk, offset); offset += chunk.length; }
      chunks = [];
      emit({ type: "release" });
      if (!samples.length) {
        emit({ type: "error" });
        options.onError(new Error("No audio captured. Hold the button longer and try again."));
        return;
      }
      options.onAudio(encodeWav(samples, rate), state.token, samples.length / rate);
    };
    const press = async () => {
      if (!["idle", "error"].includes(state.phase)) return;
      emit({ type: "press" });
      const token = state.token;
      let owned = null;
      try {
        if (!env.navigator?.mediaDevices?.getUserMedia || !(env.AudioContext || env.webkitAudioContext)) {
          throw new Error("Microphone capture requires HTTPS or localhost and browser Web Audio support.");
        }
        const stream = await env.navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1 }, video: false });
        owned = { stream };
        if (state.token !== token || state.phase !== "permission") { dispose(owned); return; }
        resources = owned;
        owned.context = new (env.AudioContext || env.webkitAudioContext)({ sampleRate: 16000 });
        await owned.context.resume();
        if (state.token !== token || state.phase !== "permission") { dispose(owned); return; }
        rate = owned.context.sampleRate;
        if (rate < 8000 || rate > 48000) throw new Error("Microphone sample rate must be between 8 and 48 kHz.");
        chunks = [];
        sampleCount = 0;
        owned.source = owned.context.createMediaStreamSource(stream);
        owned.processor = owned.context.createScriptProcessor(2048, 1, 1);
        owned.gain = owned.context.createGain();
        owned.gain.gain.value = 0;
        owned.processor.onaudioprocess = (event) => {
          if (state.token !== token || state.phase !== "recording") return;
          const input = event.inputBuffer.getChannelData(0);
          const count = Math.min(input.length, MAX_SECONDS * rate - sampleCount);
          if (count > 0) { chunks.push(new Float32Array(input.subarray(0, count))); sampleCount += count; }
          if (sampleCount >= MAX_SECONDS * rate) release();
        };
        owned.source.connect(owned.processor);
        owned.processor.connect(owned.gain);
        owned.gain.connect(owned.context.destination);
        startedAt = Date.now();
        emit({ type: "ready", token });
        timer = setTimeout(release, MAX_SECONDS * 1000);
      } catch (error) {
        dispose(owned);
        if (state.token !== token) return;
        cleanup();
        emit({ type: "error", token });
        options.onError(error);
      }
    };
    return {
      press, release, cancel,
      complete: (token) => emit({ type: "done", token }),
      fail: (token) => emit({ type: "error", token }),
      state: () => state,
      elapsed: () => state.phase === "recording" ? Math.min(MAX_SECONDS, (Date.now() - startedAt) / 1000) : 0,
    };
  }

  function modelLabel(data) {
    const model = data.model ?? data.model_name ?? data.backend;
    const status = data.model_status ?? data.backend_status ?? data.transcriber_state ?? (data.available === false ? "Unavailable" : null);
    const name = typeof model === "object" && model ? model.name ?? model.id : model;
    const reported = status ?? (typeof model === "object" && model ? model.status : null);
    return [name, reported || "Status not reported"].filter(Boolean).join(" / ");
  }

  function formatMilliseconds(value) {
    return typeof value === "number" && Number.isFinite(value) ? `${Math.round(value)} ms` : "Not reported";
  }

  function playbackView(snapshot) {
    const playback = snapshot.current_playback ?? {};
    const segment = snapshot.current_segment ?? snapshot.current ?? {};
    const tick = playback.tick ?? snapshot.current_tick;
    const bar = playback.bar ?? snapshot.current_bar ?? segment.bar ?? (Number.isInteger(tick) ? Math.floor(tick / 16) : null);
    let current = segment.bar === bar ? { ...segment } : {};
    for (const collection of [snapshot.bars, snapshot.frozen_bars]) {
      for (const item of Object.values(collection ?? {})) {
        if (item.bar === bar) current = { ...current, ...item };
      }
    }
    const nonLyric = { flow_echo: "Captured rhythm replay", drums_idle: "Drums only", capture_transition: "Drums only", silence: "Silence" };
    const frozen = Object.values(snapshot.frozen_bars ?? {}).find((item) => item.bar === bar);
    return {
      bar: Number.isInteger(bar) ? bar + 1 : null,
      state: snapshot.audio?.state ?? (snapshot.stopped ? "stopped" : "Not reported"),
      text: current.text || nonLyric[current.source ?? current.choice] || "Waiting for a lyric.",
      topic: current.topic ?? snapshot.session_metadata?.topic ?? "--",
      source: current.fallback ? "fallback" : current.source ?? current.choice ?? "--",
      storyVersion: Number.isInteger(frozen?.story_version) && frozen.story_version >= 0 ? frozen.story_version : null,
      error: snapshot.runtime_error ?? snapshot.audio?.runtime_error ?? "",
    };
  }

  function mount(document) {
    const get = (id) => document.getElementById(id);
    const set = (id, text) => { const node = get(id); const next = String(text ?? "--"); if (node.textContent !== next) node.textContent = next; };
    const textValue = (value) => typeof value === "object" && value !== null ? JSON.stringify(value) : String(value ?? "--");
    const error = (section, message = "") => { set(`${section}-error`, message); get(`${section}-error`).hidden = !message; };
    const feedback = (section, message, success = false) => { set(`${section}-feedback`, message); get(`${section}-feedback`).dataset.state = success ? "success" : ""; };
    const create = (tag, value, className) => { const node = document.createElement(tag); if (value != null) node.textContent = String(value); if (className) node.className = className; return node; };
    let activeTab = "story";
    let disposed = false;
    let pollTimer = null;
    let pollBusy = false;
    let experimentBusy = false;
    let experimentState = null;
    let characterEvents = [];
    let acceptedEvents = [];
    let actorNames = {};
    let eventCursor = 0;
    let runId = null;
    let storyBusy = false;
    let pendingStory = null;
    let staleStoryDraft = null;
    let storySnapshot = null;
    let knownStorySessionId = null;
    let storySessionEpoch = 0;
    let storyRevision = 0;
    let experimentRevision = 0;
    let microphoneDestination = "transcript";
    let microphoneStorySessionId = null;
    let transcriptionAbort = null;
    let captureDisplayTimer = null;
    let pointerId = null;
    let keyboardHeld = false;
    let playbackState = null;
    let playbackContext = null;
    let playbackBusy = false;
    let playbackRevision = 0;
    const requests = new Set();
    const freshId = () => root.crypto?.randomUUID?.() || `research-${Date.now()}-${Math.random().toString(16).slice(2)}`;

    async function request(path, options = {}) {
      const controller = new AbortController();
      requests.add(controller);
      const upstreamAbort = () => controller.abort();
      options.signal?.addEventListener("abort", upstreamAbort, { once: true });
      if (options.signal?.aborted) controller.abort();
      const timer = setTimeout(() => controller.abort(), path.endsWith("/transcribe") ? 90000 : 15000);
      try {
        const response = await root.fetch(path, { cache: "no-store", ...options, signal: controller.signal });
        const body = await response.text();
        let data;
        try { data = body ? JSON.parse(body) : {}; } catch (_) { throw new Error(`Invalid JSON from ${path} (${response.status})`); }
        if (!response.ok) throw new Error(`${response.status}: ${textValue(data.detail ?? data.error ?? data.message ?? response.statusText)}`);
        if (!data || typeof data !== "object") throw new Error(`Invalid response from ${path}`);
        return data;
      } catch (failure) {
        if (failure.name === "AbortError" && !options.signal?.aborted && !disposed) throw new Error("Request timed out. Refresh to check the server state.");
        throw failure;
      } finally {
        clearTimeout(timer);
        requests.delete(controller);
        options.signal?.removeEventListener("abort", upstreamAbort);
      }
    }
    const post = (path, body, signal) => request(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body), signal });

    function updatePlaybackControls() {
      get("story-start").disabled = playbackBusy || playbackState !== "stopped";
      get("story-stop").disabled = playbackBusy || !["priming", "running"].includes(playbackState);
    }

    function renderStoryVersions() {
      const baseSession = playbackContext?.sessionId;
      const epoch = playbackContext?.coordinatorEpoch;
      // Publisher snapshots use the base ID; the story controller appends its epoch.
      const controllerSession = baseSession && Number.isInteger(epoch) && epoch >= 0 ? `${baseSession}-e${epoch}` : null;
      const sameSession = baseSession && validSessionId(storySnapshot?.session_id) &&
        (baseSession === storySnapshot.session_id || controllerSession === storySnapshot.session_id);
      const playing = sameSession ? playbackContext.storyVersion : null;
      const accepted = storySnapshot?.version;
      set("story-playback-version", playing == null ? "--" : `v${playing}`);
      set("story-latest-version", Number.isInteger(accepted) && accepted >= 0 ? `v${accepted}` : "--");
    }

    async function loadPlayback() {
      const revision = playbackRevision;
      try {
        const snapshot = await request("/api/state");
        if (revision !== playbackRevision || disposed) return;
        const view = playbackView(snapshot);
        playbackState = view.state;
        playbackContext = { sessionId: snapshot.session_id, coordinatorEpoch: snapshot.coordinator_epoch, storyVersion: view.storyVersion };
        renderStoryVersions();
        set("story-playback-state", view.state);
        set("story-playback-bar", `Bar ${view.bar ?? "--"}`);
        set("story-playback-text", view.text);
        set("story-playback-topic", view.topic);
        set("story-playback-source", view.source);
        error("playback", view.error);
      } catch (failure) {
        if (revision !== playbackRevision || disposed) return;
        playbackState = null;
        playbackContext = null;
        renderStoryVersions();
        set("story-playback-state", "Unavailable / stale");
        error("playback", failure.message);
      }
      updatePlaybackControls();
    }

    for (const action of ["start", "stop"]) {
      get(`story-${action}`).addEventListener("click", async () => {
        if (playbackBusy) return;
        playbackBusy = true;
        playbackRevision += 1;
        updatePlaybackControls();
        error("playback");
        feedback("playback", action === "start" ? "Starting..." : "Stopping...");
        try {
          const result = await request(`/api/control/${action}`, { method: "POST" });
          playbackRevision += 1;
          playbackState = result.state ?? null;
          set("story-playback-state", playbackState);
          feedback("playback", action === "start" ? "Start accepted." : "Stop accepted.", true);
          await loadPlayback();
        } catch (failure) {
          error("playback", failure.message);
          feedback("playback", "Control not confirmed. Refresh to check playback.");
        } finally { playbackBusy = false; updatePlaybackControls(); }
      });
    }

    function renderLog(id, events, empty) {
      const list = get(id);
      const signature = JSON.stringify(events);
      if (list.dataset.signature === signature) return;
      list.dataset.signature = signature;
      const following = list.scrollHeight - list.scrollTop - list.clientHeight < 40;
      list.replaceChildren(...events.map((event) => {
        const payload = event.payload && typeof event.payload === "object" ? event.payload : {};
        const row = create("li");
        const label = create("div", null, "event-label");
        label.append(create("strong", textValue(actorNames[event.actor_id] ?? event.actor_id ?? event.persona ?? event.speaker ?? event.character ?? payload.persona ?? payload.speaker ?? event.source ?? event.event_type ?? event.type ?? "Event")));
        label.append(create("span", event.kind ? `${event.kind.replaceAll("_", " ")} #${event.sequence}` : textValue(event.event_id ?? event.id ?? event.sequence ?? event.turn ?? "")));
        let content = event.text ?? event.content ?? event.message ?? payload.text ?? payload.content ?? event.detail;
        if (!content) content = [event.state, event.error, event.reason, event.request_id ? `Request ${event.request_id}` : null, event.context_version != null ? `Context ${event.context_version}` : null, event.mode, event.model].filter(Boolean).join(" / ") || textValue(payload);
        if (event.kind === "proposal") content += `\n${event.outcome ?? "pending"}${event.reason ? ` / ${event.reason}` : ""}`;
        row.append(label, create("div", textValue(content), "event-content"));
        return row;
      }));
      if (!events.length) list.append(create("li", empty, "research-empty"));
      if (following) list.scrollTop = list.scrollHeight;
    }

    function renderStory(data) {
      const story = data.story ?? data;
      const sessionId = validSessionId(story.session_id);
      if (sessionId && knownStorySessionId && knownStorySessionId !== sessionId) {
        storySessionEpoch += 1;
        if (pendingStory) {
          staleStoryDraft = pendingStory.text;
          pendingStory = null;
          feedback("story", "Story session changed. Retained draft is not retried; edit it before a new submission.");
        } else if (get("story-feedback").dataset.state === "success") {
          feedback("story", "Story session changed. Previous acceptance belongs to the earlier session.");
        }
      }
      if (sessionId) knownStorySessionId = sessionId;
      storySnapshot = story;
      renderStoryVersions();
      const context = story.context ?? story.story_context;
      const events = Array.isArray(story.events) ? story.events : Array.isArray(context?.events) ? context.events : [];
      set("story-state", story.status ?? story.state ?? (story.enabled === false ? "No directions" : "Enabled"));
      set("story-version", story.version ?? context?.version);
      set("story-model", modelLabel(story));
      set("story-count", events.length);
      set("story-context", story.continuity?.startsWith("directions_only") ? "Directions only. Playback continuity not established." : story.continuity ?? "Not reported.");
      get("story-text").maxLength = story.limits?.text_chars ?? 512;
      updateStoryBudget();
      renderLog("story-log", events.slice(-LOG_LIMIT), "No events yet.");
    }

    function updateStoryBudget() {
      const text = get("story-text").value.trim();
      const budget = storyBudget(storySnapshot, text, pendingStory?.text === text ? pendingStory.event_id : undefined);
      const sessionKnown = typeof storySnapshot?.session_id === "string" && Boolean(storySnapshot.session_id.trim());
      const staleDraft = staleStoryDraft !== null && staleStoryDraft === text;
      set("story-budget", `${budget.usedBytes} / ${budget.maxBytes} bytes used; draft ${budget.draftBytes} bytes; ${budget.eventCount} / ${budget.maxEvents} events`);
      get("story-submit").disabled = storyBusy || !sessionKnown || staleDraft || !budget.canSubmit;
      get("story-text").setCustomValidity(staleDraft ? "Edit the retained draft before submitting to the new session." : text && !budget.canSubmit ? budget.reason : "");
    }
    get("story-text").addEventListener("input", () => {
      if (get("story-text").value.trim() !== staleStoryDraft) staleStoryDraft = null;
      updateStoryBudget();
    });
    updateStoryBudget();

    async function loadStory() {
      const revision = storyRevision;
      const data = await request("/api/story");
      if (revision !== storyRevision || disposed) return;
      renderStory(data);
      error("story");
      return storySnapshot;
    }

    function validSessionId(value) {
      return typeof value === "string" && value.trim() ? value : null;
    }

    function sessionFailure(message, changed = false) {
      const failure = new Error(message);
      failure.code = changed ? "story_session_changed" : "story_session_unconfirmed";
      return failure;
    }

    function requireStorySession(sessionId) {
      if (!validSessionId(sessionId)) {
        throw sessionFailure("Story session was unavailable when this event began. Refresh and create a new event; retained text was not submitted.");
      }
      if (knownStorySessionId && knownStorySessionId !== sessionId) {
        throw sessionFailure("Story session changed. This event cannot be submitted or confirmed in the current session.", true);
      }
      if (!validSessionId(storySnapshot?.session_id)) {
        throw sessionFailure("Current story session is unconfirmed. Refresh before retrying the original event.");
      }
    }

    async function submitStory(text, eventId, { signal, source, sessionId } = {}) {
      requireStorySession(sessionId);
      const epoch = storySessionEpoch;
      const budget = storyBudget(storySnapshot, text, eventId);
      if (!budget.canSubmit) throw new Error(budget.reason);
      storyRevision += 1;
      const body = { text, event_id: eventId, session_id: sessionId };
      if (source) body.source = source;
      let result;
      try {
        result = await post("/api/story/events", body, signal);
      } catch (failure) {
        // Refresh after a rejection without ever rebinding the pending event.
        if (failure.name === "AbortError") throw failure;
        try { await loadStory(); } catch (_) { /* Keep the original submission error. */ }
        requireStorySession(sessionId);
        throw failure;
      }
      storyRevision += 1;
      let confirmed;
      try { confirmed = await loadStory(); }
      catch (_) { /* The POST may have succeeded; an unavailable GET must not discard its retry identity. */ }
      requireStorySession(sessionId);
      if (storySessionEpoch !== epoch ||
          (validSessionId(result.session_id) && result.session_id !== sessionId) ||
          (validSessionId(confirmed?.session_id) && confirmed.session_id !== sessionId)) {
        throw sessionFailure("Story acknowledgement belongs to a previous or different session. It is not confirmed in the current session.", true);
      }
      if (result.session_id !== sessionId || confirmed?.session_id !== sessionId) {
        throw sessionFailure("Story acknowledgement is unconfirmed. Text retained; retry uses the original event ID and session.");
      }
      return result;
    }

    get("story-form").addEventListener("submit", async (event) => {
      event.preventDefault();
      const text = get("story-text").value.trim();
      if (!text || storyBusy) return;
      if (staleStoryDraft === text) {
        error("story", "Story session changed. Edit the retained draft before submitting to the new session.");
        feedback("story", "Old-session draft retained; no retry sent.");
        return;
      }
      if (!pendingStory || pendingStory.text !== text) pendingStory = { text, event_id: freshId(), session_id: storySnapshot?.session_id ?? null };
      const submission = pendingStory;
      storyBusy = true;
      get("story-submit").disabled = true;
      error("story");
      feedback("story", "Submitting event...");
      try {
        const result = await submitStory(text, submission.event_id, { sessionId: submission.session_id });
        if (get("story-text").value.trim() === text) get("story-text").value = "";
        const acceptedId = result.event?.event_id ?? result.event_id ?? submission.event_id;
        const boundary = result.effective_from_bar == null ? "" : ` / effective from bar ${result.effective_from_bar + 1}`;
        feedback("story", `${result.duplicate ? "Already accepted" : "Accepted"}: ${acceptedId}${boundary}`, true);
        if (pendingStory === submission) pendingStory = null;
      } catch (failure) {
        error("story", failure.message);
        if (failure.code === "story_session_changed") {
          staleStoryDraft = text;
          if (pendingStory === submission) pendingStory = null;
          feedback("story", "Session changed. Draft retained; edit it before a new submission.");
        } else feedback("story", "Not confirmed. Retrying unchanged text uses the same event ID and session.");
      } finally {
        storyBusy = false;
        updateStoryBudget();
      }
    });

    function updateExperimentControls() {
      const running = experimentState?.running === true || ["running", "starting", "stopping"].includes(experimentState?.status ?? experimentState?.state);
      for (const name of ["characters", "coordination"]) {
        get(`${name}-start`).disabled = experimentBusy || running || experimentState?.available === false;
        get(`${name}-stop`).disabled = experimentBusy || (experimentState != null && !running);
      }
      const dialogue = document.querySelector('input[name="character-mode"]:checked').value === "dialogue";
      get("characters-policy").disabled = dialogue || experimentBusy;
      if (dialogue) get("characters-policy").value = "fixed_turn";
      set("characters-start", dialogue ? "Start 8 turns" : "Start count");
    }
    document.querySelectorAll('input[name="character-mode"]').forEach((input) => input.addEventListener("change", updateExperimentControls));

    function renderCharacters(data) {
      const state = data.characters ?? data;
      experimentState = state;
      const currentRun = state.run_id ?? state.session_id;
      if (runId !== currentRun) { characterEvents = []; acceptedEvents = []; eventCursor = 0; runId = currentRun; }
      if (Array.isArray(state.accepted)) acceptedEvents = state.accepted;
      if (Array.isArray(state.events)) characterEvents = reduceEvents(characterEvents, state.events);
      for (const name of ["characters", "coordination"]) {
        set(`${name}-state`, state.status ?? state.state ?? (state.running === true ? "Running" : state.running === false ? "Idle" : "Not reported"));
        set(`${name}-model`, modelLabel(state));
        set(`${name}-active-mode`, state.mode);
        set(`${name}-active-policy`, state.policy?.replaceAll("_", " "));
        error(name, state.error ? textValue(state.error) : "");
      }
      const personas = state.actors ?? state.personas ?? state.agents ?? [];
      const entries = Array.isArray(personas) ? personas : Object.entries(personas).map(([name, detail]) => ({ name, description: textValue(detail) }));
      actorNames = Object.fromEntries(entries.filter((persona) => persona.actor_id).map((persona) => [persona.actor_id, persona.name ?? persona.persona?.split(":")[0] ?? persona.actor_id]));
      get("characters-personas").replaceChildren(...entries.map((persona) => create("p", typeof persona === "string" ? persona : persona.persona ?? [persona.name ?? persona.label ?? persona.id, persona.description].filter(Boolean).join(": "))));
      const metrics = state.metrics ?? data.metrics ?? {};
      const metricEntries = Object.entries(metrics);
      const rows = metricEntries.map(([name, value]) => {
        const row = create("tr");
        const display = typeof value === "number" && !Number.isInteger(value) ? value.toFixed(2) : textValue(value);
        row.append(create("td", name.replaceAll("_", " ")), create("td", display));
        return row;
      });
      if (!rows.length) { const row = create("tr"); const cell = create("td", "No metrics reported."); cell.colSpan = 2; row.append(cell); rows.push(row); }
      get("coordination-metrics").replaceChildren(...rows);
      renderCharacterLogs();
      updateExperimentControls();
    }

    function renderCharacterLogs() {
      renderLog("characters-log", acceptedEvents.slice(-LOG_LIMIT), "No accepted character turns yet.");
      renderLog("coordination-log", characterEvents, "No coordination events yet.");
    }

    async function loadCharacters() {
      const revision = experimentRevision;
      const data = await request("/api/characters/status");
      if (revision !== experimentRevision || disposed) return;
      renderCharacters(data);
      const expectedRun = runId;
      const events = await request(`/api/characters/events?after=${eventCursor}${runId ? `&run_id=${encodeURIComponent(runId)}` : ""}`);
      if (revision !== experimentRevision || disposed || runId !== expectedRun) return;
      if (events.run_id !== undefined && events.run_id !== runId) return;
      const rows = Array.isArray(events) ? events : events.events ?? [];
      if (!Array.isArray(rows)) throw new Error("Invalid character events response");
      characterEvents = reduceEvents(characterEvents, rows.filter((row) => !row.run_id || !runId || row.run_id === runId));
      acceptedEvents = reduceEvents(acceptedEvents, rows.filter((row) => row.kind === "accepted"));
      if (Number.isInteger(events.next_cursor)) eventCursor = events.next_cursor;
      renderCharacterLogs();
    }

    for (const name of ["characters", "coordination"]) {
      get(`${name}-form`).addEventListener("submit", async (event) => {
        event.preventDefault();
        if (experimentBusy) return;
        const mode = name === "coordination" ? "counting" : document.querySelector('input[name="character-mode"]:checked').value;
        experimentBusy = true;
        experimentRevision += 1;
        updateExperimentControls();
        error(name);
        feedback(name, "Starting...");
        try {
          const result = await post("/api/characters/start", { mode, policy: mode === "dialogue" ? "fixed_turn" : get(`${name}-policy`).value, turns: 8 });
          characterEvents = [];
          acceptedEvents = [];
          eventCursor = 0;
          renderCharacters(result);
          experimentRevision += 1;
          feedback(name, "Start accepted.", true);
          await loadCharacters();
        } catch (failure) { error(name, failure.message); feedback(name, "Refresh to check run state before retrying."); }
        finally { experimentBusy = false; updateExperimentControls(); }
      });
      get(`${name}-stop`).addEventListener("click", async () => {
        if (experimentBusy) return;
        experimentBusy = true;
        experimentRevision += 1;
        updateExperimentControls();
        error(name);
        feedback(name, "Stopping...");
        try {
          const result = await post("/api/characters/stop", {});
          renderCharacters(result);
          experimentRevision += 1;
          feedback(name, "Stop accepted.", true);
          await loadCharacters();
        } catch (failure) { error(name, failure.message); feedback(name, "Stop not confirmed."); }
        finally { experimentBusy = false; updateExperimentControls(); }
      });
    }

    const recorder = createRecorder({
      onState(state) {
        clearInterval(captureDisplayTimer);
        captureDisplayTimer = null;
        if (state.phase === "transcribing") { pointerId = null; keyboardHeld = false; }
        set("mic-state", ({ idle: "Idle", permission: "Permission pending", recording: "Recording", transcribing: "Transcribing", error: "Error" })[state.phase]);
        const active = ["permission", "recording", "transcribing"].includes(state.phase);
        get("mic-cancel").disabled = !active;
        get("mic-destination").disabled = active;
        get("mic-hold").disabled = state.phase === "transcribing";
        get("mic-hold").setAttribute("aria-pressed", String(state.phase === "recording"));
        set("mic-hold", state.phase === "recording" ? "Release to transcribe" : state.phase === "permission" ? "Permission pending" : "Hold to record");
        if (state.phase === "recording") {
          feedback("mic", "Recording");
          captureDisplayTimer = setInterval(() => {
            const seconds = recorder.elapsed();
            set("mic-duration", `${seconds.toFixed(1)} / 15 s`);
            get("mic-progress").value = seconds;
          }, 100);
        }
      },
      onError(failure) {
        const message = failure.name === "NotAllowedError" || failure.name === "SecurityError"
          ? "Microphone permission denied. Allow microphone access in browser settings, then hold to record again."
          : failure.name === "NotFoundError" ? "No microphone found. Connect a microphone and try again." : failure.message;
        error("microphone", message);
        feedback("mic", "Recording stopped.");
      },
      async onAudio(wav, token, seconds) {
        const destination = microphoneDestination;
        const sessionId = microphoneStorySessionId;
        set("mic-duration", `${seconds.toFixed(1)} / 15 s`);
        get("mic-progress").value = seconds;
        feedback("mic", "Transcribing...");
        transcriptionAbort = new AbortController();
        const controller = transcriptionAbort;
        try {
          const result = await request("/api/microphone/transcribe", { method: "POST", headers: { "Content-Type": "audio/wav", "X-Utterance-ID": freshId() }, body: wav, signal: controller.signal });
          if (recorder.state().token !== token) return;
          const text = String(result.text ?? "").trim();
          set("mic-transcript", text || "No speech detected.");
          set("mic-latency", formatMilliseconds(result.transcription_ms));
          const eventId = result.event_id || freshId();
          set("mic-event-id", `Event: ${eventId}`);
          if (text && destination === "story") {
            feedback("mic", "Transcript received. Sending to Story...");
            await submitStory(text, eventId, { signal: controller.signal, source: "microphone", sessionId });
            if (recorder.state().token !== token) return;
            feedback("mic", "Transcript added to Story.", true);
          } else feedback("mic", text ? "Transcribed." : "No speech detected.", Boolean(text));
          recorder.complete(token);
        } catch (failure) {
          if (recorder.state().token !== token || failure.name === "AbortError") return;
          error("microphone", failure.message);
          feedback("mic", "Request not confirmed. Transcript remains available.");
          recorder.fail(token);
        } finally { if (transcriptionAbort === controller) transcriptionAbort = null; }
      },
    });

    function cancelCapture() {
      const wasActive = ["permission", "recording", "transcribing"].includes(recorder.state().phase);
      transcriptionAbort?.abort();
      transcriptionAbort = null;
      recorder.cancel();
      if (wasActive) feedback("mic", "Cancelled.");
    }
    function pressCapture() {
      if (!["idle", "error"].includes(recorder.state().phase)) return;
      error("microphone");
      microphoneDestination = document.querySelector('input[name="destination"]:checked').value;
      // Bind at the explicit press, before permission, capture, or ASR can yield.
      microphoneStorySessionId = storySnapshot?.session_id ?? null;
      set("mic-duration", "0.0 / 15 s");
      get("mic-progress").value = 0;
      feedback("mic", "Waiting for microphone permission...");
      void recorder.press();
    }
    const hold = get("mic-hold");
    hold.addEventListener("pointerdown", (event) => {
      if (event.button !== 0 || pointerId != null || keyboardHeld) return;
      event.preventDefault();
      hold.focus();
      pointerId = event.pointerId;
      hold.setPointerCapture(event.pointerId);
      pressCapture();
    });
    hold.addEventListener("pointerup", (event) => {
      if (pointerId !== event.pointerId) return;
      pointerId = null;
      recorder.release();
    });
    hold.addEventListener("pointercancel", () => { pointerId = null; cancelCapture(); });
    hold.addEventListener("lostpointercapture", () => { if (pointerId != null) { pointerId = null; cancelCapture(); } });
    hold.addEventListener("contextmenu", (event) => event.preventDefault());
    hold.addEventListener("keydown", (event) => {
      if (![" ", "Enter"].includes(event.key)) return;
      event.preventDefault();
      if (event.repeat || keyboardHeld || pointerId != null) return;
      keyboardHeld = true;
      pressCapture();
    });
    hold.addEventListener("keyup", (event) => {
      if (![" ", "Enter"].includes(event.key) || !keyboardHeld) return;
      event.preventDefault();
      keyboardHeld = false;
      recorder.release();
    });
    hold.addEventListener("blur", () => {
      // Disabling the hold button after release must not abort transcription.
      if (["permission", "recording"].includes(recorder.state().phase)) cancelCapture();
      pointerId = null;
      keyboardHeld = false;
    });
    get("mic-cancel").addEventListener("click", cancelCapture);
    root.addEventListener("blur", cancelCapture);
    document.addEventListener("keydown", (event) => { if (event.key === "Escape") cancelCapture(); });

    async function refresh(name) {
      try {
        if (name === "story") { await loadPlayback(); await loadStory(); }
        else if (name === "microphone") {
          const data = await request("/api/microphone");
          set("mic-model", modelLabel(data));
          if (data.available === false) error("microphone", data.reason ?? data.error ?? "Transcription backend unavailable.");
          if (document.querySelector('input[name="destination"]:checked').value === "story") {
            try { await loadStory(); } catch (failure) { error("story", failure.message); }
          }
        } else await loadCharacters();
      } catch (failure) {
        if (!disposed && failure.name !== "AbortError") {
          error(name, failure.message);
          if (name === "story") set("story-state", "Unavailable / stale");
          if (["characters", "coordination"].includes(name)) set(`${name}-state`, "Unavailable / stale");
          if (name === "microphone") set("mic-model", "Unavailable / stale");
        }
      }
    }
    async function poll() {
      clearTimeout(pollTimer);
      if (disposed || document.hidden) return;
      if (pollBusy) { pollTimer = setTimeout(poll, 500); return; }
      pollBusy = true;
      await refresh(activeTab);
      pollBusy = false;
      if (!disposed) pollTimer = setTimeout(poll, 1500);
    }
    const tabs = Array.from(document.querySelectorAll("[data-tab]"));
    function selectTab(name) {
      if (name !== "microphone" && activeTab === "microphone") cancelCapture();
      activeTab = name;
      for (const tab of tabs) {
        const selected = tab.dataset.tab === name;
        tab.setAttribute("aria-selected", String(selected));
        tab.tabIndex = selected ? 0 : -1;
        get(`panel-${tab.dataset.tab}`).hidden = !selected;
      }
      void poll();
    }
    tabs.forEach((tab, index) => {
      tab.addEventListener("click", () => selectTab(tab.dataset.tab));
      tab.addEventListener("keydown", (event) => {
        let next;
        if (event.key === "ArrowRight") next = (index + 1) % tabs.length;
        if (event.key === "ArrowLeft") next = (index + tabs.length - 1) % tabs.length;
        if (event.key === "Home") next = 0;
        if (event.key === "End") next = tabs.length - 1;
        if (next == null) return;
        event.preventDefault();
        tabs[next].focus();
        selectTab(tabs[next].dataset.tab);
      });
    });
    document.querySelectorAll("[data-refresh]").forEach((button) => button.addEventListener("click", () => void poll()));
    document.addEventListener("visibilitychange", () => { if (document.hidden) { cancelCapture(); clearTimeout(pollTimer); } else void poll(); });
    root.addEventListener("pagehide", () => {
      disposed = true;
      cancelCapture();
      clearTimeout(pollTimer);
      for (const controller of requests) controller.abort();
    });
    root.addEventListener("pageshow", () => { if (disposed) { disposed = false; void poll(); } });
    void poll();
  }

  root.StreamMuseResearch = { encodeWav, reduceEvents, reduceCapture, storyBudget, createRecorder, modelLabel, formatMilliseconds, playbackView, mount };
  if (typeof document !== "undefined" && document.getElementById("research-root")) mount(document);
})(window);
