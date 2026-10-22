(function attachRapPlan(root) {
  "use strict";

  function makePayload(topic, templateId) {
    const payload = {};
    if (topic.trim()) payload.topic = topic.trim();
    if (templateId) payload.template_id = templateId;
    if (!Object.keys(payload).length) throw new Error("Enter a topic or choose a flow.");
    return payload; // Omit from_bar: the server resolves ASAP atomically with its playhead.
  }

  function editStatus(edit, bars, currentBar) {
    const target = edit.effective_from_bar ?? edit.from_bar;
    const committed = bars.filter((bar) => bar.frozen && bar.bar >= target).sort((a, b) => a.bar - b.bar);
    const matches = (bar) => (!edit.topic || bar.topic === edit.topic)
      && (!edit.template_id || bar.template_id === edit.template_id)
      && bar.choice !== "espeak_local_lyric" && bar.choice !== "silence";
    const played = committed.filter((bar) => currentBar !== null && bar.bar <= currentBar);
    const applied = played.find(matches);
    if (applied) return { state: "applied", text: `Requested settings played in bar ${applied.bar + 1}${applied.choice === "espeak_stream_lyric" ? " (eSpeak vocals)" : ""}.` };
    const ready = committed.find(matches);
    if (ready) return { state: "ready", text: `Updated audio queued for bar ${ready.bar + 1}.` };
    const missed = played.at(-1);
    if (missed) return { state: "delayed", text: `Bar ${missed.bar + 1} used ${missed.choice === "silence" ? "silence" : missed.choice === "espeak_local_lyric" ? "local fallback lyrics" : "an older setting"}. Change still pending.` };
    return { state: "pending", text: `Change requested from bar ${target + 1}. Awaiting updated audio.` };
  }

  function mount(document) {
    const get = (id) => document.getElementById(id);
    const form = get("plan-form");
    if (!form) return null;
    const topic = get("plan-topic"), select = get("plan-template"), button = get("plan-apply");
    const state = {
      info: null, edit: null, busy: false, connected: false, runtime: "stopped",
      bars: [], bar: null, epoch: null, session: null, error: null, errorKind: null,
      preview: null, loading: false, reloadPending: false, revision: 0,
      lastSequence: 0, eventFloor: -1,
    };
    const set = (id, text) => { if (get(id).textContent !== text) get(id).textContent = text; };

    function render() {
      const supported = state.info?.supported === true;
      const active = supported && state.connected && state.runtime === "running";
      topic.disabled = select.disabled = !active || state.busy;
      button.disabled = !active || state.busy || (!topic.value.trim() && !select.value);
      set("plan-mode", supported ? "ASAP" : "Unavailable");
      const frozen = state.bars.filter((bar) => bar.frozen).map((bar) => bar.bar + 1);
      const next = Math.max(state.info?.next_editable_bar ?? 0, ...frozen);
      set("plan-target", supported ? `Next editable: bar ${next + 1}` : "Live editing requires v3");
      set("plan-apply", state.busy ? "Sending..." : "Apply");
      let status = { state: "idle", text: state.info ? "No pending change." : "Loading editing controls..." };
      if (state.edit) status = editStatus(state.edit, state.bars, state.bar);
      if (!supported && state.info) status = { state: "idle", text: "This session does not support live edits." };
      else if (!state.connected) status = { state: "idle", text: "Waiting for connection." };
      else if (state.runtime !== "running") status = { state: "idle", text: state.runtime === "priming" ? "Preparing playback..." : "Start playback to edit upcoming bars." };
      if (state.error) status = { state: "error", text: state.error };
      get("plan-status").dataset.state = status.state;
      set("plan-status", status.text);
      const edit = state.edit;
      set("plan-request", edit ? `Requested: ${[edit.topic, edit.template_id && (state.info?.templates.find((item) => item.template_id === edit.template_id)?.name || edit.template_id)].filter(Boolean).join(" / ")}` : "");
    }

    function preview() {
      const flow = state.info?.templates.find((item) => item.template_id === select.value);
      const key = flow?.template_id || "";
      if (state.preview === key) return;
      state.preview = key;
      const track = get("plan-preview");
      track.replaceChildren();
      track.hidden = !flow;
      set("plan-flow-detail", flow ? `${flow.slots.length} syllables / ${flow.beats_per_bar} beats` : "Current flow unchanged");
      if (!flow) return;
      const ticks = flow.beats_per_bar * flow.ticks_per_beat;
      track.style.setProperty("--plan-ticks", ticks);
      track.setAttribute("aria-label", `${flow.name}: syllable timing and stress`);
      for (let tick = 0; tick < ticks; tick += 1) {
        const slot = flow.slots.find((item) => item.tick_in_bar === tick);
        const cell = document.createElement("span");
        cell.className = "plan-pulse";
        if (tick % flow.ticks_per_beat === 0) cell.classList.add("plan-beat");
        if (slot) {
          const fill = document.createElement("i");
          fill.style.height = `${25 + slot.target_stress * 75}%`;
          cell.append(fill);
        }
        cell.title = `Tick ${tick + 1}: ${slot ? `stress ${slot.target_stress}` : "rest"}`;
        track.append(cell);
      }
    }

    async function load() {
      if (state.loading) { state.reloadPending = true; return; }
      state.loading = true;
      const revision = state.revision;
      try {
        const response = await fetch("/api/control/plan", { cache: "no-store" });
        if (!response.ok) throw new Error(`Editing controls unavailable (${response.status}).`);
        const info = await response.json();
        if (revision !== state.revision) { state.reloadPending = true; return; }
        state.info = info;
        if (state.errorKind === "discovery") { state.error = null; state.errorKind = null; }
        if (info.last_edit && (!state.edit || info.last_edit.edit_seq >= state.edit.edit_seq)) state.edit = info.last_edit;
        const previous = select.value;
        select.replaceChildren();
        const unchanged = document.createElement("option");
        unchanged.value = ""; unchanged.textContent = "Keep current flow"; select.append(unchanged);
        for (const flow of info.templates || []) {
          const option = document.createElement("option");
          option.value = flow.template_id; option.textContent = `${flow.name} (${flow.slots.length})`; select.append(option);
        }
        select.value = previous;
        state.preview = null;
        preview();
      } catch (error) {
        if (revision === state.revision) { state.error = error.message; state.errorKind = "discovery"; }
      } finally {
        state.loading = false; render();
        if (state.reloadPending) { state.reloadPending = false; load(); }
      }
    }

    async function submit(event) {
      event.preventDefault();
      if (button.disabled || state.busy) return;
      let payload;
      try { payload = makePayload(topic.value, select.value); }
      catch (error) { state.error = error.message; state.errorKind = "submit"; render(); return; }
      state.busy = true; state.error = null;
      const revision = state.revision;
      render();
      try {
        const response = await fetch("/api/control/plan", {
          method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload),
        });
        const body = await response.json();
        if (!response.ok) throw new Error(typeof body.detail === "string" ? body.detail : `Edit rejected (${response.status}).`);
        if (revision !== state.revision) return;
        if (!state.edit || body.edit_seq >= state.edit.edit_seq) state.edit = { ...payload, ...body };
        topic.value = ""; select.value = ""; preview();
      } catch (error) { if (revision === state.revision) { state.error = error.message; state.errorKind = "submit"; } }
      finally { state.busy = false; render(); }
    }

    form.addEventListener("submit", submit);
    topic.addEventListener("input", () => { state.error = null; render(); });
    select.addEventListener("change", () => { state.error = null; preview(); render(); });
    return {
      load,
      connection(connected) { state.connected = connected; if (connected) load(); render(); },
      update(snapshot, events, current, bars) {
        const epoch = `${snapshot.session_id || ""}:${snapshot.coordinator_epoch ?? 0}`;
        if (state.epoch !== null && state.epoch !== epoch) {
          state.eventFloor = state.session === snapshot.session_id ? state.lastSequence : -1;
          state.lastSequence = 0;
          state.edit = null; state.error = null; state.revision += 1;
          topic.value = ""; select.value = ""; preview();
          if (state.connected) load();
        }
        state.epoch = epoch;
        state.session = snapshot.session_id;
        state.lastSequence = Math.max(state.lastSequence, snapshot.last_sequence || 0, ...events.map((event) => event.sequence || 0));
        state.bars = bars;
        state.bar = Number.isInteger(snapshot.audio?.current_bar) ? snapshot.audio.current_bar : current.bar;
        state.runtime = snapshot.audio?.state || "stopped";
        for (const event of events) {
          if (event.sequence <= state.eventFloor) continue;
          if (event.session_id && event.session_id !== snapshot.session_id) continue;
          if (event.payload?.coordinator_epoch !== undefined && event.payload.coordinator_epoch !== (snapshot.coordinator_epoch ?? 0)) continue;
          if (event.event_type === "plan_edited" && (!state.edit || event.payload.edit_seq > state.edit.edit_seq)) state.edit = event.payload;
        }
        render();
      },
      reset() {
        state.edit = null; state.error = null; state.revision += 1;
        state.eventFloor = state.lastSequence;
        topic.value = ""; select.value = ""; preview(); load(); render();
      },
    };
  }

  root.StreamMuseRapPlan = { makePayload, editStatus, mount };
})(typeof window !== "undefined" ? window : globalThis);
