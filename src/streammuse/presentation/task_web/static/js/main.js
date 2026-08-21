(() => {
  "use strict";

  const byId = (id) => document.getElementById(id);
  const elements = {
    task: byId("task-name"), connection: byId("connection-state"),
    connectionLabel: byId("connection-label"), turnLabel: byId("turn-label"),
    prompt: byId("prompt"), display: byId("display-value"),
    timeline: byId("timeline"), fill: byId("timeline-fill"),
    humanResponse: byId("human-response"), humanResult: byId("human-result"),
    llmResponse: byId("llm-response"), llmResult: byId("llm-result"),
    asr: byId("asr-line"), speech: byId("speech-line"),
    mode: byId("mode-status"), stats: byId("stats-status"),
    session: byId("session-status"),
  };

  let animationFrame = null;
  let reconnectAttempt = 0;
  let currentSession = null;
  let sessionFinished = false;

  function setConnection(label, className) {
    elements.connectionLabel.textContent = label;
    elements.connection.className = `connection ${className}`;
  }

  function stopTimeline() {
    if (animationFrame !== null) cancelAnimationFrame(animationFrame);
    animationFrame = null;
  }

  function startTimeline(turn) {
    stopTimeline();
    const deadline = Number(turn.deadline_ms || 0);
    const initialElapsed = Number(turn.elapsed_ms_at_snapshot || 0);
    if (!turn.active || deadline <= 0) {
      elements.timeline.className = "timeline idle";
      elements.fill.style.transform = "scaleX(0)";
      return;
    }
    const started = performance.now() - initialElapsed;
    const tick = (now) => {
      const elapsed = Math.max(0, now - started);
      if (elapsed <= deadline) {
        elements.timeline.className = "timeline";
        elements.fill.style.transform = `scaleX(${Math.max(0, 1 - elapsed / deadline)})`;
      } else {
        elements.timeline.className = "timeline overrun";
        elements.fill.style.transform = `scaleX(${Math.min(1, (elapsed - deadline) / deadline)})`;
      }
      animationFrame = requestAnimationFrame(tick);
    };
    animationFrame = requestAnimationFrame(tick);
  }

  function renderResponse(actor, response) {
    const text = actor === "human" ? elements.humanResponse : elements.llmResponse;
    const result = actor === "human" ? elements.humanResult : elements.llmResult;
    if (!response) return;
    text.textContent = response.response || "[no response]";
    result.textContent = response.is_valid ? "OK" : "MISS";
    result.className = `result ${response.is_valid ? "ok" : "miss"}`;
  }

  function render(snapshot) {
    const state = snapshot.payload || {};
    currentSession = snapshot.session_id;
    sessionFinished = state.status === "finished";
    elements.task.textContent = state.task || "Interactive task";
    const turn = state.current_turn || {};
    if (turn.actor) {
      elements.turnLabel.textContent = `${turn.actor === "human" ? "Your" : "LLM"} turn`;
      elements.prompt.textContent = turn.prompt || "Thinking";
    } else {
      elements.turnLabel.textContent = state.status === "initializing" ? "Initializing game" : "Waiting for game";
      elements.prompt.textContent = state.status === "initializing" ? "Preparing audio and model resources" : "Viewer is ready";
    }
    const hasDisplay = turn.display_value !== null && turn.display_value !== undefined;
    elements.display.hidden = !hasDisplay;
    elements.display.textContent = hasDisplay ? String(turn.display_value) : "";
    startTimeline(turn);

    const responses = state.responses || {};
    renderResponse("human", responses.human);
    renderResponse("llm", responses.llm);
    const asr = state.asr;
    elements.asr.hidden = !asr;
    elements.asr.textContent = asr ? `ASR: ${asr.raw_transcript || "[no speech]"}` : "";
    const speech = state.speech_output;
    elements.speech.hidden = !speech;
    elements.speech.textContent = speech ? `Speech output: ${speech.status || "active"}` : "";

    const config = state.config || {};
    const stage = state.stage || {};
    const stageText = stage.stage_index === null || stage.stage_index === undefined ? "" : ` · stage ${Number(stage.stage_index) + 1}`;
    elements.mode.textContent = `${config.deadline_mode || "waiting"}${stageText}`;
    const stats = state.stats || {};
    elements.stats.textContent = `${stats.turn_count || 0} turns · ${stats.valid_count || 0} ok · ${stats.deadline_miss_count || 0} deadline misses`;
    elements.session.textContent = state.status === "finished" ? `Finished: ${(state.session_result || {}).stop_reason || "completed"}` : state.status || "waiting";
  }

  function connect() {
    const token = new URLSearchParams(window.location.search).get("token") || "";
    const scheme = window.location.protocol === "https:" ? "wss" : "ws";
    const socket = new WebSocket(`${scheme}://${window.location.host}/ws?token=${encodeURIComponent(token)}`);
    setConnection("Connecting", "");
    socket.addEventListener("open", () => { reconnectAttempt = 0; setConnection("Connected", "connected"); });
    socket.addEventListener("message", (event) => {
      let message;
      try { message = JSON.parse(event.data); } catch (_) { return; }
      if (message.type !== "snapshot") return;
      render(message);
      socket.send(JSON.stringify({ type: "viewer_ready", session_id: message.session_id }));
    });
    socket.addEventListener("close", () => {
      if (sessionFinished) {
        setConnection("Session complete", "");
        stopTimeline();
        return;
      }
      setConnection("Reconnecting", "disconnected");
      stopTimeline();
      const delay = Math.min(5000, 250 * (2 ** reconnectAttempt));
      reconnectAttempt += 1;
      window.setTimeout(connect, delay);
    });
  }

  connect();
})();
