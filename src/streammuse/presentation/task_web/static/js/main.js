(() => {
  "use strict";

  const byId = (id) => document.getElementById(id);
  const elements = {
    task: byId("task-name"), connection: byId("connection-state"),
    connectionLabel: byId("connection-label"), turnLabel: byId("turn-label"),
    setupPanel: byId("setup-panel"), setupTitle: byId("setup-title"),
    setupPrompt: byId("setup-prompt"), candidateList: byId("candidate-list"),
    playfield: byId("playfield"), roundProgress: byId("round-progress"),
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
    text.textContent = response.display_text || response.response || "[no response]";
    result.textContent = response.is_valid ? "OK" : "MISS";
    result.className = `result ${response.is_valid ? "ok" : "miss"}`;
  }

  function render(snapshot) {
    const state = snapshot.payload || {};
    const setup = state.setup || {};
    const progress = state.progress || {};
    const publicState = progress.public_state || {};
    currentSession = snapshot.session_id;
    sessionFinished = state.status === "finished";
    elements.task.textContent = publicState.scope_title || state.task || "Interactive task";

    const showingSetup = state.status === "setup";
    elements.setupPanel.hidden = !showingSetup;
    elements.playfield.hidden = showingSetup;
    if (showingSetup) {
      elements.setupTitle.textContent = setup.title || publicState.scope_title || "Character scope";
      elements.setupPrompt.textContent = setup.prompt || "Choose one person privately.";
      elements.candidateList.replaceChildren();
      const choices = Array.isArray(setup.choices) ? setup.choices : [];
      choices.forEach((choice) => {
        const item = document.createElement("li");
        item.textContent = String(choice);
        elements.candidateList.appendChild(item);
      });
    }

    const turn = state.current_turn || {};
    if (turn.actor) {
      const pending = publicState.pending_action || {};
      if (turn.actor === "human" && pending.kind === "guess") {
        elements.turnLabel.textContent = "Confirm the guess";
      } else if (turn.actor === "human") {
        elements.turnLabel.textContent = "Answer the question";
      } else {
        elements.turnLabel.textContent = "LLM asking or guessing";
      }
      elements.prompt.textContent = turn.prompt || "Thinking";
    } else {
      elements.turnLabel.textContent = state.status === "initializing" ? "Initializing game" : "Waiting for game";
      elements.prompt.textContent = state.status === "initializing" ? "Preparing audio and model resources" : "Viewer is ready";
    }
    const maxRounds = Number(publicState.max_rounds || 0);
    const currentRound = Number(
      turn.logical_round || publicState.current_round || 0
    );
    elements.roundProgress.hidden = maxRounds <= 0;
    elements.roundProgress.textContent = maxRounds > 0
      ? `Round ${currentRound} / ${maxRounds}`
      : "";
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
    const roundStats = Number(stats.logical_round_count || 0);
    const rounds = maxRounds > 0 ? `${roundStats} rounds · ` : "";
    elements.stats.textContent = `${rounds}${stats.turn_count || 0} turns · ${stats.valid_count || 0} ok · ${stats.deadline_miss_count || 0} deadline misses`;
    const result = state.session_result || {};
    if (state.status === "finished" && result.successful_character_id) {
      const name = publicState.successful_character_name
        || result.successful_character_id;
      elements.session.textContent = `${name} guessed in round ${result.successful_guess_round}`;
    } else {
      elements.session.textContent = state.status === "finished"
        ? `Finished: ${result.stop_reason || "completed"}`
        : state.status || "waiting";
    }
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
