// StreamMUSE web UI client — minimal vanilla JS.

const $ = (id) => document.getElementById(id);

const statusPill = $("status-pill");
const canvas = $("roll");
const ctx = canvas.getContext("2d");

const mTick = $("m-tick");
const mBar = $("m-bar");
const mBeat = $("m-beat");
const sRtt = $("s-rtt");
const sSrv = $("s-srv");
const sNet = $("s-net");
const sHits = $("s-hits");
const sTicks = $("s-ticks");

const keySelect = $("pk-key");
const promptSelect = $("pk-prompt");
const strategySelect = $("pk-strategy");
const injectBtn = $("pk-inject");
const pickerMsg = $("pk-msg");

// State
let currentTick = 0;
let ticksPerBeat = 4;
let beatsPerBar = 4;
let windowBeats = 4;
const activeNotes = new Map(); // key `${source}:${pitch}` -> { pitch, startTick, source }
const finishedNotes = []; // {pitch, startTick, endTick, source}

// Config comes through WS once; default fallbacks above.
function onConfig(cfg) {
  if (cfg.ticks_per_beat) ticksPerBeat = cfg.ticks_per_beat;
  if (cfg.beats_per_bar) beatsPerBar = cfg.beats_per_bar;
}

function setConnection(connected) {
  statusPill.classList.toggle("connected", connected);
  statusPill.classList.toggle("disconnected", !connected);
  statusPill.textContent = connected ? "connected" : "disconnected";
}

function onNote(msg) {
  const key = `${msg.source}:${msg.pitch}`;
  if (msg.event === "on") {
    activeNotes.set(key, { pitch: msg.pitch, startTick: msg.tick, source: msg.source });
  } else if (msg.event === "off") {
    const open = activeNotes.get(key);
    if (open) {
      finishedNotes.push({
        pitch: open.pitch,
        startTick: open.startTick,
        endTick: msg.tick,
        source: open.source,
      });
      activeNotes.delete(key);
    }
  }
  purgeOld();
}

function onTick(msg) {
  currentTick = msg.tick;
  mTick.textContent = msg.tick;
  mBar.textContent = msg.bar;
  mBeat.textContent = msg.beat;
  purgeOld();
}

function purgeOld() {
  const windowTicks = windowBeats * ticksPerBeat;
  const cutoff = currentTick - windowTicks * 2;
  while (finishedNotes.length && finishedNotes[0].endTick < cutoff) {
    finishedNotes.shift();
  }
}

function onStats(msg) {
  if (msg.round_trip_ms != null) sRtt.textContent = msg.round_trip_ms.toFixed(1);
  if (msg.server_process_ms != null) sSrv.textContent = msg.server_process_ms.toFixed(1);
  if (msg.network_latency_ms != null) sNet.textContent = msg.network_latency_ms.toFixed(1);
  if (msg.total_hits != null) sHits.textContent = msg.total_hits;
  if (msg.total_ticks != null) sTicks.textContent = msg.total_ticks;
}

// --- WS ---
function connect() {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  const ws = new WebSocket(`${proto}//${location.host}/ws`);
  ws.onopen = () => setConnection(true);
  ws.onclose = () => {
    setConnection(false);
    setTimeout(connect, 1000);
  };
  ws.onerror = () => setConnection(false);
  ws.onmessage = (ev) => {
    let msg;
    try {
      msg = JSON.parse(ev.data);
    } catch {
      return;
    }
    if (msg.type === "note") onNote(msg);
    else if (msg.type === "tick") onTick(msg);
    else if (msg.type === "stats") onStats(msg);
    else if (msg.type === "config") onConfig(msg);
  };
}

// --- Rendering ---
function drawRoll() {
  const w = canvas.width;
  const h = canvas.height;
  ctx.fillStyle = "#1f2327";
  ctx.fillRect(0, 0, w, h);

  const windowTicks = windowBeats * ticksPerBeat;
  const startTick = currentTick - windowTicks + 1;
  const tickW = w / windowTicks;

  // Gridlines — one per beat
  ctx.strokeStyle = "#2e333a";
  ctx.lineWidth = 1;
  for (let b = 0; b <= windowBeats; b++) {
    const x = b * ticksPerBeat * tickW;
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, h);
    ctx.stroke();
  }

  // Pitch range: focus on C3 (48) to C6 (84) — 36 semitones, adjust vertically
  const pitchLow = 36;
  const pitchHigh = 96;
  const pitchRange = pitchHigh - pitchLow;
  const pitchH = h / pitchRange;

  function drawNote(n) {
    if (n.pitch < pitchLow || n.pitch > pitchHigh) return;
    const nStart = Math.max(n.startTick, startTick);
    const nEnd = (n.endTick != null) ? n.endTick : currentTick;
    if (nEnd < startTick) return;
    const x1 = (nStart - startTick) * tickW;
    const x2 = (nEnd - startTick) * tickW;
    const y = h - (n.pitch - pitchLow) * pitchH - pitchH;
    ctx.fillStyle = n.source === "user" ? "#7bd88f" : "#ff9466";
    ctx.fillRect(x1, y, Math.max(2, x2 - x1), Math.max(2, pitchH - 1));
  }

  for (const n of finishedNotes) drawNote(n);
  for (const n of activeNotes.values()) drawNote({ ...n, endTick: null });

  // Play-head
  ctx.strokeStyle = "#4f8cff";
  ctx.lineWidth = 2;
  const playX = (currentTick - startTick + 1) * tickW;
  ctx.beginPath();
  ctx.moveTo(playX, 0);
  ctx.lineTo(playX, h);
  ctx.stroke();

  requestAnimationFrame(drawRoll);
}

// --- Prompt picker ---
async function loadPrompts() {
  try {
    const resp = await fetch("/api/prompts");
    if (!resp.ok) throw new Error(await resp.text());
    const data = await resp.json();
    const keys = Object.keys(data.keys || {}).sort();

    keySelect.innerHTML = "";
    for (const k of keys) {
      const opt = document.createElement("option");
      opt.value = k;
      opt.textContent = `${k} (${(data.keys[k] || []).length})`;
      keySelect.appendChild(opt);
    }
    updatePrompts(data.keys || {});

    keySelect.addEventListener("change", () => updatePrompts(data.keys || {}));
  } catch (e) {
    pickerMsg.textContent = `failed to load prompts: ${e.message || e}`;
    pickerMsg.className = "msg err";
  }
}

function updatePrompts(keys) {
  const key = keySelect.value;
  const list = keys[key] || [];
  promptSelect.innerHTML = "";
  // first option = strategy
  const autoOpt = document.createElement("option");
  autoOpt.value = "";
  autoOpt.textContent = "(use strategy)";
  promptSelect.appendChild(autoOpt);
  list.forEach((p, i) => {
    const opt = document.createElement("option");
    opt.value = String(i);
    opt.textContent = p.name || `#${i}`;
    promptSelect.appendChild(opt);
  });
}

async function doInject() {
  injectBtn.disabled = true;
  pickerMsg.className = "msg";
  pickerMsg.textContent = "injecting…";
  const key = keySelect.value;
  let strategy = strategySelect.value;
  const chosen = promptSelect.value;
  if (chosen !== "") strategy = parseInt(chosen, 10);
  try {
    const resp = await fetch("/api/inject", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ key, strategy }),
    });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.detail || "inject failed");
    pickerMsg.textContent = `${data.message} (mel ${data.injected_melody}, acc ${data.injected_acc})`;
    pickerMsg.className = "msg ok";
  } catch (e) {
    pickerMsg.textContent = `inject failed: ${e.message || e}`;
    pickerMsg.className = "msg err";
  } finally {
    injectBtn.disabled = false;
  }
}

injectBtn.addEventListener("click", doInject);

connect();
loadPrompts();
requestAnimationFrame(drawRoll);
