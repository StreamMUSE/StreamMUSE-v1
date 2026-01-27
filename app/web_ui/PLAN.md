# StreamMUSE Web UI Implementation Plan

## Overview

A web-based demo UI for the StreamMUSE real-time accompaniment system featuring:
- Falling notes piano visualization
- Hyperparameter controls
- Start/restart controls
- Real-time stats monitoring
- Visual indicators for backup levels, late notes, etc.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Browser (Web UI)                        │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │ Piano Visualizer│  │  Control Panel  │  │ Stats Panel │ │
│  │ (Canvas/JS)     │  │  (HTML/JS)      │  │ (HTML/JS)   │ │
│  └────────┬────────┘  └────────┬────────┘  └──────┬──────┘ │
│           │                    │                   │        │
│           └────────────────────┼───────────────────┘        │
│                                │ WebSocket                  │
└────────────────────────────────┼────────────────────────────┘
                                 │
┌────────────────────────────────┼────────────────────────────┐
│              Python Backend (FastAPI + WebSocket)           │
│  ┌─────────────────┐  ┌────────┴────────┐  ┌─────────────┐ │
│  │ WebSocket       │  │ Existing Client │  │ Web Server  │ │
│  │ Output Handler  │◄─┤ Logic (reused)  │  │ (static)    │ │
│  └─────────────────┘  └────────┬────────┘  └─────────────┘ │
│                                │ HTTP                       │
└────────────────────────────────┼────────────────────────────┘
                                 │
┌────────────────────────────────┼────────────────────────────┐
│                    Fake Server (FastAPI)                    │
│  ┌─────────────────────────────┴──────────────────────────┐ │
│  │ Echo melody back as accompaniment (octave down)        │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## File Structure

```
app/
├── client.py                    # (existing, minor modifications)
├── server.py                    # (existing, unchanged)
├── fake_server.py               # NEW: Echo-based mock server
├── web_client.py                # NEW: FastAPI server for web UI
├── output_handlers/
│   ├── audio_output.py          # (existing)
│   ├── websocket_output.py      # NEW: Sends events to browser
│   └── ...
└── web_ui/                      # NEW: Static web files
    ├── index.html
    ├── css/
    │   └── style.css
    └── js/
        ├── main.js              # Entry point, WebSocket handling
        ├── piano.js             # Piano keyboard + falling notes
        ├── controls.js          # Hyperparameter controls
        └── stats.js             # Stats dashboard
```

---

## Implementation Phases

### Phase 1: Fake Server + Basic Infrastructure
- [x] Create `fake_server.py` with echo logic
- [x] Create `websocket_output.py` handler
- [x] Create `web_client.py` with WebSocket + static serving
- [x] Basic HTML structure
- [x] Create `midi_utils_client.py` for client-side MIDI parsing

### Phase 2: Piano Visualization
- [x] Piano keyboard rendering (Canvas)
- [x] Falling notes animation
- [x] Color coding by note type/status
- [x] Playhead line

### Phase 3: Controls + Integration  
- [x] Start/stop/restart buttons
- [x] Hyperparameter controls
- [x] Input mode switching
- [x] Connect to existing client logic

### Phase 4: Stats + Polish
- [x] Stats dashboard
- [x] Backup/late visualization
- [ ] Note swap animations (future enhancement)

---

## Technical Details

### 1. Fake Server (`fake_server.py`)

**Purpose**: Mock server for testing without ML dependencies.

**Endpoints** (mirror real server):
- `POST /generate_accompaniment` - Echo melody as accompaniment
- `POST /inject_notes` - No-op, return success
- `POST /inject_music` - No-op, return success  
- `POST /clear_history` - No-op, return success
- `GET /injection_status` - Return default status

**Echo Logic**:
```python
def generate_echo_accompaniment(melody_notes, generation_start_tick):
    accompaniment = []
    for note in melody_notes:
        # Transpose down 1 octave, offset by 1 tick
        acc_note = {
            "pitch": max(21, note["pitch"] - 12),  # Down octave, min A0
            "tick": note["tick"] + 1,              # Slight delay
            "duration": note["duration"],
            "program": 1
        }
        accompaniment.append(acc_note)
    
    # Also generate some notes at generation_start_tick if no melody
    if not accompaniment:
        # Generate a simple pattern based on tick
        base_pitch = 48 + (generation_start_tick % 12)
        accompaniment.append({
            "pitch": base_pitch,
            "tick": generation_start_tick,
            "duration": 4,
            "program": 1
        })
    
    return accompaniment
```

**Request/Response Models**: Reuse from `server.py` (copy the Pydantic models).

---

### 2. WebSocket Output Handler (`websocket_output.py`)

**Purpose**: Send real-time events to browser for visualization.

**Message Types**:
```python
# Tick update (sent every tick)
{
    "type": "tick",
    "tick": 42,
    "bar": 2,
    "beat": 2,
    "timestamp": 1234567890.123
}

# Note event
{
    "type": "note",
    "event": "on" | "off",
    "pitch": 60,
    "velocity": 100,
    "tick": 42,
    "duration": 4,           # only for note_on
    "source": "user" | "model",
    "backup_level": 0,       # only for model notes
    "is_late": false,
    "was_swapped": false
}

# Stats update (sent periodically or on change)
{
    "type": "stats",
    "hit_rate": 0.95,
    "avg_backup_level": 0.3,
    "round_trip_ms": 45.2,
    "server_process_ms": 32.1,
    "network_latency_ms": 13.1
}

# Config update (sent on change)
{
    "type": "config",
    "tempo": 90,
    "ticks_per_beat": 4,
    "beats_per_bar": 4,
    "generation_interval_ticks": 1,
    "accompaniment_velocity": 50
}

# Status update
{
    "type": "status",
    "state": "stopped" | "running" | "listening",
    "message": "Ready to start"
}
```

**Class Interface**:
```python
class WebSocketOutputHandler:
    def __init__(self, websocket_manager):
        self.manager = websocket_manager
    
    async def send_tick(self, tick, bar, beat):
        ...
    
    async def send_note_on(self, pitch, velocity, tick, duration, source, backup_level=0):
        ...
    
    async def send_note_off(self, pitch, tick, source):
        ...
    
    async def send_stats(self, stats_dict):
        ...
    
    async def send_config(self, config_dict):
        ...
    
    async def send_status(self, state, message):
        ...
```

**Threading Consideration**: 
- The existing client runs in threads (tick_loop, input_thread, inference_thread)
- WebSocket is async (asyncio)
- Solution: Use `asyncio.run_coroutine_threadsafe()` to bridge sync→async
- Or use a thread-safe queue that the async WebSocket handler polls

**Queue-based approach** (simpler):
```python
import queue
import json

class WebSocketOutputHandler:
    def __init__(self):
        self.message_queue = queue.Queue()
    
    def send_tick(self, tick, bar, beat):
        self.message_queue.put(json.dumps({
            "type": "tick",
            "tick": tick,
            "bar": bar,
            "beat": beat
        }))
    
    # ... other methods similar
    
    def get_pending_messages(self):
        """Called by async WebSocket handler to get messages to send"""
        messages = []
        while not self.message_queue.empty():
            try:
                messages.append(self.message_queue.get_nowait())
            except queue.Empty:
                break
        return messages
```

---

### 3. Web Client Server (`web_client.py`)

**Purpose**: Serve web UI and manage client lifecycle.

**Tech Stack**:
- FastAPI for HTTP + WebSocket
- uvicorn as ASGI server
- Static file serving for HTML/CSS/JS

**Endpoints**:
```python
# Static files
GET /                    # Serve index.html
GET /css/*              # Serve CSS files
GET /js/*               # Serve JS files

# WebSocket
WS /ws                  # Real-time communication

# REST API for control
POST /api/start         # Start the client
POST /api/stop          # Stop the client
POST /api/restart       # Restart the client
GET  /api/config        # Get current config
POST /api/config        # Update config
GET  /api/status        # Get current status
POST /api/input-mode    # Set input mode (midi/keyboard/file)
```

**Client Management**:
```python
class ClientManager:
    def __init__(self):
        self.client_thread = None
        self.is_running = False
        self.config = {
            "tempo": 90,
            "ticks_per_beat": 4,
            "beats_per_bar": 4,
            "generation_interval_ticks": 1,
            "accompaniment_velocity": 50,
            "server_url": "http://localhost:8001/generate_accompaniment",
            "input_mode": "keyboard",  # keyboard, midi, file
            "midi_file_path": None
        }
        self.ws_output_handler = WebSocketOutputHandler()
    
    def start(self):
        # Create and start client thread with current config
        ...
    
    def stop(self):
        # Signal client to stop
        ...
    
    def restart(self):
        self.stop()
        self.start()
    
    def update_config(self, new_config):
        self.config.update(new_config)
        # If running, may need restart
```

**WebSocket Handler**:
```python
from fastapi import WebSocket, WebSocketDisconnect

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

# Polling loop to send messages from queue
async def ws_sender_loop(manager: ConnectionManager, ws_handler: WebSocketOutputHandler):
    while True:
        messages = ws_handler.get_pending_messages()
        for msg in messages:
            await manager.broadcast(msg)
        await asyncio.sleep(0.01)  # 10ms polling interval
```

---

### 4. Piano Visualizer (`piano.js`)

**Canvas Setup**:
```javascript
const canvas = document.getElementById('piano-canvas');
const ctx = canvas.getContext('2d');

// Dimensions
const PIANO_HEIGHT = 120;        // Height of keyboard at bottom
const WHITE_KEY_WIDTH = 24;
const BLACK_KEY_WIDTH = 14;
const BLACK_KEY_HEIGHT = 80;

// Piano range: A0 (21) to C8 (108) = 88 keys
const MIN_PITCH = 21;
const MAX_PITCH = 108;
const NUM_WHITE_KEYS = 52;

// Visible range (can be subset for zooming)
let visibleMinPitch = 36;  // C2
let visibleMaxPitch = 96;  // C7

// Timing
let pixelsPerTick = 50;           // Adjustable fall speed
let visibleTicksAhead = 32;       // Future ticks shown
let visibleTicksBehind = 16;      // Past ticks shown (faded)
let currentTick = 0;
```

**Note Storage**:
```javascript
// Notes currently being displayed
const activeNotes = new Map();  // pitch -> {startTick, duration, source, status, backupLevel, ...}
const noteHistory = [];         // All notes for rendering

// Add note from WebSocket message
function addNote(noteData) {
    const note = {
        id: `${noteData.pitch}-${noteData.tick}`,
        pitch: noteData.pitch,
        startTick: noteData.tick,
        duration: noteData.duration,
        source: noteData.source,        // 'user' or 'model'
        backupLevel: noteData.backup_level || 0,
        isLate: noteData.is_late || false,
        wasSwapped: noteData.was_swapped || false,
        status: 'scheduled'  // scheduled -> playing -> played
    };
    noteHistory.push(note);
    activeNotes.set(note.id, note);
}
```

**Color Scheme**:
```javascript
const COLORS = {
    user: '#4A90D9',           // Blue - user melody
    model_0: '#4CAF50',        // Green - on-time (backup=0)
    model_1: '#FFC107',        // Yellow - backup=1
    model_2: '#FF9800',        // Orange - backup=2
    model_3: '#F44336',        // Red - backup=3+
    late: '#9E9E9E',           // Gray - late/missed
    swapped: 'rgba(158,158,158,0.5)',  // Faded - swapped out
    
    whiteKey: '#FFFFFF',
    whiteKeyPressed: '#E3F2FD',
    blackKey: '#333333',
    blackKeyPressed: '#1565C0',
    playhead: '#FF5722'
};

function getNoteColor(note) {
    if (note.wasSwapped) return COLORS.swapped;
    if (note.isLate) return COLORS.late;
    if (note.source === 'user') return COLORS.user;
    
    // Model note - color by backup level
    const level = Math.min(note.backupLevel, 3);
    return COLORS[`model_${level}`];
}
```

**Piano Key Rendering**:
```javascript
function isBlackKey(pitch) {
    const note = pitch % 12;
    return [1, 3, 6, 8, 10].includes(note);
}

function getPitchX(pitch) {
    // Calculate x position for a given pitch
    // Count white keys from minPitch to this pitch
    let whiteKeyCount = 0;
    for (let p = visibleMinPitch; p < pitch; p++) {
        if (!isBlackKey(p)) whiteKeyCount++;
    }
    
    if (isBlackKey(pitch)) {
        // Black key is offset from the white key before it
        return whiteKeyCount * WHITE_KEY_WIDTH - BLACK_KEY_WIDTH / 2;
    }
    return whiteKeyCount * WHITE_KEY_WIDTH;
}

function drawPianoKeys() {
    const y = canvas.height - PIANO_HEIGHT;
    
    // Draw white keys first
    let x = 0;
    for (let pitch = visibleMinPitch; pitch <= visibleMaxPitch; pitch++) {
        if (!isBlackKey(pitch)) {
            const isPressed = pressedKeys.has(pitch);
            ctx.fillStyle = isPressed ? COLORS.whiteKeyPressed : COLORS.whiteKey;
            ctx.fillRect(x, y, WHITE_KEY_WIDTH - 1, PIANO_HEIGHT);
            ctx.strokeStyle = '#000';
            ctx.strokeRect(x, y, WHITE_KEY_WIDTH - 1, PIANO_HEIGHT);
            x += WHITE_KEY_WIDTH;
        }
    }
    
    // Draw black keys on top
    x = 0;
    for (let pitch = visibleMinPitch; pitch <= visibleMaxPitch; pitch++) {
        if (!isBlackKey(pitch)) {
            x += WHITE_KEY_WIDTH;
        } else {
            const isPressed = pressedKeys.has(pitch);
            ctx.fillStyle = isPressed ? COLORS.blackKeyPressed : COLORS.blackKey;
            ctx.fillRect(x - BLACK_KEY_WIDTH/2 - WHITE_KEY_WIDTH, y, BLACK_KEY_WIDTH, BLACK_KEY_HEIGHT);
        }
    }
}
```

**Falling Notes Rendering**:
```javascript
function tickToY(tick) {
    // Convert tick to y position
    // Current tick is at playhead (just above piano)
    // Future ticks are higher (smaller y)
    // Past ticks are lower (larger y, below playhead)
    const playheadY = canvas.height - PIANO_HEIGHT - 20;
    const tickDiff = tick - currentTick;
    return playheadY - (tickDiff * pixelsPerTick);
}

function drawNotes() {
    const playheadY = canvas.height - PIANO_HEIGHT - 20;
    
    for (const note of noteHistory) {
        const startY = tickToY(note.startTick);
        const endY = tickToY(note.startTick + note.duration);
        
        // Skip if completely out of view
        if (endY < 0 || startY > canvas.height) continue;
        
        const x = getPitchX(note.pitch);
        const width = isBlackKey(note.pitch) ? BLACK_KEY_WIDTH : WHITE_KEY_WIDTH - 2;
        const height = startY - endY;
        
        // Get color based on note properties
        ctx.fillStyle = getNoteColor(note);
        
        // Draw note rectangle
        ctx.fillRect(x, endY, width, height);
        
        // Border
        ctx.strokeStyle = note.wasSwapped ? '#666' : '#000';
        ctx.setLineDash(note.wasSwapped ? [5, 5] : []);
        ctx.strokeRect(x, endY, width, height);
        ctx.setLineDash([]);
        
        // Fade past notes
        if (note.startTick + note.duration < currentTick) {
            ctx.fillStyle = 'rgba(255,255,255,0.5)';
            ctx.fillRect(x, endY, width, height);
        }
    }
    
    // Draw playhead line
    ctx.strokeStyle = COLORS.playhead;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(0, playheadY);
    ctx.lineTo(canvas.width, playheadY);
    ctx.stroke();
    ctx.lineWidth = 1;
}
```

**Animation Loop**:
```javascript
function render() {
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Draw background grid (optional)
    drawGrid();
    
    // Draw falling notes
    drawNotes();
    
    // Draw piano keys
    drawPianoKeys();
    
    // Draw beat/bar markers
    drawBeatMarkers();
    
    // Request next frame
    requestAnimationFrame(render);
}

// Start rendering
render();
```

**Cleanup Old Notes**:
```javascript
function cleanupOldNotes() {
    // Remove notes that are too far in the past
    const cutoffTick = currentTick - visibleTicksBehind - 10;
    while (noteHistory.length > 0 && noteHistory[0].startTick + noteHistory[0].duration < cutoffTick) {
        const note = noteHistory.shift();
        activeNotes.delete(note.id);
    }
}
```

---

### 5. Controls (`controls.js`)

**HTML Structure**:
```html
<div id="controls">
    <div class="control-group">
        <button id="btn-start">Start</button>
        <button id="btn-stop" disabled>Stop</button>
        <button id="btn-restart" disabled>Restart</button>
    </div>
    
    <div class="control-group">
        <label>Input Mode:</label>
        <select id="input-mode">
            <option value="keyboard">Computer Keyboard</option>
            <option value="midi">MIDI Device</option>
            <option value="file">MIDI File</option>
        </select>
        <input type="file" id="midi-file" accept=".mid,.midi" style="display:none">
    </div>
    
    <div class="control-group">
        <label>Tempo (BPM):</label>
        <input type="range" id="tempo" min="40" max="200" value="90">
        <span id="tempo-value">90</span>
    </div>
    
    <div class="control-group">
        <label>Ticks per Beat:</label>
        <select id="ticks-per-beat">
            <option value="2">2</option>
            <option value="4" selected>4</option>
            <option value="8">8</option>
        </select>
    </div>
    
    <div class="control-group">
        <label>Beats per Bar:</label>
        <select id="beats-per-bar">
            <option value="3">3</option>
            <option value="4" selected>4</option>
            <option value="6">6</option>
        </select>
    </div>
    
    <div class="control-group">
        <label>Generation Interval:</label>
        <input type="range" id="gen-interval" min="1" max="8" value="1">
        <span id="gen-interval-value">1 tick</span>
    </div>
    
    <div class="control-group">
        <label>Accompaniment Velocity:</label>
        <input type="range" id="acc-velocity" min="0" max="127" value="50">
        <span id="acc-velocity-value">50</span>
    </div>
    
    <hr>
    <h4>Visual Settings</h4>
    
    <div class="control-group">
        <label>Fall Speed:</label>
        <input type="range" id="fall-speed" min="20" max="100" value="50">
        <span id="fall-speed-value">50 px/tick</span>
    </div>
    
    <div class="control-group">
        <label>Visible Range (ticks):</label>
        <input type="range" id="visible-ahead" min="8" max="64" value="32">
        <span id="visible-ahead-value">32 ahead</span>
    </div>
</div>
```

**JavaScript**:
```javascript
// Control elements
const btnStart = document.getElementById('btn-start');
const btnStop = document.getElementById('btn-stop');
const btnRestart = document.getElementById('btn-restart');

// State
let isRunning = false;

// API calls
async function startClient() {
    const config = gatherConfig();
    const response = await fetch('/api/start', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(config)
    });
    if (response.ok) {
        isRunning = true;
        updateButtonStates();
    }
}

async function stopClient() {
    const response = await fetch('/api/stop', {method: 'POST'});
    if (response.ok) {
        isRunning = false;
        updateButtonStates();
    }
}

async function restartClient() {
    await stopClient();
    await startClient();
}

function gatherConfig() {
    return {
        tempo: parseInt(document.getElementById('tempo').value),
        ticks_per_beat: parseInt(document.getElementById('ticks-per-beat').value),
        beats_per_bar: parseInt(document.getElementById('beats-per-bar').value),
        generation_interval_ticks: parseInt(document.getElementById('gen-interval').value),
        accompaniment_velocity: parseInt(document.getElementById('acc-velocity').value),
        input_mode: document.getElementById('input-mode').value
    };
}

function updateButtonStates() {
    btnStart.disabled = isRunning;
    btnStop.disabled = !isRunning;
    btnRestart.disabled = !isRunning;
}

// Event listeners
btnStart.addEventListener('click', startClient);
btnStop.addEventListener('click', stopClient);
btnRestart.addEventListener('click', restartClient);

// Slider value displays
document.getElementById('tempo').addEventListener('input', (e) => {
    document.getElementById('tempo-value').textContent = e.target.value;
});
// ... similar for other sliders
```

---

### 6. Stats Panel (`stats.js`)

**HTML Structure**:
```html
<div id="stats">
    <h3>Statistics</h3>
    
    <div class="stat-row">
        <span class="stat-label">Position:</span>
        <span id="stat-position">Bar 0, Beat 0</span>
    </div>
    
    <div class="stat-row">
        <span class="stat-label">Current Tick:</span>
        <span id="stat-tick">0</span>
    </div>
    
    <div class="stat-row">
        <span class="stat-label">Hit Rate:</span>
        <span id="stat-hit-rate">--%</span>
    </div>
    
    <div class="stat-row">
        <span class="stat-label">Avg Backup Level:</span>
        <span id="stat-backup-level">--</span>
    </div>
    
    <hr>
    <h4>Latency</h4>
    
    <div class="stat-row">
        <span class="stat-label">Round Trip:</span>
        <span id="stat-round-trip">-- ms</span>
    </div>
    
    <div class="stat-row">
        <span class="stat-label">Server Processing:</span>
        <span id="stat-server-time">-- ms</span>
    </div>
    
    <div class="stat-row">
        <span class="stat-label">Network Latency:</span>
        <span id="stat-network">-- ms</span>
    </div>
</div>
```

**JavaScript**:
```javascript
function updateStats(statsData) {
    if (statsData.tick !== undefined) {
        document.getElementById('stat-tick').textContent = statsData.tick;
    }
    if (statsData.bar !== undefined && statsData.beat !== undefined) {
        document.getElementById('stat-position').textContent = 
            `Bar ${statsData.bar}, Beat ${statsData.beat}`;
    }
    if (statsData.hit_rate !== undefined) {
        document.getElementById('stat-hit-rate').textContent = 
            `${(statsData.hit_rate * 100).toFixed(1)}%`;
    }
    if (statsData.avg_backup_level !== undefined) {
        document.getElementById('stat-backup-level').textContent = 
            statsData.avg_backup_level.toFixed(2);
    }
    if (statsData.round_trip_ms !== undefined) {
        document.getElementById('stat-round-trip').textContent = 
            `${statsData.round_trip_ms.toFixed(1)} ms`;
    }
    if (statsData.server_process_ms !== undefined) {
        document.getElementById('stat-server-time').textContent = 
            `${statsData.server_process_ms.toFixed(1)} ms`;
    }
    if (statsData.network_latency_ms !== undefined) {
        document.getElementById('stat-network').textContent = 
            `${statsData.network_latency_ms.toFixed(1)} ms`;
    }
}
```

---

### 7. WebSocket Communication (`main.js`)

```javascript
let ws = null;

function connectWebSocket() {
    ws = new WebSocket(`ws://${window.location.host}/ws`);
    
    ws.onopen = () => {
        console.log('WebSocket connected');
        updateConnectionStatus('connected');
    };
    
    ws.onclose = () => {
        console.log('WebSocket disconnected');
        updateConnectionStatus('disconnected');
        // Reconnect after delay
        setTimeout(connectWebSocket, 2000);
    };
    
    ws.onerror = (error) => {
        console.error('WebSocket error:', error);
    };
    
    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handleMessage(data);
    };
}

function handleMessage(data) {
    switch (data.type) {
        case 'tick':
            currentTick = data.tick;
            updateStats({tick: data.tick, bar: data.bar, beat: data.beat});
            cleanupOldNotes();
            break;
            
        case 'note':
            if (data.event === 'on') {
                addNote(data);
                pressedKeys.add(data.pitch);
            } else {
                pressedKeys.delete(data.pitch);
            }
            break;
            
        case 'stats':
            updateStats(data);
            break;
            
        case 'config':
            updateConfigDisplay(data);
            break;
            
        case 'status':
            updateStatus(data.state, data.message);
            break;
    }
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    connectWebSocket();
    render();  // Start animation loop
});
```

---

## Testing Plan

### Phase 1 Tests
1. Start fake server, verify endpoints respond correctly
2. Test echo logic with sample melody notes
3. Verify WebSocket connection establishes

### Phase 2 Tests  
1. Piano keys render correctly
2. Notes fall at correct speed
3. Colors match note types
4. Playhead position is correct

### Phase 3 Tests
1. Start/stop/restart work
2. Config changes take effect
3. Input mode switching works
4. Keyboard input registers

### Phase 4 Tests
1. Stats update in real-time
2. Backup level visualization is correct
3. Late notes display correctly
4. Performance is smooth (60fps target)

---

## Running the Demo

```bash
# Terminal 1: Start fake server
cd app
python fake_server.py
# Runs on http://localhost:8001

# Terminal 2: Start web client
cd app  
python web_client.py
# Runs on http://localhost:8080

# Open browser to http://localhost:8080
```

---

## Future Enhancements (Not in Demo)

- [ ] MIDI device selection dropdown
- [ ] MIDI file upload and preview
- [ ] Latency timeline graph
- [ ] Backup level distribution chart
- [ ] Export session to MIDI
- [ ] Dark/light theme toggle
- [ ] Mobile responsive layout
