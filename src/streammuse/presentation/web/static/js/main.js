/**
 * StreamMUSE Viewer — read-only WebSocket client.
 *
 * Connects to /ws, routes inbound JSON envelopes to PianoVisualizer / Stats,
 * and writes session metadata into the Session panel.
 *
 * No outbound messages, no controls, no parameter editing. The viewer is a
 * passive observer of whatever the streammuse-web process is doing.
 */

(function() {
    let ws = null;
    let reconnectAttempts = 0;
    const MAX_RECONNECT_ATTEMPTS = 10;

    function updateConnectionStatus(connected) {
        const statusEl = document.getElementById('connection-status');
        if (!statusEl) return;
        if (connected) {
            statusEl.textContent = 'Connected';
            statusEl.className = 'status connected';
        } else {
            statusEl.textContent = 'Disconnected';
            statusEl.className = 'status disconnected';
        }
    }

    function setSessionInfo(config) {
        if (!config) return;
        const tempo = document.getElementById('info-tempo');
        const input = document.getElementById('info-input');
        const model = document.getElementById('info-model');
        if (tempo && config.tempo_bpm !== undefined) {
            tempo.textContent = `${config.tempo_bpm} BPM`;
        } else if (tempo && config.tempo !== undefined) {
            tempo.textContent = `${config.tempo} BPM`;
        }
        if (input && config.input_type !== undefined) {
            input.textContent = String(config.input_type);
        } else if (input && config.input_mode !== undefined) {
            input.textContent = String(config.input_mode);
        }
        if (model && config.model_name !== undefined) {
            model.textContent = String(config.model_name);
        } else if (model && config.inference_type !== undefined) {
            model.textContent = String(config.inference_type);
        }
    }

    function connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        ws = new WebSocket(`${protocol}//${window.location.host}/ws`);

        ws.onopen = () => {
            console.log('WebSocket connected');
            updateConnectionStatus(true);
            reconnectAttempts = 0;
        };

        ws.onclose = () => {
            console.log('WebSocket disconnected');
            updateConnectionStatus(false);
            if (reconnectAttempts < MAX_RECONNECT_ATTEMPTS) {
                reconnectAttempts++;
                const delay = Math.min(1000 * Math.pow(2, reconnectAttempts), 30000);
                console.log(`Reconnecting in ${delay}ms (attempt ${reconnectAttempts})`);
                setTimeout(connectWebSocket, delay);
            }
        };

        ws.onerror = (error) => {
            console.error('WebSocket error:', error);
        };

        ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                handleMessage(data);
            } catch (e) {
                console.error('Error parsing message:', e);
            }
        };
    }

    function handleMessage(data) {
        switch (data.type) {
            case 'tick':
                PianoVisualizer.setCurrentTick(data.tick);
                Stats.updateStats({
                    tick: data.tick,
                    bar: data.bar,
                    beat: data.beat,
                });
                break;

            case 'note':
                if (data.event === 'on') {
                    PianoVisualizer.addNote(data);
                    PianoVisualizer.pressKey(data.pitch);
                } else {
                    PianoVisualizer.noteOff(data.pitch, data.tick, data.source);
                    PianoVisualizer.releaseKey(data.pitch);
                }
                break;

            case 'stats':
                Stats.updateStats(data);
                break;

            case 'config':
                console.log('Config received:', data);
                setSessionInfo(data);
                break;

            case 'status':
                console.log('Status:', data.state, data.message);
                break;

            case 'inference_result':
                // Diagnostic only — Stats panel already shows the latency
                // metrics from the 'stats' envelope.
                console.log('Inference result:', data);
                break;

            default:
                console.log('Unknown message type:', data.type);
        }
    }

    document.addEventListener('DOMContentLoaded', () => {
        connectWebSocket();
    });
})();
