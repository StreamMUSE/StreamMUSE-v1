/**
 * Main entry point for StreamMUSE Web UI
 * Handles WebSocket connection, message routing, and browser keyboard input
 */

(function() {
    let ws = null;
    let reconnectAttempts = 0;
    const MAX_RECONNECT_ATTEMPTS = 10;
    
    // Keyboard mapping (same as Python client)
    const KEY_TO_PITCH = {
        'z': 60, 'x': 62, 'c': 64, 'v': 65, 'b': 67, 'n': 69, 'm': 71,
        ',': 72, '.': 74, '/': 76,
        's': 61, 'd': 63, 'g': 66, 'h': 68, 'j': 70, 'l': 73, ';': 75,
    };
    const pressedBrowserKeys = new Set();
    
    function updateConnectionStatus(connected) {
        const statusEl = document.getElementById('connection-status');
        if (connected) {
            statusEl.textContent = 'Connected';
            statusEl.className = 'status connected';
        } else {
            statusEl.textContent = 'Disconnected';
            statusEl.className = 'status disconnected';
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
                    beat: data.beat
                });
                break;
                
            case 'note':
                if (data.event === 'on') {
                    PianoVisualizer.addNote(data);
                    PianoVisualizer.pressKey(data.pitch);
                } else {
                    PianoVisualizer.releaseKey(data.pitch);
                }
                break;
                
            case 'stats':
                Stats.updateStats(data);
                break;
                
            case 'config':
                console.log('Config received:', data.config);
                Controls.populateFromConfig(data.config);
                break;
                
            case 'status':
                console.log('Status:', data.state, data.message);
                if (data.state === 'running') {
                    Controls.setRunning(true);
                } else if (data.state === 'stopped') {
                    Controls.setRunning(false);
                }
                break;
                
            case 'inference_result':
                console.log('Inference result:', data);
                break;
                
            default:
                console.log('Unknown message type:', data.type);
        }
    }
    
    // Browser keyboard input handling
    function setupKeyboardInput() {
        document.addEventListener('keydown', (e) => {
            const key = e.key.toLowerCase();
            if (KEY_TO_PITCH[key] && !pressedBrowserKeys.has(key)) {
                pressedBrowserKeys.add(key);
                const pitch = KEY_TO_PITCH[key];
                
                // Send to server via WebSocket
                if (ws && ws.readyState === WebSocket.OPEN) {
                    ws.send(JSON.stringify({
                        type: 'keyboard_input',
                        event: 'note_on',
                        pitch: pitch,
                        velocity: 100
                    }));
                }
                
                // Also update local display immediately for responsiveness
                PianoVisualizer.pressKey(pitch);
                console.log('Key pressed:', key, '-> pitch', pitch);
            }
        });
        
        document.addEventListener('keyup', (e) => {
            const key = e.key.toLowerCase();
            if (KEY_TO_PITCH[key]) {
                pressedBrowserKeys.delete(key);
                const pitch = KEY_TO_PITCH[key];
                
                // Send to server via WebSocket
                if (ws && ws.readyState === WebSocket.OPEN) {
                    ws.send(JSON.stringify({
                        type: 'keyboard_input',
                        event: 'note_off',
                        pitch: pitch
                    }));
                }
                
                // Update local display
                PianoVisualizer.releaseKey(pitch);
                console.log('Key released:', key, '-> pitch', pitch);
            }
        });
        
        console.log('Browser keyboard input enabled. Keys: z-m (white), s/d/g/h/j/l/; (black)');
    }
    
    // Initialize on page load
    document.addEventListener('DOMContentLoaded', () => {
        connectWebSocket();
        setupKeyboardInput();
        
        fetch('/api/status')
            .then(r => r.json())
            .then(status => {
                Controls.setRunning(status.is_running);
                if (status.config) {
                    Controls.populateFromConfig(status.config);
                    console.log('Initial config from server:', status.config);
                }
            })
            .catch(e => console.error('Error fetching status:', e));
    });
})();
