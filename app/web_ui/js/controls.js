/**
 * Controls for StreamMUSE Web UI
 * Handles start/stop, config changes, etc.
 */

const Controls = (function() {
    const btnStart = document.getElementById('btn-start');
    const btnStop = document.getElementById('btn-stop');
    const btnRestart = document.getElementById('btn-restart');
    
    let isRunning = false;
    
    function gatherConfig() {
        return {
            server_url: document.getElementById('server-url').value,
            tempo: parseFloat(document.getElementById('tempo').value),
            ticks_per_beat: parseInt(document.getElementById('ticks-per-beat').value),
            beats_per_bar: parseInt(document.getElementById('beats-per-bar').value),
            generation_interval_ticks: parseInt(document.getElementById('gen-interval').value),
            generation_length_per_request: parseInt(document.getElementById('gen-length').value),
            accompaniment_velocity: parseInt(document.getElementById('acc-velocity').value),
            input_mode: document.getElementById('input-mode').value,
            metronome: document.getElementById('metronome').checked,
            listening_duration_ticks: parseInt(document.getElementById('listening-duration').value),
            prompt_dir: document.getElementById('prompt-dir').value || null,
            key_detection_method: document.getElementById('key-detection').value,
        };
    }
    
    function populateFromConfig(config) {
        if (!config) return;
        
        if (config.server_url) {
            document.getElementById('server-url').value = config.server_url;
        }
        if (config.tempo) {
            document.getElementById('tempo').value = config.tempo;
            document.getElementById('tempo-value').textContent = config.tempo;
        }
        if (config.ticks_per_beat) {
            document.getElementById('ticks-per-beat').value = config.ticks_per_beat;
        }
        if (config.beats_per_bar) {
            document.getElementById('beats-per-bar').value = config.beats_per_bar;
        }
        if (config.generation_interval_ticks) {
            document.getElementById('gen-interval').value = config.generation_interval_ticks;
            document.getElementById('gen-interval-value').textContent = config.generation_interval_ticks;
        }
        if (config.generation_length_per_request) {
            document.getElementById('gen-length').value = config.generation_length_per_request;
            document.getElementById('gen-length-value').textContent = config.generation_length_per_request;
        }
        if (config.accompaniment_velocity !== undefined) {
            document.getElementById('acc-velocity').value = config.accompaniment_velocity;
            document.getElementById('acc-velocity-value').textContent = config.accompaniment_velocity;
        }
        if (config.input_mode) {
            document.getElementById('input-mode').value = config.input_mode;
        }
        if (config.metronome !== undefined) {
            document.getElementById('metronome').checked = config.metronome;
        }
        if (config.listening_duration_ticks !== undefined) {
            document.getElementById('listening-duration').value = config.listening_duration_ticks;
            document.getElementById('listening-duration-value').textContent = config.listening_duration_ticks;
        }
        if (config.prompt_dir) {
            document.getElementById('prompt-dir').value = config.prompt_dir;
        }
        if (config.key_detection_method) {
            document.getElementById('key-detection').value = config.key_detection_method;
        }
    }
    
    async function startClient() {
        const config = gatherConfig();
        console.log('Starting with config:', config);
        try {
            const response = await fetch('/api/start', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(config)
            });
            const result = await response.json();
            if (result.success) {
                isRunning = true;
                updateButtonStates();
                PianoVisualizer.clearNotes();
            } else {
                console.error('Failed to start:', result.message);
            }
        } catch (e) {
            console.error('Error starting client:', e);
        }
    }
    
    async function stopClient() {
        try {
            const response = await fetch('/api/stop', {method: 'POST'});
            const result = await response.json();
            if (result.success) {
                isRunning = false;
                updateButtonStates();
            }
        } catch (e) {
            console.error('Error stopping client:', e);
        }
    }
    
    async function restartClient() {
        const config = gatherConfig();
        try {
            const response = await fetch('/api/restart', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(config)
            });
            const result = await response.json();
            if (result.success) {
                isRunning = true;
                updateButtonStates();
                PianoVisualizer.clearNotes();
            }
        } catch (e) {
            console.error('Error restarting client:', e);
        }
    }
    
    function updateButtonStates() {
        btnStart.disabled = isRunning;
        btnStop.disabled = !isRunning;
        btnRestart.disabled = !isRunning;
    }
    
    function setRunning(running) {
        isRunning = running;
        updateButtonStates();
    }
    
    function setupSliderDisplays() {
        const sliders = [
            {id: 'tempo', displayId: 'tempo-value'},
            {id: 'gen-interval', displayId: 'gen-interval-value'},
            {id: 'gen-length', displayId: 'gen-length-value'},
            {id: 'acc-velocity', displayId: 'acc-velocity-value'},
            {id: 'listening-duration', displayId: 'listening-duration-value'},
            {id: 'fall-speed', displayId: 'fall-speed-value'},
            {id: 'visible-ahead', displayId: 'visible-ahead-value'},
            {id: 'visible-behind', displayId: 'visible-behind-value'}
        ];
        
        sliders.forEach(({id, displayId}) => {
            const slider = document.getElementById(id);
            const display = document.getElementById(displayId);
            if (slider && display) {
                slider.addEventListener('input', (e) => {
                    display.textContent = e.target.value;
                    
                    if (id === 'fall-speed') {
                        PianoVisualizer.setPixelsPerTick(parseInt(e.target.value));
                    } else if (id === 'visible-ahead') {
                        PianoVisualizer.setVisibleTicksAhead(parseInt(e.target.value));
                    } else if (id === 'visible-behind') {
                        PianoVisualizer.setVisibleTicksBehind(parseInt(e.target.value));
                    }
                });
            }
        });
    }
    
    btnStart.addEventListener('click', startClient);
    btnStop.addEventListener('click', stopClient);
    btnRestart.addEventListener('click', restartClient);
    
    setupSliderDisplays();
    
    return {
        setRunning,
        gatherConfig,
        populateFromConfig
    };
})();
