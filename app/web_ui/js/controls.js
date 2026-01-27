/**
 * Controls for StreamMUSE Web UI
 * Handles start/stop, config changes, etc.
 */

const Controls = (function() {
    const btnStart = document.getElementById('btn-start');
    const btnStop = document.getElementById('btn-stop');
    const btnRestart = document.getElementById('btn-restart');
    
    let isRunning = false;
    
    let serverConfig = null;
    
    function gatherConfig() {
        const config = {
            tempo: parseFloat(document.getElementById('tempo').value),
            ticks_per_beat: parseInt(document.getElementById('ticks-per-beat').value),
            beats_per_bar: parseInt(document.getElementById('beats-per-bar').value),
            generation_interval_ticks: parseInt(document.getElementById('gen-interval').value),
            accompaniment_velocity: parseInt(document.getElementById('acc-velocity').value),
            input_mode: document.getElementById('input-mode').value,
            metronome: document.getElementById('metronome').checked,
        };
        if (serverConfig) {
            if (serverConfig.server_url) config.server_url = serverConfig.server_url;
            if (serverConfig.generation_length_per_request) config.generation_length_per_request = serverConfig.generation_length_per_request;
        }
        return config;
    }
    
    async function startClient() {
        const config = gatherConfig();
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
            {id: 'tempo', displayId: 'tempo-value', suffix: ''},
            {id: 'gen-interval', displayId: 'gen-interval-value', suffix: ''},
            {id: 'acc-velocity', displayId: 'acc-velocity-value', suffix: ''},
            {id: 'fall-speed', displayId: 'fall-speed-value', suffix: ''},
            {id: 'visible-ahead', displayId: 'visible-ahead-value', suffix: ''},
            {id: 'visible-behind', displayId: 'visible-behind-value', suffix: ''}
        ];
        
        sliders.forEach(({id, displayId, suffix}) => {
            const slider = document.getElementById(id);
            const display = document.getElementById(displayId);
            if (slider && display) {
                slider.addEventListener('input', (e) => {
                    display.textContent = e.target.value + suffix;
                    
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
    
    function setServerConfig(config) {
        serverConfig = config;
    }
    
    return {
        setRunning,
        gatherConfig,
        setServerConfig
    };
})();
