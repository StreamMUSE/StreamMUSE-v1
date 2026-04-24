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
            midi_input_name: document.getElementById('midi-input-device').value || null,
            midi_output_name: document.getElementById('midi-output-device').value || null,
            midi_file_path: document.getElementById('midi-file-path').value || null,
            metronome: document.getElementById('metronome').checked,
            listening_duration_ticks: parseInt(document.getElementById('listening-duration').value),
            manual_prompt_path: document.getElementById('manual-prompt').value || null,
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
        if (config.midi_input_name) {
            document.getElementById('midi-input-device').value = config.midi_input_name;
        }
        if (config.midi_output_name) {
            document.getElementById('midi-output-device').value = config.midi_output_name;
        }
        if (config.midi_file_path) {
            document.getElementById('midi-file-path').value = config.midi_file_path;
        }
        if (config.manual_prompt_path) {
            document.getElementById('manual-prompt').value = config.manual_prompt_path;
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
    
    async function loadMidiDevices() {
        try {
            const response = await fetch('/api/midi_devices');
            const data = await response.json();
            
            // Populate input devices
            const inputSelect = document.getElementById('midi-input-device');
            inputSelect.innerHTML = '<option value="">None</option>';
            data.input_devices.forEach(device => {
                const option = document.createElement('option');
                option.value = device;
                option.textContent = device;
                inputSelect.appendChild(option);
            });
            
            // Populate output devices
            const outputSelect = document.getElementById('midi-output-device');
            outputSelect.innerHTML = '<option value="">None</option>';
            data.output_devices.forEach(device => {
                const option = document.createElement('option');
                option.value = device;
                option.textContent = device;
                outputSelect.appendChild(option);
            });
        } catch (e) {
            console.error('Error loading MIDI devices:', e);
        }
    }
    
    function setupInputModeHandling() {
        const inputMode = document.getElementById('input-mode');
        const midiInputGroup = document.getElementById('midi-input-group');
        const midiFileGroup = document.getElementById('midi-file-group');
        
        inputMode.addEventListener('change', (e) => {
            const mode = e.target.value;
            midiInputGroup.style.display = mode === 'midi' ? 'block' : 'none';
            midiFileGroup.style.display = mode === 'file' ? 'block' : 'none';
        });
        
        // Trigger initial state
        inputMode.dispatchEvent(new Event('change'));
    }
    
    async function loadManualPrompts() {
        try {
            const response = await fetch('/api/manual_prompts');
            const data = await response.json();
            
            const select = document.getElementById('manual-prompt');
            select.innerHTML = '<option value="">None</option>';
            
            data.prompts.forEach(prompt => {
                const option = document.createElement('option');
                option.value = prompt.path;
                option.textContent = prompt.name;
                select.appendChild(option);
            });
            
            console.log(`Loaded ${data.prompts.length} manual prompts`);
        } catch (e) {
            console.error('Error loading manual prompts:', e);
        }
    }
    
    btnStart.addEventListener('click', startClient);
    btnStop.addEventListener('click', stopClient);
    btnRestart.addEventListener('click', restartClient);
    
    setupSliderDisplays();
    setupInputModeHandling();
    loadMidiDevices();
    loadManualPrompts();
    
    return {
        setRunning,
        gatherConfig,
        populateFromConfig,
        loadMidiDevices,
        loadManualPrompts
    };
})();
