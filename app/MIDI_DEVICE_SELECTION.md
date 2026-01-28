# MIDI Device Selection for Web Client

## Changes Made

### 1. Backend API Endpoint (`web_client.py`)

Added `/api/midi_devices` endpoint that:
- Lists available MIDI input and output devices using `mido`
- Attempts to set rtmidi backend if not already set
- Filters out system devices (starting with 'RtMidi')
- Returns empty arrays gracefully if no devices found or on error

### 2. Web UI Controls (`index.html`)

Added:
- **MIDI Input Device** dropdown (shown only when Input Mode = "MIDI Device")
- **MIDI Output Device** dropdown (always visible)
- **MIDI File Path** input field (shown only when Input Mode = "MIDI File")
- **Refresh** button to reload MIDI device list

### 3. JavaScript Control Logic (`controls.js`)

Added functions:
- `loadMidiDevices()`: Fetches device list from API and populates dropdowns
- `setupInputModeHandling()`: Shows/hides relevant controls based on input mode
- Updated `gatherConfig()` to include:
  - `midi_input_name`
  - `midi_output_name`
  - `midi_file_path`
- Updated `populateFromConfig()` to restore MIDI device selections

### 4. Config Fields

The `ClientConfig` already had these fields:
- `midi_input_name`: Selected MIDI input device
- `midi_output_name`: Selected MIDI output device
- `midi_file_path`: Path to MIDI file for file input mode

## How It Works

1. **On Page Load**:
   - `loadMidiDevices()` is called automatically
   - Fetches available devices from `/api/midi_devices`
   - Populates both input and output device dropdowns

2. **Input Mode Selection**:
   - When "MIDI Device" is selected: Shows MIDI input device selector
   - When "MIDI File" is selected: Shows file path input
   - When "Computer Keyboard" is selected: Hides both

3. **Device Selection**:
   - User can select input/output devices from dropdowns
   - Click "Refresh" to reload device list if new devices are connected
   - Selected devices are passed in config when starting session

4. **When Starting Session**:
   - `midi_input_name` is used by `read_midi_input()` if input_mode = "midi"
   - `midi_output_name` is used by `AudioOutputHandler` for audio output
   - `midi_file_path` is used by `read_midi_file_input()` if input_mode = "file"

## Testing

1. Connect a MIDI device (keyboard, controller, etc.)
2. Start web_client: `python web_client.py`
3. Open browser to `http://localhost:8080`
4. You should see connected MIDI devices in the dropdowns
5. Select "MIDI Device" as input mode to use MIDI input
6. Select an output device to hear audio through that device

## Known Limitations

- MIDI device list is loaded once on page load
- Use "Refresh" button to update if devices are connected/disconnected
- On macOS, may need to grant microphone/input permissions for MIDI input
- If no MIDI output devices available, can still use FluidSynth virtual port

## Troubleshooting

If no devices appear:
1. Ensure MIDI devices are connected and powered on
2. Check that `python-rtmidi` is installed in environment
3. On macOS: Check System Preferences > Security & Privacy > Privacy > Microphone
4. Try running `mido.get_input_names()` in Python to verify devices are visible