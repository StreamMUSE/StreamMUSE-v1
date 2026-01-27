# StreamMUSE Web Client User Guide

A comprehensive guide to using the StreamMUSE Web Client for real-time AI-powered musical accompaniment generation.

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Command Line Arguments](#command-line-arguments)
4. [Web UI Controls](#web-ui-controls)
5. [Listening Mode](#listening-mode)
6. [Prompt Library](#prompt-library)
7. [Key Detection](#key-detection)
8. [Keyboard Input Mapping](#keyboard-input-mapping)
9. [Understanding the Visualization](#understanding-the-visualization)
10. [Troubleshooting](#troubleshooting)

---

## Overview

The StreamMUSE Web Client provides a browser-based interface for interacting with the StreamMUSE real-time accompaniment generation system. It features:

- **Real-time piano visualization** with horizontal scrolling notes
- **Multiple input modes**: Computer keyboard, MIDI device, or MIDI file
- **Listening Mode**: Automatic key detection and prompt injection for better accompaniment quality
- **Live statistics**: Latency metrics, hit rate, and backup level monitoring
- **Configurable parameters**: Tempo, generation settings, visual settings, and more

---

## Quick Start

### Prerequisites

1. A running StreamMUSE inference server (default: `http://localhost:8000`)
2. Python environment with required dependencies
3. (Optional) FluidSynth or other MIDI synthesizer for audio output
4. (Optional) `music21` library for advanced key detection

### Basic Usage

```bash
# Start with defaults (no listening mode)
python web_client.py

# Start with listening mode enabled
python web_client.py \
  --server_url http://localhost:8000/generate_accompaniment \
  --tempo 120 \
  --listening_duration_ticks 64 \
  --prompt_dir ../prompts \
  --key_detection_method music21
```

Then open `http://localhost:8080` in your browser.

---

## Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--server_url` | string | `http://localhost:8000/generate_accompaniment` | URL of the StreamMUSE inference server |
| `--tempo` | float | `120.0` | Tempo in beats per minute (BPM) |
| `--ticks_per_beat` | int | `4` | Number of ticks per beat (timing resolution) |
| `--beats_per_bar` | int | `4` | Beats per bar (time signature numerator) |
| `--generation_interval_ticks` | int | `1` | How often to send generation requests (in ticks) |
| `--generation_length_per_request` | int | `5` | Number of frames to generate per request |
| `--accompaniment_velocity` | int | `50` | MIDI velocity for accompaniment notes (0-127) |
| `--listening_duration_ticks` | int | `0` | Duration of listening period (0 = disabled) |
| `--prompt_dir` | string | `None` | Path to prompt library directory |
| `--key_detection_method` | string | `lightweight` | Key detection algorithm: `lightweight` or `music21` |
| `--port` | int | `8080` | Port for the web UI server |

### Example Commands

```bash
# High-tempo jazz session
python web_client.py --tempo 180 --generation_interval_ticks 2

# Slow ballad with listening mode
python web_client.py \
  --tempo 72 \
  --listening_duration_ticks 96 \
  --prompt_dir ../prompts \
  --key_detection_method music21

# Connect to remote server
python web_client.py \
  --server_url http://192.168.1.100:8000/generate_accompaniment \
  --port 3000
```

---

## Web UI Controls

### Session Controls

| Control | Description |
|---------|-------------|
| **Start** | Begin the accompaniment session |
| **Stop** | End the current session |
| **Restart** | Stop and restart with current settings |

### Server Settings

| Control | Description |
|---------|-------------|
| **Server URL** | The inference server endpoint |
| **Input Mode** | `Computer Keyboard`, `MIDI Device`, or `MIDI File` |

### Music Parameters

| Control | Range | Description |
|---------|-------|-------------|
| **Tempo** | 40-200 BPM | Playback and generation speed |
| **Ticks per Beat** | 2, 4, 8 | Timing resolution |
| **Beats per Bar** | 3, 4, 6 | Time signature |
| **Generation Interval** | 1-8 ticks | Frequency of generation requests |
| **Generation Length** | 1-20 frames | Notes generated per request |
| **Accompaniment Velocity** | 0-127 | Volume of generated notes |
| **Metronome** | On/Off | Audible beat clicks |

### Listening Mode Settings

| Control | Range | Description |
|---------|-------|-------------|
| **Listening Duration** | 0-64 ticks | Time to collect user input (0 = disabled) |
| **Prompt Directory** | path | Location of prompt MIDI files |
| **Key Detection** | `Lightweight` / `Music21` | Algorithm for detecting musical key |

### Visual Settings

| Control | Range | Description |
|---------|-------|-------------|
| **Fall Speed** | 20-100 px/tick | Note scroll speed |
| **Visible Ahead** | 8-64 ticks | Future notes shown |
| **Visible Behind** | 4-32 ticks | Past notes shown (history) |

---

## Listening Mode

Listening Mode is a feature that improves accompaniment quality by:
1. Collecting user input for a specified duration
2. Automatically detecting the musical key
3. Injecting a matching prompt (melody + accompaniment pair) to the server
4. Starting real-time generation with better context

### How It Works

```
┌─────────────────────────────────────────────────────────────────────┐
│                        LISTENING MODE TIMELINE                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  LISTENING PERIOD              GRACE PERIOD        REAL-TIME GEN    │
│  ─────────────────────────     ────────────        ──────────────── │
│                                                                      │
│  Tick 0          Tick N        Tick N+8           Tick N+8+...      │
│  │                │             │                  │                 │
│  ▼                ▼             ▼                  ▼                 │
│  ┌────────────────┐ ┌──────────┐ ┌────────────────────────────────┐ │
│  │ Collect user   │ │ Detect   │ │ Normal accompaniment          │ │
│  │ melody notes   │ │ key &    │ │ generation begins             │ │
│  │                │ │ inject   │ │                                │ │
│  │ Metronome ON   │ │ prompt   │ │ Server uses injected context  │ │
│  │ No generation  │ │          │ │                                │ │
│  └────────────────┘ └──────────┘ └────────────────────────────────┘ │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Phase 1: Listening Period (Ticks 0 to N)

- Duration: `listening_duration_ticks` (e.g., 64 ticks = 4 bars at 4 ticks/beat, 4 beats/bar)
- User plays melody notes freely
- Metronome plays to keep time
- **No accompaniment is generated** during this phase
- All notes are collected for key analysis

### Phase 2: Grace Period (Ticks N to N+8)

- Duration: 8 ticks (fixed, ~half a bar)
- Background thread performs:
  1. **Key Detection**: Analyzes collected notes to determine musical key
  2. **Prompt Selection**: Finds a matching prompt from the library
  3. **Server Injection**: Sends prompt data to prime the model
- Metronome continues
- Still no accompaniment generation

### Phase 3: Real-Time Generation (Tick N+8 onwards)

- Normal operation begins
- Server generates accompaniment using the injected context
- Results in more musically coherent accompaniment

### Recommended Settings

| Use Case | Listening Duration | Key Detection |
|----------|-------------------|---------------|
| Quick demo | 32 ticks (2 bars) | `lightweight` |
| Standard use | 64 ticks (4 bars) | `music21` |
| Extended intro | 96 ticks (6 bars) | `music21` |

---

## Prompt Library

The prompt library contains pre-recorded melody and accompaniment pairs organized by musical key. These are used by Listening Mode to provide context to the model.

### Directory Structure

```
prompts/
├── metadata.json           # Index of all prompts
├── C_major/
│   ├── pop909_216_mel.mid  # Melody track
│   ├── pop909_216_acc.mid  # Accompaniment track
│   ├── pop909_320_mel.mid
│   ├── pop909_320_acc.mid
│   └── ...
├── G_major/
│   └── ...
├── A_minor/
│   └── ...
└── [Key]_[mode]/
    └── ...
```

### metadata.json Format

```json
{
  "C_major": [
    {
      "name": "pop909_216",
      "melody_path": "/absolute/path/to/prompts/C_major/pop909_216_mel.mid",
      "accompaniment_path": "/absolute/path/to/prompts/C_major/pop909_216_acc.mid",
      "duration_ticks": 1920,
      "source": "POP909",
      "song_number": "216",
      "melody_notes_count": 511,
      "accompaniment_notes_count": 1508
    },
    {
      "name": "pop909_320",
      "melody_path": "/absolute/path/to/prompts/C_major/pop909_320_mel.mid",
      "accompaniment_path": "/absolute/path/to/prompts/C_major/pop909_320_acc.mid",
      "duration_ticks": 1120,
      "source": "POP909",
      "song_number": "320",
      "melody_notes_count": 243,
      "accompaniment_notes_count": 852
    }
  ],
  "A_minor": [
    ...
  ]
}
```

### Prompt Entry Fields

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Unique identifier for the prompt |
| `melody_path` | string | **Absolute path** to melody MIDI file |
| `accompaniment_path` | string | **Absolute path** to accompaniment MIDI file |
| `duration_ticks` | int | Total length of the prompt in ticks |
| `source` | string | Dataset origin (e.g., "POP909", "Custom") |
| `song_number` | string | (Optional) Reference ID from source dataset |
| `melody_notes_count` | int | (Optional) Number of notes in melody |
| `accompaniment_notes_count` | int | (Optional) Number of notes in accompaniment |

### Adding New Prompts

#### Step 1: Prepare MIDI Files

Create two MIDI files for your prompt:
- `[name]_mel.mid` - Melody track only
- `[name]_acc.mid` - Accompaniment track only

Requirements:
- Single track per file
- Quantized to your target `ticks_per_beat` (default: 4)
- Start at tick 0

#### Step 2: Determine the Key

Identify the musical key of your piece (e.g., `C_major`, `F#_minor`).

Supported key formats:
- Major keys: `C_major`, `G_major`, `D_major`, `A_major`, `E_major`, `B_major`, `F_major`, `Bb_major`, `Eb_major`, `Ab_major`
- Minor keys: `A_minor`, `E_minor`, `B_minor`, `F#_minor`, `C#_minor`, `D_minor`, `G_minor`, `C_minor`, `F_minor`, `Bb_minor`

#### Step 3: Create Key Directory

```bash
mkdir -p prompts/[Key]_[mode]
# Example:
mkdir -p prompts/E_minor
```

#### Step 4: Copy MIDI Files

```bash
cp my_song_mel.mid prompts/E_minor/
cp my_song_acc.mid prompts/E_minor/
```

#### Step 5: Update metadata.json

Add an entry for your new prompt:

```json
{
  "E_minor": [
    {
      "name": "my_song",
      "melody_path": "/full/path/to/prompts/E_minor/my_song_mel.mid",
      "accompaniment_path": "/full/path/to/prompts/E_minor/my_song_acc.mid",
      "duration_ticks": 640,
      "source": "Custom"
    }
  ]
}
```

**Important**: Use **absolute paths** in metadata.json, not relative paths.

#### Step 6: Verify

```bash
python -c "
from prompt_library import PromptLibrary
lib = PromptLibrary('prompts')
print('Keys available:', list(lib.metadata.keys()))
print('E_minor prompts:', lib.get_prompts_for_key('E_minor'))
"
```

### Fallback Behavior

If no prompts exist for the detected key:
1. System falls back to `C_major` prompts
2. If `C_major` is also empty, listening mode injection is skipped
3. Generation proceeds without prompt context

---

## Key Detection

Two algorithms are available for detecting the musical key from user input.

### Lightweight (Default)

- **Algorithm**: Krumhansl-Kessler key profiles with Pearson correlation
- **Speed**: ~1ms
- **Accuracy**: Good for clear tonal music
- **Dependencies**: None (pure Python)

How it works:
1. Counts pitch classes (C, C#, D, etc.) weighted by note duration
2. Correlates the distribution against major and minor key profiles
3. Returns the key with highest correlation

### Music21

- **Algorithm**: Music21 library's `analyze('key')` function
- **Speed**: ~50-200ms
- **Accuracy**: Better for complex or ambiguous tonality
- **Dependencies**: Requires `pip install music21`

How it works:
1. Converts notes to a music21 Stream
2. Uses music21's sophisticated key analysis (based on Krumhansl-Schmuckler)
3. Falls back to lightweight method if music21 fails

### Recommendation

| Situation | Recommended Method |
|-----------|-------------------|
| Live performance (low latency needed) | `lightweight` |
| Pre-recorded or slower tempo | `music21` |
| Atonal or chromatic music | Either (may default to C major) |
| Clear tonal center | Either |

---

## Keyboard Input Mapping

When using Computer Keyboard input mode, the following keys map to MIDI pitches:

### White Keys (Bottom Row: Z-M, Comma, Period, Slash)

| Key | Note | MIDI Pitch |
|-----|------|------------|
| Z | C4 | 60 |
| X | D4 | 62 |
| C | E4 | 64 |
| V | F4 | 65 |
| B | G4 | 67 |
| N | A4 | 69 |
| M | B4 | 71 |
| , | C5 | 72 |
| . | D5 | 74 |
| / | E5 | 76 |

### Black Keys (Home Row: S, D, G, H, J, L, ;)

| Key | Note | MIDI Pitch |
|-----|------|------------|
| S | C#4 | 61 |
| D | D#4 | 63 |
| G | F#4 | 66 |
| H | G#4 | 68 |
| J | A#4 | 70 |
| L | C#5 | 73 |
| ; | D#5 | 75 |

### Visual Layout

```
    S   D       G   H   J       L   ;
    │   │       │   │   │       │   │
┌───┴───┴───┬───┴───┴───┴───┬───┴───┴───┐
│ Z │ X │ C │ V │ B │ N │ M │ , │ . │ / │
│C4 │D4 │E4 │F4 │G4 │A4 │B4 │C5 │D5 │E5 │
└───┴───┴───┴───┴───┴───┴───┴───┴───┴───┘
```

---

## Understanding the Visualization

### Layout

```
┌─────────────────────────────────────────────────────────────────────┐
│                         PIANO VISUALIZATION                          │
├────────────┬────────────────────────────────────────────────────────┤
│            │     PAST (history)    │ NOW │     FUTURE (upcoming)    │
│  KEYBOARD  │◄───────────────────────┼─────┼───────────────────────►│
│            │                        │     │                          │
│            │  ████  (played notes)  │  █  │  ████  (scheduled)      │
│            │                        │     │                          │
└────────────┴────────────────────────┴─────┴──────────────────────────┘
              ◄─── Notes scroll this direction ────
```

### Note Colors

| Color | Meaning |
|-------|---------|
| **Blue** | User melody notes |
| **Green** | Model-generated accompaniment |
| **Gray dashed** | Replaced/backed-up notes (overwritten by newer generation) |

### The "NOW" Line

- Vertical line at ~40% from the left edge
- Notes crossing this line are being played
- Left of line = already played (history)
- Right of line = scheduled to play (future)

### Backup Level

When a newer inference result arrives, it may replace notes that were scheduled but not yet played. The "backup level" indicates how far back the replacement occurred:
- **Backup 0**: Note was generated and played on time
- **Backup 1+**: Note was replaced by a newer generation

Lower average backup levels indicate better real-time performance.

---

## Troubleshooting

### No Sound

1. Check that a MIDI synthesizer (e.g., FluidSynth) is running
2. Verify the correct MIDI output port is selected
3. Check the Accompaniment Velocity setting (should be > 0)

### "Listening mode dependencies NOT available"

Install the required modules:
```bash
pip install music21  # For music21 key detection
```

### "Could not initialize prompt library"

1. Verify `--prompt_dir` points to a valid directory
2. Check that `metadata.json` exists in that directory
3. Ensure paths in metadata.json are absolute and correct

### Notes Not Generating

1. Check the server is running and accessible
2. Verify the Server URL is correct
3. Look for error messages in the terminal

### High Latency / Delayed Notes

1. Reduce `--generation_length_per_request`
2. Increase `--generation_interval_ticks`
3. Check network connection to server
4. Monitor the "Round Trip" and "Network Latency" stats

### Listening Mode Not Activating

1. Ensure `listening_duration_ticks` > 0
2. Verify `prompt_dir` is set correctly
3. Check terminal for "LISTENING MODE ACTIVE" message
4. Confirm dependencies are available (check for init messages)

---

## Tips for Best Results

1. **Start with Listening Mode**: Even 2-4 bars of input helps the model generate more coherent accompaniment

2. **Match the Tempo**: Set the tempo before starting to match your intended playing speed

3. **Use Music21 for Accuracy**: If latency permits, music21 provides better key detection

4. **Build Your Prompt Library**: Add prompts in keys you frequently play in

5. **Watch the Backup Level**: High backup levels indicate the system is struggling to keep up - try reducing generation length

6. **Use MIDI Input**: For serious playing, connect a MIDI keyboard for better expression and timing

---

## Version Information

- Web Client Version: 1.0
- Compatible with StreamMUSE Server API v1
- Last Updated: January 2025
