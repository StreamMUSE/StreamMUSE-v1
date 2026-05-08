# Generation Logging System

This document describes the comprehensive logging system for comparing Fake Realtime and Offline inference modes.

## Overview

The logging system captures complete token sequences and event timelines for both inference modes, enabling detailed comparison and debugging.

## Architecture

### Components

1. **`generation_logger.py`** - Core logging module
   - `GenerationLogger`: Main logger class for both modes
   - `GenerationLog`: Dataclass for single generation records
   - `compare_logs()`: Utility to compare two log files

2. **Backend Integration**
   - `lekai_http_backend.py`: FakeRT logging in `_generate_with_interleaved_prompt()`
   - `model.py`: Offline logging in `generate_accompaniment()`

3. **Server/Client Updates**
   - `server_lekai.py`: Accepts `input_file` parameter
   - `http_client.py`: Passes `input_file` to server
   - `run_lekai_fake_realtime.py`: Sends MIDI file path

4. **Comparison Tools**
   - `compare_generation_logs.py`: Compare two logs or directories
   - `test_logging_system.py`: Test logging functionality

## Usage

### Environment Variables

```bash
# Fake Realtime settings
export LEKAI_RT_TEMPERATURE=0.8
export LEKAI_RT_TOP_K=1          # Use 1 for deterministic output
export LEKAI_RT_TOP_P=0.95
export LEKAI_RT_REPETITION_PENALTY=1.2
export LEKAI_RT_LOG_DIR=logs/fake_rt  # Log output directory

# Offline settings
export LEKAI_OFFLINE_LOG_DIR=logs/offline
```

### Running Fake Realtime with Logging

```bash
# Start server with checkpoint
LEKAI_CHECKPOINT_PATH=models/ModelLekai/epoch_4_1104_1204/model.safetensors \
LEKAI_RT_TOP_K=1 \
LEKAI_RT_LOG_DIR=logs/fake_rt \
uv run python -m streammuse.infrastructure.inference.server_lekai

# Run FakeRT client
LEKAI_RT_TOP_K=1 \
uv run python scripts/run_lekai_fake_realtime.py \
    --midi-file-path prompts/inputs_lekai/mel/1-5.mid \
    --output-dir outputs/fake_rt_1-5/ \
    --generation-interval-ticks 4 \
    --generation-length-frames 4 \
    --server-url http://127.0.0.1:8000/generate_accompaniment
```

### Running Offline with Logging

```bash
LEKAI_OFFLINE_LOG_DIR=logs/offline \
uv run python scripts/run_lekai_offline.py \
    --npz-dir prompts/inputs_lekai/npz \
    --temperature 0.8 \
    --top-k 1 \
    --condition-idx 0
```

### Comparing Logs

```bash
# Compare single files
python scripts/compare_generation_logs.py \
    --fake-rt-log logs/fake_rt/fake_rt_gen_0016.json \
    --offline-log logs/offline/offline_gen_0000.json \
    -v

# Compare entire directories
python scripts/compare_generation_logs.py \
    --fake-rt-dir logs/fake_rt/ \
    --offline-dir logs/offline/ \
    -v \
    -o comparison_result.json
```

## Log Format

Each log file is a JSON with the following structure:

```json
{
  "mode": "fake_rt",
  "timestamp": "2026-04-11T22:30:00",
  "input_file": "/path/to/input.mid",
  "generation_start_tick": 16,
  "generation_length_frames": 4,
  "prompt_tokens": [1, 4, 5, 173, 255, 255, ...],
  "prompt_token_count": 50,
  "temperature": 0.8,
  "top_k": 1,
  "top_p": 0.95,
  "repetition_penalty": 1.2,
  "melody_events": [
    {"type": "note_on", "pitch": 60, "tick": 0}
  ],
  "melody_event_count": 1,
  "accompaniment_events": [
    {"type": "note_on", "pitch": 48, "tick": 16, "velocity": 80}
  ],
  "accompaniment_event_count": 1,
  "bpm": 120,
  "notes": "additional context"
}
```

## Comparison Output

The comparison script outputs:

```
GENERATION LOG COMPARISON
============================================================

[General]
  Fake RT file:  /path/to/input.mid
  Offline file:  /path/to/input.npz

[Generation Parameters]
  Temperature match:      True
  Top-k match:            True
  Top-p match:            True
  Repetition penalty match: True

[Event Counts]
  Melody events:     FakeRT=10, Offline=10
  Acc events:        FakeRT=4, Offline=4

[Token Sequence Comparison]
  Fake RT length:    50
  Offline length:    50
  Length match:      True
  Prefix match len:  50
  Mismatch count:    0

[Special Token Analysis]
  Fake RT first 20 tokens:
    [1, 4, 5, 173, 255, 255, 10, 20, 30, 170, ...]
  Offline first 20 tokens:
    [1, 4, 5, 173, 255, 255, 10, 20, 30, 170, ...]

============================================================
SUMMARY
============================================================
✓ All checks passed! Logs match perfectly.
============================================================
```

## Known Issues

### 1. Prompt Structure Mismatch

**Issue**: FakeRT and Offline use different prompt structures at position 4:
- FakeRT: `bar` token at position 4
- Offline: `acc_{-1}` token at position 4 (with delay_beats=-1)

**Impact**: This causes the token sequences to diverge after position 3.

**Status**: Known limitation. The interleaving pattern differs between modes.

### 2. File Mapping

**Issue**: Offline mode uses NPZ files, FakeRT uses MIDI files. File ordering may differ:
- `os.listdir()` ordering doesn't match numeric file names
- Need explicit mapping between 1.mid and 1.npz

### 3. Generation Quality

**Issue**: Deterministic generation (top_k=1) produces sparse accompaniment with many empty beats.

**Workaround**: Use temperature > 0 with top_k=1 for better quality while maintaining reproducibility.

## Debugging Workflow

1. **Run both modes with same parameters**:
   ```bash
   # Use top_k=1 for deterministic output
   export LEKAI_RT_TOP_K=1
   ```

2. **Check logs are created**:
   ```bash
   ls -la logs/fake_rt/
   ls -la logs/offline/
   ```

3. **Compare token sequences**:
   ```bash
   python scripts/compare_generation_logs.py \
       --fake-rt-log logs/fake_rt/fake_rt_gen_0016.json \
       --offline-log logs/offline/offline_gen_0000.json \
       -v
   ```

4. **Analyze mismatches**:
   - Check first mismatch position
   - Examine special token distribution
   - Verify generation parameters match

## Testing

Run the test suite:

```bash
# Test all components
uv run python scripts/test_logging_system.py --mode all

# Test specific component
uv run python scripts/test_logging_system.py --mode logger
uv run python scripts/test_logging_system.py --mode compare
uv run python scripts/test_logging_system.py --mode env
```

## Future Improvements

1. **Align Prompt Structures**: Unify FakeRT and Offline prompt construction
2. **Event-Level Comparison**: Decode tokens to MIDI events for semantic comparison
3. **Audio Diff**: Generate audio from both modes and compare
4. **Metrics**: Add musical quality metrics (pitch histogram, rhythm patterns)
