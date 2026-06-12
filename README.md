# StreamMUSE

StreamMUSE is a real-time accompaniment generation system. It reads melody input from keyboard, MIDI devices, or MIDI files, sends incremental melody context to an inference backend, and plays or records generated accompaniment.

## Environment

```bash
uv sync
```

This creates a `.venv` environment in the repository.

## Real-time Application

The real-time system is a CLI client plus an optional inference server. The main code lives under `src/streammuse/` and follows a Presentation / Application / Domain / Infrastructure structure.

### Quick Start

```bash
# 1. Start the fake inference server (no model required)
uv run python scripts/fake_inference_server.py

# 2. In another terminal, run the CLI
uv run streammuse-cli --input-mode keyboard
```

### CLI Examples

```bash
# Input modes
uv run streammuse-cli --input-mode keyboard
uv run streammuse-cli --input-mode midi_device
uv run streammuse-cli --input-mode midi_file --midi-file-path path/to/song.mid

# Output types
uv run streammuse-cli --output-type console
uv run streammuse-cli --output-type audio --midi-out-port "My Synth"
uv run streammuse-cli --output-type composite --log-dir logs

# Metronome + count-in
uv run streammuse-cli \
    --input-mode midi_file \
    --midi-file-path prompts/inputs_lekai/mel/1.mid \
    --output-type console \
    --enable-metronome \
    --count-in-beats 4

# Music injection with MIDI-file input
uv run streammuse-cli \
    --input-mode midi_file \
    --midi-file-path prompts/inputs_lekai/mel/1.mid \
    --injection-file prompts/inputs_lekai/mel/1.mid \
    --injection-length 16
```

### Lekai HTTP Server

```bash
LEKAI_CHECKPOINT_PATH=path/to/lekai_checkpoint.safetensors \
LEKAI_DEVICE=auto \
LEKAI_DTYPE=auto \
uv run python -m streammuse.infrastructure.inference.server_lekai
```

Then run the client:

```bash
uv run streammuse-cli \
    --input-mode keyboard \
    --inference-type http \
    --model-name lekai \
    --server-url http://127.0.0.1:8000/generate_accompaniment
```

### Session Logging

Using `--output-type composite --log-dir logs` creates a timestamped session directory:

```text
logs/YYYY-MM-DD/session_HHMMSS/
├── events.jsonl
├── inferences.json
├── performance.json
├── statistics.csv
├── session_config.json
├── session_summary.txt
├── melody_history.json
├── accompaniment_history.json
└── combined.mid
```

`combined.mid` contains `Melody` and `Accompaniment` tracks. With `--enable-metronome`, it also contains a `Metronome` drum track. With `--count-in-beats`, count-in clicks are recorded at the beginning of the MIDI file.

## Documentation

Start from [`docs/index.md`](docs/index.md). Key pages:

- [`docs/getting-started/configuration.md`](docs/getting-started/configuration.md)
- [`docs/reference/cli-reference.md`](docs/reference/cli-reference.md)
- [`docs/user-guide/running-realtime.md`](docs/user-guide/running-realtime.md)
- [`docs/user-guide/music-injection.md`](docs/user-guide/music-injection.md)
- [`docs/architecture/application/service.md`](docs/architecture/application/service.md)

## Dataset / Training Notes

Legacy dataset preparation, preprocessing, training, and inference scripts remain in the repository. See the existing scripts and historical logs under `developing-logs/` for experiment-specific workflows.
