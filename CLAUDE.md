# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

StreamMUSE is a real-time AI music generation system that creates accompaniment for user-played melodies. The system uses a transformer-based model (RoFormer) and consists of a client-server architecture where the client handles user input and audio output, while the server performs model inference.

## Development Commands

### Environment Setup
```bash
# Set up environment and install dependencies
uv run

# Install the adapted transformers package (required for special positional encoding)
cd transformers
pip install -e .
```

### Training
```bash
# Configure model parameters in schema/yaml/ files first
uv run training_runner.py
```

### Real-time Application

#### Server (Model Inference)
```bash
# Start inference server
CHECKPOINT_PATH=path/to/model.ckpt uvicorn app.server:app --host 0.0.0.0 --port 8000

# Optional environment variables:
# MODEL_MAX_SEQ_LEN_FRAMES=96 (default)
# GENERATION_LENGTH_FRAMES=20 (default) 
# MODEL_SIZE=0.12B (default, valid: small, 0.12B, 0.25B, 0.5B)
```

#### Client (User Interface)
```bash
# MIDI device input (default)
python app/client.py --server_url http://localhost:8000/generate_accompaniment

# Computer keyboard input
python app/client.py --use-keyboard-input --tempo 120

# MIDI file simulation
python app/client.py --midi-file-input song.mid --midi-file-delay-ticks 8

# With music injection (prefill model history)
python app/client.py --injection-file prelude.mid --injection-length 50 --use-keyboard-input
```

### Benchmarking
```bash
# Performance testing
python app/benchmark.py --output_file results/benchmark_test.csv --num_requests 100
```

### Data Preprocessing
```bash
# Extract melody/accompaniment from datasets
python exact/aria_skyline.py --input_dir <dataset_path> --output_dir <output_path> --workers <num_cores>
python exact/pop909_extract.py  # For POP909 dataset

# Convert MIDI to tensor format
python preprocess/preprocess_midi2pt_dataset.py --folders /path/to/folder1 --name dataset_name --polyphony 4
```

### Inference Testing
```bash
# Offline inference testing
python m2a_transformer_inference.py --model_path /path/to/ckpt --prompt_len 75 --n_samples 2 --temperature 1.0
```

## Architecture Overview

### Client-Server Communication
- **FastAPI Server** (`app/server.py`): Hosts transformer model, provides `/generate_accompaniment` endpoint
- **Client** (`app/client.py`): Manages user input, timing, audio output, communicates with server
- **Communication Protocol**: JSON over HTTP with detailed timing information

### Key Data Models
- **MelodyNoteEvent**: `{pitch: int, tick: int, duration: int}`
- **AccompanimentNoteEvent**: `{pitch: int, tick: int, duration: int, program: int}`
- **Timings**: Detailed server-side performance metrics using `time.perf_counter()`

### Real-time System Components

#### Input Handlers (`app/input_handlers/`)
- MIDI device input
- Computer keyboard input 
- MIDI file simulation

#### Output Handlers (`app/output_handlers/`)
- Real-time audio playback
- MIDI file recording
- JSON logging with timing data
- CLI display

#### Inference Engine (`app/inference_engines/`)
- **TransformerInferenceEngine**: Standard implementation
- **InferenceEngineStanley**: Enhanced version with injection support
- Manages model history, note quantization, tensor conversion

### Timing and Synchronization
- **Tick-based timing**: Musical time quantized to ticks (1 tick = 1/4 beat by default)
- **Generation intervals**: Model generates accompaniment every N ticks (default: 2)
- **Latency compensation**: Requests sent early to account for processing delay
- **History management**: Maintains sliding window of melody/accompaniment history

### Model Integration
- **RoFormer-based transformer**: Uses interleaved melody/accompaniment frames
- **Sequence length**: 96 frames (48 ticks) default context window
- **Generation length**: 20 frames (10 ticks) per inference call
- **Note representation**: Piano roll tensors converted to symbolic tokens

## Benchmarking System

The benchmark system (`app/benchmark.py`) measures:
- **Round-trip latency**: Client request to response time
- **Server processing**: Total server-side processing duration
- **Inference time**: Pure model inference duration
- **Network latency**: Time spent on network communication
- **Preprocessing/Postprocessing**: Data conversion overhead

Results saved as CSV (timing summary) and JSON (complete response data).

## Music Injection Feature

The system supports "music injection" to pre-populate model history:
- Load existing MIDI files into model context
- Useful for continuing compositions or style transfer
- Injection endpoints: `/inject_music`, `/injection_status`
- Client support: `--injection-file` and `--injection-length` parameters

## Data Pipeline

1. **Raw datasets**: POP909, ARIA MIDI collections
2. **Extraction**: Separate melody/accompaniment tracks using skyline algorithm
3. **Preprocessing**: Convert MIDI to tensor format with quantized durations
4. **Training**: RoFormer model on melody→accompaniment pairs
5. **Inference**: Real-time generation with sliding context window

## File Structure Focus

- `app/`: Real-time application (client/server)
- `models/`: PyTorch Lightning model definitions
- `preprocess/`: Data preprocessing pipeline
- `exact/`: Dataset extraction utilities
- `inference_engines/`: Model inference implementations
- `schema/yaml/`: Model configuration files
- `transformers/`: Modified transformers library for RoFormer