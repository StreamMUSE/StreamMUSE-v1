# StreamMUSE Application

This directory contains the StreamMUSE real-time music generation application, consisting of a server-side inference engine and a client-side interactive system.

## Architecture Overview

The system uses a client-server architecture where:
- **Server**: Hosts the transformer model and handles music generation requests
- **Client**: Manages user input, timing, and audio output while communicating with the server

## Server-Client Communication Interface

### Server Endpoints

#### 1. Generate Accompaniment
**Endpoint**: `POST /generate_accompaniment`

**Request Model** (`InferenceRequest`):
```json
{
  "melody_notes": [
    {
      "pitch": 60,
      "tick": 0,
      "duration": 4
    }
  ],
  "generation_start_tick": 16,
  "client_request_send_time": 1234567890.123
}
```

**Response Model** (`AccompanimentResponse`):
```json
{
  "accompaniment": [
    {
      "pitch": 65,
      "tick": 16,
      "duration": 4,
      "program": 0
    }
  ],
  "timings": {
    "request_arrival_time": 1234567890.124,
    "response_output_time": 1234567890.456,
    "preprocess_start_time": 1234567890.125,
    "inference_start_time": 1234567890.140,
    "inference_end_time": 1234567890.440,
    "postprocess_start_time": 1234567890.441
  },
  "generation_start_tick": 16
}
```

#### 2. Clear History
**Endpoint**: `POST /clear_history`

**Request**: No body required

**Response**:
```json
{
  "message": "History cleared successfully."
}
```

### Data Models

#### MelodyNoteEvent
Represents a user-played melody note.
- `pitch` (int): MIDI pitch number (0-127)
- `tick` (int): Timing position in ticks
- `duration` (int): Note duration in ticks

#### AccompanimentNoteEvent
Represents a generated accompaniment note.
- `pitch` (int): MIDI pitch number (0-127)
- `tick` (int): Timing position in ticks
- `duration` (int): Note duration in ticks
- `program` (int): MIDI program/instrument number

#### Timings
Detailed server-side timing information using `time.perf_counter()` timestamps.
- `request_arrival_time` (float): When server received the request
- `response_output_time` (float): When server finished processing
- `preprocess_start_time` (float): When preprocessing began
- `inference_start_time` (float): When model inference started
- `inference_end_time` (float): When model inference completed
- `postprocess_start_time` (float): When postprocessing began

### Timing Analysis

The client calculates additional timing metrics from the server response:

- **Server Processing Duration**: `response_output_time - request_arrival_time`
- **Preprocessing Duration**: `inference_start_time - preprocess_start_time`
- **Inference Duration**: `inference_end_time - inference_start_time`
- **Postprocessing Duration**: `response_output_time - postprocess_start_time`
- **Round Trip Time**: Total time from client request to response
- **Network Latency**: `round_trip_time - server_processing_duration`

### Error Handling

**HTTP 503 Service Unavailable**:
```json
{
  "error": "Inference engine not loaded"
}
```

**HTTP 422 Unprocessable Entity**:
Returned for validation errors in request data with detailed field-level error information.

## Usage Example

### Starting the Server
```bash
CHECKPOINT_PATH=path/to/model.ckpt uvicorn app.server:app --host 0.0.0.0 --port 8000
```

### Environment Variables
- `CHECKPOINT_PATH` (required): Path to the model checkpoint file
- `MODEL_MAX_SEQ_LEN_FRAMES` (optional, default=96): Maximum sequence length for the model
- `GENERATION_LENGTH_FRAMES` (optional, default=20): Number of frames to generate

### Running the Client
```bash
python app/client.py --server_url http://localhost:8000/generate_accompaniment
```

### Running Benchmarks
```bash
python app/benchmark.py --output_file results/benchmark_test.csv --num_requests 100
```

## Communication Flow

1. **Client** collects user input (MIDI or keyboard)
2. **Client** quantizes notes to tick boundaries
3. **Client** sends inference request at bar boundaries with latency offset
4. **Server** processes request through preprocessing → inference → postprocessing
5. **Server** returns generated accompaniment with detailed timing data
6. **Client** schedules and plays accompaniment notes
7. **Client** logs timing data for performance analysis

## Files

- `server.py`: FastAPI server with inference endpoints
- `client.py`: Interactive client with audio I/O and timing management
- `benchmark.py`: Performance testing tool
- `inference_engines/`: Model inference implementations
- `input_handlers/`: User input processing (MIDI, keyboard)
- `output_handlers/`: Audio output and logging systems 