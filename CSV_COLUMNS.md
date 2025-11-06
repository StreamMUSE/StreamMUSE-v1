# StreamMUSE Benchmark CSV Column Reference

## Data Flow
```
benchmark.py → server.py → inference_engine → server.py → benchmark.py
(measures RTT)  (timestamps)  (ML inference)   (response)   (RTT complete)
```

## Column Definitions

### Request Info
- **`request_id`** - Sequential request number (1, 2, 3, ...)
- **`generation_start_tick`** - Musical tick where generation begins (16)
- **`num_generated_notes`** - Number of accompaniment notes generated

### Key Performance Metrics
- **`round_trip_time`** - Total client request duration (seconds) ⭐ **Primary metric**
- **`inference_duration`** - Pure ML model execution time (seconds) ⭐ **Core bottleneck**
- **`server_processing_duration`** - Total server-side processing time (seconds)
- **`total_network_latency`** - Network overhead (RTT - server processing)

### Server Pipeline Breakdown
- **`preprocess_duration`** - Data preparation for model (seconds)
- **`postprocess_duration`** - Convert model output to notes (seconds)

### Raw Timestamps (server-side)
- **`server_request_arrival_time`** - Request received timestamp
- **`server_response_output_time`** - Response sent timestamp  
- **`server_preprocess_start_time`** - Preprocessing start
- **`server_inference_start_time`** - ML inference start
- **`server_inference_end_time`** - ML inference end
- **`server_postprocess_start_time`** - Postprocessing start

## Key Relationships
- `round_trip_time` ≈ `server_processing_duration` + `total_network_latency`
- `server_processing_duration` ≈ `preprocess_duration` + `inference_duration` + `postprocess_duration`
- **Real-time constraint**: `round_trip_time` < 125ms per tick for musical timing

## Components
- **`benchmark.py`** - Client timing measurement and request generation
- **`server.py`** - FastAPI server with detailed timestamp logging  
- **`inference_engines/`** - ML model execution and music generation