# INFERENCE_SPEC.md — Inference Pathways & HTTP Contract

**Audited commit:** `05fc2fa`
**Location:** `src/streammuse/infrastructure/inference/`
**Project directive (2026-04-24):** **Lekai is the primary inference pathway.** Stanley is retained as a secondary / historical path. The code default in `InferenceConfig.model_name` is still `"stanley"` — see `progress.txt` KNOWN BUGS for the drift.

---

## 1. Pathway Overview

```
┌──────────────────────────────────────────────────────────────────┐
│  Application layer calls inference_engine.generate_accompaniment │
└──────────────────────────────────────────────────────────────────┘
                            │
               ┌────────────┴────────────┐
               │                         │
               ▼                         ▼
    ┌──────────────────────┐   ┌─────────────────────────┐
    │ HttpInferenceClient  │   │ StanleyInferenceEngine  │  (in-process)
    │ (InferenceConfig     │   │ (InferenceConfig        │
    │  .type == "http")    │   │  .type == "stanley")    │
    └──────────────────────┘   └─────────────────────────┘
               │                         │
               │                         ▼
               │             ┌───────────────────────────┐
               │             │ LegacyInferenceEngine     │
               │             │ Stanley (RoFormer wrapper)│
               │             └───────────────────────────┘
               │                         │
               │                         ▼
               │             ┌───────────────────────────┐
               │             │ stanley_stack/            │
               │             │   m2a_transformer.py      │
               │             │   (RoFormerSymbolic…)     │
               │             └───────────────────────────┘
               │
               │  HTTP POST, routed server-side by `model_name`
               ▼
     ┌─────────────────────────────────────────────────────┐
     │ Remote Inference Server                             │
     │  ┌──────────────────┐   ┌──────────────────┐        │
     │  │ server_lekai.py  │   │ scripts/         │        │
     │  │ (PRIMARY)        │   │ fake_inference_  │        │
     │  │ + lekai_model/   │   │ server.py (dev)  │        │
     │  └──────────────────┘   └──────────────────┘        │
     └─────────────────────────────────────────────────────┘
```

---

## 2. HTTP Contract (Frozen)

The HTTP contract is spoken by `HttpInferenceClient` and **must** be honored by any server that wants to participate. `scripts/fake_inference_server.py` is the reference echo implementation — keep it in sync with any change to this spec.

### 2.1 Endpoints

| Method | Path | Purpose |
|---|---|---|
| POST | `/generate_accompaniment` | Generate accompaniment for a melody slice |
| POST | `/inject_notes` | Pre-populate model history with melody + accompaniment |
| POST | `/clear_history` | Reset model history to empty |
| GET | `/injection_status` | Query whether injection has been applied |
| GET | `/health` | Liveness probe |

Default base URL: `http://localhost:8000`

### 2.2 `POST /generate_accompaniment`

**Request:**
```json
{
  "melody_notes": [
    {
      "type": "note_on",           // or "note_off"
      "pitch": 60,                 // 0..127
      "tick": 42,
      "velocity": 100,             // optional, default 100
      "channel": 0,                // optional, default 0
      "program": 0,                // optional, default 0
      "is_placeholder": false      // optional
    }
  ],
  "generation_start_tick": 42,
  "client_request_send_time": 1746115200.123,
  "generation_length_frames": 20,
  "generation_interval_ticks": 2,
  "model_name": "lekai",           // "lekai" (primary) | "stanley"
  "inference_mode": "sliding_window",
  "checkpoint_path": null,
  "prompt_length_ticks": null
}
```

**Response:**
```json
{
  "accompaniment": [
    {
      "type": "note_on",
      "pitch": 64,
      "tick": 42,
      "velocity": 100,
      "channel": 1,
      "program": 0,
      "is_placeholder": false
    }
  ],
  "timings": {
    "request_arrival_time": 1746115200.130,
    "response_output_time": 1746115200.145,
    "preprocess_start_time": 1746115200.131,
    "inference_start_time": 1746115200.133,
    "inference_end_time": 1746115200.144,
    "postprocess_start_time": 1746115200.144,

    "round_trip_time": 0.022,                 // optional, client-computed
    "server_processing_duration": 0.015,      // optional, server-computed
    "total_network_latency": 0.007            // optional
  },
  "generation_start_tick": 42
}
```

### 2.3 `POST /inject_notes`

**Request:**
```json
{
  "melody_notes":         [ /* same shape as above */ ],
  "accompaniment_notes":  [ /* same shape as above */ ],
  "injection_length_ticks": 50
}
```

**Response:**
```json
{ "success": true, "melody_count": 32, "accompaniment_count": 28 }
```

### 2.4 `GET /injection_status`

**Response:**
```json
{ "is_injected": true, "injection_offset_ticks": 50 }
```

### 2.5 `POST /clear_history`

**Response:**
```json
{
  "success": true,
  "message": "History cleared",
  "melody_history": [],          // optional
  "accompaniment_history": []    // optional
}
```

### 2.6 `GET /health`

**Response:**
```json
{ "status": "healthy", "server": "lekai" }
```
(`"server": "fake"` for the dev echo server; `"stanley"` for a Stanley HTTP server.)

---

## 3. `HttpInferenceClient`

File: `infrastructure/inference/http_client.py`

```python
class HttpInferenceClient:
    def __init__(self, config: HttpInferenceClientConfig): ...

    def generate_accompaniment(
        self,
        melody_events: List[MusicalEvent],
        generation_start_tick: int,
        generation_length_frames: int,
        prompt_length_ticks: int | None = None,
    ) -> tuple[List[MusicalEvent], TimingInfo]:
        # 1. payload.melody_notes = [event_to_dict(e) for e in melody_events]
        # 2. requests.post(config.generate_url, json=payload, timeout=config.timeout_s)
        # 3. accompaniment = [event_from_dict(d) for d in resp["accompaniment"]]
        # 4. timing = timing_info_from_dict(resp["timings"])
        # 5. return (accompaniment, timing)

    def inject_history(
        self,
        melody_events: List[MusicalEvent],
        accompaniment_events: List[MusicalEvent],
        injection_length_ticks: int,
    ) -> None: ...

    def set_injection_offset(self, offset_ticks: int) -> None: ...

    def clear_history(self) -> Dict[str, Any]: ...
```

Configuration (`HttpInferenceClientConfig`) carries `generate_url`, `timeout_s`, `model_name`, `inference_mode`, `generation_length_frames`, and `generation_interval_ticks`.

### 3.1 Serialization

File: `infrastructure/inference/serialization.py`

- `event_to_dict(event: MusicalEvent) -> dict` — emits the `"type": "note_on"|"note_off"` JSON shape above.
- `event_from_dict(d: dict) -> MusicalEvent` — inverse; fills defaults for missing optional fields.
- `timing_info_from_dict(d: dict) -> TimingInfo` — tolerant of missing client-side fields (they're all optional).

No schema library is used here (hand-rolled). If you add Pydantic validation server-side, match the field types exactly.

---

## 4. Lekai Pathway (PRIMARY)

### 4.1 Server

File: `src/streammuse/infrastructure/inference/server_lekai.py`

FastAPI app that satisfies the contract above. Backed by `lekai_model/` (see below).

Launch:
```bash
uvicorn streammuse.infrastructure.inference.server_lekai:app --host 0.0.0.0 --port 8000
```

### 4.2 Model Package

Location: `src/streammuse/infrastructure/inference/lekai_model/`

| File | Role |
|---|---|
| `model.py` | Model architecture definition |
| `inference.py` | Generation loop (sampling, top-k, nucleus) |
| `inference_adapter.py` | Adapts the raw model output to the HTTP contract shape |
| `my_tokenizer.py` | Custom token vocabulary |
| `MidiConverter.py` | MIDI ↔ token conversion at serve time |
| `Token2Midi.py` | Inverse (token → MIDI) utility |
| `PianoDataset.py` | Dataset utilities (shared with training repo) |
| `generation_utils.py` | Sampling helpers |
| `config.py` | Model hyperparameters |

### 4.3 Client-Side Adapter

File: `infrastructure/inference/lekai_http_backend.py`

HTTP backend wrapper that can be used when the inference type is specialized per model. Today this is exercised by the batch / benchmark scripts in `scripts/`:

- `scripts/run_lekai_offline.py` — offline batch inference
- `scripts/run_lekai_batch_client.py` — batched client
- `scripts/run_lekai_fake_realtime.py` — realtime simulation using Lekai
- `scripts/benchmark_lekai_http.py` — throughput/latency benchmark

---

## 5. Stanley Pathway (SECONDARY / HISTORICAL)

### 5.1 `StanleyInferenceEngine` — Event ↔ Note Adapter

File: `infrastructure/inference/stanley_engine.py`

Implements the `InferenceEngine` protocol directly (no HTTP). Responsibilities:

1. Convert incoming `melody_events` into duration-note dicts via `events_to_notes(events, horizon_tick=generation_start_tick)`.
2. Delegate to the legacy engine.
3. Convert returned duration-notes back into `MusicalEvent` streams (`note_on` + `note_off` pair per note).
4. Wrap server timings into a `TimingInfo`.

```python
class StanleyInferenceEngine:
    def __init__(
        self,
        config: StanleyInferenceConfig,
        legacy_engine: _LegacyStanleyLike | None = None,
    ): ...
    def generate_accompaniment(...) -> tuple[list[MusicalEvent], TimingInfo]: ...
    def inject_history(...) -> None: ...
    def clear_history(self) -> dict[str, Any]: ...
```

### 5.2 `LegacyInferenceEngineStanley` — RoFormer Wrapper

File: `infrastructure/inference/stanley_legacy.py`

```python
class LegacyInferenceEngineStanley:
    def __init__(
        self,
        checkpoint_path: str,
        model_size: str,                  # "0.12B" (default) etc.
        generation_length_frames: int,
        max_polyphony: int = 4,
        model_max_seq_len_frames: int = 96,
    ): ...
    def generate_accompaniment(
        self,
        melody_notes: list[dict],          # [{"pitch", "tick", "duration", ...}]
        generation_start_tick: int,
        acc_notes: list[dict] | None = None,
        generation_length_frames: int | None = None,
        prompt_length_ticks: int | None = None,
    ) -> tuple[list[dict], float, float, float, float]: ...
```

Internals:
- Loads a `RoFormerSymbolicTransformer` checkpoint via `pytorch-lightning`.
- Moves to CUDA if available (MPS fallback on Apple Silicon; CPU otherwise — via `runtime_device.py`).
- Maintains `self.melody_history`, `self.accompaniment_history` as duration-note dicts.
- Tracks `self.injection_offset_ticks` for prompt alignment.
- **Piano-roll tensor layout**: `rolls[tick, voice, feature]` where `feature` ∈ {program, pitch, duration_idx}, `voice` ∈ `[0, max_polyphony)`. Durations are quantized against `DURATION_TEMPLATES`.
- `model.generate(...)` returns token tensors → converted back to duration-dicts.
- Return tuple: `(acc_notes, preprocess_start, inference_start, inference_end, postprocess_start)` — timestamps used by the adapter to build `TimingInfo`.

### 5.3 Model Architecture

File: `infrastructure/inference/stanley_stack/m2a_transformer.py`

- **RoFormer** (Rotary Positional Embedding). Uses the custom `transformers/` editable install for the positional encoding tweak.
- Two encoder stacks (melody context + accompaniment context) + one decoder.
- Max sequence length: 96 frames default, configurable.
- Max polyphony: 4 voices per tick.

### 5.4 Preprocessing (Offline)

Location: `infrastructure/inference/stanley_stack/preprocess/`

- `preprocess_midi2pt_dataset.py` — MIDI → PyTorch tensor dataset
- `xf_midi.py` — MIDI feature extraction
- `settings.py` — hyperparameters

These are used by the out-of-tree training repo; they live here only because the inference-side tokenization depends on the same format.

---

## 6. Fake Inference Server (Development Only)

File: `scripts/fake_inference_server.py`

FastAPI echo server for development without a model or GPU:

- `POST /generate_accompaniment`: returns `accompaniment = melody_notes` (verbatim echo) with a 10 ms `asyncio.sleep` to simulate work.
- `POST /inject_notes`: returns `{"success": true, "melody_count": N, "accompaniment_count": M}`.
- `GET /injection_status`: returns `{"is_injected": false}`.
- `POST /clear_history`: returns `{"success": true, "message": "History cleared"}`.
- `GET /health`: returns `{"status": "healthy", "server": "fake"}`.

Launch:
```bash
uv run python scripts/fake_inference_server.py
```

---

## 7. Device Selection

File: `infrastructure/inference/runtime_device.py`

Selects `cuda` / `mps` / `cpu` at load time, honoring `CUDA_VISIBLE_DEVICES` and availability. Apple Silicon builds default to MPS when PyTorch supports the required ops, otherwise CPU.

---

## 8. Non-Goals for This Layer

- No training code. Training lives in a separate repository.
- No checkpoint management. The service loads a single checkpoint on startup; rotation is external.
- No model zoo / registry. Model selection is a single string (`"stanley" | "lekai"`) on the HTTP request.
