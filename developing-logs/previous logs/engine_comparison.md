# Inference Engine Data Structure Comparison

This document compares the data structures and input formats used by the two inference engines: **Stanley (RoFormer)** and **Lekai (LLaMA)**.

## 1. High-Level API Input (Identical)

Both engines implement the same `InferenceEngine` **domain protocol**. They are designed to be interchangeable at the application level.

**Protocol Signature (`domain/interfaces/inference.py`):**
```python
def generate_accompaniment(
    self,
    melody_events: List[MusicalEvent],
    generation_start_tick: int,
    generation_length_frames: int,
    prompt_length_ticks: int | None = None,
) -> tuple[List[MusicalEvent], TimingInfo]:
```

**Canonical Input Format:**
Both engines receive and return `MusicalEvent` objects (the shared domain type):

| Field | Type | Description |
| :--- | :--- | :--- |
| `tick` | `int` | Absolute start time in ticks. |
| `pitch` | `int` | MIDI pitch (0-127). |
| `event_type` | `EventType` | `NOTE_ON` or `NOTE_OFF`. |
| `velocity` | `int` | 0–127. |
| `channel` | `int` | MIDI channel. |
| `program` | `int` | MIDI program (instrument). |
| `source` | `str` | `"user"` or `"model"`. |

**Adapter Note:** The Stanley engine (`StanleyInferenceEngine`) wraps a legacy implementation that operates on `dict[pitch, tick, duration]` format. The adapter converts `MusicalEvent` → dict on input and dict → `MusicalEvent` on output. Callers always work with `MusicalEvent` only.

**Time Resolution:**
*   Both engines typically operate on a grid of **4 ticks per beat** (16th notes).
*   The `tick` values are integers based on this resolution.

---

## 2. Internal Data Representation (Major Differences)

While the input is the same, the engines process this data very differently internally to prepare it for their respective models.

### A. Stanley Engine (RoFormer / Symbolic)

The Stanley engine converts the note list into a **Symbolic Tensor** with fixed dimensions.

*   **Representation:** `(Time, Polyphony, Features)` tensor.
*   **Shape:** `(Sequence_Length, Max_Polyphony, 3)`
*   **Features:** Each note slot contains `[Program, Pitch, Duration_Index]`.
*   **Constraints:**
    1.  **Max Polyphony:** Hard limit (default 4). If more than 4 notes occur at the same tick, **excess notes are dropped**.
    2.  **Duration Templates:** Durations are not stored as raw integers. They are quantized to the nearest value in a predefined `DURATION_TEMPLATES` list (e.g., 1, 2, 4, 8, 16...). This means unusual durations might be snapped to standard musical lengths.

### B. Lekai Engine (LLaMA / Piano Roll -> Token)

The Lekai engine converts the note list into a **Piano Roll**, which is then tokenized.

*   **Representation:** **Piano Roll** image slices converted to **Tokens**.
*   **Intermediate Shape:** `(2, 88, Time)` numpy array.
    *   Channel 0: Sustain (1 if note is active).
    *   Channel 1: Onset (1 if note starts).
*   **Tokenization:** The piano roll is compressed into a sequence of integer tokens (e.g., `[BOS, TimeSig, BPM, Bar, Mel_Tokens, Bar, Acc_Tokens...]`).
*   **Constraints:**
    1.  **Polyphony:** No hard limit on the number of simultaneous notes (up to 88, one per pitch). It does **not** drop notes based on polyphony count.
    2.  **Duration:** Durations are represented by the length of the "Sustain" line in the piano roll grid. It supports any integer duration that fits the grid resolution.

## 3. Summary of Differences

| Feature | Stanley Engine (RoFormer) | Lekai Engine (LLaMA) |
| :--- | :--- | :--- |
| **Input Format** | `List[MusicalEvent]` (adapter converts to dicts internally) | `List[MusicalEvent]` (adapter converts to piano roll internally) |
| **Polyphony Handling** | **Limited** (Max 4 notes/tick). Drops excess. | **Flexible** (Implicitly up to 88). |
| **Duration Handling** | **Quantized to Templates** (Snaps to nearest standard duration). | **Grid-based** (Exact integer ticks on grid). |
| **Model Input** | Dense Tensor `(T, P, 3)` | Sequence of Tokens (Integers) |
| **Context Window** | Fixed frame length (e.g., 384 frames). | Token sequence length (e.g., 32 beats). |

## Conclusion

*   **Compatibility:** You can pass the exact same data to both engines without changing your application code.
*   **Fidelity:** The **Lekai** engine is theoretically capable of representing more complex data (higher polyphony, non-standard durations) because it doesn't force notes into fixed polyphony slots or duration templates.
