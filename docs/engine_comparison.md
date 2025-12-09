# Inference Engine Data Structure Comparison

This document compares the data structures and input formats used by the two inference engines: **Stanley (RoFormer)** and **Lekai (LLaMA)**.

## 1. High-Level API Input (Identical)

Both engines share the same signature for the main generation method `generate_accompaniment`. They are designed to be interchangeable at the application level.

**Method Signature:**
```python
def generate_accompaniment(
    self,
    melody_notes: List[Dict],
    generation_start_tick: int,
    acc_notes: List[Dict] = None,
    ...
)
```

**Input Data Format (`melody_notes` / `acc_notes`):**
Both engines expect a list of dictionaries, where each dictionary represents a note:

| Key | Type | Description |
| :--- | :--- | :--- |
| `pitch` | `int` | MIDI pitch (0-127). |
| `tick` | `int` | Absolute start time in ticks. |
| `duration` | `int` | Duration in ticks. |

**Time Resolution:**
*   Both engines typically operate on a grid of **4 ticks per beat** (16th notes).
*   The `tick` and `duration` values are integers based on this resolution.

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
| **Input Format** | List of Dicts `{'pitch', 'tick', 'duration'}` | List of Dicts `{'pitch', 'tick', 'duration'}` |
| **Polyphony Handling** | **Limited** (Max 4 notes/tick). Drops excess. | **Flexible** (Implicitly up to 88). |
| **Duration Handling** | **Quantized to Templates** (Snaps to nearest standard duration). | **Grid-based** (Exact integer ticks on grid). |
| **Model Input** | Dense Tensor `(T, P, 3)` | Sequence of Tokens (Integers) |
| **Context Window** | Fixed frame length (e.g., 384 frames). | Token sequence length (e.g., 32 beats). |

## Conclusion

*   **Compatibility:** You can pass the exact same data to both engines without changing your application code.
*   **Fidelity:** The **Lekai** engine is theoretically capable of representing more complex data (higher polyphony, non-standard durations) because it doesn't force notes into fixed polyphony slots or duration templates.
