# lessons.md — Problem → Rule Log

**Format:** `Problem: [observation] -> Rule: [constraint to prevent recurrence]`

Read this at the start of every session. Append when a non-obvious problem produces a rule that future work must respect. Do not delete entries; supersede them with a newer entry if the constraint evolves.

Entries inferred from git history and `developing-logs/` at audit time (2026-04-24, commit `05fc2fa`). Later entries are session-authored.

---

## Inferred from Git History & developing-logs

**Problem:** The initial build of this repo mixed inference concerns across many loose scripts, making it hard to reason about what was actually on the realtime path.
**Rule:** Real-time application code lives under `src/streammuse/` in Clean Architecture layers. Scripts under `scripts/` are batch / dev / benchmark utilities and must not be imported by the service. If you need shared logic, move it into the right layer of `src/streammuse/`.

---

**Problem:** MIDI recording was repeatedly broken by changes elsewhere (commit `c9b770a` "try to fix midi recording"; multiple debug-plan / report pairs in `developing-logs/`).
**Rule:** `MidiFileOutputSink.close()` must pair every outstanding `NOTE_ON` with a `NOTE_OFF` before writing the file. If you touch the output layer, run the `tests/integration/test_simulator_midi_output.py` integration test and manually verify `combined.mid` contains the expected notes.

---

**Problem:** Apple Silicon developers could not install PyTorch with the default CUDA wheel pins (Mac adaptation commits `c9cfcc7`, `0e503f5`).
**Rule:** Platform-specific dependencies go behind `sys_platform` markers in `pyproject.toml`'s `[tool.uv.sources]`. Never hardcode CUDA wheels as unconditional requirements. Test `uv sync` on macOS before merging dependency changes.

---

**Problem:** The system appeared to generate correctly but fell seconds behind the user under load — inference requests were FIFO-queued and backlogging.
**Rule:** The inference worker uses **latest-only** coalescing: on wake-up, drain the request queue, keep the newest `generation_start_tick`, merge melody events. Do not revert to FIFO semantics without an explicit plan — this is a load-shedding invariant, not a bug.

---

**Problem:** A model inference response arriving late could overwrite user events that had already been scheduled.
**Rule:** `PlaybackScheduler.clear_future_events(from_tick, source="model")` must filter on source. User events are authoritative; only model events get invalidated by a newer inference response.

---

**Problem:** Unpaired `NOTE_ON` events at the end of a generation window produced hanging notes that never turned off.
**Rule:** Use `events_to_notes(events, horizon_tick)` (close-at-horizon policy) at every event-to-note conversion boundary. Never pass an unbounded stream through a duration-based representation.

---

**Problem:** Injection was accidentally applied alongside keyboard or MIDI-device input, producing corrupted history that confused the model.
**Rule:** Injection is only valid with `--input-mode midi_file`. The CLI validates this at startup and refuses to run otherwise. Do not relax this check without updating `INFERENCE_SPEC.md` and the injection tests.

---

**Problem:** The `transformers` package from upstream PyPI did not work with Stanley's custom RoFormer positional encoding (causes silent tokenization drift).
**Rule:** The local editable install at `transformers/` is load-bearing for the Stanley pathway. Do not replace it with upstream `transformers` even if versions appear compatible. Any upgrade must be re-patched.

---

**Problem:** Session-log directories flat-under-`logs/` made it painful to locate historical runs.
**Rule:** Session directories live at `logs/YYYY-MM-DD/session_HHMMSS/`. `SessionManager` also keeps the legacy flat layout readable for pre-existing sessions — do not delete the fallback logic until historical data is archived.

---

**Problem:** Stats and inference events were logged inconsistently, breaking analysis tooling.
**Rule:** Serialize every event through `infrastructure/inference/serialization.py` (`event_to_dict`). Every log consumer (JSON, WebSocket, inference HTTP) uses the same shape. Do not hand-roll JSON in new output sinks.

---

**Problem:** Training-time packages (tensorflow, wandb, etc.) in `pyproject.toml` made the install heavy for service-only users and confused new contributors about the repo's scope.
**Rule:** This repo is the inference service and realtime application. Training lives elsewhere. Do not add training code here even if it feels convenient. See `docs/audit/progress.txt` KNOWN BUGS for the open cleanup item.

---

**Problem:** The `README.md` fell out of sync with the actual codebase (training pipeline references, old directory names) and was trusted by new contributors as canonical.
**Rule:** The code is the truth; `docs/audit/` is the frozen map of that truth; `README.md` is a friendly on-ramp, not a spec. When they disagree, update the on-ramp — never code the on-ramp's claims.

---

<!--
  Append new entries below this line in session chronological order.
  Format:
  **Problem:** ...
  **Rule:** ...
-->
