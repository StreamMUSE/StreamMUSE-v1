# Injection Mode Report

Date: 2026-04-09
Related Plan: developing-logs/2026-4-9/injection-mode-plan.md

## 1. Scope

This report summarizes implementation and verification for the Injection Mode feature set:

1. CLI injection args and config mapping
2. CLI-side MIDI prompt parsing and injection flow
3. MidiFile input start_tick support (absolute tick continuation)
4. Input factory wiring for injection offset
5. Unit/integration test coverage and end-to-end smoke validation

---

## 2. Executive Summary

Injection mode has been implemented end-to-end for the first supported scope: `--input-mode midi_file`.

Delivered outcomes:

1. CLI supports `--injection-file`, `--injection-length`, `--inject-acc-file`.
2. Injection prompt is parsed client-side via `MidiFileInput._midi_to_notes()`, converted to `MusicalEvent`, then sent through `inject_history()`.
3. Midi file simulation can skip pre-injected prefix using `start_tick` while keeping absolute tick timing.
4. CLI enforces mode guard: injection is rejected for non-`midi_file` input modes.
5. Full test suite passes after implementation.

---

## 3. Implementation Details

## 3.1 Config and CLI Args

### Changes

1. Added new `InputConfig` fields:
   1. `injection_file: Optional[str]`
   2. `injection_length_ticks: int`
   3. `injection_acc_file: Optional[str]`
2. Added CLI args:
   1. `--injection-file`
   2. `--injection-length`
   3. `--inject-acc-file`
3. Mapped args to config with `getattr` fallback to avoid brittle test mocks.

### Key files

1. src/streammuse/application/config/models.py
2. src/streammuse/presentation/cli/config_parser.py

---

## 3.2 CLI Injection Flow

### Changes

1. Added `_notes_to_musical_events()` and reused domain `Note.to_events()`.
2. Added `_perform_injection()`:
   1. melody MIDI parse
   2. optional acc parse (`/mel/` -> `/acc/` convention)
   3. `clear_history()` then `inject_history()`
3. Updated `main()` sequencing:
   1. validate injection config
   2. create inference engine
   3. perform injection
   4. create input source
4. Added explicit restriction:
   1. if `injection_file` is set and input mode is not `midi_file`, CLI exits with error.

### Key file

1. src/streammuse/presentation/cli/cli.py

---

## 3.3 MidiFile Input Offset Support

### Changes

1. Added `start_tick` to `MidiFileInputConfig`.
2. Updated `read_events()` to filter notes with `note.tick < start_tick`.
3. Kept schedule timing absolute (`onset = note_tick + delay_ticks`) without subtracting start tick.

### Key file

1. src/streammuse/infrastructure/input/midi_file.py

---

## 3.4 Factory Wiring

### Changes

1. In midi_file branch, passed:
   1. `start_tick = injection_length_ticks` when `injection_file` is present
   2. otherwise `start_tick = 0`

### Key file

1. src/streammuse/application/factories/input_factory.py

---

## 4. Test Verification

## 4.1 Focused tests for this feature

Command:

`uv run pytest tests/unit/presentation/test_cli_config_parser.py tests/unit/infrastructure/input/test_midi_file_input.py tests/integration/test_cli_entry_point.py -q`

Result:

1. 15 passed
2. 0 failed

Coverage highlights:

1. Injection args mapped to `InputConfig`
2. `start_tick` filtering and absolute-time scheduling behavior
3. non-`midi_file` injection rejection
4. `clear_history()` is called before `inject_history()`
5. input source is created after successful injection

## 4.2 Full regression suite

Command:

`uv run pytest tests/ -q --tb=short`

Result:

1. 172 passed
2. 0 failed

---

## 5. End-to-End Smoke Validation

Environment:

1. Started fake inference server (`scripts/fake_inference_server.py`)
2. Ran CLI with injection + midi_file input

Command:

`uv run streammuse-cli --input-mode midi_file --midi-file-path prompts/inputs_lekai/mel/1.mid --injection-file prompts/inputs_lekai/mel/1.mid --injection-length 16 --output-type console --max-ticks 24`

Observed evidence:

1. Injection completed with console output:
   1. `Loaded accompaniment: 5 notes`
   2. `Injected: 5 melody notes, 5 acc notes`
2. Tick stream progressed normally.
3. First user events appeared at tick 16, matching injected prefix skip behavior.

Conclusion:

Injection mode works in the intended first-phase scope (`midi_file` input).

---

## 6. Files Changed

1. src/streammuse/application/config/models.py
2. src/streammuse/presentation/cli/config_parser.py
3. src/streammuse/presentation/cli/cli.py
4. src/streammuse/infrastructure/input/midi_file.py
5. src/streammuse/application/factories/input_factory.py
6. tests/unit/presentation/test_cli_config_parser.py
7. tests/unit/infrastructure/input/test_midi_file_input.py
8. tests/integration/test_cli_entry_point.py

---

## 7. Acceptance Mapping

1. CLI args added and exposed: DONE
2. Prompt parsing via `MidiFileInput._midi_to_notes()`: DONE
3. `inject_history()` call path with `MusicalEvent` payload: DONE
4. `start_tick` skip behavior for midi file input: DONE
5. Absolute tick continuation behavior: DONE
6. Non-midi mode guard for injection: DONE
7. Regression safety via unit/integration/full-suite tests: DONE

---

## 8. Residual Notes

1. Current design intentionally limits injection support to `midi_file` mode.
2. `_injection_offset_ticks` in HTTP client remains state-only and is not needed by this absolute-tick plan.
3. Backend history trimming remains unchanged; with default lower bound (`max(512, generation_length_frames * 16)`), typical injection lengths (16-32) are safe.
