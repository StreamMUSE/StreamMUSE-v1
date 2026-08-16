# Task 2 Report

Date: August 16, 2026

## Scope

Implemented Task 2 native timing controls in the `real_rap` worktree without renaming the Task 1 `TwoBarRenderRequest` or `SyllableTarget` contracts.

Files added:

- `src/streammuse/experiments/rap_audio_protocols/timing.py`
- `tests/unit/experiments/rap_audio_protocols/test_timing.py`

## TDD Evidence

Red:

- Ran `.venv/bin/pytest tests/unit/experiments/rap_audio_protocols/test_timing.py -q`
- Observed `ModuleNotFoundError: No module named 'streammuse.experiments.rap_audio_protocols.timing'`

Green:

- Added the timing module and kept the tests focused on the approved two-bar corpus request plus the required FastPitch mismatch failure mode.
- Re-ran `.venv/bin/pytest tests/unit/experiments/rap_audio_protocols/test_timing.py -q`
- Result: `4 passed`

## Implemented Behavior

- `moss_token_target(request)` returns `round(request.duration_seconds * 12.5)`.
- `build_ted_segments(request)`:
  - segments only on word boundaries
  - prefers phrase boundaries from `boundary_strength > 0`
  - falls back to one segment per bar when explicit phrase boundaries are unavailable
  - preserves original spacing by assigning inter-segment space to the preceding segment
- `build_fastpitch_phone_plan(request, tokenizer_labels)`:
  - aligns tokenizer labels to the exact ARPAbet phone sequence after removing blanks and spaces
  - fails closed on any phone/token mismatch
  - returns one mel-duration entry per tokenizer position, including zeros for blanks/spaces
  - keeps consonants at one frame and assigns residual timing budget to vowel positions
  - reports per-syllable anchor error and compressed consonant regions

## Notes

- The approved corpus fixture lands every FastPitch vowel center within one mel frame of the target anchors; the first syllable is off by one frame because its onset consonant cannot occur before time zero.
