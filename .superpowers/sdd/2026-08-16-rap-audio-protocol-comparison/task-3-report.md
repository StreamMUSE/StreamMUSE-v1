Status: implemented Task 3 in task-owned files only. Added common WAV normalization/resampling, exact two-bar vocal assembly, deterministic shared drum rendering, full-song mixing with shared peak limiting, canonical artifact manifests, and resumable chunk-record integrity checks.

Tests:
- `.venv/bin/pytest tests/unit/experiments/rap_audio_protocols/test_audio.py tests/unit/experiments/rap_audio_protocols/test_artifacts.py -q`
- Result: `7 passed`

Commit: `17953f61` (`feat(rap): assemble comparable offline audio artifacts`)

Concerns:
- Verification is focused at the unit-contract level for Task 3 only; no end-to-end batch script integration was exercised here.
- The manifest surface is sufficient for the brief and tests, but downstream consumers may still want a stricter schema once Task 2 wiring lands.
