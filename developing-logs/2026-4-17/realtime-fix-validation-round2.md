# Realtime Lekai Validation Round 2 (2026-04-17)

## Goal
Verify whether the latest realtime regressions are fixed, and if not, continue diagnosis and implement targeted fixes.

## Scope
- Endpoint: http://127.0.0.1:8002/generate_accompaniment
- Model: models/ModelLekai/epoch_4_1104_1204/model.safetensors
- Inputs: prompts/old_input/mel/{002.mid,003.mid,005.mid}
- Settings: tempo=120, generation_interval_ticks=4, generation_length_frames=4, max_ticks=256
- Runner: scripts/run_lekai_batch_client.py

## Diagnostic Timeline
1. Re-ran key cases after prompt-order change.
2. Found partial improvement (002/005) but stable degradation on 003.
3. Added a hybrid order variant (history acc->mel, current beat mel-first) and tested.
4. Hybrid did not solve 003 enough and hurt 005 consistency.
5. Implemented fallback decoding in backend:
   - Keep mel-first as primary decode for current beat.
   - If primary decode for the beat is empty, retry with legacy prompt (without current-beat melody token) for that beat.
   - If fallback beat is non-empty, use fallback output and update sequence in legacy interleave style for that beat.
6. Re-ran 002/003/005 and repeated trials for stability.

## Code Changes
### 1) System-side scheduling (already in place before this round)
- File: src/streammuse/application/services/real_time_music_service.py
- Effect: late model events are no longer dropped; they are scheduled at current tick.

### 2) Backend-side decoding improvements (this round)
- File: src/streammuse/infrastructure/inference/lekai_http_backend.py
- Main adjustments:
  - Historical context uses mel->acc interleave.
  - Current beat primary path: inject melody first, then generate accompaniment.
  - Empty-beat fallback: retry generation with legacy prompt and select non-empty result when available.

## Quantitative Results
Metric definition:
- req: number of inference requests in session
- nonzero: requests with accompaniment_notes_count > 0
- resp_total: sum(accompaniment_notes_count)

### Song 002
- pre (n=4): req avg 42.0, nonzero avg 1.2, resp_total avg 4.0, min-max 0-16
- promptfix (n=4): req avg 41.8, nonzero avg 17.8, resp_total avg 55.8, min-max 11-93
- fallback (n=3): req avg 41.7, nonzero avg 30.7, resp_total avg 80.7, min-max 40-123

### Song 003
- pre (n=1): req 27.0, nonzero 22.0, resp_total 108.0
- promptfix (n=4): req avg 27.0, nonzero avg 4.5, resp_total avg 7.8, min-max 3-15
- fallback (n=3): req avg 26.7, nonzero avg 16.3, resp_total avg 49.3, min-max 24-84

### Song 005
- pre (n=1): req 38.0, nonzero 21.0, resp_total 57.0
- promptfix (n=1): req 38.0, nonzero 23.0, resp_total 64.0
- fallback (n=3): req avg 37.0, nonzero avg 27.0, resp_total avg 71.7, min-max 46-88

## Interpretation
- The fallback patch consistently improves 002 and 005 over both pre and promptfix baselines.
- 003 is significantly recovered from promptfix collapse and moves much closer to pre behavior.
- The issue is no longer in the "mostly empty" failure mode for the tested cases.

## Remaining Risk
- 003 still shows variance and is below the single historical best run (108).
- Further work can optimize quality stability (seed control, decode policy tuning, or selecting better of two candidates by richer criteria than non-empty only).

## Artifacts
- Fallback outputs:
  - output/lekai_batch_fallback120_focus_002
  - output/lekai_batch_fallback120_focus_003
  - output/lekai_batch_fallback120_focus_005
  - output/lekai_batch_fallback120_focus_002_repeat1
  - output/lekai_batch_fallback120_focus_002_repeat2
  - output/lekai_batch_fallback120_focus_003_repeat1
  - output/lekai_batch_fallback120_focus_003_repeat2
  - output/lekai_batch_fallback120_focus_005_repeat1
  - output/lekai_batch_fallback120_focus_005_repeat2

## Current Verdict
For the investigated regressions, the new backend fallback strategy is an effective fix direction and should be kept as the current default while continuing quality stabilization work.
