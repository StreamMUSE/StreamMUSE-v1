# Plain Lekai Device/MIDI A/B/C Acceptance

Date: 2026-09-09. Diagnostic only; no production source changes in this audit.

The user requested a separate ordinary Realtime / pure Lekai audit before any user10/test40 rerun. This is not P+C single N=1.

## Frozen Run

- Production source: `ec843a79`; same-tick encoder fix is included. P+C closed-window changes do not extend to `RealTimeMusicService`.
- A: automatic private virtual MIDI input on Spark. B: A's exported Melody MIDI through the ordinary MIDI-file input. C: repeat B.
- Source: `prompts/old_input/mel/001.mid`, first 24 beats relative to first note_on.
- Playback/model BPM: 90/80; 4 steps/beat; 4/4; 4-beat count-in.
- Standard/sliding-window, generation interval/length 4/4 ticks, stop tick 160 exclusive.
- Seed: 1051154023138951872, reset using the existing debug API for every run.
- Sampling: temperature 1.1, top-p 0.95, top-k 50, repetition penalty 1.
- Continuation checkpoint: epoch_15_0307_1858; no Prompt checkpoint, no tonal/empty-token constraints.
- H200 GPU0 only; Spark CPU virtual MIDI client. No physical audio test. Observers synchronize tensors, so this is not a latency benchmark.

## Results

Both A/B and B/C match all 40 generation requests (ticks 4..160, step 4): actual encoded Melody roll bytes/tokens, full input tokens, output tokens and decoded raw events. All 235 sampling steps match logits, RNG state and selected tokens. The raw event sequence has 183 events, including 93 note-ons, identical across A/B/C.

A/B playback is NOT exact. A/B/C each export 90 accompaniment notes; B/C match all 90, but A has delayed onset/offset ticks. For example, the identical raw note_on@16 is scheduled at tick18 in A (`clamped_open_note`) and tick16 in B/C (`future_open_note`). The final 3 raw note-ons at tick160 are outside the playback window and must not be counted as in-window loss.

Raw input event equality is separate: A has 66 quantized events, B/C 34. A/B raw per-request events match only 21/40, whereas actual encoded model inputs match 40/40. Do not conflate these comparisons.

This is one automatic-device case and one seed, NOT a general human exact-replay guarantee. Ordinary Realtime's original beat-tail submission logic remains unchanged. No pure Lekai production fix or full cohort rerun was performed.

## Artifacts

- Local: `F:/repos/StreamMUSE-v1/remote_results/plain_lekai_device_replay_20260909`.
- H200: `/data/home/yuanxin/experiments/plain_lekai_device_replay_20260909`.
- Spark: `/home/xiaosongma/experiments/plain_lekai_device_replay_20260909`.
- Harness/analyzer: `scripts/plain_lekai_device_replay_acceptance.py`.
- Tests: `tests/test_plain_lekai_acceptance_export.py`.
- Detailed result, evidence paths and recalculation command: local artifact `README.md`.

The temporary H200 server was terminated after collection. Existing Spark Web/backend sessions were not changed.
