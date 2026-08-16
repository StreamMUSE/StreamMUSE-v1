# Aligned MOSS Warp and Stress Experiment

Date: 2026-08-17  
Branch: `feature/real_rap_audio`  
Final implementation commit: `a419010b`

## Question

Can aligned MOSS vocals sound less glitchy while preserving the intended rap timing, and can the flow template's stress values be made audible?

The previous backend independently stretched about 36 short regions per two-bar chunk. Its first anchor also represented a point inside the aligned vowel, even when the requested rhythmic event was the syllable onset. A typical first syllable therefore mapped an internal vowel point around 235 ms to a target near 10 ms. Across 75 existing chunks, the baseline contained 2,700 local regions, including 374 regions below 0.33x or above 3x and a worst reciprocal distortion of 39x.

## Implementation

The legacy renderer remains the default. Four explicit experimental modes were added:

| Mode | Anchor semantics | Warp | Stress |
|---|---|---|---|
| `piecewise_vowel_r2` | point inside aligned vowel | one R2 process per region | none |
| `continuous_vowel_r3` | point inside aligned vowel | one full-chunk R3 time map | none |
| `continuous_onset_r3` | aligned syllable onset where consonants match | one full-chunk R3 time map | none |
| `continuous_onset_r2_smooth` | aligned syllable onset | one full-chunk smoothed R2 time map | none |
| `continuous_onset_constrained_r3_stress` | aligned syllable onset | bounded, stress-weighted R3 time map | smooth gain envelope |

### Onset promotion

The strict vowel match is still used to locate each syllable. The renderer then walks backward over the planned pre-vowel phones. It promotes the anchor only when those phones exactly match the aligned phone sequence. Otherwise it retains the original vowel anchor. Word-tier fallback anchors are also retained unchanged. This makes onset alignment conservative and auditable through `anchor_kind`.

### Continuous warp

The continuous modes invoke Rubber Band once for the complete two-bar source. A source-to-target sample map contains the endpoints and all syllable anchors. R3 uses `--fine`; the R2 comparison uses `--smoothing`. This removes independently synthesized joins and lets Rubber Band maintain phase continuity across the chunk.

### Constrained timing

The constrained mode minimizes weighted squared target-anchor drift subject to local stretch ratios between 0.5x and 2.0x. Stronger flow slots receive larger objective weights, so the optimizer prefers to move weak anchors when it must trade exact timing for render quality. Optimization is performed in normalized coordinates to remain stable for 24 kHz, 128,000-frame chunks.

The final integer map reached 0.499964x to 2.0x. The tiny lower-bound violation is one-sample rounding, not an audible 2x policy violation.

### Stress rendering

Target stress is mapped linearly from -1.0 dB at stress 0 to +2.5 dB at stress 1. Transitions use a 25 ms cosine ramp. The envelope follows each syllable's effective post-regularization anchor, while diagnostics retain both requested and effective target times. Overall chunk RMS is restored after emphasis, followed by a 0.999 peak guard. Pitch is not modified. Diagnostics record the per-syllable gains, input/output RMS, and peak limiting.

## Protocol

- Existing MOSS source WAVs, MFA TextGrids, requests, lyrics, and drums were reused without regeneration.
- Twelve chunks were selected from the 75-chunk baseline: three relatively clean, three median, and six worst by `max(ratio, 1/ratio)`.
- “Clean” is relative. Even the three cleanest baseline chunks had a 4.0x to 4.5x worst local distortion because of the old endpoint-anchor behavior.
- Every selected source was rendered in all five modes, producing 60 dry vocals and 60 matched drum mixes.
- ASR used faster-whisper `large-v3`, `float16`, on H200 GPU 2.
- Metrics include WER, ASR-derived timing error, local stretch ratios, anchor-boundary sample jumps, stress/RMS correlation, clipping, and silence.
- ASR timing is an estimate from recognized word intervals, not phone-level timing ground truth.

## Pilot Results

The table below is from campaign v4. During final review, v4 exposed that the constrained mode centered stress on requested rather than post-regularization anchors. Commit `a419010b` corrected that behavior, and a complete v5 render plus large-v3 ASR pass finished on H200. The H200 network became unreachable before v5 could be downloaded, so the constrained row below is retained as pilot evidence and must not be presented as final stress evidence. The other four modes are unaffected by the correction.

Mean over the same 12 selected chunks:

| Mode | WER | ASR timing MAE | Boundary jump | Extreme regions | Worst reciprocal ratio | Stress/RMS corr. |
|---|---:|---:|---:|---:|---:|---:|
| Piecewise vowel R2 | 10.95% | 268.2 ms | 0.03549 | 13.33 | 18.84x | +0.152 |
| Continuous vowel R3 | 8.05% | **161.0 ms** | 0.00398 | 7.42 | 18.84x | -0.200 |
| Continuous onset R3 | 11.95% | 192.2 ms | 0.00289 | 7.67 | 5.86x | -0.278 |
| Continuous onset R2 smooth | 15.43% | 212.5 ms | **0.00160** | 7.67 | 5.86x | -0.165 |
| Constrained onset R3 + stress (v4 pilot) | **3.86%** | 190.8 ms | 0.00246 | **0.58** | **2.0001x** | -0.226 |

Signal checks found no silent chunks. All modes except smoothed R2 had zero clipped samples; smoothed R2 had 70 clipped samples across 12 chunks.

The constrained policy moved anchors by 52.6 ms on average, 208.5 ms at p95, and 288.3 ms maximum. This is a meaningful timing tradeoff and should remain visible to users and researchers.

## Interpretation

1. Continuous full-chunk warping substantially reduces boundary discontinuities. All continuous modes improve the boundary-jump metric by roughly an order of magnitude over piecewise R2.
2. Correcting anchor semantics matters. Onset-aware modes reduce worst reciprocal distortion from 18.84x to 5.86x before any timing relaxation.
3. Bounded R3 is the strongest warp candidate from the pilot. It has the best pilot WER and nearly eliminates extreme local ratios, at the cost of measurable anchor drift. Final v5 ASR must be read before treating the 3.86% WER as the final value.
4. Exact vowel R3 has the best ASR-derived timing estimate, so the constrained policy is not unconditionally best. Listening should decide whether its intelligibility improvement justifies 50-200 ms of weak-slot drift.
5. The v4 stress result is invalid as a final perceptual test because its envelope could target the wrong phonetic region after anchor drift. V5 corrects that defect. Stress still needs a clean ablation: constrained R3 with and without stress, plus listening ratings.

## Artifacts

Local v4 pilot root:

`output/aligned_moss_warp_experiment_20260817_v4_pilot/`

Final v5 H200 root:

`/data/home/Andrew.Yang/StreamMUSE/experiments/aligned_moss_warp_20260817_v5/`

- `index.html`: dense aggregate and per-chunk listening matrix
- `listening/*.wav`: five aggregate drum-mix A/B files
- `chunks/`: dry vocals, drum mixes, and alignment diagnostics
- `selection.json`: selected chunks and original distortion scores
- `manifest.json`: source and output provenance with 120 SHA-256 hashes
- `metrics.json`: all per-render and aggregate metrics, including ASR transcripts

All 120 manifest-referenced v4 WAV hashes were verified after H200 download. The local pilot contains 177 files total. V5 rendering and ASR reached their explicit completion markers before the network interruption, but its local transfer and independent hash verification remain pending.

## Reproduction

```bash
python scripts/run_aligned_moss_warp_experiment.py \
  --input-dir output/rap_audio_protocol_comparison_20260816 \
  --output-dir output/aligned_moss_warp_experiment

CUDA_VISIBLE_DEVICES=2 python scripts/run_aligned_moss_warp_experiment.py \
  --input-dir output/rap_audio_protocol_comparison_20260816 \
  --output-dir output/aligned_moss_warp_experiment \
  --asr-only \
  --whisper-model large-v3 \
  --whisper-device cuda \
  --whisper-compute-type float16
```

The H200 Rubber Band runtime also requires the project-defined `PATH` and `LD_LIBRARY_PATH` from `scripts/setup_rap_audio_protocols_h200.sh`.

## Next Experiment

Run a blinded comparison on at least three complete songs with these three finalists:

1. continuous vowel R3 for exact timing,
2. constrained onset R3 without stress,
3. constrained onset R3 with a stronger, vowel-centered stress envelope.

Collect separate ratings for intelligibility, rhythmic alignment, glitch severity, stress clarity, and overall rap likeness. Tune the ratio bounds and stress envelope only after examining those dimensions separately.
