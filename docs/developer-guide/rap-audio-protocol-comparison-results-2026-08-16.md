# Rap Audio Protocol Comparison Results

Corrected 2026-08-17 after the final independent code and artifact review.

## Experiment

The campaign renders the same three 50-bar, 90 BPM songs with four offline lyric-audio protocols. Each song contains 25 independently rendered two-bar chunks. The final listening format is 48 kHz: every vocal stem and mix is exactly 6,400,000 frames, or 133.333 seconds.

| Protocol | Timing mechanism |
|---|---|
| `moss_global` | MOSS-TTS generates each two-bar phrase with one global duration target. |
| `ted_local` | IndexTTS2/TED-TTS receives two one-bar duration targets. |
| `fastpitch_phoneme` | NeMo FastPitch receives an explicit duration for every tokenizer label, derived from the requested syllable slots. |
| `moss_aligned` | The exact `moss_global` source is aligned with MFA and piecewise warped with Rubber Band so vowel anchors approach requested syllable times. |

All protocols rendered 75/75 chunks successfully. The campaign contains 300 successful records, 12 vocal stems, 12 drum mixes, 3 shared drum stems, and a 12-file blinded listening package. No final chunk was silent and no evaluated chunk was marked failed.

## Objective Results

Metrics use faster-whisper `large-v3`. WER is an intelligibility proxy, not a listening score. Timing is estimated only for words that ASR matches exactly. Signed bias indicates whether recognized syllables tend to be early or late; absolute error measures timing magnitude. The p95 column is the mean of the three per-song absolute-error p95 values, not a pooled percentile.

| Protocol | WER | Matched syllables | Signed bias | Mean absolute error | Mean song abs. p95 | Duration error | Stress/RMS corr. |
|---|---:|---:|---:|---:|---:|---:|---:|
| `moss_global` | 2.5% | 1,318 | -309 ms | 371 ms | 954 ms | 105.6 ms | -0.147 |
| `ted_local` | 13.4% | 1,191 | -437 ms | 483 ms | 1,142 ms | 861.6 ms | -0.110 |
| `fastpitch_phoneme` | 124.4% | 121 | -163 ms | 215 ms | 544 ms | 4.4 ms | 0.094 |
| `moss_aligned` | 14.7% | 1,173 | -268 ms | 277 ms | 606 ms | 0.0 ms | 0.112 |

`moss_global` is by far the most intelligible according to ASR. Among the natural-voice methods, `moss_aligned` reduces mean absolute timing error from 371 ms to 277 ms and mean per-song absolute p95 from 954 ms to 606 ms, while obtaining exact chunk duration and a positive stress/RMS relationship. Its warping still degrades WER relative to the source, and the negative signed bias shows that the anchor policy remains systematically early.

The FastPitch result demonstrates near-exact duration control but not usable pronunciation: WER above 100% and only 121 measurable syllables make its timing numbers unreliable. It remains useful as an explicit-duration engineering reference, not as the leading audio protocol.

## Fallback Audit

FastPitch used three explicit, sidecar-audited recoveries for NeMo tokenizer behavior:

- Fully lowercase grapheme words are grouped from requested-count vowel nuclei: 208 word occurrences.
- Alternate all-ARPABET pronunciations are accepted only with the exact requested vowel count: 14 word occurrences.
- One adjacent-vowel expansion was required for two-syllable `ruins`; one-syllable `ai` remained collapsed.

The aligned-MOSS baseline used guarded word-tier fallback when MFA and planned phone inventories could not match exactly. Across the final campaign it recorded 152 fallback syllable anchors, 75 tick-zero target-boundary adjustments, and one 7.5 ms source-boundary adjustment. Every fallback requires a unique word owner and monotonic anchors; malformed, ambiguous, colliding, and out-of-range mappings fail closed.

## Deviations And Assumptions

- TED-TTS's requested HMM alignment path produced a `NaN` in `StreamingHMMAligner.last_center` on the pinned H200 build. The campaign uses the explicit `max_head` ablation and records that method in each song's inference configuration.
- TED output is often shorter than the requested two-bar window. Assembly pads each chunk to the shared timeline; raw duration error remains visible in the metrics.
- MOSS style instructions are best-effort because the upstream API does not guarantee rap-style instruction following.
- The first aligned-MOSS render mistakenly passed Rubber Band `--pitch 1.0`, which means a one-semitone shift. Those outputs were preserved outside the final campaign; all 75 final Protocol 4 chunks were rerendered with no pitch/frequency option, then reassembled, reevaluated, and repackaged.
- The evaluation environment used `faster-whisper==1.2.0`; the setup script's later `1.2.1` pin was not used to compute these results.
- The MOSS manifest records snapshot `cdd3b911b1585e3f2dbc7775ef10f9926f58850a`. TED uses the published IndexTTS2 checkpoint with the recorded `max_head` method; FastPitch uses NeMo `tts_en_fastpitch` and `tts_en_hifigan`. Exact external revision identifiers were not retained for every backend.
- Per-chunk synthesis and post-processing wall-clock latency was not retained in the final ledgers. This is an instrumentation gap; no retrospective latency numbers are claimed.
- No perceptual winner is claimed from objective metrics. The blinded package is the decision artifact.

## Artifacts

The local campaign directory is `output/rap_audio_protocol_comparison_20260816/`. Important files are:

- `listening.html`: blinded A-D listening interface.
- `blind_map.json`: protocol mapping, kept separate from the listening page.
- `experiment_metrics.json`: all 12 per-song/protocol metric records.
- `package_audit.json`: source/blind SHA-256 verification.
- `common/lyrics.md`: human-readable lyrics for all 75 two-bar chunks.
- `<protocol>/<song>/mix.wav`: drums plus vocals.
- `<protocol>/<song>/vocals.wav`: protocol vocal stem.
- `fastpitch_phoneme/<song>/chunk-*.timing.json`: token and duration plans.
- `moss_aligned/<song>/chunk-*.wav.alignment.json`: requested/effective anchor maps and fallback diagnostics.
