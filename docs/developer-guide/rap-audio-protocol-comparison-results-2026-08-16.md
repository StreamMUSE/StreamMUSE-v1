# Rap Audio Protocol Comparison Results

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

Metrics use faster-whisper `large-v3`. WER is an intelligibility proxy, not a listening score. Timing is estimated only for words that ASR matches exactly; the reported error is signed, so negative values mean the recognized syllable estimate preceded its target. The song p95 column is the mean of the three per-song signed p95 values.

| Protocol | WER | Matched syllables | Mean timing error | Mean song p95 | Mean duration error | Stress/RMS corr. |
|---|---:|---:|---:|---:|---:|---:|
| `moss_global` | 2.5% | 1,318 | -309 ms | 219 ms | 105.6 ms | -0.147 |
| `ted_local` | 13.4% | 1,191 | -437 ms | 193 ms | 861.6 ms | -0.110 |
| `fastpitch_phoneme` | 125.4% | 108 | -177 ms | 54 ms | 4.4 ms | 0.094 |
| `moss_aligned` | 14.6% | 1,174 | -268 ms | 19 ms | 0.0 ms | 0.121 |

`moss_global` is by far the most intelligible according to ASR. `moss_aligned` preserves enough intelligibility for 1,174 matched syllables while obtaining exact chunk duration and the strongest stress/RMS relationship. Its warping still degrades WER relative to the source, and the negative mean timing error shows that the anchor policy is systematically early.

The FastPitch result demonstrates exact duration control but not usable pronunciation: WER above 100% and only 108 measurable syllables make its apparently small timing numbers unreliable. It remains useful as an explicit-duration engineering reference, not as the leading audio protocol.

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
- `faster-whisper==1.2.0` was installed into the post-synthesis MOSS environment solely for independent evaluation.
- No perceptual winner is claimed from objective metrics. The blinded package is the decision artifact.

## Artifacts

The local campaign directory is `output/rap_audio_protocol_comparison_20260816/`. Important files are:

- `listening.html`: blinded A-D listening interface.
- `blind_map.json`: protocol mapping, kept separate from the listening page.
- `experiment_metrics.json`: all 12 per-song/protocol metric records.
- `package_audit.json`: source/blind SHA-256 verification.
- `<protocol>/<song>/mix.wav`: drums plus vocals.
- `<protocol>/<song>/vocals.wav`: protocol vocal stem.
- `fastpitch_phoneme/<song>/chunk-*.timing.json`: token and duration plans.
- `moss_aligned/<song>/chunk-*.wav.alignment.json`: requested/effective anchor maps and fallback diagnostics.
