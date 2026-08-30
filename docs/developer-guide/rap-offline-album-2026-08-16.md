# Offline Rap Album Production Report

## Deliverable

The final artifact set is `output/rap_album_10x50_90bpm_20260816_v4/`. It contains ten songs with fifty bars per song, one fixed topic per song, chosen-lyric and generation-attempt logs, generation statistics, per-syllable audio diagnostics, and one drum-backed vocal WAV per song.

## Assumptions

- "10 50 bar songs" means ten songs containing fifty bars each.
- Tempo is fixed at 90 BPM in 4/4, matching the current rap demo default.
- Every bar uses one of the existing nine-slot flow templates, for exactly nine spoken syllables per bar.
- Each song cycles the straight, syncopated, and staggered templates in a fixed eight-bar arrangement.
- One topic remains fixed for the entire fifty-bar song.
- Robotic syllable-level speech is acceptable. Exact onset timing has priority over natural phrasing or preserving every syllable tail.
- WAV satisfies the requested audio artifact; no MP4 wrapper is produced.

## Pipeline

1. Qwen2.5-7B-Instruct runs through vLLM on H200 GPU 1.
2. Each API request asks for 64 independent one-line completions against the current topic, recent lyric context, and full flow template.
3. CMUdict-based analysis rejects non-ASCII text and lines that are not exactly nine syllables.
4. Candidates must have total score at least 0.30 and stress alignment at least 0.60.
5. Selection rejects repeated word trigrams and normally caps identical opening and closing words at two uses per song.
6. One bar exhausted six batches because every otherwise-valid line reused an opening word. It used the logged `relaxed_opening_word_cap` policy while retaining every prosody and score threshold.
7. The selected syllables are mapped one-to-one onto absolute flow ticks.
8. The deterministic renderer places each syllable onset at its exact target sample and mixes procedural drums into the same bar buffer.
9. Each WAV is committed only after frame count, sample timing, and signal checks pass.

## Process Record

- The first production attempt exposed repeated three-word lyric prefixes. A global normalized-trigram gate and rotating narrative focus were added.
- The second attempt varied phrases but overused the same opening word. A two-use opening-word cap and prompt feedback were added.
- The third attempt overused one literal closing rhyme word, such as `rise` or `call`. A two-use closing-word cap and prompt feedback were added.
- The final run reached 490 strict bars before one round had no candidate solely because of the opening cap. A tested, logged opening-cap-only fallback completed the song.
- The final corpus audit found three mixed-script or typographic-apostrophe lines. They were replaced from their original H200 candidate pools and re-analyzed before audio rendering.

## Final Metrics

### Lyric generation

| Metric | Result |
|---|---:|
| Songs / bars | 10 / 500 |
| Exact nine-syllable bars | 500 / 500 |
| ASCII lyric bars | 500 / 500 |
| API attempts | 565 |
| API errors | 0 |
| Requested / returned candidates | 36,160 / 35,088 |
| Bars meeting target pool of 3 | 498 / 500 |
| Strict selections | 499 / 500 |
| Opening-cap relaxation | 1 / 500 |
| Mean selected total score | 0.531 |
| Minimum selected total score | 0.352 |
| Mean stress alignment | 0.729 |
| Minimum stress alignment | 0.602 |
| Repeated normalized trigrams | 0 |

### Audio rendering

| Metric | Result |
|---|---:|
| WAV files | 10 |
| Format | 48 kHz stereo IEEE float32 |
| Frames per song | 6,400,000 |
| Duration per song | 133.333 s |
| Total duration | 1,333.333 s (22:13.333) |
| Total WAV bytes | 512,000,440 |
| Scheduled syllable onsets | 4,500 |
| Maximum software timing error | 0 samples |
| CMUdict pronunciations | 4,417 (98.16%) |
| eSpeak G2P pronunciations | 67 (1.49%) |
| Grapheme-fragment pronunciations | 16 (0.36%) |

The renderer reports 2,836 `timing_pressure` warnings, 335 `forced_bar_fit` warnings, 983 overlapping syllable slots, and 83 pronunciation fallbacks. These are visible best-effort warnings, not onset misses. They mean the generated speech fragment was longer than its slot, required compression/truncation at the bar boundary, or used a pronunciation fallback. Every target onset still has zero software timing error.

## Artifacts

- `ARTIFACT_INDEX.md`: per-song artifact links
- `album_manifest.json`: configuration and assumptions
- `album_stats.json`: album and per-song aggregate metrics
- `<song>/lyrics.txt`: readable selected lyrics
- `<song>/chosen_lyrics.jsonl`: scores, flow template, syllables, phonemes, and absolute schedule
- `<song>/generation_attempts.jsonl`: prompts, all H200 candidates, latency, and pool counts
- `<song>/generation_stats.json`: per-song generation distributions
- `<song>/audio_render_log.jsonl`: per-bar warnings and per-syllable timing diagnostics
- `<song>/audio_stats.json`: WAV, warning, pronunciation, and render statistics
- `<song>/song.wav`: final drums plus syllable-level vocal performance

## Reproduction

Generate lyrics against the H200 vLLM endpoint:

```bash
.venv/bin/python scripts/produce_offline_rap_album.py \
  --output-dir output/rap_album_10x50_90bpm_20260816_v4 \
  --stage lyrics --bars 50 --tempo 90 --choices 64 \
  --base-url http://127.0.0.1:18001/v1 --model qwen-rap \
  --minimum-score 0.30 --minimum-stress 0.60 \
  --target-pool 3 --max-attempts 6 --timeout-s 60
```

Render locally after lyric generation:

```bash
.venv/bin/python scripts/produce_offline_rap_album.py \
  --output-dir output/rap_album_10x50_90bpm_20260816_v4 \
  --stage audio --bars 50 --tempo 90 --choices 64 \
  --base-url http://127.0.0.1:18001/v1 --model qwen-rap \
  --minimum-score 0.30 --minimum-stress 0.60 \
  --target-pool 3 --max-attempts 6 --timeout-s 60
```

## Limitations

- The score is an engineered proxy for topic relevance, stress, rhyme, continuity, and novelty. It is not a learned human rap-quality score.
- The voice is intentionally robotic and syllable-oriented. Exact timing is stronger than natural coarticulation.
- Timing-pressure and forced-fit warnings are frequent at 90 BPM because long phonemes must fit short flow slots.
- Topic progression is prompt-guided rather than planned as a global fifty-bar narrative.
