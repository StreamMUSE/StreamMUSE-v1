# Flow and Audio Controls: Delivery Evidence

## What Changed

1. **Six curated MCFlow patterns** replace synthetic choices in the live picker.
   They contain 8-12 syllable slots; two begin at ticks 1 and 2 rather than zero.
   Timing, stress, durations, boundaries, and rhyme tags match the existing
   extracted manifests. No original lyrics were added. Old IDs remain readable
   for saved scenarios; custom scenarios gain validated fallback coverage for
   missing curated syllable counts. [Catalog and provenance](../developer-guide/rap-curated-flow-catalog.md).
2. **Optional chord accompaniment:** Am-F-C-G and C-G-Am-F, one chord per bar,
   with a separate gain. These are hand-authored, not MCFlow annotations. Local
   synthesis uses exact bar frame counts and smooth envelopes. Physical playback
   bar, not response ordinal, determines harmony during exchange waiting/replay.
3. **Three local MOSS alignment policies:** relaxed, strict syllable timing, and
   strict timing plus stress. The third adds smooth gain accents up to 1.35x at
   strong slots, without changing pitch, sample positions, or pronunciation text.
   Policy changes are stopped-only and rebuild preparation; chords/gains apply
   at eligible bar boundaries. Existing topic, tempo, devices, diagnostics,
   transport, and two interaction modes are retained.
4. **Leading-rest fix:** capture and request generation already retained positive
   first ticks. The old warp treated source pre-roll as part of the stretch and
   relaxed mapping could move the first entrance. The shared warp now crops to
   the first source onset, warps only the active span, and prepends exact silence.
   Final output conversion removes resampling pre-ringing before that entrance.
   One-slot captures work too. Server-side and client-side paths share this code.
5. **Approximate syllable labels:** e.g. `whis-`, `-per`, with full word/index/count
   retained. Display-only labels do not change synthesis or scoring.

Review also found and fixed two UI acknowledgement bugs: a rebuild could hide a
rejected policy POST, and reloading could lose an alternating mix acknowledgement.
Session-start audit records now report current audio settings rather than launch
defaults. Public source anchors are explicitly in original source-audio coordinates.

## Actual Mac + H200 Tests

The Mac owns playback and local rendering. Qwen runs on H200 GPU0; MOSS/MMS runs
on GPU1. Both GPUs were empty before launch. Other jobs were untouched.
[Deployment identities and commands](2026-10-28-rap-flow-audio-deployment.md).

| Test | Result |
| --- | --- |
| Continuous, 60 BPM, strict timing + stress, chords | 20/20 committed bars used MOSS; no vocal fallback; zero underruns |
| Browser topic + delayed-template edit during playback | New topic/template reached playback with ASAP behavior |
| Browser chord-loop change | Played mix confirmed the new progression and mix revision |
| Exchange, relaxed, first tap tick 2 | 2/2 MOSS response bars, 2 echo bars, zero underruns |
| Exchange, strict, first tap tick 4 | 2/2 MOSS response bars, 3 echo bars, zero underruns |
| Exchange, strict + stress, first tap tick 2 | 2/2 MOSS response bars, 3 echo bars, zero underruns |
| Stop then Start without Reset | Exercised after each of the three exchange turns; active Mac device stream |

Exchange taps were submitted to the real local capture API at their scheduled
monotonic times. This verifies capture, quantization, requests, real models,
responses, and playback; it is not a human keyboard/listening experiment.
The browser was used for Start/Stop, policy/chord selection, and live topic/flow
editing. Device evidence reports an active Core Audio stream, not microphone
measurement or a human judgement of intelligibility.

Evidence is under `output/rap_flow_audio_20261028/`: continuous snapshot,
exchange HTTP trace and per-policy snapshots, summary, canonical session events,
and the recorded mix. The latest client is on **http://127.0.0.1:8018/**, forwarded
to H200 render port **18720**. It is left stopped for the user.

## Listening Comparisons

Each condition uses identical generated lyrics, source MOSS audio, source
alignment, drums, and mix gains. Chords are off to avoid masking differences.

| Clip | Flow / tempo | Relaxed maximum estimated timing error, by bar | Strict / strict+stress |
| --- | --- | --- | --- |
| Night Train | Delayed 8, 60 BPM | 602 / 751 ms | 0 / 0 ms |
| Glass Rooftops | Clustered 11, 90 BPM | 252 / 131 ms | 0 / 0 ms |

These are **time-map estimates**, not measured acoustic errors. The Night Train
vocal stem is digitally silent for the requested first 0.5 seconds in both bars
under all three policies. Local render times across these short tests were
approximately **55-156 ms/bar**; they exclude model generation and transport.
Source generation/fetch took 21.85 seconds for Night Train and 13.52 seconds for
Glass Rooftops, each a two-bar offline comparison. Those totals are not per-bar
steady-state realtime latency measurements.

Files, sources, lyrics, and JSON diagnostics:

- `output/rap_alignment_20261028/night_train/`
- `output/rap_alignment_20261028/glass_rooftops/`

For each clip: `*_A_relaxed_mix.wav`, `*_B_strict_mix.wav`,
`*_C_strict_stress_mix.wav`, plus vocals-only counterparts. Reproduction:
`scripts/experiments/rap_alignment_comparison.py`.

## Limits and Choices

- Strict mapping does not correct errors in forced alignment. Extreme stretching
  can still sound unnatural; listening preference decides the policy.
- Strict means syllable-onset alignment. It does not force every syllable to end
  at its annotated duration; the bar's leading rest is explicitly protected.
- Stress augmentation is an amplitude cue, not learned expressive prosody or a
  guarantee that lexical stress naturally matches the requested pattern.
- The candidate stress/rhyme ranking algorithm is unchanged. This work changes
  vocal realization, not the definition of a valid lyric candidate.
- Continuous acceptance was at 60 BPM. The 90-BPM clips were rendered offline;
  they do not establish reliable live 90-BPM operation.
- Chords are an intentionally small starting catalog and default to Off.
- Existing unrelated dirty work is preserved. Nothing was committed or pushed.

## Verification

Final full regression suite: **2,767 passed, 5 skipped** in 112.14 seconds, with
the acknowledgement and session-start metadata regressions included.
The skipped tests require their existing optional integration prerequisites.
`git diff --check` was clean. Listening quality remains for the user to assess.
