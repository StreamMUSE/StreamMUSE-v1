# Rap Audio Protocol Comparison Design

Date: 2026-08-16

## Status

Approved design for a controlled offline comparison of four lyric-to-audio
protocols. The experiment holds lyrics, MCFlow schedules, tempo, drums, song
length, and output format constant while changing only vocal synthesis and
timing control.

## Research Question

Which currently runnable audio-generation approach best preserves intelligible
whole-word pronunciation while placing the intended syllables on an externally
specified rap flow?

The comparison must distinguish three forms of timing control:

1. Global phrase-duration conditioning.
2. Native local segment or phoneme duration conditioning.
3. Post-generation forced alignment and waveform retiming.

## Fixed Corpus

Use the first three completed songs from
`output/rap_album_10x50_90bpm_20260816_v4/`:

| Song | Topic | Bars |
|---|---|---:|
| Signals Beyond Earth | space exploration | 50 |
| Pressure Below | deep ocean | 50 |
| Learning Machines | artificial intelligence | 50 |

Each protocol consumes the existing `chosen_lyrics.jsonl` records. Lyrics may
not be regenerated, rewritten, or replaced for one protocol. Every protocol
uses the same template identifiers, syllable analyses, and materialized MCFlow
slots stored in those records.

## Common Audio Contract

- Tempo: 90 BPM.
- Meter: 4/4.
- Ticks per beat: 4.
- Song length: 50 bars, excluding an optional common count-in.
- Vocal synthesis window: two bars, or 5.333 seconds.
- Output sample rate: 48,000 Hz.
- Output channels: stereo for mixes and mono or stereo for vocal stems.
- Drums: one common deterministic drum stem per song.
- Voice: the same permissively licensed LJSpeech reference where the model
  supports reference conditioning; otherwise the model's LJSpeech voice.
- Outputs: vocal stem, drum stem, mixed WAV, render log, metrics, and manifest.
- No protocol may change a lyric to improve renderability.
- No protocol may silently substitute the existing isolated-eSpeak-syllable
  renderer.

Twenty-five two-bar chunks cover each song. Model inputs may contain neighboring
text as non-spoken context when supported, but each output chunk must contain
exactly the two target bars. Chunk assembly may use fixed short crossfades that
do not move an intended downbeat.

## Protocol 1: MOSS Global Duration

Use MOSS-TTS-v1.5 to synthesize each complete two-bar phrase as connected
speech. Set the target audio-token count from the two-bar duration using the
documented 12.5 audio tokens per second rate. Provide the lyrics, a stable voice
reference, and a concise delivery instruction requesting clear, rhythmically
spoken rap with restrained pitch.

The model receives total phrase duration and textual flow guidance but no
per-syllable waveform manipulation. The resulting vocal is normalized and
placed at the two-bar chunk start. This protocol measures the strongest simple
modern-TTS baseline with native global duration control.

## Protocol 2: TED Local Duration

Use TED-TTS on its supported IndexTTS2 checkpoint. Partition each two-bar lyric
into the smallest linguistically valid segments whose boundaries are both word
boundaries and MCFlow phrase boundaries. If a flow boundary falls inside a
word, retain the complete word in one segment.

Assign each segment a target duration from the interval between its first
scheduled syllable and the next segment's first scheduled syllable, with the
last segment ending at the two-bar boundary. Pass all segments and durations in
one TED-TTS inference request. Do not split and independently concatenate the
segments outside the model. This protocol measures native intra-utterance local
duration steering.

## Protocol 3: FastPitch Phoneme Duration

Use NVIDIA NeMo's English FastPitch and HiFi-GAN checkpoints. Convert each
two-bar lyric into one continuous phoneme sequence and provide explicit model
token durations rather than accepting the predicted durations.

For every syllable:

1. Treat its vowel nucleus as the perceptual timing anchor.
2. Place the vowel nucleus at the MCFlow slot.
3. Allocate onset consonants immediately before that anchor when time exists.
4. Allocate coda consonants after the vowel and before the next syllable's
   onset region.
5. Preserve a positive minimum duration for every emitted phoneme.
6. Allocate remaining interval duration primarily to the vowel.

Flow stress controls a bounded pitch and energy accent. Strong slots receive a
larger accent than weak slots, but pitch remains in a narrow speech-like range.
The model renders the complete phoneme sequence jointly. This protocol measures
deterministic native phoneme timing without isolated-syllable synthesis.

## Protocol 4: MOSS Forced-Alignment Warp

Reuse Protocol 1's raw MOSS two-bar waveform byte-for-byte as the source. Do
not generate another MOSS sample.

Force-align the source waveform to the known lyric at word and phoneme level.
Associate each planned syllable with the aligned vowel nucleus, then create a
strictly monotonic mapping from source vowel anchors to target MCFlow slots.
Apply a pitch- and formant-preserving piecewise time warp. Prefer stretching
vowels and silence over consonant transients, and use short fixed crossfades at
piece boundaries. Reject non-monotonic or physically impossible mappings and
record the chunk as failed rather than producing misleading audio.

Forced alignment is part of this protocol's rendering method. It may not be
used to modify Protocols 1 through 3.

## Failure Policy

Every protocol must be auditable. A failed model request is retried with the
same text and deterministic seed policy. After the configured retry limit:

- record the exception, model parameters, and attempt count;
- emit silence for the affected vocal chunk so song duration remains exact;
- do not substitute another protocol's vocal;
- count the event in the protocol and song metrics.

If an upstream project cannot be installed or its published checkpoint cannot
execute after a documented integration attempt, stop that protocol and report
the blocker. Do not relabel another model as the requested protocol.

## Evaluation

Evaluation may analyze all generated outputs but may not alter them. To avoid a
circular result, Protocol 4's timing is evaluated with an alignment system that
is independent from its rendering aligner.

Record at least:

- successful chunks and failed chunks;
- synthesis and post-processing latency;
- generated duration and duration error;
- ASR word error rate;
- missing, repeated, and substituted words;
- mean, median, p95, and maximum syllable timing error where measurable;
- stress realization correlation using local energy and F0;
- peak level, clipping count, and silent-chunk count;
- local stretch distribution for Protocol 4;
- model names, revisions, licenses, seeds, and generation parameters.

Listening remains the deciding evaluation. Objective metrics provide diagnostic
evidence and must not be presented as a substitute for perceptual comparison.

## Artifacts

Write the experiment under:

```text
output/rap_audio_protocol_comparison_20260816/
├── common/
│   ├── corpus_manifest.json
│   ├── lyrics.md
│   └── <song>/drums.wav
├── moss_global/<song>/
├── ted_local/<song>/
├── fastpitch_phoneme/<song>/
├── moss_aligned/<song>/
├── experiment_metrics.json
├── COMPARISON.md
└── listening.html
```

Each protocol/song directory contains `vocals.wav`, `mix.wav`,
`render_chunks.jsonl`, and `metrics.json`. The listening page groups each song's
four methods together and initially uses neutral protocol identifiers so the
listener can compare without descriptions biasing the first impression.

## Implementation Boundaries

The repository owns:

- loading and validating the fixed corpus;
- converting MCFlow and prosody records into backend-neutral render requests;
- backend adapters for the four protocols;
- deterministic chunk assembly, drums, metrics, manifests, and listening page;
- H200 orchestration and artifact download instructions.

External model repositories and checkpoints remain outside the Git repository.
Their exact revisions and locations are recorded in the experiment manifest.
The existing realtime renderer and website are not changed by this offline
experiment.

## Acceptance Criteria

The experiment is complete when:

1. All four adapters execute against a small two-bar smoke fixture or have a
   documented upstream blocker under the failure policy.
2. Every executable protocol produces all three 50-bar vocal stems and mixes.
3. The common drum stem and lyric schedule are byte-identical across matching
   protocol comparisons.
4. Protocol 4 records the hash of the Protocol 1 source chunk it reused.
5. Every WAV has the expected sample rate and exact song-frame count.
6. Metrics and render logs account for all 25 chunks per song.
7. The local listening page exposes the resulting comparable artifacts.
