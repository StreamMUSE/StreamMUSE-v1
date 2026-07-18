# Rap Alignment Prototype Design

## Goal

Build a runnable StreamMUSE prototype that turns a topic into rap-style lyric
lines, assigns every estimated syllable to an explicit musical tick, and can
print those syllables at tempo. The first deliverable proves and exposes
beat-aligned text; it does not attempt real-time speech synthesis or keyboard
drum capture yet.

## User Value

The critical question is whether generated text can be constrained, inspected,
and scheduled as musical material. A user must be able to run one command and
see the selected line, its syllable count, its position within each 4/4 bar,
accent strength, and its time in seconds. `--play` then emits the same schedule
in real time, so the text layer can be evaluated before speech is added.

## Approaches Considered

1. Prompt an LLM to output a line with a requested syllable count. This is
   fast to demonstrate but cannot guarantee metre, because the count and stress
   are only model assertions.
2. Use a pronunciation dictionary to count syllables and assign timing after
   generation. This is more trustworthy, but the current project environment
   has none of `pronouncing`, `cmudict`, or `pyphen` installed.
3. Generate several candidates, analyse them locally, and select/schedule the
   best-fitting line. This keeps the timing mechanism deterministic and works
   with an offline phrase bank. An optional local LLM improves language quality
   without becoming the source of truth.

The prototype uses approach 3. A small rule-based analyser is the baseline.
It handles common irregular words and vowel-group heuristics, marks the first
estimated syllable of a word as stressed, and preserves the word plus syllable
ordinal in the output. A later CMUdict adapter can replace only that analyser.

## Architecture

The implementation is a new Clean Architecture vertical slice. It does not
modify `RealTimeMusicService` or the melody/accompaniment inference protocol.
It reuses `domain.timing.Tempo`, including the existing convention that one
tick equals a sixteenth note when `ticks_per_beat=4`.

```
CLI options / local chat server
             |
             v
CandidateGenerator -> RapPrototypeService -> AlignmentEngine -> terminal / JSON / --play
                         |                       ^
                         v                       |
                   RhythmPlanner ---------> 16 tick slots per 4/4 bar
                         |
                         v
                Tempo (shared StreamMUSE timing value object)
```

`domain.rap` contains immutable data contracts and the dependency-free prosody
analysis. `application.rap` creates accent-weighted beat slots and uses dynamic
programming to place syllables on monotonically increasing slots. The service
scores several candidate lines, penalizes density mismatch and rejects
overflows. `infrastructure.rap` provides a deterministic phrase bank and a
local OpenAI-compatible chat adapter that falls back to the phrase bank. The
presentation layer owns parsing, readable schedule rendering, JSON export, and
the optional real-time terminal emitter.

## Rhythm and Alignment

The CLI supports three 4/4 patterns: `boom_bap`, `straight_8`, and
`trap_sparse`. Each provides 16 accent weights, one for each StreamMUSE tick in
a bar. The planner exposes absolute tick, bar, beat, tick-in-beat, and accent
weight for every slot.

For a candidate with no more than 16 estimated syllables, the alignment engine
chooses increasing slots with dynamic programming. The first syllable is fixed
to the bar downbeat. The objective rewards stressed syllables on higher-accent
slots and penalizes departure from an even progression through the bar. It also
penalizes overly sparse lines. Candidates above 16 syllables receive an
overflow penalty and cannot win while a fitting candidate exists. This makes
the schedule measurable and repeatable rather than merely prompt-directed.

Each event retains the original word, ordinal within that word, estimated stress
flag, musical position, accent, and wall-clock offset. A multi-syllabic word is
rendered as the word on its first syllable and `.` for continuations; the JSON
keeps the full metadata required by a future forced-alignment or TTS adapter.

## CLI Contract

Add a `streammuse-rap` entry point:

```text
uv run streammuse-rap --topic "space travel" --tempo 92 --pattern boom_bap --bars 4
uv run streammuse-rap --topic "space travel" --generator local_chat --model-url http://localhost:8000/v1 --model local-model
uv run streammuse-rap --topic "space travel" --bars 2 --play --output-json rap_plan.json
```

The default generator is `phrase_bank`, so the first command works without
network access or a model. `local_chat` requests multiple one-line candidates
from the existing `LocalChatModelClient`; malformed responses or client errors
fall back to phrase-bank output and report the fallback in the rendered plan.
`--play` is intentionally terminal-only: it validates the schedule against the
tempo but does not claim to produce intelligible rap audio.

## Scope and Limits

- This slice integrates through StreamMUSE's package, CLI distribution, `Tempo`
  model, and test conventions. It intentionally runs alongside rather than
  inside the accompaniment process.
- Rhythm input is represented by a preset today. Keyboard drum-hit capture,
  tempo tracking, live topic changes, a rolling lookahead buffer, TTS duration
  control, and audio mixing are next-phase work.
- The heuristic prosody analyser is sufficient for controlled prototype
  evaluation, not a production phonetic source. The JSON schedule makes errors
  visible and creates the boundary for a dictionary or forced-alignment upgrade.

## Error Handling

Invalid tempo, bars, candidate counts, unsupported patterns, and non-4/4
pattern dimensions are rejected at the CLI or planner boundary. Empty topic
text becomes `the moment`. Empty or unusable LLM output, HTTP failure, or a
candidate that cannot fit triggers deterministic fallback rather than ending
the run. JSON files are written only after a plan is successfully constructed.

## Validation

Unit tests cover irregular and heuristic syllable counts, accent-plan shape,
overflow rejection, deterministic selection, local-chat parsing/fallback, JSON
shape, and the CLI's default offline run. A smoke command validates all three
user-facing outputs: readable schedule, JSON artifact, and real-time terminal
playback. The existing tempo unit test is run before changes to establish a
clean baseline.
