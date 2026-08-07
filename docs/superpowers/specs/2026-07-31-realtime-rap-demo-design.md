# Real-Time Rap Demo Design

## Goal

Deliver one runnable, text-only prototype that continuously produces rap bars
against a prescheduled musical clock. The system must visibly demonstrate the
full path from topic and flow template through candidate generation,
phonetic/prosodic validation, ranking, bar commitment, and tick-level lyric
emission. It must keep running when online generation is slow or produces no
acceptable candidate.

This pass optimizes for an end-to-end demonstration. It does not synthesize a
voice, accept live drum or topic input, train a model, or import the complete
MCFlow corpus.

The implementation is also a research instrument. A successful run is not
only one that appears continuous; it is one whose inputs, intermediate
representations, decisions, timing, failures, and outputs can be inspected and
replayed after the session.

## Research Basis and Claim Boundary

The design combines ideas demonstrated separately in prior work:

- MCFlow supplies symbolic observations of performed rap syllables relative to
  meter. It motivates a template representation containing duration, stress,
  phrase break, and rhyme position, but it is not a generator or drum track.
- DeepBeat demonstrates candidate retrieval followed by feature-based ranking
  over rhyme, structure, and semantics. Our first ranker uses deterministic
  hand-set weights rather than claiming a learned rap-quality model.
- Shimon the Rapper demonstrates a complete real-time pipeline using CMU
  pronunciations, phoneme-level rhyme analysis, rule-based text-to-rhythm, and
  fallback-oriented latency management.
- DeepRapper motivates explicit beat markers inside a generation
  representation, while TeleMelody motivates a replaceable intermediate
  template between content generation and musical realization.

The defensible claim for this increment is therefore narrow: generated English
text can be validated, ranked, and scheduled continuously against symbolic rap
flow templates while every decision is observable. The system does not yet
know human rap quality, model expressive pronunciation, or prove that its
ranking correlates with listener preference.

## User-Facing Demo

A separate `streammuse-rap-demo` entry point starts a FastAPI server and the rap
timeline in one process. Its scenario specifies tempo, a sequence of topics,
and a sequence of symbolic flow-template names. Defaults provide a useful demo
without additional files or services.

The terminal continuously prints:

- bar and topic transitions;
- candidate generation completion and latency;
- candidate validity and component scores;
- the selected candidate or fallback reason; and
- every emitted syllable with bar, beat, and subdivision.

The browser is an observer for the same events. It shows the current topic,
tempo, active bar, timeline slots, emitted syllables, selected line, candidate
scores, source, fallback status, and recent history. It does not control the
runtime in this increment.

## Architecture

The existing Clean Architecture boundaries remain in place:

1. **Domain:** immutable flow templates, alignment diagnostics, candidate
   scores, committed bars, and observable runtime events.
2. **Application:** candidate validation/ranking, rolling lookahead, topic and
   template scheduling, and fallback policy.
3. **Infrastructure:** phrase-bank fallback, optional OpenAI-compatible local
   chat candidate generation, and an in-process event broadcaster.
4. **Presentation:** a standalone CLI/server entry point, terminal renderer,
   WebSocket endpoint, and static browser observer.

The standalone demo owns a `Tempo` clock but uses the same absolute tick model
as `RealTimeMusicService`. The flow and planning components therefore remain
usable by the active real-time music process later without maintaining a
second timing representation.

## Flow Templates

The prototype includes a small set of manually encoded, MCFlow-inspired
templates. A template is an ordered collection of lyric-bearing slots with:

- onset within a four-beat bar;
- duration in ticks;
- target stress strength;
- optional phrase-boundary strength; and
- optional rhyme endpoint label.

These templates represent vocal cadence relative to meter, not kick, snare, or
hi-hat accompaniment. They are anonymous structural examples and contain no
MCFlow source lyrics. A future adapter can replace the built-in catalog with
templates derived from licensed MCFlow files without changing the planner.

## Generation, Alignment, and Ranking

For every future bar, the candidate generator receives the current topic,
recent committed text, and the exact number of template slots. It returns a
batch of distinct one-bar candidates.

Each candidate is analyzed locally. For this first runnable system, the
existing dependency-free syllable estimator remains the default analyzer. Its
interface is isolated so CMUdict or a phonemizer can replace it later.

Validation and selection use two stages:

1. Hard validation requires non-empty text, exactly one analyzed syllable per
   flow slot, and no duplicate of a recently committed line.
2. Valid candidates are mapped sequentially to template slots and ranked with
   visible component scores for stress alignment, phrase-boundary fit, rhyme
   endpoint quality, topic coverage, continuity, and repetition penalty.

The initial weights are deterministic configuration, not learned parameters.
The total and all component values are included in runtime events. This makes
the prototype inspectable and lets later experiments compare scoring policies.

The first score is intentionally a documented baseline rather than an opaque
quality judgment. Exact syllable count is a hard gate. Among valid candidates,
the ranker combines normalized values for stress alignment, phrase-boundary
fit, endpoint rhyme, lexical topic coverage, lexical continuity, and novelty.
The session manifest records the exact weights. Dictionary pronunciation is
preferred; out-of-vocabulary words fall back to the existing vowel-group
heuristic and are identified in the logs. This is the best practical analyzer
for the first system, not a claim that canonical dictionary pronunciation
captures dialect, elision, or intentional pronunciation bending in rap.

## Rolling Runtime and Fallback

The controller keeps at least two future bars committed. Candidate generation
runs off the tick path and never blocks emission. A bar becomes immutable once
its first tick has started.

The fallback policy is deliberately simple:

1. Select the highest-scoring valid generated candidate when one meets the
   minimum score before the bar deadline.
2. Otherwise select a prevalidated phrase-bank candidate for the exact topic
   and template slot count.
3. Record `fallback=true` and one of `generation_error`, `generation_timeout`,
   `no_valid_candidate`, or `below_quality_threshold`.

A high rejection count does not force fallback if a strong candidate survives.
The observer reports total candidates, valid candidates, selected score,
generation latency, and cumulative fallback rate.

Topics and flow templates change only at bar boundaries according to the
scenario. The first implementation generates and commits one bar at a time;
the generator receives recent history so a later multi-bar generator can be
substituted without changing playback.

## Event Flow

```text
scenario + tempo clock
  -> rolling planner requests candidates for future bar
  -> analyzer validates and scores the batch against FlowTemplate
  -> best candidate or prevalidated fallback becomes CommittedBar
  -> tick loop emits ScheduledSyllable events
  -> terminal renderer + WebSocket event broadcaster
  -> browser observer
```

The event broadcaster is a narrow observer interface. The publisher assigns
sequence and timestamps on the producing thread, then places the immutable
event onto an unbounded in-process queue. A dedicated dispatcher performs
terminal, file, state-projection, and WebSocket work in order, outside the
musical tick path. Presentation failures are isolated from the clock, and a
newly connected browser receives a current state snapshot before live events.

## Research Observability

Every session creates a self-contained artifact directory with:

- `session.json`: scenario, tempo, generator and model identity, random seed,
  template provenance, scoring weights, dependency versions, and code revision;
- `events.jsonl`: the canonical ordered event stream with sequence number,
  UTC time, monotonic time, bar, tick, request ID, and typed payload;
- `summary.json`: aggregate counts and latency/quality statistics derived from
  the event stream rather than maintained independently; and
- `bars.csv`: one row per frozen bar for direct statistical analysis.

The event stream includes session lifecycle, planning requests, exact prompts,
raw model responses, parsed candidates, pronunciation analyses, every feature
score, rejection reasons, provisional fallback reservations, bar replacement
and freeze decisions, syllable emissions, generator errors, WebSocket state,
and shutdown. Secrets such as API keys are never recorded.

The live terminal prints the same events at human-readable detail levels. The
browser state is a projection of the canonical event stream, not a separate
source of truth. At minimum it exposes the current tick and bar, topic and
template, generator state, deadline slack, all candidates and component scores,
the selected source, fallback reason, pronunciation fallback words, emitted
syllables, recent frozen bars, and cumulative metrics.

Initial metrics are defined as follows:

- candidate validity rate = valid candidates / parsed candidates;
- fallback rate = fallback bars / frozen bars;
- deadline miss rate = generated batches completed after bar freeze / requests;
- generation latency = response completion monotonic time minus request start;
- deadline slack = planned bar start time minus response completion time;
- pronunciation fallback rate = heuristic-analyzed words / analyzed words;
- emission jitter = actual emission time minus the tick's planned wall time;
- score distributions = per-component and total scores for all candidates; and
- repetition rate = repeated normalized bigrams / generated bigrams over the
  configured recent-bar window.

These metrics describe system behavior. They do not establish lyrical quality.
A later listener study or annotated ranking dataset is required to validate
whether the scoring function predicts human judgments.

## Error Handling

- Candidate-generator exceptions and timeouts become fallback decisions.
- Malformed candidates are rejected with explicit reasons.
- A fallback catalog is validated at startup; the demo fails early only when
  it cannot guarantee a bar for a configured topic/template pair.
- WebSocket disconnects do not affect generation or playback.
- Shutdown stops the clock and background generator without waiting
  indefinitely for an HTTP request.

## Verification

Focused unit tests cover template construction, exact-slot validation,
component scoring, selection, fallback reasons, topic/template scheduling, and
immutable started bars. Runtime tests use a fake clock and scripted generators
to prove that slow or invalid generation cannot create a gap. Presentation
tests cover the state snapshot and serialized event contract.

Golden event-stream tests verify that a scripted session is replayable and
that `summary.json` and `bars.csv` can be regenerated solely from
`session.json` and `events.jsonl`. Property-style checks verify event sequence
monotonicity, one frozen decision per bar, no post-freeze replacement, and no
emitted syllable without a frozen bar and matching flow slot.

The final smoke test starts `streammuse-rap-demo`, observes multiple topic and
template transitions in terminal output, connects to the browser WebSocket,
and confirms that bars continue through a forced generator failure.
