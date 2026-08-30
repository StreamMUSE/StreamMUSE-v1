# Research Real-Time Rap Prototype

## Scope

`streammuse-rap-demo` is an observable, text-first experiment for continuous
beat-aligned lyric planning. A scenario supplies tempo, topics, flow templates,
and prevalidated fallback lines. A candidate generator proposes text for a
future bar; deterministic prosody, exact syllable gating, alignment, and
ranking decide whether that text may replace the fallback. The clock then
freezes and emits the chosen syllables at the template ticks.

The prototype demonstrates an inspectable symbolic real-time pipeline. It does
not establish that the selected text sounds like good rap to listeners. There
is currently no speech synthesis, live drum input, or human quality label in
the decision loop.

## Run Sessions

All commands below use the built-in looping 90 BPM scenario: four bars about
space, four about the deep sea, and four about code, each with a different
hand-authored nine-slot flow template. Each run creates a unique directory
under `logs/rap/`.

### Deterministic phrase bank

This is the offline baseline. It requires no model server and is the first
command to use when checking timing, fallback, recording, or monitor behavior.

```bash
uv run streammuse-rap-demo \
  --generator phrase_bank \
  --candidate-count 8 \
  --lookahead-bars 2 \
  --minimum-score 0.55 \
  --seed 20260807 \
  --max-bars 12 \
  --terminal-detail full \
  --log-dir logs/rap \
  --port 8012
```

### Real local-chat model

Start an OpenAI-compatible server. This vLLM example uses the same served name
as the demo defaults; substitute a locally available instruct model when
needed.

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --host 127.0.0.1 \
  --port 8001 \
  --served-model-name qwen-rap
```

In a second terminal, run:

```bash
uv run streammuse-rap-demo \
  --generator local_chat \
  --model-url http://127.0.0.1:8001/v1 \
  --model qwen-rap \
  --timeout-s 5 \
  --candidate-count 8 \
  --lookahead-bars 2 \
  --minimum-score 0.55 \
  --seed 20260807 \
  --max-bars 12 \
  --terminal-detail full \
  --log-dir logs/rap \
  --port 8012
```

The model is a candidate source, not the timing authority. Its exact prompt,
raw response, latency, parsing result, and every candidate evaluation are
recorded. A model failure, late response, invalid candidate set, or score below
the threshold leaves the prevalidated fallback in place.

### Forced generator failure

This mode deliberately exercises the continuity path without depending on a
network failure:

```bash
uv run streammuse-rap-demo \
  --generator scripted_failure \
  --lookahead-bars 2 \
  --seed 20260807 \
  --max-bars 12 \
  --terminal-detail full \
  --log-dir logs/rap \
  --port 8012
```

All bars should still freeze and emit from the fallback catalog. Treat missing
bars as a system defect; frequent fallback in this deliberate run is expected.

### Continuous run

`--max-bars 0` runs a looping scenario until interrupted. Use `Ctrl-C` for a
clean shutdown and artifact derivation.

```bash
uv run streammuse-rap-demo \
  --generator local_chat \
  --model-url http://127.0.0.1:8001/v1 \
  --model qwen-rap \
  --max-bars 0 \
  --terminal-layout auto \
  --terminal-detail full \
  --log-dir logs/rap \
  --port 8012
```

`--max-bars 0` is valid only for looping scenarios. Pass `--scenario PATH` to
use a validated JSON schedule. Non-looping schedules require a finite
`--max-bars` no larger than their declared length.

## Read the Live Monitors

### Terminal

`--terminal-layout auto` uses the dense Rich view on a sufficiently wide TTY
and structured append-only lines when output is redirected or narrow. Force a
mode with `split` or `stream`. `--terminal-detail summary`, `candidates`, and
`full` progressively add the candidate gate, exact prompts, raw model response,
score components, provenance, and event trace.

The performance side answers what is committed to the clock:

- **Reserved/armed** is the prevalidated fallback available before generation.
- **Replaced** means a candidate passed the gate, ranking, threshold, and
  deadline while the bar was still mutable.
- **Frozen** is authoritative for playback; later responses cannot alter it.
- **Flow** shows all 16 sixteenth-note positions, the chosen slot ticks, target
  stresses, boundaries, rhyme group, scheduled syllables, and active tick.
- **Health** reports planning state, latency, deadline slack, jitter, and
  fallback activity. These diagnose real-time behavior, not lyric quality.

The research side answers why a line was chosen:

- **Request** includes target bar, topic, seed, candidate count, recent context,
  and the complete immutable flow template used by alignment and scoring.
- **Prompt** is the exact system/user context sent to local chat; the template
  ID alone is not used as a substitute for slot timing.
- **Batch** contains source, raw response, candidate count, latency, late flag,
  warning, and generator error.
- **Candidates** retain valid and rejected lines, syllable analyses, OOV source,
  rejection reasons, every weighted score component, and selected state.
- **Trace** is a bounded projection of the ordered canonical event stream. The
  complete stream remains in `events.jsonl`.

### Browser

Unless `--no-web` is passed, open [http://127.0.0.1:8012](http://127.0.0.1:8012)
for the read-only monitor. `--host` and `--port` select another bind address.
The first viewport presents the same clock, flow, committed line, generation
health, and aggregate rates as the terminal; lower sections expose candidates,
frozen-bar history, and chronological events.

The browser receives a full snapshot first and then ordered canonical events
over `/ws`. `GET /api/state` returns the current JSON-safe projection and
`GET /api/session` returns session identity, artifact directory, and non-secret
runtime metadata. JavaScript does not plan, score, freeze, or align lyrics. A
reconnect obtains a fresh snapshot, so the browser is an observer rather than
experimental state.

## Candidate Gate and Score

A candidate is scored only when its analyzed syllable count exactly equals the
flow slot count and its normalized text does not duplicate prior frozen text.
Processing failures and hard rejections remain evidence; they are not silently
dropped. Valid candidates receive six component values in `[0, 1]`:

| Component | Default weight | Exact implementation |
|---|---:|---|
| `stress_alignment` | 0.30 | `1 - weighted absolute stress error / total slot weight`; primary, secondary, and unstressed lexical syllables map to `1.0`, `0.5`, and `0.0`, while stronger target slots carry more error weight. |
| `boundary_fit` | 0.10 | Mean over slots with a boundary target: `1.0` for punctuation after the syllable, `0.6` for a word ending, otherwise `0.0`; returns `1.0` when there are no targets. |
| `rhyme_quality` | 0.20 | `0.5` before a segment rhyme anchor exists, `1.0` for an exact end-phone tail, `0.6` for the same vowel-phone sequence, otherwise `0.0`. |
| `topic_coverage` | 0.20 | Fraction of normalized non-stopword topic tokens present in the candidate; returns `1.0` for an empty content-token target. |
| `lexical_continuity` | 0.15 | Non-topic content words shared with the two most recent bars, divided by two and capped at `1.0`; returns `0.5` with no history. |
| `novelty` | 0.05 | One minus the maximum word-bigram Jaccard similarity to any of the four most recent bars; returns `1.0` with no history. |

The total is `sum(component value * recorded weight)`. The highest valid total
wins; equal totals preserve generator order. If no candidate is valid or the
best total is below `minimum_score`, selection returns an explicit fallback
reason. These formulas are transparent engineering proxies. In particular,
lexical overlap is not semantic coherence, token coverage is not topical
quality, and phone-tail similarity is not a listener judgment of rhyme.

## Recorded Metrics

`summary.json` stores every rate as `{numerator, denominator, rate}`. `rate` is
`null` when its denominator is zero. Request, candidate, and bar identities are
deduplicated before aggregation.

| Metric | Numerator | Denominator |
|---|---|---|
| `candidate_validity` | Unique candidate evaluations whose `valid` field is true, for requests with a received batch. | Sum of each unique batch's declared `candidate_count`; when absent, the number of unique evaluations for that request. |
| `fallback` | Unique frozen bars whose committed payload has `fallback: true`. | All unique frozen bars. |
| `deadline_miss` | Unique planned requests whose received batch has `late: true`. | All unique `bar_planning_started` request IDs. |
| `generator_error` | Unique planned requests with a batch `error_type` or a `generation_failed` event. | All unique `bar_planning_started` request IDs. |
| `pronunciation_fallback` | Analyzed words whose recorded source is not `cmudict_first_pronunciation`. | Analyzed words with a recognized string-valued pronunciation source in unique candidate evaluations. |
| `repetition` | Normalized bigram occurrences in a frozen bar that appeared in any of the preceding `repetition_window_bars` frozen bars. | All normalized bigram occurrences in frozen-bar text. |

`bars.fallback_rate` repeats the fallback rate for convenient top-level use.
Latency distributions include `count`, linearly interpolated `p50` and `p95`,
and `max`:

- `generation_latency_ms`: numeric latency observations on unique candidate
  batches.
- `deadline_slack_ms`: numeric batch slack observations; negative values are
  already late.
- `emission_jitter_ms`: numeric jitter observations on emitted syllables.

Fallback rate measures how often the safety path was committed. It combines
model failure, deadline pressure, hard-gate rejection, and threshold policy;
it does not measure whether fallback or generated lyrics sound better. Latency
and deadline rates measure system timing under the recorded hardware/model
configuration. They do not measure rhythm naturalness, intelligibility, rhyme,
topic relevance, or overall rap quality.

## Artifacts and Regeneration

Each session directory contains:

- `session.json`: scenario, seed, tempo, exact used templates and provenance,
  generator/model identity, weights, threshold, timeout, lookahead, repetition
  window, environment, package version, and Git revision/dirty state.
- `events.jsonl`: append-first canonical ordered evidence. A truncated final
  JSON prefix from a process crash is recoverable; interior corruption,
  sequence gaps, duplicate sequence values, and mixed session IDs are errors.
- `summary.json`: deterministic aggregate metrics derived at clean close.
- `bars.csv`: deterministic one-row-per-frozen-bar evidence.

Regenerate derived artifacts without changing the originals:

```bash
uv run python scripts/summarize_rap_session.py logs/rap/<session-id>
```

This writes `summary.regenerated.json` and `bars.regenerated.csv`. The analyzer
uses `session.json`'s `repetition_window_bars` and rejects disagreement with the
canonical `session_started` event. It calls the same derivation and writer APIs
as the recorder, so matching evidence produces JSON identical apart from a
possible final newline and data-equivalent CSV rows.

For a strict local comparison:

```bash
diff -u \
  logs/rap/<session-id>/summary.json \
  logs/rap/<session-id>/summary.regenerated.json
diff -u \
  logs/rap/<session-id>/bars.csv \
  logs/rap/<session-id>/bars.regenerated.csv
```

To compare experiments, hold scenario contents, seed, generator/model identity
and parameters, score weights, threshold, timeout, lookahead, repetition
window, and code revision constant. Compare the manifests first, then rates and
latency distributions, then candidate/bar evidence. A local-chat seed is not a
complete reproducibility guarantee when the serving engine, model revision,
hardware, or sampling implementation differs. Phrase-bank runs are the
deterministic control.

## Anonymous MCFlow Structure

MCFlow-derived templates are opt-in local inputs. Extraction records anonymous
structural timing, stress, boundary, rhyme, source hash, and quantization error;
it does not copy lyrics or performer identity into the catalog.

```bash
uv run python scripts/extract_mcflow_templates.py \
  --mcflow-dir /path/to/local/mcflow \
  --output /tmp/mcflow-templates.json \
  --max-quantization-error-ticks 0.25

uv run python scripts/build_mcflow_sample_catalog.py \
  --mcflow-dir /path/to/local/mcflow \
  --catalog-output /tmp/mcflow-sample-catalog.json \
  --report-output /tmp/mcflow-sample-report.json \
  --per-bucket 10
```

The current live demo still uses the recorded hand-authored built-in templates.
The anonymous catalog supports template research and provenance inspection; it
is not evidence that MCFlow lyrics trained or conditioned the generator.

## Limitations and Claims

### Demonstrated by the prototype

- A monotonic clock can reserve, replace, freeze, and emit complete symbolic
  bars while primary generation occurs off the tick path.
- Every emitted syllable is assigned to one recorded slot in the frozen flow
  template; fallback preserves bar continuity when generation cannot be used.
- Exact model context, raw responses, analyses, hard gates, score components,
  timing decisions, and fallback reasons can be monitored and recorded.
- Aggregate summaries and bar tables can be regenerated from canonical events
  under the recorded manifest contract.

### Not yet demonstrated

- Human-perceived rap quality, natural flow, groove, rhyme quality, topical
  coherence, intelligibility, or preference for the selected candidate.
- Generalization across performers, dialects, genres, tempi, triplets, or live
  drum patterns.
- Expressive pronunciation or audio synchronization.

Current technical limitations are:

1. Pronunciation primarily uses a dictionary and takes one pronunciation
   choice rather than modeling contextual or dialect-dependent alternatives.
2. Out-of-vocabulary words use a vowel-group heuristic; syllable count, stress,
   and rhyme phones can therefore be wrong.
3. Topic coverage and continuity are lexical overlap proxies, not semantic or
   discourse models.
4. The live templates are hand-authored; anonymous MCFlow extraction has not
   yet become an automatically sampled live template source.
5. Timing is quantized to 16 subdivisions per 4/4 bar. Triplets, swing
   microtiming, variable tempo, and expressive duration are not represented.
6. There is no duration-controlled or expressive audio/TTS output.
7. There is no live rhythm/topic input in this experiment and no human
   validation of the ranking target.

The next research step is to show listeners paired candidates with the same
topic and flow, collect pairwise preferences for flow naturalness, relevance,
and overall quality, and fit a learned ranker to the already logged candidate
features. Evaluation must use held-out sessions/listeners and compare the
learned ranker with the transparent fixed-weight baseline. Until that study,
score totals should be reported as proxy values rather than quality scores.
