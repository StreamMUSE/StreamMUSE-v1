# Rap Alignment Prototype

## Purpose

`streammuse-rap` is a text-first prototype for the proposed live rap system.
It turns a topic into several candidate rap-style lines, estimates their
syllables, assigns each syllable to a StreamMUSE musical tick, and exposes the
result as a terminal schedule and optional JSON artifact. It is intentionally
separate from the accompaniment process so that beat-aligned text can be judged
before adding speech synthesis, keyboard drum capture, or audio mixing.

This page documents the original standalone scheduler and its first integration
with `streammuse-cli`. For the newer continuous scenario runner, exact flow
context, candidate diagnostics, live terminal/browser monitors, canonical
artifacts, and reproducible analysis, see
[`research-realtime-rap.md`](research-realtime-rap.md). The newer runner is the
research entry point; this command remains useful as a small alignment smoke
test.

## Run It

The default path needs no model server:

```bash
uv run streammuse-rap --topic "space travel" --tempo 92 --pattern boom_bap --bars 4
```

Write the schedule a future speech component would consume:

```bash
uv run streammuse-rap --topic "space travel" --bars 2 --output-json rap-plan.json
```

Print the same syllable events at their planned times:

```bash
uv run streammuse-rap --topic "ocean research" --tempo 120 --pattern trap_sparse --bars 1 --play
```

An optional OpenAI-compatible local server can improve candidate language. It
never controls timing: failed, empty, malformed, or overfull responses fall
back to the phrase bank.

```bash
uv run streammuse-rap \
  --topic "space travel" \
  --generator local_chat \
  --model-url http://localhost:8000/v1 \
  --model local-model
```

## What the Output Means

Every row has an absolute StreamMUSE tick, `B<bar> <beat>.<tick-in-beat>`
position, seconds from the start, pattern accent, and syllable label. With the
current default (`4` ticks per beat and `4` beats per bar), one row represents
one sixteenth-note slot. A multi-syllable word appears on its first row and `.`
marks later syllables from that word. The JSON preserves the word, syllable
ordinal, stress estimate, accent, and exact timing fields.

The three patterns are deliberately small proxies for live drumming:

- `boom_bap`: strong downbeats with moderate backbeats.
- `straight_8`: regular eighth-note pulse.
- `trap_sparse`: displaced late-sixteenth accents around strong downbeats.

## How It Makes Alignment Decisions

The phrase bank or local model supplies multiple candidate lines. The local
prosody pass estimates syllables and treats the first syllable of each word as
stressed. A dynamic-programming scheduler then selects strictly increasing
slots, fixes the first syllable to the bar downbeat, rewards stressed syllables
on high-accent slots, and penalizes uneven progression through the bar. A line
with more than 16 syllables cannot produce a partial schedule and cannot beat a
fitting candidate.

This is an explicit prototype tradeoff: no `pronouncing`, `cmudict`, or
`pyphen` package is installed in the project environment, so the first version
uses a small irregular-word table plus vowel-group heuristic. Its output is
therefore useful for testable scheduling, not a production phonetic guarantee.

## What to Evaluate First

When trying the prototype, inspect these in order:

1. **Tick validity:** each syllable has one increasing tick, starts at the bar
   downbeat, and never spills into a nonexistent slot.
2. **Density:** lines should use enough of the 16 slots to sound continuous
   without exceeding the bar. Sparse or overfull lines are visible immediately
   in the schedule and JSON.
3. **Accent fit:** word onsets marked with `*` during `--play` should fall on
   the pattern's prominent accents often enough to create a plausible flow.
4. **Generation latency:** a live system needs lookahead. The local phrase bank
   provides deterministic instant candidates; measure the optional chat server
   separately before allowing it onto the audio path.
5. **Language quality:** topic relevance is intentionally secondary today. The
   local model may improve it, but the scheduler remains the quality gate.

## Real-Time CLI Mode

The same schedule now runs inside the active StreamMUSE process when
`--rap-topic` is present:

```bash
uv run streammuse-cli \
  --input-mode keyboard \
  --rap-topic "space travel" \
  --rap-pattern boom_bap \
  --rap-generator phrase_bank \
  --rap-lookahead-bars 2
```

`RealTimeMusicService` calls the rolling controller immediately after its
normal tick output. The controller preloads phrase-bank fallback bars and
emits each aligned syllable at that same absolute tick. With
`--rap-generator local_chat`, one background request at a time may improve a
future bar, but the request never blocks a tick and cannot replace a bar once
it has started. Omit `--rap-topic` to leave the existing music flow unchanged.

## Current Limits and Next Increment

The real-time mode still does not derive accents from keyboard drum hits,
estimate variable tempo, accept live topic changes, synthesize audio, or mix
voice into accompaniment output. It uses the preset patterns and heuristic
prosody from the standalone command. The next useful increment is a
timestamped drum-hit buffer that updates the accent-weighted slot plan while
preserving a one- to two-bar lyric lookahead, followed by a
duration-controlled TTS layer.

The continuous research runner now addresses lookahead generation, immutable
bar freezing, fallback continuity, and structured observation, but it does not
remove the core scientific limitations above. Its dictionary/OOV prosody,
fixed sixteenth-note grid, lexical score proxies, and hand-authored live
templates remain engineering approximations. No current result validates
human-perceived rap quality or expressive audio timing.

## Development Record

The prototype design and implementation plan are recorded in:

- `docs/superpowers/specs/2026-07-17-rap-alignment-prototype-design.md`
- `docs/superpowers/plans/2026-07-17-rap-alignment-prototype.md`

The design compared prompt-only metre requests, a pronunciation-dictionary
adapter, and candidate generation followed by deterministic scheduling. The
last approach was selected because it works offline and makes every timing
decision inspectable.

### Observed Results: 2026-07-17

- Baseline shared-timing check: `uv run pytest
  tests/unit/domain/timing/test_tempo.py -q` reported `12 passed` before the
  prototype was added.
- Environment check: `pronouncing`, `cmudict`, and `pyphen` are all absent
  from the active project environment. The implementation therefore adds no
  package or network requirement.
- Final focused/regression verification:

  ```bash
  uv run pytest \
    tests/unit/domain/rap \
    tests/unit/application/rap \
    tests/unit/infrastructure/rap \
    tests/unit/presentation/rap \
    tests/unit/domain/timing/test_tempo.py \
    tests/unit/infrastructure/inference/test_local_chat_model_client.py -q
  ```

  Result: `43 passed in 0.28s`.

- Full repository regression gate: `uv run pytest tests/ -q --tb=no` reported
  `292 passed, 1 skipped in 7.44s`. The only warning is the existing
  `pretty_midi` notice that `pkg_resources` is deprecated.

- Offline smoke path: `space travel` at 92 BPM with `boom_bap` rendered a
  15-syllable candidate on ticks `0` through `15` with one intentional free
  slot, and wrote `/tmp/rap-alignment-space-travel.json`. The artifact includes
  the future voice-adapter fields documented above.
- Alternate-pattern/timed path: `ocean research` at 120 BPM with `trap_sparse`
  rendered a 16-syllable line and then emitted the same 16 scheduled events
  over 1.875 seconds with `--play`.
- Local-chat failure path: no server was listening on `localhost:8000`; the
  command printed the connection failure as a warning and immediately produced
  the offline phrase-bank plan. The adapter failure test covers the same
  fallback behaviour without relying on a running server.

Two smoke runs surfaced prosody errors that were fixed with red-green tests:
the `-s` forms `moves` and `makes` were initially overcounted, and the original
`-le` heuristic treated `while`, `whole`, and `file` as syllabic. The analyser
now keeps calibrated exceptions for the template verbs and distinguishes
silent-e `-le` endings from consonant-plus-syllabic `-le` endings such as
`table` and `simple`.

### Real-Time Integration Result: 2026-07-17

- Design and execution records:
  - `docs/superpowers/specs/2026-07-17-realtime-rap-scheduler-design.md`
  - `docs/superpowers/plans/2026-07-17-realtime-rap-scheduler.md`
- Focused test gate after integration:

  ```bash
  uv run pytest \
    tests/unit/application/test_real_time_music_service.py \
    tests/unit/application/rap/test_realtime.py \
    tests/unit/presentation/test_cli_config_parser.py \
    tests/integration/test_cli_entry_point.py -q
  ```

  Result: `41 passed`. These tests cover the tick-hook order, initial and
  rolling fallback bars, non-blocking primary generation, late-result
  rejection, parser mapping, and CLI wiring.

- End-to-end finite run used `scripts/fake_inference_server.py` and:

  ```bash
  uv run streammuse-cli \
    --input-mode list \
    --tempo 300 \
    --rap-topic "space travel" \
    --rap-pattern straight_8 \
    --rap-lookahead-bars 2 \
    --max-ticks 20 \
    --log-dir /tmp/streammuse-rap-realtime-smoke
  ```

  The console emitted `[RAP B1 ...]` events on ticks `0` through `15`, then
  `[RAP B2 1.1]` at tick `16`, directly following each corresponding `[tick]`
  line. Accompaniment requests continued concurrently with observed round-trip
  times of about `25` to `56` ms. The fake server was stopped after the run.

- Full repository regression after the integration: `uv run pytest tests/ -q
  --tb=no` reported `300 passed, 1 skipped in 25.62s`. The only warning is the
  existing `pretty_midi` `pkg_resources` deprecation notice.

### H200 Smoke Result: 2026-07-17

The integrated path was also run in an isolated H200 `/tmp` source overlay
using the existing `streammuse-isochron` Python 3.10 environment and the fake
inference server. It used the same finite command shape as the local smoke
test: `--input-mode list --tempo 300 --rap-topic "space travel" --rap-pattern
straight_8 --rap-lookahead-bars 2 --max-ticks 20`.

The H200 console trace emitted the first bar's aligned syllables on ticks `0`
through `15`, then emitted `[RAP B2 1.1] we*` at tick `16`. Fake accompaniment
round trips were `14.7` to `19.5` ms. H200 GPUs `0`, `1`, and `2` remained at
`0 MiB`; the other GPUs already had unrelated workloads and were not touched.
The temporary fake server was stopped after the run. This is intentionally a
CPU-only scheduler test, not a GPU model-generation benchmark.

### H200 Real Lyric-Server Result: 2026-07-17

The H200 host does not contain a compatible Stanley/RoFormer accompaniment
checkpoint, so a real music-inference server cannot currently be started.
It does contain the cached `Qwen/Qwen2.5-0.5B-Instruct` weights and vLLM
`0.24.0`. The model was served on otherwise idle GPU `0` at
`http://127.0.0.1:8001/v1`, using `VLLM_USE_FLASHINFER_SAMPLER=0` to avoid the
environment's missing `ninja` executable required by FlashInfer JIT sampling.

The standalone `local_chat` generator accepted Qwen output with
`candidate_source: local_chat` and scheduled, for example, `Climbing the walls
of space, we soar high` across ticks `0` through `15`. A full CLI run then used
the real Qwen endpoint for `--rap-generator local_chat` while the fake endpoint
continued to stand in only for accompaniment. Bar 1 used the guaranteed
phrase-bank fallback; the asynchronous Qwen result replaced future bar 2 with
the model-generated line `our journey is an adventure in time` without missing
tick `16`. GPU 0 used about `7.8 GiB` while serving and returned to `0 MiB`
after the temporary server was stopped.
