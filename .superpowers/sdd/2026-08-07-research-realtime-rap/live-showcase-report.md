# Live Rap Showcase Vertical Slice

## Status

Implemented the complete symbolic realtime rap research prototype. The runtime prevalidates fallbacks before ticks, keeps generation/prosody/ranking off the tick path, freezes each bar before emission, retains fallback on invalid/error/late results, and stops finite sessions cleanly.

The fast-generator planning frontier is bounded. With `lookahead_bars=2`, bar 0 activity can reserve at most bars 0 through 2 and plan bars 1 and 2. A finite `--max-bars 2` session reserves/plans only bars 0 and 1.

## Implemented

- Scenario-aware `RollingRapController` using `RapScenario`, built-in templates, CMU prosody, transparent ranking, score thresholds, and prevalidated fallback lines.
- Canonical `RapEventType`/`RapEvent` plus queue-backed ordered publisher/dispatcher.
- Absolute-monotonic `RapTickLoop`, finite/unbounded lifecycle, terminal rendering, dependency assembly, session manifest, and `streammuse-rap-demo` entry point.
- Generator modes `phrase_bank`, `local_chat`, and `scripted_failure`.
- Compatibility assembly for the existing main StreamMUSE CLI and unchanged `streammuse-rap` prototype.
- Canonical state projector, JSONL recorder, session manifest, summary metrics, bar CSV, Rich terminal dashboard/stream, WebSocket observer, and read-only responsive web monitor.
- Exact prompt, template, candidate gate, score components, fallback, latency, deadline, and syllable-emission observability.

## Verification

Final repository suite:

```text
uv run pytest tests/ -q --tb=short
977 passed, 4 skipped, 1 pre-existing pretty_midi/pkg_resources warning
```

Bounded-lookahead regressions:

```text
uv run pytest tests/unit/application/rap/test_realtime.py tests/unit/presentation/rap_demo/test_cli.py -q
11 passed
```

Phrase-bank smoke:

```text
uv run streammuse-rap-demo --generator phrase_bank --max-bars 2 --candidate-count 4 --terminal-detail candidates --log-dir /tmp/streammuse-rap-bounded --no-web
```

Observed: reservations only for bars 0 and 1, one plan for bar 1, four explicit syllable-count rejections, two fallback freezes, 18 syllable emissions, and `SESSION stop`.

Local-chat smoke command:

```text
uv run streammuse-rap-demo --generator local_chat --model-url http://127.0.0.1:8001/v1 --model qwen-rap --candidate-count 8 --lookahead-bars 2 --minimum-score 0.55 --max-bars 2 --terminal-detail full --log-dir /tmp/streammuse-rap-qwen --no-web
```

The endpoint refused the connection from this execution context. The demo visibly logged `generation_error`, retained the fallback, emitted both complete bars, and stopped cleanly. A follow-up `curl -sS --max-time 3 http://127.0.0.1:8001/v1/models` also exited 7.

## Current Boundaries

- This prototype schedules symbolic syllables and terminal/web visualization; live drum input, TTS/vocal synthesis, and audio output remain future integrations.
- Syllable/stress analysis and ranking are transparent heuristics over CMU pronunciations and flow targets, not a learned model of rap quality.
- The first bar uses a prevalidated fallback by design so playback can begin without waiting for the first model response.

## Final H200 Qwen Verification (2026-08-08)

The final implementation was archived at `ffb5bc74`, copied to the isolated H200 directory `/data/home/Andrew.Yang/StreamMUSE/rap_demo_runs/real_rap_ffb5bc74`, and installed editable without modifying the existing `StreamMUSE-v1` checkout. Qwen2.5-7B-Instruct was served as `qwen-rap-gpu4` on H200 GPU 4 at `http://127.0.0.1:8002/v1`; GPUs 1, 3, 5, and 6 were unused at launch.

```text
streammuse-rap-demo --generator local_chat \
  --model-url http://127.0.0.1:8002/v1 --model qwen-rap-gpu4 \
  --timeout-s 3 --candidate-count 8 --lookahead-bars 2 \
  --minimum-score 0.3 --seed 20260808 --max-bars 3 \
  --terminal-layout stream --terminal-detail full \
  --log-dir /data/home/Andrew.Yang/StreamMUSE/rap_demo_runs/real_rap_ffb5bc74/h200_logs \
  --no-web
```

Observed session: `rap-20260808T040122Z-97a25a2b`.

- Every planning flow summary displayed ticks `[0, 2, 3, 5, 7, 8, 10, 13, 15]`, durations `[2, 1, 2, 2, 1, 2, 3, 2, 1]`, stress `[1.0, 0.2, 0.7, 0.2, 0.6, 1.0, 0.2, 0.7, 0.9]`, boundary strengths `[0, 0, 0, 0, 0, 0, 0, 0, 3]`, and rhyme group `A`. Exact model context also showed the template ID, tick/stress/boundary/rhyme arrays, notation `S . w M | . w . M | S . w . | . M . S`, recent lines, and deterministic seed.
- The model returned four parsed candidates in each of two rounds despite eight requested. Three of eight parsed candidates passed the exact nine-syllable gate (37.5% validity).
- Bar 2 selected `Dreams of discovery never die` at total score `0.411`; bar 3 selected `out past planets where silence calls deep` at `0.342`.
- Generation latency was 199.8 ms and 217.5 ms (p50 208.7 ms); no deadline miss or generator error occurred.
- Model diagnostics were visible and persisted: round usage was `332/29` and `339/34` prompt/completion tokens, with `requested_8_received_4` warnings. The summary totals are 671 prompt tokens, 63 completion tokens, and two warnings.
- Three bars froze with one initial fallback, for a 33.3% fallback rate. All 27 scheduled syllables emitted.
- Emission jitter p50 was 0.289 ms, p95 0.479 ms, and maximum 2.842 ms.
- Artifacts contain 98 canonical events, three `bars.csv` rows, a validated manifest, deterministic cumulative metrics, one pronunciation fallback out of 50 analyzed words, and an unredacted `output_length_policy`.
- Because deployment used a source archive rather than a Git checkout, the session manifest reports unknown Git state; this report binds the archive and run directory to local commit `ffb5bc74`.

The current system now includes the recorder, canonical projector, derived metrics, Rich terminal, WebSocket monitor, read-only web UI, and presentation-error events. This run demonstrates real-time symbolic generation and deadline-safe selection; it does not validate human-perceived rap quality.
