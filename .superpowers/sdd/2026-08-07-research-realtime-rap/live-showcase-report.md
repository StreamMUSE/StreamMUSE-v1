# Live Rap Showcase Vertical Slice

## Status

Implemented the urgent terminal-only Tasks 7/8 vertical slice with the minimal Task 6 event publisher/dispatcher. The runtime prevalidates built-in fallbacks before ticks, keeps generation/prosody/ranking off the tick path, freezes before emission, retains fallback on invalid/error/late results, and stops finite sessions cleanly.

The fast-generator planning frontier is bounded. With `lookahead_bars=2`, bar 0 activity can reserve at most bars 0 through 2 and plan bars 1 and 2. A finite `--max-bars 2` session reserves/plans only bars 0 and 1.

## Implemented

- Scenario-aware `RollingRapController` using `RapScenario`, built-in templates, CMU prosody, transparent ranking, score thresholds, and prevalidated fallback lines.
- Canonical `RapEventType`/`RapEvent` plus queue-backed ordered publisher/dispatcher.
- Absolute-monotonic `RapTickLoop`, finite/unbounded lifecycle, terminal rendering, dependency assembly, session manifest, and `streammuse-rap-demo` entry point.
- Generator modes `phrase_bank`, `local_chat`, and `scripted_failure`.
- Compatibility assembly for the existing main StreamMUSE rap option and unchanged `streammuse-rap` prototype.
- No recorder, derived metrics, state projector, CSV, WebSocket, or web UI.

## Verification

Focused and affected suite:

```text
uv run pytest tests/unit/domain/rap tests/unit/application/rap tests/unit/infrastructure/rap tests/unit/presentation/rap tests/unit/presentation/rap_demo tests/unit/presentation/test_cli_config_parser.py tests/integration/test_cli_entry_point.py -q
186 passed, 1 warning
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

## Concerns

- Successful Qwen candidate evaluation/replacement was not observed because the local endpoint was unreachable during final smoke verification.
- Session output intentionally contains only `session.json`; recorder/JSONL/summary/CSV artifacts are deferred by the urgent scope.
- The terminal dispatcher disables a failing sink but does not yet republish a `presentation_error`; full Task 6 remains deferred.

## Final H200 Qwen Verification (2026-08-08)

This section supersedes the earlier urgent-slice limitations above. The final committed tree was archived at `ad61b7a5`, copied to the isolated H200 directory `/data/home/Andrew.Yang/StreamMUSE/rap_demo_runs/real_rap_ad61b7a5`, and installed editable without modifying the existing `StreamMUSE-v1` checkout. Qwen2.5-7B-Instruct was already served as `qwen-rap-gpu4` on H200 GPU 4 at `http://127.0.0.1:8002/v1`; GPUs 1, 5, and 6 were unused.

```text
streammuse-rap-demo --generator local_chat \
  --model-url http://127.0.0.1:8002/v1 --model qwen-rap-gpu4 \
  --timeout-s 3 --candidate-count 8 --lookahead-bars 2 \
  --minimum-score 0.3 --seed 20260807 --max-bars 3 \
  --terminal-layout stream --terminal-detail full --log-dir h200_logs --no-web
```

Observed session: `rap-20260808T032245Z-4722d150`.

- Every planning and model row displayed the same `baseline_syncopated_9` template: ticks `[0, 2, 3, 5, 7, 8, 10, 13, 15]`, stress `[1.0, 0.2, 0.7, 0.2, 0.6, 1.0, 0.2, 0.7, 0.9]`, boundary strengths `[0, 0, 0, 0, 0, 0, 0, 0, 3]`, rhyme group `A`, and notation `S . w M | . w . M | S . w . | . M . S`.
- The model returned four parsed candidates in each of two rounds despite eight requested. Two of eight parsed candidates passed the exact nine-syllable gate (25% validity).
- Bar 2 selected `Dreaming of distant worlds, out of sight` at total score `0.495`; bar 3 selected `out past planets where silence calls loud` at `0.342`.
- Generation latency was 290.1 ms and 246.9 ms (p50 268.5 ms); no deadline miss or generator error occurred.
- Three bars froze with one initial fallback, for a 33.3% fallback rate. All 27 scheduled syllables emitted.
- Emission jitter p50 was 0.268 ms, p95 0.764 ms, and maximum 3.916 ms.
- Artifacts contain 98 canonical events, three `bars.csv` rows, a validated manifest, and deterministic summary data.

The current system now includes the recorder, canonical projector, derived metrics, Rich terminal, WebSocket monitor, read-only web UI, and presentation-error events. This run demonstrates real-time symbolic generation and deadline-safe selection; it does not validate human-perceived rap quality.
