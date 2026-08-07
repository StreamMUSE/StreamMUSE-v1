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
