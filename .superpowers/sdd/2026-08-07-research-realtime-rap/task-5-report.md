# Task 5 Implementation Report: Request-Aware Generation and Prevalidated Fallback

## Status

Implemented Task 5 on `feature/real_rap`. Generation is now request-aware and
observable, and scenario fallback lines can be prevalidated before the realtime
path uses them.

## Changed Files

- `src/streammuse/domain/rap/models.py`
  - Added frozen `CandidateRequest` and enriched frozen `CandidateBatch`.
  - Validates request dimensions, batch diagnostics, and public value shapes.
- `src/streammuse/domain/rap/__init__.py`
  - Exports `CandidateRequest`.
- `src/streammuse/application/rap/service.py`
  - Replaced the positional generator protocol with `generate(request)` and
    migrated `RapPrototypeService`.
- `src/streammuse/application/rap/realtime.py`
  - Minimally migrated `RollingRapController` generator calls to complete
    `CandidateRequest` values; Task 7 remains responsible for its redesign.
- `src/streammuse/infrastructure/rap/generators.py`
  - Implemented request-aware phrase-bank and local-chat generation, prompt
    diagnostics, raw response/timing/token propagation, non-filtering parsing,
    and sanitized explicit error batches.
- `src/streammuse/infrastructure/rap/fallback.py`
  - Added `PrevalidatedFallbackCatalog` and explicit
    `PrevalidatedFallbackLine(source="prevalidated_fallback")`.
- `src/streammuse/infrastructure/rap/__init__.py`
  - Exports fallback types.
- `src/streammuse/presentation/rap/cli.py`
  - Removed the obsolete hidden-local-chat fallback constructor argument.
- `src/streammuse/presentation/cli/cli.py`
  - Keeps phrase bank as the controller fallback while constructing local chat
    without an internal fallback.
- `tests/unit/infrastructure/rap/test_generators.py`
  - Covers request/prompt structure, raw diagnostics, explicit errors,
    authorization redaction, phrase-bank batches, and parsing behavior.
- `tests/unit/infrastructure/rap/test_fallback.py`
  - Covers validation, round-robin selection, repeated normalized keys, source,
    and missing lookup behavior.
- `tests/unit/application/rap/test_service.py`
- `tests/unit/application/rap/test_realtime.py`
  - Migrated generator doubles to the new protocol.

## RED/GREEN Evidence

RED command:

```text
UV_CACHE_DIR=/tmp/streammuse-uv-cache uv run python -m pytest \
  tests/unit/infrastructure/rap/test_generators.py \
  tests/unit/infrastructure/rap/test_fallback.py -v
```

It failed at collection as intended because `CandidateRequest` and
`streammuse.infrastructure.rap.fallback` did not exist. This proved the new
tests exercised the missing public contract rather than pre-existing behavior.

GREEN command after implementation:

```text
UV_CACHE_DIR=/tmp/streammuse-uv-cache uv run python -m pytest \
  tests/unit/infrastructure/rap/test_generators.py \
  tests/unit/infrastructure/rap/test_fallback.py -v
# 9 passed
```

## Request and Prompt Contract

`CandidateGenerator.generate` now accepts exactly one `CandidateRequest`:
`request_id`, absolute `target_bar`, `topic`, `template_id`, required syllable
count, candidate count, frozen context lines, and seed. The request validates
nonempty identifiers/topics/templates, a nonnegative bar, positive dimensions,
tuple-of-string history, and an integer seed.

The local-chat prompt includes request id, target bar, template id, topic,
candidate count, the literal exact spoken-syllable requirement, frozen history,
and deterministic seed. For example, a nine-slot request tells the model to
produce lines with `exactly 9 spoken syllables` and includes prior text such as
`stars cross the night`.

`CandidateBatch` preserves the request id, parsed source-order candidates,
source, full non-secret prompt, `ChatModelResponse.text` exactly as
`raw_response`, latency, prompt-token count, completion-token count, warning,
and explicit error diagnostics. Candidate parsing only removes list markers,
empty lines, and duplicate lines while preserving source order. It does not
perform syllable filtering; Task 4 scoring owns structural validity.

## Explicit Errors and Raw Diagnostics

`LocalChatCandidateGenerator` no longer accepts or invokes a phrase-bank
fallback. Transport, client, response, or parsing failures produce an empty
batch with `source="local_chat"`, `error_type="generation_error"`, and a
sanitized error message. The musical controller can therefore record the
failure and select its own fallback. Error sanitization redacts authorization,
API-key, token, and password values; no authorization data is copied into the
batch. An empty model response is also an explicit empty error batch.

Phrase-bank generation remains an observable source and returns a complete
batch for the provided request, with an empty prompt/raw response and zero
latency.

## Fallback Validation, Selection, and Source

`PrevalidatedFallbackCatalog.build(scenario, templates, analyzer)` analyzes
every configured fallback line immediately. A line whose syllable count differs
from its template slots raises a stable `ValueError` before runtime. Lookup
requires a validated `CandidateRequest`, normalizes the topic, validates the
template key and request syllable count, and returns a
`PrevalidatedFallbackLine` with source `prevalidated_fallback`.

Repeated segments sharing a normalized topic/template append their configured
lines rather than overwriting earlier ones. `line_for(request)` selects with
`request.target_bar % len(lines)`, giving deterministic absolute-bar
round-robin behavior. It never reports a fallback line as generated.

## Compatibility Migrations

`RapPrototypeService` constructs one deterministic complete request for its
existing one-shot candidate generation. `RollingRapController` constructs a
request per fallback or primary target bar, including the bar’s slot count,
source-specific request id, and deterministic bar seed. It still uses the
current phrase bank as its controller-level fallback. No rolling-planner policy
or timing behavior was redesigned; that is deferred to Task 7.

Both RAP CLIs construct `LocalChatCandidateGenerator(client)` directly. The
realtime CLI continues to pass the phrase bank separately to the controller.

## Demo-Ready Examples

- A local-chat request for bar 4, `baseline_syncopated_9`, topic `space`, and
  history `("stars cross the night",)` records its prompt, exact model text,
  latency, and token counts in its batch.
- A connection failure returns an empty `local_chat` batch with
  `generation_error`; it does not silently produce phrase-bank text.
- A scenario with fallback lines `("one", "two")` for a one-slot template
  selects `one`, `two`, `one` at absolute bars 0, 1, and 2, each with source
  `prevalidated_fallback`.

## Tests and Verification

```text
UV_CACHE_DIR=/tmp/streammuse-uv-cache uv run python -m pytest \
  tests/unit/infrastructure/rap/test_generators.py \
  tests/unit/infrastructure/rap/test_fallback.py -v
# 9 passed

UV_CACHE_DIR=/tmp/streammuse-uv-cache uv run python -m pytest \
  tests/unit/domain/rap tests/unit/application/rap \
  tests/unit/infrastructure/rap tests/unit/presentation/rap -q --tb=no
# 128 passed

UV_CACHE_DIR=/tmp/streammuse-uv-cache uv run python -m pytest \
  tests/unit/presentation/rap/test_rap_cli.py \
  tests/unit/presentation/test_cli_config_parser.py \
  tests/integration/test_cli_entry_point.py -q --tb=no
# 24 passed; one existing pretty_midi/pkg_resources deprecation warning

git diff --check
# passed
```

## Self-Review

- No RAP generator remains on the positional `(topic, count)` protocol.
- No local-chat path invokes phrase-bank fallback behavior.
- The batch preserves raw model text and timing/token diagnostics on successful
  `ChatModelResponse` paths.
- Fallback validation occurs before lookup and checks every configured line.
- Repeated normalized fallback keys retain every configured line in stable
  scenario order.
- Fallback source is explicit and distinct from model generation.
- The minimal service/controller migration retains existing RAP tests and CLI
  behavior.

## Limitations and Concerns

- `CandidateBatch.raw_response` is the exact `ChatModelResponse.text` supplied
  by the client. The existing local client already strips its HTTP content
  before constructing that response; retaining byte-exact HTTP bodies would
  require widening the shared chat-client contract.
- The current phrase-bank templates are intentionally not syllable-filtered;
  Task 4/Task 7 scoring and selection own validity and fallback choice.
- The new catalog is ready for Task 7’s scenario-aware controller but is not
  yet wired into the legacy rolling planner, preserving the requested minimal
  compatibility migration.
