# Character and Concurrent Counting Implementation

Date: 2026-11-05. Workspace: the existing `real_rap` worktree under
`MBZUAI/StreamMUSE-NewSystem/.worktrees/real_rap`.

## Scope and Ownership

Implemented an isolated, text-only two-character backend, optional FastAPI router,
two experiment CLIs, and focused tests. No commits, pushes, extra agents, model
services, or GPU jobs were launched by this worker. Existing dirty work was left
in place. No audio, stress policy, parent server, parent CLI, or static UI files
were edited by this worker. The parent owns mounting, GUI integration, deployment,
and H200 scheduling.

Changed paths, relative to the worktree:

- `src/streammuse/application/rap/characters.py`
- `src/streammuse/presentation/rap_demo/characters_api.py`
- `scripts/experiments/research_day/characters.py`
- `scripts/experiments/research_day/concurrent_counting.py`
- `tests/unit/application/rap/test_characters.py`
- `tests/unit/presentation/test_characters_api.py`
- `tests/unit/experiments/test_research_characters.py`
- `docs/reports/2026-11-05-character-implementation.md`

## Backend Contract

`CharacterExperiment(ExperimentConfig(...))` owns exactly two `CharacterState`
objects, each with its own persona and private proposal history. Its `run()` is
async; `stop()` requests transport cancellation; `snapshot()` and
`event_page(after=0)` return detached JSON-compatible snapshots. Mutation belongs
to the owning asyncio event loop. Each experiment instance runs once.

Each actor gets a distinct `LocalChatModelClient`. Both target the same endpoint
and model. Each generation has its own actor-specific prompt; `generate_choices`
and API-level `n=2` are not used. Production has no fake or mock fallback.

- URL: `STREAMMUSE_RESEARCH_MODEL_URL`, default `http://127.0.0.1:18711/v1`.
- Model: `STREAMMUSE_RESEARCH_MODEL`, default `qwen-rap`.
- CLI overrides: `--model-url`, `--model`, `--seed`.
- Temperature 0; seed recorded; `chat_template_kwargs.enable_thinking=false`.
- Maximum two in-flight requests, one per actor; no automatic HTTP retries.
- Each run permits at most 100 model calls and 60 seconds of generation time.
- Cancellation closes both clients before terminal status is published. HTTP
  cancellation/closure does not establish that GPU execution stopped.

Full accepted and proposal ledgers are retained. Request contexts are projected
to at most 1,400 UTF-8 content bytes, with short excerpts and explicit omitted
history counts. Dialogue allows 96 output tokens; counting allows 24. This is a
conservative byte budget, not a tokenizer-derived preflight count. Actual prompt
and completion token counts are retained when returned by the service. Each
request records its SHA-256 message-context hash, context version, visible parent
event IDs, byte count, monotonic start/end, wall duration, and client latency.

Only accepted output becomes a shared observation. An actor's unpublished drafts
are never copied into the other actor's prompt. Operators can inspect all public
proposal logs; "private" here means isolated model contexts, not secret operator
storage. Persona adherence and voluntary model disclosure are not guaranteed.

## Policies and Interpretation

Dialogue defaults to eight alternating turns, with a configurable range of 1..12.
Counting always targets 1..10, regardless of the dialogue-only `turns` field.
All accepted counting entries must have the next integer and a different actor
from the preceding accepted entry. Completion therefore implies both actors
participated. There is no eleventh accepted count.

| Policy | Execution | Invalid or racing output |
| --- | --- | --- |
| `fixed_turn` | Schedule the next eligible actor alone | Keep proposal; retry within the call/time budget |
| `naive` | Both actors prepare from the same frozen state | Keep raw publication attempts; fail on the first rule violation, without repair |
| `coordinated` | Both actors prepare from the same frozen state | Reject outdated versions or illegal values; retain rejections and retry |

Concurrent policies use two-request waves, waiting for both results before the
next wave. Results are processed in completion order, not actor order. A `WAIT`
response is logged and does not publish a number. The naive accepted transcript
is only the audited valid prefix; `publication_attempts` is the uncorrected record.

The host referee enforces/audits accepted-prefix correctness. Successful
coordinated runs are not evidence of unaided model coordination or arithmetic
intelligence. Summed request durations can exceed wall duration under concurrency.
This is not a matched-snapshot dialogue speedup experiment, a continuous shared-KV
system, partial-message listening, or simultaneous acoustic conversation.

## API for Parent Mounting

```python
from streammuse.presentation.rap_demo.characters_api import create_characters_router

app.include_router(create_characters_router())
```

Mounting alone creates no clients or model traffic. Router lifespan cleanup stops
an active run when the parent application shuts down. The parent reported mounting
the router on Mac port 8019; this worker did not alter that server.

| Route | Request | Response |
| --- | --- | --- |
| `GET /api/characters/status` | None | Current or last run snapshot; initial `idle` |
| `POST /api/characters/start` | `{"mode":"dialogue","policy":"fixed_turn","turns":8}` | HTTP 202 and new run snapshot |
| `POST /api/characters/stop` | None | Snapshot after transport cleanup; idempotent |
| `GET /api/characters/events` | `?after=0&run_id=<optional>` | `{run_id,state,events,next_cursor}` |

`mode` is `dialogue` or `counting`; policy is `fixed_turn`, `naive`, or
`coordinated`, defaulting to `fixed_turn`. Dialogue only supports `fixed_turn`.
Bad bodies return 422; overlapping runs return 409. Model unavailability appears
as terminal `failed` with an explicit `error` and error proposal, not generated
substitute output. Start acceptance is not a model health guarantee.

States: `idle`, `running`, `stopping`, `completed`, `failed`, `cancelled`,
`timed_out`. Each controller keeps one run, not persistent multi-run history.
Model traffic is bounded per controller/process, not globally across CLI and API.

Snapshot keys include `run_id,state,mode,policy,model_url,model,actors,accepted,
proposals,publication_attempts,metrics,error`. Proposal keys include
`request_id,actor_id,text,number,context_version,context_hash,prompt_utf8_bytes,
started_at,finished_at,duration_ms,model_latency_ms,prompt_tokens,completion_tokens,
outcome,reason`. Outcomes are `accepted,rejected,wait,cancelled,error`.

Accepted events include `event_id,sequence,actor_id,request_id,text,number,version`.
All event records include `kind`, `run_id`, and `client_monotonic_time`.
Sequence cursors restart per run; a mismatching optional `run_id` resets the cursor.
This is polling, not SSE. Event kinds include `run_started,request_started,
publication_attempt,proposal,accepted,cancellation_requested,transport_closed,
run_finished`.

## Experiment Commands and Artifacts

Run from the worktree, using the existing environment:

```sh
PYTHONPATH=src .venv/bin/python scripts/experiments/research_day/characters.py --turns 8
PYTHONPATH=src .venv/bin/python scripts/experiments/research_day/concurrent_counting.py --policy all --trials 3
```

The parent must reserve model/GPU availability before invoking these commands.
`--policy all` executes policies sequentially for each trial, using the same seed
across that trial's policies. Seeds do not imply replay determinism. Exit code is
0 only when every requested run completes, 1 for any failed/timed-out run, and 130
for keyboard interruption. Naive failure is expected evidence, not a CLI crash.

Output defaults to `output/research_day_20261105/{characters,counting}/` with a
unique directory per run. Files: `manifest.json`, `summary.json`, `events.jsonl`,
`proposals.jsonl`, `accepted.jsonl`, `publication_attempts.jsonl`. Counting also
writes `comparison.json`, updated after each completed run. Reusing an output
root replaces that comparison index, but preserves uniquely named run directories.
Graceful cancellation retains per-run artifacts. A hard process kill can prevent
final artifact flushing; this is not a durable streaming journal.

## Verification and Cancellation-Test Repair

Tests were added before implementation and initially failed on the missing new
modules. Focused backend/API/CLI tests then passed. Two further failing lifecycle
tests demonstrated premature terminal state and cleanup-error propagation; both
were fixed without changing policies or schemas.

The parent later reported a broad run with 1,829 passes and one failure in
`test_cancellation_aborts_both_requests_and_accepts_no_late_output`. Production
returned `cancelled` with no accepted entries, but one test handler missed its
cancellation observation: `await asyncio.to_thread(barrier.wait)` was outside its
`try/except CancelledError`. The parent can stop immediately after barrier release,
before that handler resumes. The test now observes cancellation around both awaits.
No production cancellation behavior was changed for that repair.

After the repair, the following command passed **32 tests in each of 10 fresh
pytest processes**, totaling 320 passing executions and no failures:

```sh
PYTHONPATH=src .venv/bin/python -m pytest tests/unit/application/rap/test_characters.py tests/unit/presentation/test_characters_api.py tests/unit/experiments/test_research_characters.py -q -p no:cacheprovider
```

The focused regression command also includes the existing client suite:

```sh
PYTHONPATH=src .venv/bin/python -m pytest tests/unit/infrastructure/inference/test_local_chat_model_client.py tests/unit/application/rap/test_characters.py tests/unit/presentation/test_characters_api.py tests/unit/experiments/test_research_characters.py -q -p no:cacheprovider
```

That command passed 62 tests after lifecycle fixes. The existing client's two
temporary-loopback-server tests initially hit sandbox socket permissions; the
authorized unsandboxed rerun passed. No test in this focused set contacts H200.
Both CLI `--help` entry points were executed successfully. `git diff --check`
passed. A fully green broad rap suite is not claimed here; the parent's final
broad rerun remains the authority.

## Parent-Generated Real-Model Evidence

This worker inspected the parent's existing artifacts without issuing model calls:

- Dialogue: `output/research_day_20261105/characters/fixed_turn-ea596c47fa8a455a8d055c0ba6feb12f/summary.json`.
- Counting: `output/research_day_20261105/counting/comparison.json` and its nine referenced run directories.

The `qwen-rap` dialogue completed eight turns in 7.240 seconds, with eight model
calls. Reported prompt tokens ranged from 128 to 303, below the parent's 2,048-token
server context. The last two dialogue lines were identical. This demonstrates
the working isolated-context/shared-transcript path, not strong persona quality.

| Policy | Completed | Calls per run | Valid prefix | Wall seconds, seeds 0/1/2 | Stale proposals |
| --- | --- | --- | --- | --- | --- |
| `fixed_turn` | 3/3 | 10 | 10 | 6.993 / 7.211 / 8.020 | 0 / 0 / 0 |
| `naive` | 0/3 | 2 | 1 | 0.954 / 1.348 / 0.897 | Not version-filtered |
| `coordinated` | 3/3 | 20 | 10 | 5.844 / 6.670 / 5.580 | 6 / 7 / 4 |

Every naive run attempted duplicate `1` publication and stopped with the violation
retained. Coordinated runs rejected ten proposals apiece in total, including the
listed stale proposals. These three-trial, fixed-policy-order pilots are not a
general speedup or reliability claim. Their successful accepted transcripts are
explicitly dependent on the host referee.
