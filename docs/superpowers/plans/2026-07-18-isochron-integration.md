# Isochron Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Isochron prompt continuation and complete the shared runtime composition root without changing default StreamMUSE behavior.

**Architecture:** Port Isochron-only modules first, then add disabled-by-default configuration and parallel server/client endpoints. Preserve current target implementations in every overlapping file. Migrate CLI and web to a characterized `RuntimeSessionBuilder` only after both service modes pass focused tests.

**Tech Stack:** Python 3.10+, FastAPI, Pydantic, PyTorch, pytest, existing StreamMUSE Clean Architecture layers.

## Global Constraints

- Freeze the 15 failing node IDs at `1734ea9b`; introduce no new failures or collection errors, and do not turn previously passing tests into skips. Newly ported environment-gated Isochron tests may retain their existing skips.
- Do not change code or tests solely to repair those baseline failures.
- Standard inference remains the default and must preserve current construction arguments.
- Prompt continuation is opt-in; Rap plus prompt continuation is rejected explicitly.
- Do not import `docs/.vitepress/cache`, `docs/.vitepress/dist`, or generated `egg-info` changes.

---

### Task 1: Port the isolated prompt-continuation core

**Files:**
- Create: `src/streammuse/domain/musical/sequence.py`
- Create: `src/streammuse/application/services/prompt_continuation_realtime_service.py`
- Create: `src/streammuse/infrastructure/inference/lekai_continuation_model/*`
- Create: `src/streammuse/infrastructure/inference/lekai_prompt_continuation/*`
- Create: `src/streammuse/infrastructure/inference/prompt_continuation_http_client.py`
- Test: corresponding Isochron unit and integration tests

**Interfaces:**
- Consumes: existing `MusicalEvent`, `Tempo`, input and output protocols
- Produces: `PromptContinuationRealtimeService` and prompt-continuation backend/client APIs

- [ ] Port focused tests before their production modules and run them to confirm collection fails because the modules are absent.
- [ ] Port only the production modules needed by those tests.
- [ ] Run the prompt service, scheduler, engine, backend, sequence, and client unit tests until they pass.
- [ ] Run the full suite and compare exact failures to the frozen baseline.
- [ ] Commit the isolated core and tests.

### Task 2: Add opt-in configuration and server/client routing

**Files:**
- Modify: `src/streammuse/application/config/models.py`
- Modify: `src/streammuse/application/config/__init__.py`
- Modify: `src/streammuse/presentation/cli/config_parser.py`
- Modify: `src/streammuse/infrastructure/inference/server_lekai.py`
- Test: parser, server, and prompt HTTP client tests

**Interfaces:**
- Consumes: Task 1 backend/client APIs
- Produces: disabled-by-default `continuation_mode`, `prompt_length_ticks`, and `midi_file_trim_leading_rest` settings plus prompt routes

- [ ] Add failing tests for defaults, explicit continuation selection, prompt routes, and Rap/continuation rejection.
- [ ] Verify the new tests fail for missing configuration/routes.
- [ ] Add the minimum configuration fields and parallel server routes while retaining standard route bodies.
- [ ] Run focused parser/server/client tests.
- [ ] Run the full baseline-differential suite.
- [ ] Commit configuration and routing.

### Task 3: Add opt-in MIDI trimming and continuation entry-path assembly

**Files:**
- Modify: `src/streammuse/infrastructure/input/midi_file.py`
- Modify: `src/streammuse/application/factories/input_factory.py`
- Modify: `src/streammuse/presentation/cli/cli.py`
- Modify: `src/streammuse/presentation/web/server.py`
- Test: MIDI input, CLI entry-point, and web tests

**Interfaces:**
- Consumes: Task 2 configuration and Task 1 service/client
- Produces: explicitly selected continuation runtime while preserving the default standard branch

- [ ] Add failing tests proving default MIDI timing is unchanged and trimming occurs only when enabled.
- [ ] Add failing CLI/web tests for explicit continuation selection and unchanged standard selection.
- [ ] Implement opt-in trimming without moving the target clock anchor.
- [ ] Add the smallest conditional continuation construction path.
- [ ] Run focused and full baseline-differential tests.
- [ ] Commit entry-path integration.

### Task 4: Complete the single runtime composition root

**Files:**
- Modify: `src/streammuse/application/runtime/builder.py`
- Modify: `src/streammuse/application/runtime/session.py`
- Modify: `src/streammuse/presentation/cli/cli.py`
- Modify: `src/streammuse/presentation/web/server.py`
- Modify: `tests/unit/application/runtime/test_runtime_builder.py`
- Test: CLI and web characterization tests

**Interfaces:**
- Consumes: standard and continuation service constructors plus optional Rap observer
- Produces: one `RuntimeSessionBuilder` composition root for CLI and web

- [ ] Add characterization tests that capture current standard CLI/web construction arguments, lifecycle horizons, artifact tiers, and Rap observer attachment.
- [ ] Add continuation builder tests and verify they fail before builder changes.
- [ ] Extend the builder to reproduce the characterized standard runtime and select continuation only when configured.
- [ ] Switch CLI to the builder; run characterization, focused, and full tests.
- [ ] Switch web to the builder; repeat the gates.
- [ ] Remove only construction duplication proven unreachable by tests.
- [ ] Commit the composition-root migration separately.

### Task 5: Port consistency tooling and curated documentation

**Files:**
- Create: `tests/consistency/two_stage_runners.py`
- Create: `tests/consistency/test_two_stage_prompt_continuation_consistency.py`
- Create: `scripts/run_lekai_prompt_continuation_offline.py`
- Modify: `scripts/run_lekai_offline.py` only when required by shared helpers
- Modify: `pyproject.toml` and `uv.lock` only for verified direct dependencies
- Create/Modify: source documentation describing the integrated mode

**Interfaces:**
- Consumes: integrated continuation runtime
- Produces: reproducible offline/realtime comparison and user instructions

- [ ] Port consistency tests before tooling and confirm missing-runner failures.
- [ ] Port the minimum runners/scripts required to pass them.
- [ ] Audit direct imports before changing dependencies.
- [ ] Exclude generated docs, package metadata, and large benchmark artifacts.
- [ ] Run focused consistency tests and the complete baseline comparison.
- [ ] Commit tooling and documentation.

### Task 6: Final differential verification

**Files:**
- Review all changes relative to `origin/new_system_stanley`

**Interfaces:**
- Consumes: all previous tasks
- Produces: a reviewable integration branch with evidence of no new regressions

- [ ] Run `git diff --check` and review every changed standard-path file.
- [ ] Run all prompt-continuation, runtime-builder, CLI, web, Rap, and Voice focused tests.
- [ ] Run `pytest tests/ -q --tb=no` and compare failure and skip node IDs to the frozen baseline.
- [ ] Confirm default CLI/web configuration still selects standard inference.
- [ ] Confirm the branch contains no generated VitePress or `egg-info` additions.
- [ ] Present the verified branch for review without pushing or merging unless explicitly requested.
