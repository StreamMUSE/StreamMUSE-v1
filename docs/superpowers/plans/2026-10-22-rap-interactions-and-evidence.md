# Rap Interactions and Scheduling Evidence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Work in disjoint scopes and review integration before release.

**Goal:** Deliver two distinct interaction modes and an empirically informed scheduling model by 07:30 America/Los_Angeles, with an 08:00 hard stop on 2026-10-22.

**Architecture:** Preserve the continuous v3 controller. Add a separate alternating coordinator, rap-specific timeline, typed runtime settings, and compact additions to the existing GUI. Research tests run separately and do not define feature acceptance.

**Tech Stack:** Existing Python/FastAPI/sounddevice backend, browser JavaScript, vLLM and MOSS on H200, pytest.

**Spec:** `docs/superpowers/specs/2026-10-22-rap-interaction-delivery-design.md` and `docs/superpowers/specs/2026-10-22-realtime-scheduling-model-design.md`.

## Global Constraints

- Continuous guided generation and alternating exchange are separate modes.
- Mac client/audio; H200 models. GPUs 0, 1, 2 only after empty check.
- Keep piano timing, old renderer entry points, monitoring, and fallback provenance intact.
- Capture one complete 16-tick bar; neutral tap stress 0.5; default response two bars, configurable 1-8.
- Echo does not advance logical model progress. First ready response starts without waiting for the entire turn.
- Tempo changes at future bar boundaries, 30-180 BPM. Device/mode/response changes while stopped.
- Never describe a playback cursor as an acoustic syllable measurement.
- Existing speculative templates remain explicitly UNTESTED.
- Preserve user changes; agents do not commit or change branch independently.
- Target initial integrated tests by 06:35, reserve remaining time for real tests, review, repair, and reports.

## Review Focus

- Old asynchronous results after Stop/Reset must not start another turn.
- Echo delays must not count as generated bars or erase intended deadlines.
- Tempo changes must not alter past sample positions or enqueue wrong-duration audio.
- Browser Space in an editable field or auto-repeat must not capture rhythm.
- Missing hardware, primary-render failure, and legacy backends must remain visible and usable.

## Ownership and Interfaces

The coordinator owns `runtime.py`, `rap_demo/server.py`, `rap_demo/cli.py`, and new `application/rap/settings.py`. Other workers own the scopes below. Communicate cross-scope edits before making them. All work is in the existing isolated `.worktrees/real_rap` worktree.

### Task 1: Versioned Timing and v3 Backend

Owner: timing worker. Files: new `domain/rap/timeline.py`, `application/rap/playback.py`, `bar_stream.py`, `audio_rendering.py`, `domain/rap/lyric_stream.py`, `application/rap/lyric_session.py`, `presentation/rap_lyric_stream_server.py`, `infrastructure/rap/lyric_stream_client.py`, corresponding tests.

- [x] Add failing tests for static equivalence, faster/slower future changes, repeated edits, immutable past, and legacy rejection.
- [x] Implement rap-only piecewise timeline and per-bar immutable frame snapshots; preserve fixed Tempo compatibility.
- [x] Add negotiated tempo updates and bounded turn production; expose controller settings hooks with requested/effective bar and revision.
- [x] Verify protocol round trips, stale preparation rejection, precise frame counts, and existing v3 tests.
- [x] Report exact integration hooks and limitations to coordinator; do not edit assembly/runtime/UI.
- [x] Repair live SSH acknowledgment failure and three independent review findings; rerun live tempo edits.

### Task 2: Alternating Capture and Echo

Owner: interaction worker. Files: new `domain/rap/interaction.py`, `application/rap/interaction.py`, `infrastructure/rap/flow_echo.py`, dedicated interaction tests.

- [x] Test arm-next-bar, full-bar capture, ties-earlier quantization, duplicates, empty input, timeout, stale epoch, and logical/physical mapping.
- [x] Implement separate coordinator compatible with current controller lifecycle where practical; expose `interaction_state()`, `arm_capture()`, and `record_tap(...)`.
- [x] Reuse source and preparation services through adapters; keep drums/echo live, submit once, start first ready response, bound later waits.
- [x] Test two generated bars separated by three echoes still deliver exactly two generated bars and no echo commit to server.
- [x] Send exact assembly/API integration instructions; coordinate protocol needs with timing worker through coordinator.
- [x] Repair four independent review findings and persist capture-finalization evidence.

### Task 3: Existing GUI Enhancements

Owner: GUI worker. Files: `presentation/rap_demo/static/**`, `infrastructure/rap/templates.py`, browser/template tests only.

- [x] Preserve current monitoring and plan controls; add stopped-only mode/device/response controls and live BPM/gain settings.
- [x] Add focused keyboard capture and turn-state display; no capture in form controls or held key repeats.
- [x] Make stress levels readable and add six provenance-correct UNTESTED variants.
- [x] Add distinct requested/estimated/cursor diagnostics with unavailable states.
- [x] Verify desktop/mobile layout and relevant JS tests; coordinate backend JSON contract with coordinator.

### Task 4: Audio Preparation and Diagnostics

Owner: audio worker. Files: `application/rap/bar_audio.py`, `bar_renderer.py`, `chunk_audio.py`, `domain/rap/audio.py`, `infrastructure/rap/audio_output.py`, `application/rap/monitoring_payloads.py` and related tests. No playback/controller edits.

- [x] Add tests for independent stem levels, limiter, unavailable estimates, sparse-anchor drift, and output-device validation.
- [x] Preserve warp-derived estimates through preparation; distinguish requested from estimated and observed.
- [x] Expose reusable mix configuration and supported output-device enumeration/switching while stopped.
- [x] Preserve defaults for old code paths; let settings explicitly activate increased drum level.
- [x] Report exact interfaces for runtime/GUI wiring, and test old eSpeak/MOSS preparation paths.

### Task 5: Lifecycle and Integration

Owner: coordinator. Files: runtime/server/CLI/settings and associated tests.

- [x] Pin reported Stop/Start behavior and finite replay with tests before changing code.
- [x] Add typed GET/POST `/api/control/settings`, GET `/api/control/interaction`, POST `/api/control/interaction/arm`, POST `/api/control/interaction/tap`.
- [x] Assemble mode-specific controller, connect timeline/audio settings, reject unsupported settings visibly.
- [x] Ensure Start after natural completion automatically establishes a fresh epoch; manual Stop resumes.
- [x] Integrate all worker changes and run full tests plus browser and actual Mac/H200 workflows.

### Task 6: Research Evidence and Reduction

Owner: research worker. Files: new `scripts/experiments/rap_scheduling_evidence.py`, `docs/research/2026-10-22-*`, tests of experiment analysis, ignored/raw artifacts; no product code.

- [x] Inventory measured knobs and select short paired tests of candidate budget, stage costs/concurrency, and deadline/lookahead sensitivity.
- [x] Launch isolated authorized model services on empty GPU0 (GPU2 if needed), report ports and commands for integration. Do not kill unrelated processes or use shared service pid files blindly.
- [x] Preserve raw per-round data, valid/selectable counts, stage latency, failures, environment/model configuration, and reproducible commands.
- [x] Separate real measurements, deterministic delay-injection experiments, and trace-based counterfactuals.
- [x] Reduce the broad inventory to roughly 5-8 defensible parameter groups, with relationships and evidence/uncertainty. No significance claims unsupported by sample size.
- [x] Produce a separate research progress/results report with experiment coverage and untested dimensions.

### Task 7: Acceptance, Review, and Two Reports

- [x] Run full pytest suite and focused browser tests after integration.
- [x] Real H200/Mac continuous edits and alternating turns; record misses/replays/fallbacks rather than hiding them.
- [x] Fresh code/spec review, repair high-impact findings, rerun affected checks.
- [x] Update quick guide with exact commands and current limitations.
- [x] Produce `docs/reports/2026-10-22-feature-delivery.md` and `docs/reports/2026-10-22-scheduling-research.md`, update throughout execution and finalize near deadline.
- [x] Record source state and evidence without claiming live success for untested paths. No final commit/push requested or performed.

## Execution Log

- 05:14 PDT: clean rap worktree at bde2f62c; main checkout left untouched.
- 05:15 PDT: H200 GPUs 0-7 show 0 MiB and no compute processes. Authorized GPUs allocated: research GPU0, integration GPU1, spare GPU2; recheck before launch.
- User explicitly requests continued work until close to cutoff and separate reports for feature and research tracks.
- User permits at most one usage-limit reset, only if needed. Demo reset tests are unaffected.
- Latest clarification: stop early only when both tracks are complete and reasonably exhaustively tested; otherwise continue to the deadline.
