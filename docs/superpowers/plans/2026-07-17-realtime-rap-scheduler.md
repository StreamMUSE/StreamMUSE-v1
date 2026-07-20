# Real-Time Rap Scheduler Implementation Plan

> **For implementation:** Follow this plan with `superpowers:executing-plans`, using focused red-green tests for each task.

**Goal:** Integrate beat-aligned rap text into the existing `streammuse-cli` tick loop as an optional, rolling, non-blocking live layer.

**Architecture:** Add a small optional tick-observer lifecycle to `RealTimeMusicService`. A new application-level `RollingRapController` maintains locally aligned future bars, emits syllables through a presentation callback, and uses a one-worker background executor only for optional local-chat candidates. CLI configuration owns all assembly.

**Tech Stack:** Python 3.11, dataclasses, `ThreadPoolExecutor`, pytest, existing StreamMUSE Clean Architecture layers.

---

### Task 1: Add an optional tick-observer lifecycle to the music service

**Files:**
- Modify: `src/streammuse/application/services/real_time_music_service.py`
- Modify: `tests/unit/application/test_real_time_music_service.py`

1. Write a failing test with an observer and tracing output. It must assert
   `output_tick(0)` is observed before `observer.on_tick(0)`.
2. Run `uv run pytest tests/unit/application/test_real_time_music_service.py -q` and confirm the new test fails because the service has no observer argument.
3. Define a small structural `TickObserver` protocol near the service, add an optional constructor argument, call `start()` before worker threads, invoke `on_tick(tick)` immediately after `output_tick`, and call `close()` once during `stop()`.
4. Run the focused test file and confirm it passes.

### Task 2: Build the rolling, non-blocking rap controller

**Files:**
- Create: `src/streammuse/application/rap/realtime.py`
- Modify: `src/streammuse/application/rap/__init__.py`
- Create: `tests/unit/application/rap/test_realtime.py`

1. Write failing tests for initial fallback scheduling/emission, future-bar refill with absolute ticks, non-blocking behavior with a blocking local generator, replacement only before target-bar start, and close behavior.
2. Run `uv run pytest tests/unit/application/rap/test_realtime.py -q` and confirm collection fails before implementation.
3. Implement `RollingRapController` with a phrase-bank fallback, `choose_best_line`, `build_bar_slots`, per-bar storage, selected-text tracking, a single optional background future, and callback exception isolation. Use asynchronous results only when `CandidateBatch.source == "local_chat"` and their bar has not started.
4. Run the focused controller test file and confirm it passes.

### Task 3: Add optional rap configuration and CLI assembly

**Files:**
- Modify: `src/streammuse/application/config/models.py`
- Modify: `src/streammuse/application/config/__init__.py`
- Modify: `src/streammuse/presentation/cli/config_parser.py`
- Modify: `src/streammuse/presentation/cli/cli.py`
- Modify: `tests/unit/presentation/test_cli_config_parser.py`
- Modify: `tests/integration/test_cli_entry_point.py`

1. Add failing parser tests for disabled-by-default `RapConfig`, explicit values, and positive-value clamping.
2. Add a failing integration test that supplies `RapConfig(topic=...)`, patches the controller factory/constructor, and asserts the service receives the controller.
3. Run the two focused files and confirm failures.
4. Introduce immutable `RapConfig`, parse all CLI flags, construct a phrase-bank or `LocalChatCandidateGenerator`, supply a console callback and client cleanup callback, pass the controller to `RealTimeMusicService`, and include rap settings in session config.
5. Run the focused parser and integration files and confirm they pass.

### Task 4: Verify the integrated live path and document operation

**Files:**
- Modify: `docs/developer-guide/rap-alignment-prototype.md`

1. Run `uv run pytest tests/unit/application/test_real_time_music_service.py tests/unit/application/rap/test_realtime.py tests/unit/presentation/test_cli_config_parser.py tests/integration/test_cli_entry_point.py -q`.
2. Run a finite smoke command against the fake inference server or patched local test path that exercises `--rap-topic --max-ticks` and records the emitted `[RAP ...]` lines.
3. Run `uv run pytest tests/ -q --tb=no` for full regression evidence.
4. Add the precise invocation, observed timing behavior, and current limits to the developer guide.
