"""Tests for deadline-safe rolling two-bar rap audio preparation."""

from __future__ import annotations

from collections import deque
from concurrent.futures import Future, TimeoutError
from dataclasses import replace
import json
from threading import Event, Thread, current_thread

import pytest

import streammuse.application.rap.chunk_realtime as chunk_realtime_module
from streammuse.application.rap.chunk_realtime import RollingRapChunkController
from streammuse.domain.rap import (
    AudioFormat,
    FlowProvenance,
    FlowSlot,
    FlowTemplate,
    PcmAudio,
    PreparedRapBar,
    PreparedRapChunk,
    RapEventType,
    RapScenario,
    RemoteCandidatePolicy,
    ScheduledSyllable,
    ScenarioSegment,
    materialize_flow,
)
from streammuse.domain.timing import Tempo
from streammuse.infrastructure.rap.fallback import PrevalidatedFallbackCatalog
from streammuse.infrastructure.rap.prosody import CmuProsodyAnalyzer
from streammuse.infrastructure.rap.templates import TemplateCatalog


class ManualClock:
    def __init__(self, value: float = 0.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value


class ManualFuture(Future):
    def __init__(self) -> None:
        super().__init__()
        self.result_calls: list[float | None] = []

    def result(self, timeout: float | None = None):
        self.result_calls.append(timeout)
        if timeout is not None and not self.done():
            raise TimeoutError
        return super().result(timeout=timeout)

    def cancel(self) -> bool:
        # Model a worker that has already started and cannot be cancelled.
        return False


class ManualExecutor:
    def __init__(self, *, run_submissions: int = 0) -> None:
        self.run_submissions = run_submissions
        self.tasks: list[tuple[ManualFuture, object, tuple[object, ...]]] = []
        self.shutdown_calls: list[tuple[bool, bool]] = []

    def submit(self, fn, *args):
        future = ManualFuture()
        self.tasks.append((future, fn, args))
        if len(self.tasks) <= self.run_submissions:
            self.complete(len(self.tasks) - 1)
        return future

    def complete(self, index: int) -> None:
        future, fn, args = self.tasks[index]
        if future.done():
            return
        try:
            future.set_result(fn(*args))
        except BaseException as error:
            future.set_exception(error)

    def shutdown(self, *, wait: bool, cancel_futures: bool) -> None:
        self.shutdown_calls.append((wait, cancel_futures))


class RecordingPublisher:
    def __init__(self) -> None:
        self.events: list[dict[str, object]] = []

    def emit(self, event_type, **kwargs):
        self.events.append({"event_type": event_type, **kwargs})


class RecordingRenderer:
    def __init__(self, *, on_render=None) -> None:
        self.plans = []
        self.on_render = on_render

    def render(self, plan) -> PreparedRapBar:
        self.plans.append(plan)
        if self.on_render is not None:
            self.on_render(plan)
        return _prepared_bar(plan.bar, plan.text, plan.source, plan.scheduled, plan.fallback_reason)


class RecordingStrategy:
    def __init__(self, outcomes=()) -> None:
        self.outcomes = deque(outcomes)
        self.requests = []
        self.deadlines: list[float] = []
        self.fallback_render_counts: list[int] = []
        self.renderer: RecordingRenderer | None = None
        self.abort_calls = 0
        self.close_calls = 0

    def prepare(self, request, *, deadline_monotonic: float):
        self.requests.append(request)
        self.deadlines.append(deadline_monotonic)
        self.fallback_render_counts.append(len(self.renderer.plans) if self.renderer is not None else -1)
        outcome = self.outcomes.popleft() if self.outcomes else _chunk_for(request)
        if isinstance(outcome, BaseException):
            raise outcome
        if callable(outcome):
            return outcome(request)
        return outcome

    def abort(self) -> None:
        self.abort_calls += 1

    def close(self) -> None:
        self.close_calls += 1


class BlockingAbortStrategy(RecordingStrategy):
    def __init__(self) -> None:
        super().__init__()
        self.abort_entered = Event()
        self.release_abort = Event()
        self.abort_threads: list[Thread] = []

    def abort(self) -> None:
        self.abort_calls += 1
        self.abort_threads.append(current_thread())
        self.abort_entered.set()
        if not self.release_abort.wait(timeout=2.0):
            raise RuntimeError("test did not release blocking abort")


def _template() -> FlowTemplate:
    return FlowTemplate(
        template_id="one_slot",
        name="One slot",
        ticks_per_beat=4,
        beats_per_bar=4,
        slots=(FlowSlot(0, 16, 1.0, rhyme_group="A"),),
        provenance=FlowProvenance(kind="test", source="unit-test"),
    )


def _scenario(*, loop: bool = True, bars: int = 4) -> RapScenario:
    return RapScenario(
        scenario_id="chunk-test",
        tempo_bpm=120.0,
        segments=(ScenarioSegment(0, bars, "space", "one_slot", ("beat", "light", "stars", "drums")),),
        loop=loop,
    )


def _prepared_bar(bar: int, text: str, source: str, scheduled=(), fallback_reason=None) -> PreparedRapBar:
    audio_format = AudioFormat()
    return PreparedRapBar(
        bar=bar,
        text=text,
        source=source,
        fallback_reason=fallback_reason,
        scheduled=tuple(scheduled),
        audio=PcmAudio(audio_format, 1, bytes(audio_format.channels * audio_format.sample_width_bytes)),
        diagnostics=(),
        warnings=(),
        render_latency_ms=1.0,
    )


def _chunk_for(request, texts: tuple[str, str] | None = None) -> PreparedRapChunk:
    texts = texts or ("stars", "space")
    analyzer = CmuProsodyAnalyzer()
    bars = tuple(
        _prepared_bar(
            item.bar,
            text,
            "moss_aligned_remote",
            (
                ScheduledSyllable(
                    materialize_flow(item.flow_template, item.bar)[0],
                    analyzer.analyze(text).syllables[0],
                ),
            ),
        )
        for item, text in zip(request.bars, texts, strict=True)
    )
    return PreparedRapChunk(
        request_id=request.request_id,
        chunk_index=request.chunk_index,
        renderer="moss_aligned_remote",
        bars=bars,
        diagnostics={
            "stage_timings_ms": {"total": 12.0},
            "warnings": (),
        },
    )


def _invalid_chunk(request) -> PreparedRapChunk:
    return replace(_chunk_for(request), request_id="wrong-request")


def _controller(
    *,
    strategy: RecordingStrategy | None = None,
    executor: ManualExecutor | None = None,
    clock: ManualClock | None = None,
    scenario: RapScenario | None = None,
    planning_bar_limit: int | None = None,
    startup_timeout_seconds: float = 3.0,
    rolling_timeout_seconds: float = 5.0,
    preparation_executor: ManualExecutor | None = None,
    cancellation_executor: ManualExecutor | None = None,
    renderer: RecordingRenderer | None = None,
    on_enqueue=None,
):
    clock = clock or ManualClock()
    executor = executor or ManualExecutor()
    renderer = renderer or RecordingRenderer()
    strategy = strategy or RecordingStrategy()
    strategy.renderer = renderer
    scenario = scenario or _scenario()
    template = _template()
    templates = TemplateCatalog.from_templates((template,))
    analyzer = CmuProsodyAnalyzer()
    queued: list[PreparedRapBar] = []

    def enqueue(prepared: PreparedRapBar) -> None:
        queued.append(prepared)
        if on_enqueue is not None:
            on_enqueue(prepared)

    publisher = RecordingPublisher()
    controller = RollingRapChunkController(
        tempo=Tempo(120.0, 4, 4),
        scenario=scenario,
        templates=templates,
        fallback_catalog=PrevalidatedFallbackCatalog.build(scenario, templates, analyzer),
        analyzer=analyzer,
        fallback_renderer=renderer,
        preparation_strategy=strategy,
        publisher=publisher,
        enqueue=enqueue,
        session_id="session-1",
        policy=RemoteCandidatePolicy.realtime_default(),
        seed=7,
        planning_bar_limit=planning_bar_limit,
        startup_timeout_seconds=startup_timeout_seconds,
        rolling_timeout_seconds=rolling_timeout_seconds,
        executor=executor,
        preparation_executor=preparation_executor or ManualExecutor(run_submissions=100),
        cancellation_executor=cancellation_executor or ManualExecutor(run_submissions=100),
        monotonic=clock,
    )
    return controller, queued, strategy, renderer, executor, clock, publisher


def _events(publisher: RecordingPublisher, event_type: RapEventType) -> list[dict[str, object]]:
    return [item for item in publisher.events if item["event_type"] == event_type]


def _complete_pair_two_staging_when_async(preparation_executor: ManualExecutor) -> None:
    if preparation_executor.tasks:
        preparation_executor.complete(0)


def test_startup_renders_both_fallbacks_before_remote_submission_and_prefers_on_time_remote() -> None:
    executor = ManualExecutor(run_submissions=1)
    controller, queued, strategy, renderer, _, clock, publisher = _controller(executor=executor)
    clock.value = 10.0

    controller.start()

    assert [plan.bar for plan in renderer.plans[:2]] == [0, 1]
    assert strategy.fallback_render_counts[0] == 2
    assert strategy.deadlines[0] == 13.0
    assert strategy.requests[0].context_lines == ()
    assert [bar.bar for bar in queued] == [0, 1]
    assert [bar.source for bar in queued] == ["moss_aligned_remote", "moss_aligned_remote"]
    assert len(_events(publisher, RapEventType.CHUNK_COMMITTED)) == 1


def test_startup_timeout_commits_ready_fallback_pair_without_gap_then_submits_next_chunk() -> None:
    controller, queued, strategy, renderer, executor, _, publisher = _controller()

    controller.start()

    assert [bar.bar for bar in queued] == [0, 1]
    assert [bar.source for bar in queued] == ["prevalidated_fallback", "prevalidated_fallback"]
    assert len(renderer.plans) == 4
    assert len(executor.tasks) == 2
    assert strategy.abort_calls == 1
    assert executor.tasks[0][0].result_calls == [3.0]
    assert executor.tasks[1][0].result_calls == []
    assert strategy.requests == []
    executor.complete(1)
    assert strategy.requests[0].bars[0].bar == 2
    assert strategy.requests[0].context_lines == tuple(bar.text for bar in queued)
    assert len(_events(publisher, RapEventType.CHUNK_FALLBACK_ACTIVATED)) == 1


def test_running_chunk_is_polled_without_blocking_and_committed_atomically_at_pair_boundary() -> None:
    executor = ManualExecutor(run_submissions=1)
    controller, queued, strategy, _, _, clock, _ = _controller(executor=executor)
    controller.start()
    future = executor.tasks[1][0]

    clock.value = 1.0
    controller.on_tick(0)
    assert future.result_calls == []
    assert len(queued) == 2

    clock.value = 3.8
    executor.complete(1)
    controller.on_tick(31)

    assert future.result_calls == [None]
    assert [bar.bar for bar in queued] == [0, 1, 2, 3]
    assert [bar.source for bar in queued[2:]] == ["moss_aligned_remote", "moss_aligned_remote"]
    assert len(executor.tasks) == 3
    executor.complete(2)
    assert strategy.requests[-1].context_lines == tuple(bar.text for bar in queued)


def test_completion_at_immutable_deadline_is_late_and_never_changes_later_context() -> None:
    executor = ManualExecutor(run_submissions=1)
    controller, queued, strategy, _, _, clock, publisher = _controller(executor=executor)
    controller.start()

    clock.value = 3.875
    executor.complete(1)
    assert strategy.deadlines[1] == 3.875
    controller.on_tick(31)

    assert [bar.source for bar in queued[2:]] == ["prevalidated_fallback", "prevalidated_fallback"]
    executor.complete(2)
    assert strategy.requests[-1].context_lines == tuple(bar.text for bar in queued)
    rejection = _events(publisher, RapEventType.CHUNK_REMOTE_REJECTED)[-1]
    assert rejection["payload"]["state"] == "late"
    assert rejection["payload"]["deadline_slack_ms"] == 0.0


@pytest.mark.parametrize(
    "outcome",
    (RuntimeError("remote failed"), _invalid_chunk, object(), None, {"not": "a chunk"}),
)
def test_failed_or_invalid_remote_commits_both_local_fallbacks(outcome) -> None:
    executor = ManualExecutor(run_submissions=1)
    strategy = RecordingStrategy((_chunk_for, outcome))
    controller, queued, _, _, _, clock, publisher = _controller(strategy=strategy, executor=executor)
    controller.start()
    clock.value = 2.0
    executor.complete(1)

    controller.on_tick(31)

    assert [bar.bar for bar in queued] == [0, 1, 2, 3]
    assert [bar.source for bar in queued[2:]] == ["prevalidated_fallback", "prevalidated_fallback"]
    assert len(_events(publisher, RapEventType.CHUNK_REMOTE_REJECTED)) == 1
    assert len(_events(publisher, RapEventType.CHUNK_FALLBACK_ACTIVATED)) == 1


def test_pending_remote_at_boundary_falls_back_without_waiting_in_tick_callback() -> None:
    executor = ManualExecutor(run_submissions=1)
    controller, queued, strategy, _, _, clock, _ = _controller(executor=executor)
    controller.start()
    future = executor.tasks[1][0]
    clock.value = 4.0

    controller.on_tick(31)

    assert future.result_calls == []
    assert [bar.source for bar in queued[2:]] == ["prevalidated_fallback", "prevalidated_fallback"]
    assert strategy.abort_calls == 1


def test_tick_delivers_ready_fallback_before_blocking_remote_abort_runs_off_observer() -> None:
    executor = ManualExecutor(run_submissions=1)
    cancellation_executor = ManualExecutor()
    strategy = BlockingAbortStrategy()
    controller, queued, _, _, _, clock, _ = _controller(
        strategy=strategy,
        executor=executor,
        cancellation_executor=cancellation_executor,
    )
    controller.start()
    clock.value = 4.0
    observer_threads: list[Thread] = []
    observer_returned = Event()

    def observe_boundary() -> None:
        observer_threads.append(current_thread())
        controller.on_tick(31)
        observer_returned.set()

    observer = Thread(target=observe_boundary)
    observer.start()
    observer.join(timeout=1.0)
    assert observer_returned.is_set()
    assert [bar.bar for bar in queued] == [0, 1, 2, 3]
    assert strategy.abort_calls == 0
    assert len(cancellation_executor.tasks) == 1

    abort_worker = Thread(target=lambda: cancellation_executor.complete(0))
    abort_worker.start()
    assert strategy.abort_entered.wait(timeout=1.0)
    strategy.release_abort.set()
    abort_worker.join(timeout=1.0)

    assert strategy.abort_threads == [abort_worker]
    assert strategy.abort_threads != observer_threads


def test_running_tick_schedules_following_fallback_render_without_executing_it_inline() -> None:
    executor = ManualExecutor(run_submissions=1)
    preparation_executor = ManualExecutor()
    controller, _, _, renderer, _, clock, _ = _controller(
        executor=executor,
        preparation_executor=preparation_executor,
    )
    controller.start()
    assert len(renderer.plans) == 2
    assert len(preparation_executor.tasks) == 1
    assert len(executor.tasks) == 1

    preparation_executor.complete(0)
    assert len(renderer.plans) == 4
    assert len(executor.tasks) == 2
    clock.value = 3.0
    executor.complete(1)

    controller.on_tick(31)

    assert len(renderer.plans) == 4
    assert len(preparation_executor.tasks) == 2
    assert len(executor.tasks) == 2
    preparation_executor.complete(1)
    assert len(renderer.plans) == 6
    assert len(executor.tasks) == 3


@pytest.mark.parametrize(
    ("completion_tick", "completion_time", "commit_tick"),
    ((62, 7.875, 63), (63, 8.0, 64)),
)
def test_fallback_staging_at_or_after_guard_commits_on_next_observed_tick_without_stranding(
    completion_tick: int,
    completion_time: float,
    commit_tick: int,
) -> None:
    executor = ManualExecutor(run_submissions=1)
    preparation_executor = ManualExecutor()
    controller, queued, _, _, _, clock, _ = _controller(
        executor=executor,
        preparation_executor=preparation_executor,
    )
    controller.start()
    _complete_pair_two_staging_when_async(preparation_executor)
    clock.value = 3.0
    executor.complete(1)
    controller.on_tick(31)
    pair_four_staging = len(preparation_executor.tasks) - 1

    controller.on_tick(completion_tick)
    clock.value = completion_time
    preparation_executor.complete(pair_four_staging)
    controller.on_tick(commit_tick)

    assert [bar.bar for bar in queued] == [0, 1, 2, 3, 4, 5]
    assert [bar.source for bar in queued[4:]] == ["prevalidated_fallback", "prevalidated_fallback"]
    assert controller._pending is None


def test_playback_handoff_sets_origin_before_async_pair_two_staging() -> None:
    clock = ManualClock()
    preparation_executor = ManualExecutor()
    executor = ManualExecutor(run_submissions=1)

    def observe_enqueue(prepared: PreparedRapBar) -> None:
        if prepared.bar == 1:
            clock.value = 1.0

    controller, _, strategy, renderer, _, _, _ = _controller(
        clock=clock,
        executor=executor,
        preparation_executor=preparation_executor,
        on_enqueue=observe_enqueue,
    )

    controller.start()

    assert [plan.bar for plan in renderer.plans] == [0, 1]
    assert len(preparation_executor.tasks) == 1
    preparation_executor.complete(0)
    executor.complete(1)
    assert strategy.deadlines[1] == pytest.approx(4.875)
    assert strategy.requests[1].remaining_budget_ms == 3_875


def test_finite_controller_stops_submitting_at_even_limit_and_looping_controller_wraps_segments() -> None:
    executor = ManualExecutor(run_submissions=1)
    finite = _scenario(loop=False, bars=2)
    controller, queued, strategy, _, _, _, _ = _controller(
        executor=executor,
        scenario=finite,
        planning_bar_limit=2,
    )
    controller.start()
    assert [bar.bar for bar in queued] == [0, 1]
    assert len(executor.tasks) == 1
    assert len(strategy.requests) == 1

    with pytest.raises(ValueError, match="even"):
        _controller(planning_bar_limit=3)

    looping_executor = ManualExecutor(run_submissions=1)
    looping, _, looping_strategy, _, _, _, _ = _controller(
        executor=looping_executor,
        scenario=_scenario(loop=True, bars=2),
    )
    looping.start()
    looping_executor.complete(1)
    assert tuple(item.bar for item in looping_strategy.requests[-1].bars) == (2, 3)


def test_stop_retains_committed_successor_for_resume_and_cancels_useful_remote_wait() -> None:
    executor = ManualExecutor(run_submissions=1)
    controller, queued, strategy, _, _, _, _ = _controller(executor=executor)
    controller.start()

    controller.request_stop(successor_bar=1)

    assert strategy.abort_calls == 1
    controller.resume_audio(1)
    assert [bar.bar for bar in queued] == [0, 1, 1]
    controller.resume_after_stop()
    assert len(executor.tasks) == 3


def test_stop_finishes_inflight_local_staging_needed_for_successor_resume() -> None:
    executor = ManualExecutor(run_submissions=1)
    preparation_executor = ManualExecutor()
    controller, queued, _, _, _, clock, _ = _controller(
        executor=executor,
        preparation_executor=preparation_executor,
    )
    controller.start()
    _complete_pair_two_staging_when_async(preparation_executor)
    clock.value = 3.0
    executor.complete(1)
    controller.on_tick(31)
    pair_four_staging = len(preparation_executor.tasks) - 1

    controller.request_stop(successor_bar=4)
    preparation_executor.complete(pair_four_staging)
    controller.resume_audio(4)

    assert [bar.bar for bar in queued] == [0, 1, 2, 3, 4]
    assert queued[-1].source == "prevalidated_fallback"


def test_reset_establishes_new_epoch_and_ignores_old_completion() -> None:
    executor = ManualExecutor(run_submissions=1)
    controller, queued, _, _, _, clock, _ = _controller(executor=executor)
    controller.start()
    old_future = executor.tasks[1][0]

    epoch = controller.reset()
    assert epoch == 1
    controller.start()
    queued_after_restart = list(queued)
    clock.value = 1.0
    executor.complete(1)
    controller.on_tick(0)

    assert old_future.done()
    assert queued == queued_after_restart


@pytest.mark.parametrize("transition", ("reset", "stop", "close"))
def test_queued_old_epoch_remote_never_crosses_strategy_execution_boundary(transition: str) -> None:
    executor = ManualExecutor(run_submissions=1)
    controller, _, strategy, _, _, _, publisher = _controller(executor=executor)
    controller.start()
    requests_before_transition = len(strategy.requests)

    if transition == "reset":
        controller.reset()
    elif transition == "stop":
        controller.request_stop(successor_bar=1)
    else:
        controller.close()
    events_after_transition = len(publisher.events)

    executor.complete(1)

    assert len(strategy.requests) == requests_before_transition
    assert len(publisher.events) == events_after_transition


def test_reset_fences_worker_paused_after_first_execution_boundary_check() -> None:
    executor = ManualExecutor(run_submissions=1)
    controller, _, strategy, _, _, _, _ = _controller(executor=executor)
    controller.start()
    requests_before_reset = len(strategy.requests)
    first_check_released_lock = Event()
    release_worker = Event()

    class FirstExitBarrierLock:
        def __init__(self, delegate) -> None:
            self.delegate = delegate
            self.armed = True

        def __enter__(self):
            self.delegate.acquire()
            return self

        def __exit__(self, *_args) -> None:
            self.delegate.release()
            if self.armed:
                self.armed = False
                first_check_released_lock.set()
                assert release_worker.wait(timeout=2.0)

    controller._lock = FirstExitBarrierLock(controller._lock)
    remote_worker = Thread(target=lambda: executor.complete(1))
    remote_worker.start()
    assert first_check_released_lock.wait(timeout=1.0)

    reset_done = Event()
    reset_worker = Thread(target=lambda: (controller.reset(), reset_done.set()))
    reset_worker.start()
    reset_returned_before_worker_handoff = reset_done.wait(timeout=0.1)
    release_worker.set()
    remote_worker.join(timeout=1.0)
    reset_worker.join(timeout=1.0)

    assert not reset_returned_before_worker_handoff
    assert not remote_worker.is_alive()
    assert not reset_worker.is_alive()
    assert len(strategy.requests) == requests_before_reset


def test_reset_cannot_complete_inside_remote_submission_handoff(monkeypatch) -> None:
    executor = ManualExecutor(run_submissions=1)
    preparation_executor = ManualExecutor()
    controller, _, strategy, _, _, clock, publisher = _controller(
        executor=executor,
        preparation_executor=preparation_executor,
    )
    controller.start()
    _complete_pair_two_staging_when_async(preparation_executor)
    clock.value = 3.0
    executor.complete(1)
    controller.on_tick(31)
    pair_four_staging = len(preparation_executor.tasks) - 1

    create_entered = Event()
    release_create = Event()
    original_create = chunk_realtime_module.RemoteRapChunkRequest.create

    class BlockingRequestFactory:
        @staticmethod
        def create(**kwargs):
            create_entered.set()
            if not release_create.wait(timeout=2.0):
                raise RuntimeError("test did not release request creation")
            return original_create(**kwargs)

    monkeypatch.setattr(chunk_realtime_module, "RemoteRapChunkRequest", BlockingRequestFactory)
    staging_worker = Thread(target=lambda: preparation_executor.complete(pair_four_staging))
    staging_worker.start()
    assert create_entered.wait(timeout=1.0)

    reset_done = Event()
    reset_event_count: list[int] = []

    def reset_controller() -> None:
        controller.reset()
        reset_event_count.append(len(publisher.events))
        reset_done.set()

    reset_worker = Thread(target=reset_controller)
    reset_worker.start()
    reset_completed_inside_handoff = reset_done.wait(timeout=0.1)
    release_create.set()
    staging_worker.join(timeout=1.0)
    reset_worker.join(timeout=1.0)
    assert not staging_worker.is_alive()
    assert not reset_worker.is_alive()

    requests_after_reset = len(strategy.requests)
    events_after_reset = reset_event_count[0]
    executor.complete(len(executor.tasks) - 1)

    assert not reset_completed_inside_handoff
    assert len(strategy.requests) == requests_after_reset
    assert len(publisher.events) == events_after_reset


def test_reset_rejects_queued_old_epoch_fallback_preparation_before_it_emits_or_submits() -> None:
    executor = ManualExecutor(run_submissions=1)
    preparation_executor = ManualExecutor()
    controller, _, _, _, _, clock, publisher = _controller(
        executor=executor,
        preparation_executor=preparation_executor,
    )
    controller.start()
    _complete_pair_two_staging_when_async(preparation_executor)
    clock.value = 3.0
    executor.complete(1)
    controller.on_tick(31)
    stale_staging = len(preparation_executor.tasks) - 1
    event_count = len(publisher.events)

    controller.reset()
    preparation_executor.complete(stale_staging)

    assert len(publisher.events) == event_count
    assert len(executor.tasks) == 2


def test_close_aborts_and_permanently_closes_strategy_and_executor_once() -> None:
    controller, _, strategy, _, executor, _, _ = _controller()
    controller.start()

    controller.close()
    controller.close()

    assert strategy.abort_calls == 2  # startup timeout plus close
    assert strategy.close_calls == 1
    assert executor.shutdown_calls == [(True, True)]


def test_chunk_events_are_bounded_and_existing_bar_and_tick_events_remain_available() -> None:
    executor = ManualExecutor(run_submissions=1)
    controller, _, _, _, _, _, publisher = _controller(executor=executor)
    controller.start()
    controller.on_tick(0)

    event_types = {item["event_type"] for item in publisher.events}
    assert {
        RapEventType.CHUNK_REQUEST_SUBMITTED,
        RapEventType.CHUNK_REMOTE_COMPLETED,
        RapEventType.CHUNK_COMMITTED,
        RapEventType.BAR_FROZEN,
        RapEventType.BAR_AUDIO_READY,
        RapEventType.BAR_AUDIO_COMMITTED,
        RapEventType.TICK,
    } <= event_types
    for event in publisher.events:
        assert not _contains_bytes(event)
    committed = _events(publisher, RapEventType.CHUNK_COMMITTED)[0]["payload"]
    assert committed["renderer_decision"] == "moss_aligned_remote"
    assert committed["selected_lines"] == ["stars", "space"]
    assert len(committed["flows"]) == 2


def test_chunk_event_strings_and_nested_aggregates_have_hard_bounds() -> None:
    long_line = "line-" + "x" * 100_000
    long_warning = "warning-" + "y" * 100_000
    long_key = "stage-" + "z" * 10_000

    def oversized_chunk(request) -> PreparedRapChunk:
        return replace(
            _chunk_for(request, texts=(long_line, long_line)),
            diagnostics={
                "stage_timings_ms": {f"{long_key}-{index}": float(index) for index in range(100)},
                "warnings": tuple(f"{long_warning}-{index}" for index in range(100)),
            },
        )

    executor = ManualExecutor(run_submissions=1)
    strategy = RecordingStrategy((_chunk_for, oversized_chunk))
    controller, _, _, _, _, clock, publisher = _controller(strategy=strategy, executor=executor)
    controller.start()
    clock.value = 2.0
    executor.complete(1)
    controller.on_tick(31)

    payload = _events(publisher, RapEventType.CHUNK_COMMITTED)[-1]["payload"]
    assert set(payload) == {
        "state",
        "renderer_decision",
        "chunk_index",
        "bars",
        "selected_lines",
        "flows",
        "stage_timings_ms",
        "deadline_slack_ms",
        "warnings",
    }
    assert all(len(item.encode("utf-8")) <= 512 for item in payload["selected_lines"])
    assert len(payload["warnings"]) <= 8
    assert all(len(item.encode("utf-8")) <= 256 for item in payload["warnings"])
    assert len(payload["stage_timings_ms"]) <= 8
    assert all(len(key.encode("utf-8")) <= 64 for key in payload["stage_timings_ms"])
    assert len(json.dumps(payload, ensure_ascii=False).encode("utf-8")) <= 24_000
    _assert_bounded_collections(payload)


def test_chunk_rejection_bounds_arbitrary_exception_text() -> None:
    executor = ManualExecutor(run_submissions=1)
    strategy = RecordingStrategy((_chunk_for, RuntimeError("remote-error-" + "e" * 100_000)))
    controller, _, _, _, _, clock, publisher = _controller(strategy=strategy, executor=executor)
    controller.start()
    clock.value = 2.0
    executor.complete(1)
    controller.on_tick(31)

    rejected = _events(publisher, RapEventType.CHUNK_REMOTE_REJECTED)[-1]["payload"]
    committed = _events(publisher, RapEventType.CHUNK_COMMITTED)[-1]["payload"]
    assert all(len(item.encode("utf-8")) <= 256 for item in rejected["warnings"])
    assert all(len(item.encode("utf-8")) <= 256 for item in committed["warnings"])
    assert len(json.dumps(rejected, ensure_ascii=False).encode("utf-8")) <= 24_000


def _contains_bytes(value) -> bool:
    if isinstance(value, bytes):
        return True
    if isinstance(value, dict):
        return any(_contains_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_bytes(item) for item in value)
    return False


def _assert_bounded_collections(value, *, depth: int = 0) -> None:
    assert depth <= 6
    if isinstance(value, dict):
        assert len(value) <= 16
        for item in value.values():
            _assert_bounded_collections(item, depth=depth + 1)
    elif isinstance(value, (list, tuple)):
        assert len(value) <= 32
        for item in value:
            _assert_bounded_collections(item, depth=depth + 1)
