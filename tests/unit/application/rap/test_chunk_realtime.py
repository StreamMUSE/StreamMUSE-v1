"""Tests for deadline-safe rolling two-bar rap audio preparation."""

from __future__ import annotations

from collections import deque
from concurrent.futures import Future, TimeoutError
from dataclasses import replace

import pytest

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
    def __init__(self) -> None:
        self.plans = []

    def render(self, plan) -> PreparedRapBar:
        self.plans.append(plan)
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
):
    clock = clock or ManualClock()
    executor = executor or ManualExecutor()
    renderer = RecordingRenderer()
    strategy = strategy or RecordingStrategy()
    strategy.renderer = renderer
    scenario = scenario or _scenario()
    template = _template()
    templates = TemplateCatalog.from_templates((template,))
    analyzer = CmuProsodyAnalyzer()
    queued: list[PreparedRapBar] = []
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
        enqueue=queued.append,
        session_id="session-1",
        policy=RemoteCandidatePolicy.realtime_default(),
        seed=7,
        planning_bar_limit=planning_bar_limit,
        startup_timeout_seconds=startup_timeout_seconds,
        rolling_timeout_seconds=rolling_timeout_seconds,
        executor=executor,
        preparation_executor=preparation_executor or ManualExecutor(run_submissions=100),
        monotonic=clock,
    )
    return controller, queued, strategy, renderer, executor, clock, publisher


def _events(publisher: RecordingPublisher, event_type: RapEventType) -> list[dict[str, object]]:
    return [item for item in publisher.events if item["event_type"] == event_type]


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


@pytest.mark.parametrize("outcome", (RuntimeError("remote failed"), _invalid_chunk))
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


def test_running_tick_schedules_following_fallback_render_without_executing_it_inline() -> None:
    executor = ManualExecutor(run_submissions=1)
    preparation_executor = ManualExecutor()
    controller, _, _, renderer, _, clock, _ = _controller(
        executor=executor,
        preparation_executor=preparation_executor,
    )
    controller.start()
    assert len(renderer.plans) == 4
    clock.value = 3.0
    executor.complete(1)

    controller.on_tick(31)

    assert len(renderer.plans) == 4
    assert len(preparation_executor.tasks) == 1
    assert len(executor.tasks) == 2
    preparation_executor.complete(0)
    controller.on_tick(32)
    assert len(renderer.plans) == 6
    assert len(executor.tasks) == 3


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
    looping, _, _, _, _, _, _ = _controller(executor=looping_executor, scenario=_scenario(loop=True, bars=2))
    looping.start()
    looping_executor.complete(1)
    assert tuple(item.bar for item in looping_executor.tasks[1][2][0].bars) == (2, 3)


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


def test_reset_rejects_queued_old_epoch_fallback_preparation_before_it_emits_or_submits() -> None:
    executor = ManualExecutor(run_submissions=1)
    preparation_executor = ManualExecutor()
    controller, _, _, _, _, clock, publisher = _controller(
        executor=executor,
        preparation_executor=preparation_executor,
    )
    controller.start()
    clock.value = 3.0
    executor.complete(1)
    controller.on_tick(31)
    event_count = len(publisher.events)

    controller.reset()
    preparation_executor.complete(0)

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


def _contains_bytes(value) -> bool:
    if isinstance(value, bytes):
        return True
    if isinstance(value, dict):
        return any(_contains_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_bytes(item) for item in value)
    return False
