"""Tests for the scenario-aware rolling rap controller."""

from __future__ import annotations

from concurrent.futures import Future
from threading import Event
from time import monotonic

from streammuse.application.rap.monitoring import RapEventDispatcher, RapEventPublisher
from streammuse.application.rap.realtime import RollingRapController
from streammuse.domain.rap import (
    CandidateBatch,
    CandidateRequest,
    FlowProvenance,
    FlowSlot,
    FlowTemplate,
    RapScenario,
    ScenarioSegment,
    ScoreWeights,
)
from streammuse.domain.timing import Tempo
from streammuse.infrastructure.rap.fallback import PrevalidatedFallbackCatalog
from streammuse.infrastructure.rap.prosody import CmuProsodyAnalyzer
from streammuse.infrastructure.rap.templates import TemplateCatalog


class FixedGenerator:
    def __init__(
        self,
        candidates: tuple[str, ...] = ("space",),
        *,
        source: str = "local_chat",
        error_type: str | None = None,
    ) -> None:
        self.candidates = candidates
        self.source = source
        self.error_type = error_type
        self.requests: list[CandidateRequest] = []

    def generate(self, request: CandidateRequest) -> CandidateBatch:
        self.requests.append(request)
        return CandidateBatch(
            request_id=request.request_id,
            candidates=self.candidates,
            source=self.source,
            prompt=({"role": "user", "content": "prompt"},),
            raw_response="\n".join(self.candidates),
            latency_ms=4.0,
            error_type=self.error_type,
            error_message="scripted failure" if self.error_type else None,
        )


class BlockingGenerator(FixedGenerator):
    def __init__(self) -> None:
        super().__init__()
        self.release = Event()
        self.started = Event()

    def generate(self, request: CandidateRequest) -> CandidateBatch:
        self.started.set()
        self.release.wait(timeout=1.0)
        return super().generate(request)


class ManualExecutor:
    def __init__(self) -> None:
        self.future: Future | None = None
        self._fn = None
        self.shutdown_calls: list[tuple[bool, bool]] = []

    def submit(self, fn, *args):
        self._fn = lambda: fn(*args)
        self.future = Future()
        return self.future

    def complete(self) -> None:
        assert self.future is not None
        assert self._fn is not None
        self.future.set_result(self._fn())

    def shutdown(self, *, wait: bool, cancel_futures: bool) -> None:
        self.shutdown_calls.append((wait, cancel_futures))


def _templates() -> TemplateCatalog:
    return TemplateCatalog.from_templates(
        (
            FlowTemplate(
                template_id="one_slot",
                name="One slot",
                ticks_per_beat=4,
                beats_per_bar=4,
                slots=(FlowSlot(0, 16, 1.0, rhyme_group="A"),),
                provenance=FlowProvenance(kind="test", source="test"),
            ),
        )
    )


def _scenario() -> RapScenario:
    return RapScenario(
        scenario_id="test",
        tempo_bpm=120.0,
        segments=(ScenarioSegment(0, 4, "space", "one_slot", ("beat", "light")),),
    )


def _controller(
    *,
    primary=None,
    executor=None,
    lookahead_bars: int = 2,
    minimum_score: float = 0.0,
    planning_bar_limit: int | None = None,
):
    analyzer = CmuProsodyAnalyzer()
    scenario = _scenario()
    templates = _templates()
    publisher = RapEventPublisher("test-session")
    events = []
    dispatcher = RapEventDispatcher(publisher.queue, sinks=(events.append,))
    dispatcher.start()
    emitted = []
    controller = RollingRapController(
        tempo=Tempo(120.0, 4, 4),
        scenario=scenario,
        templates=templates,
        fallback_catalog=PrevalidatedFallbackCatalog.build(scenario, templates, analyzer),
        analyzer=analyzer,
        weights=ScoreWeights(),
        publisher=publisher,
        primary_generator=primary,
        candidate_count=3,
        lookahead_bars=lookahead_bars,
        minimum_score=minimum_score,
        seed=7,
        planning_bar_limit=planning_bar_limit,
        emit=emitted.append,
        executor=executor,
    )
    return controller, emitted, events, dispatcher


def _finish(controller, dispatcher) -> None:
    controller.close()
    dispatcher.flush_and_close()


def _types(events) -> list[str]:
    return [event.event_type.value for event in events]


def test_start_reserves_prevalidated_fallback_through_lookahead() -> None:
    controller, _emitted, events, dispatcher = _controller(lookahead_bars=2)
    controller.start()
    _finish(controller, dispatcher)

    assert controller.bar_state(0) == "reserved"
    assert controller.bar_state(1) == "reserved"
    assert _types(events).count("bar_reserved") == 2
    assert controller.bar_for(0).source == "prevalidated_fallback"


def test_fallback_bars_freeze_before_emission_without_gaps() -> None:
    controller, emitted, events, dispatcher = _controller()
    controller.start()
    for tick in range(33):
        controller.on_tick(tick)
    _finish(controller, dispatcher)

    assert [item.slot.bar for item in emitted] == [0, 1, 2]
    assert all(controller.bar_for(bar).frozen for bar in range(3))
    assert _types(events).count("bar_frozen") == 3
    frozen_sequences = {event.bar: event.sequence for event in events if event.event_type.value == "bar_frozen"}
    for event in (item for item in events if item.event_type.value == "syllable_emitted"):
        assert frozen_sequences[event.bar] < event.sequence


def test_valid_primary_replaces_only_unfrozen_reservation_and_logs_components() -> None:
    executor = ManualExecutor()
    controller, _emitted, events, dispatcher = _controller(primary=FixedGenerator(), executor=executor)
    controller.start()
    executor.complete()
    controller.on_tick(0)
    _finish(controller, dispatcher)

    assert controller.bar_for(1).source == "local_chat"
    assert controller.bar_for(1).text == "space"
    assert "bar_replaced" in _types(events)
    evaluated = next(event for event in events if event.event_type.value == "candidate_evaluated")
    assert evaluated.payload["valid"] is True
    assert evaluated.payload["selected"] is True
    assert evaluated.payload["components"]
    assert evaluated.payload["word_analysis_sources"] == [{"word": "space", "source": "cmudict_first_pronunciation"}]


def test_realtime_request_owns_the_reserved_template_used_for_ranking() -> None:
    executor = ManualExecutor()
    generator = FixedGenerator()
    controller, _emitted, _events, dispatcher = _controller(primary=generator, executor=executor)
    controller.start()
    executor.complete()
    controller.on_tick(0)
    _finish(controller, dispatcher)

    assert generator.requests[0].flow_template is controller.bar_for(1).template


def test_invalid_and_error_batches_retain_fallback_with_visible_reasons() -> None:
    for generator, expected in (
        (FixedGenerator(("too many",)), "no_valid_candidate"),
        (FixedGenerator((), error_type="generation_error"), "generation_error"),
    ):
        executor = ManualExecutor()
        controller, _emitted, events, dispatcher = _controller(primary=generator, executor=executor)
        controller.start()
        executor.complete()
        controller.on_tick(0)
        controller.on_tick(16)
        _finish(controller, dispatcher)

        assert controller.bar_for(1).source == "prevalidated_fallback"
        frozen = next(event for event in events if event.event_type.value == "bar_frozen" and event.bar == 1)
        assert frozen.payload["fallback_reason"] == expected
        if expected == "generation_error":
            assert "generation_failed" in _types(events)


def test_late_result_is_logged_and_cannot_replace_frozen_bar() -> None:
    executor = ManualExecutor()
    controller, _emitted, events, dispatcher = _controller(primary=FixedGenerator(), executor=executor)
    controller.start()
    controller.on_tick(16)
    executor.complete()
    controller.on_tick(17)
    _finish(controller, dispatcher)

    assert controller.bar_for(1).source == "prevalidated_fallback"
    batch = next(event for event in events if event.event_type.value == "candidate_batch_received")
    assert batch.payload["late"] is True


def test_slow_generation_never_blocks_tick_path() -> None:
    primary = BlockingGenerator()
    controller, emitted, _events, dispatcher = _controller(primary=primary)
    controller.start()
    assert primary.started.wait(timeout=0.5)

    started = monotonic()
    controller.on_tick(0)
    elapsed = monotonic() - started

    primary.release.set()
    _finish(controller, dispatcher)
    assert elapsed < 0.1
    assert [item.slot.tick for item in emitted] == [0]


def test_fast_generation_keeps_reservations_and_plans_inside_bounded_lookahead() -> None:
    executor = ManualExecutor()
    controller, _emitted, events, dispatcher = _controller(primary=FixedGenerator(), executor=executor)
    controller.start()

    executor.complete()
    controller.on_tick(0)
    executor.complete()
    for tick in range(1, 16):
        controller.on_tick(tick)
    _finish(controller, dispatcher)

    reserved_bars = [event.bar for event in events if event.event_type.value == "bar_reserved"]
    planned_bars = [event.bar for event in events if event.event_type.value == "bar_planning_started"]
    assert reserved_bars == [0, 1, 2]
    assert planned_bars == [1, 2]
    assert controller.bar_state(3) == "unreserved"


def test_finite_session_never_reserves_or_plans_past_its_last_emitted_bar() -> None:
    executor = ManualExecutor()
    controller, _emitted, events, dispatcher = _controller(
        primary=FixedGenerator(),
        executor=executor,
        planning_bar_limit=2,
    )
    controller.start()
    executor.complete()
    for tick in range(32):
        controller.on_tick(tick)
    _finish(controller, dispatcher)

    assert {event.bar for event in events if event.bar is not None} <= {0, 1}
    assert controller.bar_state(2) == "unreserved"
