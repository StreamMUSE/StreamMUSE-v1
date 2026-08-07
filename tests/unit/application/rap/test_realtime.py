"""Tests for the rolling tick-driven rap controller."""

from __future__ import annotations

from concurrent.futures import Future
from threading import Event
from time import monotonic

from streammuse.application.rap.realtime import RollingRapController
from streammuse.domain.rap import CandidateBatch, CandidateRequest
from streammuse.domain.timing import Tempo


class FixedGenerator:
    def __init__(self, *, source: str = "phrase_bank", candidates: tuple[str, ...] | None = None) -> None:
        self.source = source
        self.candidates = candidates or (
            "steady drums carry the city through the night",
            "we move with rhythm and hold the light",
            "bright echoes travel through the open room",
        )

    def generate(self, request: CandidateRequest) -> CandidateBatch:
        return CandidateBatch(
            request_id=request.request_id,
            candidates=self.candidates,
            source=self.source,
            prompt=(),
            raw_response="",
            latency_ms=0.0,
        )


class BlockingGenerator:
    def __init__(self) -> None:
        self.release = Event()
        self.started = Event()

    def generate(self, request: CandidateRequest) -> CandidateBatch:
        self.started.set()
        self.release.wait(timeout=1.0)
        return FixedGenerator(source="local_chat").generate(request)


class ManualExecutor:
    def __init__(self) -> None:
        self.future: Future[CandidateBatch] | None = None
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


def _controller(*, primary=None, executor=None):
    emitted = []
    controller = RollingRapController(
        tempo=Tempo(bpm=120.0, ticks_per_beat=4, beats_per_bar=4),
        topic="city lights",
        pattern="boom_bap",
        fallback_generator=FixedGenerator(),
        primary_generator=primary,
        candidate_count=3,
        lookahead_bars=2,
        emit=emitted.append,
        executor=executor,
    )
    return controller, emitted


def test_controller_preloads_and_emits_fallback_bars_on_absolute_ticks() -> None:
    controller, emitted = _controller()

    controller.start()
    assert controller.line_for_bar(0).events[0].slot.tick == 0
    assert controller.line_for_bar(1).events[0].slot.tick == 16

    controller.on_tick(0)
    controller.on_tick(16)

    assert [event.slot.tick for event in emitted] == [0, 16]
    assert controller.line_for_bar(2).events[0].slot.tick == 32
    controller.close()


def test_slow_primary_generation_never_blocks_tick_delivery() -> None:
    primary = BlockingGenerator()
    controller, emitted = _controller(primary=primary)
    controller.start()
    assert primary.started.wait(timeout=0.5)

    started = monotonic()
    controller.on_tick(0)

    assert monotonic() - started < 0.1
    assert [event.slot.tick for event in emitted] == [0]
    primary.release.set()
    controller.close()


def test_completed_primary_replaces_only_a_future_fallback_bar() -> None:
    executor = ManualExecutor()
    controller, _emitted = _controller(
        primary=FixedGenerator(
            source="local_chat",
            candidates=("future stars carry rhythm through the night",),
        ),
        executor=executor,
    )
    controller.start()
    fallback_text = controller.line_for_bar(1).text

    executor.complete()
    controller.on_tick(0)

    assert controller.line_source_for_bar(1) == "local_chat"
    assert controller.line_for_bar(1).text != fallback_text
    controller.close()


def test_late_primary_result_cannot_replace_a_started_bar() -> None:
    executor = ManualExecutor()
    controller, _emitted = _controller(
        primary=FixedGenerator(
            source="local_chat",
            candidates=("future stars carry rhythm through the night",),
        ),
        executor=executor,
    )
    controller.start()
    fallback_text = controller.line_for_bar(1).text

    controller.on_tick(16)
    executor.complete()
    controller.on_tick(17)

    assert controller.line_source_for_bar(1) == "phrase_bank"
    assert controller.line_for_bar(1).text == fallback_text
    controller.close()
    assert executor.shutdown_calls == [(False, True)]
