"""Tests for fallback-first atomic rap bar audio coordination."""

from __future__ import annotations

from threading import Event, Thread
from time import monotonic, sleep

import pytest

from streammuse.application.rap.audio_coordination import BarAudioCoordinator
from streammuse.application.rap.realtime import PlannedRapBar
from streammuse.domain.rap import (
    AudioFormat,
    PcmAudio,
    PreparedRapBar,
    RapEventType,
)


class RecordingPublisher:
    def __init__(self) -> None:
        self.events: list[tuple[RapEventType, int | None, dict[str, object]]] = []

    def emit(self, event_type: RapEventType, **kwargs: object) -> None:
        self.events.append((event_type, kwargs.get("bar"), kwargs.get("payload", {})))


class ControlledBarRenderer:
    def __init__(self) -> None:
        self.started: dict[tuple[int, str], Event] = {}
        self.release: dict[tuple[int, str], Event] = {}
        self.calls: list[tuple[int, str]] = []

    def render(self, plan: PlannedRapBar) -> PreparedRapBar:
        key = (plan.bar, plan.source)
        self.calls.append(key)
        self.started.setdefault(key, Event()).set()
        assert self.release.setdefault(key, Event()).wait(timeout=1.0)
        return prepared(plan)

    def complete(self, plan: PlannedRapBar) -> None:
        self.release.setdefault((plan.bar, plan.source), Event()).set()

    def wait_started(self, plan: PlannedRapBar) -> None:
        assert self.started.setdefault((plan.bar, plan.source), Event()).wait(timeout=1.0)


def planned_bar(*, bar: int, source: str) -> PlannedRapBar:
    return PlannedRapBar(
        bar=bar,
        segment=None,  # type: ignore[arg-type]
        template=None,  # type: ignore[arg-type]
        analysis=None,  # type: ignore[arg-type]
        scheduled=(),
        text=f"{source} line",
        source=source,
        fallback_reason="generation_pending" if source == "prevalidated_fallback" else None,
    )


def prepared(plan: PlannedRapBar) -> PreparedRapBar:
    audio_format = AudioFormat()
    return PreparedRapBar(
        bar=plan.bar,
        text=plan.text,
        source=plan.source,
        fallback_reason=plan.fallback_reason,
        scheduled=plan.scheduled,
        audio=PcmAudio(audio_format, 1, bytes(8)),
        diagnostics=(),
        warnings=(),
        render_latency_ms=12.5,
    )


def test_fallback_render_can_start_while_primary_worker_is_busy() -> None:
    renderer = ControlledBarRenderer()
    coordinator = BarAudioCoordinator(renderer, publisher=RecordingPublisher())
    fallback_one = planned_bar(bar=1, source="prevalidated_fallback")
    primary_one = planned_bar(bar=1, source="local_chat")
    fallback_two = planned_bar(bar=2, source="prevalidated_fallback")
    try:
        coordinator.reserve_fallback(fallback_one)
        renderer.wait_started(fallback_one)
        coordinator.submit_primary(primary_one)
        renderer.wait_started(primary_one)
        renderer.complete(fallback_one)
        assert coordinator.commit(1).source == "prevalidated_fallback"

        coordinator.reserve_fallback(fallback_two)

        renderer.wait_started(fallback_two)
    finally:
        renderer.complete(primary_one)
        renderer.complete(fallback_two)
        coordinator.close()


def test_fallback_render_does_not_queue_behind_primary_work_from_other_bars() -> None:
    renderer = ControlledBarRenderer()
    coordinator = BarAudioCoordinator(renderer, publisher=RecordingPublisher())
    fallback_one = planned_bar(bar=1, source="prevalidated_fallback")
    primary_one = planned_bar(bar=1, source="local_chat")
    fallback_two = planned_bar(bar=2, source="prevalidated_fallback")
    primary_two = planned_bar(bar=2, source="local_chat")
    fallback_three = planned_bar(bar=3, source="prevalidated_fallback")
    try:
        coordinator.reserve_fallback(fallback_one)
        renderer.wait_started(fallback_one)
        coordinator.submit_primary(primary_one)
        renderer.wait_started(primary_one)
        renderer.complete(fallback_one)

        coordinator.reserve_fallback(fallback_two)
        renderer.wait_started(fallback_two)
        renderer.complete(fallback_two)
        coordinator.submit_primary(primary_two)

        coordinator.reserve_fallback(fallback_three)

        renderer.wait_started(fallback_three)
    finally:
        for plan in (primary_one, primary_two, fallback_three):
            renderer.complete(plan)
        coordinator.close()


def test_ready_primary_is_observable_once_and_committed_atomically() -> None:
    renderer = ControlledBarRenderer()
    publisher = RecordingPublisher()
    coordinator = BarAudioCoordinator(renderer, publisher=publisher)
    fallback = planned_bar(bar=1, source="prevalidated_fallback")
    primary = planned_bar(bar=1, source="local_chat")
    try:
        coordinator.reserve_fallback(fallback)
        coordinator.submit_primary(primary)
        renderer.wait_started(fallback)
        renderer.wait_started(primary)
        renderer.complete(primary)
        observed = _wait_for_primary(coordinator, 1)
        assert observed.source == "local_chat"
        assert coordinator.poll_primary(1) is None
        renderer.complete(fallback)

        assert coordinator.commit(1) is observed
        assert [event[0] for event in publisher.events].count(RapEventType.BAR_AUDIO_READY) == 1
    finally:
        coordinator.close()


def test_committed_bar_rejects_late_primary_audio() -> None:
    renderer = ControlledBarRenderer()
    coordinator = BarAudioCoordinator(renderer, publisher=RecordingPublisher())
    fallback = planned_bar(bar=2, source="prevalidated_fallback")
    primary = planned_bar(bar=2, source="local_chat")
    try:
        coordinator.reserve_fallback(fallback)
        coordinator.submit_primary(primary)
        renderer.wait_started(fallback)
        renderer.wait_started(primary)
        renderer.complete(fallback)
        committed = coordinator.commit(2)
        renderer.complete(primary)

        assert coordinator.commit(2) is committed
        assert coordinator.poll_primary(2) is None
    finally:
        coordinator.close()


def test_reset_cancels_uncommitted_work_and_close_is_permanent() -> None:
    renderer = ControlledBarRenderer()
    coordinator = BarAudioCoordinator(renderer, publisher=RecordingPublisher())
    fallback = planned_bar(bar=3, source="prevalidated_fallback")
    replacement = planned_bar(bar=4, source="prevalidated_fallback")
    coordinator.reserve_fallback(fallback)
    renderer.wait_started(fallback)

    coordinator.reset()
    renderer.complete(fallback)
    coordinator.reserve_fallback(replacement)
    renderer.wait_started(replacement)
    renderer.complete(replacement)
    assert coordinator.commit(4).bar == 4

    coordinator.close()
    with pytest.raises(RuntimeError, match="closed"):
        coordinator.reserve_fallback(planned_bar(bar=5, source="prevalidated_fallback"))


def test_concurrent_duplicate_fallback_reservations_render_once() -> None:
    renderer = ControlledBarRenderer()
    coordinator = BarAudioCoordinator(renderer, publisher=RecordingPublisher())
    fallback = planned_bar(bar=6, source="prevalidated_fallback")
    workers = [Thread(target=lambda: coordinator.reserve_fallback(fallback)) for _ in range(8)]
    try:
        for worker in workers:
            worker.start()
        renderer.wait_started(fallback)
        renderer.complete(fallback)
        for worker in workers:
            worker.join(timeout=1.0)

        assert all(not worker.is_alive() for worker in workers)
        assert renderer.calls == [(6, "prevalidated_fallback")]
        assert coordinator.commit(6).source == "prevalidated_fallback"
    finally:
        coordinator.close()


def _wait_for_primary(coordinator: BarAudioCoordinator, bar: int) -> PreparedRapBar:
    deadline = monotonic() + 1.0
    while monotonic() < deadline:
        result = coordinator.poll_primary(bar)
        if result is not None:
            return result
        sleep(0.005)
    raise AssertionError("primary audio did not become ready")
