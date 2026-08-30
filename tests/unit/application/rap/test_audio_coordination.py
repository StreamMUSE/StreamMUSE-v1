"""Tests for fallback-first atomic rap bar audio coordination."""

from __future__ import annotations

from io import StringIO
from threading import Event, Thread
from time import monotonic, sleep

import pytest

from streammuse.application.rap.audio_coordination import BarAudioCoordinator
from streammuse.application.rap.monitoring import RapEventDispatcher, RapEventPublisher, RapStateProjector
from streammuse.application.rap.realtime import PlannedRapBar
from streammuse.domain.rap import (
    AudioFormat,
    AudioWarning,
    AudioWarningCode,
    AudioWarningSeverity,
    PcmAudio,
    PreparedRapBar,
    RapEventType,
    SyllablePlacementDiagnostic,
)
from streammuse.presentation.rap_demo.terminal_dashboard import build_dashboard
from streammuse.presentation.rap_demo.terminal_state import TerminalRapStateProjector
from rich.console import Console


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
        self.warnings: dict[tuple[int, str], tuple[AudioWarning, ...]] = {}

    def render(self, plan: PlannedRapBar) -> PreparedRapBar:
        key = (plan.bar, plan.source)
        self.calls.append(key)
        self.started.setdefault(key, Event()).set()
        assert self.release.setdefault(key, Event()).wait(timeout=1.0)
        return prepared(plan, warnings=self.warnings.get(key, ()))

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


def prepared(
    plan: PlannedRapBar,
    *,
    warnings: tuple[AudioWarning, ...] = (),
    diagnostics: tuple[SyllablePlacementDiagnostic, ...] = (),
) -> PreparedRapBar:
    audio_format = AudioFormat()
    return PreparedRapBar(
        bar=plan.bar,
        text=plan.text,
        source=plan.source,
        fallback_reason=plan.fallback_reason,
        scheduled=plan.scheduled,
        audio=PcmAudio(audio_format, 1, bytes(8)),
        diagnostics=diagnostics,
        warnings=warnings,
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


def test_primary_audio_completed_after_absolute_deadline_cannot_beat_ready_fallback() -> None:
    now = [0.0]
    renderer = ControlledBarRenderer()
    publisher = RecordingPublisher()
    coordinator = BarAudioCoordinator(renderer, publisher=publisher, monotonic=lambda: now[0])
    fallback = planned_bar(bar=3, source="prevalidated_fallback")
    primary = planned_bar(bar=3, source="local_chat")
    try:
        coordinator.reserve_fallback(fallback)
        renderer.wait_started(fallback)
        now[0] = 9.0
        renderer.complete(fallback)
        _wait_for_render_completion(publisher, role="fallback")

        coordinator.submit_primary(primary)
        renderer.wait_started(primary)
        now[0] = 10.001
        renderer.complete(primary)
        assert _wait_for_primary(coordinator, 3).source == "local_chat"

        committed = coordinator.commit(3, deadline_monotonic=10.0)

        assert committed.source == "prevalidated_fallback"
    finally:
        renderer.complete(fallback)
        renderer.complete(primary)
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


def test_render_events_mark_stale_reset_work_unaccepted_without_ready_or_warnings() -> None:
    renderer = ControlledBarRenderer()
    publisher = RecordingPublisher()
    coordinator = BarAudioCoordinator(renderer, publisher=publisher)
    accepted = planned_bar(bar=7, source="prevalidated_fallback")
    stale = planned_bar(bar=8, source="prevalidated_fallback")
    warnings = (
        AudioWarning(
            code=AudioWarningCode.PRONUNCIATION_FALLBACK,
            severity=AudioWarningSeverity.WARNING,
            message="fallback pronunciation",
            word="streammuse",
        ),
        AudioWarning(
            code=AudioWarningCode.TIMING_PRESSURE,
            severity=AudioWarningSeverity.WARNING,
            message="compressed syllable",
            compression_ratio=1.2,
        ),
    )
    renderer.warnings[(7, "prevalidated_fallback")] = warnings
    renderer.warnings[(8, "prevalidated_fallback")] = warnings
    try:
        coordinator.reserve_fallback(accepted)
        renderer.wait_started(accepted)
        renderer.complete(accepted)
        coordinator.commit(7)

        accepted_events = [event for event in publisher.events if event[1] == 7]
        assert [event[0] for event in accepted_events] == [
            RapEventType.AUDIO_RENDER_STARTED,
            RapEventType.AUDIO_RENDER_COMPLETED,
            RapEventType.BAR_AUDIO_READY,
            RapEventType.PRONUNCIATION_FALLBACK,
            RapEventType.TIMING_PRESSURE,
        ]
        assert accepted_events[1][2]["accepted"] is True
        assert accepted_events[2][2]["warnings"] == ["pronunciation_fallback", "timing_pressure"]

        coordinator.reserve_fallback(stale)
        renderer.wait_started(stale)
        coordinator.reset()
        renderer.complete(stale)
        _wait_for_event_count(publisher, bar=8, count=2)

        stale_events = [event for event in publisher.events if event[1] == 8]
        assert [event[0] for event in stale_events] == [
            RapEventType.AUDIO_RENDER_STARTED,
            RapEventType.AUDIO_RENDER_COMPLETED,
        ]
        assert stale_events[1][2]["accepted"] is False
        assert stale_events[1][2]["warnings"] == ["pronunciation_fallback", "timing_pressure"]
    finally:
        coordinator.close()


def test_stale_render_completion_retains_its_original_coordinator_epoch() -> None:
    renderer = ControlledBarRenderer()
    publisher = RecordingPublisher()
    coordinator = BarAudioCoordinator(renderer, publisher=publisher)
    stale = planned_bar(bar=8, source="prevalidated_fallback")
    current = planned_bar(bar=0, source="prevalidated_fallback")
    try:
        coordinator.reserve_fallback(stale)
        renderer.wait_started(stale)
        coordinator.reset()
        renderer.complete(stale)
        _wait_for_event_count(publisher, bar=8, count=2)
        coordinator.reserve_fallback(current)
        renderer.wait_started(current)
        renderer.complete(current)
        coordinator.commit(0)

        stale_events = [event for event in publisher.events if event[1] == 8]
        current_events = [event for event in publisher.events if event[1] == 0]
        assert stale_events[0][2]["coordinator_epoch"] == 0
        assert stale_events[1][2]["coordinator_epoch"] == 0
        assert stale_events[1][2]["accepted"] is False
        assert current_events[0][2]["coordinator_epoch"] == 1
        assert current_events[1][2]["coordinator_epoch"] == 1
    finally:
        coordinator.close()


def test_coordinator_produces_complete_warning_and_latency_evidence_for_terminal_and_monitor() -> None:
    class Renderer:
        def render(self, plan: PlannedRapBar) -> PreparedRapBar:
            return prepared(
                plan,
                diagnostics=(
                    SyllablePlacementDiagnostic(
                        bar=plan.bar,
                        slot_index=3,
                        word="StreamMUSE",
                        target_sample=12_345,
                        source_frames=5_000,
                        fitted_frames=4_000,
                        available_frames=4_000,
                        compression_ratio=1.25,
                        overlap_frames=48,
                        pronunciation_source="espeak_g2p",
                        renderer_phonemes=("str", "i:", "mju:z"),
                        synthesis_latency_ms=7.5,
                    ),
                ),
                warnings=(
                    AudioWarning(AudioWarningCode.PRONUNCIATION_FALLBACK, AudioWarningSeverity.WARNING, "fallback", slot_index=3, word="StreamMUSE", action="fallback"),
                    AudioWarning(AudioWarningCode.TIMING_PRESSURE, AudioWarningSeverity.WARNING, "pressure", slot_index=3, word="StreamMUSE", available_ms=80.0, rendered_ms=100.0, compression_ratio=1.25, overlap_ms=1.0, action="compress"),
                    AudioWarning(AudioWarningCode.FORCED_BAR_FIT, AudioWarningSeverity.WARNING, "fit", slot_index=3, word="StreamMUSE", action="pitch_preserving_stretch_to_bar"),
                    AudioWarning(AudioWarningCode.SYNTHESIS_FAILED, AudioWarningSeverity.ERROR, "failed", slot_index=3, word="StreamMUSE", action="empty_pcm"),
                ),
            )

    monitor = RapStateProjector()
    terminal = TerminalRapStateProjector()
    publisher = RapEventPublisher("producer-path")
    dispatcher = RapEventDispatcher(publisher.queue, sinks=(monitor, terminal))
    dispatcher.start()
    coordinator = BarAudioCoordinator(Renderer(), publisher=publisher)
    plan = planned_bar(bar=0, source="local_chat")
    try:
        coordinator.reserve_fallback(plan)
        assert coordinator.commit(0).bar == 0
    finally:
        coordinator.close()
        dispatcher.flush_and_close()

    monitor_state = monitor.snapshot()
    terminal_state = terminal.state
    warnings = terminal_state.audio_warnings
    assert {warning["type"] for warning in warnings} == {
        "pronunciation_fallback",
        "timing_pressure",
        "forced_bar_fit",
        "synthesis_failed",
    }
    assert all(warning["target_sample"] == 12_345 for warning in warnings)
    assert all(warning["renderer_phonemes"] == ("str", "i:", "mju:z") for warning in warnings)
    assert all(warning["source"] == "espeak_g2p" for warning in warnings)
    assert monitor_state["latencies"]["synthesis_latency_ms"]["total"] == 7.5
    assert monitor_state["latencies"]["bar_render_latency_ms"]["count"] == 1
    rendered_payload = BarAudioCoordinator._audio_payload(
        prepared(plan, diagnostics=(
            SyllablePlacementDiagnostic(
                bar=plan.bar,
                slot_index=3,
                word="StreamMUSE",
                target_sample=12_345,
                source_frames=5_000,
                fitted_frames=4_000,
                available_frames=4_000,
                compression_ratio=1.25,
                overlap_frames=48,
                pronunciation_source="espeak_g2p",
            ),
        )),
        role="fallback",
        accepted=True,
        coordinator_epoch=0,
    )
    assert rendered_payload["vocal_syllable_count"] == 1
    assert rendered_payload["vocal_source_frames"] == 5_000
    assert rendered_payload["vocal_fitted_frames"] == 4_000
    assert rendered_payload["pronunciation_sources"] == {"espeak_g2p": 1}

    stream = StringIO()
    Console(file=stream, force_terminal=False, width=120).print(build_dashboard(terminal_state, detail="full", width=120))
    assert "target_sample=12345" in stream.getvalue()
    assert "source=espeak_g2p" in stream.getvalue()


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


def _wait_for_render_completion(publisher: RecordingPublisher, *, role: str) -> None:
    deadline = monotonic() + 1.0
    while monotonic() < deadline:
        if any(
            event_type == RapEventType.AUDIO_RENDER_COMPLETED and payload.get("role") == role
            for event_type, _bar, payload in publisher.events
        ):
            return
        sleep(0.001)
    raise AssertionError(f"{role} render did not complete")


def _wait_for_event_count(publisher: RecordingPublisher, *, bar: int, count: int) -> None:
    deadline = monotonic() + 1.0
    while monotonic() < deadline:
        if len([event for event in publisher.events if event[1] == bar]) >= count:
            return
        sleep(0.005)
    raise AssertionError(f"expected {count} events for bar {bar}")
