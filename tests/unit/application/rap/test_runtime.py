"""Tests for the standalone rap tick loop."""

import pytest

from streammuse.application.rap.monitoring import RapEventDispatcher, RapEventPublisher
from streammuse.application.rap.runtime import RapDemoDependencies, RapTickLoop
from streammuse.domain.timing import Tempo


class FakeClock:
    def __init__(self) -> None:
        self.now = 10.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def test_tick_loop_uses_absolute_monotonic_deadlines_without_drift() -> None:
    clock = FakeClock()
    ticks: list[int] = []
    loop = RapTickLoop(Tempo(120.0, 4, 4), on_tick=ticks.append, clock=clock, sleep=clock.sleep)

    loop.run(max_ticks=5)

    assert ticks == [0, 1, 2, 3, 4]
    assert clock.sleeps == pytest.approx([0.125, 0.125, 0.125, 0.125])


def test_tick_loop_compensates_for_callback_time_instead_of_accumulating_drift() -> None:
    clock = FakeClock()

    def on_tick(_tick: int) -> None:
        clock.now += 0.025

    loop = RapTickLoop(Tempo(120.0, 4, 4), on_tick=on_tick, clock=clock, sleep=clock.sleep)
    loop.run(max_ticks=3)

    assert clock.sleeps == pytest.approx([0.1, 0.1])


def test_demo_session_start_records_resolved_repetition_window(tmp_path) -> None:
    class Controller:
        def start(self) -> None:
            return None

        def close(self) -> None:
            return None

    class TickLoop:
        def run(self, max_ticks: int | None = None) -> None:
            return None

        def stop(self) -> None:
            return None

    events = []
    publisher = RapEventPublisher("session")
    dispatcher = RapEventDispatcher(publisher.queue, sinks=(events.append,))
    dispatcher.start()
    dependencies = RapDemoDependencies(
        Tempo(120.0, 4, 4),
        Controller(),
        publisher,
        dispatcher,
        TickLoop(),
        tmp_path,
        repetition_window_bars=7,
    )
    dependencies.run(max_bars=1)

    assert events[0].payload["repetition_window_bars"] == 7


def test_demo_dependencies_reject_nonpositive_repetition_window(tmp_path) -> None:
    class Controller:
        def start(self) -> None:
            return None

        def close(self) -> None:
            return None

    class TickLoop:
        def run(self, max_ticks: int | None = None) -> None:
            return None

        def stop(self) -> None:
            return None

    publisher = RapEventPublisher("session")
    dispatcher = RapEventDispatcher(publisher.queue, sinks=())
    with pytest.raises(ValueError, match="repetition_window_bars"):
        RapDemoDependencies(Tempo(120.0, 4, 4), Controller(), publisher, dispatcher, TickLoop(), tmp_path, 0)
