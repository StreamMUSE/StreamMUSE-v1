"""Tests for the standalone rap tick loop."""

import pytest

from streammuse.application.rap.runtime import RapTickLoop
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
