from __future__ import annotations

import pytest

from streammuse.domain.timing import Tempo
from streammuse.presentation.rap_demo.cli import _install_scripted_edits, _parse_scripted_edits


def test_scripted_edits_parse_notice_and_fields() -> None:
    assert _parse_scripted_edits("12:topic=deep sea;20@1:template=t9,topic=ocean") == [
        (12, 2, "deep sea", None),
        (20, 1, "ocean", "t9"),
    ]
    with pytest.raises(ValueError):
        _parse_scripted_edits("12@0:topic=x")
    with pytest.raises(ValueError):
        _parse_scripted_edits("12:mood=x")


def test_scripted_edits_apply_mid_bar_notice_bars_ahead() -> None:
    class _Controller:
        def __init__(self) -> None:
            self.edits = []

        def on_tick(self, tick: int) -> None:
            pass

        def update_plan(self, **kwargs) -> None:
            self.edits.append((self.tick, kwargs["from_bar"]))

    controller = _Controller()
    tempo = Tempo(90.0, 4, 4)
    _install_scripted_edits(controller, "6:topic=a;5@1:topic=b", tempo)
    for tick in range(0, 7 * tempo.ticks_per_bar):
        controller.tick = tick
        controller.on_tick(tick)

    half = tempo.ticks_per_bar // 2
    assert controller.edits == [(4 * tempo.ticks_per_bar + half, 6), (4 * tempo.ticks_per_bar + half, 5)]
