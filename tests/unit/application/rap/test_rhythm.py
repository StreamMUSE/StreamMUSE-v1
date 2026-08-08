"""Tests for accent-weighted rap rhythm slots."""

import pytest

from streammuse.application.rap.rhythm import available_patterns, build_bar_slots, flow_template_for_pattern
from streammuse.domain.timing import Tempo


def test_boom_bap_slots_share_streammuse_tick_coordinates() -> None:
    slots = build_bar_slots(Tempo(bpm=92, ticks_per_beat=4, beats_per_bar=4), "boom_bap", bar=2)

    assert len(slots) == 16
    assert slots[0].tick == 32
    assert slots[0].bar == 2
    assert slots[0].beat == 0
    assert slots[4].beat == 1
    assert slots[0].accent > slots[1].accent


def test_available_patterns_exposes_supported_presets() -> None:
    assert available_patterns() == ("boom_bap", "straight_8", "trap_sparse")


def test_boom_bap_adapts_to_a_validated_sixteen_slot_flow_template() -> None:
    template = flow_template_for_pattern(Tempo(bpm=92, ticks_per_beat=4, beats_per_bar=4), "boom_bap")

    assert template.template_id == "legacy_boom_bap"
    assert len(template.slots) == 16
    assert template.slots[-1].boundary_strength == 3
    assert template.slots[-1].rhyme_group == "A"


def test_build_bar_slots_rejects_non_sixteenth_bar_resolution() -> None:
    with pytest.raises(ValueError, match="exactly 16 ticks per bar"):
        build_bar_slots(Tempo(bpm=92, ticks_per_beat=2, beats_per_bar=4), "boom_bap", bar=0)


def test_build_bar_slots_rejects_unknown_pattern() -> None:
    with pytest.raises(ValueError, match="unsupported rap pattern"):
        build_bar_slots(Tempo(bpm=92, ticks_per_beat=4, beats_per_bar=4), "unknown", bar=0)
