"""Tests for immutable beat-aligned flow templates."""

import pytest

from streammuse.domain.rap import FlowProvenance, FlowSlot, FlowTemplate, materialize_flow


def test_materialize_flow_preserves_relative_structure_at_absolute_bar() -> None:
    template = FlowTemplate(
        template_id="test_syncopated",
        name="Test syncopated",
        ticks_per_beat=4,
        beats_per_bar=4,
        provenance=FlowProvenance(kind="hand_authored_test", source="unit-test"),
        slots=(
            FlowSlot(tick_in_bar=0, duration_ticks=2, target_stress=1.0),
            FlowSlot(tick_in_bar=3, duration_ticks=1, target_stress=0.2),
            FlowSlot(tick_in_bar=8, duration_ticks=2, target_stress=0.9, rhyme_group="A"),
        ),
    )

    slots = materialize_flow(template, bar=2)

    assert [slot.tick for slot in slots] == [32, 35, 40]
    assert [slot.slot_index for slot in slots] == [0, 1, 2]
    assert slots[-1].rhyme_group == "A"


@pytest.mark.parametrize(
    ("slots", "message"),
    (
        ((), "flow template requires an id and at least one slot"),
        (
            (
                FlowSlot(tick_in_bar=3, duration_ticks=1, target_stress=0.5),
                FlowSlot(tick_in_bar=3, duration_ticks=1, target_stress=0.5),
            ),
            "flow slots must have unique increasing onsets",
        ),
        ((FlowSlot(tick_in_bar=16, duration_ticks=1, target_stress=0.5),), "flow slot onset lies outside the bar"),
        ((FlowSlot(tick_in_bar=0, duration_ticks=0, target_stress=0.5),), "flow slot duration must be positive"),
        ((FlowSlot(tick_in_bar=0, duration_ticks=1, target_stress=1.1),), "target stress must be between zero and one"),
        ((FlowSlot(tick_in_bar=0, duration_ticks=1, target_stress=0.5, boundary_strength=6),), "boundary strength must be between zero and five"),
    ),
)
def test_flow_template_rejects_invalid_slots(slots: tuple[FlowSlot, ...], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        FlowTemplate(
            template_id="invalid",
            name="Invalid",
            ticks_per_beat=4,
            beats_per_bar=4,
            slots=slots,
            provenance=FlowProvenance(kind="test", source="unit-test"),
        )


def test_materialize_flow_rejects_negative_bar() -> None:
    template = FlowTemplate(
        template_id="one_slot",
        name="One slot",
        ticks_per_beat=4,
        beats_per_bar=4,
        slots=(FlowSlot(tick_in_bar=0, duration_ticks=1, target_stress=1.0),),
        provenance=FlowProvenance(kind="test", source="unit-test"),
    )

    with pytest.raises(ValueError, match="bar must not be negative"):
        materialize_flow(template, bar=-1)


@pytest.mark.parametrize(
    ("ticks_per_beat", "beats_per_bar"),
    ((3, 4), (4, 3)),
)
def test_flow_template_requires_four_ticks_per_beat_and_four_beats_per_bar(
    ticks_per_beat: int, beats_per_bar: int
) -> None:
    with pytest.raises(ValueError, match="flow templates require four ticks per beat and four beats per bar"):
        FlowTemplate(
            template_id="wrong_timing",
            name="Wrong timing",
            ticks_per_beat=ticks_per_beat,
            beats_per_bar=beats_per_bar,
            slots=(FlowSlot(tick_in_bar=0, duration_ticks=1, target_stress=1.0),),
            provenance=FlowProvenance(kind="test", source="unit-test"),
        )
