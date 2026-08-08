"""Tests for deterministic flow context in lyric-generation prompts."""

from streammuse.application.rap.rhythm import flow_template_for_pattern
from streammuse.domain.rap import FlowProvenance, FlowSlot, FlowTemplate
from streammuse.domain.timing import Tempo
from streammuse.infrastructure.rap.flow_prompt import describe_flow, format_flow_for_prompt
from streammuse.infrastructure.rap.templates import BUILTIN_TEMPLATES


def test_syncopated_template_formats_exact_ticks_stress_boundary_and_rhyme() -> None:
    template = BUILTIN_TEMPLATES.get("baseline_syncopated_9")

    description = describe_flow(template)
    rendered = format_flow_for_prompt(template)

    assert description.ticks == (0, 2, 3, 5, 7, 8, 10, 13, 15)
    assert description.stresses == (1.0, 0.2, 0.7, 0.2, 0.6, 1.0, 0.2, 0.7, 0.9)
    assert description.notation == "S . w M | . w . M | S . w . | . M . S"
    assert "phrase boundary strength 3" in rendered
    assert "rhyme group A" in rendered


def test_equal_length_templates_with_different_timing_have_different_prompts() -> None:
    straight = BUILTIN_TEMPLATES.get("baseline_straight_9")
    syncopated = BUILTIN_TEMPLATES.get("baseline_syncopated_9")

    assert format_flow_for_prompt(straight) != format_flow_for_prompt(syncopated)


def test_legacy_stresses_are_serialized_without_precision_loss() -> None:
    template = flow_template_for_pattern(Tempo(bpm=92, ticks_per_beat=4, beats_per_bar=4), "boom_bap")

    rendered = format_flow_for_prompt(template)

    assert (
        "Target stress: [1.0, 0.15, 0.35, 0.15, 0.75, 0.15, 0.3, 0.15, "
        "0.9, 0.15, 0.4, 0.15, 0.75, 0.15, 0.3, 0.2]"
    ) in rendered


def test_flow_description_preserves_per_slot_boundaries_and_rhyme_roles() -> None:
    template = FlowTemplate(
        template_id="multi_phrase",
        name="Internal phrase and rhyme roles",
        ticks_per_beat=4,
        beats_per_bar=4,
        slots=(
            FlowSlot(0, 2, 1.0, boundary_strength=0, rhyme_group="A"),
            FlowSlot(4, 2, 0.35, boundary_strength=2, rhyme_group="B"),
            FlowSlot(8, 2, 0.75),
            FlowSlot(15, 1, 0.15, boundary_strength=4, rhyme_group="C"),
        ),
        provenance=FlowProvenance(kind="test", source="unit-test"),
    )

    description = describe_flow(template)
    rendered = format_flow_for_prompt(template)

    assert description.boundary_strengths == (0, 2, 0, 4)
    assert description.rhyme_groups == ("A", "B", None, "C")
    assert "Boundary strengths: [0, 2, 0, 4]" in rendered
    assert 'Rhyme groups: ["A", "B", null, "C"]' in rendered
    assert "Final slot: phrase boundary strength 4, rhyme group C" in rendered
