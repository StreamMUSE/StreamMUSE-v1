"""Tests for deterministic flow context in lyric-generation prompts."""

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
