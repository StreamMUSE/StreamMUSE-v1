"""Pure prompt serialization for validated rap flow templates."""

from __future__ import annotations

import json
from dataclasses import dataclass

from streammuse.domain.rap import FlowTemplate


@dataclass(frozen=True)
class FlowPromptDescription:
    """Exact and compact prompt-ready flow details."""

    ticks: tuple[int, ...]
    stresses: tuple[float, ...]
    boundary_strengths: tuple[int, ...]
    rhyme_groups: tuple[str | None, ...]
    notation: str
    final_boundary_strength: int
    final_rhyme_group: str | None


def describe_flow(template: FlowTemplate) -> FlowPromptDescription:
    """Describe exact slot values and compact beat notation for a template."""
    ticks = tuple(slot.tick_in_bar for slot in template.slots)
    stresses = tuple(slot.target_stress for slot in template.slots)
    boundary_strengths = tuple(slot.boundary_strength for slot in template.slots)
    rhyme_groups = tuple(slot.rhyme_group for slot in template.slots)
    grid = ["."] * (template.ticks_per_beat * template.beats_per_bar)
    for slot in template.slots:
        grid[slot.tick_in_bar] = "S" if slot.target_stress >= 0.85 else "M" if slot.target_stress >= 0.5 else "w"
    beats = (
        " ".join(grid[start : start + template.ticks_per_beat])
        for start in range(0, len(grid), template.ticks_per_beat)
    )
    final = template.slots[-1]
    return FlowPromptDescription(
        ticks=ticks,
        stresses=stresses,
        boundary_strengths=boundary_strengths,
        rhyme_groups=rhyme_groups,
        notation=" | ".join(beats),
        final_boundary_strength=final.boundary_strength,
        final_rhyme_group=final.rhyme_group,
    )


def format_flow_for_prompt(template: FlowTemplate) -> str:
    """Serialize exact flow values and compact notation for model context."""
    flow = describe_flow(template)
    ticks = ", ".join(str(value) for value in flow.ticks)
    stresses = ", ".join(repr(float(value)) for value in flow.stresses)
    boundary_strengths = ", ".join(str(value) for value in flow.boundary_strengths)
    rhyme_groups = json.dumps(flow.rhyme_groups)
    rhyme = flow.final_rhyme_group or "none"
    return (
        f"Flow template: {template.template_id}\n"
        f"Syllable ticks: [{ticks}]\n"
        f"Target stress: [{stresses}]\n"
        f"Boundary strengths: [{boundary_strengths}]\n"
        f"Rhyme groups: {rhyme_groups}\n"
        f"Pattern: {flow.notation}\n"
        f"Final slot: phrase boundary strength {flow.final_boundary_strength}, rhyme group {rhyme}"
    )
