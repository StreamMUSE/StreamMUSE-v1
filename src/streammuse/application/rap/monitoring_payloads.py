"""Stable JSON-ready monitoring payload builders for rap planning."""

from __future__ import annotations

from typing import Any

from streammuse.domain.rap import FlowTemplate, ScheduledSyllable


def flow_template_payload(template: FlowTemplate) -> dict[str, Any]:
    """Return every flow alignment field used to make a planning decision."""

    return {
        "template_id": template.template_id,
        "name": template.name,
        "ticks_per_beat": template.ticks_per_beat,
        "beats_per_bar": template.beats_per_bar,
        "provenance": {
            "kind": template.provenance.kind,
            "source": template.provenance.source,
            "source_hash": template.provenance.source_hash,
            "quantization_error_ticks": template.provenance.quantization_error_ticks,
        },
        "slots": [
            {
                "slot_index": index,
                "tick_in_bar": slot.tick_in_bar,
                "duration_ticks": slot.duration_ticks,
                "target_stress": slot.target_stress,
                "boundary_strength": slot.boundary_strength,
                "rhyme_group": slot.rhyme_group,
            }
            for index, slot in enumerate(template.slots)
        ],
    }


def scheduled_syllables_payload(
    scheduled: tuple[ScheduledSyllable, ...], *, bar: int
) -> list[dict[str, Any]]:
    """Return scheduled lyric syllables relative to their owning bar."""

    return [
        {
            "slot_index": item.slot.slot_index,
            "tick_in_bar": item.slot.tick - (bar * 16),
            "target_stress": item.slot.accent,
            "label": item.syllable.label,
            "word": item.syllable.word,
            "stress": item.syllable.stress,
            "stressed": item.syllable.stressed,
        }
        for item in scheduled
    ]
