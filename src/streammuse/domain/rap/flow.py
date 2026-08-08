"""Immutable templates for beat-aligned rap flow."""

from __future__ import annotations

from dataclasses import dataclass

from streammuse.domain.rap.models import BeatSlot


@dataclass(frozen=True)
class FlowSlot:
    """One syllable target relative to the start of a bar."""

    tick_in_bar: int
    duration_ticks: int
    target_stress: float
    boundary_strength: int = 0
    rhyme_group: str | None = None


@dataclass(frozen=True)
class FlowProvenance:
    """Provenance of a flow template used in a reproducible run."""

    kind: str
    source: str
    source_hash: str | None = None
    quantization_error_ticks: float = 0.0


@dataclass(frozen=True)
class FlowTemplate:
    """Validated flow targets for one bar."""

    template_id: str
    name: str
    ticks_per_beat: int
    beats_per_bar: int
    slots: tuple[FlowSlot, ...]
    provenance: FlowProvenance

    def __post_init__(self) -> None:
        if not isinstance(self.slots, tuple) or not all(isinstance(slot, FlowSlot) for slot in self.slots):
            raise ValueError("flow template slots must be a tuple of FlowSlot values")
        if not isinstance(self.provenance, FlowProvenance):
            raise ValueError("flow template provenance must be a FlowProvenance")
        if self.ticks_per_beat != 4 or self.beats_per_bar != 4:
            raise ValueError("flow templates require four ticks per beat and four beats per bar")
        ticks_per_bar = self.ticks_per_beat * self.beats_per_bar
        ticks = [slot.tick_in_bar for slot in self.slots]
        if not self.template_id or not self.slots:
            raise ValueError("flow template requires an id and at least one slot")
        if ticks != sorted(set(ticks)):
            raise ValueError("flow slots must have unique increasing onsets")
        if any(tick < 0 or tick >= ticks_per_bar for tick in ticks):
            raise ValueError("flow slot onset lies outside the bar")
        if any(slot.duration_ticks <= 0 for slot in self.slots):
            raise ValueError("flow slot duration must be positive")
        if any(not 0.0 <= slot.target_stress <= 1.0 for slot in self.slots):
            raise ValueError("target stress must be between zero and one")
        if any(not 0 <= slot.boundary_strength <= 5 for slot in self.slots):
            raise ValueError("boundary strength must be between zero and five")


def materialize_flow(template: FlowTemplate, bar: int) -> tuple[BeatSlot, ...]:
    """Project a relative flow template into absolute beat slots for one bar."""
    if bar < 0:
        raise ValueError("bar must not be negative")

    ticks_per_bar = template.ticks_per_beat * template.beats_per_bar
    first_tick = bar * ticks_per_bar
    return tuple(
        BeatSlot(
            bar=bar,
            tick=first_tick + slot.tick_in_bar,
            beat=slot.tick_in_bar // template.ticks_per_beat,
            tick_in_beat=slot.tick_in_bar % template.ticks_per_beat,
            accent=slot.target_stress,
            duration_ticks=slot.duration_ticks,
            boundary_strength=slot.boundary_strength,
            rhyme_group=slot.rhyme_group,
            template_id=template.template_id,
            slot_index=index,
        )
        for index, slot in enumerate(template.slots)
    )
