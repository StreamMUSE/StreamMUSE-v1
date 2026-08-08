"""Rhythm presets expressed in StreamMUSE tick coordinates."""

from __future__ import annotations

from streammuse.domain.rap import BeatSlot, FlowProvenance, FlowSlot, FlowTemplate
from streammuse.domain.timing import Tempo


_PATTERN_ACCENTS: dict[str, tuple[float, ...]] = {
    "boom_bap": (
        1.00,
        0.15,
        0.35,
        0.15,
        0.75,
        0.15,
        0.30,
        0.15,
        0.90,
        0.15,
        0.40,
        0.15,
        0.75,
        0.15,
        0.30,
        0.20,
    ),
    "straight_8": (
        1.00,
        0.15,
        0.65,
        0.15,
        0.85,
        0.15,
        0.65,
        0.15,
        0.95,
        0.15,
        0.65,
        0.15,
        0.85,
        0.15,
        0.65,
        0.15,
    ),
    "trap_sparse": (
        1.00,
        0.10,
        0.25,
        0.45,
        0.55,
        0.10,
        0.20,
        0.15,
        0.95,
        0.10,
        0.25,
        0.55,
        0.60,
        0.10,
        0.30,
        0.20,
    ),
}

# The active CLI's historical rhythm names select an explicit research flow.
# Keeping this mapping public makes that compatibility decision inspectable.
LEGACY_PATTERN_TEMPLATE_IDS: dict[str, str] = {
    "boom_bap": "baseline_syncopated_9",
    "straight_8": "baseline_straight_9",
    "trap_sparse": "baseline_staggered_9",
}


def available_patterns() -> tuple[str, ...]:
    """Return the stable presentation order of supported rhythm presets."""
    return tuple(_PATTERN_ACCENTS)


def build_bar_slots(tempo: Tempo, pattern: str, bar: int) -> tuple[BeatSlot, ...]:
    """Create the lyric-bearing slots for one 4/4 bar."""
    if pattern not in _PATTERN_ACCENTS:
        raise ValueError(f"unsupported rap pattern: {pattern}")
    if bar < 0:
        raise ValueError("bar must not be negative")

    accents = _PATTERN_ACCENTS[pattern]
    if tempo.ticks_per_bar != len(accents):
        raise ValueError("rap patterns require exactly 16 ticks per bar")

    first_tick = bar * tempo.ticks_per_bar
    return tuple(
        BeatSlot(
            bar=bar,
            tick=first_tick + tick_in_bar,
            beat=tick_in_bar // tempo.ticks_per_beat,
            tick_in_beat=tick_in_bar % tempo.ticks_per_beat,
            accent=accent,
        )
        for tick_in_bar, accent in enumerate(accents)
    )


def flow_template_for_pattern(tempo: Tempo, pattern: str) -> FlowTemplate:
    """Adapt a legacy named rhythm preset into an authoritative flow template."""
    slots = build_bar_slots(tempo, pattern, bar=0)
    final = len(slots) - 1
    return FlowTemplate(
        template_id=f"legacy_{pattern}",
        name=f"Legacy {pattern} rhythm preset",
        ticks_per_beat=tempo.ticks_per_beat,
        beats_per_bar=tempo.beats_per_bar,
        slots=tuple(
            FlowSlot(
                tick_in_bar=slot.tick,
                duration_ticks=1,
                target_stress=slot.accent,
                boundary_strength=3 if index == final else 0,
                rhyme_group="A" if index == final else None,
            )
            for index, slot in enumerate(slots)
        ),
        provenance=FlowProvenance(kind="legacy_rhythm_preset", source=pattern),
    )
