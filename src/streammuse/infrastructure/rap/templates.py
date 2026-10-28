"""Curated MCFlow choices with lookup-only legacy flow compatibility."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path

from streammuse.domain.rap.flow import FlowProvenance, FlowSlot, FlowTemplate
from streammuse.infrastructure.rap.mcflow import load_extracted_templates


class TemplateCatalog:
    """Lookup table for validated immutable flow templates."""

    def __init__(
        self, templates: Iterable[FlowTemplate], *, legacy_templates: Iterable[FlowTemplate] = (),
    ) -> None:
        by_id = {template.template_id: template for template in templates}
        if not by_id:
            raise ValueError("template catalog must not be empty")
        self._public_templates = tuple(by_id.values())
        self._by_id = {template.template_id: template for template in legacy_templates}
        self._by_id.update(by_id)

    @classmethod
    def from_templates(cls, templates: Iterable[FlowTemplate]) -> "TemplateCatalog":
        """Build a catalog from the supplied templates."""
        return cls(templates)

    def get(self, template_id: str) -> FlowTemplate:
        """Return a template or report a stable configuration error."""
        try:
            return self._by_id[template_id]
        except KeyError as exc:
            raise ValueError(f"unknown flow template: {template_id}") from exc

    def values(self) -> tuple[FlowTemplate, ...]:
        """Return selectable templates in catalog order, excluding lookup-only legacy IDs."""
        return self._public_templates


def _slots(ticks: tuple[int, ...], *, stresses: tuple[float, ...]) -> tuple[FlowSlot, ...]:
    if len(ticks) != len(stresses):
        raise ValueError("tick and stress arrays must have equal length")
    last = len(ticks) - 1
    return tuple(
        FlowSlot(
            tick_in_bar=tick,
            duration_ticks=max(1, (ticks[index + 1] - tick) if index < last else 16 - tick),
            target_stress=stresses[index],
            boundary_strength=3 if index == last else 0,
            rhyme_group="A" if index == last else None,
        )
        for index, tick in enumerate(ticks)
    )


_BASELINE_TEMPLATES = (
        FlowTemplate(
            template_id="baseline_straight_9",
            name="Straight nine-slot baseline",
            ticks_per_beat=4,
            beats_per_bar=4,
            slots=_slots((0, 2, 4, 6, 8, 10, 12, 14, 15), stresses=(1, 0, 0.7, 0, 1, 0, 0.7, 0, 0.9)),
            provenance=FlowProvenance(kind="hand_authored_mcflow_inspired", source="StreamMUSE baseline"),
        ),
        FlowTemplate(
            template_id="baseline_syncopated_9",
            name="Syncopated nine-slot baseline",
            ticks_per_beat=4,
            beats_per_bar=4,
            slots=_slots((0, 2, 3, 5, 7, 8, 10, 13, 15), stresses=(1, 0.2, 0.7, 0.2, 0.6, 1, 0.2, 0.7, 0.9)),
            provenance=FlowProvenance(kind="hand_authored_mcflow_inspired", source="StreamMUSE baseline"),
        ),
        FlowTemplate(
            template_id="baseline_staggered_9",
            name="Staggered nine-slot baseline",
            ticks_per_beat=4,
            beats_per_bar=4,
            slots=_slots((0, 1, 4, 6, 7, 9, 11, 12, 14), stresses=(1, 0.2, 0.8, 0.2, 0.6, 0.9, 0.2, 0.7, 0.9)),
            provenance=FlowProvenance(kind="hand_authored_mcflow_inspired", source="StreamMUSE baseline"),
        ),
)

_STRESS_VARIANTS = (
    ("straight", "backbeat", (0.25, 0.15, 1, 0.25, 0.5, 0.15, 1, 0.25, 0.5)),
    ("straight", "rising", (0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 1)),
    ("syncopated", "backbeat", (0.25, 0.15, 1, 0.25, 0.5, 0.15, 0.5, 1, 0.25)),
    ("syncopated", "rising", (0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 1)),
    ("staggered", "backbeat", (0.25, 0.15, 1, 0.25, 0.5, 0.15, 0.5, 1, 0.25)),
    ("staggered", "rising", (0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 1)),
)


def _stress_variants() -> tuple[FlowTemplate, ...]:
    baselines = {template.template_id: template for template in _BASELINE_TEMPLATES}
    return tuple(
        FlowTemplate(
            template_id=f"untested_{grid}_{accent}_9",
            name=f"UNTESTED {grid.title()} {accent} stress",
            ticks_per_beat=4,
            beats_per_bar=4,
            slots=_slots(
                tuple(slot.tick_in_bar for slot in baselines[f"baseline_{grid}_9"].slots),
                stresses=stresses,
            ),
            provenance=FlowProvenance(
                kind="hand_authored_stress_variant",
                source=f"UNTESTED handcrafted stress variant of baseline_{grid}_9; no listening validation",
            ),
        )
        for grid, accent, stresses in _STRESS_VARIANTS
    )


_CURATED_NAMES = {
    "mcflow_4b52a40dac67813b8d53": "Basketball m01 - Syncopated 9",
    "mcflow_0a782d792d272ce35c37": "Basketball m03 - Delayed 10",
    "mcflow_4fa0466611a111afb52f": "Basketball m20 - Delayed 8",
    "mcflow_5db8ad408d09359b078d": "Boom Boom Pow m17 - Long-short 8",
    "mcflow_778c4db0b4f291db7620": "Boom Boom Pow m35 - Dense 12",
    "mcflow_522b72072b7907f3288f": "U Can't Touch This m07 - Clustered 11",
}


def _curated_templates() -> tuple[FlowTemplate, ...]:
    templates = load_extracted_templates(Path(__file__).parent / "data" / "curated_mcflow_templates.json")
    return tuple(replace(template, name=_CURATED_NAMES[template.template_id]) for template in templates)


BUILTIN_TEMPLATES = TemplateCatalog(
    _curated_templates(), legacy_templates=(*_BASELINE_TEMPLATES, *_stress_variants()),
)
