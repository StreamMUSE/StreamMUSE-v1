"""Built-in hand-authored rap flow templates."""

from __future__ import annotations

from collections.abc import Iterable

from streammuse.domain.rap.flow import FlowProvenance, FlowSlot, FlowTemplate


class TemplateCatalog:
    """Lookup table for validated immutable flow templates."""

    def __init__(self, templates: Iterable[FlowTemplate]) -> None:
        by_id = {template.template_id: template for template in templates}
        if not by_id:
            raise ValueError("template catalog must not be empty")
        self._by_id = by_id

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


BUILTIN_TEMPLATES = TemplateCatalog.from_templates(
    (
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
)
