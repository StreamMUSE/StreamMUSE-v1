"""Deterministic density-balanced selection of anonymous MCFlow templates."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

from streammuse.domain.rap.flow import FlowTemplate
from streammuse.infrastructure.rap.mcflow import ExtractionResult, flow_template_to_dict, write_extracted_templates


SAMPLE_CATALOG_SCHEMA_VERSION = "streammuse.mcflow_sample.v1"
DENSITY_BANDS = {"sparse": (4, 7), "medium": (8, 11), "dense": (12, 16)}

StructuralFlowSignature = tuple[tuple[int, int, float, int, int | None], ...]


@dataclass(frozen=True)
class DensityBandCount:
    """Aggregate availability and selection counts for one density band."""

    available: int
    selected: int
    underfilled: int

    def to_dict(self) -> dict[str, int]:
        return {"available": self.available, "selected": self.selected, "underfilled": self.underfilled}


@dataclass(frozen=True)
class SampleCatalogReport:
    """Aggregate-only accounting for one deterministic sample selection."""

    input_templates: int
    structurally_unique_templates: int
    duplicates_removed: int
    out_of_range_templates: int
    selected_templates: int
    anonymous_source_file_count: int
    requested_per_bucket: int
    band_counts: Mapping[str, DensityBandCount] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "band_counts", MappingProxyType(dict(self.band_counts)))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": SAMPLE_CATALOG_SCHEMA_VERSION,
            "density_ranges": {name: list(slot_range) for name, slot_range in DENSITY_BANDS.items()},
            "requested_per_bucket": self.requested_per_bucket,
            "input_templates": self.input_templates,
            "structurally_unique_templates": self.structurally_unique_templates,
            "duplicates_removed": self.duplicates_removed,
            "out_of_range_templates": self.out_of_range_templates,
            "selected_templates": self.selected_templates,
            "anonymous_source_file_count": self.anonymous_source_file_count,
            "bands": {name: self.band_counts[name].to_dict() for name in DENSITY_BANDS},
        }


@dataclass(frozen=True)
class SampleCatalogSelection:
    """Immutable templates selected for a local catalog and their aggregate report."""

    templates: tuple[FlowTemplate, ...]
    report: SampleCatalogReport


def structural_flow_signature(template: FlowTemplate) -> StructuralFlowSignature:
    """Return a metadata-free signature with canonical first-occurrence rhymes."""
    _validate_extracted_template(template)
    rhyme_indices: dict[str, int] = {}
    slots: list[tuple[int, int, float, int, int | None]] = []
    for slot in template.slots:
        rhyme_index: int | None = None
        if slot.rhyme_group is not None:
            rhyme_index = rhyme_indices.setdefault(slot.rhyme_group, len(rhyme_indices))
        slots.append(
            (
                slot.tick_in_bar,
                slot.duration_ticks,
                slot.target_stress,
                slot.boundary_strength,
                rhyme_index,
            )
        )
    return tuple(slots)


def select_sample_templates(
    templates: Iterable[FlowTemplate], *, per_bucket: int = 10
) -> SampleCatalogSelection:
    """Deduplicate and select deterministic representative structures by density."""
    if isinstance(per_bucket, bool) or not isinstance(per_bucket, int) or per_bucket < 1:
        raise ValueError("per_bucket must be a positive integer")

    input_templates = tuple(templates)
    representatives: dict[StructuralFlowSignature, FlowTemplate] = {}
    source_hashes: set[str] = set()
    for template in input_templates:
        signature = structural_flow_signature(template)
        source_hash = template.provenance.source_hash
        if source_hash is None:
            raise ValueError("anonymous templates require a source hash")
        source_hashes.add(source_hash)
        existing = representatives.get(signature)
        if existing is None or template.template_id < existing.template_id:
            representatives[signature] = template

    grouped: dict[str, list[tuple[StructuralFlowSignature, FlowTemplate]]] = {
        name: [] for name in DENSITY_BANDS
    }
    out_of_range = 0
    for signature, template in sorted(representatives.items(), key=lambda item: _signature_sort_key(item[0])):
        band = _density_band(len(template.slots))
        if band is None:
            out_of_range += 1
        else:
            grouped[band].append((signature, template))

    selected: list[FlowTemplate] = []
    counts: dict[str, DensityBandCount] = {}
    for name in DENSITY_BANDS:
        available = grouped[name]
        chosen = _evenly_spaced(available, per_bucket)
        selected.extend(template for _, template in chosen)
        counts[name] = DensityBandCount(
            available=len(available), selected=len(chosen), underfilled=max(0, per_bucket - len(chosen))
        )

    report = SampleCatalogReport(
        input_templates=len(input_templates),
        structurally_unique_templates=len(representatives),
        duplicates_removed=len(input_templates) - len(representatives),
        out_of_range_templates=out_of_range,
        selected_templates=len(selected),
        anonymous_source_file_count=len(source_hashes),
        requested_per_bucket=per_bucket,
        band_counts=counts,
    )
    return SampleCatalogSelection(templates=tuple(selected), report=report)


def write_sample_catalog(
    extraction: ExtractionResult,
    selection: SampleCatalogSelection,
    catalog_path: str | Path,
    report_path: str | Path,
) -> None:
    """Persist a standard extracted catalog and a validated aggregate-only sidecar."""
    catalog_destination = Path(catalog_path)
    report_destination = Path(report_path)
    if catalog_destination.resolve() == report_destination.resolve():
        raise ValueError("catalog and report destinations must differ")
    _validate_write_contract(extraction, selection)
    catalog_destination.parent.mkdir(parents=True, exist_ok=True)
    report_destination.parent.mkdir(parents=True, exist_ok=True)
    sampled_extraction = ExtractionResult(
        templates=selection.templates,
        rejections=extraction.rejections,
        parsed_files=extraction.parsed_files,
    )
    write_extracted_templates(sampled_extraction, catalog_destination)
    report_destination.write_text(json.dumps(selection.report.to_dict(), sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _validate_extracted_template(template: FlowTemplate) -> None:
    if not isinstance(template, FlowTemplate):
        raise ValueError("sample catalog requires FlowTemplate values")
    if template.ticks_per_beat != 4 or template.beats_per_bar != 4:
        raise ValueError("flow templates require four ticks per beat and four beats per bar")
    flow_template_to_dict(template)


def _density_band(slot_count: int) -> str | None:
    for name, (minimum, maximum) in DENSITY_BANDS.items():
        if minimum <= slot_count <= maximum:
            return name
    return None


def _signature_sort_key(signature: StructuralFlowSignature) -> tuple[tuple[int, int, float, int, int], ...]:
    return tuple(
        (onset, duration, stress, boundary, -1 if rhyme_index is None else rhyme_index)
        for onset, duration, stress, boundary, rhyme_index in signature
    )


def _evenly_spaced(
    values: list[tuple[StructuralFlowSignature, FlowTemplate]], limit: int
) -> list[tuple[StructuralFlowSignature, FlowTemplate]]:
    if len(values) <= limit:
        return values
    if limit == 1:
        return [values[0]]
    denominator = limit - 1
    return [values[(index * (len(values) - 1) + denominator // 2) // denominator] for index in range(limit)]


def _validate_write_contract(extraction: ExtractionResult, selection: SampleCatalogSelection) -> None:
    try:
        expected = select_sample_templates(
            extraction.templates, per_bucket=selection.report.requested_per_bucket
        )
    except (AttributeError, ValueError) as exc:
        raise ValueError("inconsistent sample catalog selection") from exc
    if selection.templates != expected.templates or selection.report != expected.report:
        raise ValueError("inconsistent sample catalog selection")
