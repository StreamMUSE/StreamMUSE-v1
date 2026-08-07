"""Tests for deterministic, anonymous MCFlow sample selection."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from streammuse.domain.rap.flow import FlowProvenance, FlowSlot, FlowTemplate
from streammuse.infrastructure.rap.mcflow import (
    ExtractionRejection,
    ExtractionResult,
    load_extracted_templates,
)
from streammuse.infrastructure.rap.sample_catalog import (
    DENSITY_BANDS,
    structural_flow_signature,
    select_sample_templates,
    write_sample_catalog,
)
from streammuse.infrastructure.rap.templates import TemplateCatalog


def _template(
    ordinal: int,
    *,
    slots: int = 4,
    source_seed: str = "a",
    rhyme_labels: tuple[str | None, ...] | None = None,
    stress_offset: float = 0.0,
    duration_offset: int = 0,
    boundary_offset: int = 0,
    onset_offset: int = 0,
    error: float = 0.0,
) -> FlowTemplate:
    source_hash = hashlib.sha256(source_seed.encode("ascii")).hexdigest()
    template_id = "mcflow_" + hashlib.sha256(f"{source_hash}:{ordinal}".encode("ascii")).hexdigest()[:20]
    labels = rhyme_labels or tuple("A" if index == slots - 1 else None for index in range(slots))
    onsets = tuple(index * (16 // slots) for index in range(slots))
    if onset_offset:
        onsets = tuple((tick + onset_offset if index == 1 else tick) for index, tick in enumerate(onsets))
    return FlowTemplate(
        template_id=template_id,
        name=f"anonymous_measure_{ordinal}",
        ticks_per_beat=4,
        beats_per_bar=4,
        slots=tuple(
            FlowSlot(
                tick_in_bar=tick,
                duration_ticks=max(1, 16 // slots + (duration_offset if index == 0 else 0)),
                target_stress=0.5 + stress_offset if index == 0 else 0.0,
                boundary_strength=boundary_offset if index == slots - 1 else 0,
                rhyme_group=labels[index],
            )
            for index, tick in enumerate(onsets)
        ),
        provenance=FlowProvenance(
            kind="mcflow_extracted_anonymous",
            source="anonymous_mcflow",
            source_hash=source_hash,
            quantization_error_ticks=error,
        ),
    )


def test_structural_signature_ignores_anonymous_provenance_and_canonicalizes_rhymes() -> None:
    """Catches IDs, source metadata, or arbitrary rhyme labels affecting deduplication."""
    first = _template(1, source_seed="first", rhyme_labels=("A", None, "B", "A"), error=0.25)
    equivalent = _template(9, source_seed="second", rhyme_labels=("C", None, "D", "C"), error=0.0)

    assert structural_flow_signature(first) == structural_flow_signature(equivalent)

    selection = select_sample_templates((equivalent, first))

    assert selection.templates == (min(first, equivalent, key=lambda template: template.template_id),)
    assert selection.report.structurally_unique_templates == 1
    assert selection.report.duplicates_removed == 1


@pytest.mark.parametrize(
    "changed",
    (
        lambda: _template(2, onset_offset=1),
        lambda: _template(3, duration_offset=1),
        lambda: _template(4, stress_offset=0.1),
        lambda: _template(5, boundary_offset=1),
        lambda: _template(6, rhyme_labels=("A", None, "B", "B")),
        lambda: _template(7, slots=5),
    ),
)
def test_structural_signature_retains_each_flow_dimension(changed) -> None:
    """Catches deduplication that collapses meaningful timing or prosodic structure."""
    baseline = _template(1, rhyme_labels=("A", None, "B", "A"))

    assert structural_flow_signature(baseline) != structural_flow_signature(changed())


def test_selection_balances_bands_excludes_out_of_range_and_is_order_independent() -> None:
    """Catches input-order sampling or accidental inclusion outside the three density bands."""
    sparse = tuple(_template(index, source_seed=f"sparse-{index}", stress_offset=index / 100) for index in range(1, 6))
    medium = tuple(_template(index + 10, slots=8, source_seed=f"medium-{index}", stress_offset=index / 100) for index in range(1, 3))
    dense = tuple(_template(index + 20, slots=12, source_seed=f"dense-{index}", stress_offset=index / 100) for index in range(1, 3))
    selection = select_sample_templates((*sparse, *medium, *dense, _template(40, slots=3)), per_bucket=3)
    reordered = select_sample_templates(tuple(reversed((*sparse, *medium, *dense, _template(40, slots=3)))), per_bucket=3)

    assert DENSITY_BANDS == {"sparse": (4, 7), "medium": (8, 11), "dense": (12, 16)}
    assert [len(template.slots) for template in selection.templates] == [4, 4, 4, 8, 8, 12, 12]
    assert selection.templates == reordered.templates
    assert selection.report.out_of_range_templates == 1
    assert selection.report.band_counts["sparse"].available == 5
    assert selection.report.band_counts["sparse"].selected == 3
    assert selection.report.band_counts["sparse"].underfilled == 0
    assert selection.report.band_counts["medium"].underfilled == 1
    assert selection.report.band_counts["dense"].underfilled == 1
    sparse_signatures = sorted(structural_flow_signature(template) for template in sparse)
    selected_sparse = [structural_flow_signature(template) for template in selection.templates[:3]]
    assert selected_sparse == [sparse_signatures[0], sparse_signatures[2], sparse_signatures[-1]]


def test_selection_sorts_mixed_rhyme_topologies_deterministically() -> None:
    """Catches ordering that tries to compare an absent rhyme directly to a rhyme index."""
    unlabeled = _template(1, rhyme_labels=(None, None, None, None))
    labeled = _template(2, rhyme_labels=("A", None, None, None))

    first = select_sample_templates((labeled, unlabeled), per_bucket=2)
    second = select_sample_templates((unlabeled, labeled), per_bucket=2)

    assert first.templates == second.templates
    assert len(first.templates) == 2


@pytest.mark.parametrize("per_bucket", (0, -1, 1.5, True))
def test_selection_rejects_invalid_bucket_limits(per_bucket: object) -> None:
    """Catches ambiguous or nonsensical sampling limits."""
    with pytest.raises(ValueError, match="per_bucket"):
        select_sample_templates((_template(1),), per_bucket=per_bucket)  # type: ignore[arg-type]


def test_selection_rejects_nonanonymous_and_non_4x4_templates() -> None:
    """Catches hand-authored or malformed templates crossing the extractor boundary."""
    anonymous = _template(1)
    nonanonymous = replace(anonymous, name="not anonymous")
    malformed = object.__new__(FlowTemplate)
    object.__setattr__(malformed, "template_id", anonymous.template_id)
    object.__setattr__(malformed, "name", anonymous.name)
    object.__setattr__(malformed, "ticks_per_beat", 3)
    object.__setattr__(malformed, "beats_per_bar", 4)
    object.__setattr__(malformed, "slots", anonymous.slots)
    object.__setattr__(malformed, "provenance", anonymous.provenance)

    with pytest.raises(ValueError, match="anonymous"):
        select_sample_templates((nonanonymous,))
    with pytest.raises(ValueError, match="four ticks"):
        select_sample_templates((malformed,))


def test_write_sample_catalog_retains_extraction_metadata_and_emits_aggregate_only_report(tmp_path: Path) -> None:
    """Catches a sampled catalog that cannot reload or a report that exposes identities."""
    templates = (_template(1), _template(2, slots=8), _template(3, slots=12))
    extraction = ExtractionResult(
        templates=templates,
        rejections=(
            ExtractionRejection(
                source_hash=templates[0].provenance.source_hash or "",
                measure_ordinal=4,
                error_code="empty_measure",
                detail="measure has no lyric-bearing slots",
            ),
        ),
        parsed_files=2,
    )
    selection = select_sample_templates(templates, per_bucket=2)
    catalog_path = tmp_path / "nested" / "catalog.json"
    report_path = tmp_path / "nested" / "report.json"

    write_sample_catalog(extraction, selection, catalog_path, report_path)

    assert TemplateCatalog.from_templates(load_extracted_templates(catalog_path)).get(selection.templates[0].template_id)
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert catalog["aggregate"] == {"parsed_files": 2, "accepted_templates": 3, "rejected_measures": 1}
    assert report["schema_version"] == "streammuse.mcflow_sample.v1"
    assert report["density_ranges"] == {"dense": [12, 16], "medium": [8, 11], "sparse": [4, 7]}
    assert report["requested_per_bucket"] == 2
    assert all(forbidden not in json.dumps(report) for forbidden in ("source_hash", "template_id", "anonymous_measure", "lyric", "ipa", "artist", "title", "path"))


def test_write_sample_catalog_rejects_an_inconsistent_selection_report(tmp_path: Path) -> None:
    """Catches sidecars that claim counts inconsistent with their selected catalog."""
    template = _template(1)
    extraction = ExtractionResult(templates=(template,), rejections=(), parsed_files=1)
    selection = select_sample_templates((template,))

    with pytest.raises(ValueError, match="inconsistent"):
        write_sample_catalog(
            extraction,
            replace(selection, report=replace(selection.report, selected_templates=99)),
            tmp_path / "catalog.json",
            tmp_path / "report.json",
        )
