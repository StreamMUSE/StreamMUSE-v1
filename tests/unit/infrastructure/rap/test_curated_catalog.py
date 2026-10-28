"""Curated MCFlow choices, source fidelity, and default fallback coverage."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from streammuse.domain.rap import CandidateRequest
from streammuse.domain.rap.flow import materialize_flow
from streammuse.infrastructure.rap import templates as template_module
from streammuse.infrastructure.rap.fallback import PrevalidatedFallbackCatalog
from streammuse.infrastructure.rap.mcflow import load_extracted_templates
from streammuse.infrastructure.rap.prosody import CmuProsodyAnalyzer, HeuristicProsodyAnalyzer
from streammuse.infrastructure.rap.sample_catalog import structural_flow_signature
from streammuse.infrastructure.rap.scenarios import default_scenario, load_scenario
from streammuse.infrastructure.rap.templates import BUILTIN_TEMPLATES, TemplateCatalog


# IDs, onsets, and canonical-object hashes recorded from the supplied manifests,
# independently of the packaged catalog. Hashes cover all source slot/provenance fields.
SOURCE_PATTERNS = (
    ("mcflow_4b52a40dac67813b8d53", (0, 2, 3, 5, 7, 8, 10, 11, 15),
     "8aac83732cfdaed7f947b1ab94116d96318a1495378b1c03e529046fb967fff7"),
    ("mcflow_0a782d792d272ce35c37", (1, 3, 4, 5, 7, 8, 10, 11, 14, 15),
     "0b26a2a6728559865eba7e4da1ef44927db81bc5793f06b44d66f5dab949cea1"),
    ("mcflow_4fa0466611a111afb52f", (2, 4, 5, 7, 8, 9, 11, 14),
     "f09b10eb8ce1706f1f6308f48c28d111e3cdc8551ceb4732e8abd183fcf5f4d1"),
    ("mcflow_5db8ad408d09359b078d", (0, 3, 6, 10, 11, 12, 13, 14),
     "1f9b0ab0da929f424a5cc08f503cbca15b344357ea84e966eaef40a756fa4dc6"),
    ("mcflow_778c4db0b4f291db7620", (0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14),
     "ba5acced4887b3914e63383accb0eef14f4949274af06cd93cd1b3394893a4d6"),
    ("mcflow_522b72072b7907f3288f", (0, 1, 2, 4, 6, 8, 9, 10, 12, 13, 14),
     "add3d7b04db7240674f3e2f3a1b7417da9a2726f072c52db18eccc0f58182ad6"),
)
CURATED_IDS = tuple(row[0] for row in SOURCE_PATTERNS)
CATALOG_PATH = Path(template_module.__file__).parent / "data" / "curated_mcflow_templates.json"


def test_public_catalog_enumerates_only_the_six_curated_choices_in_order() -> None:
    assert callable(getattr(BUILTIN_TEMPLATES, "values", None)), "catalog needs public enumeration"
    choices = BUILTIN_TEMPLATES.values()

    assert isinstance(choices, tuple)
    assert tuple(template.template_id for template in choices) == CURATED_IDS
    assert all(BUILTIN_TEMPLATES.get(template.template_id) is template for template in choices)
    assert all(template.provenance.kind == "mcflow_extracted_anonymous" for template in choices)
    assert all(not template.name.startswith("anonymous_measure_") for template in choices)


def test_custom_catalog_enumeration_uses_its_own_templates() -> None:
    baseline = BUILTIN_TEMPLATES.get("baseline_straight_9")
    catalog = TemplateCatalog.from_templates(iter((baseline,)))

    assert callable(getattr(catalog, "values", None)), "custom catalogs need the same public API"
    assert catalog.values() == (baseline,)
    assert catalog.values() == (baseline,)  # enumeration must not exhaust a generator


@pytest.mark.parametrize("template_id,onsets,source_digest", SOURCE_PATTERNS)
def test_curated_data_preserves_exact_source_structure_and_provenance(
    template_id: str, onsets: tuple[int, ...], source_digest: str,
) -> None:
    assert CATALOG_PATH.is_file(), "curated structure must ship independently of output/ manifests"
    payload = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    raw = next(item for item in payload["templates"] if item["template_id"] == template_id)
    digest = hashlib.sha256(json.dumps(raw, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert digest == source_digest

    # The standard loader enforces a structural-only whitelist (no lyrics or IPA).
    extracted = next(item for item in load_extracted_templates(CATALOG_PATH) if item.template_id == template_id)
    public = BUILTIN_TEMPLATES.get(template_id)
    assert replace(public, name=extracted.name) == extracted
    assert tuple(slot.tick_in_bar for slot in public.slots) == onsets
    assert tuple(slot.tick for slot in materialize_flow(public, 3)) == tuple(48 + tick for tick in onsets)


def test_curated_choices_are_diverse_useful_bars_with_two_delayed_entrances() -> None:
    assert CATALOG_PATH.is_file()
    templates = load_extracted_templates(CATALOG_PATH)

    assert len(templates) == len({structural_flow_signature(template) for template in templates}) == 6
    assert len({tuple(slot.tick_in_bar for slot in template.slots) for template in templates}) == 6
    assert all(8 <= len(template.slots) <= 12 for template in templates)
    assert sorted(template.slots[0].tick_in_bar for template in templates if template.slots[0].tick_in_bar) == [1, 2]


def test_default_scenario_uses_all_curated_choices_and_keeps_the_topic_loop() -> None:
    scenario = default_scenario()

    assert tuple(segment.template_id for segment in scenario.segments) == CURATED_IDS
    assert [scenario.segment_for_bar(bar).topic for bar in (0, 4, 8, 12)] == ["space", "deep sea", "code", "space"]
    assert scenario.total_bars == 12
    assert scenario.loop is True


@pytest.mark.parametrize("analyzer_type", (CmuProsodyAnalyzer, HeuristicProsodyAnalyzer))
def test_default_fallbacks_cover_every_curated_flow_even_after_a_topic_edit(analyzer_type) -> None:
    scenario = default_scenario()
    assert set(CURATED_IDS) == {segment.template_id for segment in scenario.segments}
    catalog = PrevalidatedFallbackCatalog.build(scenario, BUILTIN_TEMPLATES, analyzer_type())

    for segment in scenario.segments:
        template = BUILTIN_TEMPLATES.get(segment.template_id)
        for topic in (segment.topic, "a newly edited topic"):
            request = CandidateRequest(
                request_id=f"fallback-{template.template_id}", target_bar=segment.start_bar,
                topic=topic, flow_template=template, count=1, context_lines=(), seed=1,
            )
            fallback = catalog.line_for_any(request)
            assert len(fallback.analysis.syllables) == len(template.slots)
            assert fallback.source == "prevalidated_fallback"
            assert fallback.topic == segment.topic


@pytest.mark.parametrize("template_id", CURATED_IDS)
def test_saved_scenario_can_load_a_curated_template(tmp_path, template_id: str) -> None:
    assert template_id in {segment.template_id for segment in default_scenario().segments}
    segment = next(segment for segment in default_scenario().segments if segment.template_id == template_id)
    path = tmp_path / "scenario.json"
    path.write_text(json.dumps({
        "scenario_id": "curated", "tempo_bpm": 100.0, "loop": False,
        "segments": [{"start_bar": 0, "bars": 1, "topic": segment.topic,
                      "template_id": template_id, "fallback_lines": list(segment.fallback_lines)}],
    }), encoding="utf-8")

    loaded = load_scenario(path)
    assert loaded.segment_for_bar(0).template_id == template_id
    PrevalidatedFallbackCatalog.build(loaded, BUILTIN_TEMPLATES, CmuProsodyAnalyzer())
