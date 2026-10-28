"""Tests for built-in rap flow templates and scenario assembly."""

import json

import pytest

from streammuse.infrastructure.rap.scenarios import default_scenario, load_scenario
from streammuse.infrastructure.rap.templates import BUILTIN_TEMPLATES, TemplateCatalog
from streammuse.domain.rap import CandidateRequest
from streammuse.infrastructure.rap.fallback import PrevalidatedFallbackCatalog
from streammuse.infrastructure.rap.prosody import CmuProsodyAnalyzer


def test_builtin_templates_have_nine_slots_and_honest_provenance() -> None:
    for template_id in ("baseline_straight_9", "baseline_syncopated_9", "baseline_staggered_9"):
        template = BUILTIN_TEMPLATES.get(template_id)

        assert len(template.slots) == 9
        assert template.provenance.kind == "hand_authored_mcflow_inspired"
        assert template.provenance.source == "StreamMUSE baseline"


@pytest.mark.parametrize("grid", ("straight", "syncopated", "staggered"))
@pytest.mark.parametrize("accent", ("backbeat", "rising"))
def test_stress_variants_preserve_onsets_and_untested_handcrafted_provenance(grid, accent):
    baseline = BUILTIN_TEMPLATES.get(f"baseline_{grid}_9")
    variant = BUILTIN_TEMPLATES.get(f"untested_{grid}_{accent}_9")

    assert [slot.tick_in_bar for slot in variant.slots] == [slot.tick_in_bar for slot in baseline.slots]
    assert [slot.duration_ticks for slot in variant.slots] == [slot.duration_ticks for slot in baseline.slots]
    assert [slot.target_stress for slot in variant.slots] != [slot.target_stress for slot in baseline.slots]
    assert "UNTESTED" in variant.name
    assert variant.provenance.kind == "hand_authored_stress_variant"
    assert "UNTESTED" in variant.provenance.source
    assert baseline.template_id in variant.provenance.source
    assert variant.provenance.source_hash is None


def test_catalog_rejects_empty_catalog_and_unknown_template() -> None:
    with pytest.raises(ValueError, match="template catalog must not be empty"):
        TemplateCatalog.from_templates(())

    with pytest.raises(ValueError, match="unknown flow template: unknown"):
        BUILTIN_TEMPLATES.get("unknown")


def test_default_scenario_is_valid_against_builtin_templates() -> None:
    scenario = default_scenario()

    assert scenario.tempo_bpm == 90.0
    assert scenario.total_bars == 12
    assert [scenario.segment_for_bar(bar).topic for bar in (0, 4, 8, 12)] == ["space", "deep sea", "code", "space"]
    assert all(BUILTIN_TEMPLATES.get(segment.template_id) for segment in scenario.segments)
    assert all(line.strip() for segment in scenario.segments for line in segment.fallback_lines)


def test_default_cli_fallback_catalog_can_resolve_every_new_stress_variant() -> None:
    catalog = PrevalidatedFallbackCatalog.build(default_scenario(), BUILTIN_TEMPLATES, CmuProsodyAnalyzer())
    for grid in ("straight", "syncopated", "staggered"):
        for accent in ("backbeat", "rising"):
            template = BUILTIN_TEMPLATES.get(f"untested_{grid}_{accent}_9")
            request = CandidateRequest(
                request_id=f"test-{template.template_id}", target_bar=0, topic="new topic",
                flow_template=template, count=1, context_lines=(), seed=1,
            )
            fallback = catalog.line_for_any(request)
            assert len(fallback.analysis.syllables) == 9
            assert fallback.source == "prevalidated_fallback"
            assert fallback.topic != "new topic"


def test_load_scenario_builds_valid_json_scenario(tmp_path) -> None:
    path = tmp_path / "scenario.json"
    path.write_text(
        json.dumps(
            {
                "scenario_id": "custom",
                "tempo_bpm": 100.0,
                "loop": False,
                "segments": [
                    {
                        "start_bar": 0,
                        "bars": 1,
                        "topic": "space",
                        "template_id": "baseline_straight_9",
                        "fallback_lines": ["space dreams rise while bright stars cross dark night"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    scenario = load_scenario(path)

    assert scenario.scenario_id == "custom"
    assert scenario.loop is False
    assert scenario.segment_for_bar(0).template_id == "baseline_straight_9"


@pytest.mark.parametrize(
    "payload",
    (
        {
            "scenario_id": "bad_range",
            "tempo_bpm": 92.0,
            "segments": [{"start_bar": 1, "bars": 1, "topic": "space", "template_id": "baseline_straight_9", "fallback_lines": ["space dreams rise while bright stars cross dark night"]}],
        },
        {
            "scenario_id": "bad_template",
            "tempo_bpm": 92.0,
            "segments": [{"start_bar": 0, "bars": 1, "topic": "space", "template_id": "missing", "fallback_lines": ["space dreams rise while bright stars cross dark night"]}],
        },
        {
            "scenario_id": "empty_fallback",
            "tempo_bpm": 92.0,
            "segments": [{"start_bar": 0, "bars": 1, "topic": "space", "template_id": "baseline_straight_9", "fallback_lines": [""]}],
        },
    ),
)
def test_load_scenario_rejects_invalid_dependency_assembly(tmp_path, payload: dict[str, object]) -> None:
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError):
        load_scenario(path)


@pytest.mark.parametrize(
    "payload",
    (
        {
            "scenario_id": "string_loop",
            "tempo_bpm": 92.0,
            "loop": "false",
            "segments": [{"start_bar": 0, "bars": 1, "topic": "space", "template_id": "baseline_straight_9", "fallback_lines": ["space dreams rise while bright stars cross dark night"]}],
        },
        {
            "scenario_id": "fractional_bar",
            "tempo_bpm": 92.0,
            "segments": [{"start_bar": 0.5, "bars": 1, "topic": "space", "template_id": "baseline_straight_9", "fallback_lines": ["space dreams rise while bright stars cross dark night"]}],
        },
        {
            "scenario_id": "null_topic",
            "tempo_bpm": 92.0,
            "segments": [{"start_bar": 0, "bars": 1, "topic": None, "template_id": "baseline_straight_9", "fallback_lines": ["space dreams rise while bright stars cross dark night"]}],
        },
    ),
)
def test_load_scenario_rejects_malformed_json_field_types(tmp_path, payload: dict[str, object]) -> None:
    path = tmp_path / "malformed.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="invalid scenario JSON"):
        load_scenario(path)
