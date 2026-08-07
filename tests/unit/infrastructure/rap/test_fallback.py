"""Tests for startup validation and deterministic fallback lookup."""

import pytest

from streammuse.domain.rap import (
    CandidateRequest,
    FlowProvenance,
    FlowSlot,
    FlowTemplate,
    RapScenario,
    ScenarioSegment,
)
from streammuse.infrastructure.rap.fallback import PrevalidatedFallbackCatalog
from streammuse.infrastructure.rap.prosody import HeuristicProsodyAnalyzer
from streammuse.infrastructure.rap.templates import TemplateCatalog


def _templates() -> TemplateCatalog:
    return TemplateCatalog.from_templates(
        (
            FlowTemplate(
                template_id="one_slot",
                name="One slot",
                ticks_per_beat=4,
                beats_per_bar=4,
                slots=(FlowSlot(0, 16, 1.0),),
                provenance=FlowProvenance(kind="test", source="test"),
            ),
        )
    )


def _request(bar: int, *, topic: str = "space") -> CandidateRequest:
    return CandidateRequest(
        request_id=f"request-{bar}",
        target_bar=bar,
        topic=topic,
        template_id="one_slot",
        required_syllables=1,
        count=1,
        context_lines=(),
        seed=1,
    )


def test_catalog_returns_prevalidated_lines_round_robin_by_absolute_bar() -> None:
    scenario = RapScenario(
        scenario_id="fallbacks",
        tempo_bpm=120.0,
        segments=(ScenarioSegment(0, 4, "space", "one_slot", ("one", "two")),),
    )
    catalog = PrevalidatedFallbackCatalog.build(scenario, _templates(), HeuristicProsodyAnalyzer())

    assert catalog.line_for(_request(0)).text == "one"
    assert catalog.line_for(_request(1)).text == "two"
    assert catalog.line_for(_request(2)).text == "one"
    assert catalog.line_for(_request(2)).source == "prevalidated_fallback"


def test_catalog_preserves_repeated_normalized_topic_template_lines() -> None:
    scenario = RapScenario(
        scenario_id="repeated",
        tempo_bpm=120.0,
        segments=(
            ScenarioSegment(0, 1, "Space!", "one_slot", ("one",)),
            ScenarioSegment(1, 1, "space", "one_slot", ("two",)),
        ),
    )
    catalog = PrevalidatedFallbackCatalog.build(scenario, _templates(), HeuristicProsodyAnalyzer())

    assert [catalog.line_for(_request(bar, topic="SPACE")).text for bar in range(4)] == ["one", "two", "one", "two"]


def test_catalog_rejects_syllable_mismatch_during_build() -> None:
    scenario = RapScenario(
        scenario_id="invalid",
        tempo_bpm=120.0,
        segments=(ScenarioSegment(0, 1, "space", "one_slot", ("one two",)),),
    )

    with pytest.raises(ValueError, match="fallback lines do not match one_slot: \['one two'\]"):
        PrevalidatedFallbackCatalog.build(scenario, _templates(), HeuristicProsodyAnalyzer())


def test_catalog_rejects_unknown_lookup_key() -> None:
    scenario = RapScenario(
        scenario_id="fallbacks",
        tempo_bpm=120.0,
        segments=(ScenarioSegment(0, 1, "space", "one_slot", ("one",)),),
    )
    catalog = PrevalidatedFallbackCatalog.build(scenario, _templates(), HeuristicProsodyAnalyzer())

    with pytest.raises(ValueError, match="no prevalidated fallback"):
        catalog.line_for(_request(0, topic="ocean"))
