"""Tests for prescheduled rap scenarios."""

import pytest

from streammuse.domain.rap import RapScenario, ScenarioSegment


def test_scenario_changes_only_at_bar_boundaries_and_loops() -> None:
    scenario = RapScenario(
        scenario_id="test",
        tempo_bpm=96.0,
        loop=True,
        segments=(
            ScenarioSegment(start_bar=0, bars=2, topic="space", template_id="a", fallback_lines=("one two",)),
            ScenarioSegment(start_bar=2, bars=1, topic="deep sea", template_id="b", fallback_lines=("three four",)),
        ),
    )

    assert scenario.segment_for_bar(0).topic == "space"
    assert scenario.segment_for_bar(2).topic == "deep sea"
    assert scenario.segment_for_bar(3).topic == "space"


def test_non_looping_scenario_rejects_bar_after_its_final_segment() -> None:
    scenario = RapScenario(
        scenario_id="fixed",
        tempo_bpm=96.0,
        loop=False,
        segments=(ScenarioSegment(0, 1, "space", "a", ("one two",)),),
    )

    with pytest.raises(IndexError, match="bar 1 lies outside scenario fixed"):
        scenario.segment_for_bar(1)


def test_scenario_rejects_negative_bar() -> None:
    scenario = RapScenario(
        scenario_id="fixed",
        tempo_bpm=96.0,
        loop=False,
        segments=(ScenarioSegment(0, 1, "space", "a", ("one two",)),),
    )

    with pytest.raises(ValueError, match="bar must not be negative"):
        scenario.segment_for_bar(-1)
