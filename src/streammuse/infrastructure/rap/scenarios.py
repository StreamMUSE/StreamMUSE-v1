"""Built-in and external JSON rap scenarios."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from streammuse.domain.rap.scenario import RapScenario, ScenarioSegment
from streammuse.infrastructure.rap.templates import BUILTIN_TEMPLATES, TemplateCatalog


def default_scenario() -> RapScenario:
    """Return the fixed looping research demonstration scenario."""
    scenario = RapScenario(
        scenario_id="default_research_demo",
        tempo_bpm=90.0,
        loop=True,
        segments=(
            ScenarioSegment(0, 4, "space", "baseline_syncopated_9", ("space dreams rise while bright stars cross dark night",)),
            ScenarioSegment(4, 4, "deep sea", "baseline_straight_9", ("deep sea winds move while moon lights guide ships",)),
            ScenarioSegment(8, 4, "code", "baseline_staggered_9", ("code sparks grow as quick hands shape new sound",)),
        ),
    )
    _validate_scenario(scenario, BUILTIN_TEMPLATES)
    return scenario


def load_scenario(path: Path | str) -> RapScenario:
    """Load and validate an externally supplied JSON scenario."""
    with Path(path).open(encoding="utf-8") as source:
        payload = json.load(source)
    scenario = _scenario_from_payload(payload)
    _validate_scenario(scenario, BUILTIN_TEMPLATES)
    return scenario


def _scenario_from_payload(payload: Any) -> RapScenario:
    if not isinstance(payload, dict):
        raise ValueError("scenario JSON must contain an object")
    try:
        scenario_id = _require_string(payload, "scenario_id")
        tempo_bpm = _require_number(payload, "tempo_bpm")
        loop = _optional_bool(payload, "loop", default=True)
        raw_segments = payload["segments"]
        if not isinstance(raw_segments, list):
            raise ValueError("scenario segments must be a list")
        return RapScenario(
            scenario_id=scenario_id,
            tempo_bpm=tempo_bpm,
            loop=loop,
            segments=tuple(_segment_from_payload(segment) for segment in raw_segments),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("invalid scenario JSON") from exc


def _segment_from_payload(payload: Any) -> ScenarioSegment:
    if not isinstance(payload, dict):
        raise ValueError("scenario segment must be an object")
    fallback_lines = payload["fallback_lines"]
    if not isinstance(fallback_lines, list) or not all(isinstance(line, str) for line in fallback_lines):
        raise ValueError("scenario fallback lines must be a list of strings")
    return ScenarioSegment(
        start_bar=_require_int(payload, "start_bar"),
        bars=_require_int(payload, "bars"),
        topic=_require_string(payload, "topic"),
        template_id=_require_string(payload, "template_id"),
        fallback_lines=tuple(fallback_lines),
    )


def _require_string(payload: dict[str, Any], key: str) -> str:
    value = payload[key]
    if not isinstance(value, str):
        raise ValueError(f"scenario {key} must be a string")
    return value


def _require_int(payload: dict[str, Any], key: str) -> int:
    value = payload[key]
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"scenario {key} must be an integer")
    return value


def _require_number(payload: dict[str, Any], key: str) -> float:
    value = payload[key]
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"scenario {key} must be a number")
    return float(value)


def _optional_bool(payload: dict[str, Any], key: str, *, default: bool) -> bool:
    value = payload.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"scenario {key} must be a boolean")
    return value


def _validate_scenario(scenario: RapScenario, templates: TemplateCatalog) -> None:
    if not scenario.scenario_id:
        raise ValueError("scenario requires an id")
    if scenario.tempo_bpm <= 0:
        raise ValueError("scenario tempo must be positive")
    if not scenario.segments:
        raise ValueError("scenario requires at least one segment")

    expected_start = 0
    for segment in scenario.segments:
        if segment.start_bar != expected_start or segment.bars <= 0:
            raise ValueError("scenario segments must be contiguous positive bar ranges")
        if not segment.topic or not segment.template_id:
            raise ValueError("scenario segments require a topic and template id")
        if not segment.fallback_lines or any(not line.strip() for line in segment.fallback_lines):
            raise ValueError("scenario segments require nonempty fallback lines")
        templates.get(segment.template_id)
        expected_start += segment.bars

    if scenario.total_bars <= 0:
        raise ValueError("scenario loop length must be positive")
