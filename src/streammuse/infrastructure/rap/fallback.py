"""Validated deterministic fallback lyrics for the scenario-aware planner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from streammuse.domain.rap import CandidateRequest, ProsodyAnalysis, RapScenario, normalize_text
from streammuse.infrastructure.rap.scenarios import default_scenario
from streammuse.infrastructure.rap.templates import TemplateCatalog


class ProsodyAnalyzer(Protocol):
    """Analyze a fallback line before the realtime path can use it."""

    def analyze(self, text: str) -> ProsodyAnalysis:
        """Return immutable prosody analysis for ``text``."""


@dataclass(frozen=True)
class PrevalidatedFallbackLine:
    """A validated fallback line with explicit non-generated provenance."""

    analysis: ProsodyAnalysis
    source: str = "prevalidated_fallback"
    topic: str | None = None

    @property
    def text(self) -> str:
        """Return the original configured line text."""
        return self.analysis.text


class PrevalidatedFallbackCatalog:
    """Lookup all startup-validated fallbacks by normalized topic and template."""

    def __init__(self, lines: dict[tuple[str, str], tuple[ProsodyAnalysis, ...]]) -> None:
        self._lines = {key: tuple(value) for key, value in lines.items()}

    @classmethod
    def build(
        cls,
        scenario: RapScenario,
        templates: TemplateCatalog,
        analyzer: ProsodyAnalyzer,
    ) -> "PrevalidatedFallbackCatalog":
        """Analyze every configured fallback and reject structural mismatches."""
        lines: dict[tuple[str, str], list[ProsodyAnalysis]] = {}
        for segment in scenario.segments:
            if not segment.fallback_lines:
                raise ValueError(f"fallback segment {segment.template_id} requires at least one line")
            template = templates.get(segment.template_id)
            analyses = tuple(analyzer.analyze(line) for line in segment.fallback_lines)
            invalid = [analysis.text for analysis in analyses if len(analysis.syllables) != len(template.slots)]
            if invalid:
                raise ValueError(f"fallback lines do not match {template.template_id}: {invalid}")
            key = (normalize_topic(segment.topic), template.template_id)
            lines.setdefault(key, []).extend(analyses)
        if not any(lines.values()):
            raise ValueError("prevalidated fallback catalog requires at least one line")

        # Live picker edits can outgrow a legacy scenario's syllable counts.
        # Fill only missing counts so configured lines keep their existing priority.
        covered_counts = {len(analysis.syllables) for choices in lines.values() for analysis in choices}
        selectable = {template.template_id: template for template in templates.values()}
        for segment in default_scenario().segments:
            template = selectable.get(segment.template_id)
            if template is None or len(template.slots) in covered_counts:
                continue
            analyses = tuple(analyzer.analyze(line) for line in segment.fallback_lines)
            invalid = [analysis.text for analysis in analyses if len(analysis.syllables) != len(template.slots)]
            if invalid:
                raise ValueError(f"fallback lines do not match {template.template_id}: {invalid}")
            lines[(normalize_topic(segment.topic), template.template_id)] = list(analyses)
            covered_counts.add(len(template.slots))

        return cls({key: tuple(value) for key, value in lines.items()})

    def line_for(self, bar_context: CandidateRequest) -> PrevalidatedFallbackLine:
        """Return a deterministic prevalidated fallback for one absolute target bar."""
        if not isinstance(bar_context, CandidateRequest):
            raise TypeError("fallback lookup requires a CandidateRequest")
        key = (normalize_topic(bar_context.topic), bar_context.template_id)
        try:
            choices = self._lines[key]
        except KeyError as exc:
            raise ValueError(
                f"no prevalidated fallback for topic {key[0]!r} and template {bar_context.template_id!r}"
            ) from exc
        if not choices:
            raise ValueError(f"no prevalidated fallback lines for template {bar_context.template_id!r}")
        analysis = choices[bar_context.target_bar % len(choices)]
        if len(analysis.syllables) != bar_context.required_syllables:
            raise ValueError(
                "fallback request syllable count does not match configured template: "
                f"{bar_context.required_syllables}!={len(analysis.syllables)}"
            )
        return PrevalidatedFallbackLine(analysis=analysis, topic=bar_context.topic)


    def line_for_any(self, bar_context: CandidateRequest) -> PrevalidatedFallbackLine:
        """``line_for``, else a line of the same template under any topic, else any line with the
        template's syllable count, so a live edit to a new topic or template still has a fallback."""
        try:
            return self.line_for(bar_context)
        except ValueError:
            pass
        needed = bar_context.required_syllables
        same_template = [
            (topic, analysis)
            for (topic, template_id), lines in sorted(self._lines.items())
            if template_id == bar_context.template_id
            for analysis in lines
            if len(analysis.syllables) == needed
        ]
        same_count = [
            (topic, analysis)
            for (topic, _), lines in sorted(self._lines.items())
            for analysis in lines
            if len(analysis.syllables) == needed
        ]
        for pool in (same_template, same_count):
            if pool:
                topic, analysis = pool[bar_context.target_bar % len(pool)]
                return PrevalidatedFallbackLine(analysis=analysis, topic=topic)
        raise ValueError(f"no prevalidated fallback line has {needed} syllables")


def normalize_topic(topic: str) -> str:
    """Normalize a scenario or request topic at the fallback lookup boundary."""
    if not isinstance(topic, str):
        raise ValueError("fallback topic must be a string")
    normalized = normalize_text(topic)
    if not normalized:
        raise ValueError("fallback topic must contain at least one word")
    return normalized
