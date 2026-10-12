from __future__ import annotations

import pytest

from streammuse.application.rap.lyric_writer import (
    LyricWriteFailed,
    LyricWriteRequest,
    LyricWriterPolicy,
    SequentialLyricWriter,
)
from streammuse.domain.rap import (
    CandidateBatch,
    FlowProvenance,
    FlowSlot,
    FlowTemplate,
    ProsodyAnalysis,
    ScoreWeights,
    Syllable,
    normalize_text,
)


def _flow(rhyme: str | None = "A") -> FlowTemplate:
    return FlowTemplate(
        template_id=f"four-{rhyme}",
        name="four",
        ticks_per_beat=4,
        beats_per_bar=4,
        slots=tuple(
            FlowSlot(tick_in_bar=tick, duration_ticks=4, target_stress=1.0, rhyme_group=rhyme if tick == 12 else None)
            for tick in (0, 4, 8, 12)
        ),
        provenance=FlowProvenance(kind="test", source="unit-test"),
    )


class _Analyzer:
    """One syllable per word; the rhyme tail is the last word's final two letters."""

    def analyze(self, text: str) -> ProsodyAnalysis:
        normalized = normalize_text(text)
        words = normalized.split()
        return ProsodyAnalysis(
            text=text,
            normalized_text=normalized,
            syllables=tuple(Syllable(word, 0, 1, 1, ("AH1",), "fake") for word in words),
            end_rhyme_tail=(words[-1][-2:].upper() + "1",) if words else (),
            oov_words=(),
            heuristic_words=(),
            punctuation_boundary_after=(),
        )


class _Generator:
    def __init__(self, waves: list[tuple[str, ...] | Exception]) -> None:
        self.waves = list(waves)
        self.requests = []

    def generate(self, request):
        self.requests.append(request)
        result = self.waves[len(self.requests) - 1]
        if isinstance(result, Exception):
            raise result
        return CandidateBatch(
            request_id=request.request_id, candidates=result, source="fake", prompt=(), raw_response="", latency_ms=0.0
        )


def _request(bar: int = 1, previous: str | None = "we sail the night", **overrides) -> LyricWriteRequest:
    values = dict(
        session_id="s",
        bar=bar,
        revision=0,
        topic="sea",
        flow_template=_flow(),
        history=("we sail the night",) if previous else (),
        seed=7,
        previous_line=previous,
        previous_template=_flow() if previous else None,
    )
    values.update(overrides)
    return LyricWriteRequest(**values)


_WEIGHTS = ScoreWeights(
    stress_alignment=0.0, boundary_fit=0.0, rhyme_quality=1.0, topic_coverage=0.0, lexical_continuity=0.0, novelty=0.0
)


def test_odd_bar_in_a_couplet_rhymes_with_the_previous_line_and_picks_the_rhyme() -> None:
    generator = _Generator([("blue waves roll on", "the sea feels so", "deep tides take flight")])
    writer = SequentialLyricWriter(generator, _Analyzer(), _WEIGHTS, LyricWriterPolicy(candidates=3, minimum_valid=1))

    written = writer.write(_request(bar=1))

    assert generator.requests[0].rhyme_with == "we sail the night"
    assert written.record.text == "deep tides take flight"  # 'ht' matches 'night'
    assert written.record.bar == 1
    assert written.record.provenance["rhyme_with"] == "we sail the night"
    assert [row["status"] for row in written.ledger] == ["valid", "valid", "selected"]


def test_even_bar_starts_a_new_couplet_and_chain_scheme_always_rhymes() -> None:
    couplet = SequentialLyricWriter(_Generator([]), _Analyzer(), _WEIGHTS, LyricWriterPolicy())
    chain = SequentialLyricWriter(_Generator([]), _Analyzer(), _WEIGHTS, LyricWriterPolicy(rhyme_scheme="chain"))

    assert couplet.rhyme_target(_request(bar=2)) is None
    assert chain.rhyme_target(_request(bar=2)) == "we sail the night"
    assert couplet.rhyme_target(_request(bar=3, previous_template=_flow("B"))) is None
    assert couplet.rhyme_target(_request(bar=1, previous=None)) is None


def test_rescue_wave_runs_only_when_too_few_candidates_pass() -> None:
    generator = _Generator([("too short", "four words right here"), ("five words is too long", "now four words fit")])
    writer = SequentialLyricWriter(
        generator, _Analyzer(), _WEIGHTS, LyricWriterPolicy(candidates=2, rescue_candidates=2, minimum_valid=2)
    )

    written = writer.write(_request(bar=0, previous=None))

    assert [item.count for item in generator.requests] == [2, 2]
    assert written.waves == 2
    assert written.record.provenance["candidates_valid"] == 2


def test_generation_error_is_recorded_and_no_valid_candidate_fails_with_the_ledger() -> None:
    generator = _Generator([RuntimeError("vllm down"), ("too short",)])
    writer = SequentialLyricWriter(
        generator, _Analyzer(), _WEIGHTS, LyricWriterPolicy(candidates=2, rescue_candidates=1, minimum_valid=1)
    )

    with pytest.raises(LyricWriteFailed) as error:
        writer.write(_request(bar=0, previous=None))

    statuses = [row["status"] for row in error.value.ledger]
    assert statuses == ["generation_error", "rejected"]


def test_duplicate_of_history_is_rejected() -> None:
    generator = _Generator([("we sail the night", "we sail the light")])
    writer = SequentialLyricWriter(generator, _Analyzer(), _WEIGHTS, LyricWriterPolicy(candidates=2, minimum_valid=1))

    written = writer.write(_request(bar=1))

    assert written.record.text == "we sail the light"
