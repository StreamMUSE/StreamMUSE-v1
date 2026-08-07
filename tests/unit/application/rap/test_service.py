"""Tests for multi-bar rap-plan assembly."""

import pytest

from streammuse.application.rap.service import RapPrototypeService
from streammuse.domain.rap import CandidateBatch, CandidateRequest
from streammuse.domain.timing import Tempo
from streammuse.infrastructure.rap.generators import PhraseBankGenerator


class FixedGenerator:
    def generate(self, request: CandidateRequest) -> CandidateBatch:
        return CandidateBatch(
            request_id=request.request_id,
            candidates=(
                "space travel in the midnight all the speakers keep it bright",
                "we bring space travel through the rhythm making every pulse ignite",
            ),
            source="fixed",
            prompt=(),
            raw_response="",
            latency_ms=0.0,
        )


def test_phrase_bank_plan_contains_a_unique_aligned_line_per_requested_bar() -> None:
    plan = RapPrototypeService(
        Tempo(bpm=92, ticks_per_beat=4, beats_per_bar=4),
        "boom_bap",
        PhraseBankGenerator(),
    ).build_plan("space travel", bars=2, candidate_count=8)

    assert len(plan.lines) == 2
    assert all(line.events for line in plan.lines)
    assert len({line.text for line in plan.lines}) == 2
    assert [line.events[0].slot.tick for line in plan.lines] == [0, 16]


def test_service_preserves_generator_metadata_and_warning() -> None:
    generator = FixedGenerator()

    plan = RapPrototypeService(
        Tempo(bpm=92, ticks_per_beat=4, beats_per_bar=4),
        "straight_8",
        generator,
    ).build_plan("space travel", bars=1, candidate_count=2)

    assert plan.candidate_source == "fixed"
    assert plan.warning is None
    assert plan.pattern == "straight_8"


@pytest.mark.parametrize("bars,candidate_count", [(0, 2), (1, 0)])
def test_service_rejects_nonpositive_plan_dimensions(bars: int, candidate_count: int) -> None:
    service = RapPrototypeService(
        Tempo(bpm=92, ticks_per_beat=4, beats_per_bar=4),
        "boom_bap",
        FixedGenerator(),
    )

    with pytest.raises(ValueError, match="positive"):
        service.build_plan("space travel", bars=bars, candidate_count=candidate_count)
