from __future__ import annotations

import pytest

from streammuse.experiments.rap_audio_protocols.contracts import (
    SyllableTarget,
    TwoBarRenderRequest,
)


def _syllable(tick: int) -> SyllableTarget:
    return SyllableTarget(
        word="pulse",
        index_in_word=0,
        phonemes=("P", "AH1", "L", "S"),
        lexical_stress=1,
        target_stress=1.0,
        boundary_strength=0,
        absolute_tick=tick,
        tick_in_chunk=tick,
        target_seconds=tick / 6,
    )


def test_two_bar_request_accepts_variable_mcflow_density() -> None:
    request = TwoBarRenderRequest(
        song_id="mcflow_demo",
        chunk_index=0,
        start_bar=0,
        end_bar=2,
        text="Pulse pulse pulse pulse pulse.",
        syllables=tuple(_syllable(tick) for tick in (0, 3, 8, 16, 27)),
    )

    assert len(request.syllables) == 5
    assert request.duration_seconds == pytest.approx(16 / 3)


def test_two_bar_request_uses_and_serializes_its_tempo() -> None:
    request = TwoBarRenderRequest(
        song_id="mcflow_demo",
        chunk_index=0,
        start_bar=0,
        end_bar=2,
        text="Pulse.",
        syllables=(_syllable(0),),
        tempo_bpm=133.0,
    )

    assert request.duration_seconds == pytest.approx(480 / 133)
    assert request.to_payload()["tempo_bpm"] == 133.0


def test_two_bar_request_rejects_an_empty_vocal_schedule() -> None:
    with pytest.raises(ValueError, match="at least one syllable"):
        TwoBarRenderRequest(
            song_id="mcflow_demo",
            chunk_index=0,
            start_bar=0,
            end_bar=2,
            text="",
            syllables=(),
        )
