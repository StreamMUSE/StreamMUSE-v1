"""Tests for the rap audio protocol comparison corpus loader."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from streammuse.domain.timing import Tempo
from streammuse.experiments.rap_audio_protocols import ProtocolId, SongCorpus, load_song_corpus, sha256_hex


FIXTURE_PATH = Path("tests/fixtures/rap_audio_protocols/two_bar_records.jsonl")


def _fixture_records() -> list[dict[str, Any]]:
    return [json.loads(line) for line in FIXTURE_PATH.read_text(encoding="utf-8").splitlines()]


def _write_records(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text("\n".join(json.dumps(record, sort_keys=True) for record in records) + "\n", encoding="utf-8")


def test_load_song_corpus_builds_one_backend_neutral_two_bar_request() -> None:
    corpus = load_song_corpus(FIXTURE_PATH, song_id="01_space_exploration", expected_bars=2)
    request = corpus.two_bar_requests()[0]

    assert corpus.song_id == "01_space_exploration"
    assert corpus.tempo == Tempo(90.0, 4, 4)
    assert {protocol.value for protocol in ProtocolId} == {
        "moss_global",
        "ted_local",
        "fastpitch_phoneme",
        "moss_aligned",
    }
    assert request.chunk_index == 0
    assert request.start_bar == 0
    assert request.end_bar == 2
    assert request.duration_seconds == pytest.approx(16 / 3)
    assert len(request.syllables) == 18
    assert [item.absolute_tick for item in request.syllables] == sorted(
        item.absolute_tick for item in request.syllables
    )
    assert request.syllables[0].word == "rocket"
    assert request.syllables[0].index_in_word == 0
    assert request.syllables[0].tick_in_chunk == 0
    assert request.syllables[0].target_seconds == pytest.approx(0.0)
    assert request.syllables[-1].word == "sea"
    assert request.syllables[-1].tick_in_chunk == 31
    assert request.syllables[-1].target_seconds == pytest.approx(31 / 6)
    assert request.text.split() == (
        "Rocket blasts liftoff, silence breaks free, Shadows stretch beneath vast cosmic sea,"
    ).split()

    canonical = request.canonical_json_bytes()
    assert canonical == request.canonical_json_bytes()
    assert request.sha256 == hashlib.sha256(canonical).hexdigest()
    assert sha256_hex(request) == request.sha256


def test_load_song_corpus_rejects_noncontiguous_bars(tmp_path: Path) -> None:
    records = _fixture_records()
    records[1]["bar"] = 2
    path = tmp_path / "noncontiguous.jsonl"
    _write_records(path, records)

    with pytest.raises(ValueError, match="contiguous"):
        load_song_corpus(path, song_id="01_space_exploration", expected_bars=2)


def test_load_song_corpus_rejects_schedule_word_order_mismatch(tmp_path: Path) -> None:
    records = _fixture_records()
    records[0]["schedule"][2]["word"] = "thrusters"
    path = tmp_path / "word-mismatch.jsonl"
    _write_records(path, records)

    with pytest.raises(ValueError, match="reconstruct"):
        load_song_corpus(path, song_id="01_space_exploration", expected_bars=2)


def test_load_song_corpus_rejects_non_nine_syllable_bar(tmp_path: Path) -> None:
    records = _fixture_records()
    records[0]["syllable_count"] = 8
    records[0]["schedule"] = records[0]["schedule"][:-1]
    path = tmp_path / "bad-syllables.jsonl"
    _write_records(path, records)

    with pytest.raises(ValueError, match="nine syllables"):
        load_song_corpus(path, song_id="01_space_exploration", expected_bars=2)


def test_load_song_corpus_resets_within_word_index_for_repeated_surface_words(tmp_path: Path) -> None:
    records = _fixture_records()
    records[0]["normalized_text"] = "rocket breaks rocket silence breaks free"
    records[0]["text"] = "Rocket breaks rocket silence breaks free,"
    records[0]["schedule"][2]["word"] = "breaks"
    records[0]["schedule"][3]["word"] = "rocket"
    records[0]["schedule"][3]["phonemes"] = ["R", "AA1", "K"]
    records[0]["schedule"][4]["word"] = "rocket"
    records[0]["schedule"][4]["phonemes"] = ["AH0", "T"]
    records[0]["schedule"][7]["word"] = "breaks"
    path = tmp_path / "repeated-words.jsonl"
    _write_records(path, records)

    corpus = load_song_corpus(path, song_id="01_space_exploration", expected_bars=2)
    chunk = corpus.two_bar_requests()[0]
    rocket_indices = [item.index_in_word for item in chunk.syllables if item.word == "rocket"]

    assert rocket_indices == [0, 1, 0, 1]


def test_song_corpus_rejects_any_tempo_other_than_90_bpm() -> None:
    valid = load_song_corpus(FIXTURE_PATH, song_id="01_space_exploration", expected_bars=2)

    with pytest.raises(ValueError, match="90 BPM"):
        SongCorpus(
            song_id=valid.song_id,
            tempo=Tempo(92.0, 4, 4),
            requests=valid.two_bar_requests(),
        )
