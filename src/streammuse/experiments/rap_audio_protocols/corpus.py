"""Strict loading for the immutable rap audio comparison corpus."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from streammuse.domain.timing import Tempo
from streammuse.experiments.rap_audio_protocols.contracts import SongCorpus, SyllableTarget, TwoBarRenderRequest
from streammuse.infrastructure.rap.templates import BUILTIN_TEMPLATES


def load_song_corpus(path: Path | str, *, song_id: str, expected_bars: int = 50) -> SongCorpus:
    """Load chosen lyric records and convert them into backend-neutral two-bar requests."""
    if expected_bars <= 0 or expected_bars % 2 != 0:
        raise ValueError("expected_bars must be a positive even count")

    records = _read_jsonl(path)
    if len(records) != expected_bars:
        raise ValueError(f"expected {expected_bars} bars, found {len(records)}")

    tempo = Tempo(90.0, 4, 4)
    for index, record in enumerate(records):
        _validate_record(record, expected_bar=index, tempo=tempo)

    requests = tuple(
        _build_request(song_id=song_id, tempo=tempo, chunk_index=chunk_index, bars=records[start : start + 2])
        for chunk_index, start in enumerate(range(0, len(records), 2))
    )
    return SongCorpus(song_id=song_id, tempo=tempo, requests=requests)


def _read_jsonl(path: Path | str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as source:
        for line in source:
            stripped = line.strip()
            if stripped:
                payload = json.loads(stripped)
                if not isinstance(payload, dict):
                    raise ValueError("corpus records must be JSON objects")
                records.append(payload)
    return records


def _validate_record(record: dict[str, Any], *, expected_bar: int, tempo: Tempo) -> None:
    bar = int(record["bar"])
    if bar != expected_bar:
        raise ValueError("corpus bars must be contiguous and zero-based")

    schedule = record["schedule"]
    if not isinstance(schedule, list):
        raise ValueError("corpus schedule must be a list")

    syllable_count = int(record["syllable_count"])
    if syllable_count != 9 or len(schedule) != 9:
        raise ValueError("every corpus bar must contain exactly nine syllables")

    template = BUILTIN_TEMPLATES.get(str(record["template_id"]))
    if len(template.slots) != 9:
        raise ValueError("comparison corpus requires nine-slot templates")

    normalized_words = str(record["normalized_text"]).split()
    schedule_words = _reconstructed_words(schedule)
    if schedule_words != normalized_words:
        raise ValueError("schedule words must reconstruct the lyric's analyzed word order")

    for position, item in enumerate(schedule):
        slot_index = int(item["slot_index"])
        if slot_index != position:
            raise ValueError("schedule slots must be ordered by slot_index")
        slot = template.slots[slot_index]
        if int(item["tick_in_bar"]) != slot.tick_in_bar:
            raise ValueError("schedule ticks must match the declared MCFlow template")
        if float(item["target_stress"]) != float(slot.target_stress):
            raise ValueError("schedule target stress must match the declared MCFlow template")
        absolute_tick = int(item["absolute_tick"])
        if absolute_tick != bar * tempo.ticks_per_bar + slot.tick_in_bar:
            raise ValueError("schedule absolute ticks must match tempo-derived bar positions")


def _reconstructed_words(schedule: list[dict[str, Any]]) -> list[str]:
    words: list[str] = []
    last_word: str | None = None
    for item in schedule:
        word = str(item["word"])
        if word != last_word:
            words.append(word)
            last_word = word
    return words


def _build_request(
    *,
    song_id: str,
    tempo: Tempo,
    chunk_index: int,
    bars: list[dict[str, Any]],
) -> TwoBarRenderRequest:
    start_bar = int(bars[0]["bar"])
    end_bar = int(bars[-1]["bar"]) + 1
    start_tick = start_bar * tempo.ticks_per_bar
    syllables = tuple(
        _build_syllable_target(record, item, start_tick=start_tick, tempo=tempo)
        for record in bars
        for item in record["schedule"]
    )
    text = " ".join(str(record["text"]).strip() for record in bars)
    return TwoBarRenderRequest(
        song_id=song_id,
        chunk_index=chunk_index,
        start_bar=start_bar,
        end_bar=end_bar,
        text=text,
        syllables=syllables,
    )


def _build_syllable_target(
    record: dict[str, Any],
    item: dict[str, Any],
    *,
    start_tick: int,
    tempo: Tempo,
) -> SyllableTarget:
    template = BUILTIN_TEMPLATES.get(str(record["template_id"]))
    slot_index = int(item["slot_index"])
    slot = template.slots[slot_index]
    word = str(item["word"])
    index_in_word = 0
    for earlier in reversed(record["schedule"][:slot_index]):
        if str(earlier["word"]) != word:
            break
        index_in_word += 1
    absolute_tick = int(item["absolute_tick"])
    tick_in_chunk = absolute_tick - start_tick
    return SyllableTarget(
        word=word,
        index_in_word=index_in_word,
        phonemes=tuple(str(phone) for phone in item["phonemes"]),
        lexical_stress=int(item["lexical_stress"]),
        target_stress=float(item["target_stress"]),
        boundary_strength=int(slot.boundary_strength),
        absolute_tick=absolute_tick,
        tick_in_chunk=tick_in_chunk,
        target_seconds=tick_in_chunk * tempo.seconds_per_tick,
    )
