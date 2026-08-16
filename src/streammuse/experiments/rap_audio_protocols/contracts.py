"""Immutable contracts for offline rap audio protocol comparisons."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from typing import Any

from streammuse.domain.timing import Tempo


def _json_default(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    to_payload = getattr(value, "to_payload", None)
    if callable(to_payload):
        return to_payload()
    if is_dataclass(value):
        return asdict(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def canonical_json_dumps(value: Any) -> str:
    """Return a canonical JSON string with stable key ordering."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=_json_default)


def canonical_json_bytes(value: Any) -> bytes:
    """Return canonical UTF-8 encoded JSON bytes."""
    return canonical_json_dumps(value).encode("utf-8")


def sha256_hex(value: Any) -> str:
    """Return the SHA-256 of the canonical JSON form of a value."""
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


class ProtocolId(str, Enum):
    MOSS_GLOBAL = "moss_global"
    TED_LOCAL = "ted_local"
    FASTPITCH_PHONEME = "fastpitch_phoneme"
    MOSS_ALIGNED = "moss_aligned"


@dataclass(frozen=True)
class SyllableTarget:
    word: str
    index_in_word: int
    phonemes: tuple[str, ...]
    lexical_stress: int
    target_stress: float
    boundary_strength: int
    absolute_tick: int
    tick_in_chunk: int
    target_seconds: float

    def to_payload(self) -> dict[str, Any]:
        return {
            "word": self.word,
            "index_in_word": self.index_in_word,
            "phonemes": list(self.phonemes),
            "lexical_stress": self.lexical_stress,
            "target_stress": self.target_stress,
            "boundary_strength": self.boundary_strength,
            "absolute_tick": self.absolute_tick,
            "tick_in_chunk": self.tick_in_chunk,
            "target_seconds": self.target_seconds,
        }


@dataclass(frozen=True)
class TwoBarRenderRequest:
    song_id: str
    chunk_index: int
    start_bar: int
    end_bar: int
    text: str
    syllables: tuple[SyllableTarget, ...]

    def __post_init__(self) -> None:
        if self.end_bar - self.start_bar != 2:
            raise ValueError("two-bar render requests must span exactly two bars")
        if len(self.syllables) != 18:
            raise ValueError("two-bar render requests must contain exactly 18 syllables")
        if any(item.tick_in_chunk < 0 or item.tick_in_chunk >= 32 for item in self.syllables):
            raise ValueError("two-bar render requests must stay within a 32-tick chunk")

    @property
    def duration_seconds(self) -> float:
        return 32 * Tempo(90.0, 4, 4).seconds_per_tick

    def to_payload(self) -> dict[str, Any]:
        return {
            "song_id": self.song_id,
            "chunk_index": self.chunk_index,
            "start_bar": self.start_bar,
            "end_bar": self.end_bar,
            "duration_seconds": self.duration_seconds,
            "text": self.text,
            "syllables": [item.to_payload() for item in self.syllables],
        }

    def canonical_json_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_payload())

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json_bytes()).hexdigest()


@dataclass(frozen=True)
class ChunkRenderRecord:
    protocol_id: ProtocolId
    song_id: str
    chunk_index: int
    request_sha256: str
    success: bool
    output_path: str | None = None
    output_sha256: str | None = None
    source_chunk_sha256: str | None = None
    sample_rate_hz: int | None = None
    attempts: int = 0
    error: str | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "protocol_id": self.protocol_id.value,
            "song_id": self.song_id,
            "chunk_index": self.chunk_index,
            "request_sha256": self.request_sha256,
            "success": self.success,
            "output_path": self.output_path,
            "output_sha256": self.output_sha256,
            "source_chunk_sha256": self.source_chunk_sha256,
            "sample_rate_hz": self.sample_rate_hz,
            "attempts": self.attempts,
            "error": self.error,
        }

    def canonical_json_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_payload())

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json_bytes()).hexdigest()


@dataclass(frozen=True)
class SongCorpus:
    song_id: str
    tempo: Tempo
    requests: tuple[TwoBarRenderRequest, ...]

    def __post_init__(self) -> None:
        if self.tempo.bpm != 90.0:
            raise ValueError("rap audio comparison corpus requires exactly 90 BPM")
        if self.tempo.ticks_per_beat != 4 or self.tempo.beats_per_bar != 4:
            raise ValueError("rap audio comparison corpus requires 4/4 with four ticks per beat")

    def two_bar_requests(self) -> tuple[TwoBarRenderRequest, ...]:
        return self.requests
