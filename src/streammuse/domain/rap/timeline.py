"""Immutable rap-only tempo map and exact, self-contained per-bar timing."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from streammuse.domain.timing import Tempo


def _nonnegative_int(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def validate_live_bpm(bpm: float) -> None:
    if isinstance(bpm, bool) or not isinstance(bpm, (int, float)) or not isfinite(bpm) or not 30 <= bpm <= 180:
        raise ValueError("BPM must be finite and between 30 and 180")


@dataclass(frozen=True)
class BarTimingSnapshot:
    revision: int
    bar: int
    bpm: float
    sample_rate_hz: int
    start_frame: int
    end_frame: int
    tick_offsets: tuple[int, ...]

    def __post_init__(self) -> None:
        for name in ("revision", "bar", "start_frame", "end_frame"):
            _nonnegative_int(getattr(self, name), name)
        if not isfinite(self.bpm) or self.bpm <= 0:
            raise ValueError("snapshot BPM must be positive and finite")
        if isinstance(self.sample_rate_hz, bool) or not isinstance(self.sample_rate_hz, int) or self.sample_rate_hz <= 0:
            raise ValueError("sample rate must be a positive integer")
        if (not isinstance(self.tick_offsets, tuple) or len(self.tick_offsets) != 17
                or self.tick_offsets[0] != 0 or self.tick_offsets[-1] != self.frame_count
                or any(isinstance(tick, bool) or not isinstance(tick, int) for tick in self.tick_offsets)
                or any(a >= b for a, b in zip(self.tick_offsets, self.tick_offsets[1:]))):
            raise ValueError("snapshot needs 17 strictly increasing tick boundaries")

    @property
    def frame_count(self) -> int:
        return self.end_frame - self.start_frame

    @property
    def tempo(self) -> Tempo:
        return Tempo(self.bpm, 4, 4)

    @property
    def ticks_per_bar(self) -> int:
        return 16

    @property
    def ticks_per_beat(self) -> int:
        return 4

    @property
    def beats_per_bar(self) -> int:
        return 4

    def to_payload(self) -> dict[str, object]:
        return {"revision": self.revision, "bar": self.bar, "bpm": self.bpm,
                "sample_rate_hz": self.sample_rate_hz, "start_frame": self.start_frame,
                "end_frame": self.end_frame, "tick_offsets": list(self.tick_offsets)}


@dataclass(frozen=True)
class _Segment:
    bar: int
    seconds: float
    bpm: float
    revision: int


@dataclass(frozen=True)
class RapTimeline:
    tempo: Tempo
    revision: int = 0
    _segments: tuple[_Segment, ...] = ()

    def __post_init__(self) -> None:
        if self.tempo.ticks_per_beat != 4 or self.tempo.beats_per_bar != 4:
            raise ValueError("rap timeline requires a 16-tick 4/4 bar")
        if not isfinite(self.tempo.bpm):
            raise ValueError("timeline BPM must be finite")
        _nonnegative_int(self.revision, "revision")
        if not self._segments:
            object.__setattr__(self, "_segments", (_Segment(0, 0.0, self.tempo.bpm, self.revision),))

    @property
    def ticks_per_bar(self) -> int:
        return 16

    @property
    def ticks_per_beat(self) -> int:
        return 4

    @property
    def beats_per_bar(self) -> int:
        return 4

    def _segment(self, bar: int) -> _Segment:
        return next(segment for segment in reversed(self._segments) if segment.bar <= bar)

    def tick_to_seconds(self, tick: int) -> float:
        _nonnegative_int(tick, "tick")
        segment = self._segment(tick // 16)
        return segment.seconds + (tick - segment.bar * 16) * (60.0 / segment.bpm / 4)

    def with_tempo(self, bpm: float, from_bar: int) -> "RapTimeline":
        validate_live_bpm(bpm)
        _nonnegative_int(from_bar, "from_bar")
        boundary = self.tick_to_seconds(from_bar * 16)
        revision = self.revision + 1
        segments = tuple(segment for segment in self._segments if segment.bar < from_bar)
        return RapTimeline(self.tempo, revision, segments + (_Segment(from_bar, boundary, float(bpm), revision),))

    def snapshot(self, bar: int, sample_rate_hz: int) -> BarTimingSnapshot:
        _nonnegative_int(bar, "bar")
        segment = self._segment(bar)
        start = round(self.tick_to_seconds(bar * 16) * sample_rate_hz)
        boundaries = tuple(round(self.tick_to_seconds(bar * 16 + tick) * sample_rate_hz) for tick in range(17))
        return BarTimingSnapshot(segment.revision, bar, segment.bpm, sample_rate_hz,
                                 start, boundaries[-1], tuple(frame - start for frame in boundaries))
