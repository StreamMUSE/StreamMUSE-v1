"""Versioned musical coverage, independent of generation or delivery granularity."""

from dataclasses import dataclass, field
import json
from math import isfinite


@dataclass(frozen=True)
class InferenceRequest:
    session_id: str
    epoch: int
    request_id: str
    input_version: int
    start_tick: int
    duration_ticks: int
    bpm: float
    ticks_per_beat: int
    beats_per_bar: int
    topic: str
    template_id: str
    flow_ticks: tuple[int, ...] = ()
    history: tuple[str, ...] = ()
    config: dict = field(default_factory=dict)

    def __post_init__(self):
        for name in ("duration_ticks", "ticks_per_beat", "beats_per_bar"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("epoch", "input_version", "start_tick"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if not isfinite(self.bpm) or self.bpm <= 0:
            raise ValueError("bpm must be finite and positive")
        if any(type(tick) is not int or not 0 <= tick < self.duration_ticks for tick in self.flow_ticks):
            raise ValueError("flow onset lies outside request")


@dataclass(frozen=True)
class AudioSpan:
    request_id: str
    output_id: str
    epoch: int
    input_version: int
    start_tick: int
    duration_ticks: int
    sample_rate: int
    channels: int
    pcm: bytes
    revision: int = 0
    safe_split_ticks: tuple[int, ...] = ()
    declared_silence: bool = False
    text: str = ""
    syllables: tuple[dict, ...] = ()
    stage_timings: dict = field(default_factory=dict)
    final: bool = False
    first_vocal_tick: int | None = None

    def __post_init__(self):
        for name in ("duration_ticks", "sample_rate", "channels"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("epoch", "input_version", "start_tick", "revision"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative")
        if not self.output_id or not self.request_id:
            raise ValueError("span requires request and output identities")
        if not isinstance(self.pcm, bytes) or not self.pcm or len(self.pcm) % (self.channels * 4):
            raise ValueError("span requires nonempty complete float32 frames")
        if len(self.pcm) > 64 * 1024 * 1024:
            raise ValueError("span exceeds PCM payload bound")
        if tuple(sorted(set(self.safe_split_ticks))) != self.safe_split_ticks or any(
            type(tick) is not int or not 0 < tick < self.duration_ticks for tick in self.safe_split_ticks
        ):
            raise ValueError("safe split ticks must be ordered interior offsets")
        if self.first_vocal_tick is not None and (type(self.first_vocal_tick) is not int
                                                 or not 0 <= self.first_vocal_tick < self.duration_ticks):
            raise ValueError("first vocal onset lies outside span")
        try:
            json.dumps({"stage_timings": self.stage_timings, "syllables": self.syllables}, allow_nan=False)
        except (ValueError, TypeError) as error:
            raise ValueError("span telemetry must be finite JSON data") from error

    @property
    def end_tick(self):
        return self.start_tick + self.duration_ticks

    @property
    def frame_count(self):
        return len(self.pcm) // (4 * self.channels)
