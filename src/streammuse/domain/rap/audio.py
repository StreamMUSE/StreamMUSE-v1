"""Immutable domain contracts for optional realtime rap audio."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import TYPE_CHECKING

from streammuse.domain.rap.chords import CHORD_LOOP_IDS
from streammuse.domain.rap.models import ScheduledSyllable

if TYPE_CHECKING:
    from streammuse.domain.rap.timeline import BarTimingSnapshot


class AudioWarningSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class AudioWarningCode(str, Enum):
    PRONUNCIATION_FALLBACK = "pronunciation_fallback"
    TIMING_PRESSURE = "timing_pressure"
    FORCED_BAR_FIT = "forced_bar_fit"
    SYNTHESIS_FAILED = "synthesis_failed"
    AUDIO_DEADLINE_MISS = "audio_deadline_miss"
    AUDIO_UNDERRUN = "audio_underrun"
    AUDIO_DEVICE_FAILED = "audio_device_failed"


class PlaybackState(str, Enum):
    STOPPED = "stopped"
    PRIMING = "priming"
    RUNNING = "running"
    STOP_REQUESTED = "stop_requested"
    CLOSED = "closed"


@dataclass(frozen=True)
class AudioFormat:
    sample_rate_hz: int = 48_000
    channels: int = 2
    sample_width_bytes: int = 4


@dataclass(frozen=True)
class PcmAudio:
    format: AudioFormat
    frame_count: int
    data: bytes

    def __post_init__(self) -> None:
        expected_bytes = self.frame_count * self.format.channels * self.format.sample_width_bytes
        if len(self.data) != expected_bytes:
            raise ValueError(f"PCM data frame byte length must be {expected_bytes}, got {len(self.data)}")

    @property
    def duration_seconds(self) -> float:
        return self.frame_count / self.format.sample_rate_hz


@dataclass(frozen=True)
class AudioWarning:
    code: AudioWarningCode
    severity: AudioWarningSeverity
    message: str
    bar: int | None = None
    slot_index: int | None = None
    word: str | None = None
    available_ms: float | None = None
    rendered_ms: float | None = None
    compression_ratio: float | None = None
    overlap_ms: float | None = None
    action: str | None = None


@dataclass(frozen=True)
class SyllableRenderRequest:
    bar: int
    slot_index: int
    word: str
    index_in_word: int
    syllable_count: int
    phonemes: tuple[str, ...]
    stress: int
    analysis_source: str
    voice: str
    speed_wpm: int
    pitch: int


@dataclass(frozen=True)
class RenderedSyllable:
    request: SyllableRenderRequest
    audio: PcmAudio
    renderer_phonemes: tuple[str, ...]
    pronunciation_source: str
    synthesis_latency_ms: float
    warnings: tuple[AudioWarning, ...] = ()


@dataclass(frozen=True)
class SyllablePlacementDiagnostic:
    bar: int
    slot_index: int
    word: str
    target_sample: int
    source_frames: int
    fitted_frames: int
    available_frames: int
    compression_ratio: float
    overlap_frames: int
    pronunciation_source: str
    software_error_samples: int = 0
    renderer_phonemes: tuple[str, ...] = ()
    synthesis_latency_ms: float = 0.0
    rendered_frames: int | None = None
    cropped_frames: int = 0


@dataclass(frozen=True)
class RapMixConfig:
    """Versioned linear gains; construction alone preserves the legacy mix."""

    vocal_gain: float = 0.80
    drum_gain: float = 0.55
    master_gain: float = 1.0
    peak_limit: float = 0.95
    revision: int = 0
    chord_loop: str = "off"
    chord_gain: float = 0.20

    def __post_init__(self) -> None:
        for name, maximum in (("vocal_gain", 2.0), ("drum_gain", 2.0), ("master_gain", 1.0), ("peak_limit", 1.0), ("chord_gain", 2.0)):
            value = getattr(self, name)
            if type(value) not in (int, float) or not 0 <= value <= maximum or not isfinite(value):
                raise ValueError(f"{name} must be finite and between 0 and {maximum}")
        if self.peak_limit == 0:
            raise ValueError("peak_limit must be positive")
        if type(self.revision) is not int or self.revision < 0:
            raise ValueError("mix revision must be a non-negative integer")
        if not isinstance(self.chord_loop, str) or self.chord_loop not in CHORD_LOOP_IDS:
            raise ValueError(f"chord_loop must be one of {CHORD_LOOP_IDS}")

    @classmethod
    def with_drum_boost(cls, *, revision: int = 0) -> RapMixConfig:
        """Explicit settings preset: add 3 dB to the prior drum contribution."""
        return cls(drum_gain=0.55 * 10 ** (3 / 20), revision=revision)


@dataclass(frozen=True)
class RenderedOnsetEstimate:
    """A source-alignment/time-map estimate, never an acoustic observation."""

    slot_index: int
    requested_sample: int
    estimated_sample: float | None = None
    source_seconds: float | None = None
    confidence: float | None = None
    method: str = "unavailable"
    unavailable_reason: str | None = None


@dataclass(frozen=True)
class RapWarpDiagnostics:
    """Applied warp evidence in original source-audio and output-bar coordinates."""

    source_onsets: tuple[float, ...]
    source_anchors: tuple[float, ...]
    target_anchors: tuple[float, ...]
    local_warp_ratios: tuple[float, ...]
    policy: str
    warp_ms: float
    r3_attempt_deviations: tuple[int, ...] = ()


@dataclass(frozen=True)
class PreparedRapBar:
    bar: int
    text: str
    source: str
    fallback_reason: str | None
    scheduled: tuple[ScheduledSyllable, ...]
    audio: PcmAudio
    diagnostics: tuple[SyllablePlacementDiagnostic, ...]
    warnings: tuple[AudioWarning, ...]
    render_latency_ms: float
    vocal_stem: PcmAudio | None = None
    drum_stem: PcmAudio | None = None
    mix_config: RapMixConfig | None = None
    onset_estimates: tuple[RenderedOnsetEstimate, ...] = ()
    warp_diagnostics: RapWarpDiagnostics | None = None
    timing: BarTimingSnapshot | None = None
    chord_stem: PcmAudio | None = None


class AudioPlaybackNoticeKind(str, Enum):
    BAR_STARTED = "bar_started"
    BAR_COMPLETED = "bar_completed"
    STOPPED = "stopped"
    UNDERRUN = "underrun"
    DEVICE_FAILED = "device_failed"


@dataclass(frozen=True)
class AudioPlaybackNotice:
    kind: AudioPlaybackNoticeKind
    bar: int | None
    absolute_frame: int
    queue_depth: int
    message: str
    buffered_seconds: float = 0.0


@dataclass(frozen=True)
class AudioPlaybackSnapshot:
    state: PlaybackState
    current_bar: int | None
    last_completed_bar: int | None
    frame_in_bar: int
    absolute_frame: int
    queue_depth: int
    underrun_count: int
    buffered_seconds: float = 0.0


@dataclass(frozen=True)
class SpeechPrefix:
    """Speech a synthesis continues from: the previous line's 24 kHz mono WAV and its text."""

    text: str
    wav: bytes

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("speech prefix text must not be empty")
        if not self.wav:
            raise ValueError("speech prefix audio must not be empty")
