"""Immutable domain contracts for optional realtime rap audio."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from streammuse.domain.rap.models import ScheduledSyllable


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


@dataclass(frozen=True)
class AudioPlaybackSnapshot:
    state: PlaybackState
    current_bar: int | None
    frame_in_bar: int
    absolute_frame: int
    queue_depth: int
    underrun_count: int
