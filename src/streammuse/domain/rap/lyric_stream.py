"""Protocol v3: a per-bar lyric stream and per-bar remote audio.

The server writes one lyric line per bar and streams each line to the client.
A line is identified by its ``lyric_id``: the bar, its revision (a listener's
theme or flow edit creates a new revision of every not-yet-committed bar) and
a hash of its content. Audio is rendered per lyric line and fetched by that id.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from math import isfinite
from types import MappingProxyType
from typing import Mapping

from streammuse.domain.rap.flow import FlowTemplate, materialize_flow
from streammuse.domain.rap.models import ScheduledSyllable
from streammuse.domain.rap.timeline import validate_live_bpm
from streammuse.domain.rap.remote_chunk import (
    _canonical_json_bytes,
    _flow_from_payload,
    _flow_payload,
    _freeze_json,
    _is_int,
    _is_real,
    _keys,
    _mapping,
    _nonblank,
    _scheduled_from_payload,
    _scheduled_payload,
    _thaw,
)

LYRIC_STREAM_SCHEMA_VERSION = "streammuse.rap_lyric_stream.v3"
BAR_AUDIO_SCHEMA_VERSION = "streammuse.rap_bar_audio.v3"
BAR_AUDIO_SAMPLE_RATE_HZ = 24_000
TEMPO_UPDATES_CAPABILITY = "tempo_updates_v1"
BOUNDED_TURNS_CAPABILITY = "bounded_turns_v1"
LYRIC_STREAM_CAPABILITIES = (TEMPO_UPDATES_CAPABILITY, BOUNDED_TURNS_CAPABILITY)
# A one-bar MOSS source is about 2.7 s at 90 BPM; a generous bound still keeps
# a corrupt manifest from claiming a huge buffer.
MAX_BAR_SOURCE_SECONDS = 15.0
_MAX_TEXT_CHARS = 512
_SHA256_HEX = frozenset("0123456789abcdef")


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _SHA256_HEX


def _require_4_4(template: FlowTemplate) -> FlowTemplate:
    if not isinstance(template, FlowTemplate):
        raise ValueError("flow_template must be a FlowTemplate")
    if template.ticks_per_beat != 4 or template.beats_per_bar != 4:
        raise ValueError("lyric stream flow templates must use 4/4 timing")
    return _flow_from_payload(_flow_payload(template))


@dataclass(frozen=True)
class BarLyricRecord:
    """One chosen lyric line for one bar, at one revision."""

    session_id: str
    bar: int
    revision: int
    topic: str
    flow_template: FlowTemplate
    text: str
    scheduled: tuple[ScheduledSyllable, ...]
    score: float
    provenance: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not _nonblank(self.session_id):
            raise ValueError("lyric session_id must be non-empty")
        for name in ("bar", "revision"):
            value = getattr(self, name)
            if not _is_int(value) or value < 0:
                raise ValueError(f"lyric {name} must be a non-negative integer")
        if not _nonblank(self.topic):
            raise ValueError("lyric topic must be non-empty")
        if not isinstance(self.text, str) or not self.text.strip() or len(self.text) > _MAX_TEXT_CHARS:
            raise ValueError("lyric text must be non-empty and bounded")
        object.__setattr__(self, "flow_template", _require_4_4(self.flow_template))
        if not isinstance(self.scheduled, tuple) or not self.scheduled:
            raise ValueError("lyric scheduled syllables must be a non-empty tuple")
        expected_slots = materialize_flow(self.flow_template, self.bar)
        if tuple(item.slot for item in self.scheduled) != expected_slots:
            raise ValueError("lyric scheduled syllables must fill the flow's slots for this bar")
        if not _is_real(self.score):
            raise ValueError("lyric score must be finite")
        object.__setattr__(self, "provenance", _freeze_json(dict(self.provenance), "lyric provenance"))

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": LYRIC_STREAM_SCHEMA_VERSION,
            "session_id": self.session_id,
            "bar": self.bar,
            "revision": self.revision,
            "topic": self.topic,
            "flow_template": _flow_payload(self.flow_template),
            "text": self.text,
            "scheduled": [_scheduled_payload(item) for item in self.scheduled],
        }

    @property
    def lyric_id(self) -> str:
        return hashlib.sha256(_canonical_json_bytes(self.identity_payload())).hexdigest()

    def to_payload(self) -> dict[str, object]:
        return {
            **self.identity_payload(),
            "lyric_id": self.lyric_id,
            "score": self.score,
            "provenance": _thaw(self.provenance),
        }

    @classmethod
    def from_payload(cls, value: object) -> "BarLyricRecord":
        payload = _mapping(value, "lyric record")
        _keys(
            payload,
            {
                "schema_version",
                "session_id",
                "bar",
                "revision",
                "topic",
                "flow_template",
                "text",
                "scheduled",
                "lyric_id",
                "score",
                "provenance",
            },
            "lyric record",
        )
        if payload["schema_version"] != LYRIC_STREAM_SCHEMA_VERSION:
            raise ValueError("unsupported lyric stream schema version")
        scheduled = payload["scheduled"]
        if not isinstance(scheduled, list):
            raise ValueError("lyric scheduled must be an array")
        record = cls(
            payload["session_id"],  # type: ignore[arg-type]
            payload["bar"],  # type: ignore[arg-type]
            payload["revision"],  # type: ignore[arg-type]
            payload["topic"],  # type: ignore[arg-type]
            _flow_from_payload(payload["flow_template"]),
            payload["text"],  # type: ignore[arg-type]
            tuple(_scheduled_from_payload(item) for item in scheduled),
            payload["score"],  # type: ignore[arg-type]
            _mapping(payload["provenance"], "lyric provenance"),
        )
        if record.lyric_id != payload["lyric_id"]:
            raise ValueError("lyric_id does not match the lyric content")
        return record


@dataclass(frozen=True)
class PlanEdit:
    """A listener's change to the theme and/or flow from ``from_bar`` onward."""

    edit_seq: int
    from_bar: int
    topic: str | None = None
    flow_template: FlowTemplate | None = None

    def __post_init__(self) -> None:
        for name in ("edit_seq", "from_bar"):
            value = getattr(self, name)
            if not _is_int(value) or value < 0:
                raise ValueError(f"plan edit {name} must be a non-negative integer")
        if self.topic is None and self.flow_template is None:
            raise ValueError("plan edit must change the topic or the flow template")
        if self.topic is not None and not _nonblank(self.topic):
            raise ValueError("plan edit topic must be non-empty")
        if self.flow_template is not None:
            object.__setattr__(self, "flow_template", _require_4_4(self.flow_template))

    def to_payload(self) -> dict[str, object]:
        return {
            "edit_seq": self.edit_seq,
            "from_bar": self.from_bar,
            "topic": self.topic,
            "flow_template": None if self.flow_template is None else _flow_payload(self.flow_template),
        }

    @classmethod
    def from_payload(cls, value: object) -> "PlanEdit":
        payload = _mapping(value, "plan edit")
        _keys(payload, {"edit_seq", "from_bar", "topic", "flow_template"}, "plan edit")
        flow = payload["flow_template"]
        return cls(
            payload["edit_seq"],  # type: ignore[arg-type]
            payload["from_bar"],  # type: ignore[arg-type]
            payload["topic"],  # type: ignore[arg-type]
            None if flow is None else _flow_from_payload(flow),
        )


@dataclass(frozen=True)
class TempoUpdate:
    """Tempo/render identity changes independently of lyric content identity."""

    edit_seq: int
    from_bar: int
    bpm: float

    def __post_init__(self) -> None:
        for name in ("edit_seq", "from_bar"):
            value = getattr(self, name)
            if not _is_int(value) or value < 0:
                raise ValueError(f"tempo update {name} must be a non-negative integer")
        validate_live_bpm(self.bpm)

    def to_payload(self) -> dict[str, object]:
        return {"edit_seq": self.edit_seq, "from_bar": self.from_bar, "bpm": self.bpm}

    @classmethod
    def from_payload(cls, value: object) -> "TempoUpdate":
        payload = _mapping(value, "tempo update")
        _keys(payload, {"edit_seq", "from_bar", "bpm"}, "tempo update")
        return cls(payload["edit_seq"], payload["from_bar"], payload["bpm"])


@dataclass(frozen=True)
class BarAudioManifest:
    """The raw MOSS line for one lyric record plus its measured syllable onsets."""

    lyric_id: str
    bar: int
    tempo_bpm: float
    frame_count: int
    source_sha256: str
    source_onsets: tuple[float, ...]
    onset_confidence: tuple[float, ...]
    stage_timings_ms: Mapping[str, float]
    diagnostics: Mapping[str, object] = field(default_factory=dict)
    sample_rate_hz: int = BAR_AUDIO_SAMPLE_RATE_HZ
    schema_version: str = BAR_AUDIO_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != BAR_AUDIO_SCHEMA_VERSION:
            raise ValueError("unsupported bar audio schema version")
        if not _is_sha256(self.lyric_id) or not _is_sha256(self.source_sha256):
            raise ValueError("bar audio ids must be lowercase SHA-256 hex digests")
        if not _is_int(self.bar) or self.bar < 0:
            raise ValueError("bar audio bar must be a non-negative integer")
        if not _is_real(self.tempo_bpm) or self.tempo_bpm <= 0:
            raise ValueError("bar audio tempo must be positive")
        if self.sample_rate_hz != BAR_AUDIO_SAMPLE_RATE_HZ:
            raise ValueError("bar audio must use the 24 kHz contract rate")
        if (
            not _is_int(self.frame_count)
            or self.frame_count <= 0
            or self.frame_count > MAX_BAR_SOURCE_SECONDS * self.sample_rate_hz
        ):
            raise ValueError("bar audio frame count is out of range")
        onsets = tuple(self.source_onsets)
        confidence = tuple(self.onset_confidence)
        duration = self.frame_count / self.sample_rate_hz
        if (
            not onsets
            or len(onsets) != len(confidence)
            or any(not _is_real(item) or item < 0 or item >= duration for item in onsets)
            or any(not _is_real(item) or not 0 <= item <= 1 for item in confidence)
        ):
            raise ValueError("bar audio onsets and confidences must be consistent and in range")
        object.__setattr__(self, "source_onsets", onsets)
        object.__setattr__(self, "onset_confidence", confidence)
        timings = dict(self.stage_timings_ms)
        if any(
            not isinstance(name, str) or not _is_real(value) or not isfinite(value) or value < 0
            for name, value in timings.items()
        ):
            raise ValueError("bar audio stage timings must be finite and non-negative")
        object.__setattr__(self, "stage_timings_ms", MappingProxyType(timings))
        object.__setattr__(self, "diagnostics", _freeze_json(dict(self.diagnostics), "bar audio diagnostics"))

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "lyric_id": self.lyric_id,
            "bar": self.bar,
            "tempo_bpm": self.tempo_bpm,
            "sample_rate_hz": self.sample_rate_hz,
            "frame_count": self.frame_count,
            "source_sha256": self.source_sha256,
            "source_onsets": list(self.source_onsets),
            "onset_confidence": list(self.onset_confidence),
            "stage_timings_ms": dict(self.stage_timings_ms),
            "diagnostics": _thaw(self.diagnostics),
        }

    def canonical_json_bytes(self) -> bytes:
        return _canonical_json_bytes(self.to_payload())

    @classmethod
    def from_payload(cls, value: object) -> "BarAudioManifest":
        payload = _mapping(value, "bar audio manifest")
        _keys(
            payload,
            {
                "schema_version",
                "lyric_id",
                "bar",
                "tempo_bpm",
                "sample_rate_hz",
                "frame_count",
                "source_sha256",
                "source_onsets",
                "onset_confidence",
                "stage_timings_ms",
                "diagnostics",
            },
            "bar audio manifest",
        )
        for name in ("source_onsets", "onset_confidence"):
            if not isinstance(payload[name], list):
                raise ValueError(f"bar audio {name} must be an array")
        return cls(
            lyric_id=payload["lyric_id"],  # type: ignore[arg-type]
            bar=payload["bar"],  # type: ignore[arg-type]
            tempo_bpm=payload["tempo_bpm"],  # type: ignore[arg-type]
            frame_count=payload["frame_count"],  # type: ignore[arg-type]
            source_sha256=payload["source_sha256"],  # type: ignore[arg-type]
            source_onsets=tuple(payload["source_onsets"]),  # type: ignore[arg-type]
            onset_confidence=tuple(payload["onset_confidence"]),  # type: ignore[arg-type]
            stage_timings_ms=_mapping(payload["stage_timings_ms"], "bar audio timings"),  # type: ignore[arg-type]
            diagnostics=_mapping(payload["diagnostics"], "bar audio diagnostics"),
            sample_rate_hz=payload["sample_rate_hz"],  # type: ignore[arg-type]
            schema_version=payload["schema_version"],  # type: ignore[arg-type]
        )
