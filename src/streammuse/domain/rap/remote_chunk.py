"""Immutable transport contracts for remote two-bar rap rendering."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from math import isfinite
from types import MappingProxyType
from typing import Mapping

from streammuse.domain.rap.audio import PreparedRapBar
from streammuse.domain.rap.flow import FlowProvenance, FlowSlot, FlowTemplate, materialize_flow
from streammuse.domain.rap.models import BeatSlot, ScheduledSyllable, Syllable


REMOTE_CHUNK_SCHEMA_VERSION = "streammuse.rap_chunk.v1"
REMOTE_CHUNK_SAMPLE_RATE_HZ = 24_000
MAX_REMOTE_DIAGNOSTIC_SUMMARIES = 8


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _is_int(value: object) -> bool:
    return type(value) is int


def _is_real(value: object) -> bool:
    return type(value) in (int, float) and isfinite(value)


def _nonblank(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _freeze_json(value: object, name: str) -> object:
    """Freeze a recursively JSON-safe value before it becomes wire diagnostics."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if _is_real(value):
        return value
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise ValueError(f"{name} object keys must be strings")
        return MappingProxyType({key: _freeze_json(item, name) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item, name) for item in value)
    raise ValueError(f"{name} must contain only finite JSON values")


def _thaw(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _keys(payload: Mapping[str, object], expected: set[str], name: str) -> None:
    if set(payload) != expected:
        raise ValueError(f"{name} payload keys must be exactly {sorted(expected)}")


def _flow_payload(flow: FlowTemplate) -> dict[str, object]:
    return {
        "template_id": flow.template_id,
        "name": flow.name,
        "ticks_per_beat": flow.ticks_per_beat,
        "beats_per_bar": flow.beats_per_bar,
        "slots": [
            {
                "tick_in_bar": item.tick_in_bar,
                "duration_ticks": item.duration_ticks,
                "target_stress": item.target_stress,
                "boundary_strength": item.boundary_strength,
                "rhyme_group": item.rhyme_group,
            }
            for item in flow.slots
        ],
        "provenance": {
            "kind": flow.provenance.kind,
            "source": flow.provenance.source,
            "source_hash": flow.provenance.source_hash,
            "quantization_error_ticks": flow.provenance.quantization_error_ticks,
        },
    }


def _flow_from_payload(value: object) -> FlowTemplate:
    payload = _mapping(value, "flow_template")
    _keys(payload, {"template_id", "name", "ticks_per_beat", "beats_per_bar", "slots", "provenance"}, "flow_template")
    slots = payload["slots"]
    if not isinstance(slots, list):
        raise ValueError("flow_template slots must be an array")
    if not _nonblank(payload["template_id"]) or not _nonblank(payload["name"]):
        raise ValueError("flow template identity must be non-empty strings")
    if not _is_int(payload["ticks_per_beat"]) or not _is_int(payload["beats_per_bar"]):
        raise ValueError("flow template timing must use integers")
    flow_slots = []
    for value in slots:
        slot = _mapping(value, "flow slot")
        _keys(slot, {"tick_in_bar", "duration_ticks", "target_stress", "boundary_strength", "rhyme_group"}, "flow slot")
        if (
            not _is_int(slot["tick_in_bar"])
            or not _is_int(slot["duration_ticks"])
            or not _is_real(slot["target_stress"])
            or not _is_int(slot["boundary_strength"])
            or (slot["rhyme_group"] is not None and not _nonblank(slot["rhyme_group"]))
        ):
            raise ValueError("flow slot primitives are invalid")
        flow_slots.append(FlowSlot(**slot))  # type: ignore[arg-type]
    provenance = _mapping(payload["provenance"], "flow provenance")
    _keys(provenance, {"kind", "source", "source_hash", "quantization_error_ticks"}, "flow provenance")
    if (
        not _nonblank(provenance["kind"])
        or not _nonblank(provenance["source"])
        or (provenance["source_hash"] is not None and not _nonblank(provenance["source_hash"]))
        or not _is_real(provenance["quantization_error_ticks"])
    ):
        raise ValueError("flow provenance primitives are invalid")
    return FlowTemplate(
        template_id=payload["template_id"],  # type: ignore[arg-type]
        name=payload["name"],  # type: ignore[arg-type]
        ticks_per_beat=payload["ticks_per_beat"],  # type: ignore[arg-type]
        beats_per_bar=payload["beats_per_bar"],  # type: ignore[arg-type]
        slots=tuple(flow_slots),
        provenance=FlowProvenance(**provenance),  # type: ignore[arg-type]
    )


@dataclass(frozen=True)
class RemoteCandidatePolicy:
    profile: str
    initial_candidates: int
    rescue_candidates: int
    maximum_candidates: int
    minimum_valid_candidates: int
    minimum_score: float
    render_reserve_ms: int

    def __post_init__(self) -> None:
        if not isinstance(self.profile, str) or not self.profile:
            raise ValueError("policy profile must be a non-empty string")
        values = (self.initial_candidates, self.rescue_candidates, self.maximum_candidates, self.minimum_valid_candidates)
        if any(not _is_int(value) or value < 0 for value in values):
            raise ValueError("candidate policy counts must be non-negative integers")
        if self.maximum_candidates <= 0 or self.minimum_valid_candidates <= 0:
            raise ValueError("maximum and minimum valid candidate counts must be positive")
        if self.minimum_valid_candidates > self.maximum_candidates:
            raise ValueError("minimum_valid_candidates cannot exceed maximum_candidates")
        if self.initial_candidates + self.rescue_candidates > self.maximum_candidates:
            raise ValueError("candidate policy waves cannot exceed maximum_candidates")
        if not _is_int(self.render_reserve_ms) or self.render_reserve_ms < 0:
            raise ValueError("render_reserve_ms must be a non-negative integer")
        if not _is_real(self.minimum_score):
            raise ValueError("minimum_score must be finite")

    @classmethod
    def realtime_default(cls) -> RemoteCandidatePolicy:
        return cls("realtime_default", 4, 2, 6, 1, 0.0, 1_500)

    def to_payload(self) -> dict[str, object]:
        return {
            "profile": self.profile,
            "initial_candidates": self.initial_candidates,
            "rescue_candidates": self.rescue_candidates,
            "maximum_candidates": self.maximum_candidates,
            "minimum_valid_candidates": self.minimum_valid_candidates,
            "minimum_score": self.minimum_score,
            "render_reserve_ms": self.render_reserve_ms,
        }

    @classmethod
    def from_payload(cls, value: object) -> RemoteCandidatePolicy:
        payload = _mapping(value, "candidate policy")
        _keys(payload, set(cls("x", 1, 0, 1, 1, 0.0, 0).to_payload()), "candidate policy")
        return cls(**payload)  # type: ignore[arg-type]


@dataclass(frozen=True)
class RemoteCandidateStats:
    requested_count: int
    parseable_count: int
    valid_count: int
    selectable_count: int
    top_candidates: tuple[Mapping[str, object], ...]
    rejections: tuple[Mapping[str, object], ...]

    def __post_init__(self) -> None:
        counts = (self.requested_count, self.parseable_count, self.valid_count, self.selectable_count)
        if any(not _is_int(value) or value < 0 for value in counts):
            raise ValueError("candidate counts must be non-negative integers")
        if not self.selectable_count <= self.valid_count <= self.parseable_count <= self.requested_count:
            raise ValueError("candidate counts must be monotonically decreasing")
        if len(self.top_candidates) > min(MAX_REMOTE_DIAGNOSTIC_SUMMARIES, self.selectable_count):
            raise ValueError("top_candidates must not exceed selectable_count")
        if len(self.rejections) > min(MAX_REMOTE_DIAGNOSTIC_SUMMARIES, self.requested_count - self.selectable_count):
            raise ValueError("rejections must not exceed rejected candidates")
        for name, values, validator in (
            ("top_candidates", self.top_candidates, _validate_top_candidate),
            ("rejections", self.rejections, _validate_rejection),
        ):
            if not isinstance(values, tuple) or len(values) > MAX_REMOTE_DIAGNOSTIC_SUMMARIES:
                raise ValueError(f"{name} must be a bounded tuple")
            if not all(isinstance(value, Mapping) for value in values):
                raise ValueError(f"{name} must contain mappings")
            object.__setattr__(self, name, tuple(validator(value) for value in values))

    def to_payload(self) -> dict[str, object]:
        return {
            "requested_count": self.requested_count,
            "parseable_count": self.parseable_count,
            "valid_count": self.valid_count,
            "selectable_count": self.selectable_count,
            "top_candidates": [_thaw(value) for value in self.top_candidates],
            "rejections": [_thaw(value) for value in self.rejections],
        }

    @classmethod
    def from_payload(cls, value: object) -> RemoteCandidateStats:
        payload = _mapping(value, "candidate stats")
        _keys(payload, {"requested_count", "parseable_count", "valid_count", "selectable_count", "top_candidates", "rejections"}, "candidate stats")
        top_candidates, rejections = payload["top_candidates"], payload["rejections"]
        if not isinstance(top_candidates, list) or not isinstance(rejections, list):
            raise ValueError("candidate summaries and rejections must be arrays")
        return cls(
            payload["requested_count"], payload["parseable_count"], payload["valid_count"], payload["selectable_count"],
            tuple(_mapping(item, "candidate summary") for item in top_candidates),
            tuple(_mapping(item, "candidate rejection") for item in rejections),
        )  # type: ignore[arg-type]


def _validate_component_scores(value: object, name: str) -> Mapping[str, object]:
    scores = _mapping(value, name)
    if not scores or not all(_nonblank(key) and _is_real(score) for key, score in scores.items()):
        raise ValueError(f"{name} must be non-empty finite named scores")
    return _freeze_json(scores, name)  # type: ignore[return-value]


def _validate_top_candidate(value: Mapping[str, object]) -> Mapping[str, object]:
    _keys(value, {"bar", "candidate_id", "text", "score", "component_scores", "source_order"}, "top candidate")
    if (
        not _is_int(value["bar"])
        or value["bar"] < 0
        or not _nonblank(value["candidate_id"])
        or not _nonblank(value["text"])
        or not _is_real(value["score"])
        or not _is_int(value["source_order"])
        or value["source_order"] < 0
    ):
        raise ValueError("top candidate primitives are invalid")
    return _freeze_json({**value, "component_scores": _validate_component_scores(value["component_scores"], "top candidate component_scores")}, "top candidate")  # type: ignore[return-value]


def _validate_rejection(value: Mapping[str, object]) -> Mapping[str, object]:
    _keys(value, {"bar", "candidate_id", "text", "reasons", "source_order"}, "candidate rejection")
    reasons = value["reasons"]
    if (
        not _is_int(value["bar"])
        or value["bar"] < 0
        or not _nonblank(value["candidate_id"])
        or not isinstance(value["text"], str)
        or not _is_int(value["source_order"])
        or value["source_order"] < 0
        or not isinstance(reasons, (list, tuple))
        or not reasons
        or not all(_nonblank(reason) for reason in reasons)
    ):
        raise ValueError("candidate rejection primitives are invalid")
    return _freeze_json(value, "candidate rejection")  # type: ignore[return-value]


_REQUIRED_STAGE_TIMINGS = {"generation", "evaluation", "moss", "mfa", "warp", "packaging", "total"}
_REQUIRED_ALIGNMENT_DIAGNOSTICS = {"fallback_counts", "source_anchors", "target_anchors", "local_warp_ratios"}
_REQUIRED_AUDIO_DIAGNOSTICS = {"sample_rate_hz", "frame_count", "duration_seconds", "peak"}


@dataclass(frozen=True)
class RemoteRapChunkDiagnostics:
    accepted_request_budget_ms: int
    resolved_policy: RemoteCandidatePolicy
    candidate_stats: RemoteCandidateStats
    stage_timings_ms: Mapping[str, float]
    alignment_diagnostics: Mapping[str, object]
    audio_diagnostics: Mapping[str, object]
    model_tool_versions: Mapping[str, str]
    warnings: tuple[str, ...]

    def __post_init__(self) -> None:
        if not _is_int(self.accepted_request_budget_ms) or self.accepted_request_budget_ms <= 0:
            raise ValueError("accepted_request_budget_ms must be a positive integer")
        if not isinstance(self.resolved_policy, RemoteCandidatePolicy):
            raise ValueError("resolved_policy must be a RemoteCandidatePolicy")
        if not isinstance(self.candidate_stats, RemoteCandidateStats):
            raise ValueError("candidate_stats must be RemoteCandidateStats")
        if not isinstance(self.stage_timings_ms, Mapping) or set(self.stage_timings_ms) != _REQUIRED_STAGE_TIMINGS:
            raise ValueError("stage_timings_ms must contain every required stage")
        if any(not _is_real(value) or value < 0 for value in self.stage_timings_ms.values()):
            raise ValueError("stage timings must be finite non-negative numbers")
        if any(self.stage_timings_ms["total"] < value for key, value in self.stage_timings_ms.items() if key != "total"):
            raise ValueError("total stage timing must cover every component stage")
        alignment = _mapping(self.alignment_diagnostics, "alignment_diagnostics")
        if set(alignment) != _REQUIRED_ALIGNMENT_DIAGNOSTICS:
            raise ValueError("alignment_diagnostics must contain every required field")
        fallback_counts = _mapping(alignment["fallback_counts"], "alignment fallback_counts")
        source_anchors = alignment["source_anchors"]
        target_anchors = alignment["target_anchors"]
        ratios = alignment["local_warp_ratios"]
        if not all(_nonblank(key) and _is_int(value) and value >= 0 for key, value in fallback_counts.items()):
            raise ValueError("alignment fallback counts must be non-negative integers")
        if (
            not isinstance(source_anchors, (list, tuple))
            or not isinstance(target_anchors, (list, tuple))
            or not source_anchors
            or len(source_anchors) != len(target_anchors)
            or any(not _is_real(item) for item in source_anchors)
            or any(not _is_real(item) for item in target_anchors)
            or not isinstance(ratios, (list, tuple))
            or any(not _is_real(item) or item <= 0 for item in ratios)
        ):
            raise ValueError("alignment anchors and ratios must be finite consistent arrays")
        audio = _mapping(self.audio_diagnostics, "audio_diagnostics")
        if set(audio) != _REQUIRED_AUDIO_DIAGNOSTICS:
            raise ValueError("audio_diagnostics must contain every required field")
        if not _is_int(audio["sample_rate_hz"]) or audio["sample_rate_hz"] <= 0 or not _is_int(audio["frame_count"]) or audio["frame_count"] <= 0:
            raise ValueError("audio diagnostics require positive sample rate and frame count")
        if not _is_real(audio["duration_seconds"]) or not _is_real(audio["peak"]) or not 0 <= audio["peak"] <= 1:
            raise ValueError("audio duration and peak must be finite normalized numbers")
        sample_tolerance = 1 / audio["sample_rate_hz"]
        if abs(audio["duration_seconds"] - audio["frame_count"] / audio["sample_rate_hz"]) > sample_tolerance:
            raise ValueError("audio duration must match frame count within one sample")
        required_versions = {"moss", "mfa", "rubberband"}
        if not isinstance(self.model_tool_versions, Mapping) or not required_versions.issubset(self.model_tool_versions) or not all(_nonblank(key) and _nonblank(value) for key, value in self.model_tool_versions.items()):
            raise ValueError("model_tool_versions must include non-empty moss, mfa, and rubberband versions")
        if not isinstance(self.warnings, tuple) or not all(isinstance(item, str) for item in self.warnings):
            raise ValueError("warnings must be a tuple of strings")
        object.__setattr__(self, "stage_timings_ms", _freeze_json(self.stage_timings_ms, "stage_timings_ms"))
        object.__setattr__(self, "alignment_diagnostics", _freeze_json(alignment, "alignment_diagnostics"))
        object.__setattr__(self, "audio_diagnostics", _freeze_json(audio, "audio_diagnostics"))
        object.__setattr__(self, "model_tool_versions", _freeze_json(self.model_tool_versions, "model_tool_versions"))
        object.__setattr__(self, "warnings", _freeze_json(self.warnings, "warnings"))

    def to_payload(self) -> dict[str, object]:
        return {
            "accepted_request_budget_ms": self.accepted_request_budget_ms,
            "resolved_policy": self.resolved_policy.to_payload(),
            "candidate_stats": self.candidate_stats.to_payload(),
            "stage_timings_ms": _thaw(self.stage_timings_ms),
            "alignment_diagnostics": _thaw(self.alignment_diagnostics),
            "audio_diagnostics": _thaw(self.audio_diagnostics),
            "model_tool_versions": _thaw(self.model_tool_versions),
            "warnings": list(self.warnings),
        }

    @classmethod
    def from_payload(cls, value: object) -> RemoteRapChunkDiagnostics:
        payload = _mapping(value, "remote chunk diagnostics")
        _keys(payload, {"accepted_request_budget_ms", "resolved_policy", "candidate_stats", "stage_timings_ms", "alignment_diagnostics", "audio_diagnostics", "model_tool_versions", "warnings"}, "remote chunk diagnostics")
        warnings = payload["warnings"]
        if not isinstance(warnings, list):
            raise ValueError("warnings must be an array")
        return cls(
            payload["accepted_request_budget_ms"],
            RemoteCandidatePolicy.from_payload(payload["resolved_policy"]),
            RemoteCandidateStats.from_payload(payload["candidate_stats"]),
            _mapping(payload["stage_timings_ms"], "stage_timings_ms"),  # type: ignore[arg-type]
            _mapping(payload["alignment_diagnostics"], "alignment_diagnostics"),
            _mapping(payload["audio_diagnostics"], "audio_diagnostics"),
            _mapping(payload["model_tool_versions"], "model_tool_versions"),  # type: ignore[arg-type]
            tuple(warnings),  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class RemoteRapBarRequest:
    bar: int
    topic: str
    flow_template: FlowTemplate

    def __post_init__(self) -> None:
        if not _is_int(self.bar) or self.bar < 0:
            raise ValueError("bar must be a non-negative integer")
        if not isinstance(self.topic, str) or not self.topic:
            raise ValueError("topic must be a non-empty string")
        if not isinstance(self.flow_template, FlowTemplate):
            raise ValueError("flow_template must be a FlowTemplate")
        try:
            object.__setattr__(self, "flow_template", _flow_from_payload(_flow_payload(self.flow_template)))
        except ValueError as error:
            if self.flow_template.ticks_per_beat != 4 or self.flow_template.beats_per_bar != 4:
                raise ValueError("remote chunk flow templates must use 4/4 timing") from error
            raise

    def to_payload(self) -> dict[str, object]:
        return {"bar": self.bar, "topic": self.topic, "flow_template": _flow_payload(self.flow_template)}

    @classmethod
    def from_payload(cls, value: object) -> RemoteRapBarRequest:
        payload = _mapping(value, "bar request")
        _keys(payload, {"bar", "topic", "flow_template"}, "bar request")
        return cls(payload["bar"], payload["topic"], _flow_from_payload(payload["flow_template"]))  # type: ignore[arg-type]


@dataclass(frozen=True)
class RemoteRapChunkRequest:
    schema_version: str
    session_id: str
    request_id: str
    chunk_index: int
    bars: tuple[RemoteRapBarRequest, RemoteRapBarRequest]
    tempo_bpm: float
    output_sample_rate_hz: int
    expected_frame_count: int
    remaining_budget_ms: int
    policy: RemoteCandidatePolicy
    context_lines: tuple[str, ...]
    seed: int

    def __post_init__(self) -> None:
        if self.schema_version != REMOTE_CHUNK_SCHEMA_VERSION:
            raise ValueError("unsupported remote chunk schema version")
        if not isinstance(self.session_id, str) or not self.session_id:
            raise ValueError("session_id must be a non-empty string")
        if not _is_int(self.chunk_index) or self.chunk_index < 0:
            raise ValueError("chunk_index must be a non-negative integer")
        if not isinstance(self.bars, tuple) or len(self.bars) != 2 or not all(isinstance(bar, RemoteRapBarRequest) for bar in self.bars):
            raise ValueError("remote chunk requests require two consecutive bars")
        if self.bars[1].bar != self.bars[0].bar + 1:
            raise ValueError("remote chunk requests require two consecutive bars")
        if any(bar.flow_template.ticks_per_beat != 4 or bar.flow_template.beats_per_bar != 4 for bar in self.bars):
            raise ValueError("remote chunk flow templates must use 4/4 timing")
        if not _is_real(self.tempo_bpm) or self.tempo_bpm <= 0:
            raise ValueError("tempo_bpm must be positive and finite")
        if not _is_int(self.output_sample_rate_hz) or self.output_sample_rate_hz != REMOTE_CHUNK_SAMPLE_RATE_HZ:
            raise ValueError("output_sample_rate_hz must be 24000")
        if not _is_int(self.expected_frame_count) or self.expected_frame_count != self.frame_count_for(self.tempo_bpm, self.output_sample_rate_hz):
            raise ValueError("expected_frame_count must match the two-bar timing contract")
        if not _is_int(self.remaining_budget_ms) or self.remaining_budget_ms <= 0:
            raise ValueError("remaining_budget_ms must be a positive integer")
        if not isinstance(self.policy, RemoteCandidatePolicy):
            raise ValueError("policy must be a RemoteCandidatePolicy")
        if not isinstance(self.context_lines, tuple) or not all(isinstance(item, str) for item in self.context_lines):
            raise ValueError("context_lines must be a tuple of strings")
        if not _is_int(self.seed):
            raise ValueError("seed must be an integer")
        if self.request_id != self.request_id_for(self.identity_payload()):
            raise ValueError("request_id does not match the canonical request body")

    @staticmethod
    def frame_count_for(tempo_bpm: float, sample_rate_hz: int = REMOTE_CHUNK_SAMPLE_RATE_HZ) -> int:
        if not _is_real(tempo_bpm) or tempo_bpm <= 0 or not _is_int(sample_rate_hz) or sample_rate_hz <= 0:
            raise ValueError("tempo_bpm and sample_rate_hz must be positive finite wire numbers")
        return round(sample_rate_hz * 8 * 60 / tempo_bpm)

    @staticmethod
    def request_id_for(identity_payload: Mapping[str, object]) -> str:
        return hashlib.sha256(_canonical_json_bytes(identity_payload)).hexdigest()

    @classmethod
    def create(
        cls,
        *,
        session_id: str,
        chunk_index: int,
        bars: tuple[RemoteRapBarRequest, RemoteRapBarRequest],
        tempo_bpm: float,
        remaining_budget_ms: int,
        policy: RemoteCandidatePolicy,
        context_lines: tuple[str, ...],
        seed: int,
        output_sample_rate_hz: int = REMOTE_CHUNK_SAMPLE_RATE_HZ,
    ) -> RemoteRapChunkRequest:
        if not _is_real(tempo_bpm) or tempo_bpm <= 0:
            raise ValueError("tempo_bpm must be positive and finite")
        expected_frame_count = cls.frame_count_for(tempo_bpm, output_sample_rate_hz)
        identity = {
            "schema_version": REMOTE_CHUNK_SCHEMA_VERSION,
            "session_id": session_id,
            "chunk_index": chunk_index,
            "bars": [item.to_payload() for item in bars],
            "tempo_bpm": tempo_bpm,
            "output_sample_rate_hz": output_sample_rate_hz,
            "expected_frame_count": expected_frame_count,
            "policy": policy.to_payload(),
            "context_lines": list(context_lines),
            "seed": seed,
        }
        return cls(
            REMOTE_CHUNK_SCHEMA_VERSION, session_id, cls.request_id_for(identity), chunk_index, bars, tempo_bpm,
            output_sample_rate_hz, expected_frame_count, remaining_budget_ms, policy, context_lines, seed,
        )

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "session_id": self.session_id,
            "chunk_index": self.chunk_index,
            "bars": [item.to_payload() for item in self.bars],
            "tempo_bpm": self.tempo_bpm,
            "output_sample_rate_hz": self.output_sample_rate_hz,
            "expected_frame_count": self.expected_frame_count,
            "policy": self.policy.to_payload(),
            "context_lines": list(self.context_lines),
            "seed": self.seed,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self.identity_payload(), "request_id": self.request_id, "remaining_budget_ms": self.remaining_budget_ms}

    def canonical_json_bytes(self) -> bytes:
        return _canonical_json_bytes(self.to_payload())

    def transport_attempt(self) -> RemoteRapChunkTransportAttempt:
        """Capture immutable bytes for this request's initial transport attempt and retries."""
        return RemoteRapChunkTransportAttempt.from_request(self)

    @classmethod
    def from_payload(cls, value: object) -> RemoteRapChunkRequest:
        payload = _mapping(value, "remote chunk request")
        _keys(payload, {"schema_version", "session_id", "request_id", "chunk_index", "bars", "tempo_bpm", "output_sample_rate_hz", "expected_frame_count", "remaining_budget_ms", "policy", "context_lines", "seed"}, "remote chunk request")
        bars, lines = payload["bars"], payload["context_lines"]
        if not isinstance(bars, list) or len(bars) != 2 or not isinstance(lines, list):
            raise ValueError("remote chunk request bars and context_lines must be arrays")
        return cls(
            payload["schema_version"], payload["session_id"], payload["request_id"], payload["chunk_index"],
            (RemoteRapBarRequest.from_payload(bars[0]), RemoteRapBarRequest.from_payload(bars[1])),
            payload["tempo_bpm"], payload["output_sample_rate_hz"], payload["expected_frame_count"],
            payload["remaining_budget_ms"], RemoteCandidatePolicy.from_payload(payload["policy"]), tuple(lines), payload["seed"],
        )  # type: ignore[arg-type]


@dataclass(frozen=True)
class RemoteRapChunkTransportAttempt:
    """An immutable request body that must be reused verbatim for retries."""

    request_id: str
    body: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not self.request_id:
            raise ValueError("transport attempt request_id must be a non-empty string")
        if not isinstance(self.body, bytes):
            raise ValueError("transport attempt body must be bytes")
        try:
            request = RemoteRapChunkRequest.from_payload(json.loads(self.body.decode("utf-8")))
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as error:
            raise ValueError("transport attempt body must be a valid canonical request") from error
        if request.request_id != self.request_id or request.canonical_json_bytes() != self.body:
            raise ValueError("transport attempt body does not match its request identity")

    @classmethod
    def from_request(cls, request: RemoteRapChunkRequest) -> RemoteRapChunkTransportAttempt:
        if not isinstance(request, RemoteRapChunkRequest):
            raise ValueError("transport attempts require a RemoteRapChunkRequest")
        return cls(request.request_id, request.canonical_json_bytes())

    def retry_body(self) -> bytes:
        """Return the original canonical body, including its original budget, unchanged."""
        return self.body


def _scheduled_payload(item: ScheduledSyllable) -> dict[str, object]:
    return {
        "slot": item.slot.__dict__.copy(),
        "syllable": {
            "word": item.syllable.word,
            "index_in_word": item.syllable.index_in_word,
            "syllable_count": item.syllable.syllable_count,
            "stress": item.syllable.stress,
            "phonemes": list(item.syllable.phonemes),
            "analysis_source": item.syllable.analysis_source,
        },
    }


def _scheduled_from_payload(value: object) -> ScheduledSyllable:
    payload = _mapping(value, "scheduled syllable")
    _keys(payload, {"slot", "syllable"}, "scheduled syllable")
    slot, syllable = _mapping(payload["slot"], "scheduled slot"), _mapping(payload["syllable"], "scheduled syllable data")
    _keys(slot, {"bar", "tick", "beat", "tick_in_beat", "accent", "duration_ticks", "boundary_strength", "rhyme_group", "template_id", "slot_index"}, "scheduled slot")
    _keys(syllable, {"word", "index_in_word", "syllable_count", "stress", "phonemes", "analysis_source"}, "scheduled syllable data")
    phonemes = syllable["phonemes"]
    if not isinstance(phonemes, list):
        raise ValueError("scheduled syllable phonemes must be an array")
    return ScheduledSyllable(BeatSlot(**slot), Syllable(syllable["word"], syllable["index_in_word"], syllable["syllable_count"], syllable["stress"], tuple(phonemes), syllable["analysis_source"]))  # type: ignore[arg-type]


def _validate_scheduled_syllable(item: ScheduledSyllable, bar: int, template_id: str) -> None:
    if not isinstance(item, ScheduledSyllable) or not isinstance(item.slot, BeatSlot) or not isinstance(item.syllable, Syllable):
        raise ValueError("selected bar scheduled must contain ScheduledSyllable values")
    slot = item.slot
    if (
        not _is_int(slot.bar)
        or slot.bar != bar
        or not _is_int(slot.tick)
        or slot.tick < 0
        or not _is_int(slot.beat)
        or not 0 <= slot.beat < 4
        or not _is_int(slot.tick_in_beat)
        or not 0 <= slot.tick_in_beat < 4
        or slot.tick != slot.bar * 16 + slot.beat * 4 + slot.tick_in_beat
        or not _is_real(slot.accent)
        or not 0 <= slot.accent <= 1
        or not _is_int(slot.duration_ticks)
        or slot.duration_ticks <= 0
        or not _is_int(slot.boundary_strength)
        or not 0 <= slot.boundary_strength <= 5
        or (slot.rhyme_group is not None and not _nonblank(slot.rhyme_group))
        or slot.template_id != template_id
        or not _is_int(slot.slot_index)
        or slot.slot_index < 0
    ):
        raise ValueError("selected bar slot primitives are invalid")
    syllable = item.syllable
    if (
        not _nonblank(syllable.word)
        or not _is_int(syllable.index_in_word)
        or not _is_int(syllable.syllable_count)
        or syllable.syllable_count <= 0
        or not 0 <= syllable.index_in_word < syllable.syllable_count
        or not _is_int(syllable.stress)
        or not 0 <= syllable.stress <= 2
        or not isinstance(syllable.phonemes, tuple)
        or not all(_nonblank(phoneme) for phoneme in syllable.phonemes)
        or not _nonblank(syllable.analysis_source)
    ):
        raise ValueError("selected bar syllable primitives are invalid")


@dataclass(frozen=True)
class RemoteSelectedBar:
    bar: int
    text: str
    flow_template_id: str
    scheduled: tuple[ScheduledSyllable, ...]
    score: float
    diagnostics: Mapping[str, object] = MappingProxyType({})

    def __post_init__(self) -> None:
        if not _is_int(self.bar) or self.bar < 0:
            raise ValueError("selected bar must have a non-negative bar identity")
        if not _nonblank(self.text) or not _nonblank(self.flow_template_id):
            raise ValueError("selected bar identity must be valid")
        if not isinstance(self.scheduled, tuple) or not self.scheduled:
            raise ValueError("selected bar scheduled must be a tuple of ScheduledSyllable values")
        for item in self.scheduled:
            _validate_scheduled_syllable(item, self.bar, self.flow_template_id)
        ticks = [item.slot.tick for item in self.scheduled]
        if ticks != sorted(ticks) or len(ticks) != len(set(ticks)):
            raise ValueError("selected bar scheduled slots must be ordered and unique")
        if not _is_real(self.score):
            raise ValueError("selected bar score must be finite")
        if not isinstance(self.diagnostics, Mapping):
            raise ValueError("selected bar diagnostics must be a mapping")
        component_scores = self.diagnostics.get("component_scores")
        if not isinstance(component_scores, Mapping) or not component_scores:
            raise ValueError("selected bar diagnostics must include component_scores")
        _validate_component_scores(component_scores, "component_scores")
        object.__setattr__(self, "diagnostics", _freeze_json(self.diagnostics, "selected bar diagnostics"))

    @classmethod
    def create(cls, request: RemoteRapBarRequest, *, text: str, scheduled: tuple[ScheduledSyllable, ...], score: float, diagnostics: Mapping[str, object] | None = None) -> RemoteSelectedBar:
        if tuple(item.slot for item in scheduled) != materialize_flow(request.flow_template, request.bar):
            raise ValueError("selected bar identity must match the requested materialized flow")
        return cls(
            request.bar,
            text,
            request.flow_template.template_id,
            scheduled,
            score,
            {"component_scores": {"total": score}} if diagnostics is None else diagnostics,
        )

    def to_payload(self) -> dict[str, object]:
        return {"bar": self.bar, "text": self.text, "flow_template_id": self.flow_template_id, "scheduled": [_scheduled_payload(item) for item in self.scheduled], "score": self.score, "diagnostics": _thaw(self.diagnostics)}

    @classmethod
    def from_payload(cls, value: object) -> RemoteSelectedBar:
        payload = _mapping(value, "selected bar")
        _keys(payload, {"bar", "text", "flow_template_id", "scheduled", "score", "diagnostics"}, "selected bar")
        scheduled = payload["scheduled"]
        if not isinstance(scheduled, list):
            raise ValueError("selected bar scheduled must be an array")
        return cls(payload["bar"], payload["text"], payload["flow_template_id"], tuple(_scheduled_from_payload(item) for item in scheduled), payload["score"], _mapping(payload["diagnostics"], "selected bar diagnostics"))  # type: ignore[arg-type]


@dataclass(frozen=True)
class RemoteRapChunkManifest:
    request_id: str
    chunk_index: int
    tempo_bpm: float
    output_sample_rate_hz: int
    expected_frame_count: int
    selected_bars: tuple[RemoteSelectedBar, RemoteSelectedBar]
    diagnostics: RemoteRapChunkDiagnostics
    vocal_sha256: str
    schema_version: str = REMOTE_CHUNK_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != REMOTE_CHUNK_SCHEMA_VERSION or not isinstance(self.request_id, str) or not self.request_id:
            raise ValueError("manifest schema_version and request_id must be valid")
        if not _is_int(self.chunk_index) or self.chunk_index < 0 or not _is_real(self.tempo_bpm) or self.tempo_bpm <= 0:
            raise ValueError("manifest chunk_index and tempo_bpm must be valid")
        if (
            not _is_int(self.output_sample_rate_hz)
            or not _is_int(self.expected_frame_count)
            or self.output_sample_rate_hz != REMOTE_CHUNK_SAMPLE_RATE_HZ
            or self.expected_frame_count != RemoteRapChunkRequest.frame_count_for(self.tempo_bpm)
        ):
            raise ValueError("manifest timing does not match the two-bar 24 kHz contract")
        if not isinstance(self.selected_bars, tuple) or len(self.selected_bars) != 2 or not all(isinstance(item, RemoteSelectedBar) for item in self.selected_bars) or self.selected_bars[1].bar != self.selected_bars[0].bar + 1:
            raise ValueError("manifest requires two consecutive selected bars")
        if not isinstance(self.diagnostics, RemoteRapChunkDiagnostics):
            raise ValueError("manifest diagnostics must be RemoteRapChunkDiagnostics")
        if self.diagnostics.audio_diagnostics["sample_rate_hz"] != self.output_sample_rate_hz or self.diagnostics.audio_diagnostics["frame_count"] != self.expected_frame_count:
            raise ValueError("manifest audio diagnostics must match the declared audio format")
        if not isinstance(self.vocal_sha256, str) or len(self.vocal_sha256) != 64 or any(item not in "0123456789abcdef" for item in self.vocal_sha256):
            raise ValueError("manifest vocal_sha256 must be a lowercase SHA-256 hex digest")

    def to_payload(self) -> dict[str, object]:
        return {"schema_version": self.schema_version, "request_id": self.request_id, "chunk_index": self.chunk_index, "tempo_bpm": self.tempo_bpm, "output_sample_rate_hz": self.output_sample_rate_hz, "expected_frame_count": self.expected_frame_count, "selected_bars": [item.to_payload() for item in self.selected_bars], "diagnostics": self.diagnostics.to_payload(), "vocal_sha256": self.vocal_sha256}

    def canonical_json_bytes(self) -> bytes:
        return _canonical_json_bytes(self.to_payload())

    @classmethod
    def from_payload(cls, value: object) -> RemoteRapChunkManifest:
        payload = _mapping(value, "remote chunk manifest")
        _keys(payload, {"schema_version", "request_id", "chunk_index", "tempo_bpm", "output_sample_rate_hz", "expected_frame_count", "selected_bars", "diagnostics", "vocal_sha256"}, "remote chunk manifest")
        bars = payload["selected_bars"]
        if not isinstance(bars, list) or len(bars) != 2:
            raise ValueError("manifest selected_bars must be a two-item array")
        return cls(payload["request_id"], payload["chunk_index"], payload["tempo_bpm"], payload["output_sample_rate_hz"], payload["expected_frame_count"], (RemoteSelectedBar.from_payload(bars[0]), RemoteSelectedBar.from_payload(bars[1])), RemoteRapChunkDiagnostics.from_payload(payload["diagnostics"]), payload["vocal_sha256"], payload["schema_version"])  # type: ignore[arg-type]


@dataclass(frozen=True)
class PreparedRapChunk:
    request_id: str
    chunk_index: int
    renderer: str
    bars: tuple[PreparedRapBar, PreparedRapBar]
    diagnostics: Mapping[str, object]

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not self.request_id or not isinstance(self.chunk_index, int) or self.chunk_index < 0 or not isinstance(self.renderer, str) or not self.renderer:
            raise ValueError("prepared chunk identity must be valid")
        if not isinstance(self.bars, tuple) or len(self.bars) != 2 or not all(isinstance(item, PreparedRapBar) for item in self.bars) or self.bars[1].bar != self.bars[0].bar + 1:
            raise ValueError("prepared chunk requires two consecutive bars")
        if not isinstance(self.diagnostics, Mapping):
            raise ValueError("prepared chunk diagnostics must be a mapping")
        object.__setattr__(self, "diagnostics", _freeze(self.diagnostics))
