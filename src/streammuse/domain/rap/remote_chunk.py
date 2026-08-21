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


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


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
    flow_slots = []
    for value in slots:
        slot = _mapping(value, "flow slot")
        _keys(slot, {"tick_in_bar", "duration_ticks", "target_stress", "boundary_strength", "rhyme_group"}, "flow slot")
        flow_slots.append(FlowSlot(**slot))  # type: ignore[arg-type]
    provenance = _mapping(payload["provenance"], "flow provenance")
    _keys(provenance, {"kind", "source", "source_hash", "quantization_error_ticks"}, "flow provenance")
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
        if any(not isinstance(value, int) or value < 0 for value in values):
            raise ValueError("candidate policy counts must be non-negative integers")
        if self.maximum_candidates <= 0 or self.minimum_valid_candidates <= 0:
            raise ValueError("maximum and minimum valid candidate counts must be positive")
        if self.minimum_valid_candidates > self.maximum_candidates:
            raise ValueError("minimum_valid_candidates cannot exceed maximum_candidates")
        if self.initial_candidates + self.rescue_candidates > self.maximum_candidates:
            raise ValueError("candidate policy waves cannot exceed maximum_candidates")
        if not isinstance(self.render_reserve_ms, int) or self.render_reserve_ms < 0:
            raise ValueError("render_reserve_ms must be a non-negative integer")
        if not isfinite(self.minimum_score):
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
class RemoteRapBarRequest:
    bar: int
    topic: str
    flow_template: FlowTemplate

    def __post_init__(self) -> None:
        if not isinstance(self.bar, int) or self.bar < 0:
            raise ValueError("bar must be a non-negative integer")
        if not isinstance(self.topic, str) or not self.topic:
            raise ValueError("topic must be a non-empty string")
        if not isinstance(self.flow_template, FlowTemplate):
            raise ValueError("flow_template must be a FlowTemplate")

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
        if not isinstance(self.chunk_index, int) or self.chunk_index < 0:
            raise ValueError("chunk_index must be a non-negative integer")
        if not isinstance(self.bars, tuple) or len(self.bars) != 2 or not all(isinstance(bar, RemoteRapBarRequest) for bar in self.bars):
            raise ValueError("remote chunk requests require two consecutive bars")
        if self.bars[1].bar != self.bars[0].bar + 1:
            raise ValueError("remote chunk requests require two consecutive bars")
        if any(bar.flow_template.ticks_per_beat != 4 or bar.flow_template.beats_per_bar != 4 for bar in self.bars):
            raise ValueError("remote chunk flow templates must use 4/4 timing")
        if not isfinite(self.tempo_bpm) or self.tempo_bpm <= 0:
            raise ValueError("tempo_bpm must be positive and finite")
        if self.output_sample_rate_hz != REMOTE_CHUNK_SAMPLE_RATE_HZ:
            raise ValueError("output_sample_rate_hz must be 24000")
        if self.expected_frame_count != self.frame_count_for(self.tempo_bpm, self.output_sample_rate_hz):
            raise ValueError("expected_frame_count must match the two-bar timing contract")
        if not isinstance(self.remaining_budget_ms, int) or self.remaining_budget_ms <= 0:
            raise ValueError("remaining_budget_ms must be a positive integer")
        if not isinstance(self.policy, RemoteCandidatePolicy):
            raise ValueError("policy must be a RemoteCandidatePolicy")
        if not isinstance(self.context_lines, tuple) or not all(isinstance(item, str) for item in self.context_lines):
            raise ValueError("context_lines must be a tuple of strings")
        if not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
        if self.request_id != self.request_id_for(self.identity_payload()):
            raise ValueError("request_id does not match the canonical request body")

    @staticmethod
    def frame_count_for(tempo_bpm: float, sample_rate_hz: int = REMOTE_CHUNK_SAMPLE_RATE_HZ) -> int:
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
        if not isfinite(tempo_bpm) or tempo_bpm <= 0:
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


@dataclass(frozen=True)
class RemoteSelectedBar:
    bar: int
    text: str
    flow_template_id: str
    scheduled: tuple[ScheduledSyllable, ...]
    score: float
    diagnostics: Mapping[str, object] = MappingProxyType({})

    def __post_init__(self) -> None:
        if not isinstance(self.bar, int) or self.bar < 0:
            raise ValueError("selected bar must have a non-negative bar identity")
        if not isinstance(self.text, str) or not isinstance(self.flow_template_id, str) or not self.flow_template_id:
            raise ValueError("selected bar identity must be valid")
        if not isinstance(self.scheduled, tuple) or not all(isinstance(item, ScheduledSyllable) for item in self.scheduled):
            raise ValueError("selected bar scheduled must be a tuple of ScheduledSyllable values")
        if any(item.slot.bar != self.bar or item.slot.template_id != self.flow_template_id for item in self.scheduled):
            raise ValueError("selected bar scheduled events must preserve bar identity and flow template")
        if not isfinite(self.score):
            raise ValueError("selected bar score must be finite")
        if not isinstance(self.diagnostics, Mapping):
            raise ValueError("selected bar diagnostics must be a mapping")
        object.__setattr__(self, "diagnostics", _freeze(self.diagnostics))

    @classmethod
    def create(cls, request: RemoteRapBarRequest, *, text: str, scheduled: tuple[ScheduledSyllable, ...], score: float, diagnostics: Mapping[str, object] | None = None) -> RemoteSelectedBar:
        if tuple(item.slot for item in scheduled) != materialize_flow(request.flow_template, request.bar):
            raise ValueError("selected bar identity must match the requested materialized flow")
        return cls(request.bar, text, request.flow_template.template_id, scheduled, score, {} if diagnostics is None else diagnostics)

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
    diagnostics: Mapping[str, object]
    vocal_sha256: str
    schema_version: str = REMOTE_CHUNK_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != REMOTE_CHUNK_SCHEMA_VERSION or not isinstance(self.request_id, str) or not self.request_id:
            raise ValueError("manifest schema_version and request_id must be valid")
        if not isinstance(self.chunk_index, int) or self.chunk_index < 0 or not isfinite(self.tempo_bpm) or self.tempo_bpm <= 0:
            raise ValueError("manifest chunk_index and tempo_bpm must be valid")
        if self.output_sample_rate_hz != REMOTE_CHUNK_SAMPLE_RATE_HZ or self.expected_frame_count != RemoteRapChunkRequest.frame_count_for(self.tempo_bpm):
            raise ValueError("manifest timing does not match the two-bar 24 kHz contract")
        if not isinstance(self.selected_bars, tuple) or len(self.selected_bars) != 2 or not all(isinstance(item, RemoteSelectedBar) for item in self.selected_bars) or self.selected_bars[1].bar != self.selected_bars[0].bar + 1:
            raise ValueError("manifest requires two consecutive selected bars")
        if not isinstance(self.diagnostics, Mapping):
            raise ValueError("manifest diagnostics must be a mapping")
        if not isinstance(self.vocal_sha256, str) or len(self.vocal_sha256) != 64 or any(item not in "0123456789abcdef" for item in self.vocal_sha256):
            raise ValueError("manifest vocal_sha256 must be a lowercase SHA-256 hex digest")
        object.__setattr__(self, "diagnostics", _freeze(self.diagnostics))

    def to_payload(self) -> dict[str, object]:
        return {"schema_version": self.schema_version, "request_id": self.request_id, "chunk_index": self.chunk_index, "tempo_bpm": self.tempo_bpm, "output_sample_rate_hz": self.output_sample_rate_hz, "expected_frame_count": self.expected_frame_count, "selected_bars": [item.to_payload() for item in self.selected_bars], "diagnostics": _thaw(self.diagnostics), "vocal_sha256": self.vocal_sha256}

    def canonical_json_bytes(self) -> bytes:
        return _canonical_json_bytes(self.to_payload())

    @classmethod
    def from_payload(cls, value: object) -> RemoteRapChunkManifest:
        payload = _mapping(value, "remote chunk manifest")
        _keys(payload, {"schema_version", "request_id", "chunk_index", "tempo_bpm", "output_sample_rate_hz", "expected_frame_count", "selected_bars", "diagnostics", "vocal_sha256"}, "remote chunk manifest")
        bars = payload["selected_bars"]
        if not isinstance(bars, list) or len(bars) != 2:
            raise ValueError("manifest selected_bars must be a two-item array")
        return cls(payload["request_id"], payload["chunk_index"], payload["tempo_bpm"], payload["output_sample_rate_hz"], payload["expected_frame_count"], (RemoteSelectedBar.from_payload(bars[0]), RemoteSelectedBar.from_payload(bars[1])), _mapping(payload["diagnostics"], "manifest diagnostics"), payload["vocal_sha256"], payload["schema_version"])  # type: ignore[arg-type]


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
