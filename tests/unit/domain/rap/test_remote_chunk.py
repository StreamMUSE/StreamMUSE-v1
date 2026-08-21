from __future__ import annotations

from dataclasses import FrozenInstanceError
from math import inf, nan

import pytest

from streammuse.domain.rap import (
    AudioFormat,
    FlowProvenance,
    FlowSlot,
    FlowTemplate,
    PcmAudio,
    PreparedRapBar,
    RemoteCandidatePolicy,
    RemoteCandidateStats,
    RemoteRapBarRequest,
    RemoteRapChunkDiagnostics,
    RemoteRapChunkRequest,
    RemoteRapChunkTransportAttempt,
    RemoteRapChunkManifest,
    RemoteSelectedBar,
    ScheduledSyllable,
    Syllable,
    PreparedRapChunk,
    materialize_flow,
)


@pytest.fixture
def flow() -> FlowTemplate:
    return FlowTemplate(
        template_id="test_flow",
        name="Test flow",
        ticks_per_beat=4,
        beats_per_bar=4,
        slots=(
            FlowSlot(tick_in_bar=0, duration_ticks=1, target_stress=1.0),
            FlowSlot(tick_in_bar=4, duration_ticks=2, target_stress=0.5, rhyme_group="A"),
        ),
        provenance=FlowProvenance(kind="test", source="unit-test", source_hash="abc123"),
    )


def bar_request(bar: int, flow: FlowTemplate, topic: str = "space") -> RemoteRapBarRequest:
    return RemoteRapBarRequest(bar=bar, topic=topic, flow_template=flow)


def remote_request(
    flow: FlowTemplate,
    *,
    remaining_budget_ms: int = 5_000,
    tempo_bpm: float = 90.0,
) -> RemoteRapChunkRequest:
    return RemoteRapChunkRequest.create(
        session_id="session-1",
        chunk_index=0,
        bars=(bar_request(0, flow), bar_request(1, flow)),
        tempo_bpm=tempo_bpm,
        remaining_budget_ms=remaining_budget_ms,
        policy=RemoteCandidatePolicy.realtime_default(),
        context_lines=("previous line",),
        seed=7,
    )


def diagnostics(flow: FlowTemplate) -> RemoteRapChunkDiagnostics:
    return RemoteRapChunkDiagnostics(
        accepted_request_budget_ms=5_000,
        resolved_policy=RemoteCandidatePolicy.realtime_default(),
        candidate_stats=RemoteCandidateStats(
            requested_count=6,
            parseable_count=5,
            valid_count=3,
            selectable_count=2,
            top_candidates=({"text": "orbit", "score": 0.9},),
            rejections=({"text": "noise", "reason": "flow_mismatch"},),
        ),
        stage_timings_ms={
            "generation": 1.0,
            "evaluation": 1.0,
            "moss": 1.0,
            "mfa": 1.0,
            "warp": 1.0,
            "packaging": 1.0,
            "total": 7.0,
        },
        alignment_diagnostics={
            "fallback_counts": {"word": 0},
            "source_anchors": [0.0],
            "target_anchors": [0.0],
            "local_warp_ratios": [1.0],
        },
        audio_diagnostics={
            "sample_rate_hz": 24_000,
            "frame_count": 128_000,
            "duration_seconds": 128_000 / 24_000,
            "peak": 0.5,
        },
        model_tool_versions={"moss": "test", "mfa": "test", "rubberband": "test"},
        warnings=(),
    )


def test_remote_request_requires_two_consecutive_bars(flow: FlowTemplate) -> None:
    with pytest.raises(ValueError, match="two consecutive bars"):
        RemoteRapChunkRequest.create(
            session_id="session-1",
            chunk_index=0,
            bars=(bar_request(0, flow), bar_request(2, flow)),
            tempo_bpm=90.0,
            remaining_budget_ms=5_000,
            policy=RemoteCandidatePolicy.realtime_default(),
            context_lines=(),
            seed=7,
        )


def test_remote_request_requires_four_four_flow_templates(flow: FlowTemplate) -> None:
    invalid_flow = FlowTemplate.__new__(FlowTemplate)
    object.__setattr__(invalid_flow, "template_id", flow.template_id)
    object.__setattr__(invalid_flow, "name", flow.name)
    object.__setattr__(invalid_flow, "ticks_per_beat", 3)
    object.__setattr__(invalid_flow, "beats_per_bar", 4)
    object.__setattr__(invalid_flow, "slots", flow.slots)
    object.__setattr__(invalid_flow, "provenance", flow.provenance)

    with pytest.raises(ValueError, match="4/4"):
        RemoteRapChunkRequest.create(
            session_id="session-1",
            chunk_index=0,
            bars=(bar_request(0, invalid_flow), bar_request(1, invalid_flow)),
            tempo_bpm=90.0,
            remaining_budget_ms=5_000,
            policy=RemoteCandidatePolicy.realtime_default(),
            context_lines=(),
            seed=7,
        )


@pytest.mark.parametrize("tempo_bpm", (0.0, -1.0, inf, nan))
def test_remote_request_requires_positive_finite_tempo(flow: FlowTemplate, tempo_bpm: float) -> None:
    with pytest.raises(ValueError, match="tempo_bpm"):
        remote_request(flow, tempo_bpm=tempo_bpm)


@pytest.mark.parametrize("remaining_budget_ms", (0, -1))
def test_remote_request_requires_positive_budget(flow: FlowTemplate, remaining_budget_ms: int) -> None:
    with pytest.raises(ValueError, match="remaining_budget_ms"):
        remote_request(flow, remaining_budget_ms=remaining_budget_ms)


def test_remote_request_id_excludes_budget_while_retry_body_keeps_original_budget(flow: FlowTemplate) -> None:
    original = remote_request(flow, remaining_budget_ms=5_000)
    same_identity_new_budget = remote_request(flow, remaining_budget_ms=1_000)
    changed = RemoteRapChunkRequest.create(
        session_id="session-1",
        chunk_index=0,
        bars=(bar_request(0, flow, topic="ocean"), bar_request(1, flow, topic="space")),
        tempo_bpm=90.0,
        remaining_budget_ms=5_000,
        policy=RemoteCandidatePolicy.realtime_default(),
        context_lines=("previous line",),
        seed=7,
    )

    assert same_identity_new_budget.request_id == original.request_id
    assert same_identity_new_budget.to_payload()["remaining_budget_ms"] == 1_000
    assert original.to_payload()["remaining_budget_ms"] == 5_000
    assert original.to_payload() == original.to_payload()
    assert changed.request_id != original.request_id
    assert RemoteRapChunkRequest.from_payload(original.to_payload()) == original


def test_transport_attempt_reuses_original_bytes_when_equivalent_request_has_lower_budget(flow: FlowTemplate) -> None:
    original = remote_request(flow, remaining_budget_ms=5_000)
    attempt = RemoteRapChunkTransportAttempt.from_request(original)
    equivalent_lower_budget = remote_request(flow, remaining_budget_ms=1_000)

    assert equivalent_lower_budget.request_id == original.request_id
    assert equivalent_lower_budget.canonical_json_bytes() != attempt.body
    assert attempt.request_id == original.request_id
    assert attempt.body == original.canonical_json_bytes()
    assert attempt.retry_body() == attempt.body


def test_remote_request_derives_exact_24khz_two_bar_frame_count(flow: FlowTemplate) -> None:
    request = remote_request(flow)

    assert request.output_sample_rate_hz == 24_000
    assert request.expected_frame_count == 128_000


def test_selected_bar_preserves_requested_bar_identity_and_flow(flow: FlowTemplate) -> None:
    request = remote_request(flow)
    scheduled = tuple(
        ScheduledSyllable(slot=slot, syllable=Syllable("orbit", 0, 1, 1))
        for slot in materialize_flow(flow, bar=0)
    )

    selected = RemoteSelectedBar.create(request.bars[0], text="orbit orbit", scheduled=scheduled, score=0.9)
    assert selected.bar == 0
    assert selected.flow_template_id == "test_flow"
    assert RemoteSelectedBar.from_payload(selected.to_payload()) == selected
    with pytest.raises(ValueError, match="bar identity"):
        RemoteSelectedBar.create(request.bars[1], text="orbit", scheduled=scheduled, score=0.9)


def test_manifest_and_prepared_chunk_keep_diagnostics_immutable(flow: FlowTemplate) -> None:
    request = remote_request(flow)
    scheduled = tuple(
        ScheduledSyllable(slot=slot, syllable=Syllable("orbit", 0, 1, 1))
        for slot in materialize_flow(flow, bar=0)
    )
    selected = RemoteSelectedBar.create(request.bars[0], text="orbit orbit", scheduled=scheduled, score=0.9)
    second_selected = RemoteSelectedBar.create(
        request.bars[1],
        text="orbit orbit",
        scheduled=tuple(
            ScheduledSyllable(slot=slot, syllable=Syllable("orbit", 0, 1, 1))
            for slot in materialize_flow(flow, bar=1)
        ),
        score=0.9,
    )
    manifest = RemoteRapChunkManifest(
        request_id=request.request_id,
        chunk_index=request.chunk_index,
        tempo_bpm=request.tempo_bpm,
        output_sample_rate_hz=request.output_sample_rate_hz,
        expected_frame_count=request.expected_frame_count,
        selected_bars=(selected, second_selected),
        diagnostics=diagnostics(flow),
        vocal_sha256="a" * 64,
    )
    prepared_bar = PreparedRapBar(
        bar=0,
        text="orbit",
        source="remote",
        fallback_reason=None,
        scheduled=(),
        audio=PcmAudio(AudioFormat(), 0, b""),
        diagnostics=(),
        warnings=(),
        render_latency_ms=0.0,
    )
    second_prepared_bar = PreparedRapBar(
        bar=1,
        text="orbit",
        source="remote",
        fallback_reason=None,
        scheduled=(),
        audio=PcmAudio(AudioFormat(), 0, b""),
        diagnostics=(),
        warnings=(),
        render_latency_ms=0.0,
    )
    chunk = PreparedRapChunk(
        request_id=request.request_id,
        chunk_index=0,
        renderer="moss_aligned_remote",
        bars=(prepared_bar, second_prepared_bar),
        diagnostics={"source": "remote"},
    )

    with pytest.raises(FrozenInstanceError):
        manifest.diagnostics.accepted_request_budget_ms = 1  # type: ignore[misc]
    with pytest.raises(TypeError):
        chunk.diagnostics["source"] = "local"  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        chunk.renderer = "espeak"  # type: ignore[misc]


def test_manifest_requires_complete_diagnostic_envelope(flow: FlowTemplate) -> None:
    request = remote_request(flow)
    selected = tuple(
        RemoteSelectedBar.create(
            request.bars[bar],
            text="orbit orbit",
            scheduled=tuple(
                ScheduledSyllable(slot=slot, syllable=Syllable("orbit", 0, 1, 1))
                for slot in materialize_flow(flow, bar=bar)
            ),
            score=0.9,
        )
        for bar in range(2)
    )

    with pytest.raises(ValueError, match="diagnostics"):
        RemoteRapChunkManifest(
            request_id=request.request_id,
            chunk_index=request.chunk_index,
            tempo_bpm=request.tempo_bpm,
            output_sample_rate_hz=request.output_sample_rate_hz,
            expected_frame_count=request.expected_frame_count,
            selected_bars=selected,
            diagnostics={},  # type: ignore[arg-type]
            vocal_sha256="a" * 64,
        )


def test_selected_bar_requires_component_scores(flow: FlowTemplate) -> None:
    request = remote_request(flow)
    scheduled = tuple(
        ScheduledSyllable(slot=slot, syllable=Syllable("orbit", 0, 1, 1))
        for slot in materialize_flow(flow, bar=0)
    )

    with pytest.raises(ValueError, match="component_scores"):
        RemoteSelectedBar(
            bar=0,
            text="orbit orbit",
            flow_template_id=flow.template_id,
            scheduled=scheduled,
            score=0.9,
            diagnostics={},
        )


def test_prepared_chunk_requires_exactly_two_consecutive_bars() -> None:
    prepared_bar = PreparedRapBar(
        bar=0,
        text="orbit",
        source="remote",
        fallback_reason=None,
        scheduled=(),
        audio=PcmAudio(AudioFormat(), 0, b""),
        diagnostics=(),
        warnings=(),
        render_latency_ms=0.0,
    )

    with pytest.raises(ValueError, match="two consecutive bars"):
        PreparedRapChunk(
            request_id="request-1",
            chunk_index=0,
            renderer="moss_aligned_remote",
            bars=(prepared_bar, prepared_bar),
            diagnostics={},
        )
