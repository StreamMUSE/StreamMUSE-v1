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


_ARTIFACT_IDS = {
    "request": "request.json",
    "candidate_ledger": "candidate_ledger.json",
    "source_wav": "source.wav",
    "mms_alignment": "mms_alignment.json",
    "alignment": "alignment.json",
    "aligned_wav": "aligned.wav",
    "vocal_wav": "vocal.wav",
    "manifest": "manifest.json",
    "server_timing": "server_timing.json",
    "response_package": "response.zip",
}


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
            top_candidates=(
                {
                    "bar": 0,
                    "candidate_id": "candidate-1",
                    "text": "orbit",
                    "score": 0.9,
                    "component_scores": {"flow": 0.9},
                    "source_order": 0,
                },
            ),
            rejections=(
                {
                    "bar": 0,
                    "candidate_id": "candidate-2",
                    "text": "noise",
                    "reasons": ("flow_mismatch",),
                    "source_order": 1,
                },
            ),
        ),
        stage_timings_ms={
            "generation": 1.0,
            "evaluation": 1.0,
            "moss": 1.0,
            "aligner": 1.0,
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
        model_tool_versions={"moss": "test", "aligner": "test", "rubberband": "test"},
        warnings=(),
        monitoring_summary={
            "schema_version": "streammuse.rap_chunk_monitor.v1",
            "alignment_method": "torchaudio.pipelines.MMS_FA",
            "alignment_confidence": 0.9,
            "source_wav_sha256": "a" * 64,
            "artifact_ids": _ARTIFACT_IDS,
        },
    )


def _request_payload(flow: FlowTemplate) -> dict[str, object]:
    return remote_request(flow).to_payload()


def _refresh_request_id(payload: dict[str, object]) -> None:
    identity = {key: value for key, value in payload.items() if key not in {"request_id", "remaining_budget_ms"}}
    payload["request_id"] = RemoteRapChunkRequest.request_id_for(identity)


def _selected_payload(flow: FlowTemplate) -> dict[str, object]:
    request = remote_request(flow)
    selected = RemoteSelectedBar.create(
        request.bars[0],
        text="orbit orbit",
        scheduled=tuple(
            ScheduledSyllable(slot=slot, syllable=Syllable("orbit", 0, 1, 1, ("AO1",), "cmu"))
            for slot in materialize_flow(flow, bar=0)
        ),
        score=0.9,
    )
    return selected.to_payload()


def _nested_json(depth: int) -> object:
    value: object = "leaf"
    for _ in range(depth):
        value = {"next": value}
    return value


@pytest.mark.parametrize(
    "mutate",
    (
        lambda payload: payload.__setitem__("remaining_budget_ms", True),
        lambda payload: payload.__setitem__("chunk_index", True),
        lambda payload: payload.__setitem__("tempo_bpm", True),
        lambda payload: payload.__setitem__("output_sample_rate_hz", True),
        lambda payload: payload.__setitem__("expected_frame_count", True),
        lambda payload: payload.__setitem__("seed", True),
        lambda payload: payload["policy"].__setitem__("initial_candidates", True),  # type: ignore[union-attr]
        lambda payload: payload["policy"].__setitem__("minimum_score", True),  # type: ignore[union-attr]
    ),
)
def test_request_and_policy_wire_numbers_reject_booleans(flow: FlowTemplate, mutate) -> None:
    payload = _request_payload(flow)
    mutate(payload)
    _refresh_request_id(payload)

    with pytest.raises(ValueError):
        RemoteRapChunkRequest.from_payload(payload)


@pytest.mark.parametrize(
    "mutate",
    (
        lambda payload: payload["bars"][0].__setitem__("bar", True),  # type: ignore[index,union-attr]
        lambda payload: payload["bars"][0]["flow_template"].__setitem__("ticks_per_beat", True),  # type: ignore[index,union-attr]
        lambda payload: payload["bars"][0]["flow_template"]["slots"][0].__setitem__("tick_in_bar", True),  # type: ignore[index,union-attr]
        lambda payload: payload["bars"][0]["flow_template"]["slots"][0].__setitem__("target_stress", True),  # type: ignore[index,union-attr]
        lambda payload: payload["bars"][0]["flow_template"]["provenance"].__setitem__("quantization_error_ticks", True),  # type: ignore[index,union-attr]
    ),
)
def test_flow_wire_numbers_reject_booleans(flow: FlowTemplate, mutate) -> None:
    payload = _request_payload(flow)
    mutate(payload)
    _refresh_request_id(payload)

    with pytest.raises(ValueError):
        RemoteRapChunkRequest.from_payload(payload)


@pytest.mark.parametrize(
    "flow_slot",
    (
        FlowSlot(tick_in_bar=0, duration_ticks=1, target_stress=True),
        FlowSlot(tick_in_bar=0, duration_ticks=1, target_stress=0.5, boundary_strength=True),
    ),
)
def test_locally_created_flow_wire_numbers_reject_booleans(flow_slot: FlowSlot) -> None:
    invalid_flow = FlowTemplate(
        template_id="test_flow",
        name="Test flow",
        ticks_per_beat=4,
        beats_per_bar=4,
        slots=(flow_slot,),
        provenance=FlowProvenance(kind="test", source="unit-test"),
    )

    with pytest.raises(ValueError):
        RemoteRapBarRequest(0, "space", invalid_flow)


@pytest.mark.parametrize(
    "mutate",
    (
        lambda payload: payload.__setitem__("bar", True),
        lambda payload: payload.__setitem__("score", True),
        lambda payload: payload["scheduled"][0]["slot"].__setitem__("tick", True),  # type: ignore[index,union-attr]
        lambda payload: payload["scheduled"][0]["slot"].__setitem__("accent", True),  # type: ignore[index,union-attr]
        lambda payload: payload["scheduled"][0]["syllable"].__setitem__("index_in_word", True),  # type: ignore[index,union-attr]
        lambda payload: payload["diagnostics"]["component_scores"].__setitem__("total", True),  # type: ignore[index,union-attr]
    ),
)
def test_selected_bar_wire_numbers_reject_booleans(flow: FlowTemplate, mutate) -> None:
    payload = _selected_payload(flow)
    mutate(payload)

    with pytest.raises(ValueError):
        RemoteSelectedBar.from_payload(payload)


@pytest.mark.parametrize(
    "mutate",
    (
        lambda payload: payload.__setitem__("accepted_request_budget_ms", True),
        lambda payload: payload["candidate_stats"].__setitem__("requested_count", True),  # type: ignore[union-attr]
        lambda payload: payload["candidate_stats"]["top_candidates"][0].__setitem__("score", True),  # type: ignore[index,union-attr]
        lambda payload: payload["stage_timings_ms"].__setitem__("moss", True),  # type: ignore[union-attr]
        lambda payload: payload["alignment_diagnostics"]["fallback_counts"].__setitem__("word", True),  # type: ignore[index,union-attr]
        lambda payload: payload["alignment_diagnostics"]["source_anchors"].__setitem__(0, True),  # type: ignore[index,union-attr]
        lambda payload: payload["audio_diagnostics"].__setitem__("frame_count", True),  # type: ignore[union-attr]
    ),
)
def test_diagnostic_wire_numbers_reject_booleans(flow: FlowTemplate, mutate) -> None:
    payload = diagnostics(flow).to_payload()
    mutate(payload)

    with pytest.raises(ValueError):
        RemoteRapChunkDiagnostics.from_payload(payload)


@pytest.mark.parametrize(
    "mutate",
    (
        lambda payload: payload.pop("monitoring_summary"),
        lambda payload: payload["monitoring_summary"].__setitem__("unexpected", True),  # type: ignore[union-attr]
        lambda payload: payload["monitoring_summary"].__setitem__("schema_version", "unknown"),  # type: ignore[union-attr]
        lambda payload: payload["monitoring_summary"].__setitem__("alignment_confidence", True),  # type: ignore[union-attr]
        lambda payload: payload["monitoring_summary"].__setitem__("alignment_confidence", 1.1),  # type: ignore[union-attr]
        lambda payload: payload["monitoring_summary"].__setitem__("source_wav_sha256", "not-a-hash"),  # type: ignore[union-attr]
        lambda payload: payload["monitoring_summary"]["artifact_ids"].pop("manifest"),  # type: ignore[index,union-attr]
        lambda payload: payload["monitoring_summary"]["artifact_ids"].__setitem__("manifest", "/absolute/manifest.json"),  # type: ignore[index,union-attr]
    ),
)
def test_monitoring_summary_is_a_strict_versioned_wire_contract(
    flow: FlowTemplate, mutate
) -> None:
    payload = diagnostics(flow).to_payload()
    mutate(payload)

    with pytest.raises(ValueError):
        RemoteRapChunkDiagnostics.from_payload(payload)


def test_monitoring_summary_round_trips_without_package_only_alignment_anchors(
    flow: FlowTemplate,
) -> None:
    payload = diagnostics(flow).to_payload()

    restored = RemoteRapChunkDiagnostics.from_payload(payload)

    assert restored.monitoring_summary == {
        "schema_version": "streammuse.rap_chunk_monitor.v1",
        "alignment_method": "torchaudio.pipelines.MMS_FA",
        "alignment_confidence": 0.9,
        "source_wav_sha256": "a" * 64,
        "artifact_ids": _ARTIFACT_IDS,
    }


@pytest.mark.parametrize(
    "mutate",
    (
        lambda payload: payload.__setitem__("text", " "),
        lambda payload: payload.__setitem__("scheduled", []),
        lambda payload: payload.__setitem__("scheduled", list(reversed(payload["scheduled"]))),
        lambda payload: payload["scheduled"].__setitem__(1, payload["scheduled"][0]),  # type: ignore[index,union-attr]
        lambda payload: payload["scheduled"][0]["slot"].__setitem__("duration_ticks", 0),  # type: ignore[index,union-attr]
        lambda payload: payload["scheduled"][0]["slot"].__setitem__("accent", 1.1),  # type: ignore[index,union-attr]
        lambda payload: payload["scheduled"][0]["syllable"].__setitem__("word", ""),  # type: ignore[index,union-attr]
        lambda payload: payload["scheduled"][0]["syllable"].__setitem__("index_in_word", 1),  # type: ignore[index,union-attr]
        lambda payload: payload["scheduled"][0]["syllable"].__setitem__("syllable_count", 0),  # type: ignore[index,union-attr]
        lambda payload: payload["scheduled"][0]["syllable"].__setitem__("stress", 3),  # type: ignore[index,union-attr]
        lambda payload: payload["scheduled"][0]["syllable"].__setitem__("phonemes", [""]),  # type: ignore[index,union-attr]
        lambda payload: payload["scheduled"][0]["syllable"].__setitem__("analysis_source", ""),  # type: ignore[index,union-attr]
        lambda payload: payload["scheduled"][1]["slot"].__setitem__("slot_index", 0),  # type: ignore[index,union-attr]
        lambda payload: payload["scheduled"][1]["slot"].__setitem__("slot_index", 2),  # type: ignore[index,union-attr]
    ),
)
def test_selected_bar_rejects_invalid_success_payload(flow: FlowTemplate, mutate) -> None:
    payload = _selected_payload(flow)
    mutate(payload)

    with pytest.raises(ValueError):
        RemoteSelectedBar.from_payload(payload)


@pytest.mark.parametrize(
    "mutate",
    (
        lambda payload: payload["candidate_stats"]["top_candidates"][0].pop("component_scores"),  # type: ignore[index,union-attr]
        lambda payload: payload["candidate_stats"].__setitem__("top_candidates", payload["candidate_stats"]["top_candidates"] * 3),  # type: ignore[index,union-attr]
        lambda payload: payload["candidate_stats"].__setitem__("rejections", payload["candidate_stats"]["rejections"] * 5),  # type: ignore[index,union-attr]
        lambda payload: payload["candidate_stats"]["top_candidates"][0]["component_scores"].__setitem__("flow", nan),  # type: ignore[index,union-attr]
        lambda payload: payload["candidate_stats"]["top_candidates"][0]["component_scores"].__setitem__("flow", {"not_json_safe"}),  # type: ignore[index,union-attr]
    ),
)
def test_candidate_summaries_require_bounded_json_safe_contract(flow: FlowTemplate, mutate) -> None:
    payload = diagnostics(flow).to_payload()
    mutate(payload)

    with pytest.raises(ValueError):
        RemoteRapChunkDiagnostics.from_payload(payload)


@pytest.mark.parametrize(
    "mutate",
    (
        lambda payload: payload["alignment_diagnostics"].__setitem__("source_anchors", []),  # type: ignore[union-attr]
        lambda payload: payload["alignment_diagnostics"].__setitem__("target_anchors", [0.0, 1.0]),  # type: ignore[union-attr]
        lambda payload: payload["alignment_diagnostics"].__setitem__("local_warp_ratios", [0.0]),  # type: ignore[union-attr]
        lambda payload: payload["audio_diagnostics"].__setitem__("duration_seconds", 0.0),  # type: ignore[union-attr]
        lambda payload: payload["audio_diagnostics"].__setitem__("peak", 1.1),  # type: ignore[union-attr]
        lambda payload: payload["model_tool_versions"].pop("aligner"),  # type: ignore[union-attr]
        lambda payload: payload["stage_timings_ms"].__setitem__("total", 0.0),  # type: ignore[union-attr]
        lambda payload: payload["warnings"].append({"not_json_safe"}),  # type: ignore[union-attr]
    ),
)
def test_diagnostics_enforce_cross_field_and_json_safety(flow: FlowTemplate, mutate) -> None:
    payload = diagnostics(flow).to_payload()
    mutate(payload)

    with pytest.raises(ValueError):
        RemoteRapChunkDiagnostics.from_payload(payload)


def test_selected_bar_rejects_excessively_nested_diagnostics(flow: FlowTemplate) -> None:
    payload = _selected_payload(flow)
    payload["diagnostics"]["detail"] = _nested_json(33)  # type: ignore[index,union-attr]

    with pytest.raises(ValueError, match="nesting"):
        RemoteSelectedBar.from_payload(payload)


def test_transport_attempt_normalizes_nested_json_recursion() -> None:
    with pytest.raises(ValueError, match="transport attempt body"):
        RemoteRapChunkTransportAttempt("request-1", b"[" * 1_100 + b"0" + b"]" * 1_100)


@pytest.mark.parametrize("tempo_bpm", (5e-324, 1e308, 1.0))
def test_frame_count_rejects_unrepresentable_pcm16_transport(flow: FlowTemplate, tempo_bpm: float) -> None:
    with pytest.raises(ValueError, match="frame count"):
        remote_request(flow, tempo_bpm=tempo_bpm)


def test_diagnostics_accept_modular_aligner_version(flow: FlowTemplate) -> None:
    payload = diagnostics(flow).to_payload()
    versions = payload["model_tool_versions"]
    versions.pop("aligner")  # type: ignore[union-attr]
    versions["aligner"] = "mms-fa"  # type: ignore[index,union-attr]

    decoded = RemoteRapChunkDiagnostics.from_payload(payload)

    assert "mfa" not in decoded.model_tool_versions
    assert decoded.model_tool_versions["aligner"] == "mms-fa"


def test_diagnostics_accept_generic_aligner_stage_timing(flow: FlowTemplate) -> None:
    payload = diagnostics(flow).to_payload()
    decoded = RemoteRapChunkDiagnostics.from_payload(payload)

    assert "mfa" not in decoded.stage_timings_ms
    assert decoded.stage_timings_ms["aligner"] == 1.0


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
