from __future__ import annotations

import hashlib
import io
import json
import struct
import wave
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from threading import Event

import pytest
from fastapi.testclient import TestClient

from streammuse.application.rap.chunk_orchestration import (
    NoValidCandidates,
    PhraseRenderFailed,
    RemoteChunkRenderArtifact,
    RenderBudgetExpired,
)
from streammuse.domain.rap import (
    FlowProvenance,
    FlowSlot,
    FlowTemplate,
    RemoteCandidatePolicy,
    RemoteCandidateStats,
    RemoteRapBarRequest,
    RemoteRapChunkDiagnostics,
    RemoteRapChunkManifest,
    RemoteRapChunkRequest,
    RemoteSelectedBar,
    ScheduledSyllable,
    Syllable,
    materialize_flow,
)
from streammuse.infrastructure.rap.chunk_package import RAP_CHUNK_PACKAGE_MEDIA_TYPE
from streammuse.presentation.rap_render_server import build_parser, create_rap_render_app, main


def _request(*, remaining_budget_ms: int = 5_000) -> RemoteRapChunkRequest:
    flow = FlowTemplate(
        template_id="test-flow",
        name="Test flow",
        ticks_per_beat=4,
        beats_per_bar=4,
        slots=(FlowSlot(tick_in_bar=0, duration_ticks=4, target_stress=1.0),),
        provenance=FlowProvenance(kind="test", source="unit-test"),
    )
    return RemoteRapChunkRequest.create(
        session_id="session-1",
        chunk_index=0,
        bars=(RemoteRapBarRequest(0, "space", flow), RemoteRapBarRequest(1, "space", flow)),
        tempo_bpm=90.0,
        remaining_budget_ms=remaining_budget_ms,
        policy=RemoteCandidatePolicy.realtime_default(),
        context_lines=("previous line",),
        seed=7,
    )


def _wav(request: RemoteRapChunkRequest) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(24_000)
        output.writeframes(struct.pack("<h", 1_000) * request.expected_frame_count)
    return buffer.getvalue()


def _artifact(request: RemoteRapChunkRequest, workspace: Path) -> RemoteChunkRenderArtifact:
    vocal_wav = _wav(request)
    selected_bars = tuple(
        RemoteSelectedBar.create(
            bar,
            text="orbit",
            scheduled=tuple(
                ScheduledSyllable(slot, Syllable("orbit", 0, 1, 1, ("AO1",), "test"))
                for slot in materialize_flow(bar.flow_template, bar.bar)
            ),
            score=0.9,
        )
        for bar in request.bars
    )
    manifest = RemoteRapChunkManifest(
        request_id=request.request_id,
        chunk_index=request.chunk_index,
        tempo_bpm=request.tempo_bpm,
        output_sample_rate_hz=request.output_sample_rate_hz,
        expected_frame_count=request.expected_frame_count,
        selected_bars=selected_bars,  # type: ignore[arg-type]
        diagnostics=RemoteRapChunkDiagnostics(
            accepted_request_budget_ms=request.remaining_budget_ms,
            resolved_policy=request.policy,
            candidate_stats=RemoteCandidateStats(2, 2, 2, 2, (), ()),
            stage_timings_ms={
                "generation": 1.0,
                "evaluation": 2.0,
                "moss": 3.0,
                "aligner": 4.0,
                "warp": 5.0,
                "packaging": 6.0,
                "total": 21.0,
            },
            alignment_diagnostics={
                "fallback_counts": {"word": 0},
                "source_anchors": [0.0],
                "target_anchors": [0.0],
                "local_warp_ratios": [1.0],
            },
            audio_diagnostics={
                "sample_rate_hz": 24_000,
                "frame_count": request.expected_frame_count,
                "duration_seconds": request.expected_frame_count / 24_000,
                "peak": 0.5,
            },
            model_tool_versions={"moss": "test", "aligner": "mms-test", "rubberband": "test"},
            warnings=(),
        ),
        vocal_sha256=hashlib.sha256(vocal_wav).hexdigest(),
    )
    return RemoteChunkRenderArtifact(
        manifest=manifest,
        vocal_wav=vocal_wav,
        candidate_ledger=({"candidate_id": "candidate-1", "prompt": "private"},),
        workspace=workspace,
    )


class FakeOrchestrator:
    def __init__(self, workspace_root: Path, result: object | None = None) -> None:
        self.workspace_root = workspace_root
        self.result = result
        self.calls = 0
        self.started = Event()
        self.release: Event | None = None

    def render(self, request: RemoteRapChunkRequest) -> RemoteChunkRenderArtifact:
        self.calls += 1
        self.started.set()
        if self.release is not None:
            assert self.release.wait(timeout=5)
        if isinstance(self.result, BaseException):
            raise self.result
        workspace = self.workspace_root / request.request_id
        workspace.mkdir(parents=True, exist_ok=True)
        (workspace / "source.wav").write_bytes(b"source wav")
        (workspace / "active-aligner.json").write_text('{"engine":"mms"}', encoding="utf-8")
        (workspace / "reference.TextGrid").write_text("File type = \\\"ooTextFile\\\"", encoding="utf-8")
        return self.result if isinstance(self.result, RemoteChunkRenderArtifact) else _artifact(request, workspace)


def _client(tmp_path: Path, orchestrator: FakeOrchestrator) -> TestClient:
    return TestClient(
        create_rap_render_app(
            orchestrator,
            health={
                "protocol_version": "remote-rap-chunk/v1",
                "schema_version": "1",
                "ready": True,
                "vllm": {"ready": True, "model": "test-model"},
                "moss": {"ready": True, "model": "test-moss"},
                "aligner": {"ready": True, "identity": "mms-test"},
                "rubberband": {"ready": True, "version": "test"},
                "candidate_profile": "realtime",
                "warmup": {"complete": True},
            },
            artifact_root=tmp_path / "artifacts",
        )
    )


def _post(client: TestClient, request: RemoteRapChunkRequest):
    return client.post(
        "/v1/rap/chunks/render",
        content=request.canonical_json_bytes(),
        headers={"Content-Type": "application/json", "Idempotency-Key": request.request_id},
    )


def test_health_exposes_compatible_readiness_fields(tmp_path: Path) -> None:
    client = _client(tmp_path, FakeOrchestrator(tmp_path / "worker"))

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["protocol_version"] == "remote-rap-chunk/v1"
    assert response.json()["schema_version"] == "1"
    assert response.json()["ready"] is True
    assert response.json()["aligner"]["identity"] == "mms-test"
    assert response.json()["candidate_profile"] == "realtime"


def test_render_returns_canonical_binary_package_and_atomic_artifacts(tmp_path: Path) -> None:
    request = _request()
    orchestrator = FakeOrchestrator(tmp_path / "worker")
    client = _client(tmp_path, orchestrator)

    response = _post(client, request)

    workspace = tmp_path / "artifacts" / request.request_id
    assert response.status_code == 200
    assert response.headers["content-type"] == RAP_CHUNK_PACKAGE_MEDIA_TYPE
    assert response.headers["x-streammuse-request-id"] == request.request_id
    assert response.headers["content-length"] == str(len(response.content))
    assert "total;dur=21" in response.headers["server-timing"]
    assert (workspace / "request.json").read_bytes() == request.canonical_json_bytes()
    assert json.loads((workspace / "candidate_ledger.json").read_text(encoding="utf-8"))[0]["candidate_id"] == "candidate-1"
    assert (workspace / "source.wav").read_bytes() == b"source wav"
    assert json.loads((workspace / "alignment.json").read_text(encoding="utf-8"))["fallback_counts"] == {"word": 0}
    assert (workspace / "reference.TextGrid").exists()
    assert (workspace / "aligned.wav").exists()
    assert (workspace / "response.zip").read_bytes() == response.content


@pytest.mark.parametrize(
    ("failure", "status", "code"),
    (
        (RenderBudgetExpired("request body secret"), 422, "budget_exhausted"),
        (NoValidCandidates("request body secret"), 422, "no_valid_candidates"),
        (PhraseRenderFailed("request body secret"), 503, "render_failed"),
    ),
)
def test_known_render_failures_return_bounded_error_payloads(
    tmp_path: Path, failure: BaseException, status: int, code: str
) -> None:
    request = _request()
    client = _client(tmp_path, FakeOrchestrator(tmp_path / "worker", failure))

    response = _post(client, request)

    assert response.status_code == status
    assert response.json() == {"error": {"code": code, "message": "rap chunk render could not be completed"}}
    assert not (tmp_path / "artifacts" / request.request_id / "response.zip").exists()


def test_malformed_request_and_unexpected_failure_are_sanitized(tmp_path: Path) -> None:
    client = _client(tmp_path, FakeOrchestrator(tmp_path / "worker", RuntimeError("Bearer top-secret prompt")))

    malformed = client.post(
        "/v1/rap/chunks/render",
        content=b"{}",
        headers={"Content-Type": "application/json", "Idempotency-Key": "x"},
    )
    unexpected = _post(client, _request())

    assert malformed.status_code == 422
    assert malformed.json() == {"error": {"code": "invalid_request", "message": "invalid rap chunk request"}}
    assert unexpected.status_code == 500
    assert unexpected.json() == {"error": {"code": "internal_error", "message": "rap chunk render failed"}}
    assert "top-secret" not in unexpected.text
    assert "prompt" not in unexpected.text


def test_matching_in_flight_requests_share_one_render(tmp_path: Path) -> None:
    request = _request()
    orchestrator = FakeOrchestrator(tmp_path / "worker")
    orchestrator.release = Event()
    client = _client(tmp_path, orchestrator)

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(_post, client, request)
        assert orchestrator.started.wait(timeout=5)
        second = executor.submit(_post, client, request)
        orchestrator.release.set()
        responses = (first.result(timeout=5), second.result(timeout=5))

    assert [response.status_code for response in responses] == [200, 200]
    assert responses[0].content == responses[1].content
    assert orchestrator.calls == 1


def test_completed_request_returns_byte_identical_cached_package(tmp_path: Path) -> None:
    request = _request()
    orchestrator = FakeOrchestrator(tmp_path / "worker")
    client = _client(tmp_path, orchestrator)

    first = _post(client, request)
    second = _post(client, request)

    assert first.status_code == second.status_code == 200
    assert second.content == first.content
    assert orchestrator.calls == 1


def test_same_id_with_different_canonical_body_is_rejected(tmp_path: Path) -> None:
    request = _request()
    changed_budget = replace(request, remaining_budget_ms=request.remaining_budget_ms - 1)
    client = _client(tmp_path, FakeOrchestrator(tmp_path / "worker"))

    assert _post(client, request).status_code == 200
    response = _post(client, changed_budget)

    assert response.status_code == 409
    assert response.json() == {"error": {"code": "idempotency_conflict", "message": "request ID is already bound to another request"}}


def test_same_id_with_different_in_flight_body_is_rejected(tmp_path: Path) -> None:
    request = _request()
    changed_budget = replace(request, remaining_budget_ms=request.remaining_budget_ms - 1)
    orchestrator = FakeOrchestrator(tmp_path / "worker")
    orchestrator.release = Event()
    client = _client(tmp_path, orchestrator)

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(_post, client, request)
        assert orchestrator.started.wait(timeout=5)
        conflict = _post(client, changed_budget)
        orchestrator.release.set()
        successful = first.result(timeout=5)

    assert successful.status_code == 200
    assert conflict.status_code == 409
    assert orchestrator.calls == 1


def test_cli_defaults_to_loopback_and_refuses_public_bind_without_opt_in() -> None:
    defaults = build_parser().parse_args(
        ["--vllm-model", "vllm", "--moss-model", "moss", "--moss-reference-wav", "voice.wav"]
    )

    status = main(
        [
            "--host",
            "0.0.0.0",
            "--vllm-model",
            "vllm",
            "--moss-model",
            "moss",
            "--moss-reference-wav",
            "voice.wav",
        ]
    )

    assert defaults.host == "127.0.0.1"
    assert status == 2


def test_cli_composes_through_injected_factory_without_model_imports(tmp_path: Path) -> None:
    calls: dict[str, object] = {}

    def compose(config):
        calls["config"] = config
        return FakeOrchestrator(tmp_path / "worker"), {"ready": True}

    def serve(app, **kwargs):
        calls["app"] = app
        calls["serve"] = kwargs

    status = main(
        [
            "--artifact-root",
            str(tmp_path / "artifacts"),
            "--vllm-model",
            "vllm",
            "--moss-model",
            "moss",
            "--moss-reference-wav",
            "voice.wav",
        ],
        composition_factory=compose,
        serve=serve,
    )

    assert status == 0
    assert calls["config"].host == "127.0.0.1"  # type: ignore[union-attr]
    assert calls["serve"] == {"host": "127.0.0.1", "port": 8020, "log_level": "info"}
