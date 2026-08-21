from __future__ import annotations

import builtins
import hashlib
import io
import json
import struct
import wave
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from threading import Event, Thread
from types import SimpleNamespace

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
from streammuse.infrastructure.rap.chunk_package import (
    RAP_CHUNK_PACKAGE_MEDIA_TYPE,
    decode_chunk_package,
)
from streammuse.presentation import rap_render_server
from streammuse.presentation.rap_render_server import (
    build_parser,
    create_rap_render_app,
    main,
)


_FULL_MMS_ALIGNMENT_BYTES = (
    b'{"aligner":{"identity":"torchaudio.pipelines.MMS_FA","version":"2.8.0"},'
    b'"character_spans":[{"character":"o","end_seconds":0.1,"score":0.97,'
    b'"start_seconds":0.0,"word":"orbit"}],"normalized_transcript":"orbit orbit",'
    b'"warnings":[],"word_spans":[{"end_seconds":0.5,"score":0.93,'
    b'"start_seconds":0.0,"word":"orbit"}]}\n'
)


def _request(
    *, remaining_budget_ms: int = 5_000, session_id: str = "session-1"
) -> RemoteRapChunkRequest:
    flow = FlowTemplate(
        template_id="test-flow",
        name="Test flow",
        ticks_per_beat=4,
        beats_per_bar=4,
        slots=(FlowSlot(tick_in_bar=0, duration_ticks=4, target_stress=1.0),),
        provenance=FlowProvenance(kind="test", source="unit-test"),
    )
    return RemoteRapChunkRequest.create(
        session_id=session_id,
        chunk_index=0,
        bars=(
            RemoteRapBarRequest(0, "space", flow),
            RemoteRapBarRequest(1, "space", flow),
        ),
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


def _artifact(
    request: RemoteRapChunkRequest, workspace: Path
) -> RemoteChunkRenderArtifact:
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
                "packaging": 0.0,
                "total": 15.0,
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
            model_tool_versions={
                "moss": "test",
                "aligner": "mms-test",
                "rubberband": "test",
            },
            warnings=("packaging timing is provisional",),
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
    def __init__(
        self,
        workspace_root: Path,
        result: object | None = None,
        *,
        source_name: str = "source.wav",
        alignment_name: str = "mms_alignment.json",
    ) -> None:
        self.workspace_root = workspace_root
        self.result = result
        self.source_name = source_name
        self.alignment_name = alignment_name
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
        artifact = (
            self.result
            if isinstance(self.result, RemoteChunkRenderArtifact)
            else _artifact(request, workspace)
        )
        (workspace / self.source_name).write_bytes(b"source wav")
        (workspace / self.alignment_name).write_bytes(_FULL_MMS_ALIGNMENT_BYTES)
        (workspace / "vocal.wav").write_bytes(artifact.vocal_wav)
        (workspace / "reference.TextGrid").write_text(
            'File type = \\"ooTextFile\\"', encoding="utf-8"
        )
        return artifact


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
        headers={
            "Content-Type": "application/json",
            "Idempotency-Key": request.request_id,
        },
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


def test_health_retains_required_defaults_when_optional_summaries_are_absent(
    tmp_path: Path,
) -> None:
    app = create_rap_render_app(
        FakeOrchestrator(tmp_path / "worker"),
        {"ready": True},
        artifact_root=tmp_path / "artifacts",
    )

    response = TestClient(app).get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "protocol_version": "remote-rap-chunk/v1",
        "schema_version": "streammuse.rap_chunk.v1",
        "ready": True,
    }


def test_health_uses_recursive_allowlists_for_public_scalar_summaries(
    tmp_path: Path,
) -> None:
    secret = "must-not-escape"
    health = {
        "protocol_version": "remote-rap-chunk/v1",
        "schema_version": "streammuse.rap_chunk.v1",
        "ready": True,
        "vllm": {
            "ready": True,
            "status": "serving",
            "model": "Qwen-test",
            "vllm_url": secret,
            "endpoint_url": secret,
            "api_key": secret,
            "access_token": secret,
            "model_path": secret,
            "cache_dir": secret,
            "credentials": secret,
            "arbitrary_secret": secret,
            "warmup": {
                "status": "complete",
                "endpoint_url": secret,
                "nested": {"api_key": secret},
            },
        },
        "moss": {
            "ready": True,
            "identity": "MOSS-TTS",
            "model": "MOSS-v1.5",
            "model_path": secret,
        },
        "aligner": {
            "ready": True,
            "identity": "MMS_FA",
            "version": "2.8.0",
            "cache_dir": secret,
        },
        "rubberband": {
            "ready": True,
            "identity": "Rubber Band",
            "version": "3.3.0",
            "credentials": {"access_token": secret},
        },
        "candidate_profile": "realtime",
        "warmup": {
            "ready": True,
            "status": "complete",
            "api_key": secret,
            "nested_variants": {"arbitrary": secret},
        },
        "unapproved": {"status": secret},
    }
    app = create_rap_render_app(
        FakeOrchestrator(tmp_path / "worker"),
        health,
        artifact_root=tmp_path / "artifacts",
    )

    response = TestClient(app).get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "protocol_version": "remote-rap-chunk/v1",
        "schema_version": "streammuse.rap_chunk.v1",
        "ready": True,
        "vllm": {
            "ready": True,
            "status": "serving",
            "model": "Qwen-test",
            "warmup": {"status": "complete"},
        },
        "moss": {"ready": True, "identity": "MOSS-TTS", "model": "MOSS-v1.5"},
        "aligner": {"ready": True, "identity": "MMS_FA", "version": "2.8.0"},
        "rubberband": {
            "ready": True,
            "identity": "Rubber Band",
            "version": "3.3.0",
        },
        "candidate_profile": "realtime",
        "warmup": {"ready": True, "status": "complete"},
    }
    assert secret not in response.text


def test_render_returns_canonical_binary_package_and_atomic_artifacts(
    tmp_path: Path,
) -> None:
    request = _request()
    orchestrator = FakeOrchestrator(tmp_path / "worker")
    client = _client(tmp_path, orchestrator)

    response = _post(client, request)

    workspace = tmp_path / "artifacts" / request.request_id
    assert response.status_code == 200
    assert response.headers["content-type"] == RAP_CHUNK_PACKAGE_MEDIA_TYPE
    assert response.headers["x-streammuse-request-id"] == request.request_id
    assert response.headers["content-length"] == str(len(response.content))
    assert (workspace / "request.json").read_bytes() == request.canonical_json_bytes()
    assert (
        json.loads((workspace / "candidate_ledger.json").read_text(encoding="utf-8"))[
            0
        ]["candidate_id"]
        == "candidate-1"
    )
    assert (workspace / "source.wav").read_bytes() == b"source wav"
    assert (workspace / "mms_alignment.json").read_bytes() == _FULL_MMS_ALIGNMENT_BYTES
    assert (workspace / "vocal.wav").read_bytes() == _wav(request)
    assert not (workspace / "render_failure.json").exists()
    assert json.loads((workspace / "alignment.json").read_text(encoding="utf-8")) == {
        "fallback_counts": {"word": 0},
        "local_warp_ratios": [1.0],
        "source_anchors": [0.0],
        "target_anchors": [0.0],
    }
    assert (workspace / "reference.TextGrid").exists()
    assert (workspace / "aligned.wav").exists()
    assert (workspace / "response.zip").read_bytes() == response.content
    decoded = decode_chunk_package(
        response.content, expected_request_id=request.request_id
    )
    timings = decoded.manifest.diagnostics.stage_timings_ms
    assert timings["packaging"] > 0.0
    assert timings["total"] >= max(timings.values())
    assert (
        "packaging timing is provisional" not in decoded.manifest.diagnostics.warnings
    )
    assert response.headers["server-timing"] == f"total;dur={timings['total']:.3f}"


@pytest.mark.parametrize(
    ("source_name", "alignment_name"),
    (
        ("moss-source.wav", "mms_alignment.json"),
        ("source.wav", "alignment.json"),
    ),
)
def test_render_rejects_legacy_task_3_artifact_names(
    tmp_path: Path, source_name: str, alignment_name: str
) -> None:
    request = _request()
    orchestrator = FakeOrchestrator(
        tmp_path / "worker",
        source_name=source_name,
        alignment_name=alignment_name,
    )
    client = _client(tmp_path, orchestrator)

    response = _post(client, request)

    workspace = tmp_path / "artifacts" / request.request_id
    assert response.status_code == 503
    assert response.json() == {
        "error": {
            "code": "render_failed",
            "message": "rap chunk render could not be completed",
        }
    }
    assert not (workspace / "response.zip").exists()


def test_atomic_replace_fsyncs_containing_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "artifact.json"
    events: list[tuple[str, int | None]] = []
    directory_fds: set[int] = set()
    original_open = rap_render_server.os.open
    original_fsync = rap_render_server.os.fsync
    original_close = rap_render_server.os.close
    original_replace = rap_render_server.os.replace

    def tracked_open(path, flags):
        fd = original_open(path, flags)
        if Path(path) == tmp_path:
            directory_fds.add(fd)
            events.append(("directory_open", fd))
        return fd

    def tracked_fsync(fd):
        events.append(("directory_fsync" if fd in directory_fds else "file_fsync", fd))
        return original_fsync(fd)

    def tracked_close(fd):
        if fd in directory_fds:
            events.append(("directory_close", fd))
        return original_close(fd)

    def tracked_replace(source, destination):
        events.append(("replace", None))
        return original_replace(source, destination)

    monkeypatch.setattr(rap_render_server.os, "open", tracked_open)
    monkeypatch.setattr(rap_render_server.os, "fsync", tracked_fsync)
    monkeypatch.setattr(rap_render_server.os, "close", tracked_close)
    monkeypatch.setattr(rap_render_server.os, "replace", tracked_replace)

    rap_render_server._ArtifactStore._atomic_write(output, b"durable")

    labels = [name for name, _fd in events]
    assert output.read_bytes() == b"durable"
    assert labels.index("replace") < labels.index("directory_fsync")
    assert labels.index("directory_fsync") < labels.index("directory_close")


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
    assert response.json() == {
        "error": {"code": code, "message": "rap chunk render could not be completed"}
    }
    assert not (tmp_path / "artifacts" / request.request_id / "response.zip").exists()


def test_malformed_request_and_unexpected_failure_are_sanitized(tmp_path: Path) -> None:
    client = _client(
        tmp_path,
        FakeOrchestrator(tmp_path / "worker", RuntimeError("Bearer top-secret prompt")),
    )

    malformed = client.post(
        "/v1/rap/chunks/render",
        content=b"{}",
        headers={"Content-Type": "application/json", "Idempotency-Key": "x"},
    )
    unexpected = _post(client, _request())

    assert malformed.status_code == 422
    assert malformed.json() == {
        "error": {"code": "invalid_request", "message": "invalid rap chunk request"}
    }
    assert unexpected.status_code == 500
    assert unexpected.json() == {
        "error": {"code": "internal_error", "message": "rap chunk render failed"}
    }
    assert "top-secret" not in unexpected.text
    assert "prompt" not in unexpected.text


def test_request_parser_calls_from_payload_without_runtime_method_probing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _request()
    orchestrator = FakeOrchestrator(tmp_path / "worker")
    client = _client(tmp_path, orchestrator)

    def reject_probe(cls, payload):
        raise AssertionError("from_dict must not be probed")

    monkeypatch.setattr(
        RemoteRapChunkRequest,
        "from_dict",
        classmethod(reject_probe),
        raising=False,
    )

    response = _post(client, request)

    assert response.status_code == 200
    assert orchestrator.calls == 1


def test_oversized_content_length_is_rejected_before_body_streaming(
    tmp_path: Path,
) -> None:
    request = _request()
    orchestrator = FakeOrchestrator(tmp_path / "worker")
    client = _client(tmp_path, orchestrator)

    response = client.post(
        "/v1/rap/chunks/render",
        content=request.canonical_json_bytes(),
        headers={
            "Content-Type": "application/json",
            "Content-Length": str(64 * 1024 + 1),
            "Idempotency-Key": request.request_id,
        },
    )

    assert response.status_code == 413
    assert response.json() == {
        "error": {
            "code": "request_too_large",
            "message": "rap chunk request exceeds size limit",
        }
    }
    assert orchestrator.calls == 0


def test_streamed_request_overflow_is_rejected_with_bounded_413(
    tmp_path: Path,
) -> None:
    orchestrator = FakeOrchestrator(tmp_path / "worker")
    client = _client(tmp_path, orchestrator)
    chunks = iter((b"{", b"x" * (64 * 1024), b"}"))

    response = client.post(
        "/v1/rap/chunks/render",
        content=chunks,
        headers={"Content-Type": "application/json", "Idempotency-Key": "unused"},
    )

    assert response.status_code == 413
    assert response.json() == {
        "error": {
            "code": "request_too_large",
            "message": "rap chunk request exceeds size limit",
        }
    }
    assert orchestrator.calls == 0


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


def test_cache_io_for_one_request_does_not_hold_global_idempotency_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first_request = _request(session_id="session-a")
    second_request = _request(session_id="session-b")
    orchestrator = FakeOrchestrator(tmp_path / "worker")
    store = rap_render_server._ArtifactStore(tmp_path / "artifacts", orchestrator)
    store.render_or_load(first_request, first_request.canonical_json_bytes())
    store.render_or_load(second_request, second_request.canonical_json_bytes())
    blocked_path = tmp_path / "artifacts" / first_request.request_id / "response.zip"
    cache_read_started = Event()
    release_cache_read = Event()
    second_finished = Event()
    original_read_bytes = Path.read_bytes

    def controlled_read_bytes(path: Path) -> bytes:
        if path == blocked_path:
            cache_read_started.set()
            assert release_cache_read.wait(timeout=5)
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", controlled_read_bytes)

    first = Thread(
        target=store.render_or_load,
        args=(first_request, first_request.canonical_json_bytes()),
    )

    def load_second() -> None:
        store.render_or_load(second_request, second_request.canonical_json_bytes())
        second_finished.set()

    second = Thread(target=load_second)
    first.start()
    assert cache_read_started.wait(timeout=5)
    second.start()
    completed_while_first_read_blocked = second_finished.wait(timeout=0.5)
    release_cache_read.set()
    first.join(timeout=5)
    second.join(timeout=5)

    assert completed_while_first_read_blocked
    assert not first.is_alive()
    assert not second.is_alive()


def test_overlapping_failure_survives_diagnostic_write_failure_for_all_waiters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _request()
    render_failure = PhraseRenderFailed("typed render failure")
    orchestrator = FakeOrchestrator(tmp_path / "worker", render_failure)
    orchestrator.release = Event()
    waiter_started = Event()

    class ObservedFuture(Future):
        def result(self, timeout=None):
            waiter_started.set()
            return super().result(timeout=1)

    def fail_diagnostic_write(self, request_id, error):
        raise OSError("diagnostic disk failure")

    monkeypatch.setattr(rap_render_server, "Future", ObservedFuture)
    monkeypatch.setattr(
        rap_render_server._ArtifactStore,
        "_persist_failure",
        fail_diagnostic_write,
    )
    store = rap_render_server._ArtifactStore(tmp_path / "artifacts", orchestrator)
    failures: dict[str, BaseException] = {}

    def invoke(name: str) -> None:
        try:
            store.render_or_load(request, request.canonical_json_bytes())
        except BaseException as error:
            failures[name] = error

    owner = Thread(target=invoke, args=("owner",))
    waiter = Thread(target=invoke, args=("waiter",))
    owner.start()
    assert orchestrator.started.wait(timeout=5)
    waiter.start()
    assert waiter_started.wait(timeout=5)
    orchestrator.release.set()
    owner.join(timeout=5)
    waiter.join(timeout=5)

    assert not owner.is_alive()
    assert not waiter.is_alive()
    assert failures == {"owner": render_failure, "waiter": render_failure}
    assert orchestrator.calls == 1


def test_completed_request_returns_byte_identical_cached_package(
    tmp_path: Path,
) -> None:
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
    changed_budget = replace(
        request, remaining_budget_ms=request.remaining_budget_ms - 1
    )
    client = _client(tmp_path, FakeOrchestrator(tmp_path / "worker"))

    assert _post(client, request).status_code == 200
    response = _post(client, changed_budget)

    assert response.status_code == 409
    assert response.json() == {
        "error": {
            "code": "idempotency_conflict",
            "message": "request ID is already bound to another request",
        }
    }


def test_same_id_with_different_in_flight_body_is_rejected(tmp_path: Path) -> None:
    request = _request()
    changed_budget = replace(
        request, remaining_budget_ms=request.remaining_budget_ms - 1
    )
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
        [
            "--vllm-model",
            "vllm",
            "--moss-model",
            "moss",
            "--moss-reference-wav",
            "voice.wav",
        ]
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
    assert defaults.vllm_url == "http://127.0.0.1:8000/v1"
    assert status == 2


def test_cli_resolves_default_server_before_composing_resident_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_import = builtins.__import__
    composition_calls: list[rap_render_server.RapRenderServerConfig] = []

    def fail_uvicorn_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "uvicorn":
            raise ImportError("uvicorn unavailable")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fail_uvicorn_import)

    with pytest.raises(ImportError, match="uvicorn unavailable"):
        main(
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
            composition_factory=lambda config: composition_calls.append(config),
        )

    assert composition_calls == []


def test_cli_composes_through_injected_factory_without_model_imports(
    tmp_path: Path,
) -> None:
    calls: dict[str, object] = {}
    close_calls: list[str] = []
    composition = SimpleNamespace(
        orchestrator=FakeOrchestrator(tmp_path / "worker"),
        health={"ready": True},
        close=lambda: close_calls.append("close"),
    )

    def compose(config):
        calls["config"] = config
        return composition

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
    assert close_calls == ["close"]


def test_cli_closes_composed_worker_when_server_raises(tmp_path: Path) -> None:
    close_calls: list[str] = []
    composition = SimpleNamespace(
        orchestrator=FakeOrchestrator(tmp_path / "worker"),
        health={"ready": True},
        close=lambda: close_calls.append("close"),
    )

    def fail_server(*_args, **_kwargs):
        raise RuntimeError("server failed")

    with pytest.raises(RuntimeError, match="server failed"):
        main(
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
            composition_factory=lambda _config: composition,
            serve=fail_server,
        )

    assert close_calls == ["close"]


def test_real_worker_composition_loads_warms_and_owns_resident_components(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: dict[str, object] = {}

    class FakeClientConfig:
        def __init__(self, **kwargs):
            calls["client_config"] = kwargs

    class FakeClient:
        def __init__(self, config):
            calls["client"] = self

        def close(self):
            calls["client_close"] = int(calls.get("client_close", 0)) + 1

    class FakeGenerator:
        def __init__(self, client, **kwargs):
            calls["generator"] = (client, kwargs)

    class FakeAnalyzer:
        def __init__(self):
            calls["analyzer"] = self

    class FakeWeights:
        def __init__(self):
            calls["weights"] = self

    class FakePlanner:
        def __init__(self, generator, analyzer, weights):
            calls["planner"] = (self, generator, analyzer, weights)

    class FakeMossInstance:
        def synthesize(self, request, output_wav):
            calls["moss_warmup"] = (request, output_wav)
            output_wav.write_bytes(b"warm MOSS WAV")
            return SimpleNamespace(model_revision="moss-revision")

        def close(self):
            calls["moss_close"] = int(calls.get("moss_close", 0)) + 1

    moss = FakeMossInstance()

    class FakeMoss:
        @classmethod
        def load(cls, **kwargs):
            calls["moss_load"] = kwargs
            return moss

    class FakeAlignerInstance:
        def warmup(self, source_wav, transcript):
            calls["aligner_warmup"] = (source_wav.read_bytes(), transcript)
            return {"aligner": "MMS_FA", "version": "mms-version", "aligned": True}

        def close(self):
            calls["aligner_close"] = int(calls.get("aligner_close", 0)) + 1

    aligner = FakeAlignerInstance()

    class FakeAligner:
        @classmethod
        def load(cls, **kwargs):
            calls["aligner_load"] = kwargs
            return aligner

    class FakeRenderer:
        def __init__(self, **kwargs):
            calls["renderer"] = (self, kwargs)

    class ComposedOrchestrator:
        def __init__(self, planner, renderer, *, workspace_root):
            calls["orchestrator"] = (self, planner, renderer, workspace_root)

    dependencies = SimpleNamespace(
        LocalChatModelClient=FakeClient,
        LocalChatModelClientConfig=FakeClientConfig,
        IndependentChoiceCandidateGenerator=FakeGenerator,
        CmuProsodyAnalyzer=FakeAnalyzer,
        ScoreWeights=FakeWeights,
        ChunkCandidatePlanner=FakePlanner,
        PersistentMossSynthesizer=FakeMoss,
        MmsForcedAligner=FakeAligner,
        MossAlignedPhraseRenderer=FakeRenderer,
        RapChunkOrchestrator=ComposedOrchestrator,
    )
    monkeypatch.setattr(
        rap_render_server,
        "_load_worker_dependencies",
        lambda: dependencies,
        raising=False,
    )
    monkeypatch.setattr(
        rap_render_server,
        "_probe_vllm",
        lambda base_url, model: {
            "ready": True,
            "status": "serving",
            "identity": "vLLM",
            "version": "vllm-version",
            "model": model,
        },
        raising=False,
    )
    monkeypatch.setattr(
        rap_render_server,
        "_probe_rubberband",
        lambda: {
            "ready": True,
            "status": "available",
            "identity": "Rubber Band",
            "version": "rubberband-version",
        },
        raising=False,
    )
    monkeypatch.setattr(
        rap_render_server,
        "_configure_aligner_cache",
        lambda path: calls.setdefault("aligner_cache", path),
        raising=False,
    )
    config = rap_render_server.RapRenderServerConfig(
        host="127.0.0.1",
        port=8020,
        artifact_root=tmp_path / "artifacts",
        vllm_url="http://127.0.0.1:8000/v1",
        vllm_model="Qwen-test",
        moss_model="MOSS-test",
        moss_device="cuda:1",
        moss_reference_wav=tmp_path / "reference.wav",
        aligner_device="cuda:2",
        aligner_cache=tmp_path / "mms-cache",
        candidate_profile="realtime",
    )

    composition = rap_render_server._compose_real_worker(config)

    assert calls["client_config"] == {
        "base_url": "http://127.0.0.1:8000/v1",
        "model": "Qwen-test",
        "timeout_s": 30.0,
    }
    assert calls["generator"][1] == {  # type: ignore[index]
        "max_tokens_per_choice": 32,
        "temperature": 1.0,
    }
    assert calls["moss_load"] == {
        "model_id": "MOSS-test",
        "device": "cuda:1",
        "reference_wav": tmp_path / "reference.wav",
    }
    assert calls["aligner_cache"] == tmp_path / "mms-cache"
    assert calls["aligner_load"] == {"device": "cuda:2"}
    warmup_request, _warmup_path = calls["moss_warmup"]  # type: ignore[misc]
    assert warmup_request.text == "warm voice"
    assert calls["aligner_warmup"] == (b"warm MOSS WAV", "warm voice")
    renderer, renderer_kwargs = calls["renderer"]  # type: ignore[misc]
    assert renderer_kwargs == {
        "synthesizer": moss,
        "aligner": aligner,
        "rubberband_version": "rubberband-version",
    }
    orchestrator, planner, wired_renderer, workspace_root = calls["orchestrator"]  # type: ignore[misc]
    assert composition.orchestrator is orchestrator
    assert wired_renderer is renderer
    assert workspace_root == tmp_path / "artifacts"
    assert composition.health == {
        "protocol_version": "remote-rap-chunk/v1",
        "schema_version": "streammuse.rap_chunk.v1",
        "ready": True,
        "vllm": {
            "ready": True,
            "status": "serving",
            "identity": "vLLM",
            "version": "vllm-version",
            "model": "Qwen-test",
        },
        "moss": {
            "ready": True,
            "status": "warmed",
            "identity": "PersistentMossSynthesizer",
            "version": "moss-revision",
            "model": "MOSS-test",
            "warmup": "complete",
        },
        "aligner": {
            "ready": True,
            "status": "warmed",
            "identity": "MMS_FA",
            "version": "mms-version",
            "warmup": "complete",
        },
        "rubberband": {
            "ready": True,
            "status": "available",
            "identity": "Rubber Band",
            "version": "rubberband-version",
        },
        "candidate_profile": "realtime",
        "warmup": {"ready": True, "status": "complete"},
    }

    composition.close()
    composition.close()

    assert calls["client_close"] == 1
    assert calls["moss_close"] == 1
    assert calls["aligner_close"] == 1
