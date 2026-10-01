from __future__ import annotations

import builtins
import hashlib
import io
import json
import struct
import time
import wave
import zipfile
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from threading import Event, Lock, Thread
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from streammuse.application.rap.execution import SynthesisExecutionContext
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
    RAP_CHUNK_OPUS_PACKAGE_MEDIA_TYPE,
    decode_chunk_package,
)
from streammuse.infrastructure.rap.producer_manifest import ProducerManifestV1
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


def _producer_manifest(*, backend: str = "inprocess") -> ProducerManifestV1:
    return ProducerManifestV1(
        backend=backend,
        backend_implementation_revision="test-adapter.v1",
        streammuse_revision={"commit": "test-commit", "patch_sha256": "1" * 64},
        model={"id": "test-moss", "revision": "test-model-revision"},
        runtime={
            "identity": "test-runtime",
            "version": "1.0",
            "revision": "test-runtime-revision",
            "config_sha256": "2" * 64,
        },
        generation={
            "seed_policy_version": "streammuse.moss_seed.v1",
            "settings": {"audio_top_k": 25},
        },
        reference={"audio_sha256": "3" * 64, "text_sha256": None},
        alignment={
            "identity": "MMS_FA",
            "version": "test",
            "warp_policy": "gentle_sparse_r3",
        },
        output={
            "sample_rate_hz": 24_000,
            "wire_audio_codec": "pcm",
            "public_schema_version": "streammuse.rap_chunk.v1",
        },
    )


def _namespace(root: Path) -> Path:
    return root / _producer_manifest().fingerprint


class _PublicationInterrupted(BaseException):
    pass


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

    def render(
        self,
        request: RemoteRapChunkRequest,
        *,
        execution=None,
    ) -> RemoteChunkRenderArtifact:
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
            producer_manifest=_producer_manifest(),
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


def _sglang_cli_args(tmp_path: Path) -> list[str]:
    return [
        "--artifact-root",
        str(tmp_path / "artifacts"),
        "--vllm-model",
        "Qwen-test",
        "--moss-model",
        "OpenMOSS-Team/MOSS-TTS-v1.5",
        "--moss-reference-wav",
        str(tmp_path / "reference.wav"),
        "--moss-serving-backend",
        "sglang-omni",
        "--moss-sglang-url",
        "http://127.0.0.1:8030",
        "--moss-reference-text-file",
        str(tmp_path / "reference.txt"),
        "--moss-sglang-reference-uri",
        "file:///models/streammuse/reference.wav",
        "--moss-sglang-reference-sha256",
        "1" * 64,
        "--moss-model-revision",
        "moss-snapshot-20260903",
        "--moss-runtime-version",
        "sglang-omni-0.1.4",
        "--moss-runtime-revision",
        "omni-commit-a1",
        "--moss-sglang-version",
        "sglang-0.5.2",
        "--moss-sglang-revision",
        "sglang-commit-b2",
        "--moss-runtime-environment-sha256",
        "2" * 64,
        "--moss-runtime-config",
        str(tmp_path / "moss_tts.yaml"),
        "--moss-runtime-config-sha256",
        "3" * 64,
    ]


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
        producer_manifest=_producer_manifest(),
        artifact_root=tmp_path / "artifacts",
    )

    response = TestClient(app).get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "protocol_version": "remote-rap-chunk/v1",
        "schema_version": "streammuse.rap_chunk.v1",
        "ready": True,
        "state": "ready",
    }


def test_health_uses_recursive_allowlists_for_public_scalar_summaries(
    tmp_path: Path,
) -> None:
    secret = "must-not-escape"
    health = {
        "protocol_version": "remote-rap-chunk/v1",
        "schema_version": "streammuse.rap_chunk.v1",
        "ready": True,
        "state": "ready",
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
        producer_manifest=_producer_manifest(),
        artifact_root=tmp_path / "artifacts",
    )

    response = TestClient(app).get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "protocol_version": "remote-rap-chunk/v1",
        "schema_version": "streammuse.rap_chunk.v1",
        "ready": True,
        "state": "ready",
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


def test_health_bounds_types_and_never_exposes_private_model_paths(
    tmp_path: Path,
) -> None:
    private_model = tmp_path / "private-models" / "MOSS-v1.5"
    long_status = "w" * 512
    app = create_rap_render_app(
        FakeOrchestrator(tmp_path / "worker"),
        {
            "ready": 1,
            "candidate_profile": "realtime",
            "vllm": {
                "ready": True,
                "identity": str(tmp_path / "private" / "vllm"),
                "model": "Qwen/test-model",
            },
            "moss": {
                "ready": True,
                "status": long_status,
                "identity": "PersistentMossSynthesizer",
                "version": float("nan"),
                "model": str(private_model),
                "profile": "resident",
            },
            "aligner": {"ready": True, "version": float("inf")},
            "rubberband": {"ready": "true", "version": "3.3.0"},
            "warmup": {"ready": True, "status": "complete", "elapsed": 1.0},
        },
        producer_manifest=_producer_manifest(),
        artifact_root=tmp_path / "artifacts",
    )

    response = TestClient(app).get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "protocol_version": "remote-rap-chunk/v1",
        "schema_version": "streammuse.rap_chunk.v1",
        "ready": False,
        "state": "ready",
        "candidate_profile": "realtime",
        "vllm": {"ready": True, "model": "Qwen/test-model"},
        "moss": {
            "ready": True,
            "status": "w" * 128,
            "identity": "PersistentMossSynthesizer",
            "model": "MOSS-v1.5",
            "profile": "resident",
        },
        "aligner": {"ready": True},
        "rubberband": {"version": "3.3.0"},
        "warmup": {"ready": True, "status": "complete"},
    }
    assert str(tmp_path) not in response.text
    assert "private-models" not in response.text


def test_render_returns_canonical_binary_package_and_atomic_artifacts(
    tmp_path: Path,
) -> None:
    request = _request()
    orchestrator = FakeOrchestrator(tmp_path / "worker")
    client = _client(tmp_path, orchestrator)

    response = _post(client, request)

    workspace = _namespace(tmp_path / "artifacts") / request.request_id
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


def test_opus_enabled_server_returns_separate_variant_without_changing_canonical_cache(
    tmp_path: Path,
) -> None:
    class Codec:
        encoder_identity = "ffmpeg test / libopus"

        def encode_pcm16_mono_24khz(self, pcm: bytes, *, expected_frame_count: int) -> bytes:
            assert len(pcm) == expected_frame_count * 2
            return b"encoded-opus"

        def decode_to_pcm16_mono_24khz(self, encoded: bytes, *, expected_frame_count: int) -> bytes:
            assert encoded == b"encoded-opus"
            return struct.pack("<h", 1_000) * expected_frame_count

    request = _request()
    app = create_rap_render_app(
        FakeOrchestrator(tmp_path / "worker"),
        {"ready": True},
        producer_manifest=_producer_manifest(),
        artifact_root=tmp_path / "artifacts",
        wire_audio_codec="opus",
        opus_codec=Codec(),
    )
    client = TestClient(app)

    pcm_response = _post(client, request)

    response = client.post(
        "/v1/rap/chunks/render",
        content=request.canonical_json_bytes(),
        headers={
            "Content-Type": "application/json",
            "Idempotency-Key": request.request_id,
            "Accept": f"{RAP_CHUNK_OPUS_PACKAGE_MEDIA_TYPE}, {RAP_CHUNK_PACKAGE_MEDIA_TYPE};q=0.9",
        },
    )

    workspace = _namespace(tmp_path / "artifacts") / request.request_id
    assert pcm_response.status_code == 200
    assert pcm_response.headers["content-type"] == RAP_CHUNK_PACKAGE_MEDIA_TYPE
    assert response.status_code == 200
    assert response.headers["content-type"] == RAP_CHUNK_OPUS_PACKAGE_MEDIA_TYPE
    assert (workspace / "response.zip").is_file()
    assert (workspace / "response.opus.zip").read_bytes() == response.content


@pytest.mark.parametrize("quality", ("0", "0.0", "0.00", "not-a-number"))
def test_opus_accept_zero_or_malformed_quality_falls_back_to_pcm(
    tmp_path: Path, quality: str
) -> None:
    class Codec:
        encoder_identity = "ffmpeg test / libopus"

        def encode_pcm16_mono_24khz(self, pcm: bytes, *, expected_frame_count: int) -> bytes:
            return b"encoded-opus"

    request = _request()
    app = create_rap_render_app(
        FakeOrchestrator(tmp_path / "worker"),
        {"ready": True},
        producer_manifest=_producer_manifest(),
        artifact_root=tmp_path / "artifacts",
        wire_audio_codec="opus",
        opus_codec=Codec(),
    )

    response = TestClient(app).post(
        "/v1/rap/chunks/render",
        content=request.canonical_json_bytes(),
        headers={
            "Content-Type": "application/json",
            "Idempotency-Key": request.request_id,
            "Accept": f"{RAP_CHUNK_OPUS_PACKAGE_MEDIA_TYPE};q={quality}",
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == RAP_CHUNK_PACKAGE_MEDIA_TYPE


def test_opus_variant_encoding_uses_threadpool_and_sanitizes_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    original_run_in_threadpool = rap_render_server.run_in_threadpool

    async def observed_run_in_threadpool(func, *args, **kwargs):
        calls.append(func.__name__)
        return await original_run_in_threadpool(func, *args, **kwargs)

    class Codec:
        encoder_identity = "ffmpeg test / libopus"

        def encode_pcm16_mono_24khz(self, pcm: bytes, *, expected_frame_count: int) -> bytes:
            raise RuntimeError("private encoder path=/secret")

    monkeypatch.setattr(rap_render_server, "run_in_threadpool", observed_run_in_threadpool)
    request = _request()
    app = create_rap_render_app(
        FakeOrchestrator(tmp_path / "worker"),
        {"ready": True},
        producer_manifest=_producer_manifest(),
        artifact_root=tmp_path / "artifacts",
        wire_audio_codec="opus",
        opus_codec=Codec(),
    )

    response = TestClient(app, raise_server_exceptions=False).post(
        "/v1/rap/chunks/render",
        content=request.canonical_json_bytes(),
        headers={
            "Content-Type": "application/json",
            "Idempotency-Key": request.request_id,
            "Accept": RAP_CHUNK_OPUS_PACKAGE_MEDIA_TYPE,
        },
    )

    assert calls == ["load_or_create_opus"]
    assert response.status_code == 500
    assert "secret" not in response.text


def test_first_opus_encodes_for_unrelated_requests_do_not_share_global_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first_entered = Event()
    second_entered = Event()
    same_request_waiting = Event()
    release = Event()
    encode_calls = 0

    class Codec:
        encoder_identity = "ffmpeg test / libopus"

        def encode_pcm16_mono_24khz(self, pcm: bytes, *, expected_frame_count: int) -> bytes:
            nonlocal encode_calls
            encode_calls += 1
            (first_entered if encode_calls == 1 else second_entered).set()
            assert release.wait(timeout=5)
            return b"encoded-opus"

    store = rap_render_server._ArtifactStore(
        tmp_path / "artifacts",
        FakeOrchestrator(tmp_path / "worker"),
        _producer_manifest(),
        opus_codec=Codec(),
    )
    first_request = _request(session_id="first")
    second_request = _request(session_id="second")
    first_stored = store.render_or_load(
        first_request, first_request.canonical_json_bytes()
    )
    second_stored = store.render_or_load(
        second_request, second_request.canonical_json_bytes()
    )

    class ObservedFuture(Future):
        def result(self, timeout=None):
            same_request_waiting.set()
            return super().result(timeout=timeout)

    monkeypatch.setattr(rap_render_server, "Future", ObservedFuture)
    first = Thread(target=store.load_or_create_opus, args=(first_request, first_stored))
    same_request = Thread(
        target=store.load_or_create_opus, args=(first_request, first_stored)
    )
    second = Thread(target=store.load_or_create_opus, args=(second_request, second_stored))
    first.start()
    assert first_entered.wait(timeout=1)
    same_request.start()
    assert same_request_waiting.wait(timeout=1)
    second.start()
    second_started_while_first_blocked = second_entered.wait(timeout=0.5)
    release.set()
    first.join(timeout=5)
    same_request.join(timeout=5)
    second.join(timeout=5)

    assert second_started_while_first_blocked
    assert encode_calls == 2
    assert not first.is_alive()
    assert not same_request.is_alive()
    assert not second.is_alive()


def test_canonical_regeneration_invalidates_stale_opus_derivative(tmp_path: Path) -> None:
    class Codec:
        encoder_identity = "ffmpeg test / libopus"

        def encode_pcm16_mono_24khz(self, pcm: bytes, *, expected_frame_count: int) -> bytes:
            return b"encoded-opus"

    request = _request()
    orchestrator = FakeOrchestrator(tmp_path / "worker")
    store = rap_render_server._ArtifactStore(
        tmp_path / "artifacts",
        orchestrator,
        _producer_manifest(),
        opus_codec=Codec(),
    )
    stored = store.render_or_load(request, request.canonical_json_bytes())
    store.load_or_create_opus(request, stored)
    workspace = store.namespace_root / request.request_id
    (workspace / "response.zip").unlink()
    (workspace / "complete.v1.json").unlink()

    store.render_or_load(request, request.canonical_json_bytes())

    assert not (workspace / "response.opus.zip").exists()
    assert orchestrator.calls == 2


def test_overlapping_old_opus_encode_cannot_replace_new_canonical_derivative(
    tmp_path: Path,
) -> None:
    old_entered = Event()
    new_entered = Event()
    release_old = Event()
    encode_calls = 0

    class Codec:
        encoder_identity = "ffmpeg test / libopus"

        def encode_pcm16_mono_24khz(self, pcm: bytes, *, expected_frame_count: int) -> bytes:
            nonlocal encode_calls
            encode_calls += 1
            if encode_calls == 1:
                old_entered.set()
                assert release_old.wait(timeout=5)
                return b"old-opus"
            new_entered.set()
            return b"new-opus"

    request = _request()
    orchestrator = FakeOrchestrator(tmp_path / "worker")
    store = rap_render_server._ArtifactStore(
        tmp_path / "artifacts",
        orchestrator,
        _producer_manifest(),
        opus_codec=Codec(),
    )
    old_stored = store.render_or_load(request, request.canonical_json_bytes())
    workspace = store.namespace_root / request.request_id
    old_results: list[bytes] = []
    new_results: list[bytes] = []
    errors: list[BaseException] = []

    def encode(stored, results: list[bytes]) -> None:
        try:
            results.append(store.load_or_create_opus(request, stored))
        except BaseException as error:
            errors.append(error)

    old_thread = Thread(target=encode, args=(old_stored, old_results))
    old_thread.start()
    assert old_entered.wait(timeout=1)

    new_workspace = tmp_path / "worker" / request.request_id
    new_workspace.mkdir(parents=True, exist_ok=True)
    new_artifact = _artifact(request, new_workspace)
    new_vocal = new_artifact.vocal_wav[:-2] + struct.pack("<h", 2_000)
    new_artifact = replace(
        new_artifact,
        vocal_wav=new_vocal,
        manifest=replace(
            new_artifact.manifest,
            vocal_sha256=hashlib.sha256(new_vocal).hexdigest(),
        ),
    )
    orchestrator.result = new_artifact
    (workspace / "response.zip").unlink()
    (workspace / "complete.v1.json").unlink()
    new_stored = store.render_or_load(request, request.canonical_json_bytes())

    new_thread = Thread(target=encode, args=(new_stored, new_results))
    new_thread.start()
    new_started_before_old_finished = new_entered.wait(timeout=0.5)
    release_old.set()
    old_thread.join(timeout=5)
    new_thread.join(timeout=5)

    assert new_started_before_old_finished
    assert not old_thread.is_alive()
    assert not new_thread.is_alive()
    assert not errors
    assert len(old_results) == 1
    assert len(new_results) == 1
    assert (workspace / "response.opus.zip").read_bytes() == new_results[0]
    assert new_results[0] != old_results[0]


def test_packaging_timing_uses_measured_final_publication_equivalent(
    tmp_path: Path,
) -> None:
    request = _request()
    orchestrator = FakeOrchestrator(tmp_path / "worker")
    clock_values = iter((10.000, 10.002, 10.005, 10.006))
    clock_calls: list[float] = []

    def clock() -> float:
        value = next(clock_values)
        clock_calls.append(value)
        return value

    store = rap_render_server._ArtifactStore(
        tmp_path / "artifacts", orchestrator, _producer_manifest(), clock=clock
    )

    stored = store.render_or_load(request, request.canonical_json_bytes())

    workspace = store.namespace_root / request.request_id
    decoded = decode_chunk_package(
        stored.package, expected_request_id=request.request_id
    )
    timings = decoded.manifest.diagnostics.stage_timings_ms
    persisted_manifest = json.loads(
        (workspace / "manifest.json").read_text(encoding="utf-8")
    )
    persisted_server_timing = json.loads(
        (workspace / "server_timing.json").read_text(encoding="utf-8")
    )
    assert clock_calls == [10.000, 10.002, 10.005, 10.006]
    assert timings["packaging"] == pytest.approx(9.0)
    assert timings["total"] == pytest.approx(24.0)
    assert persisted_manifest == decoded.manifest.to_payload()
    assert persisted_server_timing == {"server_timing": "total;dur=24.000"}
    assert stored.server_timing == "total;dur=24.000"
    assert (workspace / "response.zip").read_bytes() == stored.package
    assert not tuple(workspace.glob(".*measurement*"))


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

    workspace = _namespace(tmp_path / "artifacts") / request.request_id
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


def test_shared_renderer_workspace_fsyncs_canonical_task3_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _request()
    artifact_root = tmp_path / "artifacts"
    orchestrator = FakeOrchestrator(_namespace(artifact_root))
    opened_paths: dict[int, Path] = {}
    fsynced_paths: list[Path] = []
    original_open = rap_render_server.os.open
    original_fsync = rap_render_server.os.fsync

    def tracked_open(path, flags, *args, **kwargs):
        fd = original_open(path, flags, *args, **kwargs)
        opened_paths[fd] = Path(path)
        return fd

    def tracked_fsync(fd):
        if fd in opened_paths:
            fsynced_paths.append(opened_paths[fd])
        return original_fsync(fd)

    monkeypatch.setattr(rap_render_server.os, "open", tracked_open)
    monkeypatch.setattr(rap_render_server.os, "fsync", tracked_fsync)
    store = rap_render_server._ArtifactStore(
        artifact_root, orchestrator, _producer_manifest()
    )

    store.render_or_load(request, request.canonical_json_bytes())

    workspace = store.namespace_root / request.request_id
    assert {
        workspace / "source.wav",
        workspace / "mms_alignment.json",
        workspace / "vocal.wav",
        workspace,
    }.issubset(set(fsynced_paths))
    assert (workspace / "source.wav").read_bytes() == b"source wav"
    assert (workspace / "mms_alignment.json").read_bytes() == (
        _FULL_MMS_ALIGNMENT_BYTES
    )
    assert (workspace / "vocal.wav").read_bytes() == _wav(request)


@pytest.mark.parametrize("failure_type", (OSError, _PublicationInterrupted))
def test_post_rename_publication_failure_unpublishes_and_retries_for_all_waiters(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_type: type[BaseException],
) -> None:
    request = _request()
    orchestrator = FakeOrchestrator(tmp_path / "worker")
    orchestrator.release = Event()
    waiter_started = Event()
    publication_renamed = Event()
    failure = failure_type("response publication interrupted")
    original_replace = rap_render_server.os.replace
    original_fsync_directory = rap_render_server._ArtifactStore._fsync_directory
    failed_once = False
    rollback_fsync_states: list[bool] = []
    result_calls = 0
    result_calls_lock = Lock()

    class ObservedFuture(Future):
        def result(self, timeout=None):
            nonlocal result_calls
            with result_calls_lock:
                result_calls += 1
                if result_calls == 2:
                    waiter_started.set()
            return super().result(timeout=5)

    def replace_then_fail(source, destination):
        nonlocal failed_once
        if Path(destination).name == "response.zip" and not failed_once:
            failed_once = True
            original_replace(source, destination)
            publication_renamed.set()
            raise failure
        return original_replace(source, destination)

    def tracked_fsync_directory(path: Path) -> None:
        if publication_renamed.is_set():
            response_path = Path(path) / "response.zip"
            rollback_fsync_states.append(response_path.exists())
        original_fsync_directory(path)

    monkeypatch.setattr(rap_render_server, "Future", ObservedFuture)
    monkeypatch.setattr(rap_render_server.os, "replace", replace_then_fail)
    monkeypatch.setattr(
        rap_render_server._ArtifactStore,
        "_fsync_directory",
        staticmethod(tracked_fsync_directory),
    )
    store = rap_render_server._ArtifactStore(
        tmp_path / "artifacts", orchestrator, _producer_manifest()
    )
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

    workspace = store.namespace_root / request.request_id
    assert not owner.is_alive()
    assert not waiter.is_alive()
    assert failures["owner"] is failure
    assert failures["waiter"] is failure
    assert not (workspace / "response.zip").exists()
    assert False in rollback_fsync_states
    assert orchestrator.calls == 1

    recovered = store.render_or_load(request, request.canonical_json_bytes())

    assert (workspace / "response.zip").read_bytes() == recovered.package
    assert orchestrator.calls == 2


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
    assert not (
        _namespace(tmp_path / "artifacts") / request.request_id / "response.zip"
    ).exists()


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
    store = rap_render_server._ArtifactStore(
        tmp_path / "artifacts", orchestrator, _producer_manifest()
    )
    store.render_or_load(first_request, first_request.canonical_json_bytes())
    store.render_or_load(second_request, second_request.canonical_json_bytes())
    blocked_path = store.namespace_root / first_request.request_id / "response.zip"
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
    result_calls = 0
    result_calls_lock = Lock()

    class ObservedFuture(Future):
        def result(self, timeout=None):
            nonlocal result_calls
            with result_calls_lock:
                result_calls += 1
                if result_calls == 2:
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
    store = rap_render_server._ArtifactStore(
        tmp_path / "artifacts", orchestrator, _producer_manifest()
    )
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


def test_private_moss_sidecar_is_strict_and_never_enters_public_package(
    tmp_path: Path,
) -> None:
    request = _request()
    orchestrator = FakeOrchestrator(tmp_path / "worker")
    client = _client(tmp_path, orchestrator)

    first = _post(client, request)

    workspace = _namespace(tmp_path / "artifacts") / request.request_id
    sidecar_path = workspace / "internal" / "moss_synthesis.v1.json"
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert sidecar["schema_version"] == "streammuse.moss_synthesis.v1"
    assert sidecar["producer_fingerprint"] == _producer_manifest().fingerprint
    assert sidecar["moss_backend"] == "inprocess"
    assert sidecar["cache_hit"] is False
    assert sidecar["synthesis_outcome"] == "success"
    assert sidecar["upstream_abort_confirmed"] is False
    assert sidecar["cancellation_grace_exceeded"] is False
    assert sidecar["recovery_outcome"] == "not_required"
    with zipfile.ZipFile(io.BytesIO(first.content)) as package:
        assert all(not name.startswith("internal/") for name in package.namelist())

    sidecar["moss_backend"] = "sglang-omni"
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    second = _post(client, request)

    assert second.status_code == 500
    assert second.json() == {
        "error": {
            "code": "internal_error",
            "message": "rap chunk render failed",
        }
    }
    assert orchestrator.calls == 1


def test_same_request_is_isolated_across_producer_namespaces(tmp_path: Path) -> None:
    request = _request()
    artifact_root = tmp_path / "artifacts"
    first_orchestrator = FakeOrchestrator(tmp_path / "worker-a")
    second_orchestrator = FakeOrchestrator(tmp_path / "worker-b")
    first_manifest = _producer_manifest(backend="inprocess")
    second_manifest = replace(
        _producer_manifest(backend="sglang-omni"),
        backend_implementation_revision="test-sglang-adapter.v1",
    )
    first_client = TestClient(
        create_rap_render_app(
            first_orchestrator,
            {"ready": True},
            producer_manifest=first_manifest,
            artifact_root=artifact_root,
        )
    )
    second_client = TestClient(
        create_rap_render_app(
            second_orchestrator,
            {"ready": True},
            producer_manifest=second_manifest,
            artifact_root=artifact_root,
        )
    )

    assert _post(first_client, request).status_code == 200
    assert _post(second_client, request).status_code == 200

    assert first_orchestrator.calls == 1
    assert second_orchestrator.calls == 1
    assert (
        artifact_root / first_manifest.fingerprint / request.request_id
    ).is_dir()
    assert (
        artifact_root / second_manifest.fingerprint / request.request_id
    ).is_dir()


def test_one_waiter_can_leave_without_cancelling_shared_owner(tmp_path: Path) -> None:
    request = _request()
    orchestrator = FakeOrchestrator(tmp_path / "worker")
    orchestrator.release = Event()
    app = create_rap_render_app(
        orchestrator,
        {"ready": True},
        producer_manifest=_producer_manifest(),
        artifact_root=tmp_path / "artifacts",
    )
    store = app.state.rap_artifact_store
    first_execution = SynthesisExecutionContext.from_timeout(
        5.0, correlation_id=request.request_id
    )
    second_execution = SynthesisExecutionContext.from_timeout(
        5.0, correlation_id=request.request_id
    )

    first = store.join(
        request,
        request.canonical_json_bytes(),
        execution=first_execution,
    )
    assert orchestrator.started.wait(timeout=5)
    second = store.join(
        request,
        request.canonical_json_bytes(),
        execution=second_execution,
    )
    first_execution.cancel("client_disconnected")
    first.detach("client_disconnected")

    assert first.state.owner_execution.cancelled is False
    orchestrator.release.set()
    assert second.future.result(timeout=5).package
    second.detach()
    assert orchestrator.calls == 1
    store.close()


def test_last_waiter_cancel_degrades_when_owner_ignores_grace(
    tmp_path: Path,
) -> None:
    request = _request()
    orchestrator = FakeOrchestrator(tmp_path / "worker")
    orchestrator.release = Event()
    app = create_rap_render_app(
        orchestrator,
        {"ready": True},
        producer_manifest=_producer_manifest(),
        artifact_root=tmp_path / "artifacts",
        cancellation_grace_seconds=0.02,
    )
    store = app.state.rap_artifact_store
    execution = SynthesisExecutionContext.from_timeout(
        5.0, correlation_id=request.request_id
    )
    waiter = store.join(
        request,
        request.canonical_json_bytes(),
        execution=execution,
    )
    assert orchestrator.started.wait(timeout=5)

    execution.cancel("client_disconnected")
    waiter.detach("client_disconnected")
    deadline = time.monotonic() + 1.0
    while (
        app.state.rap_runtime_state.snapshot()["state"] != "degraded"
        and time.monotonic() < deadline
    ):
        time.sleep(0.005)

    health = app.state.rap_runtime_state.snapshot()
    assert health["ready"] is False
    assert health["state"] == "degraded"
    assert health["restart_required"] is True
    with pytest.raises(rap_render_server._ServiceDegraded):
        store.join(
            request,
            request.canonical_json_bytes(),
            execution=SynthesisExecutionContext.from_timeout(
                5.0, correlation_id=request.request_id
            ),
        )

    orchestrator.release.set()
    with pytest.raises(Exception):
        waiter.future.result(timeout=5)
    workspace = _namespace(tmp_path / "artifacts") / request.request_id
    sidecar = json.loads(
        (workspace / "internal" / "moss_synthesis.v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert sidecar["upstream_abort_confirmed"] is False
    assert sidecar["cancellation_grace_exceeded"] is True
    assert sidecar["recovery_outcome"] == "restart_required"
    store.close()


def test_unconfirmed_transport_close_degrades_after_local_owner_exits(
    tmp_path: Path,
) -> None:
    request = _request()

    class AbortUnconfirmedOrchestrator(FakeOrchestrator):
        def render(self, request, *, execution=None):
            assert execution is not None
            self.calls += 1
            self.started.set()
            execution.add_cancel_callback(
                lambda: execution.record_cancellation_outcome(
                    "transport_closed_abort_unconfirmed"
                )
            )
            while not execution.cancelled:
                time.sleep(0.001)
            execution.checkpoint()

    orchestrator = AbortUnconfirmedOrchestrator(tmp_path / "worker")
    manifest = _producer_manifest(backend="sglang-omni")
    app = create_rap_render_app(
        orchestrator,
        {"ready": True},
        producer_manifest=manifest,
        artifact_root=tmp_path / "artifacts",
        cancellation_grace_seconds=0.03,
    )
    store = app.state.rap_artifact_store
    execution = SynthesisExecutionContext.from_timeout(
        5.0, correlation_id=request.request_id
    )
    waiter = store.join(
        request,
        request.canonical_json_bytes(),
        execution=execution,
    )
    assert orchestrator.started.wait(timeout=5)

    execution.cancel("client_disconnected")
    waiter.detach("client_disconnected")
    with pytest.raises(Exception):
        waiter.future.result(timeout=5)
    deadline = time.monotonic() + 1.0
    while (
        app.state.rap_runtime_state.snapshot()["state"] != "degraded"
        and time.monotonic() < deadline
    ):
        time.sleep(0.005)

    assert app.state.rap_runtime_state.snapshot()["state"] == "degraded"
    sidecar = json.loads(
        (
            tmp_path
            / "artifacts"
            / manifest.fingerprint
            / request.request_id
            / "internal"
            / "moss_synthesis.v1.json"
        ).read_text(encoding="utf-8")
    )
    assert sidecar["cancellation_outcome"] == (
        "transport_closed_abort_unconfirmed"
    )
    assert sidecar["cancellation_grace_exceeded"] is True
    assert sidecar["recovery_outcome"] == "restart_required"
    store.close()


@pytest.mark.parametrize(
    "evidence",
    (
        {
            "cancellation_outcome": "not_requested",
            "recovery_outcome": "abort_unconfirmed",
        },
        {
            "cancellation_outcome": "cancel_requested",
            "recovery_outcome": "not_required",
        },
        {
            "cancellation_outcome": "upstream_abort_confirmed",
            "upstream_abort_confirmed": True,
            "recovery_outcome": "abort_unconfirmed",
        },
    ),
)
def test_private_sidecar_rejects_inconsistent_recovery_evidence(
    evidence: dict[str, object],
) -> None:
    with pytest.raises(
        rap_render_server._CacheIntegrityError,
        match="recovery evidence is inconsistent",
    ):
        rap_render_server._private_moss_metadata(
            evidence,
            backend="sglang-omni",
            correlation_id="request-id",
            model_revision="model-revision",
            reference_audio_sha256="1" * 64,
            reference_text_sha256="2" * 64,
            config_sha256="3" * 64,
        )


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
    assert defaults.moss_warp_policy == "gentle_sparse_r3"
    assert status == 2


def test_sglang_cli_requires_and_preserves_complete_pinned_identity(
    tmp_path: Path,
) -> None:
    args = build_parser().parse_args(_sglang_cli_args(tmp_path))

    config = rap_render_server._server_config_from_args(args)

    assert config.moss_serving_backend == "sglang-omni"
    assert config.moss_device == "external"
    assert config.moss_sglang_url == "http://127.0.0.1:8030"
    assert config.moss_runtime_version == "sglang-omni-0.1.4"
    assert config.moss_runtime_revision == "omni-commit-a1"
    assert config.moss_sglang_version == "sglang-0.5.2"
    assert config.moss_sglang_revision == "sglang-commit-b2"
    assert config.moss_runtime_environment_sha256 == "2" * 64


def test_sglang_cli_rejects_missing_pin_public_endpoint_and_remote_reference(
    tmp_path: Path,
) -> None:
    complete = _sglang_cli_args(tmp_path)
    missing_environment = complete[:]
    index = missing_environment.index("--moss-runtime-environment-sha256")
    del missing_environment[index : index + 2]
    with pytest.raises(ValueError, match="requires"):
        rap_render_server._server_config_from_args(
            build_parser().parse_args(missing_environment)
        )

    public_endpoint = complete[:]
    index = public_endpoint.index("--moss-sglang-url")
    public_endpoint[index + 1] = "https://moss.example.test"
    with pytest.raises(ValueError, match="HTTP origin"):
        rap_render_server._server_config_from_args(
            build_parser().parse_args(public_endpoint)
        )

    remote_reference = complete[:]
    index = remote_reference.index("--moss-sglang-reference-uri")
    remote_reference[index + 1] = "https://example.test/reference.wav"
    with pytest.raises(ValueError, match="local file URI"):
        rap_render_server._server_config_from_args(
            build_parser().parse_args(remote_reference)
        )


def test_inprocess_cli_rejects_sglang_only_flags(tmp_path: Path) -> None:
    args = build_parser().parse_args(
        [
            "--vllm-model",
            "Qwen-test",
            "--moss-model",
            "MOSS-test",
            "--moss-reference-wav",
            str(tmp_path / "reference.wav"),
            "--moss-runtime-version",
            "should-not-be-here",
        ]
    )

    with pytest.raises(ValueError, match="only be used with sglang-omni"):
        rap_render_server._server_config_from_args(args)


def _mlx_cli_args(tmp_path: Path) -> list[str]:
    return [
        "--vllm-model",
        "qwen-rap",
        "--moss-model",
        "OpenMOSS-Team/MOSS-TTS-v1.5",
        "--moss-reference-wav",
        str(tmp_path / "reference.wav"),
        "--moss-serving-backend",
        "mlx",
        "--moss-mlx-url",
        "http://127.0.0.1:8030",
        "--moss-model-revision",
        "cdd3b911b1585e3f2dbc7775ef10f9926f58850a",
        "--mlx-version",
        "0.32.3",
        "--mlx-audio-version",
        "0.5.7",
        "--mlx-audio-commit",
        "94c7716212b2228f178d2f9c7619a591fd1b0b78",
        "--mlx-moss-quantization",
        "affine-q8-g64",
        "--mlx-moss-weights-sha256",
        "4" * 64,
        "--mlx-moss-audio-tokenizer-revision",
        "3cd226ba2947efa357ef453bcad111b6eafba782",
    ]


def test_mlx_cli_requires_and_preserves_complete_pinned_runtime(tmp_path: Path) -> None:
    config = rap_render_server._server_config_from_args(
        build_parser().parse_args(_mlx_cli_args(tmp_path))
    )

    assert config.moss_serving_backend == "mlx"
    assert config.moss_device == "external"
    assert config.moss_mlx_url == "http://127.0.0.1:8030"
    assert dict(config.moss_mlx_runtime) == {
        "mlx_version": "0.32.3",
        "mlx_audio_version": "0.5.7",
        "mlx_audio_commit": "94c7716212b2228f178d2f9c7619a591fd1b0b78",
        "quantization": "affine-q8-g64",
        "weights_sha256": "4" * 64,
        "audio_tokenizer_revision": "3cd226ba2947efa357ef453bcad111b6eafba782",
    }


@pytest.mark.parametrize(
    "flag",
    (
        "--moss-mlx-url",
        "--mlx-version",
        "--mlx-audio-version",
        "--mlx-audio-commit",
        "--mlx-moss-quantization",
        "--mlx-moss-weights-sha256",
        "--mlx-moss-audio-tokenizer-revision",
        "--moss-model-revision",
    ),
)
def test_mlx_cli_rejects_any_missing_pin(tmp_path: Path, flag: str) -> None:
    args = _mlx_cli_args(tmp_path)
    index = args.index(flag)
    del args[index : index + 2]

    with pytest.raises(ValueError, match=f"mlx requires.*{flag}"):
        rap_render_server._server_config_from_args(build_parser().parse_args(args))


def test_mlx_cli_rejects_public_endpoint_relative_reference_and_mutable_pins(
    tmp_path: Path,
) -> None:
    public = _mlx_cli_args(tmp_path)
    public[public.index("--moss-mlx-url") + 1] = "http://192.168.1.20:8030"
    with pytest.raises(ValueError, match="HTTP origin"):
        rap_render_server._server_config_from_args(build_parser().parse_args(public))

    relative = _mlx_cli_args(tmp_path)
    relative[relative.index("--moss-reference-wav") + 1] = "reference.wav"
    with pytest.raises(ValueError, match="absolute"):
        rap_render_server._server_config_from_args(build_parser().parse_args(relative))

    floating = _mlx_cli_args(tmp_path)
    floating[floating.index("--mlx-audio-commit") + 1] = "main"
    with pytest.raises(ValueError, match="--mlx-audio-commit"):
        rap_render_server._server_config_from_args(build_parser().parse_args(floating))

    bad_hash = _mlx_cli_args(tmp_path)
    bad_hash[bad_hash.index("--mlx-moss-weights-sha256") + 1] = "abc"
    with pytest.raises(ValueError, match="SHA-256"):
        rap_render_server._server_config_from_args(build_parser().parse_args(bad_hash))


def test_mlx_and_sglang_flags_are_mutually_exclusive(tmp_path: Path) -> None:
    mlx_with_sglang = [*_mlx_cli_args(tmp_path), "--moss-sglang-url", "http://127.0.0.1:9000"]
    with pytest.raises(ValueError, match="only be used with sglang-omni"):
        rap_render_server._server_config_from_args(build_parser().parse_args(mlx_with_sglang))

    sglang_with_mlx = [*_sglang_cli_args(tmp_path), "--mlx-version", "0.32.3"]
    with pytest.raises(ValueError, match="only be used with mlx"):
        rap_render_server._server_config_from_args(build_parser().parse_args(sglang_with_mlx))

    inprocess_with_mlx = [
        "--vllm-model",
        "qwen-rap",
        "--moss-model",
        "moss",
        "--moss-reference-wav",
        str(tmp_path / "reference.wav"),
        "--moss-mlx-url",
        "http://127.0.0.1:8030",
    ]
    with pytest.raises(ValueError, match="only be used with mlx"):
        rap_render_server._server_config_from_args(build_parser().parse_args(inprocess_with_mlx))


def test_inprocess_cli_defaults_to_auto_devices(tmp_path: Path) -> None:
    config = rap_render_server._server_config_from_args(
        build_parser().parse_args(
            [
                "--vllm-model",
                "qwen-rap",
                "--moss-model",
                "moss",
                "--moss-reference-wav",
                str(tmp_path / "reference.wav"),
            ]
        )
    )

    assert config.moss_device == "auto"
    assert config.aligner_device == "auto"


@pytest.mark.parametrize(
    ("cuda", "mps", "expected"),
    ((True, True, "cuda"), (False, True, "mps"), (False, False, "cpu")),
)
def test_auto_device_prefers_cuda_then_mps_then_cpu(
    monkeypatch: pytest.MonkeyPatch, cuda: bool, mps: bool, expected: str
) -> None:
    from streammuse.infrastructure.inference import runtime_device

    monkeypatch.setattr(runtime_device.torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(runtime_device, "is_mps_available", lambda: mps)

    assert rap_render_server._resolve_torch_device("auto") == expected
    assert rap_render_server._resolve_torch_device("cuda:1") == "cuda:1"


def test_mlx_producer_identity_is_distinct_from_sglang(tmp_path: Path) -> None:
    mlx_config = rap_render_server._server_config_from_args(
        build_parser().parse_args(_mlx_cli_args(tmp_path))
    )
    sglang_config = rap_render_server._server_config_from_args(
        build_parser().parse_args(_sglang_cli_args(tmp_path))
    )
    common = dict(
        model_revision="cdd3b911b1585e3f2dbc7775ef10f9926f58850a",
        reference_audio_sha256="1" * 64,
        reference_text_sha256=None,
        aligner_identity="MMS_FA",
        aligner_version="mms-version",
    )

    mlx_manifest = rap_render_server._build_producer_manifest(
        mlx_config, aligner_device="mps", **common
    )
    sglang_manifest = rap_render_server._build_producer_manifest(sglang_config, **common)
    other_quantization = rap_render_server._build_producer_manifest(
        replace(
            mlx_config,
            moss_mlx_runtime={**mlx_config.moss_mlx_runtime, "quantization": "affine-q4-g64"},
        ),
        aligner_device="mps",
        **common,
    )

    payload = mlx_manifest.to_payload()
    assert payload["backend"] == "mlx"
    assert payload["runtime"]["identity"] == "MLX/mlx-audio"
    assert payload["runtime"]["weights_sha256"] == "4" * 64
    assert payload["alignment"]["device"] == "mps"
    assert mlx_manifest.fingerprint != sglang_manifest.fingerprint
    assert mlx_manifest.fingerprint != other_quantization.fingerprint


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
        producer_manifest=_producer_manifest(),
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
        producer_manifest=_producer_manifest(),
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
    reference_bytes = _wav(_request())

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
        def synthesize(self, request, output_wav, *, execution):
            calls["moss_warmup"] = (request, output_wav)
            output_wav.write_bytes(reference_bytes)
            return SimpleNamespace(
                model_revision="moss-revision",
                reference_voice_sha256=hashlib.sha256(reference_bytes).hexdigest(),
            )

        def close(self):
            calls["moss_close"] = int(calls.get("moss_close", 0)) + 1

    moss = FakeMossInstance()

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
    def load_inprocess(config):
        calls["moss_load"] = {
            "model_id": config.moss_model,
            "device": config.moss_device,
            "reference_wav": config.moss_reference_wav,
        }
        return moss

    monkeypatch.setattr(
        rap_render_server,
        "_load_inprocess_moss_synthesizer",
        load_inprocess,
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
    reference_wav = tmp_path / "reference.wav"
    reference_wav.write_bytes(reference_bytes)
    config = rap_render_server.RapRenderServerConfig(
        host="127.0.0.1",
        port=8020,
        artifact_root=tmp_path / "artifacts",
        vllm_url="http://127.0.0.1:8000/v1",
        vllm_model="Qwen-test",
        moss_model=str(tmp_path / "private-models" / "MOSS-test"),
        moss_device="cuda:1",
        moss_reference_wav=reference_wav,
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
        "model_id": str(tmp_path / "private-models" / "MOSS-test"),
        "device": "cuda:1",
        "reference_wav": tmp_path / "reference.wav",
    }
    assert calls["aligner_cache"] == tmp_path / "mms-cache"
    assert calls["aligner_load"] == {"device": "cuda:2"}
    warmup_request, _warmup_path = calls["moss_warmup"]  # type: ignore[misc]
    assert warmup_request.text == "warm voice"
    assert calls["aligner_warmup"] == (reference_bytes, "warm voice")
    renderer, renderer_kwargs = calls["renderer"]  # type: ignore[misc]
    assert renderer_kwargs == {
        "synthesizer": moss,
        "aligner": aligner,
        "rubberband_version": "rubberband-version",
        "warp_policy": "gentle_sparse_r3",
    }
    orchestrator, planner, wired_renderer, workspace_root = calls["orchestrator"]  # type: ignore[misc]
    assert composition.orchestrator is orchestrator
    assert wired_renderer is renderer
    assert workspace_root == tmp_path / "artifacts" / composition.producer_manifest.fingerprint
    assert composition.health["ready"] is True
    assert composition.health["state"] == "ready"
    assert composition.health["supported_schema_versions"] == (
        "streammuse.rap_chunk.v1,streammuse.rap_chunk.v2"
    )
    assert composition.health["backend"] == "inprocess"
    assert composition.health["producer_fingerprint"] == composition.producer_manifest.fingerprint
    assert composition.health["vllm"] == {
        "ready": True,
        "status": "serving",
        "identity": "vLLM",
        "version": "vllm-version",
        "model": "Qwen-test",
    }
    moss_health = composition.health["moss"]
    assert moss_health["identity"] == "PersistentMossSynthesizer"
    assert moss_health["model"] == "MOSS-test"
    assert moss_health["model_revision"] == "moss-revision"
    assert moss_health["reference_audio_sha256"] == hashlib.sha256(
        reference_bytes
    ).hexdigest()
    assert moss_health["warmup"] == "complete"
    assert composition.health["aligner"] == {
        "ready": True,
        "status": "warmed",
        "identity": "MMS_FA",
        "version": "mms-version",
        "device": "cuda:2",
        "warmup": "complete",
    }
    assert composition.health["rubberband"] == {
        "ready": True,
        "status": "available",
        "identity": "Rubber Band",
        "version": "rubberband-version",
    }

    composition.close()
    composition.close()

    assert calls["client_close"] == 1
    assert calls["moss_close"] == 1
    assert calls["aligner_close"] == 1


@pytest.mark.parametrize(
    ("valid_warmup", "probe_failure"),
    ((True, False), (False, False), (True, True)),
)
def test_sglang_composition_is_lazy_ordered_and_validates_warmup_audio(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    valid_warmup: bool,
    probe_failure: bool,
) -> None:
    events: list[str] = []
    reference_bytes = _wav(_request())
    reference_wav = tmp_path / "reference.wav"
    reference_wav.write_bytes(reference_bytes)
    reference_text = tmp_path / "reference.txt"
    reference_text.write_text("reference words", encoding="utf-8")
    runtime_config = tmp_path / "moss_tts.yaml"
    runtime_config.write_text("model: pinned\n", encoding="utf-8")

    class Closable:
        def __init__(self, name: str) -> None:
            self.name = name

        def close(self) -> None:
            events.append(f"close:{self.name}")

    client = Closable("client")

    class FakeSynthesizer(Closable):
        def __init__(self) -> None:
            super().__init__("sglang")

        def probe(self):
            events.append("sglang_probe")
            if probe_failure:
                raise RuntimeError("probe failed")
            return {
                "identity": "SGLang-Omni/MOSS-TTS",
                "version": "omni-server-1",
                "model_revision": "moss-snapshot-1",
            }

        def synthesize(self, request, output_wav, *, execution):
            events.append("moss_warmup")
            output_wav.write_bytes(reference_bytes if valid_warmup else b"invalid")
            return SimpleNamespace(
                model_revision="moss-snapshot-1",
                reference_voice_sha256=hashlib.sha256(reference_bytes).hexdigest(),
            )

    synthesizer = FakeSynthesizer()

    class FakeAligner(Closable):
        def __init__(self) -> None:
            super().__init__("aligner")

        def warmup(self, source_wav, transcript):
            events.append("aligner_warmup")
            assert source_wav.read_bytes() == reference_bytes
            assert transcript == "warm voice"
            return {"aligner": "MMS_FA", "version": "mms-1"}

    aligner = FakeAligner()

    def load_aligner(**kwargs):
        assert kwargs == {"device": "cuda:2"}
        events.append("aligner_load")
        return aligner

    class FakeRenderer(Closable):
        def __init__(self, **kwargs) -> None:
            super().__init__("renderer")
            assert kwargs["synthesizer"] is synthesizer
            assert kwargs["aligner"] is aligner

    class FakeOrchestrator:
        def __init__(self, planner, renderer, *, workspace_root) -> None:
            self.workspace_root = workspace_root

    dependencies = SimpleNamespace(
        LocalChatModelClientConfig=lambda **kwargs: kwargs,
        LocalChatModelClient=lambda _config: client,
        IndependentChoiceCandidateGenerator=lambda *_args, **_kwargs: object(),
        CmuProsodyAnalyzer=lambda: object(),
        ScoreWeights=lambda: object(),
        ChunkCandidatePlanner=lambda *_args: object(),
        MmsForcedAligner=SimpleNamespace(load=load_aligner),
        MossAlignedPhraseRenderer=FakeRenderer,
        RapChunkOrchestrator=FakeOrchestrator,
    )
    monkeypatch.setattr(
        rap_render_server,
        "_load_worker_dependencies",
        lambda: dependencies,
    )

    def load_sglang(config, **kwargs):
        events.append("sglang_load")
        assert kwargs["reference_audio_sha256"] == hashlib.sha256(
            reference_bytes
        ).hexdigest()
        assert kwargs["reference_text"] == "reference words"
        return synthesizer

    monkeypatch.setattr(
        rap_render_server,
        "_load_sglang_moss_synthesizer",
        load_sglang,
    )
    monkeypatch.setattr(
        rap_render_server,
        "_load_inprocess_moss_synthesizer",
        lambda _config: pytest.fail("in-process MOSS must remain lazy"),
    )

    def probe_vllm(_url, _model):
        events.append("vllm_probe")
        return {"ready": True, "identity": "vLLM", "version": "vllm-1"}

    def probe_rubberband():
        events.append("r3_probe")
        return {
            "ready": True,
            "identity": "Rubber Band",
            "version": "3.3.0",
        }

    monkeypatch.setattr(rap_render_server, "_probe_vllm", probe_vllm)
    monkeypatch.setattr(rap_render_server, "_probe_rubberband", probe_rubberband)
    config = rap_render_server.RapRenderServerConfig(
        host="127.0.0.1",
        port=8020,
        artifact_root=tmp_path / "artifacts",
        vllm_url="http://127.0.0.1:8000/v1",
        vllm_model="Qwen-test",
        moss_model="OpenMOSS-Team/MOSS-TTS-v1.5",
        moss_device="external",
        moss_reference_wav=reference_wav,
        aligner_device="cuda:2",
        aligner_cache=None,
        candidate_profile="realtime",
        moss_serving_backend="sglang-omni",
        moss_sglang_url="http://127.0.0.1:8030",
        moss_reference_text_file=reference_text,
        moss_sglang_reference_uri="file:///models/streammuse/reference.wav",
        moss_sglang_reference_sha256=hashlib.sha256(reference_bytes).hexdigest(),
        moss_model_revision="moss-snapshot-1",
        moss_runtime_version="sglang-omni-0.1.4",
        moss_runtime_revision="omni-commit-a1",
        moss_sglang_version="sglang-0.5.2",
        moss_sglang_revision="sglang-commit-b2",
        moss_runtime_environment_sha256="4" * 64,
        moss_runtime_config=runtime_config,
        moss_runtime_config_sha256=hashlib.sha256(
            runtime_config.read_bytes()
        ).hexdigest(),
    )

    if probe_failure:
        with pytest.raises(RuntimeError, match="probe failed"):
            rap_render_server._compose_real_worker(config)
        assert "moss_warmup" not in events
        assert events[-2:] == ["close:sglang", "close:client"]
        return

    if not valid_warmup:
        with pytest.raises(RuntimeError, match="invalid audio"):
            rap_render_server._compose_real_worker(config)
        assert "aligner_load" not in events
        assert events[-2:] == ["close:sglang", "close:client"]
        return

    composition = rap_render_server._compose_real_worker(config)

    assert events[:6] == [
        "vllm_probe",
        "sglang_load",
        "sglang_probe",
        "moss_warmup",
        "aligner_load",
        "aligner_warmup",
    ]
    assert events[6] == "r3_probe"
    assert composition.health["backend"] == "sglang-omni"
    assert composition.health["moss"]["model_revision"] == "moss-snapshot-1"
    assert composition.producer_manifest.runtime == {
        "identity": "SGLang-Omni/SGLang",
        "version": "sglang-omni-0.1.4",
        "revision": "omni-commit-a1",
        "sglang_version": "sglang-0.5.2",
        "sglang_revision": "sglang-commit-b2",
        "environment_sha256": "4" * 64,
        "config_sha256": hashlib.sha256(runtime_config.read_bytes()).hexdigest(),
    }
    composition.close()
    composition.close()
    assert events.count("close:sglang") == 1
    assert events.count("close:aligner") == 1
    assert events.count("close:client") == 1


def test_rubberband_probe_runs_a_real_time_map_operation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}

    def run(argv, **kwargs):
        calls["argv"] = argv
        calls["run_kwargs"] = kwargs
        return SimpleNamespace(stdout="rubberband 3.3.0\n", stderr="")

    class FakeStretcher:
        def __init__(self, *, timeout_seconds):
            calls["timeout_seconds"] = timeout_seconds

        def stretch(self, source, target_frames, time_map):
            calls["source_frames"] = source.frame_count
            calls["target_frames"] = target_frames
            calls["time_map"] = time_map
            size = target_frames * source.format.channels * source.format.sample_width_bytes
            data = (source.data * ((size // len(source.data)) + 1))[:size]
            return replace(source, frame_count=target_frames, data=data)

    from streammuse.infrastructure.rap import time_stretch

    monkeypatch.setattr(rap_render_server.subprocess, "run", run)
    monkeypatch.setattr(
        time_stretch,
        "RubberBandTimeMapStretcher",
        FakeStretcher,
    )

    health = rap_render_server._probe_rubberband()

    assert calls["argv"] == ["rubberband", "--version"]
    assert calls["source_frames"] == 2_400
    assert calls["target_frames"] == 2_520
    assert calls["time_map"] == ((0, 0), (2_399, 2_519))
    assert health == {
        "ready": True,
        "status": "available",
        "identity": "Rubber Band",
        "version": "rubberband 3.3.0",
    }


def _v2_request() -> RemoteRapChunkRequest:
    from streammuse.domain.rap import REMOTE_CHUNK_SCHEMA_VERSION_V2

    v1 = _request()
    return RemoteRapChunkRequest.create(
        session_id="session-1",
        chunk_index=0,
        bars=v1.bars,
        tempo_bpm=v1.tempo_bpm,
        remaining_budget_ms=v1.remaining_budget_ms,
        policy=v1.policy,
        context_lines=v1.context_lines,
        seed=v1.seed,
        schema_version=REMOTE_CHUNK_SCHEMA_VERSION_V2,
    )


_V2_SOURCE_FRAMES = 100_000


def _v2_artifact(
    request: RemoteRapChunkRequest, workspace: Path
) -> RemoteChunkRenderArtifact:
    v1 = _artifact(request, workspace)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(24_000)
        output.writeframes(struct.pack("<h", 700) * _V2_SOURCE_FRAMES)
    source_wav = buffer.getvalue()
    base = v1.manifest.diagnostics
    diagnostics = replace(
        base,
        stage_timings_ms={k: v for k, v in base.stage_timings_ms.items() if k != "warp"},
        alignment_diagnostics={
            "fallback_counts": {"word": 0},
            "source_onsets": [0.1, 0.6],
            "onset_confidence": [0.9, 0.8],
        },
        audio_diagnostics={
            "sample_rate_hz": 24_000,
            "frame_count": _V2_SOURCE_FRAMES,
            "duration_seconds": _V2_SOURCE_FRAMES / 24_000,
            "peak": 0.02,
        },
        model_tool_versions={"moss": "test", "aligner": "mms-test"},
        schema_version=request.schema_version,
    )
    manifest = replace(
        v1.manifest,
        diagnostics=diagnostics,
        vocal_sha256=hashlib.sha256(source_wav).hexdigest(),
        schema_version=request.schema_version,
    )
    return replace(v1, manifest=manifest, vocal_wav=source_wav)


class _SourceOnlyOrchestrator(FakeOrchestrator):
    """Mimics the v2 renderer: source + MMS evidence, never a warped vocal."""

    def render(self, request, *, execution=None):
        if request.schema_version.endswith("v1"):
            return super().render(request, execution=execution)
        self.calls += 1
        workspace = self.workspace_root / request.request_id
        workspace.mkdir(parents=True, exist_ok=True)
        artifact = _v2_artifact(request, workspace)
        (workspace / self.source_name).write_bytes(artifact.vocal_wav)
        (workspace / self.alignment_name).write_bytes(_FULL_MMS_ALIGNMENT_BYTES)
        return artifact


def test_v2_render_publishes_the_source_phrase_without_a_server_warp(
    tmp_path: Path,
) -> None:
    request = _v2_request()
    client = _client(tmp_path, _SourceOnlyOrchestrator(tmp_path / "worker"))

    response = _post(client, request)

    assert response.status_code == 200, response.text
    workspace = _namespace(tmp_path / "artifacts") / request.request_id
    decoded = decode_chunk_package(response.content, expected_request_id=request.request_id)
    assert decoded.manifest.schema_version == request.schema_version
    assert decoded.manifest.audio_frame_count == _V2_SOURCE_FRAMES
    assert decoded.vocal_wav == (workspace / "source.wav").read_bytes()
    assert "warp" not in decoded.manifest.diagnostics.stage_timings_ms
    assert decoded.manifest.diagnostics.alignment_diagnostics["source_onsets"] == (0.1, 0.6)
    assert not (workspace / "vocal.wav").exists()
    assert not (workspace / "aligned.wav").exists()
    manifest = json.loads((workspace / "manifest.json").read_text(encoding="utf-8"))
    assert "source_sha256" in manifest and "vocal_sha256" not in manifest


def test_v1_and_v2_requests_for_the_same_chunk_never_share_a_cache_entry(
    tmp_path: Path,
) -> None:
    orchestrator = _SourceOnlyOrchestrator(tmp_path / "worker")
    client = _client(tmp_path, orchestrator)
    v1, v2 = _request(), _v2_request()

    first_v1, first_v2 = _post(client, v1), _post(client, v2)
    second_v1, second_v2 = _post(client, v1), _post(client, v2)

    assert v1.request_id != v2.request_id
    assert orchestrator.calls == 2
    assert first_v1.content == second_v1.content
    assert first_v2.content == second_v2.content
    assert decode_chunk_package(
        first_v1.content, expected_request_id=v1.request_id
    ).manifest.schema_version == v1.schema_version
    assert decode_chunk_package(
        first_v2.content, expected_request_id=v2.request_id
    ).manifest.schema_version == v2.schema_version


def test_public_health_passes_supported_schema_versions_through(tmp_path: Path) -> None:
    supported = "streammuse.rap_chunk.v1,streammuse.rap_chunk.v2"
    client = TestClient(
        create_rap_render_app(
            FakeOrchestrator(tmp_path / "worker"),
            health={"ready": True, "supported_schema_versions": supported},
            producer_manifest=_producer_manifest(),
            artifact_root=tmp_path / "artifacts",
        )
    )

    assert client.get("/health").json()["supported_schema_versions"] == supported
