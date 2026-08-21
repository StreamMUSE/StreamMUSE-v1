"""Private HTTP service for bounded remote two-bar rap rendering."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import time
import uuid
from concurrent.futures import Future
from contextlib import ExitStack
from dataclasses import dataclass, replace
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Callable, Mapping, Protocol

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, Response

from streammuse.domain.rap import (
    REMOTE_CHUNK_SCHEMA_VERSION,
    RemoteRapChunkManifest,
    RemoteRapChunkRequest,
)

if TYPE_CHECKING:
    from streammuse.application.rap.chunk_orchestration import (
        RemoteChunkRenderArtifact,
    )


_RESPONSE_FILE = "response.zip"
_REQUEST_FILE = "request.json"
_FAILURE_FILE = "failure.json"
_CANDIDATE_LEDGER_FILE = "candidate_ledger.json"
_ALIGNMENT_FILE = "alignment.json"
_MMS_ALIGNMENT_FILE = "mms_alignment.json"
_ALIGNED_WAV_FILE = "aligned.wav"
_SOURCE_WAV_FILE = "source.wav"
_VOCAL_WAV_FILE = "vocal.wav"
_MANIFEST_FILE = "manifest.json"
_SERVER_TIMING_FILE = "server_timing.json"
_MEASUREMENT_MANIFEST_FILE = ".manifest.measurement.json"
_MEASUREMENT_TIMING_FILE = ".server_timing.measurement.json"
_MEASUREMENT_PACKAGE_FILE = ".response.measurement.zip"
MAX_RAP_CHUNK_REQUEST_BYTES = 64 * 1024
_MAX_HEALTH_STRING_LENGTH = 128
_LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}
_HEALTH_COMPONENT_KEYS = {"vllm", "moss", "aligner", "rubberband"}
_HEALTH_SUMMARY_KEYS = {
    "ready",
    "status",
    "identity",
    "version",
    "model",
    "profile",
}
_INVALID_HEALTH_VALUE = object()
_CANDIDATE_PROFILES = {
    "realtime": {"max_tokens_per_choice": 32, "temperature": 1.0},
}


class _ChunkOrchestrator(Protocol):
    def render(self, request: RemoteRapChunkRequest) -> RemoteChunkRenderArtifact:
        """Render exactly one validated two-bar artifact."""


class _IdempotencyConflict(RuntimeError):
    pass


class _RequestTooLarge(RuntimeError):
    pass


@dataclass(frozen=True)
class _StoredResponse:
    package: bytes
    server_timing: str


@dataclass(frozen=True)
class RapRenderServerConfig:
    host: str
    port: int
    artifact_root: Path
    vllm_url: str
    vllm_model: str
    moss_model: str
    moss_device: str
    moss_reference_wav: Path
    aligner_device: str
    aligner_cache: Path | None
    candidate_profile: str


@dataclass
class _WorkerComposition:
    orchestrator: _ChunkOrchestrator
    health: Mapping[str, object]
    _resources: ExitStack

    def close(self) -> None:
        self._resources.close()


class _ArtifactStore:
    """Coordinates idempotent responses while keeping render work outside its lock."""

    def __init__(
        self,
        root: Path,
        orchestrator: _ChunkOrchestrator,
        *,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._orchestrator = orchestrator
        self._clock = clock
        self._lock = Lock()
        self._in_flight: dict[str, tuple[bytes, Future[_StoredResponse]]] = {}

    def render_or_load(
        self, request: RemoteRapChunkRequest, canonical_request: bytes
    ) -> _StoredResponse:
        request_id = request.request_id
        with self._lock:
            existing = self._in_flight.get(request_id)
            if existing is not None:
                existing_request, future = existing
                if existing_request != canonical_request:
                    raise _IdempotencyConflict
                owner = False
            else:
                future = Future()
                self._in_flight[request_id] = (canonical_request, future)
                owner = True

        if not owner:
            return future.result()

        try:
            response = self._load_completed(request_id, canonical_request)
            if response is None:
                self._write_request(request_id, canonical_request)
                artifact = self._orchestrator.render(request)
                response = self._persist_success(request, artifact)
        except BaseException as error:
            try:
                if not isinstance(error, _IdempotencyConflict):
                    self._persist_failure(request_id, error)
            except BaseException:
                pass
            finally:
                future.set_exception(error)
            raise
        else:
            future.set_result(response)
            return response
        finally:
            with self._lock:
                if self._in_flight.get(request_id, (None, None))[1] is future:
                    self._in_flight.pop(request_id, None)

    def _load_completed(
        self, request_id: str, canonical_request: bytes
    ) -> _StoredResponse | None:
        workspace = self._workspace(request_id)
        request_path = workspace / _REQUEST_FILE
        if request_path.exists():
            try:
                recorded_request = request_path.read_bytes()
            except OSError:
                recorded_request = b""
            if recorded_request != canonical_request:
                raise _IdempotencyConflict
        package_path = workspace / _RESPONSE_FILE
        if request_path.exists() and package_path.is_file():
            server_timing = "cache;dur=0"
            timing_path = workspace / _SERVER_TIMING_FILE
            if timing_path.is_file():
                try:
                    timing_payload = json.loads(timing_path.read_bytes())
                    recorded_timing = timing_payload.get("server_timing")
                    if isinstance(recorded_timing, str):
                        server_timing = recorded_timing
                except (AttributeError, json.JSONDecodeError, OSError):
                    pass
            return _StoredResponse(package_path.read_bytes(), server_timing)
        return None

    def _persist_success(
        self, request: RemoteRapChunkRequest, artifact: RemoteChunkRenderArtifact
    ) -> _StoredResponse:
        from streammuse.application.rap.chunk_orchestration import PhraseRenderFailed
        from streammuse.infrastructure.rap.chunk_package import encode_chunk_package

        if artifact.manifest.request_id != request.request_id:
            raise PhraseRenderFailed("render artifact request identity mismatch")
        packaging_started = self._clock()
        workspace = self._workspace(request.request_id)
        self._copy_renderer_artifacts(artifact.workspace, workspace)
        self._preserve_renderer_artifact(
            artifact.workspace,
            workspace,
            _SOURCE_WAV_FILE,
            "render artifact is missing source WAV",
        )
        self._preserve_renderer_artifact(
            artifact.workspace,
            workspace,
            _MMS_ALIGNMENT_FILE,
            "render artifact is missing MMS alignment JSON",
        )
        self._preserve_renderer_artifact(
            artifact.workspace,
            workspace,
            _VOCAL_WAV_FILE,
            "render artifact is missing vocal WAV",
        )

        self._write_json(workspace / _CANDIDATE_LEDGER_FILE, artifact.candidate_ledger)
        self._write_json(
            workspace / _ALIGNMENT_FILE,
            artifact.manifest.diagnostics.alignment_diagnostics,
        )
        self._atomic_write(workspace / _ALIGNED_WAV_FILE, artifact.vocal_wav)

        # The manifest must contain packaging time, so measure a first pass that
        # mirrors the final manifest/package/timing/response publication sequence.
        measurement_manifest = _finalize_manifest_timing(artifact.manifest, 0.001)
        measurement_paths = (
            workspace / _MEASUREMENT_MANIFEST_FILE,
            workspace / _MEASUREMENT_TIMING_FILE,
            workspace / _MEASUREMENT_PACKAGE_FILE,
        )
        measurement_started = self._clock()
        try:
            self._write_json(measurement_paths[0], measurement_manifest.to_payload())
            measurement_package = encode_chunk_package(
                measurement_manifest, artifact.vocal_wav
            )
            self._write_json(
                measurement_paths[1],
                {"server_timing": _server_timing(measurement_manifest)},
            )
            self._atomic_write(measurement_paths[2], measurement_package)
        finally:
            measurement_finished = self._clock()
            for measurement_path in measurement_paths:
                self._durably_unpublish(measurement_path)
            cleanup_finished = self._clock()

        measured_prefix_ms = (measurement_started - packaging_started) * 1000.0
        measured_publication_ms = (measurement_finished - measurement_started) * 1000.0
        measured_cleanup_ms = (cleanup_finished - measurement_finished) * 1000.0
        packaging_ms = max(
            0.001,
            measured_prefix_ms
            + measured_publication_ms
            + measured_cleanup_ms
            + measured_publication_ms,
        )

        final_manifest = _finalize_manifest_timing(artifact.manifest, packaging_ms)
        self._write_json(workspace / _MANIFEST_FILE, final_manifest.to_payload())
        package = encode_chunk_package(final_manifest, artifact.vocal_wav)
        server_timing = _server_timing(final_manifest)
        self._write_json(
            workspace / _SERVER_TIMING_FILE, {"server_timing": server_timing}
        )
        # The final atomic replacement is the sole successful-cache marker.
        self._publish_response(workspace / _RESPONSE_FILE, package)
        return _StoredResponse(package, server_timing)

    def _persist_failure(self, request_id: str, error: BaseException) -> None:
        from streammuse.application.rap.chunk_orchestration import NoValidCandidates

        workspace = self._workspace(request_id)
        if isinstance(error, NoValidCandidates):
            self._write_json(workspace / _CANDIDATE_LEDGER_FILE, error.candidate_ledger)
        code, _status = _error_spec(error)
        self._write_json(workspace / _FAILURE_FILE, {"code": code})

    def _write_request(self, request_id: str, canonical_request: bytes) -> None:
        self._atomic_write(
            self._workspace(request_id) / _REQUEST_FILE, canonical_request
        )

    def _workspace(self, request_id: str) -> Path:
        path = self._root / request_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _copy_renderer_artifacts(self, source: Path, destination: Path) -> None:
        source = Path(source)
        if not source.exists() or source.resolve() == destination.resolve():
            return
        for path in source.rglob("*"):
            if not path.is_file() or path.is_symlink():
                continue
            relative = path.relative_to(source)
            if relative.parts and relative.parts[0] == "..":
                continue
            target = destination / relative
            if target.name in {
                _RESPONSE_FILE,
                _REQUEST_FILE,
                _FAILURE_FILE,
                _CANDIDATE_LEDGER_FILE,
                _ALIGNMENT_FILE,
                _MMS_ALIGNMENT_FILE,
                _ALIGNED_WAV_FILE,
                _SOURCE_WAV_FILE,
                _VOCAL_WAV_FILE,
                _MANIFEST_FILE,
                _SERVER_TIMING_FILE,
                _MEASUREMENT_MANIFEST_FILE,
                _MEASUREMENT_TIMING_FILE,
                _MEASUREMENT_PACKAGE_FILE,
            }:
                continue
            self._atomic_copy(path, target)

    def _preserve_renderer_artifact(
        self,
        source: Path,
        destination: Path,
        name: str,
        missing_message: str,
    ) -> None:
        from streammuse.application.rap.chunk_orchestration import PhraseRenderFailed

        source_path = Path(source) / name
        if not source_path.is_file():
            raise PhraseRenderFailed(missing_message)
        target = destination / name
        if source_path.resolve() == target.resolve():
            self._fsync_existing_file(target)
        else:
            self._atomic_copy(source_path, target)

    @staticmethod
    def _atomic_copy(source: Path, destination: Path) -> None:
        with source.open("rb") as input_file:
            _ArtifactStore._atomic_write(destination, input_file.read())

    @staticmethod
    def _write_json(path: Path, value: object) -> None:
        _ArtifactStore._atomic_write(
            path,
            json.dumps(
                _json_value(value),
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8"),
        )

    @staticmethod
    def _atomic_write(path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("xb") as output:
                output.write(data)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, path)
            _ArtifactStore._fsync_directory(path.parent)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _publish_response(path: Path, data: bytes) -> None:
        try:
            _ArtifactStore._atomic_write(path, data)
        except BaseException:
            try:
                _ArtifactStore._durably_unpublish(path)
            except BaseException:
                pass
            raise

    @staticmethod
    def _durably_unpublish(path: Path) -> None:
        path.unlink(missing_ok=True)
        _ArtifactStore._fsync_directory(path.parent)

    @staticmethod
    def _fsync_existing_file(path: Path) -> None:
        file_fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(file_fd)
        finally:
            os.close(file_fd)
        _ArtifactStore._fsync_directory(path.parent)

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        directory_fd = os.open(path, flags)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)


def create_rap_render_app(
    orchestrator: _ChunkOrchestrator,
    health: Mapping[str, object] | Callable[[], Mapping[str, object]],
    *,
    artifact_root: str | Path = "rap-chunk-artifacts",
) -> FastAPI:
    """Create the testable private render service without composing model dependencies."""
    from streammuse.infrastructure.rap.chunk_package import (
        RAP_CHUNK_PACKAGE_MEDIA_TYPE,
    )

    store = _ArtifactStore(Path(artifact_root), orchestrator)
    app = FastAPI(
        title="StreamMUSE Private Rap Renderer", docs_url=None, redoc_url=None
    )

    @app.get("/health")
    async def get_health() -> dict[str, object]:
        value = health() if callable(health) else health
        return _public_health(value)

    @app.post("/v1/rap/chunks/render")
    async def render_chunk(http_request: Request) -> Response:
        try:
            request_body = await _read_bounded_request_body(http_request)
        except _RequestTooLarge:
            return _error_response(
                "request_too_large", 413, "rap chunk request exceeds size limit"
            )
        try:
            request = _parse_request(request_body)
            idempotency_key = http_request.headers.get("Idempotency-Key")
            if idempotency_key != request.request_id:
                raise ValueError("Idempotency-Key must equal request_id")
        except (
            UnicodeDecodeError,
            json.JSONDecodeError,
            RecursionError,
            TypeError,
            ValueError,
        ):
            return _error_response("invalid_request", 422, "invalid rap chunk request")

        try:
            stored = await run_in_threadpool(
                store.render_or_load, request, request.canonical_json_bytes()
            )
        except _IdempotencyConflict:
            return _error_response(
                "idempotency_conflict",
                409,
                "request ID is already bound to another request",
            )
        except Exception as error:
            code, status = _error_spec(error)
            message = (
                "rap chunk render could not be completed"
                if code != "internal_error"
                else "rap chunk render failed"
            )
            return _error_response(code, status, message)

        return Response(
            content=stored.package,
            media_type=RAP_CHUNK_PACKAGE_MEDIA_TYPE,
            headers={
                "X-StreamMUSE-Request-ID": request.request_id,
                "Content-Length": str(len(stored.package)),
                "Server-Timing": stored.server_timing,
            },
        )

    return app


def _parse_request(body: bytes) -> RemoteRapChunkRequest:
    payload = json.loads(body.decode("utf-8"))
    return RemoteRapChunkRequest.from_payload(payload)


async def _read_bounded_request_body(request: Request) -> bytes:
    content_length = request.headers.get("Content-Length")
    if content_length is not None:
        try:
            if int(content_length) > MAX_RAP_CHUNK_REQUEST_BYTES:
                raise _RequestTooLarge
        except ValueError:
            pass

    body = bytearray()
    async for chunk in request.stream():
        if len(body) + len(chunk) > MAX_RAP_CHUNK_REQUEST_BYTES:
            raise _RequestTooLarge
        body.extend(chunk)
    return bytes(body)


def _json_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    return value


def _error_spec(error: Exception) -> tuple[str, int]:
    from streammuse.application.rap.chunk_orchestration import (
        NoValidCandidates,
        PhraseRenderFailed,
        RenderBudgetExpired,
    )

    if isinstance(error, RenderBudgetExpired):
        return "budget_exhausted", 422
    if isinstance(error, NoValidCandidates):
        return "no_valid_candidates", 422
    if isinstance(error, PhraseRenderFailed):
        return "render_failed", 503
    return "internal_error", 500


def _error_response(code: str, status: int, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status, content={"error": {"code": code, "message": message}}
    )


def _finalize_manifest_timing(
    manifest: RemoteRapChunkManifest, packaging_ms: float
) -> RemoteRapChunkManifest:
    timings = dict(manifest.diagnostics.stage_timings_ms)
    previous_total = float(timings["total"])
    timings["packaging"] = packaging_ms
    timings["total"] = max(
        previous_total + packaging_ms,
        sum(value for name, value in timings.items() if name != "total"),
    )
    warnings = tuple(
        warning
        for warning in manifest.diagnostics.warnings
        if warning != "packaging timing is provisional"
    )
    diagnostics = replace(
        manifest.diagnostics,
        stage_timings_ms=timings,
        warnings=warnings,
    )
    return replace(manifest, diagnostics=diagnostics)


def _server_timing(manifest: RemoteRapChunkManifest) -> str:
    total = manifest.diagnostics.stage_timings_ms["total"]
    return f"total;dur={total:.3f}"


def _public_health(value: Mapping[str, object]) -> dict[str, object]:
    public = {
        "protocol_version": "remote-rap-chunk/v1",
        "schema_version": REMOTE_CHUNK_SCHEMA_VERSION,
        "ready": False,
    }
    for key in ("protocol_version", "schema_version", "ready", "candidate_profile"):
        if key not in value:
            continue
        scalar_key = "profile" if key == "candidate_profile" else key
        item = _bounded_health_scalar(scalar_key, value[key])
        if item is not _INVALID_HEALTH_VALUE:
            public[key] = item
    for key in _HEALTH_COMPONENT_KEYS:
        item = value.get(key)
        if isinstance(item, Mapping):
            public[key] = _public_health_summary(item)
    if "warmup" in value:
        warmup = value["warmup"]
        if isinstance(warmup, Mapping):
            public["warmup"] = _public_health_summary(warmup)
        else:
            bounded = _bounded_health_scalar("warmup", warmup)
            if bounded is not _INVALID_HEALTH_VALUE:
                public["warmup"] = bounded
    return public


def _public_health_summary(value: Mapping[str, object]) -> dict[str, object]:
    summary: dict[str, object] = {}
    for key in _HEALTH_SUMMARY_KEYS:
        if key not in value:
            continue
        item = _bounded_health_scalar(key, value[key])
        if item is not _INVALID_HEALTH_VALUE:
            summary[key] = item
    if "warmup" in value:
        warmup = value["warmup"]
        if isinstance(warmup, Mapping):
            summary["warmup"] = _public_health_summary(warmup)
        else:
            bounded = _bounded_health_scalar("warmup", warmup)
            if bounded is not _INVALID_HEALTH_VALUE:
                summary["warmup"] = bounded
    return summary


def _bounded_health_scalar(key: str, value: object) -> object:
    if key == "ready":
        return value if type(value) is bool else _INVALID_HEALTH_VALUE
    if key != "warmup" and not isinstance(value, str):
        return _INVALID_HEALTH_VALUE
    if isinstance(value, str):
        text = value.strip()
        if not text or any(ord(character) < 32 for character in text):
            return _INVALID_HEALTH_VALUE
        lowered = text.casefold()
        if "://" in text or lowered.startswith(
            ("bearer ", "api_key=", "access_token=", "token=")
        ):
            return _INVALID_HEALTH_VALUE
        if _looks_like_absolute_path(text):
            if key != "model":
                return _INVALID_HEALTH_VALUE
            text = text.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
            if not text:
                return _INVALID_HEALTH_VALUE
        return text[:_MAX_HEALTH_STRING_LENGTH]
    if type(value) is int:
        return value if abs(value) <= 1_000_000_000 else _INVALID_HEALTH_VALUE
    if type(value) is float:
        if math.isfinite(value) and abs(value) <= 1_000_000_000.0:
            return value
    if type(value) is bool:
        return value
    return _INVALID_HEALTH_VALUE


def _looks_like_absolute_path(value: str) -> bool:
    return value.startswith(("/", "\\", "~/", "~\\")) or (
        len(value) >= 3 and value[1] == ":" and value[2] in "/\\"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve private StreamMUSE rap chunks")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8020)
    parser.add_argument("--allow-public-bind", action="store_true")
    parser.add_argument("--artifact-root", default="rap-chunk-artifacts")
    parser.add_argument("--vllm-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--vllm-model", required=True)
    parser.add_argument("--moss-model", required=True)
    parser.add_argument("--moss-device", default="cuda")
    parser.add_argument("--moss-reference-wav", required=True)
    parser.add_argument("--aligner-device", default="cuda")
    parser.add_argument("--aligner-cache")
    parser.add_argument(
        "--candidate-profile", choices=tuple(_CANDIDATE_PROFILES), default="realtime"
    )
    return parser


def main(
    argv: list[str] | None = None,
    *,
    composition_factory: Callable[[RapRenderServerConfig], object] | None = None,
    serve: Callable[..., object] | None = None,
) -> int:
    args = build_parser().parse_args(argv)
    if args.host not in _LOOPBACK_HOSTS and not args.allow_public_bind:
        print("refusing non-loopback bind without --allow-public-bind", file=sys.stderr)
        return 2
    config = RapRenderServerConfig(
        host=args.host,
        port=args.port,
        artifact_root=Path(args.artifact_root),
        vllm_url=args.vllm_url,
        vllm_model=args.vllm_model,
        moss_model=args.moss_model,
        moss_device=args.moss_device,
        moss_reference_wav=Path(args.moss_reference_wav),
        aligner_device=args.aligner_device,
        aligner_cache=Path(args.aligner_cache) if args.aligner_cache else None,
        candidate_profile=args.candidate_profile,
    )
    run_server = serve
    if run_server is None:
        import uvicorn

        run_server = uvicorn.run
    compose = composition_factory or _compose_real_worker
    composition = compose(config)
    try:
        run_server(
            create_rap_render_app(
                composition.orchestrator,
                composition.health,
                artifact_root=config.artifact_root,
            ),
            host=config.host,
            port=config.port,
            log_level="info",
        )
    finally:
        composition.close()
    return 0


def _compose_real_worker(
    config: RapRenderServerConfig,
) -> _WorkerComposition:
    """Load, warm, and own one resident H200 render composition."""
    dependencies = _load_worker_dependencies()
    resources = ExitStack()
    try:
        config.artifact_root.mkdir(parents=True, exist_ok=True)
        client_config = dependencies.LocalChatModelClientConfig(
            base_url=config.vllm_url,
            model=config.vllm_model,
            timeout_s=30.0,
        )
        client = dependencies.LocalChatModelClient(client_config)
        _register_close(resources, client)
        vllm_health = _probe_vllm(config.vllm_url, config.vllm_model)

        profile = _CANDIDATE_PROFILES[config.candidate_profile]
        generator = dependencies.IndependentChoiceCandidateGenerator(
            client,
            max_tokens_per_choice=profile["max_tokens_per_choice"],
            temperature=profile["temperature"],
        )
        analyzer = dependencies.CmuProsodyAnalyzer()
        planner = dependencies.ChunkCandidatePlanner(
            generator,
            analyzer,
            dependencies.ScoreWeights(),
        )

        synthesizer = dependencies.PersistentMossSynthesizer.load(
            model_id=config.moss_model,
            device=config.moss_device,
            reference_wav=config.moss_reference_wav,
        )
        _register_close(resources, synthesizer)
        if config.aligner_cache is not None:
            _configure_aligner_cache(config.aligner_cache)
        aligner = dependencies.MmsForcedAligner.load(device=config.aligner_device)
        _register_close(resources, aligner)
        rubberband_health = _probe_rubberband()

        warmup_request = _warmup_render_request()
        with tempfile.TemporaryDirectory(
            prefix=".streammuse-rap-warmup-", dir=config.artifact_root
        ) as temporary_directory:
            warmup_wav = Path(temporary_directory) / "moss-warmup.wav"
            moss_warmup = synthesizer.synthesize(warmup_request, warmup_wav)
            aligner_warmup = aligner.warmup(warmup_wav, warmup_request.text)

        renderer = dependencies.MossAlignedPhraseRenderer(
            synthesizer=synthesizer,
            aligner=aligner,
            rubberband_version=str(rubberband_health["version"]),
        )
        _register_close(resources, renderer)
        orchestrator = dependencies.RapChunkOrchestrator(
            planner,
            renderer,
            workspace_root=config.artifact_root,
        )
        moss_version = str(getattr(moss_warmup, "model_revision", "unknown"))
        health = {
            "protocol_version": "remote-rap-chunk/v1",
            "schema_version": REMOTE_CHUNK_SCHEMA_VERSION,
            "ready": True,
            "vllm": dict(vllm_health),
            "moss": {
                "ready": True,
                "status": "warmed",
                "identity": "PersistentMossSynthesizer",
                "version": moss_version,
                "model": _public_model_identity(config.moss_model),
                "warmup": "complete",
            },
            "aligner": {
                "ready": True,
                "status": "warmed",
                "identity": str(aligner_warmup["aligner"]),
                "version": str(aligner_warmup["version"]),
                "warmup": "complete",
            },
            "rubberband": dict(rubberband_health),
            "candidate_profile": config.candidate_profile,
            "warmup": {"ready": True, "status": "complete"},
        }
        return _WorkerComposition(orchestrator, health, resources)
    except BaseException:
        resources.close()
        raise


def _load_worker_dependencies() -> object:
    """Import H200-only dependencies only while composing the real worker."""
    from types import SimpleNamespace

    from streammuse.application.rap.chunk_orchestration import (
        ChunkCandidatePlanner,
        RapChunkOrchestrator,
    )
    from streammuse.domain.rap import ScoreWeights
    from streammuse.infrastructure.inference.local_chat_client import (
        LocalChatModelClient,
        LocalChatModelClientConfig,
    )
    from streammuse.infrastructure.rap.generators import (
        IndependentChoiceCandidateGenerator,
    )
    from streammuse.infrastructure.rap.mms_forced_alignment import MmsForcedAligner
    from streammuse.infrastructure.rap.moss_aligned_phrase import (
        MossAlignedPhraseRenderer,
    )
    from streammuse.infrastructure.rap.moss_tts import PersistentMossSynthesizer
    from streammuse.infrastructure.rap.prosody import CmuProsodyAnalyzer

    return SimpleNamespace(
        LocalChatModelClient=LocalChatModelClient,
        LocalChatModelClientConfig=LocalChatModelClientConfig,
        IndependentChoiceCandidateGenerator=IndependentChoiceCandidateGenerator,
        CmuProsodyAnalyzer=CmuProsodyAnalyzer,
        ScoreWeights=ScoreWeights,
        ChunkCandidatePlanner=ChunkCandidatePlanner,
        PersistentMossSynthesizer=PersistentMossSynthesizer,
        MmsForcedAligner=MmsForcedAligner,
        MossAlignedPhraseRenderer=MossAlignedPhraseRenderer,
        RapChunkOrchestrator=RapChunkOrchestrator,
    )


def _public_model_identity(model: str) -> str:
    if _looks_like_absolute_path(model):
        model = model.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
    return model[:_MAX_HEALTH_STRING_LENGTH] or "unknown"


def _register_close(resources: ExitStack, owner: object) -> None:
    close = getattr(owner, "close", None)
    if callable(close):
        resources.callback(close)


def _configure_aligner_cache(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    import torch

    torch.hub.set_dir(str(path))


def _probe_vllm(base_url: str, model: str) -> Mapping[str, object]:
    import httpx

    response = httpx.get(f"{base_url.rstrip('/')}/models", timeout=5.0)
    response.raise_for_status()
    payload = response.json()
    models = payload.get("data") if isinstance(payload, Mapping) else None
    model_ids = {
        item.get("id")
        for item in models or ()
        if isinstance(item, Mapping) and isinstance(item.get("id"), str)
    }
    if model not in model_ids:
        raise RuntimeError("configured vLLM model is not ready")
    return {
        "ready": True,
        "status": "serving",
        "identity": "vLLM",
        "version": response.headers.get("server", "unknown"),
        "model": model,
    }


def _probe_rubberband() -> Mapping[str, object]:
    completed = subprocess.run(
        ["rubberband", "--version"],
        check=True,
        capture_output=True,
        text=True,
        timeout=5.0,
    )
    version = (completed.stdout or completed.stderr).strip().splitlines()
    return {
        "ready": True,
        "status": "available",
        "identity": "Rubber Band",
        "version": (version[0] if version else "unknown")[:128],
    }


def _warmup_render_request() -> object:
    from streammuse.experiments.rap_audio_protocols.contracts import (
        SyllableTarget,
        TwoBarRenderRequest,
    )

    return TwoBarRenderRequest(
        song_id="streammuse-h200-warmup",
        chunk_index=0,
        start_bar=0,
        end_bar=2,
        text="warm voice",
        syllables=(
            SyllableTarget(
                word="warm",
                index_in_word=0,
                phonemes=("W", "AO1", "R", "M"),
                lexical_stress=1,
                target_stress=1.0,
                boundary_strength=0,
                absolute_tick=1,
                tick_in_chunk=1,
                target_seconds=0.25,
            ),
            SyllableTarget(
                word="voice",
                index_in_word=0,
                phonemes=("V", "OY1", "S"),
                lexical_stress=1,
                target_stress=1.0,
                boundary_strength=2,
                absolute_tick=4,
                tick_in_chunk=4,
                target_seconds=0.75,
            ),
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main())
