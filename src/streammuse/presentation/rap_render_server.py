"""Private HTTP service for bounded remote two-bar rap rendering."""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Callable, Mapping, Protocol

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, Response

from streammuse.application.rap.chunk_orchestration import (
    NoValidCandidates,
    PhraseRenderFailed,
    RemoteChunkRenderArtifact,
    RenderBudgetExpired,
)
from streammuse.domain.rap import REMOTE_CHUNK_SCHEMA_VERSION, RemoteRapChunkRequest
from streammuse.infrastructure.rap.chunk_package import (
    RAP_CHUNK_PACKAGE_MEDIA_TYPE,
    encode_chunk_package,
)


_RESPONSE_FILE = "response.zip"
_REQUEST_FILE = "request.json"
_FAILURE_FILE = "failure.json"
_CANDIDATE_LEDGER_FILE = "candidate_ledger.json"
_ALIGNMENT_FILE = "alignment.json"
_ALIGNED_WAV_FILE = "aligned.wav"
_SOURCE_WAV_FILE = "source.wav"
_LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}
_HEALTH_KEYS = {
    "protocol_version",
    "schema_version",
    "ready",
    "vllm",
    "moss",
    "aligner",
    "rubberband",
    "candidate_profile",
    "warmup",
}
_SENSITIVE_HEALTH_KEYS = {
    "authorization",
    "cache",
    "credential",
    "password",
    "path",
    "reference_wav",
    "secret",
    "token",
    "url",
}


class _ChunkOrchestrator(Protocol):
    def render(self, request: RemoteRapChunkRequest) -> RemoteChunkRenderArtifact:
        """Render exactly one validated two-bar artifact."""


class _IdempotencyConflict(RuntimeError):
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


class _ArtifactStore:
    """Coordinates idempotent responses while keeping render work outside its lock."""

    def __init__(self, root: Path, orchestrator: _ChunkOrchestrator) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._orchestrator = orchestrator
        self._lock = Lock()
        self._in_flight: dict[str, tuple[bytes, Future[_StoredResponse]]] = {}

    def render_or_load(
        self, request: RemoteRapChunkRequest, canonical_request: bytes
    ) -> _StoredResponse:
        request_id = request.request_id
        with self._lock:
            cached = self._load_completed_locked(request_id, canonical_request)
            if cached is not None:
                return cached
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
            self._write_request(request_id, canonical_request)
            artifact = self._orchestrator.render(request)
            response = self._persist_success(request, artifact)
        except Exception as error:
            self._persist_failure(request_id, error)
            future.set_exception(error)
            raise
        else:
            future.set_result(response)
            return response
        finally:
            with self._lock:
                if self._in_flight.get(request_id, (None, None))[1] is future:
                    self._in_flight.pop(request_id, None)

    def _load_completed_locked(
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
            return _StoredResponse(package_path.read_bytes(), "cache;dur=0")
        return None

    def _persist_success(
        self, request: RemoteRapChunkRequest, artifact: RemoteChunkRenderArtifact) -> _StoredResponse:
        if artifact.manifest.request_id != request.request_id:
            raise PhraseRenderFailed("render artifact request identity mismatch")
        workspace = self._workspace(request.request_id)
        self._copy_renderer_artifacts(artifact.workspace, workspace)
        if not (workspace / _SOURCE_WAV_FILE).is_file():
            raise PhraseRenderFailed("render artifact is missing source WAV")

        self._write_json(workspace / _CANDIDATE_LEDGER_FILE, artifact.candidate_ledger)
        self._write_json(
            workspace / _ALIGNMENT_FILE,
            artifact.manifest.diagnostics.alignment_diagnostics,
        )
        self._write_json(workspace / "manifest.json", artifact.manifest.to_payload())
        self._atomic_write(workspace / _ALIGNED_WAV_FILE, artifact.vocal_wav)
        package = encode_chunk_package(artifact.manifest, artifact.vocal_wav)
        # The final atomic replacement is the sole successful-cache marker.
        self._atomic_write(workspace / _RESPONSE_FILE, package)
        return _StoredResponse(package, _server_timing(artifact))

    def _persist_failure(self, request_id: str, error: Exception) -> None:
        workspace = self._workspace(request_id)
        if isinstance(error, NoValidCandidates):
            self._write_json(workspace / _CANDIDATE_LEDGER_FILE, error.candidate_ledger)
        code, _status = _error_spec(error)
        self._write_json(workspace / _FAILURE_FILE, {"code": code})

    def _write_request(self, request_id: str, canonical_request: bytes) -> None:
        self._atomic_write(self._workspace(request_id) / _REQUEST_FILE, canonical_request)

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
                _ALIGNED_WAV_FILE,
            }:
                continue
            self._atomic_copy(path, target)

    @staticmethod
    def _atomic_copy(source: Path, destination: Path) -> None:
        with source.open("rb") as input_file:
            _ArtifactStore._atomic_write(destination, input_file.read())

    @staticmethod
    def _write_json(path: Path, value: object) -> None:
        _ArtifactStore._atomic_write(
            path,
            json.dumps(
                _json_value(value), ensure_ascii=True, separators=(",", ":"), sort_keys=True
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
        finally:
            temporary.unlink(missing_ok=True)


def create_rap_render_app(
    orchestrator: _ChunkOrchestrator,
    health: Mapping[str, object] | Callable[[], Mapping[str, object]],
    *,
    artifact_root: str | Path = "rap-chunk-artifacts",
) -> FastAPI:
    """Create the testable private render service without composing model dependencies."""
    store = _ArtifactStore(Path(artifact_root), orchestrator)
    app = FastAPI(title="StreamMUSE Private Rap Renderer", docs_url=None, redoc_url=None)

    @app.get("/health")
    async def get_health() -> dict[str, object]:
        value = health() if callable(health) else health
        return _public_health(value)

    @app.post("/v1/rap/chunks/render")
    async def render_chunk(http_request: Request) -> Response:
        try:
            request = _parse_request(await http_request.body())
            idempotency_key = http_request.headers.get("Idempotency-Key")
            if idempotency_key != request.request_id:
                raise ValueError("Idempotency-Key must equal request_id")
        except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, TypeError, ValueError):
            return _error_response("invalid_request", 422, "invalid rap chunk request")

        try:
            stored = await run_in_threadpool(
                store.render_or_load, request, request.canonical_json_bytes()
            )
        except _IdempotencyConflict:
            return _error_response(
                "idempotency_conflict", 409, "request ID is already bound to another request"
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
    parser = getattr(RemoteRapChunkRequest, "from_dict", None)
    if callable(parser):
        return parser(payload)
    return RemoteRapChunkRequest.from_payload(payload)


def _json_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    return value


def _error_spec(error: Exception) -> tuple[str, int]:
    if isinstance(error, RenderBudgetExpired):
        return "budget_exhausted", 422
    if isinstance(error, NoValidCandidates):
        return "no_valid_candidates", 422
    if isinstance(error, PhraseRenderFailed):
        return "render_failed", 503
    return "internal_error", 500


def _error_response(code: str, status: int, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message}})


def _server_timing(artifact: RemoteChunkRenderArtifact) -> str:
    total = artifact.manifest.diagnostics.stage_timings_ms["total"]
    return f"total;dur={total:.3f}"


def _public_health(value: Mapping[str, object]) -> dict[str, object]:
    public = {
        "protocol_version": "remote-rap-chunk/v1",
        "schema_version": REMOTE_CHUNK_SCHEMA_VERSION,
        "ready": False,
    }
    for key in _HEALTH_KEYS:
        if key in value:
            public[key] = _sanitize_health_value(value[key])
    return public


def _sanitize_health_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            str(key): _sanitize_health_value(item)
            for key, item in value.items()
            if str(key).lower() not in _SENSITIVE_HEALTH_KEYS
        }
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_sanitize_health_value(item) for item in value]
    return str(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve private StreamMUSE rap chunks")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8020)
    parser.add_argument("--allow-public-bind", action="store_true")
    parser.add_argument("--artifact-root", default="rap-chunk-artifacts")
    parser.add_argument("--vllm-url", default="http://127.0.0.1:8000")
    parser.add_argument("--vllm-model", required=True)
    parser.add_argument("--moss-model", required=True)
    parser.add_argument("--moss-device", default="cuda")
    parser.add_argument("--moss-reference-wav", required=True)
    parser.add_argument("--aligner-device", default="cuda")
    parser.add_argument("--aligner-cache")
    parser.add_argument("--candidate-profile", default="realtime")
    return parser


def main(
    argv: list[str] | None = None,
    *,
    composition_factory: Callable[[RapRenderServerConfig], tuple[_ChunkOrchestrator, Mapping[str, object]]] | None = None,
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
    compose = composition_factory or _compose_real_worker
    orchestrator, health = compose(config)
    run_server = serve
    if run_server is None:
        import uvicorn

        run_server = uvicorn.run
    run_server(
        create_rap_render_app(orchestrator, health, artifact_root=config.artifact_root),
        host=config.host,
        port=config.port,
        log_level="info",
    )
    return 0


def _compose_real_worker(
    config: RapRenderServerConfig,
) -> tuple[_ChunkOrchestrator, Mapping[str, object]]:
    """Late-bound seam for Task 3's resident MOSS/MMS composition.

    Keeping this import boundary local lets the server module and CLI remain
    importable on the Mac, where MOSS and MMS are intentionally unavailable.
    """
    del config
    raise RuntimeError(
        "real H200 composition requires the Task 3 MOSS/MMS renderer factory"
    )


if __name__ == "__main__":
    raise SystemExit(main())
