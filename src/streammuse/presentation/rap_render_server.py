"""Private HTTP service for bounded remote two-bar rap rendering."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FutureTimeout
from contextlib import ExitStack
from dataclasses import dataclass, field, replace
from pathlib import Path
from threading import Event, Lock, Thread
from typing import TYPE_CHECKING, Callable, Mapping, Protocol
from urllib.parse import urlsplit

import numpy as np
from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, Response

from streammuse.domain.rap import (
    REMOTE_CHUNK_SCHEMA_VERSION,
    REMOTE_CHUNK_SCHEMA_VERSION_V2,
    REMOTE_CHUNK_SCHEMA_VERSIONS,
    RemoteRapChunkManifest,
    RemoteRapChunkRequest,
)
from streammuse.application.rap.execution import (
    ExecutionCancelled,
    ExecutionDeadlineExceeded,
    SynthesisExecutionContext,
)
from streammuse.infrastructure.rap.producer_manifest import (
    ProducerManifestError,
    ProducerManifestV1,
    initialize_producer_namespace,
    verify_producer_namespace,
)

if TYPE_CHECKING:
    from streammuse.application.rap.chunk_orchestration import (
        RemoteChunkRenderArtifact,
    )


_RESPONSE_FILE = "response.zip"
_OPUS_RESPONSE_FILE = "response.opus.zip"
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
_COMPLETE_FILE = "complete.v1.json"
_MOSS_SIDECAR_FILE = "internal/moss_synthesis.v1.json"
_MEASUREMENT_MANIFEST_FILE = ".manifest.measurement.json"
_MEASUREMENT_TIMING_FILE = ".server_timing.measurement.json"
_MEASUREMENT_PACKAGE_FILE = ".response.measurement.zip"
MAX_RAP_CHUNK_REQUEST_BYTES = 64 * 1024
_MAX_HEALTH_STRING_LENGTH = 128
_LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}
_HEALTH_COMPONENT_KEYS = {"vllm", "moss", "aligner", "rubberband"}
_HEALTH_SUMMARY_KEYS = {
    "backend",
    "endpoint_host",
    "endpoint_port",
    "ready",
    "reason_code",
    "reference_audio_sha256",
    "reference_text_sha256",
    "restart_required",
    "status",
    "state",
    "identity",
    "version",
    "model",
    "model_revision",
    "producer_fingerprint",
    "profile",
    "server_version",
    "warmup_time_ms",
}
_HEALTH_TOP_KEYS = {
    "backend",
    "candidate_profile",
    "producer_fingerprint",
    "protocol_version",
    "ready",
    "reason_code",
    "restart_required",
    "schema_version",
    "state",
    "supported_schema_versions",
}
# Health keeps reporting the v1 envelope for older clients; chunk contracts the
# server accepts are advertised separately (comma-separated).
_SUPPORTED_SCHEMA_VERSIONS_TEXT = ",".join(REMOTE_CHUNK_SCHEMA_VERSIONS)
_INVALID_HEALTH_VALUE = object()
_CANDIDATE_PROFILES = {
    "realtime": {"max_tokens_per_choice": 32, "temperature": 1.0},
}
_MOSS_WARP_POLICIES = ("gentle_sparse_r3", "all_onsets_r3")
_MOSS_SERVING_BACKENDS = ("inprocess", "sglang-omni", "mlx")
_PRIVATE_SIDECAR_MAX_BYTES = 64 * 1024


class _ChunkOrchestrator(Protocol):
    def render(
        self,
        request: RemoteRapChunkRequest,
        *,
        execution: SynthesisExecutionContext,
    ) -> RemoteChunkRenderArtifact:
        """Render exactly one validated two-bar artifact."""


class _IdempotencyConflict(RuntimeError):
    pass


class _RequestTooLarge(RuntimeError):
    pass


class _CacheIntegrityError(RuntimeError):
    pass


class _ServiceDegraded(RuntimeError):
    pass


@dataclass(frozen=True)
class _StoredResponse:
    package: bytes
    server_timing: str


@dataclass
class _InFlightRender:
    key: tuple[str, str]
    canonical_request: bytes
    future: Future[_StoredResponse]
    owner_execution: SynthesisExecutionContext
    queued_at: float
    completed: Event = field(default_factory=Event)
    waiters: set[str] = field(default_factory=set)
    cancellation_monitor_started: bool = False


@dataclass
class _RenderWaiter:
    store: "_ArtifactStore"
    state: _InFlightRender
    waiter_id: str
    execution: SynthesisExecutionContext
    detached: bool = False

    @property
    def future(self) -> Future[_StoredResponse]:
        return self.state.future

    def detach(self, reason: str = "waiter_complete") -> None:
        if self.detached:
            return
        self.detached = True
        self.store._detach_waiter(self, reason=reason)


class _RuntimeState:
    """Thread-safe public readiness state shared by routes and worker monitors."""

    def __init__(
        self,
        health: Mapping[str, object] | Callable[[], Mapping[str, object]],
    ) -> None:
        self._health = health
        self._lock = Lock()
        self._state = "ready"
        self._reason_code: str | None = None

    def require_ready(self) -> None:
        with self._lock:
            if self._state != "ready":
                raise _ServiceDegraded("render service requires an explicit restart")

    def degrade(self, reason_code: str) -> None:
        bounded = _bounded_reason_code(reason_code)
        with self._lock:
            self._state = "degraded"
            self._reason_code = bounded

    def snapshot(self) -> dict[str, object]:
        source = self._health() if callable(self._health) else self._health
        value = dict(source)
        with self._lock:
            state = self._state
            reason_code = self._reason_code
        value["state"] = state
        configured_ready = value.get("ready", False)
        value["ready"] = (
            configured_ready if type(configured_ready) is bool and state == "ready" else False
        )
        if state == "degraded":
            value["restart_required"] = True
            value["reason_code"] = reason_code or "cancellation_grace_exceeded"
            moss = value.get("moss")
            moss_health = dict(moss) if isinstance(moss, Mapping) else {}
            moss_health.update(
                {
                    "ready": False,
                    "state": "degraded",
                    "restart_required": True,
                    "reason_code": value["reason_code"],
                }
            )
            value["moss"] = moss_health
        return value


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
    moss_warp_policy: str = "gentle_sparse_r3"
    wire_audio_codec: str = "pcm"
    opus_compression_level: int = 5
    moss_serving_backend: str = "inprocess"
    moss_sglang_url: str | None = None
    moss_reference_text_file: Path | None = None
    moss_sglang_reference_uri: str | None = None
    moss_sglang_reference_sha256: str | None = None
    moss_request_timeout_s: float = 120.0
    moss_cancellation_grace_s: float = 2.0
    moss_model_revision: str | None = None
    moss_runtime_version: str | None = None
    moss_runtime_revision: str | None = None
    moss_sglang_version: str | None = None
    moss_sglang_revision: str | None = None
    moss_runtime_environment_sha256: str | None = None
    moss_runtime_patch_sha256: str | None = None
    moss_runtime_config: Path | None = None
    moss_runtime_config_sha256: str | None = None
    moss_mlx_url: str | None = None
    moss_mlx_runtime: Mapping[str, str] | None = None
    concurrent_bar_generation: bool = False


@dataclass
class _WorkerComposition:
    orchestrator: _ChunkOrchestrator
    health: Mapping[str, object]
    producer_manifest: ProducerManifestV1
    _resources: ExitStack

    def close(self) -> None:
        self._resources.close()


class _ArtifactStore:
    """Coordinates idempotent responses while keeping render work outside its lock."""

    def __init__(
        self,
        root: Path,
        orchestrator: _ChunkOrchestrator,
        producer_manifest: ProducerManifestV1,
        *,
        clock: Callable[[], float] = time.perf_counter,
        execution_clock: Callable[[], float] = time.monotonic,
        cancellation_grace_seconds: float = 2.0,
        require_ready: Callable[[], None] | None = None,
        on_uninterruptible_cancel: Callable[[str], None] | None = None,
        opus_codec: object | None = None,
    ) -> None:
        if (
            isinstance(cancellation_grace_seconds, bool)
            or not isinstance(cancellation_grace_seconds, (int, float))
            or not math.isfinite(cancellation_grace_seconds)
            or cancellation_grace_seconds <= 0
        ):
            raise ValueError("cancellation grace must be finite and positive")
        self._producer_manifest = producer_manifest
        self._producer_fingerprint = producer_manifest.fingerprint
        self._root = initialize_producer_namespace(root, producer_manifest)
        self._orchestrator = orchestrator
        self._clock = clock
        self._execution_clock = execution_clock
        self._cancellation_grace_seconds = float(cancellation_grace_seconds)
        self._require_ready = require_ready or (lambda: None)
        self._on_uninterruptible_cancel = on_uninterruptible_cancel or (
            lambda _reason: None
        )
        self._opus_codec = opus_codec
        self._lock = Lock()
        self._in_flight: dict[tuple[str, str], _InFlightRender] = {}
        self._opus_in_flight: dict[tuple[str, bytes], Future[bytes]] = {}
        self._executor = ThreadPoolExecutor(
            max_workers=4,
            thread_name_prefix="streammuse-rap-render",
        )
        self._closed = False

    @property
    def namespace_root(self) -> Path:
        return self._root

    @property
    def producer_fingerprint(self) -> str:
        return self._producer_fingerprint

    def render_or_load(
        self,
        request: RemoteRapChunkRequest,
        canonical_request: bytes,
        *,
        execution: SynthesisExecutionContext | None = None,
    ) -> _StoredResponse:
        active_execution = execution or SynthesisExecutionContext.from_timeout(
            request.remaining_budget_ms / 1000.0,
            correlation_id=request.request_id,
            clock=self._execution_clock,
        )
        waiter = self.join(request, canonical_request, execution=active_execution)
        try:
            active_execution.checkpoint()
            try:
                response = waiter.future.result(
                    timeout=max(0.001, active_execution.remaining_seconds())
                )
            except FutureTimeout as exc:
                active_execution.cancel("waiter_deadline_exceeded")
                raise ExecutionDeadlineExceeded(
                    "render waiter deadline has elapsed"
                ) from exc
            active_execution.checkpoint()
            return response
        finally:
            waiter.detach(
                "waiter_cancelled" if active_execution.cancelled else "waiter_complete"
            )

    def join(
        self,
        request: RemoteRapChunkRequest,
        canonical_request: bytes,
        *,
        execution: SynthesisExecutionContext,
    ) -> _RenderWaiter:
        if not isinstance(execution, SynthesisExecutionContext):
            raise ValueError("render waiter requires an execution context")
        execution.checkpoint()
        self._require_ready()
        key = (self._producer_fingerprint, request.request_id)
        waiter_id = uuid.uuid4().hex
        start_owner = False
        with self._lock:
            if self._closed:
                raise _ServiceDegraded("render artifact store is closed")
            state = self._in_flight.get(key)
            if state is not None:
                if state.canonical_request != canonical_request:
                    raise _IdempotencyConflict
            else:
                state = _InFlightRender(
                    key=key,
                    canonical_request=canonical_request,
                    future=Future(),
                    owner_execution=execution.owner_context(),
                    queued_at=float(self._execution_clock()),
                )
                self._in_flight[key] = state
                start_owner = True
            state.waiters.add(waiter_id)

        waiter = _RenderWaiter(self, state, waiter_id, execution)
        if start_owner:
            try:
                self._executor.submit(self._run_owner, state, request)
            except BaseException as error:
                with self._lock:
                    if self._in_flight.get(key) is state:
                        self._in_flight.pop(key, None)
                state.completed.set()
                state.future.set_exception(error)
                raise
            Thread(
                target=self._watch_owner_deadline,
                args=(state,),
                name=f"rap-deadline-{request.request_id[:12]}",
                daemon=True,
            ).start()
        return waiter

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            states = tuple(self._in_flight.values())
        for state in states:
            self._cancel_owner(state, "server_shutdown")
        self._executor.shutdown(wait=False, cancel_futures=False)

    def _run_owner(
        self,
        state: _InFlightRender,
        request: RemoteRapChunkRequest,
    ) -> None:
        execution = state.owner_execution
        queue_time_ms = max(
            0.0, (float(self._execution_clock()) - state.queued_at) * 1000.0
        )
        try:
            execution.checkpoint()
            response = self._load_completed(
                request.request_id,
                state.canonical_request,
                execution=execution,
            )
            if response is None:
                execution.checkpoint()
                self._write_request(request.request_id, state.canonical_request)
                execution.checkpoint()
                artifact = self._orchestrator.render(request, execution=execution)
                execution.checkpoint()
                response = self._persist_success(
                    request,
                    artifact,
                    execution=execution,
                    queue_time_ms=queue_time_ms,
                )
        except BaseException as error:
            try:
                if not isinstance(error, _IdempotencyConflict):
                    self._persist_failure(
                        request.request_id,
                        error,
                        execution=execution,
                    )
            except BaseException:
                pass
            if not state.future.done():
                state.future.set_exception(error)
        else:
            if not state.future.done():
                state.future.set_result(response)
        finally:
            state.completed.set()
            with self._lock:
                if self._in_flight.get(state.key) is state:
                    self._in_flight.pop(state.key, None)

    def _detach_waiter(self, waiter: _RenderWaiter, *, reason: str) -> None:
        with self._lock:
            state = waiter.state
            state.waiters.discard(waiter.waiter_id)
            should_cancel = not state.waiters and not state.future.done()
        if should_cancel:
            self._cancel_owner(state, reason)

    def _watch_owner_deadline(self, state: _InFlightRender) -> None:
        if state.completed.wait(timeout=state.owner_execution.remaining_seconds()):
            return
        self._cancel_owner(state, "owner_deadline_exceeded")

    def _cancel_owner(self, state: _InFlightRender, reason: str) -> None:
        if state.future.done():
            return
        if not state.owner_execution.cancel(reason):
            return
        with self._lock:
            if state.cancellation_monitor_started:
                return
            state.cancellation_monitor_started = True
        Thread(
            target=self._monitor_cancellation_grace,
            args=(state,),
            name=f"rap-cancel-{state.key[1][:12]}",
            daemon=True,
        ).start()

    def _monitor_cancellation_grace(self, state: _InFlightRender) -> None:
        started = time.monotonic()
        completed = state.completed.wait(timeout=self._cancellation_grace_seconds)
        outcome = state.owner_execution.cancellation_outcome
        if completed and outcome != "transport_closed_abort_unconfirmed":
            return
        remaining = self._cancellation_grace_seconds - (time.monotonic() - started)
        if remaining > 0:
            Event().wait(timeout=remaining)
        if state.owner_execution.upstream_abort_confirmed:
            return
        if (
            state.completed.is_set()
            and state.owner_execution.cancellation_outcome
            != "transport_closed_abort_unconfirmed"
        ):
            return
        state.owner_execution.record_cancellation_grace_exceeded()
        try:
            self._persist_failure(
                state.key[1],
                ExecutionCancelled("upstream release was not confirmed"),
                execution=state.owner_execution,
            )
        except BaseException:
            pass
        self._on_uninterruptible_cancel("cancellation_grace_exceeded")

    def _load_completed(
        self,
        request_id: str,
        canonical_request: bytes,
        *,
        execution: SynthesisExecutionContext | None = None,
    ) -> _StoredResponse | None:
        if execution is not None:
            execution.checkpoint()
        try:
            verify_producer_namespace(self._root, self._producer_manifest)
        except ProducerManifestError as exc:
            raise _CacheIntegrityError("producer namespace verification failed") from exc
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
        complete_path = workspace / _COMPLETE_FILE
        if request_path.exists() and (package_path.exists() or complete_path.exists()):
            if not package_path.is_file() or not complete_path.is_file():
                raise _CacheIntegrityError("completed cache artifact is incomplete")
            try:
                complete = json.loads(complete_path.read_bytes())
            except (json.JSONDecodeError, OSError) as exc:
                raise _CacheIntegrityError("cache completion marker is invalid") from exc
            expected_complete = {
                "schema_version": "streammuse.rap_cache_complete.v1",
                "request_id": request_id,
                "producer_fingerprint": self._producer_fingerprint,
                "package_sha256": hashlib.sha256(package_path.read_bytes()).hexdigest(),
            }
            if complete != expected_complete:
                raise _CacheIntegrityError("cache completion marker does not match")
            sidecar_path = workspace / _MOSS_SIDECAR_FILE
            if not sidecar_path.is_file():
                raise _CacheIntegrityError("completed cache is missing private metadata")
            self._validate_completed_moss_sidecar(sidecar_path, request_id)
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
            package = package_path.read_bytes()
            try:
                from streammuse.infrastructure.rap.chunk_package import decode_chunk_package

                decode_chunk_package(package, expected_request_id=request_id)
            except Exception as exc:
                raise _CacheIntegrityError("cached package validation failed") from exc
            if execution is not None:
                execution.checkpoint()
            return _StoredResponse(package, server_timing)
        return None

    def _persist_success(
        self,
        request: RemoteRapChunkRequest,
        artifact: RemoteChunkRenderArtifact,
        *,
        execution: SynthesisExecutionContext,
        queue_time_ms: float,
    ) -> _StoredResponse:
        from streammuse.application.rap.chunk_orchestration import PhraseRenderFailed
        from streammuse.infrastructure.rap.chunk_package import encode_chunk_package

        if artifact.manifest.request_id != request.request_id:
            raise PhraseRenderFailed("render artifact request identity mismatch")
        execution.checkpoint()
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
        source_only = artifact.manifest.schema_version == REMOTE_CHUNK_SCHEMA_VERSION_V2
        if not source_only:
            # v2 servers stop after MMS; the warped vocal is produced on the Mac.
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
        if not source_only:
            self._atomic_write(workspace / _ALIGNED_WAV_FILE, artifact.vocal_wav)
        self._write_moss_sidecar(
            request,
            artifact,
            workspace,
            queue_time_ms=queue_time_ms,
        )
        execution.checkpoint()

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
        execution.checkpoint()
        self._write_json(workspace / _MANIFEST_FILE, final_manifest.to_payload())
        package = encode_chunk_package(final_manifest, artifact.vocal_wav)
        server_timing = _server_timing(final_manifest)
        self._write_json(
            workspace / _SERVER_TIMING_FILE, {"server_timing": server_timing}
        )
        # The final atomic replacement is the sole successful-cache marker.
        execution.checkpoint()
        with self._lock:
            self._durably_unpublish(workspace / _OPUS_RESPONSE_FILE)
            self._durably_unpublish(workspace / _COMPLETE_FILE)
            response_path = workspace / _RESPONSE_FILE
            self._publish_response(response_path, package)
            try:
                execution.checkpoint()
                self._write_json(
                    workspace / _COMPLETE_FILE,
                    {
                        "schema_version": "streammuse.rap_cache_complete.v1",
                        "request_id": request.request_id,
                        "producer_fingerprint": self._producer_fingerprint,
                        "package_sha256": hashlib.sha256(package).hexdigest(),
                    },
                )
            except BaseException:
                self._durably_unpublish(response_path)
                raise
        return _StoredResponse(package, server_timing)

    def _write_moss_sidecar(
        self,
        request: RemoteRapChunkRequest,
        artifact: RemoteChunkRenderArtifact,
        workspace: Path,
        *,
        queue_time_ms: float,
    ) -> None:
        serving_metadata = dict(artifact.moss_serving_metadata)
        if not serving_metadata.get("resolved_generation_settings"):
            serving_metadata["resolved_generation_settings"] = _json_value(
                self._producer_manifest.generation["settings"]
            )
        metadata = _private_moss_metadata(
            serving_metadata,
            backend=self._producer_manifest.backend,
            correlation_id=request.request_id,
            model_revision=str(self._producer_manifest.model["revision"]),
            reference_audio_sha256=str(
                self._producer_manifest.reference["audio_sha256"]
            ),
            reference_text_sha256=self._producer_manifest.reference.get(
                "text_sha256"
            ),
            config_sha256=self._producer_manifest.runtime.get("config_sha256"),
        )
        payload = {
            "schema_version": "streammuse.moss_synthesis.v1",
            "request_id": request.request_id,
            "producer_fingerprint": self._producer_fingerprint,
            "cache_hit": False,
            "synthesis_outcome": "success",
            "queue_time_ms": queue_time_ms,
            **metadata,
        }
        self._write_private_moss_sidecar(workspace, payload)

    def _validate_completed_moss_sidecar(
        self,
        sidecar_path: Path,
        request_id: str,
    ) -> None:
        try:
            data = sidecar_path.read_bytes()
        except OSError as exc:
            raise _CacheIntegrityError("private MOSS metadata is unreadable") from exc
        if not data or len(data) > _PRIVATE_SIDECAR_MAX_BYTES:
            raise _CacheIntegrityError("private MOSS metadata size is invalid")
        try:
            raw = json.loads(data)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise _CacheIntegrityError("private MOSS metadata is invalid JSON") from exc
        canonical = _bounded_private_json(raw, depth=0)
        if not isinstance(canonical, dict):
            raise _CacheIntegrityError("private MOSS metadata is invalid")
        expected_keys = {
            "schema_version",
            "request_id",
            "producer_fingerprint",
            "cache_hit",
            "synthesis_outcome",
            "queue_time_ms",
            *_PRIVATE_MOSS_KEYS,
        }
        if set(canonical) != expected_keys:
            raise _CacheIntegrityError("private MOSS metadata schema does not match")
        expected_reference_text = self._producer_manifest.reference.get(
            "text_sha256"
        )
        trusted_values = {
            "schema_version": "streammuse.moss_synthesis.v1",
            "request_id": request_id,
            "producer_fingerprint": self._producer_fingerprint,
            "cache_hit": False,
            "synthesis_outcome": "success",
            "moss_backend": self._producer_manifest.backend,
            "correlation_id": request_id,
            "model_revision": str(self._producer_manifest.model["revision"]),
            "reference_audio_sha256": str(
                self._producer_manifest.reference["audio_sha256"]
            ),
            "reference_text_sha256": (
                expected_reference_text
                if isinstance(expected_reference_text, str)
                else "unavailable"
            ),
            "config_sha256": str(
                self._producer_manifest.runtime["config_sha256"]
            ),
        }
        if any(canonical.get(name) != value for name, value in trusted_values.items()):
            raise _CacheIntegrityError("private MOSS metadata identity does not match")
        queue_time_ms = canonical.get("queue_time_ms")
        if (
            type(queue_time_ms) not in {int, float}
            or not math.isfinite(queue_time_ms)
            or queue_time_ms < 0
        ):
            raise _CacheIntegrityError("private MOSS queue timing is invalid")
        resolved = canonical.get("resolved_generation_settings")
        producer_settings = _json_value(
            self._producer_manifest.generation["settings"]
        )
        if not isinstance(resolved, Mapping) or not isinstance(
            producer_settings, Mapping
        ):
            raise _CacheIntegrityError("private MOSS generation settings are invalid")
        if any(resolved.get(name) != value for name, value in producer_settings.items()):
            raise _CacheIntegrityError(
                "private MOSS generation settings do not match producer"
            )

    def _write_private_moss_sidecar(
        self,
        workspace: Path,
        payload: Mapping[str, object],
    ) -> None:
        data = json.dumps(
            payload,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        if len(data) > _PRIVATE_SIDECAR_MAX_BYTES:
            raise _CacheIntegrityError("private MOSS metadata exceeds its size limit")
        self._atomic_write(workspace / _MOSS_SIDECAR_FILE, data)

    def load_or_create_opus(self, request: RemoteRapChunkRequest, stored: _StoredResponse) -> bytes:
        """Derive a lossy delivery package without altering canonical cached artifacts."""
        if self._opus_codec is None:
            raise RuntimeError("Opus response requested but no Opus codec is configured")
        workspace = self._workspace(request.request_id)
        path = workspace / _OPUS_RESPONSE_FILE
        canonical_path = workspace / _RESPONSE_FILE
        canonical_identity = hashlib.sha256(stored.package).digest()
        in_flight_key = (request.request_id, canonical_identity)
        with self._lock:
            future = self._opus_in_flight.get(in_flight_key)
            if future is None:
                future = Future()
                self._opus_in_flight[in_flight_key] = future
                owner = True
            else:
                owner = False

        if not owner:
            return future.result()

        try:
            with self._lock:
                canonical_matches = (
                    canonical_path.is_file()
                    and canonical_path.read_bytes() == stored.package
                )
                package = path.read_bytes() if canonical_matches and path.is_file() else None
            if package is None:
                from streammuse.infrastructure.rap.chunk_package import (
                    decode_chunk_package,
                    encode_opus_chunk_package,
                )

                canonical = decode_chunk_package(
                    stored.package, expected_request_id=request.request_id
                )
                package = encode_opus_chunk_package(
                    canonical.manifest, canonical.vocal_wav, self._opus_codec
                )
                with self._lock:
                    if (
                        canonical_path.is_file()
                        and canonical_path.read_bytes() == stored.package
                    ):
                        self._publish_response(path, package)
        except BaseException as error:
            future.set_exception(error)
            raise
        else:
            future.set_result(package)
            return package
        finally:
            with self._lock:
                if self._opus_in_flight.get(in_flight_key) is future:
                    self._opus_in_flight.pop(in_flight_key, None)

    def _persist_failure(
        self,
        request_id: str,
        error: BaseException,
        *,
        execution: SynthesisExecutionContext | None = None,
    ) -> None:
        from streammuse.application.rap.chunk_orchestration import NoValidCandidates

        workspace = self._workspace(request_id)
        if (workspace / _COMPLETE_FILE).is_file():
            return
        if isinstance(error, NoValidCandidates):
            self._write_json(workspace / _CANDIDATE_LEDGER_FILE, error.candidate_ledger)
        code, _status = _error_spec(error)
        self._write_json(workspace / _FAILURE_FILE, {"code": code})
        cancellation_outcome = (
            execution.cancellation_outcome
            if execution is not None
            else "not_requested"
        )
        deadline_outcome = (
            execution.deadline_outcome
            if execution is not None
            else "within_deadline"
        )
        metadata = _private_moss_metadata(
            {
                "cancellation_outcome": cancellation_outcome,
                "deadline_outcome": deadline_outcome,
                "upstream_abort_confirmed": (
                    execution.upstream_abort_confirmed
                    if execution is not None
                    else False
                ),
                "cancellation_grace_exceeded": (
                    execution.cancellation_grace_exceeded
                    if execution is not None
                    else False
                ),
                "recovery_outcome": (
                    execution.recovery_outcome
                    if execution is not None
                    else "not_required"
                ),
            },
            backend=self._producer_manifest.backend,
            correlation_id=request_id,
            model_revision=str(self._producer_manifest.model["revision"]),
            reference_audio_sha256=str(
                self._producer_manifest.reference["audio_sha256"]
            ),
            reference_text_sha256=self._producer_manifest.reference.get(
                "text_sha256"
            ),
            config_sha256=self._producer_manifest.runtime.get("config_sha256"),
        )
        self._write_private_moss_sidecar(
            workspace,
            {
                "schema_version": "streammuse.moss_synthesis.v1",
                "request_id": request_id,
                "producer_fingerprint": self._producer_fingerprint,
                "cache_hit": False,
                "synthesis_outcome": "failure",
                "queue_time_ms": 0.0,
                **metadata,
            },
        )

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
                _OPUS_RESPONSE_FILE,
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
                _COMPLETE_FILE,
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
    producer_manifest: ProducerManifestV1,
    artifact_root: str | Path = "rap-chunk-artifacts",
    wire_audio_codec: str = "pcm",
    opus_codec: object | None = None,
    cancellation_grace_seconds: float = 2.0,
    monotonic_clock: Callable[[], float] = time.monotonic,
) -> FastAPI:
    """Create the testable private render service without composing model dependencies."""
    from streammuse.infrastructure.rap.chunk_package import (
        RAP_CHUNK_PACKAGE_MEDIA_TYPE,
        RAP_CHUNK_OPUS_PACKAGE_MEDIA_TYPE,
    )

    if wire_audio_codec not in {"pcm", "opus"}:
        raise ValueError("wire_audio_codec must be pcm or opus")
    if wire_audio_codec == "opus" and opus_codec is None:
        raise ValueError("Opus server mode requires an Opus codec")
    runtime_state = _RuntimeState(health)
    store = _ArtifactStore(
        Path(artifact_root),
        orchestrator,
        producer_manifest,
        execution_clock=monotonic_clock,
        cancellation_grace_seconds=cancellation_grace_seconds,
        require_ready=runtime_state.require_ready,
        on_uninterruptible_cancel=runtime_state.degrade,
        opus_codec=opus_codec,
    )
    app = FastAPI(
        title="StreamMUSE Private Rap Renderer", docs_url=None, redoc_url=None
    )
    app.state.rap_artifact_store = store
    app.state.rap_runtime_state = runtime_state
    app.add_event_handler("shutdown", store.close)

    @app.get("/health")
    async def get_health() -> dict[str, object]:
        return _public_health(runtime_state.snapshot())

    @app.post("/v1/rap/chunks/render")
    async def render_chunk(http_request: Request) -> Response:
        accepted_at = float(monotonic_clock())
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
            now = float(monotonic_clock())
            execution = SynthesisExecutionContext(
                deadline_monotonic=(
                    accepted_at + request.remaining_budget_ms / 1000.0
                ),
                correlation_id=request.request_id,
                clock=monotonic_clock,
                validation_time_monotonic=now,
            )
            execution.checkpoint()
            waiter = store.join(
                request,
                request.canonical_json_bytes(),
                execution=execution,
            )
        except ExecutionDeadlineExceeded:
            return _error_response(
                "budget_exhausted", 422, "rap chunk request budget expired"
            )
        except _ServiceDegraded:
            return _error_response(
                "service_degraded",
                503,
                "rap render service requires restart",
            )
        except _IdempotencyConflict:
            return _error_response(
                "idempotency_conflict",
                409,
                "request ID is already bound to another request",
            )

        opus_requested = (
            wire_audio_codec == "opus"
            and _accepts_media_type(http_request.headers.get("Accept", ""), RAP_CHUNK_OPUS_PACKAGE_MEDIA_TYPE)
        )
        try:
            stored = await _await_render_waiter(http_request, waiter)
            execution.checkpoint()
            package = (
                await run_in_threadpool(store.load_or_create_opus, request, stored)
                if opus_requested
                else stored.package
            )
            execution.checkpoint()
        except _IdempotencyConflict:
            return _error_response(
                "idempotency_conflict",
                409,
                "request ID is already bound to another request",
            )
        except ExecutionDeadlineExceeded:
            return _error_response(
                "budget_exhausted", 422, "rap chunk request budget expired"
            )
        except ExecutionCancelled:
            return _error_response(
                "request_cancelled", 499, "rap chunk request was cancelled"
            )
        except _ServiceDegraded:
            return _error_response(
                "service_degraded",
                503,
                "rap render service requires restart",
            )
        except Exception as error:
            code, status = _error_spec(error)
            message = (
                "rap chunk render could not be completed"
                if code != "internal_error"
                else "rap chunk render failed"
            )
            return _error_response(code, status, message)

        media_type = RAP_CHUNK_OPUS_PACKAGE_MEDIA_TYPE if opus_requested else RAP_CHUNK_PACKAGE_MEDIA_TYPE
        return Response(
            content=package,
            media_type=media_type,
            headers={
                "X-StreamMUSE-Request-ID": request.request_id,
                "Content-Length": str(len(package)),
                "Server-Timing": stored.server_timing,
            },
        )

    return app


async def _await_render_waiter(
    http_request: Request,
    waiter: _RenderWaiter,
) -> _StoredResponse:
    execution = waiter.execution
    detach_reason = "waiter_complete"
    try:
        while not waiter.future.done():
            execution.checkpoint()
            if await http_request.is_disconnected():
                detach_reason = "client_disconnected"
                execution.cancel(detach_reason)
                raise ExecutionCancelled("render client disconnected")
            await asyncio.sleep(
                max(0.001, min(0.025, execution.remaining_seconds()))
            )
        result = waiter.future.result()
        execution.checkpoint()
        return result
    except asyncio.CancelledError:
        detach_reason = "client_task_cancelled"
        execution.cancel(detach_reason)
        raise
    except ExecutionDeadlineExceeded:
        detach_reason = "waiter_deadline_exceeded"
        execution.cancel(detach_reason)
        raise
    except ExecutionCancelled:
        detach_reason = execution.cancel_reason
        raise
    finally:
        waiter.detach(detach_reason)


def _accepts_media_type(accept: str, expected: str) -> bool:
    """Only explicit, non-zero-quality media types opt into a lossy response."""
    for item in accept.lower().split(","):
        parts = [part.strip() for part in item.split(";")]
        if not parts or parts[0] != expected:
            continue
        quality = 1.0
        for parameter in parts[1:]:
            name, separator, value = parameter.partition("=")
            if name.strip() != "q":
                continue
            if not separator:
                return False
            try:
                quality = float(value.strip())
            except ValueError:
                return False
            if not math.isfinite(quality) or not 0.0 <= quality <= 1.0:
                return False
        return quality > 0.0
    return False


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


_PRIVATE_MOSS_KEYS = frozenset(
    {
        "moss_backend",
        "correlation_id",
        "moss_service_request_ms",
        "moss_http_response_headers_ms",
        "moss_http_first_body_byte_ms",
        "moss_response_download_ms",
        "moss_response_validation_ms",
        "moss_response_bytes",
        "server_identity",
        "server_version",
        "model_revision",
        "reference_audio_sha256",
        "reference_text_sha256",
        "config_sha256",
        "cancellation_outcome",
        "deadline_outcome",
        "upstream_abort_confirmed",
        "cancellation_grace_exceeded",
        "recovery_outcome",
        "streaming",
        "internal_stage_timings",
        "resolved_generation_settings",
    }
)


def _private_moss_metadata(
    value: Mapping[str, object],
    *,
    backend: str,
    correlation_id: str,
    model_revision: str,
    reference_audio_sha256: str,
    reference_text_sha256: object,
    config_sha256: object,
) -> dict[str, object]:
    unknown = set(value) - _PRIVATE_MOSS_KEYS
    if unknown:
        raise _CacheIntegrityError("private MOSS metadata contains unknown fields")
    defaults: dict[str, object] = {
        "moss_backend": backend,
        "correlation_id": correlation_id,
        "moss_service_request_ms": 0.0,
        "moss_http_response_headers_ms": 0.0,
        "moss_http_first_body_byte_ms": 0.0,
        "moss_response_download_ms": 0.0,
        "moss_response_validation_ms": 0.0,
        "moss_response_bytes": 0,
        "server_identity": backend,
        "server_version": "unknown",
        "model_revision": model_revision,
        "reference_audio_sha256": reference_audio_sha256,
        "reference_text_sha256": (
            reference_text_sha256
            if isinstance(reference_text_sha256, str)
            else "unavailable"
        ),
        "config_sha256": (
            config_sha256 if isinstance(config_sha256, str) else "unavailable"
        ),
        "cancellation_outcome": "not_requested",
        "deadline_outcome": "within_deadline",
        "upstream_abort_confirmed": False,
        "cancellation_grace_exceeded": False,
        "recovery_outcome": "not_required",
        "streaming": False,
        "internal_stage_timings": {
            "status": "unavailable",
            "unavailable_reason": "backend_did_not_report_stage_timings",
        },
        "resolved_generation_settings": {},
    }
    trusted_defaults = {
        name: defaults[name]
        for name in (
            "config_sha256",
            "model_revision",
            "reference_audio_sha256",
            "reference_text_sha256",
        )
    }
    defaults.update(value)
    for name, trusted in trusted_defaults.items():
        reported = defaults[name]
        if isinstance(reported, str) and reported in {
            "unknown",
            "unavailable",
        }:
            defaults[name] = trusted
        elif reported != trusted:
            raise _CacheIntegrityError(
                "private MOSS metadata does not match producer identity"
            )
    canonical = _bounded_private_json(defaults, depth=0)
    if not isinstance(canonical, dict):
        raise _CacheIntegrityError("private MOSS metadata is invalid")
    if canonical["moss_backend"] != backend:
        raise _CacheIntegrityError("private MOSS backend does not match producer")
    if canonical["correlation_id"] != correlation_id:
        raise _CacheIntegrityError("private MOSS correlation ID does not match request")
    for name in (
        "moss_service_request_ms",
        "moss_http_response_headers_ms",
        "moss_http_first_body_byte_ms",
        "moss_response_download_ms",
        "moss_response_validation_ms",
    ):
        item = canonical[name]
        if type(item) not in {int, float} or item < 0:
            raise _CacheIntegrityError("private MOSS timing is invalid")
    if type(canonical["moss_response_bytes"]) is not int or canonical[
        "moss_response_bytes"
    ] < 0:
        raise _CacheIntegrityError("private MOSS response size is invalid")
    if canonical["cancellation_outcome"] not in {
        "not_requested",
        "cancel_requested",
        "cancel_requested_but_not_interruptible",
        "transport_closed_abort_unconfirmed",
        "upstream_abort_confirmed",
    }:
        raise _CacheIntegrityError("private MOSS cancellation outcome is invalid")
    if canonical["deadline_outcome"] not in {
        "within_deadline",
        "deadline_exceeded",
    }:
        raise _CacheIntegrityError("private MOSS deadline outcome is invalid")
    if type(canonical["upstream_abort_confirmed"]) is not bool:
        raise _CacheIntegrityError("private MOSS abort confirmation is invalid")
    if type(canonical["cancellation_grace_exceeded"]) is not bool:
        raise _CacheIntegrityError("private MOSS cancellation grace flag is invalid")
    if canonical["recovery_outcome"] not in {
        "not_required",
        "abort_unconfirmed",
        "upstream_released",
        "restart_required",
    }:
        raise _CacheIntegrityError("private MOSS recovery outcome is invalid")
    if canonical["upstream_abort_confirmed"] != (
        canonical["cancellation_outcome"] == "upstream_abort_confirmed"
    ):
        raise _CacheIntegrityError("private MOSS abort evidence is inconsistent")
    expected_recovery = (
        "not_required"
        if canonical["cancellation_outcome"] == "not_requested"
        else "restart_required"
        if canonical["cancellation_grace_exceeded"]
        else "upstream_released"
        if canonical["upstream_abort_confirmed"]
        else "abort_unconfirmed"
    )
    if canonical["recovery_outcome"] != expected_recovery:
        raise _CacheIntegrityError("private MOSS recovery evidence is inconsistent")
    if canonical["streaming"] is not False:
        raise _CacheIntegrityError("private MOSS streaming flag is invalid")
    return canonical


def _bounded_private_json(value: object, *, depth: int) -> object:
    if depth > 8:
        raise _CacheIntegrityError("private MOSS metadata nesting is too deep")
    if value is None or type(value) in {bool, int}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise _CacheIntegrityError("private MOSS metadata contains non-finite data")
        return value
    if isinstance(value, str):
        if (
            len(value) > 1024
            or any(ord(character) < 32 for character in value)
            or _looks_like_absolute_path(value)
            or "://" in value
        ):
            raise _CacheIntegrityError("private MOSS metadata contains unsafe text")
        return value
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key or len(key) > 128:
                raise _CacheIntegrityError("private MOSS metadata key is invalid")
            result[key] = _bounded_private_json(item, depth=depth + 1)
        return result
    if isinstance(value, (list, tuple)):
        if len(value) > 1024:
            raise _CacheIntegrityError("private MOSS metadata array is too large")
        return [_bounded_private_json(item, depth=depth + 1) for item in value]
    raise _CacheIntegrityError("private MOSS metadata contains a non-JSON value")


def _error_spec(error: Exception) -> tuple[str, int]:
    from streammuse.application.rap.chunk_orchestration import (
        NoValidCandidates,
        PhraseRenderFailed,
        RenderBudgetExpired,
    )

    if isinstance(error, (RenderBudgetExpired, ExecutionDeadlineExceeded)):
        return "budget_exhausted", 422
    if isinstance(error, ExecutionCancelled):
        return "request_cancelled", 503
    if isinstance(error, _ServiceDegraded):
        return "service_degraded", 503
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
    for key in _HEALTH_TOP_KEYS:
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
    if key in {"ready", "restart_required"}:
        return value if type(value) is bool else _INVALID_HEALTH_VALUE
    if key in {"endpoint_port", "warmup_time_ms", "warmup"} and type(value) in {
        int,
        float,
    }:
        if math.isfinite(value) and abs(value) <= 1_000_000_000:
            return value
        return _INVALID_HEALTH_VALUE
    if not isinstance(value, str):
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
    return _INVALID_HEALTH_VALUE


def _bounded_reason_code(value: object) -> str:
    text = str(value).strip().lower().replace("-", "_")[:64]
    if not text or any(not (character.isalnum() or character == "_") for character in text):
        return "cancellation_grace_exceeded"
    return text


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
    parser.add_argument(
        "--moss-serving-backend",
        choices=_MOSS_SERVING_BACKENDS,
        default="inprocess",
    )
    parser.add_argument("--moss-device")
    parser.add_argument("--moss-reference-wav", required=True)
    parser.add_argument("--moss-sglang-url")
    parser.add_argument("--moss-reference-text-file")
    parser.add_argument("--moss-sglang-reference-uri")
    parser.add_argument("--moss-sglang-reference-sha256")
    parser.add_argument("--moss-request-timeout-s", type=float, default=120.0)
    parser.add_argument("--moss-cancellation-grace-s", type=float, default=2.0)
    parser.add_argument("--moss-model-revision")
    parser.add_argument("--moss-runtime-version")
    parser.add_argument("--moss-runtime-revision")
    parser.add_argument("--moss-sglang-version")
    parser.add_argument("--moss-sglang-revision")
    parser.add_argument("--moss-runtime-environment-sha256")
    parser.add_argument(
        "--moss-runtime-patch-sha256",
        help="SHA-256 of the patch applied to the pinned SGLang-Omni runtime "
        "(the launch manifest's runtime.patch_sha256); omit for an unpatched runtime",
    )
    parser.add_argument("--moss-runtime-config")
    parser.add_argument("--moss-runtime-config-sha256")
    parser.add_argument("--moss-mlx-url")
    parser.add_argument("--mlx-version")
    parser.add_argument("--mlx-audio-version")
    parser.add_argument("--mlx-audio-commit")
    parser.add_argument("--mlx-moss-quantization")
    parser.add_argument("--mlx-moss-weights-sha256")
    parser.add_argument("--mlx-moss-audio-tokenizer-revision")
    parser.add_argument(
        "--aligner-device",
        default="auto",
        help="auto picks cuda, then mps, then cpu",
    )
    parser.add_argument("--aligner-cache")
    parser.add_argument(
        "--candidate-profile", choices=tuple(_CANDIDATE_PROFILES), default="realtime"
    )
    parser.add_argument(
        "--moss-warp-policy",
        choices=_MOSS_WARP_POLICIES,
        default="gentle_sparse_r3",
    )
    parser.add_argument("--wire-audio-codec", choices=("pcm", "opus"), default="pcm")
    parser.add_argument(
        "--opus-compression-level",
        type=int,
        choices=range(11),
        default=5,
        metavar="{0..10}",
        help="libopus complexity for --wire-audio-codec opus (10 was the pre-2026-10 default)",
    )
    parser.add_argument(
        "--concurrent-bar-generation",
        action="store_true",
        help="generate both bars' initial candidate waves at once, one chat client per bar",
    )
    return parser


def main(
    argv: list[str] | None = None,
    *,
    composition_factory: Callable[[RapRenderServerConfig], object] | None = None,
    serve: Callable[..., object] | None = None,
) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.host not in _LOOPBACK_HOSTS and not args.allow_public_bind:
        print("refusing non-loopback bind without --allow-public-bind", file=sys.stderr)
        return 2
    try:
        config = _server_config_from_args(args)
    except ValueError as exc:
        parser.error(str(exc))
    run_server = serve
    if run_server is None:
        import uvicorn

        run_server = uvicorn.run
    compose = composition_factory or _compose_real_worker
    opus_codec = None
    if config.wire_audio_codec == "opus":
        from streammuse.infrastructure.rap.opus_codec import FFmpegOpusCodec

        opus_codec = FFmpegOpusCodec(compression_level=config.opus_compression_level)
        opus_codec.probe()
    composition = compose(config)
    try:
        run_server(
            create_rap_render_app(
                composition.orchestrator,
                composition.health,
                producer_manifest=composition.producer_manifest,
                artifact_root=config.artifact_root,
                wire_audio_codec=config.wire_audio_codec,
                opus_codec=opus_codec,
                cancellation_grace_seconds=config.moss_cancellation_grace_s,
            ),
            host=config.host,
            port=config.port,
            log_level="info",
        )
    finally:
        composition.close()
    return 0


def _server_config_from_args(args: argparse.Namespace) -> RapRenderServerConfig:
    backend = args.moss_serving_backend
    sglang_only = {
        "--moss-sglang-url": args.moss_sglang_url,
        "--moss-reference-text-file": args.moss_reference_text_file,
        "--moss-sglang-reference-uri": args.moss_sglang_reference_uri,
        "--moss-sglang-reference-sha256": args.moss_sglang_reference_sha256,
        "--moss-runtime-version": args.moss_runtime_version,
        "--moss-runtime-revision": args.moss_runtime_revision,
        "--moss-sglang-version": args.moss_sglang_version,
        "--moss-sglang-revision": args.moss_sglang_revision,
        "--moss-runtime-environment-sha256": (
            args.moss_runtime_environment_sha256
        ),
        "--moss-runtime-config": args.moss_runtime_config,
        "--moss-runtime-config-sha256": args.moss_runtime_config_sha256,
    }
    mlx_only = {
        "--moss-mlx-url": args.moss_mlx_url,
        "--mlx-version": args.mlx_version,
        "--mlx-audio-version": args.mlx_audio_version,
        "--mlx-audio-commit": args.mlx_audio_commit,
        "--mlx-moss-quantization": args.mlx_moss_quantization,
        "--mlx-moss-weights-sha256": args.mlx_moss_weights_sha256,
        "--mlx-moss-audio-tokenizer-revision": args.mlx_moss_audio_tokenizer_revision,
    }
    sglang_conflicts = [name for name, value in sglang_only.items() if value is not None]
    if args.moss_runtime_patch_sha256 is not None:
        # Optional (an unpatched runtime omits it), so it is not in sglang_only.
        sglang_conflicts.append("--moss-runtime-patch-sha256")
    mlx_conflicts = [name for name, value in mlx_only.items() if value is not None]
    if backend != "sglang-omni" and sglang_conflicts:
        raise ValueError(
            f"{', '.join(sglang_conflicts)} may only be used with sglang-omni"
        )
    if backend != "mlx" and mlx_conflicts:
        raise ValueError(f"{', '.join(mlx_conflicts)} may only be used with mlx")
    moss_mlx_runtime: dict[str, str] | None = None
    if backend == "inprocess":
        moss_device = args.moss_device or "auto"
    elif backend == "mlx":
        if args.moss_device is not None:
            raise ValueError("--moss-device is only valid for the inprocess backend")
        missing = [name for name, value in mlx_only.items() if value is None]
        if args.moss_model_revision is None:
            missing.append("--moss-model-revision")
        if missing:
            raise ValueError(f"mlx requires {', '.join(missing)}")
        _validate_origin_url(args.moss_mlx_url, "--moss-mlx-url", loopback_only=True)
        if not Path(args.moss_reference_wav).is_absolute():
            raise ValueError("--moss-reference-wav must be an absolute path")
        _validate_pinned_identity(args.moss_model_revision, "--moss-model-revision")
        _validate_sha256(args.mlx_moss_weights_sha256, "--mlx-moss-weights-sha256")
        for name, value in mlx_only.items():
            if name != "--moss-mlx-url":
                _validate_pinned_identity(value, name)
        moss_mlx_runtime = {
            "mlx_version": args.mlx_version,
            "mlx_audio_version": args.mlx_audio_version,
            "mlx_audio_commit": args.mlx_audio_commit,
            "quantization": args.mlx_moss_quantization,
            "weights_sha256": args.mlx_moss_weights_sha256,
            "audio_tokenizer_revision": args.mlx_moss_audio_tokenizer_revision,
        }
        moss_device = "external"
    else:
        if args.moss_device is not None:
            raise ValueError("--moss-device is only valid for the inprocess backend")
        missing = [name for name, value in sglang_only.items() if value is None]
        if args.moss_model_revision is None:
            missing.append("--moss-model-revision")
        if missing:
            raise ValueError(
                f"sglang-omni requires {', '.join(missing)}"
            )
        _validate_origin_url(
            args.moss_sglang_url,
            "--moss-sglang-url",
            loopback_only=True,
        )
        _validate_service_reference_uri(args.moss_sglang_reference_uri)
        _validate_sha256(
            args.moss_runtime_config_sha256,
            "--moss-runtime-config-sha256",
        )
        _validate_sha256(
            args.moss_sglang_reference_sha256,
            "--moss-sglang-reference-sha256",
        )
        _validate_sha256(
            args.moss_runtime_environment_sha256,
            "--moss-runtime-environment-sha256",
        )
        if args.moss_runtime_patch_sha256 is not None:
            _validate_sha256(
                args.moss_runtime_patch_sha256,
                "--moss-runtime-patch-sha256",
            )
        for value, name in (
            (args.moss_reference_wav, "--moss-reference-wav"),
            (args.moss_reference_text_file, "--moss-reference-text-file"),
            (args.moss_runtime_config, "--moss-runtime-config"),
        ):
            if not Path(value).is_absolute():
                raise ValueError(f"{name} must be an absolute path")
        _validate_pinned_identity(args.moss_model_revision, "--moss-model-revision")
        _validate_pinned_identity(args.moss_runtime_version, "--moss-runtime-version")
        _validate_pinned_identity(args.moss_runtime_revision, "--moss-runtime-revision")
        _validate_pinned_identity(args.moss_sglang_version, "--moss-sglang-version")
        _validate_pinned_identity(
            args.moss_sglang_revision,
            "--moss-sglang-revision",
        )
        moss_device = "external"

    for name, value in (
        ("--moss-request-timeout-s", args.moss_request_timeout_s),
        ("--moss-cancellation-grace-s", args.moss_cancellation_grace_s),
    ):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")

    return RapRenderServerConfig(
        host=args.host,
        port=args.port,
        artifact_root=Path(args.artifact_root),
        vllm_url=args.vllm_url,
        vllm_model=args.vllm_model,
        moss_model=args.moss_model,
        moss_device=moss_device,
        moss_reference_wav=Path(args.moss_reference_wav),
        aligner_device=args.aligner_device,
        aligner_cache=Path(args.aligner_cache) if args.aligner_cache else None,
        candidate_profile=args.candidate_profile,
        moss_warp_policy=args.moss_warp_policy,
        wire_audio_codec=args.wire_audio_codec,
        opus_compression_level=args.opus_compression_level,
        moss_serving_backend=backend,
        moss_sglang_url=args.moss_sglang_url,
        moss_reference_text_file=(
            Path(args.moss_reference_text_file)
            if args.moss_reference_text_file
            else None
        ),
        moss_sglang_reference_uri=args.moss_sglang_reference_uri,
        moss_sglang_reference_sha256=args.moss_sglang_reference_sha256,
        moss_request_timeout_s=args.moss_request_timeout_s,
        moss_cancellation_grace_s=args.moss_cancellation_grace_s,
        moss_model_revision=args.moss_model_revision,
        moss_runtime_version=args.moss_runtime_version,
        moss_runtime_revision=args.moss_runtime_revision,
        moss_sglang_version=args.moss_sglang_version,
        moss_sglang_revision=args.moss_sglang_revision,
        moss_runtime_environment_sha256=args.moss_runtime_environment_sha256,
        moss_runtime_patch_sha256=args.moss_runtime_patch_sha256,
        moss_runtime_config=(
            Path(args.moss_runtime_config) if args.moss_runtime_config else None
        ),
        moss_runtime_config_sha256=args.moss_runtime_config_sha256,
        moss_mlx_url=args.moss_mlx_url,
        moss_mlx_runtime=moss_mlx_runtime,
        concurrent_bar_generation=args.concurrent_bar_generation,
    )


def _validate_origin_url(
    value: object,
    name: str,
    *,
    loopback_only: bool = False,
) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be an HTTP origin")
    parsed = urlsplit(value)
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError(f"{name} contains an invalid port") from exc
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
        or (port is not None and not 1 <= port <= 65535)
        or (
            loopback_only
            and parsed.hostname not in {"127.0.0.1", "::1", "localhost"}
        )
    ):
        raise ValueError(f"{name} must be an HTTP origin without credentials")


def _validate_service_reference_uri(value: object) -> None:
    if not isinstance(value, str) or not value or len(value) > 2048:
        raise ValueError("--moss-sglang-reference-uri must be a local file URI")
    parsed = urlsplit(value)
    if (
        parsed.scheme != "file"
        or parsed.netloc not in {"", "localhost"}
        or not parsed.path.startswith("/")
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("--moss-sglang-reference-uri must be a local file URI")


def _validate_sha256(value: object, name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")


def _validate_pinned_identity(value: object, name: str) -> None:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > 512
        or value.strip().lower() in {"unknown", "unavailable", "main", "master", "latest"}
        or any(ord(character) < 32 for character in value)
    ):
        raise ValueError(f"{name} must identify an immutable pinned build")


def _compose_real_worker(
    config: RapRenderServerConfig,
) -> _WorkerComposition:
    """Load, warm, and own one resident H200 render composition."""
    reference_audio_bytes, reference_text = _validate_composition_inputs(config)
    reference_audio_sha256 = hashlib.sha256(reference_audio_bytes).hexdigest()
    reference_text_sha256 = (
        hashlib.sha256(reference_text.encode("utf-8")).hexdigest()
        if reference_text is not None
        else None
    )
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

        def make_generator(chat_client: object) -> object:
            return dependencies.IndependentChoiceCandidateGenerator(
                chat_client,
                max_tokens_per_choice=profile["max_tokens_per_choice"],
                temperature=profile["temperature"],
            )

        generator = make_generator(client)
        bar_generators = None
        if config.concurrent_bar_generation:
            # LocalChatModelClient allows one active request, so the second bar
            # gets its own client; the first bar reuses the primary one.
            second_client = dependencies.LocalChatModelClient(client_config)
            _register_close(resources, second_client)
            bar_generators = (generator, make_generator(second_client))
        analyzer = dependencies.CmuProsodyAnalyzer()
        planner = dependencies.ChunkCandidatePlanner(
            generator,
            analyzer,
            dependencies.ScoreWeights(),
            **({"bar_generators": bar_generators} if bar_generators is not None else {}),
        )

        if config.moss_serving_backend == "inprocess":
            synthesizer = _load_inprocess_moss_synthesizer(config)
            _register_close(resources, synthesizer)
            moss_probe: Mapping[str, object] = {
                "ready": True,
                "status": "resident",
                "identity": "PersistentMossSynthesizer",
                "version": _package_version("transformers"),
                "model": config.moss_model,
            }
        elif config.moss_serving_backend == "mlx":
            synthesizer = _load_mlx_moss_synthesizer(
                config,
                reference_audio_sha256=reference_audio_sha256,
            )
            _register_close(resources, synthesizer)
            moss_probe = synthesizer.probe()
        else:
            synthesizer = _load_sglang_moss_synthesizer(
                config,
                reference_audio_sha256=reference_audio_sha256,
                reference_text=reference_text,
                reference_text_sha256=reference_text_sha256,
            )
            _register_close(resources, synthesizer)
            moss_probe = synthesizer.probe()
        warmup_request = _warmup_render_request()
        warmup_started = time.perf_counter()
        with tempfile.TemporaryDirectory(
            prefix=".streammuse-rap-warmup-", dir=config.artifact_root
        ) as temporary_directory:
            warmup_wav = Path(temporary_directory) / "moss-warmup.wav"
            warmup_execution = SynthesisExecutionContext.from_timeout(
                min(300.0, config.moss_request_timeout_s),
                correlation_id="streammuse-rap-startup-warmup",
            )
            moss_warmup = synthesizer.synthesize(
                warmup_request,
                warmup_wav,
                execution=warmup_execution,
            )
            warmup_execution.checkpoint()
            from streammuse.infrastructure.rap.moss_tts import (
                MossInvalidOutput,
                read_valid_mono_wav,
            )

            try:
                read_valid_mono_wav(
                    warmup_wav,
                    expected_sample_rate_hz=24_000,
                    maximum_frame_count=30 * 24_000,
                )
            except MossInvalidOutput as exc:
                raise RuntimeError(
                    "MOSS startup warmup returned invalid audio"
                ) from exc
            if config.aligner_cache is not None:
                _configure_aligner_cache(config.aligner_cache)
            aligner_device = _resolve_torch_device(config.aligner_device)
            aligner = dependencies.MmsForcedAligner.load(device=aligner_device)
            _register_close(resources, aligner)
            aligner_warmup = aligner.warmup(warmup_wav, warmup_request.text)
        warmup_time_ms = max(0.0, (time.perf_counter() - warmup_started) * 1000.0)
        rubberband_health = _probe_rubberband()

        model_revision = str(getattr(moss_warmup, "model_revision", "unknown"))
        if model_revision == "unknown":
            raise RuntimeError("MOSS startup did not resolve an exact model revision")
        if (
            config.moss_model_revision is not None
            and model_revision != config.moss_model_revision
        ):
            raise RuntimeError("MOSS model revision does not match the configured pin")
        result_reference_hash = str(
            getattr(
                moss_warmup,
                "reference_voice_sha256",
                reference_audio_sha256,
            )
        )
        if result_reference_hash != reference_audio_sha256:
            raise RuntimeError("MOSS warmup used an unexpected reference audio")

        producer_manifest = _build_producer_manifest(
            config,
            model_revision=model_revision,
            reference_audio_sha256=reference_audio_sha256,
            reference_text_sha256=reference_text_sha256,
            aligner_identity=str(aligner_warmup["aligner"]),
            aligner_version=str(aligner_warmup["version"]),
            aligner_device=aligner_device,
        )
        namespace_root = initialize_producer_namespace(
            config.artifact_root,
            producer_manifest,
        )
        _probe_artifact_namespace(namespace_root)

        renderer = dependencies.MossAlignedPhraseRenderer(
            synthesizer=synthesizer,
            aligner=aligner,
            rubberband_version=str(rubberband_health["version"]),
            warp_policy=config.moss_warp_policy,
        )
        _register_close(resources, renderer)
        orchestrator = dependencies.RapChunkOrchestrator(
            planner,
            renderer,
            workspace_root=namespace_root,
        )
        endpoint = _moss_endpoint_health(config)
        health = {
            "protocol_version": "remote-rap-chunk/v1",
            "schema_version": REMOTE_CHUNK_SCHEMA_VERSION,
            "supported_schema_versions": _SUPPORTED_SCHEMA_VERSIONS_TEXT,
            "ready": True,
            "state": "ready",
            "backend": config.moss_serving_backend,
            "producer_fingerprint": producer_manifest.fingerprint,
            "vllm": dict(vllm_health),
            "moss": {
                "ready": True,
                "status": "warmed",
                "state": "ready",
                "backend": config.moss_serving_backend,
                "identity": str(moss_probe.get("identity", "unknown")),
                "version": str(moss_probe.get("version", "unknown")),
                "server_version": str(moss_probe.get("version", "unknown")),
                "model": _public_model_identity(config.moss_model),
                "model_revision": (
                    model_revision
                    if config.moss_serving_backend == "inprocess"
                    else str(moss_probe.get("model_revision", "unknown"))
                ),
                "reference_audio_sha256": reference_audio_sha256,
                "reference_text_sha256": reference_text_sha256 or "unavailable",
                "producer_fingerprint": producer_manifest.fingerprint,
                "warmup_time_ms": warmup_time_ms,
                "warmup": "complete",
                **endpoint,
            },
            "aligner": {
                "ready": True,
                "status": "warmed",
                "identity": str(aligner_warmup["aligner"]),
                "version": str(aligner_warmup["version"]),
                "device": aligner_device,
                "warmup": "complete",
            },
            "rubberband": dict(rubberband_health),
            "candidate_profile": config.candidate_profile,
            "candidate_generation": (
                "concurrent_bars" if config.concurrent_bar_generation else "serial_bars"
            ),
            "warmup": {"ready": True, "status": "complete"},
        }
        return _WorkerComposition(
            orchestrator,
            health,
            producer_manifest,
            resources,
        )
    except BaseException:
        resources.close()
        raise


def _validate_composition_inputs(
    config: RapRenderServerConfig,
) -> tuple[bytes, str | None]:
    if config.moss_serving_backend not in _MOSS_SERVING_BACKENDS:
        raise ValueError("unsupported MOSS serving backend")
    if (
        not math.isfinite(config.moss_request_timeout_s)
        or config.moss_request_timeout_s <= 0
        or not math.isfinite(config.moss_cancellation_grace_s)
        or config.moss_cancellation_grace_s <= 0
    ):
        raise ValueError("MOSS timeout configuration must be finite and positive")
    try:
        reference_audio = config.moss_reference_wav.read_bytes()
    except OSError as exc:
        raise ValueError("unable to read MOSS reference WAV") from exc
    if not reference_audio or len(reference_audio) > 64 * 1024 * 1024:
        raise ValueError("MOSS reference WAV size is invalid")
    from streammuse.infrastructure.rap.moss_tts import (
        MossInvalidOutput,
        read_valid_mono_wav_bytes,
    )

    try:
        reference_rate, reference_samples = read_valid_mono_wav_bytes(
            reference_audio
        )
    except MossInvalidOutput as exc:
        raise ValueError("MOSS reference WAV is invalid") from exc
    if reference_samples.shape[0] / reference_rate > 120.0:
        raise ValueError("MOSS reference WAV exceeds 120 seconds")

    mlx_fields = (config.moss_mlx_url, config.moss_mlx_runtime)
    if config.moss_serving_backend != "mlx" and any(
        value is not None for value in mlx_fields
    ):
        raise ValueError("only the mlx backend accepts MLX configuration")
    if config.moss_serving_backend == "mlx":
        if any(value is None for value in mlx_fields) or config.moss_model_revision is None:
            raise ValueError("mlx backend is missing pinned startup configuration")
        _validate_origin_url(config.moss_mlx_url, "MLX MOSS URL", loopback_only=True)
        _validate_pinned_identity(config.moss_model_revision, "MOSS model revision")
        if not config.moss_reference_wav.is_absolute():
            raise ValueError("mlx backend requires an absolute MOSS reference WAV path")
        sglang_fields = (
            config.moss_sglang_url,
            config.moss_reference_text_file,
            config.moss_sglang_reference_uri,
            config.moss_sglang_reference_sha256,
            config.moss_runtime_version,
            config.moss_runtime_revision,
            config.moss_sglang_version,
            config.moss_sglang_revision,
            config.moss_runtime_environment_sha256,
            config.moss_runtime_patch_sha256,
            config.moss_runtime_config,
            config.moss_runtime_config_sha256,
        )
        if any(value is not None for value in sglang_fields):
            raise ValueError("mlx backend received SGLang-only configuration")
        return reference_audio, None

    if config.moss_serving_backend == "inprocess":
        conflicts = (
            config.moss_sglang_url,
            config.moss_reference_text_file,
            config.moss_sglang_reference_uri,
            config.moss_sglang_reference_sha256,
            config.moss_runtime_version,
            config.moss_runtime_revision,
            config.moss_sglang_version,
            config.moss_sglang_revision,
            config.moss_runtime_environment_sha256,
            config.moss_runtime_patch_sha256,
            config.moss_runtime_config,
            config.moss_runtime_config_sha256,
        )
        if any(value is not None for value in conflicts):
            raise ValueError("inprocess backend received SGLang-only configuration")
        if not config.moss_device or config.moss_device == "external":
            raise ValueError("inprocess backend requires a MOSS device")
        return reference_audio, None

    required = (
        config.moss_sglang_url,
        config.moss_reference_text_file,
        config.moss_sglang_reference_uri,
        config.moss_sglang_reference_sha256,
        config.moss_model_revision,
        config.moss_runtime_version,
        config.moss_runtime_revision,
        config.moss_sglang_version,
        config.moss_sglang_revision,
        config.moss_runtime_environment_sha256,
        config.moss_runtime_config,
        config.moss_runtime_config_sha256,
    )
    if any(value is None for value in required):
        raise ValueError("sglang-omni backend is missing pinned startup configuration")
    _validate_origin_url(
        config.moss_sglang_url,
        "SGLang MOSS URL",
        loopback_only=True,
    )
    _validate_service_reference_uri(config.moss_sglang_reference_uri)
    _validate_pinned_identity(config.moss_model_revision, "MOSS model revision")
    _validate_pinned_identity(config.moss_runtime_version, "SGLang runtime version")
    _validate_pinned_identity(config.moss_runtime_revision, "SGLang runtime revision")
    _validate_pinned_identity(config.moss_sglang_version, "SGLang version")
    _validate_pinned_identity(config.moss_sglang_revision, "SGLang revision")
    _validate_sha256(config.moss_runtime_config_sha256, "SGLang config hash")
    _validate_sha256(
        config.moss_runtime_environment_sha256,
        "SGLang environment hash",
    )
    _validate_sha256(
        config.moss_sglang_reference_sha256,
        "SGLang service reference hash",
    )
    if config.moss_runtime_patch_sha256 is not None:
        _validate_sha256(config.moss_runtime_patch_sha256, "SGLang runtime patch hash")
    reference_audio_sha256 = hashlib.sha256(reference_audio).hexdigest()
    if reference_audio_sha256 != config.moss_sglang_reference_sha256:
        raise ValueError(
            "host and service-visible MOSS reference WAV hashes do not match"
        )
    assert config.moss_runtime_config is not None
    if not config.moss_runtime_config.is_absolute():
        raise ValueError("SGLang runtime config path must be absolute")
    try:
        runtime_config_bytes = config.moss_runtime_config.read_bytes()
    except OSError as exc:
        raise ValueError("unable to read SGLang runtime config") from exc
    if not runtime_config_bytes or len(runtime_config_bytes) > 1024 * 1024:
        raise ValueError("SGLang runtime config size is invalid")
    if hashlib.sha256(runtime_config_bytes).hexdigest() != config.moss_runtime_config_sha256:
        raise ValueError("SGLang runtime config SHA-256 does not match")
    assert config.moss_reference_text_file is not None
    try:
        text_bytes = config.moss_reference_text_file.read_bytes()
    except OSError as exc:
        raise ValueError("unable to read MOSS reference transcript") from exc
    if not text_bytes or len(text_bytes) > 64 * 1024:
        raise ValueError("MOSS reference transcript size is invalid")
    try:
        reference_text = text_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("MOSS reference transcript must be UTF-8") from exc
    if reference_text != reference_text.strip():
        raise ValueError("MOSS reference transcript has surrounding whitespace")
    return reference_audio, reference_text


def _load_inprocess_moss_synthesizer(config: RapRenderServerConfig) -> object:
    from streammuse.infrastructure.rap.moss_tts import PersistentMossSynthesizer

    return PersistentMossSynthesizer.load(
        model_id=config.moss_model,
        device=_resolve_torch_device(config.moss_device),
        reference_wav=config.moss_reference_wav,
    )


def _resolve_torch_device(preference: str) -> str:
    """Resolve auto to cuda, then mps, then cpu; keep explicit devices such as cuda:1."""
    if preference.strip().lower() != "auto":
        return preference
    from streammuse.infrastructure.inference.runtime_device import resolve_device

    return resolve_device("auto")


def _load_mlx_moss_synthesizer(
    config: RapRenderServerConfig,
    *,
    reference_audio_sha256: str,
) -> object:
    from streammuse.infrastructure.rap.mlx_moss_tts import (
        MlxMossConfig,
        MlxMossSynthesizer,
    )

    assert config.moss_mlx_url is not None
    assert config.moss_model_revision is not None
    assert config.moss_mlx_runtime is not None
    client_config = MlxMossConfig(
        base_url=config.moss_mlx_url,
        model_id=config.moss_model,
        model_revision=config.moss_model_revision,
        # Same host and filesystem: the service reads the exact file hashed here.
        reference_audio_uri=config.moss_reference_wav.resolve().as_uri(),
        reference_audio_sha256=reference_audio_sha256,
        runtime_identity=config.moss_mlx_runtime,
        request_timeout_seconds=config.moss_request_timeout_s,
        cancellation_grace_seconds=config.moss_cancellation_grace_s,
    )
    return MlxMossSynthesizer(client_config)


def _load_sglang_moss_synthesizer(
    config: RapRenderServerConfig,
    *,
    reference_audio_sha256: str,
    reference_text: str | None,
    reference_text_sha256: str | None,
) -> object:
    from streammuse.infrastructure.rap.sglang_moss_tts import (
        SglangMossConfig,
        SglangMossSynthesizer,
    )

    assert config.moss_sglang_url is not None
    assert config.moss_model_revision is not None
    assert config.moss_sglang_reference_uri is not None
    assert reference_text is not None
    assert reference_text_sha256 is not None
    client_config = SglangMossConfig(
        base_url=config.moss_sglang_url,
        model_id=config.moss_model,
        model_revision=config.moss_model_revision,
        reference_audio_uri=config.moss_sglang_reference_uri,
        reference_audio_sha256=reference_audio_sha256,
        reference_text=reference_text,
        reference_text_sha256=reference_text_sha256,
        request_timeout_seconds=config.moss_request_timeout_s,
        cancellation_grace_seconds=config.moss_cancellation_grace_s,
        runtime_config_sha256=str(config.moss_runtime_config_sha256),
    )
    return SglangMossSynthesizer(client_config)


def _build_producer_manifest(
    config: RapRenderServerConfig,
    *,
    model_revision: str,
    reference_audio_sha256: str,
    reference_text_sha256: str | None,
    aligner_identity: str,
    aligner_version: str,
    aligner_device: str | None = None,
) -> ProducerManifestV1:
    from streammuse.infrastructure.rap.moss_generation import (
        DEFAULT_BASE_SEED,
        SEED_POLICY_VERSION,
        producer_generation_settings,
    )

    if config.moss_serving_backend == "mlx":
        from streammuse.infrastructure.rap.mlx_moss_tts import (
            MLX_MOSS_ADAPTER_REVISION,
            runtime_identity_sha256,
        )

        assert config.moss_mlx_runtime is not None
        backend_revision = MLX_MOSS_ADAPTER_REVISION
        runtime = {
            "identity": "MLX/mlx-audio",
            "version": str(config.moss_mlx_runtime["mlx_audio_version"]),
            "revision": str(config.moss_mlx_runtime["mlx_audio_commit"]),
            **{name: str(value) for name, value in config.moss_mlx_runtime.items()},
            "config_sha256": runtime_identity_sha256(config.moss_mlx_runtime),
        }
    elif config.moss_serving_backend == "sglang-omni":
        from streammuse.infrastructure.rap.sglang_moss_tts import (
            SGLANG_MOSS_ADAPTER_REVISION,
        )

        backend_revision = SGLANG_MOSS_ADAPTER_REVISION
        runtime = {
            "identity": "SGLang-Omni/SGLang",
            "version": str(config.moss_runtime_version),
            "revision": str(config.moss_runtime_revision),
            "sglang_version": str(config.moss_sglang_version),
            "sglang_revision": str(config.moss_sglang_revision),
            "environment_sha256": str(
                config.moss_runtime_environment_sha256
            ),
            "config_sha256": str(config.moss_runtime_config_sha256),
            # The pip lock cannot see a source patch, so it is named here; an
            # unpatched runtime keeps its original fingerprint.
            **(
                {"patch_sha256": config.moss_runtime_patch_sha256}
                if config.moss_runtime_patch_sha256 is not None
                else {}
            ),
        }
    else:
        from streammuse.infrastructure.rap.moss_tts import (
            INPROCESS_MOSS_ADAPTER_REVISION,
        )

        backend_revision = INPROCESS_MOSS_ADAPTER_REVISION
        runtime_base = {
            "identity": "Transformers/PyTorch",
            "version": _package_version("transformers"),
            "revision": _package_version("torch"),
        }
        runtime = {
            **runtime_base,
            "config_sha256": hashlib.sha256(
                json.dumps(
                    runtime_base,
                    ensure_ascii=True,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest(),
        }

    return ProducerManifestV1(
        backend=config.moss_serving_backend,
        backend_implementation_revision=backend_revision,
        streammuse_revision=_streammuse_revision(),
        model={
            "id": _public_model_identity(config.moss_model),
            "revision": model_revision,
        },
        runtime=runtime,
        generation={
            "seed_policy_version": SEED_POLICY_VERSION,
            "settings": producer_generation_settings(base_seed=DEFAULT_BASE_SEED),
        },
        reference={
            "audio_sha256": reference_audio_sha256,
            "text_sha256": reference_text_sha256,
        },
        alignment={
            "identity": aligner_identity,
            "version": aligner_version,
            "warp_policy": config.moss_warp_policy,
            **({"device": aligner_device} if aligner_device is not None else {}),
        },
        output={
            "sample_rate_hz": 24_000,
            "wire_audio_codec": config.wire_audio_codec,
            # Only Opus output depends on the level, so PCM namespaces keep
            # their fingerprint (and cache) across this setting.
            **(
                {"opus_compression_level": config.opus_compression_level}
                if config.wire_audio_codec == "opus"
                else {}
            ),
            "public_schema_version": _SUPPORTED_SCHEMA_VERSIONS_TEXT,
        },
    )


def _streammuse_revision() -> Mapping[str, object]:
    root = Path(__file__).resolve().parents[3]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            timeout=5.0,
        ).stdout.decode("ascii").strip()
        diff = subprocess.run(
            [
                "git",
                "diff",
                "--binary",
                "--no-ext-diff",
                "HEAD",
                "--",
                "src/streammuse",
                "scripts/rap_audio_backends",
            ],
            cwd=root,
            check=True,
            capture_output=True,
            timeout=10.0,
        ).stdout
        untracked_output = subprocess.run(
            [
                "git",
                "ls-files",
                "--others",
                "--exclude-standard",
                "--",
                "src/streammuse",
                "scripts/rap_audio_backends",
            ],
            cwd=root,
            check=True,
            capture_output=True,
            timeout=5.0,
        ).stdout.decode("utf-8")
        digest = hashlib.sha256(diff)
        for relative_name in sorted(untracked_output.splitlines()):
            relative = Path(relative_name)
            candidate = root / relative
            if candidate.is_file():
                digest.update(relative.as_posix().encode("utf-8"))
                digest.update(b"\0")
                digest.update(candidate.read_bytes())
        return {"commit": commit, "patch_sha256": digest.hexdigest()}
    except (OSError, subprocess.SubprocessError, UnicodeError):
        digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        return {
            "commit": f"package-{_package_version('streammuse')}",
            "patch_sha256": digest,
        }


def _probe_artifact_namespace(namespace: Path) -> None:
    path = namespace / f".startup-write-probe-{uuid.uuid4().hex}"
    try:
        _ArtifactStore._atomic_write(path, b"streammuse-artifact-probe\n")
        if path.read_bytes() != b"streammuse-artifact-probe\n":
            raise RuntimeError("artifact startup probe could not be verified")
    finally:
        if path.exists():
            _ArtifactStore._durably_unpublish(path)


def _moss_endpoint_health(config: RapRenderServerConfig) -> dict[str, object]:
    url = {
        "sglang-omni": config.moss_sglang_url,
        "mlx": config.moss_mlx_url,
    }.get(config.moss_serving_backend)
    if url is None:
        return {}
    parsed = urlsplit(url)
    return {
        "endpoint_host": parsed.hostname or "unknown",
        "endpoint_port": parsed.port or (443 if parsed.scheme == "https" else 80),
    }


def _package_version(distribution: str) -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(distribution)
    except PackageNotFoundError:
        return "unknown"


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
    from streammuse.infrastructure.rap.prosody import CmuProsodyAnalyzer

    return SimpleNamespace(
        LocalChatModelClient=LocalChatModelClient,
        LocalChatModelClientConfig=LocalChatModelClientConfig,
        IndependentChoiceCandidateGenerator=IndependentChoiceCandidateGenerator,
        CmuProsodyAnalyzer=CmuProsodyAnalyzer,
        ScoreWeights=ScoreWeights,
        ChunkCandidatePlanner=ChunkCandidatePlanner,
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
    from streammuse.domain.rap import AudioFormat, PcmAudio
    from streammuse.infrastructure.rap.time_stretch import (
        RubberBandTimeMapStretcher,
    )

    source_frames = 2_400
    target_frames = 2_520
    phase = np.arange(source_frames, dtype=np.float32) / 24_000.0
    samples = (0.1 * np.sin(2.0 * np.pi * 220.0 * phase)).astype(np.float32)
    source = PcmAudio(
        AudioFormat(sample_rate_hz=24_000, channels=1),
        source_frames,
        samples.tobytes(),
    )
    stretched = RubberBandTimeMapStretcher(timeout_seconds=5.0).stretch(
        source,
        target_frames,
        ((0, 0), (source_frames - 1, target_frames - 1)),
    )
    stretched_samples = np.frombuffer(stretched.data, dtype=np.float32)
    if (
        stretched.frame_count != target_frames
        or stretched.format != source.format
        or not np.isfinite(stretched_samples).all()
        or not np.any(stretched_samples)
    ):
        raise RuntimeError("Rubber Band R3 startup probe returned invalid audio")
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
