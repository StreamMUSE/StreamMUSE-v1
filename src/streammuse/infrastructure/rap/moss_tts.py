"""Persistent MOSS phrase synthesis for the realtime H200 worker."""

from __future__ import annotations

import hashlib
import importlib
import io
import os
import re
import tempfile
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol

import numpy as np
from scipy.io import wavfile

from streammuse.experiments.rap_audio_protocols.contracts import (
    SyllableTarget,
    TwoBarRenderRequest,
)
from streammuse.application.rap.execution import (
    ExecutionStopped,
    SynthesisExecutionContext,
)
from streammuse.infrastructure.rap.moss_generation import (
    DEFAULT_BASE_SEED,
    DETERMINISTIC_SEED_CAVEAT,
    STYLE_INSTRUCTION_CAVEAT,
    resolved_generation_settings,
    seed_for_attempt,
)


INPROCESS_MOSS_ADAPTER_REVISION = "streammuse.inprocess_moss.v2"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CANCELLATION_OUTCOMES = frozenset(
    {
        "not_requested",
        "cancel_requested",
        "cancel_requested_but_not_interruptible",
        "transport_closed_abort_unconfirmed",
        "upstream_abort_confirmed",
    }
)
_DEADLINE_OUTCOMES = frozenset({"within_deadline", "deadline_exceeded"})
_RECOVERY_OUTCOMES = frozenset(
    {"not_required", "abort_unconfirmed", "upstream_released", "restart_required"}
)


RuntimeLoader = Callable[..., object]
PhraseGenerator = Callable[..., None]
SeedResolver = Callable[..., int]
TorchSeeder = Callable[..., None]


class MossSynthesisError(RuntimeError):
    """Base class for stable MOSS adapter failures."""


class MossSynthesisFailed(MossSynthesisError):
    """Raised when MOSS does not produce a valid connected-phrase WAV."""


class MossBackendUnavailable(MossSynthesisFailed):
    """Raised when a selected MOSS backend cannot serve the request."""


class MossRequestRejected(MossSynthesisFailed):
    """Raised when a backend rejects the configured synthesis contract."""


class MossInvalidOutput(MossSynthesisFailed):
    """Raised when a backend returns audio outside the source WAV contract."""


def _freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


@dataclass(frozen=True)
class MossServingMetadata:
    """Backend-neutral private timings and lifecycle evidence for one synthesis."""

    backend: str = "inprocess"
    correlation_id: str = "unavailable"
    service_request_ms: float = 0.0
    http_response_headers_ms: float = 0.0
    http_first_body_byte_ms: float = 0.0
    response_download_ms: float = 0.0
    response_validation_ms: float = 0.0
    response_bytes: int = 0
    server_identity: str = "inprocess"
    server_version: str = "unknown"
    model_revision: str = "unknown"
    reference_audio_sha256: str = "unavailable"
    reference_text_sha256: str = "unavailable"
    config_sha256: str = "unavailable"
    cancellation_outcome: str = "not_requested"
    deadline_outcome: str = "within_deadline"
    upstream_abort_confirmed: bool = False
    cancellation_grace_exceeded: bool = False
    recovery_outcome: str = "not_required"
    streaming: bool = False
    internal_stage_timings: Mapping[str, object] = field(
        default_factory=lambda: {
            "status": "unavailable",
            "unavailable_reason": "backend_did_not_report_stage_timings",
        }
    )
    resolved_generation_settings: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.backend not in {"inprocess", "sglang-omni", "mlx"}:
            raise ValueError("unsupported MOSS serving backend")
        for name in (
            "correlation_id",
            "server_identity",
            "server_version",
            "model_revision",
            "reference_audio_sha256",
            "reference_text_sha256",
            "config_sha256",
            "cancellation_outcome",
            "deadline_outcome",
            "recovery_outcome",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or not value
                or len(value) > 256
                or any(ord(character) < 32 for character in value)
            ):
                raise ValueError(f"invalid MOSS serving metadata {name}")
        for name in (
            "service_request_ms",
            "http_response_headers_ms",
            "http_first_body_byte_ms",
            "response_download_ms",
            "response_validation_ms",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(value)
                or value < 0
            ):
                raise ValueError(f"invalid MOSS serving metadata {name}")
        if (
            isinstance(self.response_bytes, bool)
            or not isinstance(self.response_bytes, int)
            or self.response_bytes < 0
        ):
            raise ValueError("invalid MOSS serving metadata response_bytes")
        if type(self.streaming) is not bool:
            raise ValueError("invalid MOSS serving metadata streaming flag")
        if self.cancellation_outcome not in _CANCELLATION_OUTCOMES:
            raise ValueError("invalid MOSS serving metadata cancellation_outcome")
        if self.deadline_outcome not in _DEADLINE_OUTCOMES:
            raise ValueError("invalid MOSS serving metadata deadline_outcome")
        if self.recovery_outcome not in _RECOVERY_OUTCOMES:
            raise ValueError("invalid MOSS serving metadata recovery_outcome")
        if type(self.upstream_abort_confirmed) is not bool:
            raise ValueError("invalid MOSS serving metadata abort confirmation")
        if type(self.cancellation_grace_exceeded) is not bool:
            raise ValueError("invalid MOSS serving metadata cancellation grace flag")
        expected_recovery = (
            "not_required"
            if self.cancellation_outcome == "not_requested"
            else "restart_required"
            if self.cancellation_grace_exceeded
            else "upstream_released"
            if self.upstream_abort_confirmed
            else "abort_unconfirmed"
        )
        if self.upstream_abort_confirmed != (
            self.cancellation_outcome == "upstream_abort_confirmed"
        ) or self.recovery_outcome != expected_recovery:
            raise ValueError("inconsistent MOSS serving cancellation evidence")
        for name in (
            "reference_audio_sha256",
            "reference_text_sha256",
            "config_sha256",
        ):
            value = getattr(self, name)
            if value not in {"unknown", "unavailable"} and not _SHA256.fullmatch(
                value
            ):
                raise ValueError(f"invalid MOSS serving metadata {name}")
        object.__setattr__(
            self, "internal_stage_timings", _freeze(self.internal_stage_timings)
        )
        object.__setattr__(
            self,
            "resolved_generation_settings",
            _freeze(self.resolved_generation_settings),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "moss_backend": self.backend,
            "correlation_id": self.correlation_id,
            "moss_service_request_ms": self.service_request_ms,
            "moss_http_response_headers_ms": self.http_response_headers_ms,
            "moss_http_first_body_byte_ms": self.http_first_body_byte_ms,
            "moss_response_download_ms": self.response_download_ms,
            "moss_response_validation_ms": self.response_validation_ms,
            "moss_response_bytes": self.response_bytes,
            "server_identity": self.server_identity,
            "server_version": self.server_version,
            "model_revision": self.model_revision,
            "reference_audio_sha256": self.reference_audio_sha256,
            "reference_text_sha256": self.reference_text_sha256,
            "config_sha256": self.config_sha256,
            "cancellation_outcome": self.cancellation_outcome,
            "deadline_outcome": self.deadline_outcome,
            "upstream_abort_confirmed": self.upstream_abort_confirmed,
            "cancellation_grace_exceeded": self.cancellation_grace_exceeded,
            "recovery_outcome": self.recovery_outcome,
            "streaming": self.streaming,
            "internal_stage_timings": _json_value(self.internal_stage_timings),
            "resolved_generation_settings": _json_value(
                self.resolved_generation_settings
            ),
        }


@dataclass(frozen=True)
class MossPhraseResult:
    """Validated raw MOSS phrase plus immutable reproduction metadata."""

    output_wav: Path
    model_id: str
    model_revision: str
    reference_voice_sha256: str
    source_wav_sha256: str
    sample_rate_hz: int
    frame_count: int
    generation_time_ms: float
    resolved_generation_settings: Mapping[str, object]
    warnings: tuple[str, ...]
    serving_metadata: MossServingMetadata = field(default_factory=MossServingMetadata)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "resolved_generation_settings",
            _freeze(self.resolved_generation_settings),
        )
        if not isinstance(self.serving_metadata, MossServingMetadata):
            raise ValueError("MOSS serving metadata is invalid")

    @property
    def duration_seconds(self) -> float:
        return self.frame_count / self.sample_rate_hz


class MossSynthesizer(Protocol):
    """Stable adapter contract used by the connected-phrase renderer."""

    def synthesize(
        self,
        request: TwoBarRenderRequest,
        output_wav: Path,
        *,
        execution: SynthesisExecutionContext,
    ) -> MossPhraseResult: ...

    def warmup(
        self, *, execution: SynthesisExecutionContext | None = None
    ) -> Mapping[str, object]: ...

    def close(self) -> None: ...


class PersistentMossSynthesizer:
    """Reuse one MOSS runtime across complete-phrase synthesis calls."""

    def __init__(
        self,
        *,
        runtime: object,
        model_id: str,
        device: str,
        reference_wav: Path,
        reference_voice_sha256: str,
        phrase_generator: PhraseGenerator,
        seed_resolver: SeedResolver,
        torch_seeder: TorchSeeder,
        torch_module: object,
        base_seed: int,
        backend_module: object,
        clock: Callable[[], float],
    ) -> None:
        self._runtime = runtime
        self._model_id = model_id
        self._device = device
        self._reference_wav = reference_wav
        self._reference_voice_sha256 = reference_voice_sha256
        self._phrase_generator = phrase_generator
        self._seed_resolver = seed_resolver
        self._torch_seeder = torch_seeder
        self._torch = torch_module
        self._base_seed = base_seed
        self._clock = clock
        self._closed = False
        self._close_lock = threading.Lock()

    @classmethod
    def load(
        cls,
        *,
        model_id: str,
        device: str,
        reference_wav: Path,
        runtime_loader: RuntimeLoader | None = None,
        phrase_generator: PhraseGenerator | None = None,
        base_seed: int | None = None,
        clock: Callable[[], float] = time.perf_counter,
        **runtime_options: Any,
    ) -> "PersistentMossSynthesizer":
        reference_path = Path(reference_wav)
        try:
            reference_bytes = reference_path.read_bytes()
        except OSError as exc:
            raise MossSynthesisFailed(
                f"unable to read MOSS reference voice: {reference_path}"
            ) from exc
        if not reference_bytes:
            raise MossSynthesisFailed("MOSS reference voice must not be empty")

        backend = importlib.import_module("scripts.rap_audio_backends.moss_backend")
        load_runtime = runtime_loader or getattr(backend, "create_runtime")
        generate = phrase_generator or getattr(backend, "_generate_chunk")
        resolve_seed = seed_for_attempt
        seed_torch = getattr(backend, "_seed_torch_best_effort")
        resolved_base_seed = (
            DEFAULT_BASE_SEED
            if base_seed is None
            else base_seed
        )
        if isinstance(resolved_base_seed, bool) or not isinstance(
            resolved_base_seed, int
        ):
            raise MossSynthesisFailed("MOSS base seed must be an integer")
        runtime = load_runtime(
            model_id=model_id,
            device=device,
            **runtime_options,
        )
        try:
            torch_module = getattr(runtime, "torch_module", None)
            if torch_module is None or not callable(
                getattr(torch_module, "manual_seed", None)
            ):
                raise MossSynthesisFailed(
                    "MOSS runtime must expose torch.manual_seed for request seeding"
                )
            return cls(
                runtime=runtime,
                model_id=model_id,
                device=device,
                reference_wav=reference_path,
                reference_voice_sha256=hashlib.sha256(reference_bytes).hexdigest(),
                phrase_generator=generate,
                seed_resolver=resolve_seed,
                torch_seeder=seed_torch,
                torch_module=torch_module,
                base_seed=resolved_base_seed,
                backend_module=backend,
                clock=clock,
            )
        except BaseException:
            close = getattr(runtime, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
            raise

    def warmup(
        self, *, execution: SynthesisExecutionContext | None = None
    ) -> Mapping[str, object]:
        """Execute one disposable phrase generation to warm model kernels."""
        started = self._clock()
        active_execution = execution or SynthesisExecutionContext.from_timeout(
            300.0, correlation_id="streammuse-moss-warmup"
        )
        active_execution.checkpoint()
        warmup = getattr(self._runtime, "warmup", None)
        if callable(warmup):
            warmup()
        with tempfile.TemporaryDirectory(prefix="streammuse-moss-warmup-") as temp_dir:
            result = self.synthesize(
                _warmup_request(),
                Path(temp_dir) / "moss-warmup.wav",
                execution=active_execution,
            )
        return MappingProxyType(
            {
                "model_id": self._model_id,
                "device": self._device,
                "generated": True,
                "sample_rate_hz": result.sample_rate_hz,
                "frame_count": result.frame_count,
                "source_wav_sha256": result.source_wav_sha256,
                "warmup_time_ms": max(0.0, (self._clock() - started) * 1000.0),
            }
        )

    def synthesize(
        self,
        request: TwoBarRenderRequest,
        output_wav: Path,
        *,
        execution: SynthesisExecutionContext | None = None,
    ) -> MossPhraseResult:
        if not isinstance(request, TwoBarRenderRequest):
            raise MossSynthesisFailed(
                "MOSS synthesis requires a two-bar render request"
            )

        active_execution = execution or SynthesisExecutionContext.from_timeout(
            3600.0,
            correlation_id=f"moss-{request.chunk_index}",
        )
        active_execution.checkpoint()
        with self._close_lock:
            if self._closed:
                raise MossBackendUnavailable("MOSS synthesizer is closed")

        output_path = Path(output_wav)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        partial_path = output_path.with_name(
            f".{output_path.stem}.partial{output_path.suffix or '.wav'}"
        )
        output_path.unlink(missing_ok=True)
        partial_path.unlink(missing_ok=True)
        started = self._clock()
        attempt = 1
        try:
            active_execution.checkpoint()
            seed = self._seed_resolver(
                base_seed=self._base_seed,
                request=request,
                attempt=attempt,
            )
            self._torch_seeder(self._torch, seed=seed)
            active_execution.checkpoint()
            with active_execution.uninterruptible():
                self._phrase_generator(
                    request=request,
                    output_path=partial_path,
                    reference_wav=self._reference_wav,
                    runtime=self._runtime,
                )
            # In-process model.generate is cooperative only at its boundaries.
            active_execution.checkpoint()
            sample_rate_hz, samples = read_valid_mono_wav(partial_path)
            frame_count = int(samples.shape[0])
            active_execution.checkpoint()
            os.replace(partial_path, output_path)
        except BaseException as exc:
            for path in (partial_path, output_path):
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass
            if not isinstance(exc, Exception):
                raise
            if isinstance(exc, ExecutionStopped):
                raise
            if isinstance(exc, MossSynthesisFailed):
                raise
            raise MossSynthesisFailed(f"MOSS phrase synthesis failed: {exc}") from exc

        generation_time_ms = max(0.0, (self._clock() - started) * 1000.0)
        settings = resolved_generation_settings(
            request, base_seed=self._base_seed, attempt=attempt
        )
        warnings = (STYLE_INSTRUCTION_CAVEAT, DETERMINISTIC_SEED_CAVEAT)
        model_revision = _model_revision(self._runtime)
        return MossPhraseResult(
            output_wav=output_path,
            model_id=self._model_id,
            model_revision=model_revision,
            reference_voice_sha256=self._reference_voice_sha256,
            source_wav_sha256=_file_sha256(output_path),
            sample_rate_hz=sample_rate_hz,
            frame_count=frame_count,
            generation_time_ms=generation_time_ms,
            resolved_generation_settings=settings,
            warnings=warnings,
            serving_metadata=MossServingMetadata(
                backend="inprocess",
                correlation_id=active_execution.correlation_id,
                service_request_ms=generation_time_ms,
                response_bytes=output_path.stat().st_size,
                server_identity="PersistentMossSynthesizer",
                model_revision=model_revision,
                reference_audio_sha256=self._reference_voice_sha256,
                resolved_generation_settings=settings,
            ),
        )

    def close(self) -> None:
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
        close = getattr(self._runtime, "close", None)
        if callable(close):
            close()


def read_valid_mono_wav(
    path: Path,
    *,
    expected_sample_rate_hz: int | None = None,
    maximum_frame_count: int | None = None,
) -> tuple[int, np.ndarray]:
    try:
        sample_rate_hz, samples = wavfile.read(path)
    except Exception as exc:
        raise MossInvalidOutput("MOSS output is not a readable WAV") from exc
    return _validate_mono_wav(
        sample_rate_hz,
        samples,
        expected_sample_rate_hz=expected_sample_rate_hz,
        maximum_frame_count=maximum_frame_count,
    )


def read_valid_mono_wav_bytes(
    data: bytes,
    *,
    expected_sample_rate_hz: int | None = None,
    maximum_frame_count: int | None = None,
) -> tuple[int, np.ndarray]:
    if not isinstance(data, bytes) or not data:
        raise MossInvalidOutput("MOSS output is not a readable WAV")
    try:
        sample_rate_hz, samples = wavfile.read(io.BytesIO(data))
    except Exception as exc:
        raise MossInvalidOutput("MOSS output is not a readable WAV") from exc
    return _validate_mono_wav(
        sample_rate_hz,
        samples,
        expected_sample_rate_hz=expected_sample_rate_hz,
        maximum_frame_count=maximum_frame_count,
    )


def _validate_mono_wav(
    sample_rate_hz: int,
    samples: object,
    *,
    expected_sample_rate_hz: int | None,
    maximum_frame_count: int | None,
) -> tuple[int, np.ndarray]:
    array = np.asarray(samples)
    if sample_rate_hz <= 0:
        raise MossInvalidOutput("MOSS output sample rate must be positive")
    if (
        expected_sample_rate_hz is not None
        and sample_rate_hz != expected_sample_rate_hz
    ):
        raise MossInvalidOutput(
            f"MOSS output sample rate must be {expected_sample_rate_hz} Hz"
        )
    if array.ndim == 2 and array.shape[1] == 1:
        array = array[:, 0]
    if array.ndim != 1:
        raise MossInvalidOutput("MOSS output must be mono")
    if array.size == 0:
        raise MossInvalidOutput("MOSS output must not be empty")
    if maximum_frame_count is not None and array.size > maximum_frame_count:
        raise MossInvalidOutput("MOSS output exceeds the configured duration limit")
    if not np.isfinite(array).all():
        raise MossInvalidOutput("MOSS output must contain only finite samples")
    if float(np.max(np.abs(array.astype(np.float64)))) == 0.0:
        raise MossInvalidOutput("MOSS output must not be silent")
    return int(sample_rate_hz), array


# Compatibility for callers predating the public shared validator.
_read_valid_mono_wav = read_valid_mono_wav


def _json_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _model_revision(runtime: object) -> str:
    model = getattr(runtime, "model", None)
    config = getattr(model, "config", None)
    for owner in (config, model, runtime):
        revision = getattr(owner, "_commit_hash", None)
        if isinstance(revision, str) and revision:
            return revision
    return "unknown"


def _warmup_request() -> TwoBarRenderRequest:
    return TwoBarRenderRequest(
        song_id="streammuse-moss-warmup",
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
