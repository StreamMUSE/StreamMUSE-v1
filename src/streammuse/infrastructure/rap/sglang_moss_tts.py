"""Bounded SGLang-Omni HTTP adapter for connected MOSS phrase synthesis."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from urllib.parse import urlsplit, urlunsplit

import httpx

from streammuse.application.rap.execution import (
    ExecutionStopped,
    SynthesisExecutionContext,
)
from streammuse.experiments.rap_audio_protocols.contracts import TwoBarRenderRequest
from streammuse.infrastructure.rap.moss_generation import (
    DEFAULT_BASE_SEED,
    DETERMINISTIC_SEED_CAVEAT,
    LANGUAGE,
    MAX_NEW_TOKENS,
    RAP_INSTRUCTION,
    STYLE_INSTRUCTION_CAVEAT,
    resolved_generation_settings,
    seed_for_attempt,
)
from streammuse.infrastructure.rap.moss_tts import (
    MossBackendUnavailable,
    MossInvalidOutput,
    MossPhraseResult,
    MossRequestRejected,
    MossServingMetadata,
    MossSynthesisFailed,
    read_valid_mono_wav,
    read_valid_mono_wav_bytes,
)


SGLANG_MOSS_ADAPTER_REVISION = "streammuse.sglang_moss_http.v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_WAV_MEDIA_TYPES = frozenset({"audio/wav", "audio/x-wav", "audio/wave"})
_MAX_REFERENCE_TEXT_BYTES = 64 * 1024
_MAX_REFERENCE_AUDIO_BYTES = 64 * 1024 * 1024
_MAX_REFERENCE_AUDIO_SECONDS = 120.0
_PROBE_RESPONSE_LIMIT = 256 * 1024


@dataclass(frozen=True)
class SglangMossConfig:
    base_url: str
    model_id: str
    model_revision: str
    reference_audio_uri: str
    reference_audio_sha256: str
    reference_text: str
    reference_text_sha256: str
    runtime_config_sha256: str
    request_timeout_seconds: float = 120.0
    connect_timeout_seconds: float = 5.0
    pool_timeout_seconds: float = 5.0
    cancellation_grace_seconds: float = 2.0
    response_byte_limit: int = 64 * 1024 * 1024
    maximum_audio_seconds: float = 30.0
    base_seed: int = DEFAULT_BASE_SEED

    def __post_init__(self) -> None:
        object.__setattr__(self, "base_url", _validated_base_url(self.base_url))
        _require_bounded_text(self.model_id, "SGLang MOSS model", maximum=512)
        _require_bounded_text(
            self.model_revision, "SGLang MOSS model revision", maximum=512
        )
        _require_pinned_identity(self.model_revision, "SGLang MOSS model revision")
        _validate_reference_uri(self.reference_audio_uri)
        _require_sha256(self.reference_audio_sha256, "reference audio")
        _require_bounded_text(
            self.reference_text,
            "SGLang MOSS reference text",
            maximum=_MAX_REFERENCE_TEXT_BYTES,
        )
        if len(self.reference_text.encode("utf-8")) > _MAX_REFERENCE_TEXT_BYTES:
            raise ValueError("SGLang MOSS reference text exceeds its byte limit")
        expected_text_hash = hashlib.sha256(
            self.reference_text.encode("utf-8")
        ).hexdigest()
        if self.reference_text_sha256 != expected_text_hash:
            raise ValueError("SGLang MOSS reference text SHA-256 does not match")
        _require_sha256(self.reference_text_sha256, "reference text")
        _require_sha256(self.runtime_config_sha256, "runtime config")
        for name in (
            "request_timeout_seconds",
            "connect_timeout_seconds",
            "pool_timeout_seconds",
            "cancellation_grace_seconds",
            "maximum_audio_seconds",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"SGLang MOSS {name} must be finite and positive")
        if (
            isinstance(self.response_byte_limit, bool)
            or not isinstance(self.response_byte_limit, int)
            or self.response_byte_limit < 1024
        ):
            raise ValueError("SGLang MOSS response byte limit is invalid")
        if (
            isinstance(self.base_seed, bool)
            or not isinstance(self.base_seed, int)
            or self.base_seed < 0
        ):
            raise ValueError("SGLang MOSS base seed must be non-negative")

    @classmethod
    def from_files(
        cls,
        *,
        base_url: str,
        model_id: str,
        model_revision: str,
        reference_audio_uri: str,
        reference_audio_file: str | Path,
        reference_text_file: str | Path,
        **options: object,
    ) -> "SglangMossConfig":
        try:
            audio_bytes = Path(reference_audio_file).read_bytes()
        except OSError as exc:
            raise ValueError("unable to read SGLang MOSS reference audio") from exc
        if not audio_bytes or len(audio_bytes) > _MAX_REFERENCE_AUDIO_BYTES:
            raise ValueError("SGLang MOSS reference audio size is invalid")
        try:
            sample_rate_hz, samples = read_valid_mono_wav_bytes(audio_bytes)
        except MossInvalidOutput as exc:
            raise ValueError("SGLang MOSS reference audio is invalid") from exc
        if samples.shape[0] / sample_rate_hz > _MAX_REFERENCE_AUDIO_SECONDS:
            raise ValueError("SGLang MOSS reference audio is too long")
        try:
            text_bytes = Path(reference_text_file).read_bytes()
        except OSError as exc:
            raise ValueError("unable to read SGLang MOSS reference text") from exc
        if not text_bytes or len(text_bytes) > _MAX_REFERENCE_TEXT_BYTES:
            raise ValueError("SGLang MOSS reference text size is invalid")
        try:
            reference_text = text_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("SGLang MOSS reference text must be UTF-8") from exc
        if reference_text != reference_text.strip():
            raise ValueError(
                "SGLang MOSS reference text must not have surrounding whitespace"
            )
        return cls(
            base_url=base_url,
            model_id=model_id,
            model_revision=model_revision,
            reference_audio_uri=reference_audio_uri,
            reference_audio_sha256=hashlib.sha256(audio_bytes).hexdigest(),
            reference_text=reference_text,
            reference_text_sha256=hashlib.sha256(text_bytes).hexdigest(),
            **options,
        )


class SglangMossSynthesizer:
    """Reuse one HTTP pool and expose the backend-neutral MOSS contract.

    Subclasses serving the same speech API from another runtime override the
    class attributes, ``_config_type``, ``_extra_payload`` and
    ``_validate_model_entry``.
    """

    _BACKEND = "sglang-omni"
    _LABEL = "SGLang MOSS"

    @classmethod
    def _config_type(cls) -> type:
        return SglangMossConfig

    def __init__(
        self,
        config: SglangMossConfig,
        *,
        client: httpx.Client | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if not isinstance(config, self._config_type()):
            raise ValueError(f"{self._LABEL} synthesizer requires validated config")
        self._config = config
        self._client = client or httpx.Client(
            base_url=config.base_url,
            follow_redirects=False,
            trust_env=False,
        )
        self._clock = clock
        self._state_lock = threading.Lock()
        self._closed = False
        self._server_identity = self._BACKEND
        self._server_version = "unknown"

    @property
    def config(self) -> SglangMossConfig:
        return self._config

    def probe(self) -> Mapping[str, object]:
        with self._state_lock:
            if self._closed:
                raise MossBackendUnavailable(f"{self._LABEL} synthesizer is closed")
        health = self._probe_get("/health", allow_empty=True)
        models = self._probe_get("/v1/models", allow_empty=False)
        model_entries = models.get("data") if isinstance(models, Mapping) else None
        if not isinstance(model_entries, list):
            raise MossBackendUnavailable(
                f"{self._LABEL} model list has an invalid data field"
            )
        matching_models = [
            item
            for item in model_entries
            if isinstance(item, Mapping) and item.get("id") == self._config.model_id
        ]
        if not matching_models:
            raise MossBackendUnavailable(
                f"configured model is absent from the {self._LABEL} model list"
            )
        if len(matching_models) != 1:
            raise MossBackendUnavailable(
                f"configured model appears more than once in the {self._LABEL} model list"
            )
        matching_model = matching_models[0]
        reported_revision = _reported_model_revision(matching_model)
        if (
            reported_revision != "unknown"
            and reported_revision != self._config.model_revision
        ):
            raise MossBackendUnavailable(
                f"{self._LABEL} model revision does not match the configured pin"
            )
        self._validate_model_entry(matching_model)
        return MappingProxyType(
            {
                "ready": True,
                "status": "serving",
                "identity": self._server_identity,
                "version": self._server_version,
                "model": self._config.model_id,
                "model_revision": reported_revision,
                "health_fields": len(health),
            }
        )

    def _validate_model_entry(self, model: Mapping[str, object]) -> None:
        """Hook for subclasses that pin more than the model revision."""

    def _extra_payload(self) -> dict[str, object]:
        return {"ref_text": self._config.reference_text}

    def warmup(
        self, *, execution: SynthesisExecutionContext | None = None
    ) -> Mapping[str, object]:
        if execution is not None:
            execution.checkpoint()
        return self.probe()

    def build_request_payload(self, request: TwoBarRenderRequest) -> dict[str, object]:
        if not isinstance(request, TwoBarRenderRequest):
            raise MossRequestRejected(
                f"{self._LABEL} synthesis requires a two-bar render request"
            )
        settings = resolved_generation_settings(
            request,
            base_seed=self._config.base_seed,
            attempt=1,
        )
        generation = settings["generation_kwargs"]
        if not isinstance(generation, Mapping):
            raise MossRequestRejected("resolved MOSS generation settings are invalid")
        return {
            "model": self._config.model_id,
            "input": request.text,
            "voice": "default",
            "response_format": "wav",
            "stream": False,
            "ref_audio": self._config.reference_audio_uri,
            **self._extra_payload(),
            "language": LANGUAGE,
            "instructions": RAP_INSTRUCTION,
            "token_count": settings["token_target"],
            "max_new_tokens": MAX_NEW_TOKENS,
            "audio_temperature": generation["audio_temperature"],
            "audio_top_p": generation["audio_top_p"],
            "audio_top_k": generation["audio_top_k"],
            "audio_repetition_penalty": generation["audio_repetition_penalty"],
            "seed": seed_for_attempt(
                base_seed=self._config.base_seed,
                request=request,
                attempt=1,
            ),
        }

    def synthesize(
        self,
        request: TwoBarRenderRequest,
        output_wav: Path,
        *,
        execution: SynthesisExecutionContext,
    ) -> MossPhraseResult:
        if not isinstance(execution, SynthesisExecutionContext):
            raise ValueError(f"{self._LABEL} synthesis requires an execution context")
        execution.checkpoint()
        with self._state_lock:
            if self._closed:
                raise MossBackendUnavailable(f"{self._LABEL} synthesizer is closed")

        payload = self.build_request_payload(request)
        output_path = Path(output_wav)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        partial_path = output_path.with_name(
            f".{output_path.stem}.{uuid.uuid4().hex}.source.partial.wav"
        )
        started = self._clock()
        headers_at = started
        first_body_at = started
        download_finished = started
        validation_finished = started
        response_bytes = 0
        response: httpx.Response | None = None
        remove_cancel_callback: Callable[[], None] | None = None
        try:
            timeout = self._request_timeout(execution)
            with self._client.stream(
                "POST",
                "/v1/audio/speech",
                json=payload,
                headers={"X-StreamMUSE-Correlation-ID": execution.correlation_id},
                timeout=timeout,
            ) as response:
                headers_at = self._clock()
                remove_cancel_callback = execution.add_cancel_callback(
                    lambda active_response=response: _close_cancelled_response(
                        active_response, execution
                    )
                )
                execution.checkpoint()
                self._validate_status(response)
                self._validate_response_headers(response)
                server = response.headers.get("server")
                if server:
                    self._server_version = _bounded_identity(server)
                with partial_path.open("xb") as output:
                    first_chunk = True
                    for chunk in response.iter_bytes():
                        execution.checkpoint()
                        if first_chunk:
                            first_body_at = self._clock()
                            first_chunk = False
                        response_bytes += len(chunk)
                        if response_bytes > self._config.response_byte_limit:
                            raise MossInvalidOutput(
                                f"{self._LABEL} response exceeds the configured byte limit"
                            )
                        output.write(chunk)
                        execution.checkpoint()
                    output.flush()
                    os.fsync(output.fileno())
                if first_chunk:
                    first_body_at = self._clock()
                download_finished = self._clock()
            response = None
            execution.checkpoint()
            sample_rate_hz, samples = read_valid_mono_wav(
                partial_path,
                expected_sample_rate_hz=24_000,
                maximum_frame_count=round(
                    self._config.maximum_audio_seconds * 24_000
                ),
            )
            validation_finished = self._clock()
            execution.checkpoint()
            os.replace(partial_path, output_path)
            _fsync_directory(output_path.parent)
        except ExecutionStopped:
            raise
        except (MossInvalidOutput, MossRequestRejected, MossBackendUnavailable):
            raise
        except httpx.TimeoutException as exc:
            _raise_execution_or_unavailable(
                execution,
                f"{self._LABEL} request timed out",
                exc,
            )
        except httpx.RequestError as exc:
            _raise_execution_or_unavailable(
                execution,
                f"{self._LABEL} request transport failed",
                exc,
            )
        except OSError as exc:
            raise MossSynthesisFailed("unable to commit SGLang MOSS output") from exc
        finally:
            if remove_cancel_callback is not None:
                remove_cancel_callback()
            if response is not None:
                response.close()
            partial_path.unlink(missing_ok=True)

        total_ms = max(0.0, (validation_finished - started) * 1000.0)
        settings = resolved_generation_settings(
            request,
            base_seed=self._config.base_seed,
            attempt=1,
        )
        return MossPhraseResult(
            output_wav=output_path,
            model_id=self._config.model_id,
            model_revision=self._config.model_revision,
            reference_voice_sha256=self._config.reference_audio_sha256,
            source_wav_sha256=_file_sha256(output_path),
            sample_rate_hz=sample_rate_hz,
            frame_count=int(samples.shape[0]),
            generation_time_ms=total_ms,
            resolved_generation_settings=settings,
            warnings=(STYLE_INSTRUCTION_CAVEAT, DETERMINISTIC_SEED_CAVEAT),
            serving_metadata=MossServingMetadata(
                backend=self._BACKEND,
                correlation_id=execution.correlation_id,
                service_request_ms=total_ms,
                http_response_headers_ms=max(0.0, (headers_at - started) * 1000.0),
                http_first_body_byte_ms=max(
                    0.0, (first_body_at - started) * 1000.0
                ),
                response_download_ms=max(
                    0.0, (download_finished - headers_at) * 1000.0
                ),
                response_validation_ms=max(
                    0.0, (validation_finished - download_finished) * 1000.0
                ),
                response_bytes=response_bytes,
                server_identity=self._server_identity,
                server_version=self._server_version,
                model_revision=self._config.model_revision,
                reference_audio_sha256=self._config.reference_audio_sha256,
                reference_text_sha256=self._config.reference_text_sha256,
                config_sha256=self._config.runtime_config_sha256,
                streaming=False,
                resolved_generation_settings=settings,
            ),
        )

    def close(self) -> None:
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
        self._client.close()

    def _probe_get(self, path: str, *, allow_empty: bool) -> Mapping[str, object]:
        timeout = httpx.Timeout(
            timeout=self._config.connect_timeout_seconds,
            connect=self._config.connect_timeout_seconds,
            pool=self._config.pool_timeout_seconds,
        )
        try:
            stream = self._client.stream("GET", path, timeout=timeout)
        except httpx.TimeoutException as exc:
            raise MossBackendUnavailable(f"{self._LABEL} probe timed out") from exc
        except httpx.RequestError as exc:
            raise MossBackendUnavailable(f"{self._LABEL} probe transport failed") from exc
        try:
            with stream as response:
                if response.status_code != 200:
                    raise MossBackendUnavailable(
                        f"{self._LABEL} probe returned an error"
                    )
                _validate_content_length(
                    response.headers.get("content-length"),
                    limit=_PROBE_RESPONSE_LIMIT,
                    label=f"{self._LABEL} probe response",
                    error_type=MossBackendUnavailable,
                )
                content = bytearray()
                for chunk in response.iter_bytes():
                    if len(content) + len(chunk) > _PROBE_RESPONSE_LIMIT:
                        raise MossBackendUnavailable(
                            f"{self._LABEL} probe response is too large"
                        )
                    content.extend(chunk)
                server = response.headers.get("server")
                if server:
                    self._server_version = _bounded_identity(server)
            if not content and allow_empty:
                return MappingProxyType({})
            try:
                value = json.loads(content)
            except (UnicodeDecodeError, ValueError) as exc:
                raise MossBackendUnavailable(
                    f"{self._LABEL} probe returned invalid JSON"
                ) from exc
            if not isinstance(value, Mapping):
                raise MossBackendUnavailable(
                    f"{self._LABEL} probe returned an invalid object"
                )
            return MappingProxyType(dict(value))
        except httpx.TimeoutException as exc:
            raise MossBackendUnavailable(f"{self._LABEL} probe timed out") from exc
        except httpx.RequestError as exc:
            raise MossBackendUnavailable(f"{self._LABEL} probe transport failed") from exc

    def _request_timeout(self, execution: SynthesisExecutionContext) -> httpx.Timeout:
        remaining = execution.remaining_seconds()
        if remaining <= 0:
            execution.checkpoint()
        request_cap = min(float(self._config.request_timeout_seconds), remaining)
        return httpx.Timeout(
            timeout=request_cap,
            connect=min(float(self._config.connect_timeout_seconds), request_cap),
            read=request_cap,
            write=request_cap,
            pool=min(float(self._config.pool_timeout_seconds), request_cap),
        )

    def _validate_status(self, response: httpx.Response) -> None:
        if response.status_code == 200:
            return
        if 400 <= response.status_code < 500 and response.status_code != 429:
            raise MossRequestRejected(
                f"{self._LABEL} rejected the request ({response.status_code})"
            )
        raise MossBackendUnavailable(
            f"{self._LABEL} service is unavailable ({response.status_code})"
        )

    def _validate_response_headers(self, response: httpx.Response) -> None:
        media_type = response.headers.get("content-type", "").partition(";")[0]
        if media_type.strip().lower() not in _WAV_MEDIA_TYPES:
            raise MossInvalidOutput(f"{self._LABEL} response media type is not WAV")
        _validate_content_length(
            response.headers.get("content-length"),
            limit=self._config.response_byte_limit,
            label=f"{self._LABEL} response",
            error_type=MossInvalidOutput,
        )


def _validated_base_url(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("SGLang MOSS base URL is invalid")
    parsed = urlsplit(value)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.hostname not in {"127.0.0.1", "::1", "localhost"}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
    ):
        raise ValueError("SGLang MOSS base URL must be an origin without credentials")
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("SGLang MOSS base URL port is invalid") from exc
    if port is not None and not 1 <= port <= 65535:
        raise ValueError("SGLang MOSS base URL port is invalid")
    netloc = f"[{parsed.hostname}]" if ":" in parsed.hostname else parsed.hostname
    if port is not None:
        netloc = f"{netloc}:{port}"
    return urlunsplit((parsed.scheme, netloc, "", "", ""))


def _validate_reference_uri(value: object) -> None:
    if not isinstance(value, str) or not value or len(value) > 2048:
        raise ValueError("SGLang MOSS reference audio URI is invalid")
    parsed = urlsplit(value)
    if parsed.scheme != "file":
        raise ValueError("SGLang MOSS reference audio URI must use file://")
    if (
        parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("SGLang MOSS reference audio URI contains forbidden data")
    if parsed.netloc not in {"", "localhost"} or not parsed.path.startswith("/"):
        raise ValueError("SGLang MOSS file reference URI must be absolute and local")


def _require_bounded_text(value: object, name: str, *, maximum: int) -> None:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > maximum
        or any(ord(character) < 32 and character not in "\n\r\t" for character in value)
    ):
        raise ValueError(f"{name} is invalid")


def _require_sha256(value: object, name: str) -> None:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"SGLang MOSS {name} SHA-256 is invalid")


def _require_pinned_identity(value: str, name: str) -> None:
    if value.strip().lower() in {
        "main",
        "master",
        "latest",
        "unknown",
        "unavailable",
    }:
        raise ValueError(f"{name} must be an immutable pin")


def _bounded_identity(value: str) -> str:
    text = " ".join(value.split())[:256]
    unsafe_path = (
        text.startswith(("/", "\\", "~/"))
        or bool(re.match(r"^[A-Za-z]:[\\/]", text))
    )
    return text if text and "://" not in text and not unsafe_path else "unknown"


def _reported_model_revision(model: Mapping[str, object]) -> str:
    for name in ("revision", "model_revision"):
        value = model.get(name)
        if isinstance(value, str):
            bounded = _bounded_identity(value)
            if bounded != "unknown":
                return bounded
    return "unknown"


def _raise_execution_or_unavailable(
    execution: SynthesisExecutionContext,
    message: str,
    cause: Exception,
) -> None:
    execution.checkpoint()
    raise MossBackendUnavailable(message) from cause


def _close_cancelled_response(
    response: httpx.Response,
    execution: SynthesisExecutionContext,
) -> None:
    execution.record_cancellation_outcome("transport_closed_abort_unconfirmed")
    response.close()


def _validate_content_length(
    value: str | None,
    *,
    limit: int,
    label: str,
    error_type: type[RuntimeError],
) -> None:
    if value is None:
        return
    if not value.isascii() or not value.isdecimal():
        raise error_type(f"{label} Content-Length is invalid")
    length = int(value)
    if length < 0 or length > limit:
        raise error_type(f"{label} exceeds the configured byte limit")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
