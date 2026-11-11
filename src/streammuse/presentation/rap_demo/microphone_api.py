"""Independent browser push-to-talk API; mounting never records or routes a story."""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from dataclasses import asdict, dataclass
from hashlib import sha256
import math
import re
from threading import Lock
from time import monotonic
from typing import Any, Protocol
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from starlette.requests import ClientDisconnect

from streammuse.infrastructure.rap.utterance_input import (
    MAX_DURATION_SECONDS,
    MAX_SAMPLE_RATE,
    MAX_WAV_BYTES,
    MIN_SAMPLE_RATE,
    FasterWhisperUtteranceTranscriber,
    InvalidUtterance,
    TranscriberUnavailable,
    UtteranceTranscript,
    validate_wav,
)


class UtteranceTranscriber(Protocol):
    def status(self) -> dict[str, Any]: ...

    def transcribe(self, payload: bytes) -> UtteranceTranscript: ...


@dataclass(frozen=True)
class _Completed:
    fingerprint: str
    status_code: int
    body: dict[str, Any]
    completed_at: float


class _MicrophoneService:
    def __init__(self, transcriber: UtteranceTranscriber, *, cache_size: int, cache_ttl_s: float):
        self.transcriber = transcriber
        self.cache_size = cache_size
        self.cache_ttl_s = cache_ttl_s
        self._lock = Lock()
        self._active_id: str | None = None
        self._completed: OrderedDict[str, _Completed] = OrderedDict()

    def status(self) -> dict[str, Any]:
        result = self.transcriber.status()
        with self._lock:
            busy = self._active_id is not None
        return {
            **result,
            "transcriber_state": result["state"],
            "state": "busy" if busy else result["state"],
            "busy": busy,
            "max_duration_seconds": MAX_DURATION_SECONDS,
            "max_bytes": MAX_WAV_BYTES,
            "min_sample_rate": MIN_SAMPLE_RATE,
            "max_sample_rate": MAX_SAMPLE_RATE,
            "channels": 1,
            "sample_width_bytes": 2,
        }

    def reserve(self, event_id: str) -> None:
        with self._lock:
            if self._active_id is not None:
                duplicate = self._active_id == event_id
                raise HTTPException(
                    409 if duplicate else 429,
                    detail={"code": "utterance_in_progress" if duplicate else "microphone_busy", "event_id": event_id},
                    headers={"Retry-After": "1"},
                )
            self._active_id = event_id

    def release(self) -> None:
        with self._lock:
            self._active_id = None

    def cached(self, event_id: str, fingerprint: str) -> _Completed | None:
        with self._lock:
            now = monotonic()
            while self._completed:
                oldest = next(iter(self._completed.values()))
                if now - oldest.completed_at < self.cache_ttl_s:
                    break
                self._completed.popitem(last=False)
            result = self._completed.get(event_id)
            if result is not None and result.fingerprint != fingerprint:
                raise HTTPException(409, detail={"code": "utterance_id_conflict", "event_id": event_id})
            return result

    def recognize(self, event_id: str, fingerprint: str, payload: bytes) -> _Completed:
        # This finally runs in the CPU worker, not the cancellable request task.
        try:
            try:
                transcript = self.transcriber.transcribe(payload)
                status_code = 200
                body = {"event_id": event_id, **asdict(transcript), "model": self.transcriber.status()["model"]}
            except TranscriberUnavailable:
                status_code = 503
                body = {"detail": {"code": "microphone_unavailable", "event_id": event_id}}
            except Exception:
                status_code = 500
                body = {"detail": {"code": "transcription_failed", "event_id": event_id}}
            result = _Completed(fingerprint, status_code, body, monotonic())
            with self._lock:
                self._completed[event_id] = result
                while len(self._completed) > self.cache_size:
                    self._completed.popitem(last=False)
            return result
        finally:
            self.release()


async def _read_bounded_body(request: Request) -> bytes:
    chunks = bytearray()
    async for chunk in request.stream():
        if len(chunks) + len(chunk) > MAX_WAV_BYTES:
            raise InvalidUtterance("WAV exceeds the upload size limit", status_code=413)
        chunks.extend(chunk)
    return bytes(chunks)


def create_microphone_router(
    *,
    transcriber: UtteranceTranscriber | None = None,
    transcription_timeout_s: float = 30.0,
    upload_timeout_s: float = 10.0,
    cache_size: int = 64,
    cache_ttl_s: float = 300.0,
) -> APIRouter:
    """Mount once per process; no playback, microphone device, or story side effects.

    X-Utterance-ID is an optional retry key, echoed as event_id. Reuse it for
    retries of the same WAV; successful responses are cached for five minutes.
    """
    for value in (transcription_timeout_s, upload_timeout_s, cache_ttl_s):
        if not math.isfinite(value) or value <= 0:
            raise ValueError("Microphone timeouts and cache TTL must be positive and finite")
    if not isinstance(cache_size, int) or cache_size < 1:
        raise ValueError("Microphone result cache must have at least one entry")
    service = _MicrophoneService(
        transcriber if transcriber is not None else FasterWhisperUtteranceTranscriber(),
        cache_size=cache_size, cache_ttl_s=cache_ttl_s,
    )
    router = APIRouter(prefix="/api/microphone", tags=["microphone"])

    @router.get("")
    async def status() -> dict[str, Any]:
        return service.status()

    @router.post("/transcribe")
    async def transcribe(request: Request) -> JSONResponse:
        event_id = request.headers.get("x-utterance-id", str(uuid4()))
        if re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", event_id) is None:
            raise HTTPException(400, detail={"code": "invalid_utterance_id"})
        content_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
        if content_type not in {"audio/wav", "audio/wave", "audio/x-wav"}:
            raise HTTPException(415, detail={"code": "expected_audio_wav", "event_id": event_id})
        length = request.headers.get("content-length")
        if length is not None:
            if not length.isascii() or not length.isdigit():
                raise HTTPException(400, detail={"code": "invalid_content_length", "event_id": event_id})
            if len(length) > 10 or int(length) > MAX_WAV_BYTES:
                raise HTTPException(413, detail={"code": "upload_too_large", "event_id": event_id})
        if not service.status()["available"]:
            raise HTTPException(503, detail={"code": "microphone_unavailable", "event_id": event_id})
        service.reserve(event_id)
        submitted = False
        try:
            payload = await asyncio.wait_for(_read_bounded_body(request), timeout=upload_timeout_s)
            validate_wav(payload)
            fingerprint = sha256(payload).hexdigest()
            cached = service.cached(event_id, fingerprint)
            if cached is not None:
                return JSONResponse(cached.body, status_code=cached.status_code)
            worker = asyncio.get_running_loop().run_in_executor(
                None, service.recognize, event_id, fingerprint, payload
            )
            submitted = True
            try:
                result = await asyncio.wait_for(asyncio.shield(worker), timeout=transcription_timeout_s)
            except asyncio.TimeoutError:
                raise HTTPException(
                    504, detail={"code": "transcription_timeout", "event_id": event_id},
                    headers={"Retry-After": "1"},
                ) from None
            return JSONResponse(result.body, status_code=result.status_code)
        except InvalidUtterance as error:
            raise HTTPException(
                error.status_code,
                detail={"code": "invalid_utterance", "message": str(error), "event_id": event_id},
            ) from None
        except asyncio.TimeoutError:
            raise HTTPException(408, detail={"code": "upload_timeout", "event_id": event_id}) from None
        except ClientDisconnect:
            raise HTTPException(400, detail={"code": "upload_disconnected", "event_id": event_id}) from None
        finally:
            if not submitted:
                service.release()

    return router
