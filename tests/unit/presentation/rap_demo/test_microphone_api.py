"""HTTP contracts for independent push-to-talk transcription."""

from __future__ import annotations

import asyncio
import io
import struct
from threading import Event
import wave

from fastapi import FastAPI
from fastapi.testclient import TestClient
import httpx
import pytest

from streammuse.infrastructure.rap.utterance_input import (
    MAX_WAV_BYTES,
    FasterWhisperUtteranceTranscriber,
    UtteranceTranscript,
)
from streammuse.presentation.rap_demo.microphone_api import create_microphone_router


def wav_bytes(value=2000, seconds=0.1, rate=16000):
    output = io.BytesIO()
    with wave.open(output, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(rate)
        writer.writeframes(struct.pack("<h", value) * int(rate * seconds))
    return output.getvalue()


class Recognizer:
    def __init__(self, *, entered=None, release=None, failure=None):
        self.calls = 0
        self.entered = entered
        self.release = release
        self.failure = failure

    def status(self):
        return {"available": True, "model": "base.en", "state": "ready", "reason": None}

    def transcribe(self, audio):
        self.calls += 1
        if self.entered:
            self.entered.set()
        if self.release:
            assert self.release.wait(3)
        if self.failure:
            raise self.failure
        return UtteranceTranscript(
            text="the city after rain", duration_seconds=0.1,
            sample_rate=16000, transcription_ms=12.0, silence=False,
        )


def app_for(transcriber=None, **kwargs):
    app = FastAPI()
    app.include_router(create_microphone_router(transcriber=transcriber or Recognizer(), **kwargs))
    return app


def post(client, payload=None, event_id=None, **kwargs):
    headers = {"Content-Type": "audio/wav"}
    if event_id:
        headers["X-Utterance-ID"] = event_id
    headers.update(kwargs.pop("headers", {}))
    return client.post(
        "/api/microphone/transcribe", content=wav_bytes() if payload is None else payload,
        headers=headers, **kwargs,
    )


def test_status_and_transcript_return_without_confirmation_or_story_mutation():
    with TestClient(app_for()) as client:
        status = client.get("/api/microphone").json()
        assert status["available"] is True
        assert status["model"] == "base.en"
        assert status["max_duration_seconds"] == 15
        result = post(client, event_id="utterance-1")
        assert result.status_code == 200
        assert result.json() == {
            "event_id": "utterance-1", "text": "the city after rain",
            "transcription_ms": 12.0, "duration_seconds": 0.1,
            "sample_rate": 16000, "silence": False, "model": "base.en",
        }


def test_repeated_id_returns_same_event_without_duplicate_transcription():
    recognizer = Recognizer()
    with TestClient(app_for(recognizer)) as client:
        first = post(client, event_id="release-123")
        second = post(client, event_id="release-123")
        assert first.status_code == second.status_code == 200
        assert first.json() == second.json()
        assert recognizer.calls == 1
        conflict = post(client, wav_bytes(value=3000), event_id="release-123")
        assert conflict.status_code == 409
        assert recognizer.calls == 1


def test_generated_ids_are_unique_for_distinct_submissions():
    with TestClient(app_for()) as client:
        first = post(client)
        second = post(client)
        assert first.json()["event_id"] != second.json()["event_id"]


@pytest.mark.parametrize(
    "payload,headers,expected",
    [
        (b"", {}, 400), (b"not a WAV", {}, 400),
        (b"/etc/passwd", {}, 400), (wav_bytes()[:-2], {}, 400),
        (wav_bytes(), {"Content-Type": "application/json"}, 415),
        (wav_bytes(), {"X-Utterance-ID": "bad id/../../"}, 400),
        (wav_bytes(), {"Content-Length": "-1"}, 400),
        (wav_bytes(), {"Content-Length": str(MAX_WAV_BYTES + 1)}, 413),
        (b"x" * (MAX_WAV_BYTES + 1), {}, 413),
        (wav_bytes(seconds=15.1, rate=8000), {}, 413),
    ],
)
def test_rejects_invalid_upload_before_recognition_and_releases_slot(payload, headers, expected):
    recognizer = Recognizer()
    with TestClient(app_for(recognizer)) as client:
        response = post(client, payload, headers=headers)
        assert response.status_code == expected
        assert recognizer.calls == 0
        assert post(client).status_code == 200


def test_unavailable_and_disabled_endpoints_do_not_crash(monkeypatch):
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
    for enabled, state in [(True, "unavailable"), (False, "disabled")]:
        recognizer = FasterWhisperUtteranceTranscriber(enabled=enabled)
        with TestClient(app_for(recognizer)) as client:
            status = client.get("/api/microphone").json()
            assert status["available"] is False
            assert status["state"] == state
            assert post(client).status_code == 503


def test_silence_returns_empty_visible_result_without_loading_model():
    recognizer = FasterWhisperUtteranceTranscriber(
        whisper_model_factory=lambda **kwargs: pytest.fail("Silence loaded model")
    )
    with TestClient(app_for(recognizer)) as client:
        response = post(client, wav_bytes(value=0))
        assert response.status_code == 200
        assert response.json()["silence"] is True
        assert response.json()["text"] == ""


def test_slow_transcription_does_not_block_status_and_rejects_parallel_work():
    async def scenario():
        entered, release = Event(), Event()
        recognizer = Recognizer(entered=entered, release=release)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app_for(recognizer)), base_url="http://test"
        ) as client:
            first = asyncio.create_task(post(client, event_id="busy-1"))
            try:
                assert await asyncio.to_thread(entered.wait, 1)
                status = await asyncio.wait_for(client.get("/api/microphone"), 0.5)
                assert status.json()["state"] == "busy"
                assert (await post(client, event_id="busy-1")).status_code == 409
                busy = await post(client, event_id="busy-2")
                assert busy.status_code == 429
                assert busy.headers["Retry-After"] == "1"
            finally:
                release.set()
                assert (await first).status_code == 200
            assert recognizer.calls == 1

    asyncio.run(scenario())


def test_timeout_keeps_slot_busy_until_actual_worker_finishes_and_retry_reuses_result():
    async def scenario():
        entered, release = Event(), Event()
        recognizer = Recognizer(entered=entered, release=release)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app_for(recognizer, transcription_timeout_s=0.05)),
            base_url="http://test",
        ) as client:
            try:
                response = await post(client, event_id="timeout-1")
                assert entered.is_set()
                assert response.status_code == 504
                assert (await client.get("/api/microphone")).json()["state"] == "busy"
                assert (await post(client)).status_code == 429
            finally:
                release.set()
            for _ in range(100):
                if (await client.get("/api/microphone")).json()["state"] != "busy":
                    break
                await asyncio.sleep(0.01)
            retry = await post(client, event_id="timeout-1")
            assert retry.status_code == 200
            assert recognizer.calls == 1

    asyncio.run(scenario())


def test_inference_failure_is_sanitized_and_releases_slot():
    recognizer = Recognizer(failure=RuntimeError("secret file /tmp/private.wav"))
    with TestClient(app_for(recognizer)) as client:
        response = post(client)
        assert response.status_code == 500
        assert "secret" not in response.text
        assert (client.get("/api/microphone")).json()["state"] == "ready"


def test_streamed_upload_is_bounded_even_without_content_length():
    async def scenario():
        async def chunks():
            yield b"x" * 800000
            yield b"x" * 800000

        recognizer = Recognizer()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app_for(recognizer)), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/microphone/transcribe", content=chunks(), headers={"Content-Type": "audio/wav"}
            )
            assert response.status_code == 413
            assert recognizer.calls == 0

    asyncio.run(scenario())


def test_slow_upload_times_out_without_leaking_busy_slot():
    async def scenario():
        async def chunks():
            yield b"RIFF"
            await asyncio.sleep(1)

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app_for(upload_timeout_s=0.02)), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/microphone/transcribe", content=chunks(), headers={"Content-Type": "audio/wav"}
            )
            assert response.status_code == 408
            assert (await post(client)).status_code == 200

    asyncio.run(scenario())


def test_cancelled_request_keeps_worker_busy_and_caches_result():
    async def scenario():
        entered, release = Event(), Event()
        recognizer = Recognizer(entered=entered, release=release)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app_for(recognizer)), base_url="http://test"
        ) as client:
            first = asyncio.create_task(post(client, event_id="cancelled-1"))
            try:
                assert await asyncio.to_thread(entered.wait, 1)
                first.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await first
                assert (await post(client)).status_code == 429
            finally:
                release.set()
            for _ in range(100):
                if not (await client.get("/api/microphone")).json()["busy"]:
                    break
                await asyncio.sleep(0.01)
            assert (await post(client, event_id="cancelled-1")).status_code == 200
            assert recognizer.calls == 1

    asyncio.run(scenario())


def test_cancelled_upload_releases_slot():
    async def scenario():
        entered = asyncio.Event()

        async def chunks():
            yield b"RIFF"
            entered.set()
            await asyncio.sleep(10)

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app_for()), base_url="http://test"
        ) as client:
            first = asyncio.create_task(client.post(
                "/api/microphone/transcribe", content=chunks(), headers={"Content-Type": "audio/wav"}
            ))
            await asyncio.wait_for(entered.wait(), 1)
            first.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first
            assert (await post(client)).status_code == 200

    asyncio.run(scenario())


def test_result_cache_is_bounded_and_expiring(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(
        "streammuse.presentation.rap_demo.microphone_api.monotonic", lambda: now[0]
    )
    recognizer = Recognizer()
    with TestClient(app_for(recognizer, cache_size=1, cache_ttl_s=5)) as client:
        assert post(client, event_id="first").status_code == 200
        assert post(client, event_id="second").status_code == 200
        assert post(client, event_id="first").status_code == 200
        assert recognizer.calls == 3
        now[0] = 106.0
        assert post(client, event_id="first").status_code == 200
        assert recognizer.calls == 4


@pytest.mark.parametrize("kwargs", [
    {"transcription_timeout_s": 0}, {"upload_timeout_s": float("inf")},
    {"cache_ttl_s": -1}, {"cache_size": 0},
])
def test_rejects_unbounded_or_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        create_microphone_router(**kwargs)
