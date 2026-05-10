from __future__ import annotations

import requests

from streammuse.domain.musical import EventType, MusicalEvent
from streammuse.infrastructure.inference.prompt_continuation_http_client import (
    PromptContinuationHttpClient,
    PromptContinuationHttpClientConfig,
    normalize_prompt_continuation_base_url,
)


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def _event(tick: int = 0, pitch: int = 60) -> MusicalEvent:
    return MusicalEvent(tick=tick, pitch=pitch, event_type=EventType.NOTE_ON, velocity=64)


def test_normalize_prompt_continuation_base_url_accepts_known_endpoints():
    assert normalize_prompt_continuation_base_url("http://x:8000") == "http://x:8000"
    assert normalize_prompt_continuation_base_url("http://x:8000/generate_accompaniment") == "http://x:8000"
    assert normalize_prompt_continuation_base_url("http://x:8000/prompt_continuation/playable") == "http://x:8000"


def test_prompt_continuation_http_client_start_append_status_playable(monkeypatch):
    calls = []

    def fake_post(url, json=None, timeout=None):
        calls.append(("POST", url, json, timeout))
        if url.endswith("/prompt_continuation/start"):
            return _FakeResponse({"phase": "prompt_running"})
        if url.endswith("/prompt_continuation/append_melody"):
            return _FakeResponse({"phase": "catchup_running"})
        if url.endswith("/clear_history"):
            return _FakeResponse({"success": True})
        raise AssertionError(url)

    def fake_get(url, timeout=None):
        calls.append(("GET", url, None, timeout))
        if url.endswith("/prompt_continuation/status"):
            return _FakeResponse({"phase": "ready", "is_playback_ready": True})
        if url.endswith("/prompt_continuation/playable"):
            return _FakeResponse(
                {
                    "accompaniment": [{"type": "note_on", "pitch": 48, "tick": 32}],
                    "status": {"phase": "ready"},
                }
            )
        raise AssertionError(url)

    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setattr(requests, "get", fake_get)

    client = PromptContinuationHttpClient(
        PromptContinuationHttpClientConfig(
            base_url="http://x:8000/generate_accompaniment",
            timeout_s=7.0,
            checkpoint_path="/tmp/ckpt.pt",
        )
    )

    assert client.base_url == "http://x:8000"
    assert client.clear_history()["success"] is True
    assert client.start(
        melody_events=[_event()],
        prompt_length_ticks=32,
        generation_interval_ticks=4,
        observed_until_tick=32,
    )["phase"] == "prompt_running"
    assert client.append_melody(melody_events=[_event(36)], observed_until_tick=36)["phase"] == "catchup_running"
    assert client.status()["is_playback_ready"] is True
    accompaniment, status = client.playable()

    assert accompaniment[0].pitch == 48
    assert status["phase"] == "ready"
    start_payload = calls[1][2]
    assert start_payload["model_name"] == "lekai_prompt_continuation"
    assert start_payload["checkpoint_path"] == "/tmp/ckpt.pt"
    assert start_payload["melody_notes"][0]["tick"] == 0
    assert calls[1][3] == 7.0
