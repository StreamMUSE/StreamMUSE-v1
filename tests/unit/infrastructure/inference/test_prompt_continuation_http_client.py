from __future__ import annotations

import requests
import pytest

from streammuse.domain.musical import EventType, MusicalEvent
from streammuse.infrastructure.inference.lekai_prompt_continuation.token_conversion import (
    event_representation_summary,
)
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
    assert normalize_prompt_continuation_base_url("http://x:8000/prompt_continuation/replay_audit") == "http://x:8000"
    assert normalize_prompt_continuation_base_url("http://x:8000/prompt_continuation/session/initialize") == "http://x:8000"


def test_prompt_continuation_http_client_initializes_generated_or_replay_session(
    monkeypatch,
):
    calls = []

    def fake_post(url, json=None, timeout=None):
        calls.append((url, json, timeout))
        return _FakeResponse({"success": True, "session_id": "session-1"})

    monkeypatch.setattr(requests, "post", fake_post)
    client = PromptContinuationHttpClient(
        PromptContinuationHttpClientConfig(base_url="http://x:8000", timeout_s=7.0)
    )

    assert client.initialize_session()["success"] is True
    assert client.initialize_session(prompt_seed=17, continuation_seed=23)[
        "session_id"
    ] == "session-1"
    assert calls == [
        ("http://x:8000/prompt_continuation/session/initialize", {}, 7.0),
        (
            "http://x:8000/prompt_continuation/session/initialize",
            {"prompt_seed": 17, "continuation_seed": 23},
            7.0,
        ),
    ]


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
        if url.endswith("/prompt_continuation/replay_audit"):
            return _FakeResponse(
                {
                    "schema_version": 1,
                    "runtime_info": {"seed_provenance_complete": True},
                    "prompt_generation_log": {"generated_tokens": [169]},
                    "continuation_generations": [{"raw_token_digest": "digest"}],
                }
            )
        if url.endswith("/prompt_continuation/playable"):
            events = [{"type": "note_on", "pitch": 48, "tick": 32, "velocity": 100}]
            return _FakeResponse(
                {
                    "accompaniment": events,
                    "status": {"phase": "ready"},
                    "representation": event_representation_summary(events),
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
            prompt_selection_mode="rule_s_if_else",
            prompt_batch_candidates=10,
            temperature=1.1,
            top_p=0.95,
            top_k=50,
            repetition_penalty=1.0,
        )
    )

    assert client.base_url == "http://x:8000"
    assert client.clear_history()["success"] is True
    assert client.start(
        melody_events=[_event()],
        prompt_length_ticks=32,
        generation_interval_ticks=4,
        observed_until_tick=32,
        bpm=96,
    )["phase"] == "prompt_running"
    assert client.append_melody(melody_events=[_event(36)], observed_until_tick=36)["phase"] == "catchup_running"
    assert client.status()["is_playback_ready"] is True
    audit = client.replay_audit()
    accompaniment, status = client.playable()

    assert accompaniment[0].pitch == 48
    assert audit["schema_version"] == 1
    assert audit["continuation_generations"][0]["raw_token_digest"] == "digest"
    assert status["phase"] == "ready"
    assert status["playable_representation_match"] is True
    assert status["server_playable_representation"]["digest"] == status["client_playable_representation"]["digest"]
    start_payload = calls[1][2]
    assert start_payload["model_name"] == "lekai_prompt_continuation"
    assert start_payload["checkpoint_path"] == "/tmp/ckpt.pt"
    assert start_payload["bpm"] == 96
    assert start_payload["prompt_selection_mode"] == "rule_s_if_else"
    assert start_payload["prompt_batch_candidates"] == 10
    assert start_payload["temperature"] == 1.1
    assert start_payload["top_p"] == 0.95
    assert start_payload["top_k"] == 50
    assert start_payload["repetition_penalty"] == 1.0
    assert start_payload["melody_notes"][0]["tick"] == 0
    assert calls[1][3] == 7.0


def test_prompt_continuation_http_client_omits_unset_generation_config(
    monkeypatch,
):
    calls = []

    def fake_post(url, json=None, timeout=None):
        calls.append(dict(json))
        return _FakeResponse({"phase": "prompt_running"})

    monkeypatch.setattr(requests, "post", fake_post)
    client = PromptContinuationHttpClient(
        PromptContinuationHttpClientConfig(base_url="http://x:8000")
    )

    client.start(
        melody_events=[],
        prompt_length_ticks=32,
        generation_interval_ticks=4,
        observed_until_tick=32,
    )

    for field_name in (
        "prompt_selection_mode",
        "prompt_batch_candidates",
        "temperature",
        "top_p",
        "top_k",
        "repetition_penalty",
    ):
        assert field_name not in calls[0]


def test_prompt_continuation_http_client_strict_representation_loop_rejects_mismatch(
    monkeypatch,
):
    def fake_get(url, timeout=None):
        if url.endswith("/prompt_continuation/playable"):
            return _FakeResponse(
                {
                    "accompaniment": [{"type": "note_on", "pitch": 48, "tick": 32, "velocity": 100}],
                    "status": {"phase": "ready"},
                    "representation": {"event_count": 1, "digest": "not-the-client-digest"},
                }
            )
        raise AssertionError(url)

    monkeypatch.setenv("LEKAI_PROMPT_CONTINUATION_STRICT_REPRESENTATION_LOOP", "1")
    monkeypatch.setattr(requests, "get", fake_get)

    client = PromptContinuationHttpClient(
        PromptContinuationHttpClientConfig(base_url="http://x:8000")
    )

    with pytest.raises(RuntimeError, match="representation mismatch"):
        client.playable()


def test_prompt_continuation_http_client_rejects_non_positive_bpm():
    client = PromptContinuationHttpClient(
        PromptContinuationHttpClientConfig(base_url="http://x:8000")
    )

    with pytest.raises(ValueError, match="bpm must be > 0"):
        client.start(
            melody_events=[],
            prompt_length_ticks=32,
            generation_interval_ticks=4,
            observed_until_tick=32,
            bpm=0,
        )
