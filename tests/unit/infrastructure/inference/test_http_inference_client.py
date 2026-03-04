import requests

from streammuse.domain.musical import EventType, MusicalEvent
from streammuse.infrastructure.inference.http_client import HttpInferenceClient, HttpInferenceClientConfig


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def test_http_inference_client_generate_posts_and_parses(monkeypatch):
    posted = {}

    def fake_post(url, json=None, timeout=None):
        posted["url"] = url
        posted["json"] = json
        posted["timeout"] = timeout
        return _FakeResponse(
            {
                "accompaniment": [{"type": "note_on", "pitch": 64, "tick": 10}],
                "timings": {
                    "request_arrival_time": 1.0,
                    "response_output_time": 2.0,
                    "preprocess_start_time": 1.1,
                    "inference_start_time": 1.2,
                    "inference_end_time": 1.3,
                    "postprocess_start_time": 1.4,
                },
                "generation_start_tick": 10,
            }
        )

    monkeypatch.setattr(requests, "post", fake_post)

    client = HttpInferenceClient(HttpInferenceClientConfig(generate_url="http://x/generate_accompaniment", timeout_s=3))
    acc, timing = client.generate_accompaniment(
        melody_events=[MusicalEvent(tick=0, pitch=60, event_type=EventType.NOTE_ON, velocity=100)],
        generation_start_tick=10,
        generation_length_frames=20,
        prompt_length_ticks=None,
    )
    assert posted["url"].endswith("/generate_accompaniment")
    assert posted["json"]["generation_start_tick"] == 10
    assert acc[0].pitch == 64
    assert timing.inference_end_time == 1.3


def test_http_inference_client_inject_and_clear(monkeypatch):
    calls = []

    def fake_post(url, json=None, timeout=None):
        calls.append((url, json))
        return _FakeResponse({"ok": True})

    monkeypatch.setattr(requests, "post", fake_post)

    client = HttpInferenceClient(HttpInferenceClientConfig(generate_url="http://x/generate_accompaniment"))
    client.inject_history([], [], injection_length_ticks=50)
    client.clear_history()

    assert calls[0][0].endswith("/inject_notes")
    assert calls[1][0].endswith("/clear_history")

