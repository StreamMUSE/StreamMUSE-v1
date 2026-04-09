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

    client = HttpInferenceClient(
        HttpInferenceClientConfig(
            generate_url="http://x/generate_accompaniment",
            timeout_s=3,
            model_name="lekai",
            inference_mode="sliding_window",
            generation_interval_ticks=4,
        )
    )
    acc, timing = client.generate_accompaniment(
        melody_events=[MusicalEvent(tick=0, pitch=60, event_type=EventType.NOTE_ON, velocity=100)],
        generation_start_tick=10,
        generation_length_frames=20,
        prompt_length_ticks=None,
    )
    assert posted["url"].endswith("/generate_accompaniment")
    assert posted["json"]["generation_start_tick"] == 10
    assert posted["json"]["generation_interval_ticks"] == 4
    assert posted["json"]["model_name"] == "lekai"
    assert posted["json"]["inference_mode"] == "sliding_window"
    assert "checkpoint_path" not in posted["json"]
    assert acc[0].pitch == 64
    assert timing.inference_end_time == 1.3


def test_http_inference_client_generate_includes_checkpoint_path(monkeypatch):
    posted = {}

    def fake_post(url, json=None, timeout=None):
        posted["json"] = json
        return _FakeResponse(
            {
                "accompaniment": [],
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

    client = HttpInferenceClient(
        HttpInferenceClientConfig(
            generate_url="http://x/generate_accompaniment",
            model_name="lekai",
            inference_mode="sliding_window",
            generation_interval_ticks=4,
            checkpoint_path="/tmp/lekai.ckpt",
        )
    )

    client.generate_accompaniment(
        melody_events=[MusicalEvent(tick=0, pitch=60, event_type=EventType.NOTE_ON, velocity=100)],
        generation_start_tick=10,
        generation_length_frames=20,
        prompt_length_ticks=None,
    )

    assert posted["json"]["checkpoint_path"] == "/tmp/lekai.ckpt"


def test_http_inference_client_inject_and_clear(monkeypatch):
    calls = []

    def fake_post(url, json=None, timeout=None):
        calls.append((url, json))
        if url.endswith("/clear_history"):
            return _FakeResponse(
                {
                    "success": True,
                    "message": "History cleared",
                    "melody_history": [{"type": "note_on", "pitch": 60, "tick": 0}],
                    "accompaniment_history": [{"type": "note_on", "pitch": 48, "tick": 0, "velocity": 80}],
                }
            )
        return _FakeResponse({"ok": True})

    monkeypatch.setattr(requests, "post", fake_post)

    client = HttpInferenceClient(HttpInferenceClientConfig(generate_url="http://x/generate_accompaniment"))
    client.inject_history([], [], injection_length_ticks=50)
    cleared = client.clear_history()

    assert calls[0][0].endswith("/inject_notes")
    assert calls[1][0].endswith("/clear_history")
    assert cleared["melody_history"][0]["pitch"] == 60
    assert cleared["accompaniment_history"][0]["pitch"] == 48

