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


def test_http_client_reset_payload_request_ids_and_metadata_consume(monkeypatch):
    calls = []

    def fake_post(url, json=None, timeout=None):
        calls.append((url, json, timeout))
        if url.endswith("/debug/reset_session"):
            return _FakeResponse(
                {
                    "success": True,
                    "session_id": "session-abc",
                    "session_epoch": 7,
                    "effective_seed": 31415,
                    "pending_boundary_generations": 0,
                }
            )
        request_id = json["request_id"]
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
                "generation_start_tick": json["generation_start_tick"],
                "metadata": {
                    "request_id": request_id,
                    "session_id": json["session_id"],
                    "session_epoch": json["session_epoch"],
                    "effective_seed": 31415,
                    "raw_tokens": [169],
                },
            }
        )

    monkeypatch.setattr(requests, "post", fake_post)
    client = HttpInferenceClient(
        HttpInferenceClientConfig(
            generate_url="http://x/generate_accompaniment",
            timeout_s=4.0,
            generation_interval_ticks=4,
            bpm=120,
            input_file="song.mid",
        )
    )

    reset = client.reset_session(seed=31415)
    assert reset["session_epoch"] == 7
    assert calls[0] == (
        "http://x/debug/reset_session",
        {"seed": 31415},
        4.0,
    )

    event = MusicalEvent(
        tick=4,
        pitch=60,
        event_type=EventType.NOTE_ON,
        velocity=91,
        channel=2,
        program=8,
    )
    client.generate_accompaniment(
        melody_events=[event],
        generation_start_tick=8,
        generation_length_frames=4,
    )
    payload = calls[1][1]
    assert payload["session_id"] == "session-abc"
    assert payload["session_epoch"] == 7
    assert payload["request_id"] == "session-abc-r000001"
    assert payload["bpm"] == 120
    assert payload["input_file"] == "song.mid"
    assert payload["melody_notes"] == [
        {
            "type": "note_on",
            "pitch": 60,
            "tick": 4,
            "velocity": 91,
            "channel": 2,
            "program": 8,
        }
    ]
    snapshot = client.last_response_metadata
    assert snapshot["raw_tokens"] == [169]
    snapshot["raw_tokens"].append(170)
    # Nested response structures are defensive; consume is destructive only for the slot.
    consumed = client.consume_last_response_metadata()
    assert consumed["request_id"] == "session-abc-r000001"
    assert consumed["raw_tokens"] == [169]
    consumed["raw_tokens"].append(171)
    assert client.consume_last_response_metadata() == {}

    client.set_next_request_id("rt-lifecycle:req_0002")
    client.generate_accompaniment(
        melody_events=[],
        generation_start_tick=12,
        generation_length_frames=4,
    )
    assert calls[2][1]["request_id"] == "rt-lifecycle:req_0002"

    client.generate_accompaniment(
        melody_events=[],
        generation_start_tick=16,
        generation_length_frames=4,
    )
    assert calls[3][1]["request_id"] == "session-abc-r000003"
