from fastapi.testclient import TestClient
import pytest

from streammuse.infrastructure.inference.server_lekai import app


client = TestClient(app)


def _base_generate_payload() -> dict:
    return {
        "melody_notes": [
            {"type": "note_on", "pitch": 60, "tick": 0},
            {"type": "note_off", "pitch": 60, "tick": 4},
        ],
        "generation_start_tick": 8,
        "generation_length_frames": 20,
        "generation_interval_ticks": 4,
        "prompt_length_ticks": None,
        "inference_mode": "sliding_window",
        "model_name": "lekai",
        "checkpoint_path": None,
    }


def test_generate_accompaniment_success():
    resp = client.post("/generate_accompaniment", json=_base_generate_payload())
    assert resp.status_code == 200
    data = resp.json()
    assert "accompaniment" in data
    assert "timings" in data
    assert data["generation_start_tick"] == 8
    for key in [
        "request_arrival_time",
        "response_output_time",
        "preprocess_start_time",
        "inference_start_time",
        "inference_end_time",
        "postprocess_start_time",
    ]:
        assert key in data["timings"]


@pytest.mark.parametrize(
    ("generation_interval_ticks", "generation_length_frames", "expected_status"),
    [
        (3, 20, 200),
        (2, 17, 422),
    ],
)
def test_lekai_validates_generation_length_only(
    generation_interval_ticks,
    generation_length_frames,
    expected_status,
):
    payload = _base_generate_payload()
    payload["generation_interval_ticks"] = generation_interval_ticks
    payload["generation_length_frames"] = generation_length_frames
    resp = client.post("/generate_accompaniment", json=payload)
    assert resp.status_code == expected_status
    if expected_status == 422:
        assert "generation_length_frames" in resp.text
        assert "multiple of 4" in resp.text


def test_non_lekai_skips_multiple_of_4_validation():
    payload = _base_generate_payload()
    payload["model_name"] = "stanley"
    payload["generation_interval_ticks"] = 2
    payload["generation_length_frames"] = 10
    resp = client.post("/generate_accompaniment", json=payload)
    assert resp.status_code == 200


def test_inject_clear_and_status():
    inject_resp = client.post(
        "/inject_notes",
        json={
            "melody_notes": [{"type": "note_on", "pitch": 60, "tick": 0}],
            "accompaniment_notes": [{"type": "note_on", "pitch": 48, "tick": 0, "velocity": 80}],
            "injection_length_ticks": 16,
        },
    )
    assert inject_resp.status_code == 200
    status_resp = client.get("/injection_status")
    assert status_resp.status_code == 200
    assert status_resp.json()["is_injected"] is True

    clear_resp = client.post("/clear_history")
    assert clear_resp.status_code == 200
    status_resp2 = client.get("/injection_status")
    assert status_resp2.status_code == 200
    assert status_resp2.json()["is_injected"] is False


def test_runtime_info_contract():
    resp = client.get("/runtime_info")
    assert resp.status_code == 200
    data = resp.json()
    for key in [
        "mode",
        "has_real_model",
        "resolved_device",
        "resolved_dtype",
        "checkpoint_path",
        "checkpoint_format",
        "fallback_reason",
        "load_time_ms",
        "warmup_time_ms",
        "use_cache",
        "runtime_model_name",
        "runtime_inference_mode",
    ]:
        assert key in data
