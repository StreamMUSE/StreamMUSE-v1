from fastapi.testclient import TestClient
import pytest

from streammuse.infrastructure.inference import server_lekai
from streammuse.infrastructure.inference.lekai_http_backend import LekaiHttpBackend
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


def _prompt_status(*, effective_bpm=None) -> dict:
    return {
        "phase": "prompt_running",
        "is_running": True,
        "is_failed": False,
        "error": None,
        "melody_event_count": 1,
        "accompaniment_event_count": 0,
        "prompt_length_ticks": 32,
        "generation_interval_ticks": 4,
        "effective_bpm": effective_bpm,
        "continuation_calls": 0,
        "melody_history_beats": 8,
        "accompaniment_history_beats": 0,
        "playable_lookahead_beats": 0,
        "target_playable_accompaniment_beats": 9,
        "beats_needed_for_playback": 9,
        "is_history_aligned": False,
        "is_playback_ready": False,
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


def test_prompt_continuation_model_uses_lekai_length_validation():
    payload = _base_generate_payload()
    payload["model_name"] = "lekai_prompt_continuation"
    payload["generation_length_frames"] = 10

    response = client.post("/generate_accompaniment", json=payload)

    assert response.status_code == 422
    assert "multiple of 4" in response.text


def test_prompt_continuation_prompt_generation_log_contract():
    response = client.get("/prompt_continuation/prompt_generation_log")

    assert response.status_code == 200
    payload = response.json()
    assert "prompt_tokens" in payload
    assert "generated_tokens" in payload
    assert "generated_acc_beats" in payload


def test_prompt_continuation_replay_audit_endpoint_is_read_only(monkeypatch):
    expected = {
        "schema_version": 1,
        "trace_capture_complete": True,
        "trace_capture_reason": "complete",
        "runtime_info": {
            "prompt_sample_seed": 7,
            "continuation_sample_seed": 9,
            "session_id": "session-1",
            "session_epoch": 1,
            "seeded_session_active": True,
            "seed_provenance_complete": True,
            "seed_provenance_reason": "complete",
        },
        "prompt_generation_log": {
            "prompt_tokens": [257, 263],
            "generated_tokens": [257, 263, 169],
        },
        "continuation_generations": [
            {
                "request_id": "continuation-1",
                "raw_tokens": [258, 169],
                "raw_token_digest": "raw-digest",
                "part0_roll_digest": "roll-digest",
                "output_event_digest": "event-digest",
            }
        ],
    }

    class _AuditBackend:
        def __init__(self):
            self.calls = 0

        def replay_audit(self):
            self.calls += 1
            return expected

    audit_backend = _AuditBackend()
    monkeypatch.setattr(server_lekai, "prompt_continuation_backend", audit_backend)

    response = TestClient(app).get("/prompt_continuation/replay_audit")

    assert response.status_code == 200
    assert response.json() == expected
    assert audit_backend.calls == 1


def test_prompt_continuation_session_initialize_generates_fresh_server_seeds(
    monkeypatch,
):
    generated = []
    calls = []

    class _SessionBackend:
        def reset_session(self, *, prompt_seed, continuation_seed):
            calls.append((prompt_seed, continuation_seed))
            return {
                "success": True,
                "prompt_seed": prompt_seed,
                "continuation_effective_seed": continuation_seed,
                "session_id": "pc-session",
                "session_epoch": 3,
                "pending_boundary_generations": 0,
                "scheduler_phase": "idle",
                "scheduler_is_running": False,
            }

    monkeypatch.delenv("LEKAI_ENABLE_DEBUG_RESET", raising=False)
    monkeypatch.setattr(
        server_lekai,
        "_new_session_seed",
        lambda: generated.append(101) or 101,
    )
    monkeypatch.setattr(server_lekai, "prompt_continuation_backend", _SessionBackend())

    response = TestClient(app).post(
        "/prompt_continuation/session/initialize",
        json={},
    )

    assert response.status_code == 200
    assert generated == [101]
    assert calls == [(101, 101)]
    assert response.json() == {
        "success": True,
        "prompt_requested_seed": 101,
        "prompt_effective_seed": 101,
        "continuation_requested_seed": 101,
        "continuation_effective_seed": 101,
        "prompt_seed_source": "system",
        "continuation_seed_source": "system",
        "session_id": "pc-session",
        "session_epoch": 3,
        "pending_boundary_generations": 0,
        "scheduler_phase": "idle",
        "scheduler_is_running": False,
    }


def test_prompt_continuation_session_initialize_accepts_saved_replay_seeds(
    monkeypatch,
):
    class _SessionBackend:
        def reset_session(self, *, prompt_seed, continuation_seed):
            return {
                "success": True,
                "prompt_seed": prompt_seed,
                "continuation_effective_seed": continuation_seed,
                "session_id": "replay-session",
                "session_epoch": 4,
                "pending_boundary_generations": 0,
                "scheduler_phase": "idle",
                "scheduler_is_running": False,
            }

    monkeypatch.setattr(server_lekai, "prompt_continuation_backend", _SessionBackend())

    response = TestClient(app).post(
        "/prompt_continuation/session/initialize",
        json={"prompt_seed": 17, "continuation_seed": 23},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["prompt_requested_seed"] == 17
    assert payload["prompt_effective_seed"] == 17
    assert payload["continuation_requested_seed"] == 23
    assert payload["continuation_effective_seed"] == 23
    assert payload["prompt_seed_source"] == "requested"
    assert payload["continuation_seed_source"] == "requested"


@pytest.mark.parametrize(
    "payload",
    [
        {"prompt_seed": 17},
        {"continuation_seed": 23},
    ],
)
def test_prompt_continuation_session_initialize_rejects_partial_seed_pair(
    monkeypatch,
    payload,
):
    class _UnexpectedBackend:
        def reset_session(self, **kwargs):
            pytest.fail(f"partial seed pair reached backend: {kwargs}")

    monkeypatch.setattr(
        server_lekai,
        "prompt_continuation_backend",
        _UnexpectedBackend(),
    )

    response = TestClient(app).post(
        "/prompt_continuation/session/initialize",
        json=payload,
    )

    assert response.status_code == 422
    assert "must either both be supplied or both be omitted" in response.json()[
        "detail"
    ]


def test_prompt_continuation_poll_endpoints_contract():
    client.post("/clear_history")

    start_response = client.post(
        "/prompt_continuation/start",
        json={
            "melody_notes": [{"type": "note_on", "pitch": 60, "tick": 0}],
            "prompt_length_ticks": 32,
            "generation_interval_ticks": 4,
            "observed_until_tick": 32,
            "inference_mode": "sliding_window",
            "model_name": "lekai_prompt_continuation",
            "checkpoint_path": None,
        },
    )
    assert start_response.status_code == 200
    assert start_response.json()["prompt_length_ticks"] == 32

    append_response = client.post(
        "/prompt_continuation/append_melody",
        json={
            "melody_notes": [{"type": "note_on", "pitch": 62, "tick": 44}],
            "observed_until_tick": 44,
        },
    )
    assert append_response.status_code == 200

    status_response = client.get("/prompt_continuation/status")
    assert status_response.status_code == 200
    assert "beats_needed_for_playback" in status_response.json()

    playable_response = client.get("/prompt_continuation/playable")
    assert playable_response.status_code == 200
    playable = playable_response.json()
    assert "accompaniment" in playable
    assert "status" in playable
    assert playable["representation"]["event_count"] == len(playable["accompaniment"])
    assert playable["representation"]["digest"]


def test_prompt_continuation_start_rejects_wrong_model_name():
    response = client.post(
        "/prompt_continuation/start",
        json={
            "melody_notes": [],
            "prompt_length_ticks": 32,
            "generation_interval_ticks": 4,
            "model_name": "lekai",
        },
    )

    assert response.status_code == 422
    assert "lekai_prompt_continuation" in response.text


@pytest.mark.parametrize(("request_bpm", "expected_bpm"), [(96, 96), (None, None)])
def test_prompt_continuation_start_forwards_optional_bpm(
    monkeypatch,
    request_bpm,
    expected_bpm,
):
    calls = []

    class _Backend:
        def start_prompt_catchup(self, **kwargs):
            calls.append(dict(kwargs))
            return _prompt_status(effective_bpm=kwargs["bpm"])

    monkeypatch.setattr(server_lekai, "prompt_continuation_backend", _Backend())
    payload = {
        "melody_notes": [{"type": "note_on", "pitch": 60, "tick": 0}],
        "prompt_length_ticks": 32,
        "generation_interval_ticks": 4,
    }
    if request_bpm is not None:
        payload["bpm"] = request_bpm

    response = client.post("/prompt_continuation/start", json=payload)

    assert response.status_code == 200
    assert calls[0]["bpm"] == expected_bpm
    assert response.json()["effective_bpm"] == expected_bpm


def test_prompt_continuation_start_rejects_non_positive_bpm(monkeypatch):
    class _Backend:
        def start_prompt_catchup(self, **kwargs):
            raise AssertionError("invalid request must not reach backend")

    monkeypatch.setattr(server_lekai, "prompt_continuation_backend", _Backend())
    response = client.post(
        "/prompt_continuation/start",
        json={
            "melody_notes": [],
            "prompt_length_ticks": 32,
            "generation_interval_ticks": 4,
            "bpm": 0,
        },
    )

    assert response.status_code == 422


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
    clear_data = clear_resp.json()
    assert "melody_history" in clear_data
    assert "accompaniment_history" in clear_data
    assert len(clear_data["melody_history"]) >= 1
    assert len(clear_data["accompaniment_history"]) >= 1

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
        "generation_interval_ticks",
        "generation_length_frames",
        "prompt_length_ticks",
        "ticks_per_beat",
    ]:
        assert key in data


def test_reset_endpoint_rotates_epoch_and_stale_session_returns_409(monkeypatch):
    monkeypatch.setenv("LEKAI_ENABLE_DEBUG_RESET", "true")
    fresh_backend = LekaiHttpBackend()
    monkeypatch.setattr(server_lekai, "backend", fresh_backend)
    local_client = TestClient(app)

    first = local_client.post("/debug/reset_session", json={"seed": 101})
    assert first.status_code == 200
    first_session = first.json()
    assert first_session["session_epoch"] == 1
    assert first_session["effective_seed"] == 101

    payload = _base_generate_payload()
    payload.update(
        {
            "session_id": first_session["session_id"],
            "session_epoch": first_session["session_epoch"],
            "request_id": "first-r000001",
        }
    )
    generated = local_client.post("/generate_accompaniment", json=payload)
    assert generated.status_code == 200
    metadata = generated.json()["metadata"]
    assert metadata["request_id"] == "first-r000001"
    assert metadata["session_id"] == first_session["session_id"]
    assert metadata["effective_seed"] == 101

    second = local_client.post("/debug/reset_session", json={"seed": 202})
    assert second.status_code == 200
    second_session = second.json()
    assert second_session["session_epoch"] == 2
    assert second_session["session_id"] != first_session["session_id"]

    stale = local_client.post("/generate_accompaniment", json=payload)
    assert stale.status_code == 409
    assert "stale session epoch" in stale.json()["detail"]

    payload.update(
        {
            "session_id": second_session["session_id"],
            "session_epoch": second_session["session_epoch"],
            "request_id": "second-r000001",
        }
    )
    current = local_client.post("/generate_accompaniment", json=payload)
    assert current.status_code == 200
    current_metadata = current.json()["metadata"]
    assert current_metadata["session_epoch"] == 2
    assert current_metadata["effective_seed"] == 202


def test_reset_endpoint_is_fail_closed_by_default(monkeypatch):
    monkeypatch.delenv("LEKAI_ENABLE_DEBUG_RESET", raising=False)
    fresh_backend = LekaiHttpBackend()
    monkeypatch.setattr(server_lekai, "backend", fresh_backend)

    response = TestClient(app).post("/debug/reset_session", json={"seed": 123})

    assert response.status_code == 403
    assert "LEKAI_ENABLE_DEBUG_RESET" in response.json()["detail"]
    assert fresh_backend.runtime_info()["session_epoch"] == 0


def test_prompt_continuation_reset_endpoint_is_dedicated_and_delegates(monkeypatch):
    monkeypatch.setenv("LEKAI_ENABLE_DEBUG_RESET", "true")

    class _ResetBackend:
        def reset_session(self, *, prompt_seed, continuation_seed):
            return {
                "success": True,
                "prompt_seed": prompt_seed,
                "continuation_effective_seed": continuation_seed,
                "session_id": "pc-session",
                "session_epoch": 4,
                "pending_boundary_generations": 0,
                "scheduler_phase": "idle",
                "scheduler_is_running": False,
            }

    monkeypatch.setattr(server_lekai, "prompt_continuation_backend", _ResetBackend())

    response = TestClient(app).post(
        "/prompt_continuation/debug/reset_session",
        json={"prompt_seed": 101, "continuation_seed": 202},
    )

    assert response.status_code == 200
    assert response.json()["prompt_seed"] == 101
    assert response.json()["continuation_effective_seed"] == 202
    assert response.json()["session_id"] == "pc-session"
    assert response.json()["session_epoch"] == 4


def test_prompt_continuation_reset_endpoint_is_fail_closed_by_default(monkeypatch):
    monkeypatch.delenv("LEKAI_ENABLE_DEBUG_RESET", raising=False)

    class _UnexpectedResetBackend:
        def reset_session(self, **kwargs):
            pytest.fail(f"reset should not be called: {kwargs}")

    monkeypatch.setattr(
        server_lekai,
        "prompt_continuation_backend",
        _UnexpectedResetBackend(),
    )

    response = TestClient(app).post(
        "/prompt_continuation/debug/reset_session",
        json={"prompt_seed": 101, "continuation_seed": 202},
    )

    assert response.status_code == 403
    assert "LEKAI_ENABLE_DEBUG_RESET" in response.json()["detail"]
