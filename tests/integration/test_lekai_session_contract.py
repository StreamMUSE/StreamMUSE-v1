from __future__ import annotations

from urllib.parse import urlparse

import requests
from fastapi.testclient import TestClient

from streammuse.application.services.real_time_music_service import (
    RealTimeMusicService,
    _INFERENCE_STOP,
)
from streammuse.domain.musical import EventType, MusicalEvent
from streammuse.domain.timing import PlaybackScheduler, Tempo
from streammuse.infrastructure.inference import server_lekai
from streammuse.infrastructure.inference.http_client import (
    HttpInferenceClient,
    HttpInferenceClientConfig,
)
from streammuse.infrastructure.inference.lekai_http_backend import LekaiHttpBackend


class _NoInput:
    def read_events(self):
        return iter(())

    def close(self):
        return None


class _LifecycleOutput:
    inference_log_detail = "full"

    def __init__(self) -> None:
        self.lifecycle = []

    def output_event(self, event, source):
        _ = event, source

    def output_tick(self, tick, bar, beat):
        _ = tick, bar, beat

    def output_stats(self, **kwargs):
        _ = kwargs

    def output_status(self, state, message=""):
        _ = state, message

    def output_config(self, config):
        _ = config

    def log_request_lifecycle(self, row):
        self.lifecycle.append(dict(row))

    def close(self):
        return None


def _route_requests_to_test_client(monkeypatch, test_client: TestClient) -> None:
    def post(url, json=None, timeout=None):
        _ = timeout
        return test_client.post(urlparse(url).path, json=json)

    def get(url, timeout=None):
        _ = timeout
        return test_client.get(urlparse(url).path)

    monkeypatch.setattr(requests, "post", post)
    monkeypatch.setattr(requests, "get", get)


def test_service_client_server_metadata_contract_survives_backend_history_trim(
    monkeypatch,
):
    monkeypatch.setenv("LEKAI_ENABLE_DEBUG_RESET", "true")
    monkeypatch.setenv("LEKAI_HISTORY_MAX_TICKS", "4")
    backend = LekaiHttpBackend()
    monkeypatch.setattr(server_lekai, "backend", backend)
    test_client = TestClient(server_lekai.app)
    _route_requests_to_test_client(monkeypatch, test_client)

    client = HttpInferenceClient(
        HttpInferenceClientConfig(
            generate_url="http://lekai.test/generate_accompaniment",
            timeout_s=2.0,
            model_name="lekai",
            inference_mode="sliding_window",
            generation_interval_ticks=4,
            bpm=120,
            input_file="contract.mid",
        )
    )
    reset = client.reset_session(seed=20260710)
    assert reset["session_epoch"] == 1

    output = _LifecycleOutput()
    service = RealTimeMusicService(
        input_source=_NoInput(),
        inference_engine=client,
        output_sink=output,
        tempo=Tempo(bpm=120.0, ticks_per_beat=4, beats_per_bar=4),
        scheduler=PlaybackScheduler(),
        now=lambda: 0.0,
        sleep=lambda _: None,
    )
    service._running = True

    requests_to_run = [
        (
            4,
            MusicalEvent(
                tick=0,
                pitch=60,
                event_type=EventType.NOTE_ON,
                velocity=90,
                channel=1,
                program=8,
            ),
        ),
        (
            20,
            MusicalEvent(
                tick=16,
                pitch=62,
                event_type=EventType.NOTE_ON,
                velocity=91,
                channel=1,
                program=8,
            ),
        ),
        (
            24,
            MusicalEvent(
                tick=20,
                pitch=64,
                event_type=EventType.NOTE_ON,
                velocity=92,
                channel=1,
                program=8,
            ),
        ),
    ]

    lifecycle_requests = []
    for generation_tick, event in requests_to_run:
        request = service._register_request(generation_tick, [event])
        lifecycle_requests.append(request)
        service._inference_request_queue.put(request)
        service._inference_request_queue.put(_INFERENCE_STOP)
        service._worker_done.clear()
        service._inference_worker()
        service._process_inference_responses(current_tick=generation_tick)

    summary = service._build_validity_summary()
    assert summary["content"]["valid"] is True
    assert summary["content"]["metadata_invalid_request_ids"] == []
    assert all(row["metadata_contract_valid"] for row in summary["requests"])

    succeeded = [row for row in output.lifecycle if row["event"] == "succeeded"]
    assert [row["response_metadata"]["request_id"] for row in succeeded] == [
        request.request_id for request in lifecycle_requests
    ]
    assert all(
        row["response_metadata"]["session_id"] == reset["session_id"]
        and row["response_metadata"]["session_epoch"] == reset["session_epoch"]
        and row["response_metadata"]["effective_seed"] == 20260710
        for row in succeeded
    )

    third_metadata = succeeded[-1]["response_metadata"]
    assert third_metadata["part0_trace_available"] is False
    assert third_metadata["input_cumulative_digest"] == lifecycle_requests[-1].input_cumulative_digest
    assert third_metadata["input_increment_digest"] == lifecycle_requests[-1].input_increment_digest
    assert third_metadata["part0_roll_digest"] != third_metadata["input_cumulative_digest"]
    assert all(int(event["tick"]) >= 20 for event in backend._melody_history)
    assert len(backend._input_digest_history) == 3
