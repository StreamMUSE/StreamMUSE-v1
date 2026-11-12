"""Callback-level inference contracts, without network or an audio device."""

from dataclasses import replace
import json
import csv
import io
from threading import Event, Thread

import numpy as np
import pytest

from streammuse.application.rap.inference import InferenceService
from streammuse.domain.rap.inference import AudioSpan


class Clock:
    now = 0.0

    def __call__(self):
        return self.now


class Producer:
    capabilities = {"renderer": "deterministic_fixture", "generation_unit": "tick",
                    "delivery_unit": "variable", "commit_quantum_ticks": 1,
                    "min_prefix_ticks": 1, "guard_ms": 0,
                    "max_immutable_lookahead_ticks": 1}

    def __init__(self):
        self.requests = []
        self.cancelled = []

    def submit(self, request, on_span, on_event):
        self.requests.append((request, on_span, on_event))

    def cancel(self, request_id):
        self.cancelled.append(request_id)

    def close(self):
        pass

    def deliver(self, start=0, ticks=1, value=0.5, request=-1, **kwargs):
        context, emit, _ = self.requests[request]
        span = AudioSpan(request_id=context.request_id, output_id=f"{context.request_id}:{start}",
                         epoch=context.epoch, input_version=context.input_version,
                         start_tick=start, duration_ticks=ticks, sample_rate=100, channels=1,
                         pcm=np.full((ticks * 25, 1), value, dtype=np.float32).tobytes(),
                         safe_split_ticks=tuple(range(1, ticks)), **kwargs)
        return emit(span)


@pytest.fixture
def setup():
    clock, producer = Clock(), Producer()
    service = InferenceService(session_id="shared", backend=producer, clock=clock,
                               sample_rate=100, channels=1,
                               drums=np.full((400, 1), 0.1, dtype=np.float32))
    yield service, producer, clock
    service.close()


def capture(service):
    service.start(owner_id="tab")
    service.audio_callback(100)
    snap = service.snapshot()
    service.tap({"owner_id": "tab", "session_id": "shared", "epoch": snap["epoch"],
                 "cycle_id": snap["cycle_id"], "event_id": "tap", "phase_elapsed_ms": 500})
    service.audio_callback(300)


def test_wait_is_silent_until_first_contiguous_decoded_prefix(setup):
    service, producer, clock = setup
    capture(service)
    assert service.snapshot()["phase"] == "WAIT"
    assert not np.any(service.audio_callback(17))
    producer.deliver(start=4)
    assert not np.any(service.audio_callback(17))
    clock.now = 4.137
    producer.deliver(start=0)
    assert service.snapshot()["phase"] == "WAIT"
    clock.now = 4.14
    assert np.allclose(service.audio_callback(5), 0.5)
    snap = service.snapshot()
    assert snap["phase"] == "MODEL"
    assert snap["current"]["scheduled_start_ms"] == 4140
    assert snap["metrics"]["ready_to_start_overhead_ms"] == pytest.approx(3)


def test_capture_preserves_leading_rest_and_repeats_over_response(setup):
    service, producer, _ = setup
    capture(service)
    request = producer.requests[-1][0]
    assert request.flow_ticks == (2, 18)
    assert request.duration_ticks == 32


@pytest.mark.parametrize("chunks", [(32,), (4,) * 8, (1,) * 32, (3, 7, 1, 21)])
def test_packet_counts_do_not_change_timeline_or_coverage(setup, chunks):
    service, producer, _ = setup
    capture(service)
    offset = 0
    for size in chunks:
        producer.deliver(start=offset, ticks=size)
        offset += size
    out = service.audio_callback(800)
    assert np.allclose(out, 0.5)
    snap = service.snapshot()
    assert snap["phase"] == "USER"
    assert snap["metrics"]["missing_duration_ms"] == 0
    assert snap["metrics"]["eligible_duration_ms"] == 8000


def test_missing_intervals_are_drums_only_and_do_not_extend_model(setup):
    service, producer, _ = setup
    capture(service)
    producer.deliver(ticks=1)
    out = service.audio_callback(800)
    assert np.allclose(out[:25], 0.5)
    assert np.allclose(out[25:], 0.1)
    assert service.snapshot()["phase"] == "USER"
    assert service.snapshot()["metrics"]["missing_duration_ms"] == 7750


def test_stop_wait_fences_late_work_and_restart_clears_capture(setup):
    service, producer, _ = setup
    capture(service)
    service.stop()
    producer.deliver(ticks=32)
    assert not np.any(service.audio_callback(25))
    service.start(owner_id="other")
    service.audio_callback(400)
    assert service.snapshot()["phase"] == "USER"
    assert len(producer.requests) == 1
    assert producer.cancelled


def test_duplicate_and_wrong_owner_taps_do_not_change_flow(setup):
    service, producer, _ = setup
    service.start(owner_id="tab")
    service.audio_callback(100)
    snap = service.snapshot()
    tap = {"owner_id": "tab", "session_id": "shared", "epoch": snap["epoch"],
           "cycle_id": snap["cycle_id"], "event_id": "once", "phase_elapsed_ms": 500}
    service.tap(tap)
    service.tap(tap)
    with pytest.raises((ValueError, RuntimeError)):
        service.tap({**tap, "owner_id": "other", "event_id": "bad"})
    service.audio_callback(300)
    assert producer.requests[-1][0].flow_ticks == (2, 18)


def test_steering_keeps_old_compatible_audio_and_reports_stale(setup):
    service, producer, _ = setup
    service.configure({"mode": "steering"})
    service.start()
    producer.deliver(ticks=32)
    service.audio_callback(25)
    service.steer({"topic": "new topic"})
    out = service.audio_callback(25)
    assert np.allclose(out, 0.5)
    snap = service.snapshot()
    assert snap["current"]["playing_version"] < snap["current"]["requested_version"]
    assert snap["metrics"]["stale_duration_ms"] == 250
    assert producer.requests[-1][0].start_tick == 1


def test_current_replacement_wins_only_before_commitment(setup):
    service, producer, _ = setup
    service.configure({"mode": "steering"})
    service.start()
    producer.deliver(ticks=32)
    service.audio_callback(25)
    service.steer({"topic": "new"})
    producer.deliver(start=1, ticks=1, value=0.8)
    assert np.allclose(service.audio_callback(25), 0.8)
    producer.deliver(start=1, ticks=1, value=0.9)
    assert np.allclose(service.audio_callback(25), 0.5)


def test_owner_expiry_stops_wait_and_continuous_has_no_lease(setup):
    service, producer, clock = setup
    capture(service)
    clock.now = 7
    service.audio_callback(1)
    assert service.snapshot()["state"] == "stopped"
    service.configure({"mode": "steering"})
    service.start()
    clock.now = 100
    producer.deliver(ticks=32)
    service.audio_callback(1)
    assert service.snapshot()["phase"] == "MODEL"


def test_snapshot_and_exports_are_json_safe_and_ttft_is_unavailable(setup):
    service, producer, _ = setup
    capture(service)
    producer.deliver(ticks=32)
    service.audio_callback(1)
    snap = service.snapshot()
    assert snap["session_id"] == "shared"
    assert snap["metrics"]["llm_ttft_ms"] is None
    json.dumps(snap, allow_nan=False)
    json.dumps(service.export_trace(), allow_nan=False)
    assert "request_id" in service.export_metrics_csv()


def test_invalid_span_frames_never_become_ready(setup):
    service, producer, _ = setup
    capture(service)
    context, emit, _ = producer.requests[-1]
    invalid = AudioSpan(request_id=context.request_id, output_id="bad", epoch=context.epoch,
                        input_version=context.input_version, start_tick=0, duration_ticks=1,
                        sample_rate=100, channels=1, pcm=b"\0" * 4)
    with pytest.raises(ValueError):
        emit(invalid)
    assert service.snapshot()["phase"] == "WAIT"


def test_completed_response_is_not_counted_as_canceled(setup):
    service, producer, _ = setup
    capture(service)
    producer.deliver(ticks=32, final=True)
    service.audio_callback(800)
    snap = service.snapshot()
    assert snap["requests"][-1]["state"] == "played"
    assert snap["metrics"]["canceled_count"] == 0


def test_each_response_gets_its_own_first_vocal_onset(setup):
    service, producer, _ = setup
    capture(service)
    producer.deliver(ticks=32, first_vocal_tick=2)
    service.audio_callback(800)
    service.audio_callback(100)
    snap = service.snapshot()
    service.tap({"owner_id": "tab", "session_id": "shared", "epoch": snap["epoch"],
                 "cycle_id": snap["cycle_id"], "event_id": "next", "phase_elapsed_ms": 500})
    service.audio_callback(300)
    assert service.snapshot()["metrics"]["first_vocal_onset_ms"] is None


def test_missing_intervals_are_merged_and_clear_playing_lyrics(setup):
    service, producer, _ = setup
    capture(service)
    producer.deliver(ticks=1, text="first line")
    service.audio_callback(25)
    service.audio_callback(25)
    service.audio_callback(25)
    snapshot = service.snapshot()
    assert snapshot["current"]["text"] == ""
    missing = [span for span in snapshot["spans"] if span["state"] == "missing"]
    assert len(missing) == 1
    assert missing[0]["start_tick"] == 1
    assert missing[0]["duration_ticks"] == 2


def test_finished_response_cannot_prime_the_next_cycle(setup):
    service, producer, _ = setup
    capture(service)
    producer.deliver(ticks=32)
    service.audio_callback(800)
    producer.deliver(ticks=32, value=0.9, revision=1)
    service.audio_callback(100)
    snap = service.snapshot()
    service.tap({"owner_id": "tab", "session_id": "shared", "epoch": snap["epoch"],
                 "cycle_id": snap["cycle_id"], "event_id": "next", "phase_elapsed_ms": 500})
    service.audio_callback(300)
    assert not np.any(service.audio_callback(25))
    assert service.snapshot()["phase"] == "WAIT"


def test_audio_failure_stops_and_exposes_recoverable_error(setup):
    service, _, _ = setup
    service.start("tab")
    service.fail("PortAudio output underflow")
    assert service.snapshot()["state"] == "error"
    assert "underflow" in service.snapshot()["error"]
    assert not np.any(service.audio_callback(25))


def test_candidate_counts_are_per_decision_not_per_packet(setup):
    service, producer, clock = setup
    capture(service)
    request, _, event = producer.requests[-1]
    clock.now = 0.137
    counts = {"type": "candidate_counts", "request_id": request.request_id, "decision_id": "choice",
              "candidates_requested": 8, "candidates_valid": 3, "candidates_parsed": 6}
    event(counts)
    event(counts)
    producer.deliver(ticks=1)
    producer.deliver(start=1, ticks=1)
    row = service.snapshot()["requests"][-1]
    assert row["valid_candidates"] == 3
    assert row["requested_candidates"] == 8
    assert row["parsed_candidates"] == 6
    assert row["evaluated_candidates"] is None
    assert row["first_valid_candidate_ms"] == pytest.approx(137)


def test_csv_contains_span_deadline_without_inventing_initial_deadline(setup):
    service, producer, clock = setup
    capture(service)
    producer.deliver(ticks=1)
    producer.deliver(start=1, ticks=31)
    clock.now = 0.1
    service.audio_callback(50)
    rows = list(csv.DictReader(io.StringIO(service.export_metrics_csv())))
    span_rows = [row for row in rows if row["record_type"] == "span"]
    assert span_rows[0]["deadline_margin_ms"] == ""
    assert float(span_rows[1]["deadline_margin_ms"]) == 350


def test_late_heartbeat_cannot_revive_expired_capture(setup):
    service, _, clock = setup
    service.start("tab")
    clock.now = 7
    with pytest.raises(RuntimeError):
        service.heartbeat({"owner_id": "tab"})


def test_output_warning_is_visible_without_stopping_music(setup):
    service, _, _ = setup
    service.start("tab")
    service.warn("PortAudio output underflow")
    assert service.snapshot()["state"] == "running"
    assert "underflow" in service.snapshot()["current"]["status"]


def test_out_of_order_final_span_does_not_mean_complete_or_prefix_ready(setup):
    service, producer, clock = setup
    capture(service)
    clock.now = 0.1
    producer.deliver(start=16, ticks=16, final=True)
    row = service.snapshot()["requests"][-1]
    assert row["latency_ms"] == 100
    assert row["complete_latency_ms"] is None
    assert row["playable_prefix_ready_ms"] is None
    clock.now = 0.2
    producer.deliver(start=0, ticks=16)
    row = service.snapshot()["requests"][-1]
    assert row["complete_latency_ms"] == 200
    assert row["playable_prefix_ready_ms"] == 200


def test_fallback_coverage_is_unknown_without_adapter_policy(setup):
    service, producer, _ = setup
    capture(service)
    producer.deliver(ticks=32)
    service.audio_callback(25)
    assert service.snapshot()["metrics"]["coverage"]["fallback_rate"] is None


def test_calibrated_tap_cannot_arrive_far_in_the_future(setup):
    service, _, _ = setup
    service.start("tab")
    snap = service.snapshot()
    with pytest.raises(ValueError, match="future"):
        service.tap({"owner_id": "tab", "session_id": "shared", "epoch": snap["epoch"],
            "cycle_id": snap["cycle_id"], "event_id": "future", "phase_elapsed_ms": 3000})


def test_snapshot_deep_copy_does_not_hold_callback_lock(setup, monkeypatch):
    service, _, _ = setup
    service.start("tab")
    entered, release, played = Event(), Event(), Event()
    from streammuse.application.rap.inference import service as module
    original = module.copy.deepcopy
    def slow_copy(value):
        entered.set()
        release.wait(2)
        return original(value)
    monkeypatch.setattr(module.copy, "deepcopy", slow_copy)
    snapshot = Thread(target=service.snapshot)
    callback = Thread(target=lambda: (service.audio_callback(1), played.set()))
    snapshot.start()
    try:
        assert entered.wait(1)
        callback.start()
        assert played.wait(0.2)
    finally:
        release.set()
        snapshot.join(2)
        callback.join(2)


def test_completed_old_request_error_cannot_stop_next_wait(setup):
    service, producer, _ = setup
    capture(service)
    old, _, event = producer.requests[-1]
    producer.deliver(ticks=32)
    service.audio_callback(800)
    service.audio_callback(100)
    snap = service.snapshot()
    service.tap({"owner_id": "tab", "session_id": "shared", "epoch": snap["epoch"],
        "cycle_id": snap["cycle_id"], "event_id": "next", "phase_elapsed_ms": 500})
    service.audio_callback(300)
    event({"type": "error", "request_id": old.request_id, "message": "late cleanup failure"})
    assert service.snapshot()["phase"] == "WAIT"
    assert service.snapshot()["error"] is None


def test_active_wall_time_excludes_idle_before_and_between_runs(setup):
    service, _, clock = setup
    clock.now = 100
    service.configure({"mode": "steering"})
    service.start()
    clock.now = 102
    service.stop()
    clock.now = 500
    assert service.snapshot()["metrics"]["coverage"]["wall_ms"] == 2000
    service.start()
    clock.now = 503
    assert service.snapshot()["metrics"]["coverage"]["wall_ms"] == 5000
    service.reset()
    assert service.snapshot()["metrics"]["coverage"]["wall_ms"] == 0


def test_continuous_new_response_records_its_own_first_vocal_origin(setup):
    service, producer, clock = setup
    service.configure({"mode": "steering"})
    service.start()
    producer.deliver(ticks=32, first_vocal_tick=0)
    service.audio_callback(25)
    clock.now = 0.25
    service.steer({"topic": "new"})
    producer.deliver(start=1, ticks=2, first_vocal_tick=1)
    service.audio_callback(50)
    row = service.snapshot()["requests"][-1]
    assert row["first_vocal_onset_ms"] == 250


def test_cycle_tells_the_backend_to_prepare_at_every_user_phase_and_steering_never_does(setup):
    service, producer, _ = setup
    calls = []
    producer.prepare = lambda: calls.append(service.snapshot()["phase"])
    capture(service)
    assert calls == ["USER"] and service.snapshot()["phase"] == "WAIT"
    producer.deliver(start=0, ticks=32)
    service.audio_callback(800)
    assert service.snapshot()["phase"] == "USER" and calls == ["USER", "USER"]
    service.audio_callback(400)  # nothing captured: USER repeats and prepares again
    assert calls == ["USER", "USER", "USER"]
    service.reset()
    service.configure({"mode": "steering"})
    service.start()
    assert len(calls) == 3


def test_a_failing_prepare_never_reaches_the_audio_callback(setup):
    service, producer, _ = setup

    def prepare():
        raise RuntimeError("renderer unreachable")

    producer.prepare = prepare
    capture(service)
    assert service.snapshot()["phase"] == "WAIT" and service.snapshot()["error"] is None
