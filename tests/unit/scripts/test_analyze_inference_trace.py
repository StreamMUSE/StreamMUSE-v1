"""The trace analysis script, on traces in the exported format."""

import json

import pytest

from scripts.analyze_inference_trace import analyze, main, render


def released(request_id, at, *, mode="cycle", version=1, start_tick=0, epoch=1, bpm=60):
    return {"type": "request_released", "at_ms": at, "epoch": epoch, "request_id": request_id,
            "context": {"epoch": epoch, "input_version": version, "bpm": bpm, "start_tick": start_tick,
                        "duration_ticks": 32, "config": {"mode": mode}}}


def backend(at, request_id, kind, **values):
    return {"type": "backend", "at_ms": at, "epoch": 1,
            "payload": {"type": kind, "request_id": request_id, **values}}


def ready(at, request_id, bar):
    return {"type": "decoded_ready", "at_ms": at, "epoch": 1, "request_id": request_id,
            "output_id": f"{request_id}:bar:{bar}", "start_tick": bar * 16, "duration_ticks": 16}


def stages(at, request_id, bar, *, fetch, render, prepare, continuation=None):
    extra = {} if continuation is None else {"continuation": continuation}
    return backend(at, request_id, "stage_timings", bar=bar, timings_ms={
        "client_fetch_decode_ms": fetch, "server_render_total_ms": render, "server_moss_ms": render - 30,
        "client_prepare_ms": prepare}, **extra)


def committed(at, request_id, version, start_tick, state="committed"):
    return {"type": "committed", "at_ms": at, "epoch": 1, "request_id": request_id, "input_version": version,
            "start_tick": float(start_tick), "duration_ticks": 16.0, "state": state, "committed_ms": at}


def cycle_trace():
    events = []
    for index, (start, lyric, done, model) in enumerate([(1000, 500, 1300, 1310), (20000, 700, 1700, 1704),
                                                         (40000, 600, 1500, 1520)], start=1):
        name = f"session:1:{index}"
        events += [
            released(name, start),
            backend(start + lyric, name, "candidate_counts", bar=0),
            backend(start + lyric + 150, name, "candidate_counts", bar=1),
            ready(start + done, name, 0),
            stages(start + done, name, 0, fetch=done - lyric - 80, render=470, prepare=80,
                   continuation={"used": index > 1, **({"handoff": True} if index > 1 else
                                                      {"reason": "no_previous_line"})}),
            {"type": "model_started", "at_ms": start + model + 1, "epoch": 1, "observed_callback_ms": start + model},
            ready(start + done + 400, name, 1),
            stages(start + done + 400, name, 1, fetch=900, render=500, prepare=70, continuation={"used": True}),
            backend(start + done + 420, name, "complete", bars_delivered=2),
        ]
    return {"schema_version": 1, "clock_domain": "client_monotonic", "units": "milliseconds", "events": events,
            "snapshot": {"capabilities": {"steering_adapter": "bounded"},
                         "metrics": {"missing_duration_ms": 0.0, "stale_duration_ms": 0.0,
                                     "eligible_duration_ms": 24000.0, "error_count": 0}}}


def test_cyclic_turns_are_split_into_lyric_audio_and_start_stages():
    report = analyze(cycle_trace())
    summary = report["summary"]
    assert summary["requests"] == 3 and summary["outcomes"] == {"complete": 3, "canceled": 0, "error": 0}
    assert summary["cycle_wait_ms"] == {"n": 3, "median": 1520, "min": 1310, "max": 1704}
    assert summary["cycle_first_lyric_ms"]["median"] == 600
    assert summary["cycle_lyric_to_ready_ms"]["median"] == 900
    assert summary["cycle_ready_to_start_ms"] == {"n": 3, "median": 10, "min": 4, "max": 20}
    assert summary["cycle_first_bar_server_render_ms"]["median"] == 470
    first = report["requests"][0]
    assert (first["first_lyric_ms"], first["first_bar_ready_ms"], first["wait_ms"]) == (500, 1300, 1310)
    assert first["all_bars_ready_ms"] == 1700 and first["first_bar_fetch_ms"] == 720
    # Stages add up to the WAIT.
    for row in report["requests"]:
        assert row["first_lyric_ms"] + row["lyric_to_ready_ms"] + row["ready_to_start_ms"] == row["wait_ms"]


def test_continuation_rate_skips_only_the_first_bar_of_the_trace():
    continuation = analyze(cycle_trace())["summary"]["continuation"]
    assert continuation["bars_reported"] == 6 and continuation["bars_counted"] == 5
    assert continuation["continued"] == 5 and continuation["handed_over"] == 2 and continuation["rate"] == 1.0
    trace = cycle_trace()
    trace["events"][13]["payload"]["continuation"] = {"used": False, "reason": "handoff_not_cached"}
    broken = analyze(trace)["summary"]["continuation"]
    assert broken["continued"] == 4 and broken["rate"] == 0.8
    assert broken["not_continued_reasons"] == {"handoff_not_cached": 1}


def test_trace_from_before_continuation_reports_says_so():
    trace = cycle_trace()
    for event in trace["events"]:
        event.get("payload", {}).pop("continuation", None)
    report = analyze(trace)
    assert report["summary"]["continuation"]["rate"] is None
    assert "no server continuation reports" in render(report)


def steering_trace():
    events = [
        released("session:1:1", 0, mode="steering"),
        committed(1000, "session:1:1", 1, 0), committed(5000, "session:1:1", 1, 16),
        released("session:1:2", 1100, mode="steering", start_tick=32),
        committed(9000, "session:1:2", 1, 32),
        # An edit one second into bar 2 (4 s per bar at 60 BPM): first half.
        {"type": "steered", "at_ms": 10000, "epoch": 1, "input_version": 2, "topic": "night market"},
        released("session:1:3", 10000, mode="steering", version=2, start_tick=48),
        backend(10001, "session:1:2", "canceled"),
        committed(13000, "session:1:3", 2, 48),
        committed(17000, "session:1:3", 2, 64),
        # A later edit three seconds into bar 4: second half; the next bar still plays the old plan.
        {"type": "steered", "at_ms": 20000, "epoch": 1, "input_version": 3, "template_id": "other"},
        released("session:1:4", 20000, mode="steering", version=3, start_tick=80),
        committed(21000, "session:1:3", 2, 80, state="stale"),
        committed(25000, "session:1:4", 3, 96),
        backend(26000, "session:1:4", "error", message="boom"),
    ]
    return {"schema_version": 1, "events": events,
            "snapshot": {"capabilities": {"steering_adapter": "session"},
                         "metrics": {"missing_duration_ms": 0.0, "stale_duration_ms": 8000.0,
                                     "eligible_duration_ms": 28000.0, "error_count": 1}}}


def test_steering_edits_report_latency_bar_offset_and_time_on_the_old_version():
    report = analyze(steering_trace())
    first, second = report["edits"]
    assert (first["input_version"], first["changed"], first["playing_bar"], first["half"]) == (2, ["topic"], 2, "first")
    assert first["position_in_bar"] == pytest.approx(0.25)
    assert (first["latency_ms"], first["performed_bar"], first["bars_later"]) == (3000, 3, 1)
    assert first["old_plan_started_after_edit_ms"] == 0
    assert (second["playing_bar"], second["half"], second["changed"]) == (4, "second", ["template_id"])
    assert (second["latency_ms"], second["bars_later"], second["old_plan_started_after_edit_ms"]) == (5000, 2, 4000)
    summary = report["summary"]
    assert summary["steering_latency_ms"] == {"n": 2, "median": 4000, "min": 3000, "max": 5000}
    assert summary["steering_bars_later"] == [1, 2] and summary["steering_adapter"] == "session"
    assert summary["outcomes"] == {"complete": 0, "canceled": 1, "error": 1}
    assert report["requests"][3]["error"] == "boom"
    assert summary["cycle_wait_ms"] is None


def test_edit_that_is_never_performed_has_no_latency():
    trace = steering_trace()
    trace["events"] = trace["events"][:-2]
    assert analyze(trace)["edits"][1]["latency_ms"] is None


def test_command_line_prints_the_report_and_writes_json(tmp_path, capsys):
    source, output = tmp_path / "trace.json", tmp_path / "report.json"
    source.write_text(json.dumps(cycle_trace()))
    assert main([str(source), "--json", str(output)]) == 0
    printed = capsys.readouterr().out
    assert "WAIT" in printed and "median 1520 ms" in printed and "5/5 bars (100.0%)" in printed
    assert json.loads(output.read_text())["summary"]["requests"] == 3
    source.write_text("{}")
    assert main([str(source)]) == 2
    assert "not an inference trace export" in capsys.readouterr().err
