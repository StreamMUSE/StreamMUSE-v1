"""Offline evidence tests only: never start the GUI, audio device, or models."""

from copy import deepcopy
import io
import json
import time

import pytest

from scripts.experiments import rap_live_edit_acceptance as live


def event(kind, seconds, bar=None, **payload):
    return {
        "session_id": "session", "sequence": round(seconds * 100),
        "event_type": kind, "monotonic_ns": round(seconds * 1e9),
        "utc_time": "2026-10-28T00:00:00Z", "bar": bar,
        "tick": None, "request_id": None, "payload": payload,
    }


@pytest.fixture
def edit():
    return {
        "index": 0, "topic": "quiet trains", "template_id": "mcflow_delayed",
        "request_start_ns": 10_000_000_000, "ack_ns": 10_200_000_000,
        "ack": {"edit_seq": 7, "effective_from_bar": 2}, "timeout_seconds": 5,
    }


@pytest.fixture
def trace():
    identity = dict(topic="quiet trains", template_id="mcflow_delayed",
                    revision=0, lyric_id="new", text="The quiet train")
    events = [
        event("bar_frozen", 9.0, 1, topic="old", template_id="old", revision=3,
              lyric_id="old", choice="remote_older"),
        event("bar_playback_started", 9.5, 1, absolute_frame=0),
        event("bar_reserved", 10.3, 2, source="lyric_stream", **identity),
        event("audio_render_completed", 10.35, 2, source="lyric_stream_espeak"),
        event("chunk_remote_completed", 10.4, 2, hashes={"lyric_id": "new"},
              stage_timings_ms={"moss": 400, "mac": 30}),
        event("bar_frozen", 10.5, 2, choice="remote_latest", source="moss_lyric_stream", latest_revision=0, **identity),
        event("bar_audio_committed", 10.6, 2, choice="remote_latest", source="moss_lyric_stream", timing_evidence={
            "bar": 2, "sample_rate_hz": 48000, "frame_count": 192000,
            "timing": {"start_frame": 192000, "end_frame": 384000},
            "requested_onsets": [{"sample": 24000}, {"sample": 48000}],
        }),
        event("bar_playback_completed", 10.9, 1, absolute_frame=192000),
        event("bar_playback_started", 11.0, 2, absolute_frame=192000),
        event("bar_playback_completed", 15.0, 2, absolute_frame=384000),
    ]
    for e in events:
        if e["bar"] == 2 and e["event_type"] in ("bar_frozen", "bar_audio_committed"):
            e["request_id"] = "new"
    return events


def test_exact_lyric_revision_join_and_leading_rest_callback_proxy(edit, trace):
    result = live.summarize_edit(trace, edit, now_ns=15_000_000_000)
    assert result["status"] == "passed"
    assert result["ack_ms"] == 200
    assert result["lyric_ready"]["latency_ms"] == 300
    assert result["first_matching_commit"]["latency_ms"] == 600
    assert result["first_matching_playback"]["latency_ms"] == 1000
    assert result["first_matching_playback"]["revision"] == 0  # Not edit_seq=7.
    onset = result["first_scheduled_vocal_onset"]
    assert onset["latency_ms"] == 1500
    assert onset["playback_sample_index"] == 216000
    assert onset["is_acoustic_observation"] is False
    assert onset["kind"] == "device_callback_schedule_proxy"
    assert result["old_bar_count"] == 1
    assert result["choice_counts"]["remote_older"] == 1
    assert result["render_stage_quantiles_ms"]["moss"]["p95"] == 400


@pytest.mark.parametrize("field,value", [
    ("topic", "wrong"), ("template_id", "wrong"), ("revision", 1),
    ("lyric_id", "wrong"), ("latest_revision", 1), ("revision", None),
])
def test_wrong_content_or_revision_never_passes_even_with_zero_underruns(edit, trace, field, value):
    next(e for e in trace if e["event_type"] == "bar_frozen" and e["bar"] == 2)["payload"][field] = value
    result = live.summarize_edit(trace, edit, now_ns=16_000_000_000)
    assert result["status"] == "timed_out"
    assert result["first_matching_playback"] is None
    assert result["first_scheduled_vocal_onset"] is None


def test_commit_is_not_playback_and_timeout_is_not_a_latency_sample(edit, trace):
    trace = [e for e in trace if e["bar"] != 2 or not e["event_type"].startswith("bar_playback")]
    result = live.summarize_edit(trace, edit, now_ns=16_000_000_000)
    assert result["status"] == "timed_out"
    assert result["first_matching_commit"] is not None
    assert result["first_matching_playback"] is None
    assert result["timeout_seconds"] == 5


def test_late_playback_cannot_turn_a_timeout_into_success(edit, trace):
    for e in trace:
        if e["bar"] == 2 and e["event_type"].startswith("bar_playback"):
            e["monotonic_ns"] += 10_000_000_000
    result = live.summarize_edit(trace, edit, now_ns=30_000_000_000)
    assert result["status"] == "timed_out"
    assert result["first_matching_playback"] is None


def test_espeak_completion_without_identity_is_not_claimed_as_revision_ready(edit, trace):
    result = live.summarize_edit(trace, edit, now_ns=15_000_000_000)
    assert result["espeak_ready"]["status"] == "unavailable_revision_identity"
    assert result["espeak_ready"]["latency_ms"] is None
    assert result["espeak_ready"]["bar_observations"][0]["latency_ms"] == 350


def test_missing_onset_or_completion_cannot_pass(edit, trace):
    missing_onsets = deepcopy(trace)
    next(e for e in missing_onsets if e["event_type"] == "bar_audio_committed")["payload"]["timing_evidence"]["requested_onsets"] = []
    assert live.summarize_edit(missing_onsets, edit, now_ns=16_000_000_000)["status"] == "timed_out"
    assert live.summarize_edit(trace[:-1], edit, now_ns=16_000_000_000)["status"] == "timed_out"


def test_elapsed_time_alone_cannot_be_reported_as_reached_vocal_onset(edit, trace):
    result = live.summarize_edit(trace[:-1], edit, now_ns=16_000_000_000)
    assert result["first_matching_playback"] is not None
    assert result["first_scheduled_vocal_onset"] is None


@pytest.mark.parametrize("end_frame", [None, 192001, 216000, 383999])
def test_completion_must_prove_full_matching_sample_extent(edit, trace, end_frame):
    trace[-1]["payload"]["absolute_frame"] = end_frame
    result = live.summarize_edit(trace, edit, now_ns=16_000_000_000)
    assert result["status"] != "passed"
    assert result["first_scheduled_vocal_onset"] is None


@pytest.mark.parametrize("choice", ["remote_older", "drums_only", "silence", None])
def test_only_current_vocal_choices_can_satisfy_the_edit(edit, trace, choice):
    for e in trace:
        if e["bar"] == 2 and e["event_type"] in ("bar_frozen", "bar_audio_committed"):
            e["payload"]["choice"] = choice
    result = live.summarize_edit(trace, edit, now_ns=16_000_000_000)
    assert result["status"] != "passed"
    assert result["first_matching_playback"] is None


@pytest.mark.parametrize("field,value", [
    ("request_id", None), ("request_id", "stale"), ("choice", "remote_older"),
    ("source", "drums_only"),
])
def test_audio_commit_must_have_same_lyric_identity_and_vocal_choice(edit, trace, field, value):
    commit = next(e for e in trace if e["event_type"] == "bar_audio_committed")
    target = commit if field == "request_id" else commit["payload"]
    target[field] = value
    result = live.summarize_edit(trace, edit, now_ns=16_000_000_000)
    assert result["status"] != "passed"
    assert result["first_matching_commit"] is None


def test_choice_counts_use_freeze_before_playback_not_a_future_replacement(edit, trace):
    trace.append(event("bar_frozen", 12, 1, choice="remote_latest"))
    result = live.summarize_edit(trace, edit, now_ns=15_000_000_000)
    assert result["choice_counts"]["remote_older"] == 1
    assert result["choice_counts"]["remote_latest"] == 1


def test_old_uncompleted_start_is_not_current_after_a_newer_bar_started(edit, trace):
    trace[:0] = [event("bar_frozen", 8, 0, choice="remote_older"),
                 event("bar_playback_started", 8.1, 0, absolute_frame=0)]
    result = live.summarize_edit(trace, edit, now_ns=15_000_000_000)
    assert result["in_flight_bars_at_request"] == [1]
    assert result["old_bar_count"] == 1


def test_late_ack_and_muted_vocals_cannot_pass(edit, trace):
    edit["ack_ns"] = 16_000_000_000
    assert live.summarize_edit(trace, edit, now_ns=16_000_000_000)["status"] == "timed_out"
    edit["ack_ns"] = 10_200_000_000
    next(e for e in trace if e["event_type"] == "bar_audio_committed")["payload"]["timing_evidence"]["mix"] = {
        "vocal_gain": 0, "master_gain": 1,
    }
    result = live.summarize_edit(trace, edit, now_ns=16_000_000_000)
    assert result["status"] == "failed"
    assert result["first_scheduled_vocal_onset"] is None


def test_underruns_fail_and_render_quantiles_do_not_double_count_commits(edit, trace):
    trace.extend([
        event("audio_underrun", 12),
        event("chunk_committed", 12.1, 2, stage_timings_ms={"moss": 400}),
        event("chunk_remote_completed", 12.2, 3, stage_timings_ms={"moss": 800}),
    ])
    result = live.summarize_edit(trace, edit, now_ns=15_000_000_000)
    assert result["status"] == "failed"
    assert result["underruns"] == 1
    assert result["render_stage_quantiles_ms"]["moss"] == {
        "n": 2, "p50": 600, "p95": 780, "p99": 796, "max": 800,
    }


def test_template_selection_requires_curated_flow_and_includes_leading_rest():
    templates = [
        {"template_id": "builtin", "slots": [{"tick_in_bar": 2}]},
        {"template_id": "mcflow_plain", "slots": [{"tick_in_bar": 0}]},
        {"template_id": "mcflow_delayed", "slots": [{"tick_in_bar": 2}]},
    ]
    selected = live.select_templates(templates)
    assert selected[0]["template_id"] == "mcflow_delayed"
    assert {t["template_id"] for t in selected} == {"mcflow_plain", "mcflow_delayed"}
    with pytest.raises(ValueError, match="leading rest"):
        live.select_templates(templates[1:2])


def test_collector_deduplicates_and_rejects_event_loss(tmp_path):
    evidence = live.EventEvidence(tmp_path)
    one, two, four = event("tick", 1), event("tick", 2), event("tick", 4)
    for seq, e in zip((1, 2, 4), (one, two, four)):
        e["sequence"] = seq
    evidence.observe({"recent_events": [one, two]}, 3_000_000_000)
    evidence.observe({"recent_events": [one, two]}, 3_000_000_000)
    evidence.observe({"recent_events": [four]}, 5_000_000_000)
    assert len(evidence.events) == 3
    assert evidence.gaps == [{"session_id": "session", "after": 2, "before": 4}]
    assert len((tmp_path / "events.jsonl").read_text().splitlines()) == 3


def test_collector_rejects_session_or_epoch_changes(tmp_path):
    for field in ("session_id", "coordinator_epoch"):
        evidence = live.EventEvidence(tmp_path)
        state = {"session_id": "session", "coordinator_epoch": 1, "recent_events": []}
        evidence.observe(state, 1)
        with pytest.raises(ValueError, match="session or epoch"):
            evidence.observe({**state, field: "changed"}, 2)


def test_espeak_selection_proves_ready_by_commit_but_not_exact_ready_time(edit, trace):
    for e in trace:
        if e["bar"] == 2 and e["event_type"] in ("bar_frozen", "bar_audio_committed"):
            e["payload"].update(choice="espeak_stream_lyric", source="lyric_stream_espeak")
    result = live.summarize_edit(trace, edit, now_ns=15_000_000_000)
    assert result["status"] == "passed"
    assert result["espeak_count"] == 1
    assert result["espeak_ready"]["latency_ms"] is None
    assert result["espeak_ready"]["confirmed_ready_by_commit_ms"] == 600


def test_cli_defaults_and_rejects_nonlocal_clock_or_invalid_counts():
    args = live.parse_args(["--output", "/tmp/evidence"])
    assert args.url == "http://127.0.0.1:8018"
    assert args.edits == 6
    assert args.bpms == [60]
    assert live.parse_args(["--output", "/tmp/evidence", "--bpm", "60", "90"]).bpms == [60, 90]
    for extra in (["--edits", "0"], ["--bpm", "nan"], ["--edit-spacing", "-1"],
                  ["--url", "http://remote:8018"]):
        with pytest.raises(SystemExit):
            live.parse_args(["--output", "/tmp/evidence", *extra])


class OfflineGui:
    """In-memory HTTP boundary; no socket, server, or device is opened."""

    def __init__(self, tmp_path, *, deliver=True, fail_cleanup=False):
        self.state = "stopped"
        self.events = []
        self.sequence = 0
        self.bar = 1
        self.deliver = deliver
        self.fail_cleanup = fail_cleanup
        self.stops = 0
        self.plan_requests = []
        self.state_reads_during_ack = 0
        self.ack_pending = False
        self.recording = tmp_path / "mixed.wav"
        self.recording.write_bytes(b"offline recording fixture")

    def emit(self, kind, bar=None, **payload):
        self.sequence += 1
        e = event(kind, 0, bar, **payload)
        e.update(sequence=self.sequence, monotonic_ns=time.monotonic_ns())
        if kind in ("bar_frozen", "bar_audio_committed"):
            e["request_id"] = f"lyric-{bar}"
        self.events.append(e)

    def __call__(self, request, timeout):
        path = live.urlsplit(request.full_url).path
        data = json.loads(request.data) if request.data else None
        if path == "/api/control/settings":
            result = {"state": self.state, "audio_output": {
                "backend": "sounddevice", "audible": self.state == "running",
                "recording": {"source": "completed_callback_pcm", "exact_completed_bars": True},
            }}
        elif path == "/api/control/stop":
            self.stops += 1
            if self.fail_cleanup:
                raise OSError("offline cleanup failed")
            self.state = "stopped"
            result = {"state": "stop_requested"}
        elif path == "/api/control/reset":
            assert self.state == "stopped", "reset must not interrupt audio"
            self.events = []
            self.bar = 1
            self.emit("session_reset")
            result = {"state": "stopped"}
        elif path == "/api/control/start":
            self.state = "running"
            for bar in (0, 1):
                self.emit("bar_frozen", bar, choice="remote_latest", text="warm")
                self.emit("bar_playback_started", bar, absolute_frame=bar * 48000)
                self.emit("bar_playback_completed", bar, absolute_frame=(bar + 1) * 48000)
            result = {"state": "priming"}
        elif path == "/api/control/plan" and data is None:
            result = {"supported": True, "templates": [
                {"template_id": "mcflow_delay", "slots": [{"tick_in_bar": 1}]},
                {"template_id": "mcflow_plain", "slots": [{"tick_in_bar": 0}]},
            ]}
        elif path == "/api/control/plan":
            self.plan_requests.append((time.monotonic_ns(), data))
            self.bar += 1
            if self.deliver:
                identity = {**data, "revision": 0, "lyric_id": f"lyric-{self.bar}", "text": "test"}
                self.emit("bar_reserved", self.bar, **identity)
                self.emit("bar_frozen", self.bar, **identity, latest_revision=0, choice="remote_latest", source="moss_lyric_stream")
                self.emit("bar_audio_committed", self.bar, choice="remote_latest", source="moss_lyric_stream", timing_evidence={
                    "bar": self.bar, "sample_rate_hz": 48000, "frame_count": 48000,
                    "requested_onsets": [{"sample": 1}],
                })
                self.emit("bar_playback_started", self.bar, absolute_frame=self.bar * 48000)
                self.emit("bar_playback_completed", self.bar, absolute_frame=(self.bar + 1) * 48000)
            self.ack_pending = True
            time.sleep(.025)
            self.ack_pending = False
            result = {"edit_seq": len(self.plan_requests), "effective_from_bar": self.bar}
        elif path == "/api/state":
            self.state_reads_during_ack += self.ack_pending
            result = {"session_id": "session", "coordinator_epoch": 1,
                      "recent_events": list(self.events), "audio": {
                          "underruns": 0, "recording_path": str(self.recording),
                      }}
        elif path == "/api/session":
            result = {"session_id": "session", "session_dir": str(self.recording.parent)}
        else:
            raise AssertionError(f"unexpected request: {path}")
        response = io.BytesIO(json.dumps(result).encode())
        response.status = 200
        return response


def run_offline(monkeypatch, tmp_path, extra=(), **backend_options):
    backend = OfflineGui(tmp_path, **backend_options)
    monkeypatch.setattr(live.urllib.request, "urlopen", backend)
    args = live.parse_args([
        "--output", str(tmp_path / "evidence"), "--edits", "2",
        "--timeout", ".12", "--warmup-timeout", ".2",
        "--edit-spacing", ".01", "--poll-interval", ".001", *extra,
    ])
    result = live.run(args)
    return backend, result, args.output


def test_run_collects_during_ack_spaces_edits_and_leaves_stopped(monkeypatch, tmp_path):
    backend, summary, output = run_offline(monkeypatch, tmp_path)
    assert summary["status"] == "passed"
    assert backend.state == "stopped"
    assert backend.stops == 1
    assert backend.state_reads_during_ack > 0
    assert backend.plan_requests[1][0] - backend.plan_requests[0][0] >= 35_000_000
    assert backend.plan_requests[0][1]["template_id"] == "mcflow_delay"
    run = summary["runs"][0]
    assert len(run["edits"]) == 2
    assert run["ready"] is True
    assert run["recording"]["source_path"] == str(backend.recording)
    assert live.Path(run["recording"]["preserved_path"]).read_bytes() == backend.recording.read_bytes()
    assert (output / "summary.json").is_file()
    http = [json.loads(s) for s in (output / "01_60bpm" / "http.jsonl").read_text().splitlines()]
    assert any(row["path"] == "/api/control/plan" and row["input"] is not None
               and row["request_start_ns"] < row["response_ns"] for row in http)
    edits = [json.loads(s) for s in (output / "01_60bpm" / "edits.jsonl").read_text().splitlines()]
    assert all(e["session_wav_path"] == run["recording"]["preserved_path"] for e in edits)
    assert edits[0]["audition"]["onset_sample"] == 96001


def test_run_timeout_is_failed_and_still_stops_and_writes_summary(monkeypatch, tmp_path):
    backend, summary, output = run_offline(monkeypatch, tmp_path, deliver=False)
    assert summary["status"] == "failed"
    assert backend.state == "stopped"
    assert all(e["status"] == "timed_out" for e in summary["runs"][0]["edits"])
    assert summary["runs"][0]["latency_quantiles_ms"]["first_matching_playback"]["n"] == 0
    assert json.loads((output / "summary.json").read_text())["status"] == "failed"


def test_cleanup_failure_is_retained_without_losing_measurements(monkeypatch, tmp_path):
    backend, summary, output = run_offline(monkeypatch, tmp_path, fail_cleanup=True)
    assert summary["status"] == "failed"
    assert summary["runs"][0]["ready"] is False
    assert "offline cleanup failed" in summary["runs"][0]["cleanup_error"]
    assert len(summary["runs"][0]["edits"]) == 2
    assert (output / "summary.json").exists()


def test_multiple_bpms_preserve_separate_recordings(monkeypatch, tmp_path):
    backend, summary, _ = run_offline(monkeypatch, tmp_path, extra=["--bpm", "60", "90"])
    assert summary["status"] == "passed"
    assert [r["bpm"] for r in summary["runs"]] == [60, 90]
    assert len({r["recording"]["preserved_path"] for r in summary["runs"]}) == 2
    assert len(backend.plan_requests) == 4


def test_http_failure_is_raw_evidence_not_an_ack(monkeypatch, tmp_path):
    backend = OfflineGui(tmp_path)

    def fail_plan(request, timeout):
        if live.urlsplit(request.full_url).path == "/api/control/plan" and request.data:
            raise TimeoutError("offline plan deadline")
        return backend(request, timeout)

    monkeypatch.setattr(live.urllib.request, "urlopen", fail_plan)
    args = live.parse_args(["--output", str(tmp_path / "evidence"), "--edits", "1",
                            "--poll-interval", ".001", "--warmup-timeout", ".2"])
    result = live.run(args)
    edit = result["runs"][0]["edits"][0]
    assert result["status"] == "failed"
    assert backend.state == "stopped"
    assert edit["ack_ns"] is None
    assert edit["ack_ms"] is None
    assert edit["http_response_ns"] >= edit["request_start_ns"]
    rows = [json.loads(s) for s in (args.output / "01_60bpm" / "http.jsonl").read_text().splitlines()]
    assert any("offline plan deadline" in row.get("error", "") for row in rows)


def test_error_before_start_still_leaves_ready_and_preserves_evidence(monkeypatch, tmp_path):
    backend = OfflineGui(tmp_path)

    def no_templates(request, timeout):
        response = backend(request, timeout)
        if live.urlsplit(request.full_url).path == "/api/control/plan":
            response = io.BytesIO(json.dumps({"supported": False}).encode())
            response.status = 200
        return response

    monkeypatch.setattr(live.urllib.request, "urlopen", no_templates)
    args = live.parse_args(["--output", str(tmp_path / "evidence")])
    result = live.run(args)
    assert result["status"] == "failed"
    assert result["runs"][0]["ready"] is True
    assert backend.state == "stopped"
    assert not backend.plan_requests


def test_stop_retries_transient_starting_conflict_without_reset(monkeypatch, tmp_path):
    backend = OfflineGui(tmp_path)
    backend.state = "running"
    attempts = []

    def transient_conflict(request, timeout):
        if live.urlsplit(request.full_url).path == "/api/control/stop":
            attempts.append(1)
            if len(attempts) == 1:
                raise live.urllib.error.HTTPError(request.full_url, 409, "starting", {},
                                                  io.BytesIO(b'{"detail":"starting"}'))
        return backend(request, timeout)

    monkeypatch.setattr(live.urllib.request, "urlopen", transient_conflict)
    test = live.LiveAcceptance("http://127.0.0.1:8018", tmp_path)
    try:
        assert test.stop()["state"] == "stopped"
        assert len(attempts) == 2
    finally:
        test.evidence.close()


def test_live_callback_mute_cannot_hide_behind_prepared_mix(tmp_path):
    poller = live.StatePoller(None, live.EventEvidence(tmp_path), .1)
    poller.settings = {"state": "running", "audio_output": {
        "backend": "sounddevice", "audible": True,
        "live_mix": {"applied": {"vocal_gain": 0}, "requested": {"vocal_gain": .8}},
    }}
    with pytest.raises(AssertionError, match="vocal"):
        poller.check()


@pytest.mark.parametrize("deliver", [True, False])
def test_old_settings_topic_gate_requires_played_revision_without_transport_changes(monkeypatch, tmp_path, deliver):
    monkeypatch.syspath_prepend(str(live.Path(live.__file__).parent))
    from scripts.experiments import rap_settings_acceptance as settings

    backend = OfflineGui(tmp_path, deliver=deliver)
    backend.state = "running"
    monkeypatch.setattr(live.urllib.request, "urlopen", backend)
    test = live.Acceptance("http://127.0.0.1:8018", tmp_path)
    template = {"template_id": "mcflow_delay", "slots": [{"tick_in_bar": 1}]}
    try:
        if deliver:
            result = settings.change_topic_flow(test, "quiet trains", template, timeout=.15, poll_interval=.001)
            assert result["status"] == "passed"
            assert result["first_matching_playback"]["lyric_id"] == "lyric-2"
            assert result["first_matching_playback"]["revision"] == 0
            assert result["matching_playback_completed"] is not None
        else:
            with pytest.raises(AssertionError, match="matching revision"):
                settings.change_topic_flow(test, "quiet trains", template, timeout=.15, poll_interval=.001)
        saved = json.loads((tmp_path / "topic_edit" / "summary.json").read_text())
        assert saved["status"] == ("passed" if deliver else "timed_out")
        assert backend.state == "running"
        assert backend.stops == 0
        assert not any(e["event_type"] == "session_reset" for e in backend.events)
    finally:
        test.evidence.close()
