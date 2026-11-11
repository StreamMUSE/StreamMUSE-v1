import json
import hashlib
import time

import pytest

from scripts.experiments.research_day import story_acceptance as harness


def event(kind, sequence, bar=1, session="s", **payload):
    return {"event_type": kind, "sequence": sequence, "bar": bar, "session_id": session,
            "monotonic_ns": sequence * 1000, "payload": payload}


def test_playback_uses_chosen_version_not_newer_draft_or_current_story():
    events = [
        event("bar_reserved", 1, lyric_id="new", story_version=2, story_event_ids=["a", "b"]),
        event("bar_frozen", 2, lyric_id="old", revision=1, text="chosen lyric", choice="remote_older",
              story_version=1, story_event_ids=["a"]),
        event("bar_playback_started", 3, absolute_frame=100),
        event("bar_frozen", 4, bar=2, lyric_id="new", revision=2, text="next lyric",
              story_version=2, story_event_ids=["a", "b"]),
        event("bar_playback_started", 5, bar=2, absolute_frame=200),
    ]
    observations = harness.playback_observations(events)
    assert observations[0]["story_version"] == 1
    assert observations[0]["text"] == "chosen lyric"
    assert observations[0]["lyric_id"] == "old"
    first = harness.first_playback_for({"event_id": "b", "sequence": 2}, observations)
    assert first["bar"] == 2
    assert first["is_acoustic_observation"] is False


def test_missing_metadata_and_cross_session_freezes_never_establish_story_version():
    events = [event("bar_frozen", 1, session="old", story_version=1, story_event_ids=["a"]),
              event("bar_playback_started", 2),
              event("bar_frozen", 3, bar=2, lyric_id="x", text="no provenance"),
              event("bar_playback_started", 4, bar=2)]
    observations = harness.playback_observations(events)
    assert all(row["story_version"] is None for row in observations)
    assert harness.first_playback_for({"event_id": "a", "sequence": 1}, observations) is None


def test_malformed_success_response_is_not_recorded_as_accepted():
    class Client:
        def request_timed(self, *args):
            return {"status": 200, "body": {"event": {"text": "missing identity"}},
                    "request_start_ns": 1, "response_ns": 2, "elapsed_ms": .001}
    result = harness._submit(Client(), 0, "map lost", 0, 0, "run")
    assert result["accepted"] is False
    assert result["error"] == "invalid story acknowledgement"


def test_defaults_are_three_events_over_twelve_bars_at_sixty(tmp_path):
    args = harness.parse_args(["--output-dir", str(tmp_path)])
    assert args.bpm == 60 and args.bars == 12 and args.event_bars == [0, 4, 8]
    assert args.edit_phase == 0
    assert len(args.events) == 3


@pytest.mark.parametrize("flags", [["--event-bars", "4", "0", "8"], ["--bars", "8"],
                                  ["--poll-interval", "0"], ["--url", "http://remote:8018"]])
def test_invalid_run_parameters_fail_before_controls(tmp_path, flags):
    with pytest.raises(SystemExit):
        harness.parse_args(["--output-dir", str(tmp_path), *flags])


def test_failure_stops_in_finally_and_preserves_evidence(tmp_path, monkeypatch):
    calls = []
    class FakeClient:
        def __init__(self, url, output):
            self.evidence = (output / "http.jsonl").open("a")
        def stop(self):
            calls.append("stop")
            return {"state": "stopped"}
        def request(self, path, payload=None):
            calls.append(path)
            raise RuntimeError("settings unavailable")
    class FakePoller:
        def __init__(self, *args):
            pass
        def close(self):
            calls.append("poller.close")
        def sample(self):
            pass
    monkeypatch.setattr(harness, "LiveAcceptance", FakeClient)
    monkeypatch.setattr(harness, "StoryPoller", FakePoller)
    args = harness.parse_args(["--output-dir", str(tmp_path / "run")])
    summary = harness.run(args)
    assert calls.count("stop") == 2
    assert calls.index("stop") < calls.index("/api/control/settings")
    assert summary["status"] == "failed" and summary["ready"] is True
    assert json.loads((args.output_dir / "summary.json").read_text())["error"]
    with pytest.raises(FileExistsError):
        harness.run(args)


def test_mocked_twelve_bar_run_submits_at_zero_four_eight_and_stops(tmp_path, monkeypatch):
    calls, accepted = [], []
    class FakeClient:
        def __init__(self, url, output):
            self.output = output
            self.evidence = (output / "http.jsonl").open("a")
        def stop(self):
            calls.append("stop")
            return {"state": "stopped"}
        def request(self, path, payload=None):
            calls.append(path)
            if path == "/api/story":
                return {"version": len(accepted), "events": list(accepted)}
            return {}
        def request_timed(self, path, payload):
            calls.append(path)
            accepted.append({**payload, "sequence": len(accepted) + 1})
            now = time.monotonic_ns()
            return {"status": 200, "body": {"status": "accepted", "event": accepted[-1],
                                            "effective_from_bar": (len(accepted) - 1) * 4 + 1},
                    "request_start_ns": now, "response_ns": now, "elapsed_ms": 0}
    class FakePoller:
        def __init__(self, client, evidence, interval):
            self.evidence, self.bar = evidence, 0
        def start(self):
            pass
        def check(self):
            now = time.monotonic_ns()
            events = [event("bar_frozen", self.bar * 2 + 1, bar=self.bar, text=f"lyric {self.bar}",
                            lyric_id=f"lyric-{self.bar}", story_version=len(accepted),
                            story_event_ids=[item["event_id"] for item in accepted]),
                      event("bar_playback_started", self.bar * 2 + 2, bar=self.bar)]
            for item in events:
                item["monotonic_ns"] = now
            self.evidence.observe({"session_id": "s", "recent_events": events}, now)
            self.bar += 1
        def close(self):
            pass
        def sample(self):
            pass
    monkeypatch.setattr(harness, "LiveAcceptance", FakeClient)
    monkeypatch.setattr(harness, "StoryPoller", FakePoller)
    args = harness.parse_args(["--output-dir", str(tmp_path), "--poll-interval", "0.001"])
    summary = harness.run(args)
    assert summary["status"] == "observed"
    assert summary["played_bar_count"] == 12
    assert [row["observed_bar_at_submission"] for row in summary["events"]] == [0, 4, 8]
    assert [row["first_playback_callback"]["bar"] for row in summary["events"]] == [1, 5, 9]
    assert calls[:3] == ["stop", "/api/control/settings", "/api/control/reset"]
    assert calls.count("stop") == 2
    assert summary["is_acoustic_observation"] is False
    assert summary["semantic_fulfillment"] == "not_evaluated"
    assert summary["audio_artifact"]["warning"]
    assert summary["audio_artifact"]["path"] is None
    assert (tmp_path / "events.jsonl").exists()
    assert (tmp_path / "chosen_lyrics.json").exists()


def test_mix_artifact_is_preserved_with_sha256_after_stop(tmp_path):
    source = tmp_path / "mixed.wav"
    data = b"RIFF-test-session-audio"
    source.write_bytes(data)
    output = tmp_path / "run"
    output.mkdir()
    session = {"metadata": {"audio": {"artifact_paths": {"wav": str(source)}}}}
    result = harness.preserve_mix_artifact(session, output, stopped=True)
    assert result["status"] == "copied"
    assert result["source_path"] == str(source)
    assert result["path"] == str(output / "story_mix.wav")
    assert result["sha256"] == hashlib.sha256(data).hexdigest()
    assert result["warning"] is None
    source.write_bytes(b"later run")
    assert (output / "story_mix.wav").read_bytes() == data


@pytest.mark.parametrize("stopped", [True, False])
def test_missing_or_unstopped_mix_is_only_a_warning(tmp_path, stopped):
    session = {"metadata": {"audio": {"artifact_paths": {"wav": str(tmp_path / "missing.wav")}}}}
    result = harness.preserve_mix_artifact(session, tmp_path, stopped=stopped)
    assert result["warning"]
    assert result["path"] is None and result["sha256"] is None


def test_mix_copy_error_is_a_warning_and_does_not_overwrite(tmp_path):
    source = tmp_path / "mixed.wav"
    source.write_bytes(b"new")
    (tmp_path / "story_mix.wav").write_bytes(b"preserve")
    session = {"metadata": {"audio": {"artifact_paths": {"wav": str(source)}}}}
    result = harness.preserve_mix_artifact(session, tmp_path, stopped=True)
    assert result["status"] == "unavailable" and result["warning"]
    assert result["path"] is None
    assert (tmp_path / "story_mix.wav").read_bytes() == b"preserve"


@pytest.mark.parametrize("value", ["-0.1", "1", "nan", "inf"])
def test_invalid_edit_phase(tmp_path, value):
    with pytest.raises(SystemExit):
        harness.parse_args(["--output-dir", str(tmp_path), "--edit-phase", value])


def test_phase_submission_waits_until_tick_twelve_and_rejects_missed_bar():
    observations = [{"bar": 4, "monotonic_ns": 10_000_000_000}]
    def timing(now, rows=observations):
        return harness.submission_timing(rows, 4, bpm=90, phase=.75, now_ns=now)
    assert timing(11_999_999_999)["eligibility"] == "waiting"
    due = timing(12_000_000_000)
    assert due["eligibility"] == "eligible"
    assert due["scheduled_submit_ns"] == 12_000_000_000
    assert due["next_bar_ns"] == 12_666_666_667
    assert timing(12_666_666_667)["eligibility"] == "late"
    assert timing(12_000_000_000, observations + [{"bar": 5, "monotonic_ns": 12_000_000_000}])["eligibility"] == "late"
    assert timing(12_000_000_000, [{"bar": 5, "monotonic_ns": 12_000_000_000}])["eligibility"] == "late"


def test_submit_records_actual_phase_and_remaining_bar_budget(monkeypatch):
    class Client:
        def request_timed(self, *args):
            return {"status": 200, "body": {"status": "accepted", "event": {"event_id": "a", "sequence": 1},
                                            "effective_from_bar": 5},
                    "request_start_ns": 12_100_000_000, "response_ns": 12_110_000_000, "elapsed_ms": 10}
    monkeypatch.setattr(harness.time, "monotonic_ns", lambda: 12_000_000_000)
    timing = harness.submission_timing([{"bar": 4, "monotonic_ns": 10_000_000_000}],
                                      4, bpm=90, phase=.75, now_ns=12_000_000_000)
    result = harness._submit(Client(), 0, "map lost", 4, 4, "run", timing=timing)
    assert result["request_offset_ms"] == 2100
    assert result["actual_request_phase"] == pytest.approx(.7875)
    assert result["available_until_next_bar_ms"] == pytest.approx(566.666667)
    assert result["phase_timing_status"] == "within_trigger_bar"


def test_late_trigger_is_not_submitted(monkeypatch):
    class Client:
        def request_timed(self, *args):
            pytest.fail("late request must not be sent")
    monkeypatch.setattr(harness.time, "monotonic_ns", lambda: 15_000_000_000)
    timing = harness.submission_timing([{"bar": 4, "monotonic_ns": 10_000_000_000}],
                                      4, bpm=90, phase=.75, now_ns=15_000_000_000)
    result = harness._submit(Client(), 0, "map lost", 4, 4, "run", timing=timing)
    assert result["accepted"] is False and result["phase_timing_status"] == "late_not_submitted"
