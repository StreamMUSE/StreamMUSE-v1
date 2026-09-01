import json
from types import SimpleNamespace

import mido
import pretty_midi
import pytest

from streammuse.domain.musical import EventType, MusicalEvent
from streammuse.infrastructure.output.composite import CompositeOutputSink
from streammuse.infrastructure.output.json_logger import JsonLoggerOutputSink
from streammuse.infrastructure.output.metronome import MetronomeOutputConfig, MetronomeOutputSink
from streammuse.infrastructure.output.midi_file import MidiFileOutputConfig, MidiFileOutputSink
from streammuse.infrastructure.output.session_logger import SessionLoggerOutputSink
from streammuse.infrastructure.output.websocket import WebSocketOutputSink


def test_websocket_output_sink_queues_messages():
    sink = WebSocketOutputSink()
    sink.output_tick(1, bar=0, beat=0)
    sink.output_event(MusicalEvent(tick=1, pitch=60, event_type=EventType.NOTE_ON, velocity=100), source="user")
    sink.output_event(MusicalEvent(tick=2, pitch=60, event_type=EventType.NOTE_OFF, velocity=0), source="user")

    msgs = [json.loads(m) for m in sink.get_pending_messages()]
    assert msgs[0]["type"] == "tick"
    assert msgs[1]["type"] == "note" and msgs[1]["event"] == "on"
    assert msgs[2]["type"] == "note" and msgs[2]["event"] == "off"


def test_midi_file_output_sink_records_and_writes(tmp_path):
    out_path = tmp_path / "out.mid"
    cfg = MidiFileOutputConfig(bpm=120.0, ticks_per_beat=4, output_path=str(out_path))
    sink = MidiFileOutputSink(cfg)
    sink.output_event(MusicalEvent(tick=0, pitch=60, event_type=EventType.NOTE_ON, velocity=100), source="user")
    sink.output_event(MusicalEvent(tick=4, pitch=60, event_type=EventType.NOTE_OFF, velocity=0), source="user")
    sink.close()
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_midi_file_output_sink_closes_unmatched_user_and_model_notes_at_final_tick(
    tmp_path,
):
    out_path = tmp_path / "unmatched.mid"
    sink = MidiFileOutputSink(
        MidiFileOutputConfig(
            bpm=120.0,
            ticks_per_beat=4,
            output_path=str(out_path),
        )
    )

    sink.output_event(
        MusicalEvent(tick=1, pitch=60, event_type=EventType.NOTE_ON, velocity=100),
        source="user",
    )
    sink.output_event(
        MusicalEvent(tick=3, pitch=48, event_type=EventType.NOTE_ON, velocity=80),
        source="model",
    )
    sink.output_tick(8, bar=0, beat=2)
    sink.close()

    midi = pretty_midi.PrettyMIDI(str(out_path))
    by_name = {instrument.name: instrument for instrument in midi.instruments}
    user_note = by_name["Melody"].notes[0]
    model_note = by_name["Accompaniment"].notes[0]
    assert user_note.start == pytest.approx(0.125, abs=1e-3)
    assert model_note.start == pytest.approx(0.375, abs=1e-3)
    assert user_note.end == pytest.approx(1.0, abs=1e-3)
    assert model_note.end == pytest.approx(1.0, abs=1e-3)


def test_midi_file_output_sink_gives_last_tick_note_positive_duration(tmp_path):
    out_path = tmp_path / "last_tick.mid"
    sink = MidiFileOutputSink(
        MidiFileOutputConfig(
            bpm=120.0,
            ticks_per_beat=4,
            output_path=str(out_path),
        )
    )

    sink.output_tick(8, bar=0, beat=2)
    sink.output_event(
        MusicalEvent(tick=8, pitch=60, event_type=EventType.NOTE_ON, velocity=100),
        source="user",
    )
    sink.close()

    note = pretty_midi.PrettyMIDI(str(out_path)).instruments[0].notes[0]
    assert note.end - note.start == pytest.approx(0.125, abs=1e-3)


def test_midi_file_output_sink_can_omit_unmatched_notes(tmp_path):
    out_path = tmp_path / "unmatched_disabled.mid"
    sink = MidiFileOutputSink(
        MidiFileOutputConfig(
            bpm=120.0,
            ticks_per_beat=4,
            output_path=str(out_path),
            close_active_notes_on_finalize=False,
        )
    )

    sink.output_event(
        MusicalEvent(tick=0, pitch=60, event_type=EventType.NOTE_ON, velocity=100),
        source="user",
    )
    sink.output_event(
        MusicalEvent(tick=2, pitch=48, event_type=EventType.NOTE_ON, velocity=80),
        source="model",
    )
    sink.output_tick(8, bar=0, beat=2)
    sink.close()

    midi = pretty_midi.PrettyMIDI(str(out_path))
    assert sum(len(instrument.notes) for instrument in midi.instruments) == 0
    assert sink._active_user == {}
    assert sink._active_model == {}


def test_midi_file_output_sink_preserves_paired_notes_and_idempotent_close(tmp_path):
    out_path = tmp_path / "paired.mid"
    sink = MidiFileOutputSink(
        MidiFileOutputConfig(
            bpm=120.0,
            ticks_per_beat=4,
            output_path=str(out_path),
        )
    )

    sink.output_event(
        MusicalEvent(tick=2, pitch=60, event_type=EventType.NOTE_ON, velocity=100),
        source="user",
    )
    sink.output_event(
        MusicalEvent(tick=6, pitch=60, event_type=EventType.NOTE_OFF, velocity=0),
        source="user",
    )
    sink.output_tick(12, bar=0, beat=3)
    sink.close()
    sink.close()

    notes = pretty_midi.PrettyMIDI(str(out_path)).instruments[0].notes
    assert len(notes) == 1
    assert notes[0].start == pytest.approx(0.25, abs=1e-3)
    assert notes[0].end == pytest.approx(0.75, abs=1e-3)


def test_midi_file_output_sink_preserves_retrigger_behavior(tmp_path):
    out_path = tmp_path / "retrigger.mid"
    sink = MidiFileOutputSink(
        MidiFileOutputConfig(
            bpm=120.0,
            ticks_per_beat=4,
            output_path=str(out_path),
        )
    )

    sink.output_event(
        MusicalEvent(tick=0, pitch=60, event_type=EventType.NOTE_ON, velocity=100),
        source="model",
    )
    sink.output_event(
        MusicalEvent(tick=4, pitch=60, event_type=EventType.NOTE_ON, velocity=80),
        source="model",
    )
    sink.output_event(
        MusicalEvent(tick=8, pitch=60, event_type=EventType.NOTE_OFF, velocity=0),
        source="model",
    )
    sink.close()

    notes = pretty_midi.PrettyMIDI(str(out_path)).instruments[0].notes
    assert [(note.start, note.end, note.velocity) for note in notes] == pytest.approx(
        [(0.0, 0.5, 100), (0.5, 1.0, 80)],
        abs=1e-3,
    )


def test_midi_file_output_sink_records_metronome_when_enabled(tmp_path):
    out_path = tmp_path / "metronome.mid"
    cfg = MidiFileOutputConfig(
        bpm=120.0,
        ticks_per_beat=4,
        beats_per_bar=4,
        output_path=str(out_path),
        record_metronome=True,
    )
    sink = MidiFileOutputSink(cfg)

    sink.output_metronome_tick(0, bar=0, beat=0)
    sink.output_metronome_tick(1, bar=0, beat=0)
    sink.output_metronome_tick(4, bar=0, beat=1)
    sink.close()

    midi = pretty_midi.PrettyMIDI(str(out_path))
    by_name = {inst.name: inst for inst in midi.instruments}
    assert "Metronome" in by_name
    assert [note.pitch for note in by_name["Metronome"].notes] == [76, 77]
    assert [note.velocity for note in by_name["Metronome"].notes] == [110, 80]
    assert all(note.end > note.start for note in by_name["Metronome"].notes)


def test_midi_file_output_sink_records_count_in_before_music(tmp_path):
    out_path = tmp_path / "count_in.mid"
    cfg = MidiFileOutputConfig(
        bpm=120.0,
        ticks_per_beat=4,
        beats_per_bar=4,
        output_path=str(out_path),
        record_metronome=True,
    )
    sink = MidiFileOutputSink(cfg)

    sink.output_metronome_tick(-4, bar=0, beat=0)
    sink.output_metronome_tick(0, bar=0, beat=0)
    sink.output_event(MusicalEvent(tick=0, pitch=60, event_type=EventType.NOTE_ON, velocity=100), source="user")
    sink.output_event(MusicalEvent(tick=4, pitch=60, event_type=EventType.NOTE_OFF, velocity=0), source="user")
    sink.close()

    midi = pretty_midi.PrettyMIDI(str(out_path))
    by_name = {inst.name: inst for inst in midi.instruments}
    assert [round(note.start, 3) for note in by_name["Metronome"].notes] == [0.0, 0.5]
    assert len(by_name["Melody"].notes) == 1
    assert round(by_name["Melody"].notes[0].start, 3) == 0.5
    assert round(by_name["Melody"].notes[0].end, 3) == 1.0


def test_composite_output_sink_fans_out_calls():
    calls = []

    class FakeSink:
        def output_event(self, event, source):
            calls.append(("event", source, event.pitch))

        def output_tick(self, tick, bar, beat):
            calls.append(("tick", tick))

        def output_stats(self, **kwargs):
            calls.append(("stats",))

        def output_status(self, state, message=""):
            calls.append(("status", state))

        def output_config(self, config):
            calls.append(("config",))

        def close(self):
            calls.append(("close",))

    s = CompositeOutputSink([FakeSink(), FakeSink()])
    s.output_tick(1, 0, 0)
    s.output_event(MusicalEvent(tick=0, pitch=60, event_type=EventType.NOTE_ON, velocity=100), "user")
    s.close()
    assert calls.count(("tick", 1)) == 2
    assert calls.count(("event", "user", 60)) == 2
    assert calls.count(("close",)) == 2


def test_composite_output_sink_fans_out_metronome_ticks():
    calls = []

    class PlainSink:
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

        def close(self):
            return None

    class MetronomeAwareSink(PlainSink):
        def output_metronome_tick(self, tick, bar, beat):
            calls.append(("metronome", tick, bar, beat))

    s = CompositeOutputSink([PlainSink(), MetronomeAwareSink()])
    s.output_metronome_tick(4, 0, 1)
    assert calls == [("metronome", 4, 0, 1)]


def test_metronome_output_sink_clicks_on_beat_boundaries(monkeypatch):
    messages = []

    class FakePort:
        def send(self, message):
            messages.append(message)

        def close(self):
            messages.append("closed")

    fake_port = FakePort()
    monkeypatch.setattr(mido, "open_output", lambda port_name=None: fake_port)

    sink = MetronomeOutputSink(
        MetronomeOutputConfig(
            port_name="metronome",
            ticks_per_beat=4,
            beats_per_bar=4,
            channel=9,
            beat_note=77,
            downbeat_note=76,
            velocity=80,
            downbeat_velocity=110,
        )
    )

    sink.output_metronome_tick(0, bar=0, beat=0)
    sink.output_metronome_tick(1, bar=0, beat=0)
    sink.output_metronome_tick(4, bar=0, beat=1)
    sink.close()

    midi_messages = [message for message in messages if isinstance(message, mido.Message)]
    assert [(m.type, m.note, m.velocity, m.channel) for m in midi_messages] == [
        ("note_on", 76, 110, 9),
        ("note_off", 76, 0, 9),
        ("note_on", 77, 80, 9),
        ("note_off", 77, 0, 9),
    ]


def test_metronome_output_sink_disables_live_click_when_port_unavailable(monkeypatch, capsys):
    open_attempts = []

    def fail_open_output(port_name=None):
        open_attempts.append(port_name)
        raise RuntimeError("no alsa sequencer")

    monkeypatch.setattr(mido, "open_output", fail_open_output)

    sink = MetronomeOutputSink(MetronomeOutputConfig(port_name=None))

    sink.output_metronome_tick(0, bar=0, beat=0)
    sink.output_metronome_tick(4, bar=0, beat=1)
    sink.close()

    captured = capsys.readouterr()
    assert open_attempts == [None]
    assert "realtime MIDI output unavailable" in captured.out
    assert "no alsa sequencer" in captured.out


def test_json_logger_writes_log_detail_marker(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    sink = JsonLoggerOutputSink(tmp_path, inference_log_detail="full")
    sink.log_inference(
        request={"timestamp": 1.0, "melody_notes": [{"pitch": 60}]},
        response={"timestamp": 2.0, "accompaniment": [{"pitch": 48}]},
        latency_ms=10.0,
        server_process_ms=5.0,
    )
    sink.close()

    data = json.loads((tmp_path / "inferences.json").read_text())
    assert data[0]["request_data"]["log_detail"] == "full"
    assert data[0]["response_data"]["accompaniment"][0]["pitch"] == 48


def test_composite_inference_log_detail_prefers_first_supporting_sink(tmp_path):
    summary_dir = tmp_path / "summary"
    full_dir = tmp_path / "full"
    summary_dir.mkdir(parents=True, exist_ok=True)
    full_dir.mkdir(parents=True, exist_ok=True)
    summary_sink = JsonLoggerOutputSink(summary_dir, inference_log_detail="summary")
    full_sink = JsonLoggerOutputSink(full_dir, inference_log_detail="full")
    composite = CompositeOutputSink([summary_sink, full_sink])
    assert composite.inference_log_detail == "summary"


def test_session_logger_close_auto_saves_metrics_after_output_config(tmp_path):
    sink = SessionLoggerOutputSink(
        session_dir=tmp_path,
        include_midi=False,
        include_json=True,
    )
    sink.output_config({"output_type": "session", "generation_length_frames": 20})
    sink.close()

    assert (tmp_path / "performance.json").exists()
    assert (tmp_path / "statistics.csv").exists()


def test_composite_close_auto_saves_session_metrics(tmp_path):
    class _NoopSink:
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

        def close(self):
            return None

    session_sink = SessionLoggerOutputSink(
        session_dir=tmp_path,
        include_midi=False,
        include_json=True,
    )
    composite = CompositeOutputSink([_NoopSink(), session_sink])
    composite.output_config({"output_type": "composite", "generation_length_frames": 20})
    composite.close()

    assert (tmp_path / "performance.json").exists()
    assert (tmp_path / "statistics.csv").exists()


def test_session_logger_writes_model_schedule_trace_and_theoretical_midi(tmp_path):
    sink = SessionLoggerOutputSink(
        session_dir=tmp_path,
        include_midi=True,
        include_json=False,
        bpm=120.0,
        ticks_per_beat=4,
        artifact_tier="normal",
    )
    sink.output_event(MusicalEvent(tick=2, pitch=60, event_type=EventType.NOTE_ON, velocity=80), "model")
    sink.output_event(MusicalEvent(tick=4, pitch=60, event_type=EventType.NOTE_OFF, velocity=0), "model")
    sink.log_model_schedule(
        [
            {
                "type": "note_on",
                "pitch": 60,
                "channel": 0,
                "program": 0,
                "velocity": 80,
                "logical_tick": 0,
                "scheduled_tick": 2,
                "policy": "clamped_partial_note",
                "action": "scheduled",
            },
            {
                "type": "note_off",
                "pitch": 60,
                "channel": 0,
                "program": 0,
                "velocity": 0,
                "logical_tick": 4,
                "scheduled_tick": 4,
                "policy": "clamped_partial_note_off",
                "action": "scheduled",
            },
        ]
    )

    sink.close()

    assert (tmp_path / "combined.mid").exists()
    assert (tmp_path / "model_schedule_trace.jsonl").exists()
    assert (tmp_path / "theoretical_model.mid").exists()
    assert not (tmp_path / "events.jsonl").exists()
    assert not (tmp_path / "inferences.json").exists()

    theoretical = pretty_midi.PrettyMIDI(str(tmp_path / "theoretical_model.mid"))
    by_name = {inst.name: inst for inst in theoretical.instruments}
    note = by_name["Theoretical Accompaniment"].notes[0]
    assert round(note.start, 3) == 0.0
    assert round(note.end, 3) == 0.5


def test_session_logger_writes_valid_empty_theoretical_midi_and_summary(tmp_path):
    sink = SessionLoggerOutputSink(
        session_dir=tmp_path,
        include_midi=True,
        include_json=True,
        bpm=120.0,
        ticks_per_beat=4,
    )
    sink.log_request_lifecycle({"event": "succeeded", "empty_success": True})
    sink.finalize_validity(
        {
            "content": {"valid": True, "empty_success": True},
            "operational": {"valid": True},
        }
    )
    sink.close()

    theoretical_path = tmp_path / "theoretical_model.mid"
    assert theoretical_path.exists()
    theoretical = pretty_midi.PrettyMIDI(str(theoretical_path))
    assert sum(len(instrument.notes) for instrument in theoretical.instruments) == 0
    summary = json.loads((tmp_path / "theoretical_model_summary.json").read_text())
    assert summary["schema_version"] == 2
    assert summary["export_status"] == "success"
    assert summary["event_count"] == 0
    assert summary["note_count"] == 0
    assert summary["empty"] is True
    assert (tmp_path / "model_schedule_trace.jsonl").read_bytes() == b""
    assert json.loads((tmp_path / "validity.json").read_text())["content"]["valid"] is True
    lifecycle = json.loads((tmp_path / "request_lifecycle.jsonl").read_text())
    assert lifecycle["empty_success"] is True


def test_session_logger_theoretical_export_failure_is_not_silently_swallowed(tmp_path):
    sink = SessionLoggerOutputSink(
        session_dir=tmp_path,
        include_midi=True,
        include_json=True,
        bpm=120.0,
        ticks_per_beat=4,
    )
    sink.log_model_schedule(
        [
            {
                "type": "note_on",
                "pitch": 60,
                "logical_tick": 0,
                "scheduled_tick": 0,
                "action": "scheduled",
                "policy": "future_note",
            }
        ]
    )
    with (tmp_path / "model_schedule_trace.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("{not-json}\n")

    with pytest.raises(ValueError, match="invalid model schedule trace JSON"):
        sink.close()

    # The JSON sink is still closed so the failed attempt keeps its diagnostic
    # evidence, but no success summary can be mistaken for a valid export.
    assert (tmp_path / "inferences.json").is_file()
    assert not (tmp_path / "theoretical_model_summary.json").exists()


def test_composite_output_sink_fans_out_model_schedule_trace(tmp_path):
    rows = [{"type": "note_on", "pitch": 60, "logical_tick": 0, "scheduled_tick": 0}]
    session_sink = SessionLoggerOutputSink(
        session_dir=tmp_path,
        include_midi=False,
        include_json=False,
    )
    composite = CompositeOutputSink([session_sink])

    composite.log_model_schedule(rows)
    composite.close()

    assert json.loads((tmp_path / "model_schedule_trace.jsonl").read_text().splitlines()[0])["pitch"] == 60


def test_session_logger_writes_system_trace_jsonl(tmp_path):
    sink = SessionLoggerOutputSink(
        session_dir=tmp_path,
        include_midi=False,
        include_json=False,
    )

    sink.log_system_trace(
        {
            "schema_version": 1,
            "mode": "realtime",
            "condition": "standard",
            "clock_domain": "service_now",
            "tick": 4,
            "nominal_tick_time_s": 1.0,
            "deadline_time_s": 1.0125,
            "decision": "note",
            "arrived_by_deadline": True,
            "arrival_time_s": 1.01,
            "logical_tick": 4,
            "scheduled_tick": 4,
            "generation_start_tick": 4,
            "request_id": "req-1",
            "action": "scheduled",
            "policy": "future_note",
            "emitted_model_note_on_count": 1,
            "explicit_rest": False,
        }
    )
    sink.close()

    rows = [
        json.loads(line)
        for line in (tmp_path / "system_trace.jsonl").read_text().splitlines()
    ]
    assert rows == [
        {
            "schema_version": 1,
            "mode": "realtime",
            "condition": "standard",
            "clock_domain": "service_now",
            "tick": 4,
            "nominal_tick_time_s": 1.0,
            "deadline_time_s": 1.0125,
            "decision": "note",
            "arrived_by_deadline": True,
            "arrival_time_s": 1.01,
            "logical_tick": 4,
            "scheduled_tick": 4,
            "generation_start_tick": 4,
            "request_id": "req-1",
            "action": "scheduled",
            "policy": "future_note",
            "emitted_model_note_on_count": 1,
            "explicit_rest": False,
        }
    ]


def test_composite_output_sink_fans_out_system_trace_and_skips_plain_sinks(tmp_path):
    class PlainSink:
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

        def close(self):
            return None

    session_sink = SessionLoggerOutputSink(
        session_dir=tmp_path,
        include_midi=False,
        include_json=False,
    )
    composite = CompositeOutputSink([PlainSink(), session_sink])

    composite.log_system_trace({"tick": 2, "decision": "missing"})
    composite.close()

    row = json.loads((tmp_path / "system_trace.jsonl").read_text())
    assert row == {"decision": "missing", "tick": 2}


def test_composite_output_sink_fans_out_lifecycle_and_validity(tmp_path):
    session_sink = SessionLoggerOutputSink(
        session_dir=tmp_path,
        include_midi=False,
        include_json=True,
    )
    composite = CompositeOutputSink([session_sink])

    composite.log_request_lifecycle({"event": "expected", "request_id": "req-1"})
    composite.finalize_validity({"content": {"valid": False}})
    composite.close()

    row = json.loads((tmp_path / "request_lifecycle.jsonl").read_text())
    assert row["request_id"] == "req-1"
    assert json.loads((tmp_path / "validity.json").read_text())["content"]["valid"] is False


def test_session_logger_keeps_core_validity_artifacts_without_debug_json(tmp_path):
    sink = SessionLoggerOutputSink(
        session_dir=tmp_path,
        include_midi=False,
        include_json=False,
        artifact_tier="normal",
    )

    sink.log_request_lifecycle({"event": "drain_completed"})
    sink.finalize_validity({"content": {"valid": True}})
    sink.close()

    assert json.loads((tmp_path / "request_lifecycle.jsonl").read_text())["event"] == "drain_completed"
    assert json.loads((tmp_path / "validity.json").read_text())["content"]["valid"] is True
