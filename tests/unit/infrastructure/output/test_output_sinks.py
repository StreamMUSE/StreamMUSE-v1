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
