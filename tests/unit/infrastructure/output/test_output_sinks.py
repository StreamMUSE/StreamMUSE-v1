import json
from types import SimpleNamespace

import pytest

from streammuse.domain.musical import EventType, MusicalEvent
from streammuse.infrastructure.output.composite import CompositeOutputSink
from streammuse.infrastructure.output.midi_file import MidiFileOutputConfig, MidiFileOutputSink
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

