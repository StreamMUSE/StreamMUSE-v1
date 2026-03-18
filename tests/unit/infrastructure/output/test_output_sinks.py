import json
from types import SimpleNamespace

import pytest

from streammuse.domain.musical import EventType, MusicalEvent
from streammuse.infrastructure.output.composite import CompositeOutputSink
from streammuse.infrastructure.output.json_logger import JsonLoggerOutputSink
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

