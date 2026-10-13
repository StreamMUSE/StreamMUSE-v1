from pathlib import Path
from queue import Queue
import sys
from threading import Lock
from types import SimpleNamespace

import mido
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from fixed_tick_midi_replay import FixedTickInput, load_midi_events


def make_midi(path, first_time=55):
    midi = mido.MidiFile(ticks_per_beat=220)
    midi.tracks = [mido.MidiTrack([mido.MetaMessage("set_tempo", tempo=666667)]),
                   mido.MidiTrack([
                       mido.MetaMessage("track_name", name="Replay Melody"),
                       mido.Message("note_on", note=65, velocity=93, channel=2, time=first_time),
                       mido.Message("note_off", note=65, velocity=12, channel=2, time=0),
                       mido.Message("note_on", note=65, velocity=71, channel=2, time=0),
                       mido.Message("note_off", note=65, channel=2, time=55),
                   ])]
    midi.save(path)


def test_native_ticks_preserve_order_velocity_and_zero_duration(tmp_path):
    path = tmp_path / "source.mid"
    make_midi(path)
    events, metadata = load_midi_events(path)
    assert metadata["midi_ppq"] == 220
    assert [e["tick"] for e in events] == [1, 1, 1, 2]
    assert [e["event_type"] for e in events] == ["note_on", "note_off", "note_on", "note_off"]
    assert [e["velocity"] for e in events] == [93, 12, 71, 64]
    assert [e["channel"] for e in events] == [2] * 4


def test_off_grid_midi_is_rejected_not_rounded(tmp_path):
    path = tmp_path / "offgrid.mid"
    make_midi(path, first_time=1)
    with pytest.raises(ValueError, match="Off-grid"):
        load_midi_events(path)


def test_ambiguous_note_tracks_are_rejected(tmp_path):
    path = tmp_path / "two_tracks.mid"
    make_midi(path)
    midi = mido.MidiFile(path)
    midi.tracks.append(midi.tracks[1].copy())
    midi.save(path)
    with pytest.raises(ValueError, match="exactly one"):
        load_midi_events(path)


def test_observer_does_not_expose_future_events(tmp_path):
    path = tmp_path / "source.mid"
    make_midi(path)
    events, _ = load_midi_events(path)
    observer = FixedTickInput(events)
    observer.service = SimpleNamespace(_event_q=Queue(), _melody_history_lock=Lock(), _melody_history=[])
    observer.start()
    observer.on_tick(0)
    assert observer.service._event_q.empty()
    observer.on_tick(1)
    assert observer.service._event_q.qsize() == 3
    assert [e.tick for e in observer.service._melody_history] == [1, 1, 1]
    observer.on_tick(2)
    assert observer.service._event_q.qsize() == 4
    assert [e["injected_at_logical_tick"] for e in observer.delivered] == [1, 1, 1, 2]


def test_observer_rejects_skipped_tick(tmp_path):
    path = tmp_path / "source.mid"
    make_midi(path)
    observer = FixedTickInput(load_midi_events(path)[0])
    observer.service = SimpleNamespace(_event_q=Queue(), _melody_history_lock=Lock(), _melody_history=[])
    observer.on_tick(0)
    with pytest.raises(ValueError, match="Nonconsecutive"):
        observer.on_tick(2)
    assert observer.error is not None


def test_existing_service_preserves_fixed_ticks_and_beat_request_boundaries():
    from streammuse.application.services.real_time_music_service import RealTimeMusicService
    from streammuse.domain.timing import PlaybackScheduler, Tempo
    from tests.unit.application.fakes import NoopInference, NoopInput, NoopOutput
    events = [{"sequence": 1, "tick": 1, "pitch": 65, "event_type": "note_on", "velocity": 91, "channel": 0},
              {"sequence": 2, "tick": 4, "pitch": 65, "event_type": "note_off", "velocity": 0, "channel": 0}]
    observer = FixedTickInput(events)
    service = RealTimeMusicService(input_source=NoopInput(), inference_engine=NoopInference(),
        output_sink=NoopOutput(), tempo=Tempo(90, 4, 4), scheduler=PlaybackScheduler(),
        generation_interval_ticks=4, generation_length_frames=4, tick_observer=observer,
        now=lambda: 1000.0, sleep=lambda seconds: None)
    observer.service = service
    service._runtime = SimpleNamespace(session_start_time=0.0)
    service._running = True
    service._tick_loop(max_ticks=8)
    first_tick, first_events = service._inference_request_queue.get_nowait()
    second_tick, second_events = service._inference_request_queue.get_nowait()
    assert first_tick == 4 and [(e.tick, e.event_type.value) for e in first_events] == [(1, "note_on")]
    assert second_tick == 8 and [(e.tick, e.event_type.value) for e in second_events] == [(4, "note_off")]


@pytest.mark.parametrize("continue_on_difference,expected_pieces,expected_status", [
    (False, 1, "stopped_on_difference"),
    (True, 2, "completed_with_differences"),
])
def test_mismatch_continuation_is_explicit_and_never_retries(
        tmp_path, monkeypatch, continue_on_difference, expected_pieces, expected_status):
    import json
    import fixed_tick_midi_replay as replay

    replay.audit.save(tmp_path / "plan.json", {"seed": 12, "cases": [{"piece": "a"}, {"piece": "b"}]})
    launched = []
    monkeypatch.setattr(replay.subprocess, "run", lambda command, **kwargs: launched.append(command))
    monkeypatch.setattr(replay, "analyze_piece", lambda root, piece: {
        "delivered_events_exact": True, "backend_exact": piece != "a",
        "sampler_exact": piece != "a", "playback_exact": True,
    })
    replay.run(SimpleNamespace(root=tmp_path, pieces=["a", "b"], server="http://127.0.0.1:18846",
                               continue_on_difference=continue_on_difference))
    progress = json.loads((tmp_path / "progress.json").read_text())
    assert progress["status"] == expected_status
    assert len(progress["pieces"]) == expected_pieces
    assert len(launched) == expected_pieces * 2
    assert [cmd[cmd.index("--repeat") + 1] for cmd in launched] == ["1", "2"] * expected_pieces
