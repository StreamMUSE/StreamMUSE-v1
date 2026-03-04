import mido

from streammuse.domain.musical import EventType
from streammuse.infrastructure.input.midi_file import MidiFileInput, MidiFileInputConfig


def test_midi_file_input_parses_and_emits_events(tmp_path):
    # Build a tiny MIDI file: one note lasting one beat.
    mid = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.Message("note_on", note=60, velocity=64, time=0))
    track.append(mido.Message("note_off", note=60, velocity=0, time=480))

    path = tmp_path / "one_note.mid"
    mid.save(str(path))

    sleeps = []

    def fake_sleep(dt: float) -> None:
        sleeps.append(dt)

    # Application uses ticks_per_beat=4 by default (quarter-beat ticks).
    cfg = MidiFileInputConfig(bpm=120.0, ticks_per_beat=4, delay_ticks=0)
    src = MidiFileInput(str(path), config=cfg, now=lambda: 0.0, sleep=fake_sleep)

    events = list(src.read_events())
    assert [e.event_type for e in events] == [EventType.NOTE_ON, EventType.NOTE_OFF]
    assert [e.pitch for e in events] == [60, 60]
    assert all(e.tick == 0 for e in events)


def test_midi_file_input_delay_ticks_sleeps(tmp_path):
    mid = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.Message("note_on", note=60, velocity=64, time=0))
    track.append(mido.Message("note_off", note=60, velocity=0, time=480))

    path = tmp_path / "one_note.mid"
    mid.save(str(path))

    sleeps = []

    def fake_sleep(dt: float) -> None:
        sleeps.append(dt)

    cfg = MidiFileInputConfig(bpm=120.0, ticks_per_beat=4, delay_ticks=4)
    src = MidiFileInput(str(path), config=cfg, now=lambda: 0.0, sleep=fake_sleep)
    _ = list(src.read_events())

    # First scheduled tick is delay_ticks, so should sleep >= 0 for that.
    assert len(sleeps) >= 1
