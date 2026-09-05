import mido
import pytest

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


def test_midi_file_input_start_tick_skips_earlier_notes(tmp_path):
    mid = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    mid.tracks.append(track)

    # Note A: output tick 0
    track.append(mido.Message("note_on", note=60, velocity=64, time=0))
    track.append(mido.Message("note_off", note=60, velocity=0, time=120))
    # Note B: output tick 16 (1920 / 120)
    track.append(mido.Message("note_on", note=62, velocity=64, time=1800))
    track.append(mido.Message("note_off", note=62, velocity=0, time=120))

    path = tmp_path / "two_notes.mid"
    mid.save(str(path))

    sleeps = []

    def fake_sleep(dt: float) -> None:
        sleeps.append(dt)

    cfg = MidiFileInputConfig(
        bpm=120.0,
        ticks_per_beat=4,
        delay_ticks=0,
        start_tick=16,
    )
    src = MidiFileInput(str(path), config=cfg, now=lambda: 0.0, sleep=fake_sleep)

    events = list(src.read_events())

    assert [e.pitch for e in events] == [62, 62]
    assert [e.event_type for e in events] == [EventType.NOTE_ON, EventType.NOTE_OFF]


def test_midi_file_input_start_tick_uses_absolute_timing(tmp_path):
    mid = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    mid.tracks.append(track)

    # Output tick 16 note
    track.append(mido.Message("note_on", note=64, velocity=64, time=1920))
    track.append(mido.Message("note_off", note=64, velocity=0, time=120))

    path = tmp_path / "start16.mid"
    mid.save(str(path))

    sleeps = []

    def fake_sleep(dt: float) -> None:
        sleeps.append(dt)

    cfg = MidiFileInputConfig(
        bpm=120.0,
        ticks_per_beat=4,
        start_tick=16,
    )
    src = MidiFileInput(str(path), config=cfg, now=lambda: 0.0, sleep=fake_sleep)
    _ = list(src.read_events())

    assert sleeps
    assert sleeps[0] == pytest.approx(16 * cfg.seconds_per_tick())


def test_midi_file_input_parsing_time_does_not_shift_schedule(tmp_path, monkeypatch):
    path = tmp_path / "synthetic.mid"
    path.write_bytes(b"placeholder")

    clock = [0.0]
    sleeps = []
    cfg = MidiFileInputConfig(bpm=120.0, ticks_per_beat=4)
    src = MidiFileInput(
        str(path),
        config=cfg,
        now=lambda: clock[0],
        sleep=lambda delay: sleeps.append(delay),
    )

    def _parse(*args, **kwargs):
        _ = args, kwargs
        clock[0] = 0.2
        return ([{"pitch": 60, "tick": 4, "duration": 1}], 4, 5)

    monkeypatch.setattr(src, "_midi_to_notes", _parse)

    _ = list(src.read_events())

    assert sleeps[0] == pytest.approx(4 * cfg.seconds_per_tick() - 0.2)


def test_midi_file_input_trim_leading_rest_preserves_preparse_clock_anchor(tmp_path):
    mid = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.Message("note_on", note=64, velocity=64, time=1920))
    track.append(mido.Message("note_off", note=64, velocity=0, time=120))

    path = tmp_path / "leading_rest.mid"
    mid.save(str(path))
    sleeps = []
    cfg = MidiFileInputConfig(
        bpm=120.0,
        ticks_per_beat=4,
        trim_leading_rest=True,
    )
    src = MidiFileInput(
        str(path),
        config=cfg,
        now=lambda: 0.0,
        sleep=lambda delay: sleeps.append(delay),
    )

    events = list(src.read_events())

    assert [event.pitch for event in events] == [64, 64]
    assert sleeps
    assert sleeps[0] == pytest.approx(1 * cfg.seconds_per_tick())


def test_midi_file_source_tick_replay_returns_only_the_requested_tick(
    tmp_path, monkeypatch
):
    path = tmp_path / "source_ticks.mid"
    path.write_bytes(b"placeholder")
    parse_calls = []
    cfg = MidiFileInputConfig(bpm=120.0, ticks_per_beat=4)
    src = MidiFileInput(
        str(path),
        config=cfg,
        now=lambda: (_ for _ in ()).throw(AssertionError("wall clock used")),
        sleep=lambda _delay: (_ for _ in ()).throw(AssertionError("sleep used")),
    )

    def _parse(*args, **kwargs):
        parse_calls.append((args, kwargs))
        return (
            [
                {"pitch": 60, "tick": 4, "duration": 2},
                {"pitch": 62, "tick": 5, "duration": 1},
            ],
            4,
            6,
        )

    monkeypatch.setattr(src, "_midi_to_notes", _parse)

    src.prepare_source_tick_replay()

    assert src.read_events_at_tick(3) == []
    assert [
        (event.event_type, event.pitch, event.tick)
        for event in src.read_events_at_tick(4)
    ] == [(EventType.NOTE_ON, 60, 4)]
    assert [
        (event.event_type, event.pitch, event.tick)
        for event in src.read_events_at_tick(5)
    ] == [(EventType.NOTE_ON, 62, 5)]
    assert [
        (event.event_type, event.pitch, event.tick)
        for event in src.read_events_at_tick(6)
    ] == [
        (EventType.NOTE_OFF, 60, 6),
        (EventType.NOTE_OFF, 62, 6),
    ]
    assert src.read_events_at_tick(4) == []
    assert len(parse_calls) == 1
