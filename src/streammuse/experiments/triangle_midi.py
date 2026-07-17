"""Deterministic MIDI evidence helpers shared by analysis and listening builds."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import mido

from streammuse.experiments.robustness_metrics import Roll


def _exact_model_tick(raw_tick: int, ticks_per_beat: int) -> int:
    scaled = int(raw_tick) * 4
    tick, remainder = divmod(scaled, int(ticks_per_beat))
    if remainder:
        raise ValueError(
            f"MIDI event at source tick {raw_tick} is off the four-ticks-per-beat grid"
        )
    return tick


def formal_triangle_note_events(
    path: str | Path, *, start_model_tick: int, end_model_tick: int
) -> list[dict[str, int]]:
    """Extract theoretical accompaniment notes with velocity and left-edge state.

    A note active at the window start is emitted at local tick zero with its
    original velocity.  The input must lie exactly on the model's four-tick
    grid; evidence extraction never rounds timing.
    """

    if start_model_tick < 0 or end_model_tick <= start_model_tick:
        raise ValueError("formal excerpt requires a non-empty non-negative window")
    midi = mido.MidiFile(str(path))
    named_tracks: list[tuple[int, mido.MidiTrack]] = []
    note_tracks: list[tuple[int, mido.MidiTrack]] = []
    for track_index, track in enumerate(midi.tracks):
        names = [
            str(message.name)
            for message in track
            if message.type == "track_name"
        ]
        if "Theoretical Accompaniment" in names:
            named_tracks.append((track_index, track))
        if any(message.type in {"note_on", "note_off"} for message in track):
            note_tracks.append((track_index, track))
    if len(named_tracks) == 1:
        track_index, target = named_tracks[0]
        if any(index != track_index for index, _track in note_tracks):
            raise ValueError(
                "theoretical_model.mid contains notes outside the theoretical track"
            )
    elif not named_tracks and not note_tracks:
        return []
    elif not named_tracks and len(note_tracks) == 1:
        track_index, target = note_tracks[0]
    else:
        raise ValueError(
            "theoretical_model.mid must contain exactly one "
            "'Theoretical Accompaniment' note track"
        )

    active: dict[tuple[int, int, int], list[tuple[int, int]]] = defaultdict(list)
    completed: list[tuple[int, int, int, int]] = []
    absolute = 0
    for message in target:
        absolute += int(message.time)
        if message.type not in {"note_on", "note_off"}:
            continue
        channel = int(message.channel)
        if channel == 9:
            raise ValueError("theoretical accompaniment unexpectedly uses the drum channel")
        tick = _exact_model_tick(absolute, midi.ticks_per_beat)
        pitch = int(message.note)
        key = (track_index, channel, pitch)
        if message.type == "note_on" and int(message.velocity) > 0:
            active[key].append((tick, int(message.velocity)))
            continue
        if not active.get(key):
            raise ValueError(
                f"unmatched note-off in theoretical accompaniment at tick {tick}, pitch {pitch}"
            )
        note_start, velocity = active[key].pop(0)
        completed.append((note_start, max(note_start + 1, tick), pitch, velocity))
    for (_track, _channel, pitch), starts in active.items():
        for note_start, velocity in starts:
            completed.append(
                (note_start, max(note_start + 1, end_model_tick), pitch, velocity)
            )

    events = [
        {
            "start_model_tick": max(note_start, start_model_tick) - start_model_tick,
            "end_model_tick": min(note_end, end_model_tick) - start_model_tick,
            "pitch": pitch,
            "velocity": velocity,
        }
        for note_start, note_end, pitch, velocity in completed
        if note_start < end_model_tick and note_end > start_model_tick
    ]
    events = [row for row in events if row["end_model_tick"] > row["start_model_tick"]]
    events.sort(
        key=lambda row: (
            row["start_model_tick"],
            row["pitch"],
            row["end_model_tick"],
            row["velocity"],
        )
    )
    return events


def triangle_note_events_roll(
    events: Sequence[Mapping[str, int]], *, end_model_tick: int
) -> Roll:
    sustain: set[tuple[int, int]] = set()
    onsets: set[tuple[int, int]] = set()
    for event in events:
        start = int(event["start_model_tick"])
        end = int(event["end_model_tick"])
        pitch = int(event["pitch"])
        onsets.add((start, pitch))
        sustain.update((tick, pitch) for tick in range(start, end))
    return Roll(end_model_tick, frozenset(sustain), frozenset(onsets))


def write_triangle_note_event_midi(
    events: Sequence[Mapping[str, int]], path: str | Path, *, bpm: int = 120
) -> None:
    """Write canonical program-0/bank-0 MIDI while preserving note velocity."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    ticks_per_beat = 480
    step = ticks_per_beat // 4
    timeline: list[tuple[int, int, int, int, mido.Message]] = []
    for index, event in enumerate(events):
        start = int(event["start_model_tick"])
        end = int(event["end_model_tick"])
        pitch = int(event["pitch"])
        velocity = int(event["velocity"])
        if start < 0 or end <= start:
            raise ValueError("canonical note event has an invalid interval")
        if not 0 <= pitch <= 127 or not 1 <= velocity <= 127:
            raise ValueError("canonical note event has an invalid pitch or velocity")
        timeline.extend(
            [
                (
                    start * step,
                    1,
                    pitch,
                    index,
                    mido.Message(
                        "note_on", channel=0, note=pitch, velocity=velocity
                    ),
                ),
                (
                    end * step,
                    0,
                    pitch,
                    index,
                    mido.Message("note_off", channel=0, note=pitch, velocity=0),
                ),
            ]
        )
    timeline.sort(key=lambda row: row[:4])
    midi = mido.MidiFile(type=1, ticks_per_beat=ticks_per_beat)
    meta = mido.MidiTrack()
    meta.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm), time=0))
    meta.append(mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0))
    midi.tracks.append(meta)
    track = mido.MidiTrack()
    # No bank-select message means the MIDI-defined default bank zero.
    track.append(mido.Message("program_change", channel=0, program=0, time=0))
    previous = 0
    for absolute, _priority, _pitch, _index, message in timeline:
        message.time = absolute - previous
        track.append(message)
        previous = absolute
    midi.tracks.append(track)
    midi.save(str(destination))


def build_formal_triangle_excerpt(
    theoretical_midi: str | Path,
    destination: str | Path,
    *,
    start_model_tick: int,
    end_model_tick: int,
    bpm: int = 120,
) -> tuple[Roll, list[dict[str, int]]]:
    """Build the exact formal comparator/canonical excerpt used by all stages."""

    events = formal_triangle_note_events(
        theoretical_midi,
        start_model_tick=start_model_tick,
        end_model_tick=end_model_tick,
    )
    roll = triangle_note_events_roll(
        events, end_model_tick=end_model_tick - start_model_tick
    )
    write_triangle_note_event_midi(events, destination, bpm=bpm)
    return roll, events
