"""MIDI file simulation input adapter implementing InputSource."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict, Iterator, List, Optional, Tuple

import mido

from streammuse.domain.musical import EventType, MusicalEvent


@dataclass(frozen=True)
class MidiFileInputConfig:
    bpm: float
    ticks_per_beat: int
    delay_ticks: int = 0
    min_pitch: int = 0
    max_pitch: int = 127
    program: Optional[int] = None
    max_tick: Optional[int] = None
    start_tick: int = 0

    def seconds_per_tick(self) -> float:
        return (60.0 / float(self.bpm)) / float(self.ticks_per_beat)


class MidiFileInput:
    """
    InputSource that simulates real-time input from a MIDI file.

    Notes are parsed from the file, then emitted as note_on/note_off events in
    real-time based on the configured tempo.

    Emits `MusicalEvent` with `tick=0`; the application layer should assign tick
    from timestamps (tempo.seconds_to_tick(elapsed)).
    """

    def __init__(
        self,
        midi_file_path: str,
        *,
        config: MidiFileInputConfig,
        velocity_default: int = 64,
        now: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._path = midi_file_path
        self._config = config
        self._velocity_default = int(velocity_default)
        self._now = now
        self._sleep = sleep
        self._closed = False

    @staticmethod
    def _midi_to_notes(
        midi_path: str,
        *,
        beat_div: int,
        min_pitch: int,
        max_pitch: int,
        program: Optional[int],
        max_tick: Optional[int],
    ) -> Tuple[List[Dict[str, int]], int, int]:
        """
        Read a MIDI file and convert to a note list in `beat_div` ticks/beat.

        Returns (notes, resolution, actual_max_tick).
        Each note dict has: pitch, tick, duration.
        """
        midi_file = mido.MidiFile(midi_path)
        resolution = int(midi_file.ticks_per_beat)
        ticks_per_output_tick = resolution / float(beat_div)

        notes: List[Dict[str, int]] = []
        active: Dict[Tuple[int, int], int] = {}

        for track in midi_file.tracks:
            current_tick = 0
            current_program = 0
            for msg in track:
                current_tick += int(msg.time)

                if msg.type == "program_change":
                    current_program = int(msg.program)
                    continue

                if msg.type == "note_on" and int(msg.velocity) > 0:
                    if program is not None and current_program != program:
                        continue
                    if min_pitch <= int(msg.note) <= max_pitch:
                        output_tick = int(round(current_tick / ticks_per_output_tick))
                        active[(int(msg.channel), int(msg.note))] = output_tick
                    continue

                if msg.type == "note_off" or (msg.type == "note_on" and int(msg.velocity) == 0):
                    key = (int(msg.channel), int(msg.note))
                    if key not in active:
                        continue
                    start_tick = active.pop(key)
                    output_tick = int(round(current_tick / ticks_per_output_tick))
                    duration = max(1, output_tick - start_tick)
                    notes.append(
                        {"pitch": int(msg.note), "tick": int(start_tick), "duration": int(duration)}
                    )

        notes.sort(key=lambda n: (n["tick"], n["pitch"]))
        actual_max_tick = max((n["tick"] + n["duration"] for n in notes), default=0)
        if max_tick is not None:
            notes = [n for n in notes if n["tick"] < max_tick]
            actual_max_tick = min(actual_max_tick, int(max_tick))
        return notes, resolution, int(actual_max_tick)

    def read_events(self) -> Iterator[MusicalEvent]:
        seconds_per_tick = self._config.seconds_per_tick()
        notes, _resolution, _max_tick = self._midi_to_notes(
            self._path,
            beat_div=self._config.ticks_per_beat,
            min_pitch=self._config.min_pitch,
            max_pitch=self._config.max_pitch,
            program=self._config.program,
            max_tick=self._config.max_tick,
        )

        # Build schedule: tick -> list[MusicalEvent]
        schedule: Dict[int, List[MusicalEvent]] = {}
        start_tick = int(self._config.start_tick)
        start_offset = int(self._config.delay_ticks)
        effective_notes = [n for n in notes if int(n["tick"]) >= start_tick]

        for n in effective_notes:
            onset = int(n["tick"]) + start_offset
            offset = onset + int(n["duration"])

            schedule.setdefault(onset, []).append(
                MusicalEvent(
                    tick=0,
                    pitch=int(n["pitch"]),
                    event_type=EventType.NOTE_ON,
                    velocity=self._velocity_default,
                )
            )
            schedule.setdefault(offset, []).append(
                MusicalEvent(
                    tick=0,
                    pitch=int(n["pitch"]),
                    event_type=EventType.NOTE_OFF,
                    velocity=0,
                )
            )

        ticks = sorted(schedule.keys())
        start_time = self._now()
        for t in ticks:
            if self._closed:
                break
            target_time = start_time + (t * seconds_per_tick)
            delay = target_time - self._now()
            if delay > 0:
                self._sleep(delay)
            for ev in schedule.get(t, []):
                if self._closed:
                    break
                yield ev

    def close(self) -> None:
        self._closed = True

