"""MIDI file simulation input adapter implementing InputSource."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict, Iterator, List, Optional, Tuple

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
    trim_leading_rest: bool = False

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

        Uses pretty_midi (same pipeline as NPZ generation) so that note
        quantization is identical to what the model was trained on.

        Returns (notes, resolution, actual_max_tick).
        Each note dict has: pitch, tick, duration.
        """
        from streammuse.infrastructure.inference.lekai_model.MidiConverter import MidiConverter

        converter = MidiConverter(ticks_per_beat=beat_div)
        pm, _meta = converter.load_midi(midi_path)
        if pm is None:
            return [], beat_div, 0

        raw_notes, actual_max_tick = converter.midi_to_notes(pm, filter_drums=True)

        notes: List[Dict[str, int]] = []
        for n in raw_notes:
            if not (min_pitch <= n["pitch"] <= max_pitch):
                continue
            if program is not None and n.get("program") != program:
                continue
            notes.append({"pitch": n["pitch"], "tick": n["tick"], "duration": n["duration"]})

        notes.sort(key=lambda n: (n["tick"], n["pitch"]))
        if max_tick is not None:
            notes = [n for n in notes if n["tick"] < max_tick]
            actual_max_tick = min(actual_max_tick, int(max_tick))
        return notes, beat_div, int(actual_max_tick)

    def read_events(self) -> Iterator[MusicalEvent]:
        # Anchor playback before parsing so file-conversion latency cannot shift
        # the entire simulated performance relative to the service timeline.
        start_time = self._now()
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
        configured_start_tick = int(self._config.start_tick)
        first_note_tick = min((int(note["tick"]) for note in notes), default=0)
        start_tick = (
            max(configured_start_tick, first_note_tick)
            if self._config.trim_leading_rest
            else configured_start_tick
        )
        start_offset = int(self._config.delay_ticks)
        effective_notes = [n for n in notes if int(n["tick"]) >= start_tick]

        for n in effective_notes:
            relative_tick = (
                int(n["tick"]) - start_tick
                if self._config.trim_leading_rest
                else int(n["tick"])
            )
            onset = relative_tick + start_offset
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
