"""MIDI file recording OutputSink adapter."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pretty_midi

from streammuse.domain.musical import EventType, MusicalEvent


@dataclass(frozen=True)
class MidiFileOutputConfig:
    bpm: float
    ticks_per_beat: int
    output_path: Optional[str] = None
    user_program: int = 0
    model_program: int = 0
    user_track_name: str = "Melody"
    model_track_name: str = "Accompaniment"

    def seconds_per_tick(self) -> float:
        return (60.0 / float(self.bpm)) / float(self.ticks_per_beat)


class MidiFileOutputSink:
    """
    Records event stream into a MIDI file.

    Notes are reconstructed from note_on/note_off pairs per pitch; if a note is
    retriggered without an intervening note_off, the previous note is closed at
    the retrigger time.
    """

    def __init__(self, config: MidiFileOutputConfig) -> None:
        self._config = config
        self._sp_tick = config.seconds_per_tick()
        self._midi = pretty_midi.PrettyMIDI(initial_tempo=float(config.bpm))

        self._user = pretty_midi.Instrument(program=int(config.user_program), name=config.user_track_name)
        self._model = pretty_midi.Instrument(program=int(config.model_program), name=config.model_track_name)
        self._midi.instruments.append(self._user)
        self._midi.instruments.append(self._model)

        self._active_user: Dict[int, List[Dict[str, float]]] = {}
        self._active_model: Dict[int, List[Dict[str, float]]] = {}
        self._max_time = 0.0

    def _time(self, tick: int) -> float:
        return float(tick) * self._sp_tick

    def _handle_event(
        self,
        event: MusicalEvent,
        *,
        instrument: pretty_midi.Instrument,
        active: Dict[int, List[Dict[str, float]]],
        default_velocity: int,
    ) -> None:
        if event.is_placeholder or event.pitch == -1:
            return
        t = self._time(event.tick)
        self._max_time = max(self._max_time, t)

        pitch = int(event.pitch)
        if event.event_type == EventType.NOTE_ON and event.velocity > 0:
            # MIDI same-pitch retriggers are ambiguous; close any still-open
            # notes at the retrigger time so a missing note_off cannot stretch
            # the previous note to the end of the file.
            if pitch in active:
                for prev in active.pop(pitch):
                    start = float(prev["start"])
                    vel = int(prev["velocity"])
                    if t > start:
                        instrument.notes.append(pretty_midi.Note(velocity=vel, pitch=pitch, start=start, end=t))
            active.setdefault(pitch, []).append(
                {"start": t, "velocity": float(event.velocity or default_velocity)}
            )
            return

        if event.event_type == EventType.NOTE_OFF:
            if pitch not in active or not active[pitch]:
                return
            prev = active[pitch].pop(0)
            if not active[pitch]:
                active.pop(pitch, None)
            start = float(prev["start"])
            vel = int(prev["velocity"])
            if t > start:
                instrument.notes.append(pretty_midi.Note(velocity=vel, pitch=pitch, start=start, end=t))

    def output_event(self, event: MusicalEvent, source: str) -> None:
        if source == "user":
            self._handle_event(event, instrument=self._user, active=self._active_user, default_velocity=100)
        else:
            self._handle_event(event, instrument=self._model, active=self._active_model, default_velocity=80)

    def output_tick(self, tick: int, bar: int, beat: int) -> None:
        return

    def output_stats(
        self,
        hit_rate=None,
        avg_backup_level=None,
        round_trip_ms=None,
        server_process_ms=None,
        network_latency_ms=None,
        total_hits=None,
        total_ticks=None,
    ) -> None:
        return

    def output_status(self, state: str, message: str = "") -> None:
        return

    def output_config(self, config) -> None:
        return

    def _finalize(self) -> None:
        end = self._max_time
        for active, inst in ((self._active_user, self._user), (self._active_model, self._model)):
            for pitch, infos in list(active.items()):
                for info in infos:
                    start = float(info["start"])
                    vel = int(info["velocity"])
                    if end > start:
                        inst.notes.append(pretty_midi.Note(velocity=vel, pitch=int(pitch), start=start, end=end))
                active.pop(pitch, None)

    def close(self) -> None:
        self._finalize()
        if self._config.output_path:
            path = Path(self._config.output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._midi.write(str(path))
