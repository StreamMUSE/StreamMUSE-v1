"""MIDI file recording OutputSink adapter."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import pretty_midi

from streammuse.domain.musical import EventType, MusicalEvent


@dataclass(frozen=True)
class MidiFileOutputConfig:
    bpm: float
    ticks_per_beat: int
    beats_per_bar: int = 4
    output_path: Optional[str] = None
    user_program: int = 0
    model_program: int = 0
    user_track_name: str = "Melody"
    model_track_name: str = "Accompaniment"
    record_metronome: bool = False
    metronome_track_name: str = "Metronome"
    metronome_beat_note: int = 77
    metronome_downbeat_note: int = 76
    metronome_velocity: int = 80
    metronome_downbeat_velocity: int = 110
    metronome_duration_ticks: int = 1
    close_active_notes_on_finalize: bool = True

    def seconds_per_tick(self) -> float:
        return (60.0 / float(self.bpm)) / float(self.ticks_per_beat)


class MidiFileOutputSink:
    """
    Records event stream into a MIDI file.

    Notes are reconstructed from note_on/note_off pairs per pitch; if a note is
    retriggered without an intervening note_off, the previous note is closed at
    the retrigger time. Active notes can be closed at the final observed timeline
    tick when the sink is finalized.
    """

    def __init__(self, config: MidiFileOutputConfig) -> None:
        self._config = config
        self._sp_tick = config.seconds_per_tick()
        self._midi = pretty_midi.PrettyMIDI(initial_tempo=float(config.bpm))

        self._user = pretty_midi.Instrument(program=int(config.user_program), name=config.user_track_name)
        self._model = pretty_midi.Instrument(program=int(config.model_program), name=config.model_track_name)
        self._midi.instruments.append(self._user)
        self._midi.instruments.append(self._model)
        self._metronome: pretty_midi.Instrument | None = None
        if config.record_metronome:
            self._metronome = pretty_midi.Instrument(
                program=0,
                is_drum=True,
                name=config.metronome_track_name,
            )
            self._midi.instruments.append(self._metronome)

        self._active_user: Dict[int, Dict[str, float]] = {}
        self._active_model: Dict[int, Dict[str, float]] = {}
        self._max_time = 0.0
        self._max_observed_tick: Optional[int] = None
        self._recording_tick_offset = 0
        self._closed = False

    def _time(self, tick: int) -> float:
        return float(int(tick) + int(self._recording_tick_offset)) * self._sp_tick

    def _observe_recording_tick(self, tick: int) -> None:
        if int(tick) < 0:
            self._recording_tick_offset = max(self._recording_tick_offset, -int(tick))

    def _observe_timeline_tick(self, tick: int) -> None:
        tick = int(tick)
        self._observe_recording_tick(tick)
        if self._max_observed_tick is None or tick > self._max_observed_tick:
            self._max_observed_tick = tick

    def _handle_event(
        self,
        event: MusicalEvent,
        *,
        instrument: pretty_midi.Instrument,
        active: Dict[int, Dict[str, float]],
        default_velocity: int,
    ) -> None:
        self._observe_timeline_tick(event.tick)
        if event.is_placeholder or event.pitch == -1:
            return
        t = self._time(event.tick)
        self._max_time = max(self._max_time, t)

        pitch = int(event.pitch)
        if event.event_type == EventType.NOTE_ON and event.velocity > 0:
            if pitch in active:
                prev = active.pop(pitch)
                start = float(prev["start"])
                vel = int(prev["velocity"])
                if t > start:
                    instrument.notes.append(pretty_midi.Note(velocity=vel, pitch=pitch, start=start, end=t))
            active[pitch] = {"start": t, "velocity": float(event.velocity or default_velocity)}
            return

        if event.event_type == EventType.NOTE_OFF:
            prev = active.pop(pitch, None)
            if prev is None:
                return
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
        _ = bar, beat
        self._observe_timeline_tick(tick)

    def output_metronome_tick(self, tick: int, bar: int, beat: int) -> None:
        _ = bar, beat
        if self._metronome is None:
            return
        self._observe_timeline_tick(tick)
        if int(tick) % int(self._config.ticks_per_beat) != 0:
            return

        ticks_per_bar = int(self._config.ticks_per_beat) * int(self._config.beats_per_bar)
        is_downbeat = ticks_per_bar > 0 and int(tick) % ticks_per_bar == 0
        pitch = self._config.metronome_downbeat_note if is_downbeat else self._config.metronome_beat_note
        velocity = (
            self._config.metronome_downbeat_velocity
            if is_downbeat
            else self._config.metronome_velocity
        )
        start = self._time(int(tick))
        end = self._time(int(tick) + max(1, int(self._config.metronome_duration_ticks)))
        self._max_time = max(self._max_time, end)
        self._metronome.notes.append(
            pretty_midi.Note(
                velocity=int(velocity),
                pitch=int(pitch),
                start=start,
                end=end,
            )
        )

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
        if self._max_observed_tick is not None:
            end = max(end, self._time(self._max_observed_tick))

        active_starts = [
            float(info["start"])
            for active in (self._active_user, self._active_model)
            for info in active.values()
        ]
        if active_starts and end <= max(active_starts):
            end = max(active_starts) + self._sp_tick

        for active, inst in ((self._active_user, self._user), (self._active_model, self._model)):
            if self._config.close_active_notes_on_finalize:
                for pitch, info in active.items():
                    start = float(info["start"])
                    vel = int(info["velocity"])
                    inst.notes.append(
                        pretty_midi.Note(
                            velocity=vel,
                            pitch=int(pitch),
                            start=start,
                            end=end,
                        )
                    )
                self._max_time = max(self._max_time, end)
            active.clear()

    def close(self) -> None:
        if self._closed:
            return
        self._finalize()
        if self._config.output_path:
            path = Path(self._config.output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._midi.write(str(path))
        self._closed = True
