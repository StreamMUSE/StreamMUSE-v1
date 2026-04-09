from __future__ import annotations

import time
from dataclasses import dataclass

import pretty_midi

from streammuse.application.config import ApplicationConfig, OutputConfig, TempoConfig
from streammuse.application.factories.output_factory import OutputSinkFactory
from streammuse.application.services.real_time_music_service import RealTimeMusicService
from streammuse.domain.interfaces.timing_info import TimingInfo
from streammuse.domain.logging import SessionManager
from streammuse.domain.musical import EventType, MusicalEvent
from streammuse.domain.timing import PlaybackScheduler, Tempo


class _ScriptedInput:
    def read_events(self):
        return iter(
            [
                MusicalEvent(tick=0, pitch=60, event_type=EventType.NOTE_ON, velocity=100),
                MusicalEvent(tick=0, pitch=60, event_type=EventType.NOTE_OFF, velocity=0),
            ]
        )

    def close(self):
        return None


class _FakeInference:
    def generate_accompaniment(self, melody_events, generation_start_tick, generation_length_frames, prompt_length_ticks=None):
        return (
            [
                MusicalEvent(
                    tick=generation_start_tick + 1,
                    pitch=48,
                    event_type=EventType.NOTE_ON,
                    velocity=80,
                    source="model",
                ),
                MusicalEvent(
                    tick=generation_start_tick + 2,
                    pitch=48,
                    event_type=EventType.NOTE_OFF,
                    velocity=0,
                    source="model",
                ),
            ],
            TimingInfo(
                request_arrival_time=0.0,
                response_output_time=0.0,
                preprocess_start_time=0.0,
                inference_start_time=0.0,
                inference_end_time=0.0,
                postprocess_start_time=0.0,
            ),
        )

    def inject_history(self, melody_events, accompaniment_events, injection_length_ticks):
        return None

    def set_injection_offset(self, offset_ticks: int):
        return None

    def clear_history(self):
        return None


@dataclass
class _MonotonicNow:
    value: float = 0.0
    step: float = 0.13

    def __call__(self) -> float:
        current = self.value
        self.value += self.step
        return current


def test_simulator_writes_combined_midi_with_two_named_tracks(tmp_path):
    config = ApplicationConfig(
        tempo=TempoConfig(bpm=110.0, ticks_per_beat=4, beats_per_bar=4),
        output=OutputConfig(type="console"),
    )
    session_manager = SessionManager(base_log_dir=str(tmp_path))
    session_dir = session_manager.create_session_directory()
    output_sink = OutputSinkFactory.create(config, session_manager=session_manager)

    now = _MonotonicNow()
    service = RealTimeMusicService(
        input_source=_ScriptedInput(),
        inference_engine=_FakeInference(),
        output_sink=output_sink,
        tempo=Tempo(bpm=110.0, ticks_per_beat=4, beats_per_bar=4),
        scheduler=PlaybackScheduler(),
        generation_interval_ticks=1,
        generation_length_frames=4,
        now=now,
        sleep=lambda _: None,
    )

    service.start(max_ticks=20)
    deadline = time.time() + 1.0
    while service.running and time.time() < deadline:
        time.sleep(0.01)
    service.stop()
    output_sink.close()

    midi_path = session_dir / "combined.mid"
    assert midi_path.exists()

    midi = pretty_midi.PrettyMIDI(str(midi_path))
    _, tempi = midi.get_tempo_changes()
    assert len(tempi) >= 1
    assert abs(float(tempi[0]) - 110.0) < 0.01

    names_to_notes = {inst.name: inst.notes for inst in midi.instruments}
    assert "Melody" in names_to_notes
    assert "Accompaniment" in names_to_notes
    assert names_to_notes["Melody"]
    assert names_to_notes["Accompaniment"]
