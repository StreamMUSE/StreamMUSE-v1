import threading

import mido

from streammuse.infrastructure.input.midi_file import MidiFileInput
from streammuse.infrastructure.inference.lekai_prompt_continuation.scheduler import (
    LekaiPromptContinuationScheduler,
)


def _note_on(pitch: int, tick: int) -> dict:
    return {"type": "note_on", "pitch": pitch, "tick": tick, "velocity": 100}


def _notes_to_event_payload(notes: list[dict[str, int]]) -> list[dict]:
    events = []
    for note in notes:
        tick = int(note["tick"])
        duration = int(note["duration"])
        pitch = int(note["pitch"])
        events.append({"type": "note_on", "pitch": pitch, "tick": tick, "velocity": 64})
        events.append({"type": "note_off", "pitch": pitch, "tick": tick + duration, "velocity": 0})
    return sorted(events, key=lambda event: (int(event["tick"]), str(event["type"]), int(event["pitch"])))


class _BlockingPromptEngine:
    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()
        self.calls = []

    def generate_prompt_accompaniment(self, melody_events, prompt_start_tick, prompt_length_ticks):
        self.calls.append(
            {
                "melody_events": melody_events,
                "prompt_start_tick": prompt_start_tick,
                "prompt_length_ticks": prompt_length_ticks,
            }
        )
        self.started.set()
        assert self.release.wait(timeout=2.0)
        return [_note_on(48, 0)]


class _RecordingContinuationEngine:
    def __init__(self):
        self.inject_calls = []
        self.generate_calls = []

    def inject_history(self, melody_events, accompaniment_events, injection_length_ticks):
        self.inject_calls.append(
            {
                "melody_events": melody_events,
                "accompaniment_events": accompaniment_events,
                "injection_length_ticks": injection_length_ticks,
            }
        )
        return {"success": True}

    def generate(self, **kwargs):
        self.generate_calls.append(kwargs)
        tick = int(kwargs["generation_start_tick"])
        return [_note_on(55, tick)], {"response_output_time": 1.0}


def test_scheduler_accepts_melody_while_prompt_is_running_then_catches_up():
    prompt_engine = _BlockingPromptEngine()
    continuation_engine = _RecordingContinuationEngine()
    scheduler = LekaiPromptContinuationScheduler(
        prompt_engine=prompt_engine,
        continuation_engine=continuation_engine,
    )

    scheduler.start(
        melody_events=[_note_on(60, 0)],
        prompt_length_ticks=32,
        generation_interval_ticks=4,
        inference_mode="sliding_window",
        model_name="lekai_prompt_continuation",
        checkpoint_path=None,
        observed_until_tick=32,
    )

    assert prompt_engine.started.wait(timeout=2.0)
    running_status = scheduler.append_melody(
        [_note_on(62, 44)],
        observed_until_tick=44,
    )
    assert running_status["phase"] == "prompt_running"
    assert running_status["melody_history_beats"] == 11

    prompt_engine.release.set()
    ready_status = scheduler.wait(timeout=2.0)

    assert ready_status["phase"] == "ready"
    assert ready_status["is_playback_ready"] is True
    assert ready_status["melody_history_beats"] == 11
    assert ready_status["accompaniment_history_beats"] == 12
    assert ready_status["continuation_calls"] == 4
    assert [call["generation_start_tick"] for call in continuation_engine.generate_calls] == [
        32,
        36,
        40,
        44,
    ]
    assert continuation_engine.inject_calls[0]["melody_events"] == [
        _note_on(60, 0),
        _note_on(62, 44),
    ]
    assert scheduler.playable_accompaniment()


def test_scheduler_uses_midi_converted_ticks_for_prompt_and_append_boundaries(tmp_path):
    midi_ticks_per_beat = 480
    output_ticks_per_beat = 4
    mid = mido.MidiFile(ticks_per_beat=midi_ticks_per_beat)
    track = mido.MidiTrack()
    mid.tracks.append(track)

    # First note starts at output tick 0. Second note starts at output tick 44,
    # i.e. after an 8-beat prompt window. This catches prompt-length mistakes
    # that fake event payloads would not expose.
    track.append(mido.Message("note_on", note=60, velocity=64, time=0))
    track.append(mido.Message("note_off", note=60, velocity=0, time=120))
    track.append(mido.Message("note_on", note=62, velocity=64, time=(44 * 120) - 120))
    track.append(mido.Message("note_off", note=62, velocity=0, time=120))

    midi_path = tmp_path / "prompt_boundary.mid"
    mid.save(str(midi_path))

    notes, resolution, max_tick = MidiFileInput._midi_to_notes(
        str(midi_path),
        beat_div=output_ticks_per_beat,
        min_pitch=0,
        max_pitch=127,
        program=None,
        max_tick=None,
    )
    assert resolution == midi_ticks_per_beat
    assert [note["tick"] for note in notes] == [0, 44]
    assert max_tick == 45

    events = _notes_to_event_payload(notes)
    prompt_events = [event for event in events if int(event["tick"]) < 32]
    append_events = [event for event in events if int(event["tick"]) >= 32]

    prompt_engine = _BlockingPromptEngine()
    continuation_engine = _RecordingContinuationEngine()
    scheduler = LekaiPromptContinuationScheduler(
        prompt_engine=prompt_engine,
        continuation_engine=continuation_engine,
    )

    scheduler.start(
        melody_events=prompt_events,
        prompt_length_ticks=32,
        generation_interval_ticks=output_ticks_per_beat,
        inference_mode="sliding_window",
        model_name="lekai_prompt_continuation",
        checkpoint_path=None,
        observed_until_tick=32,
    )
    assert prompt_engine.started.wait(timeout=2.0)

    status_after_append = scheduler.append_melody(
        append_events,
        observed_until_tick=44,
    )
    assert status_after_append["melody_history_beats"] == 11

    prompt_engine.release.set()
    ready_status = scheduler.wait(timeout=2.0)

    assert ready_status["phase"] == "ready"
    assert ready_status["accompaniment_history_beats"] == 12
    assert ready_status["continuation_calls"] == 4
    assert continuation_engine.inject_calls[0]["melody_events"] == events


def test_scheduler_clear_invalidates_running_work():
    prompt_engine = _BlockingPromptEngine()
    continuation_engine = _RecordingContinuationEngine()
    scheduler = LekaiPromptContinuationScheduler(
        prompt_engine=prompt_engine,
        continuation_engine=continuation_engine,
    )

    scheduler.start(
        melody_events=[_note_on(60, 0)],
        prompt_length_ticks=32,
        generation_interval_ticks=4,
        inference_mode="sliding_window",
        model_name="lekai_prompt_continuation",
        checkpoint_path=None,
        observed_until_tick=32,
    )
    assert prompt_engine.started.wait(timeout=2.0)

    cleared = scheduler.clear()
    prompt_engine.release.set()

    assert cleared["phase"] == "idle"
    assert scheduler.status()["phase"] == "idle"
    assert continuation_engine.generate_calls == []
