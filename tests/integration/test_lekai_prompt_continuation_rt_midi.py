from pathlib import Path

import pytest

from streammuse.infrastructure.input.midi_file import MidiFileInput
from streammuse.infrastructure.inference.lekai_prompt_continuation.scheduler import (
    LekaiPromptContinuationScheduler,
)


RT_PERIODIC_GT_MELODY = Path(
    "/data/home/yuanxin/RT-accompanimentV2/user_midi_recording_data/"
    "aligned/periodic_20260409-114946_6217163/dataset_gt_melody.mid"
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


class _PromptEngine:
    def __init__(self):
        self.calls = []

    def generate_prompt_accompaniment(self, melody_events, prompt_start_tick, prompt_length_ticks):
        self.calls.append(
            {
                "melody_events": melody_events,
                "prompt_start_tick": prompt_start_tick,
                "prompt_length_ticks": prompt_length_ticks,
            }
        )
        return [_note_on(48, 0)]


class _ContinuationEngine:
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


def test_prompt_continuation_scheduler_starts_from_real_rt_midi_sample():
    try:
        sample_exists = RT_PERIODIC_GT_MELODY.exists()
    except PermissionError as exc:
        pytest.skip(f"external RT MIDI sample is not readable: {RT_PERIODIC_GT_MELODY} ({exc})")
    if not sample_exists:
        pytest.skip(f"external RT MIDI sample not found: {RT_PERIODIC_GT_MELODY}")

    notes, resolution, max_tick = MidiFileInput._midi_to_notes(
        str(RT_PERIODIC_GT_MELODY),
        beat_div=4,
        min_pitch=0,
        max_pitch=127,
        program=None,
        max_tick=None,
    )
    assert resolution > 0
    assert max_tick > 44

    events = _notes_to_event_payload(notes)
    prompt_events = [event for event in events if int(event["tick"]) < 32]
    append_events = [event for event in events if 32 <= int(event["tick"]) <= 44]

    assert prompt_events, "classic RT sample should contain melody inside the first 8 beats"
    assert append_events, "classic RT sample should contain melody after the prompt window"

    prompt_engine = _PromptEngine()
    continuation_engine = _ContinuationEngine()
    scheduler = LekaiPromptContinuationScheduler(
        prompt_engine=prompt_engine,
        continuation_engine=continuation_engine,
    )

    scheduler.start(
        melody_events=prompt_events,
        prompt_length_ticks=32,
        generation_interval_ticks=4,
        inference_mode="sliding_window",
        model_name="lekai_prompt_continuation",
        checkpoint_path=None,
        observed_until_tick=32,
    )
    scheduler.append_melody(append_events, observed_until_tick=44)
    ready_status = scheduler.wait(timeout=2.0)

    assert prompt_engine.calls[0]["melody_events"] == prompt_events
    assert prompt_engine.calls[0]["prompt_length_ticks"] == 32
    assert continuation_engine.inject_calls[0]["melody_events"] == prompt_events + append_events
    assert ready_status["melody_history_beats"] == 11
    assert ready_status["accompaniment_history_beats"] == 12
    assert ready_status["continuation_calls"] == 4
    assert ready_status["is_playback_ready"] is True
