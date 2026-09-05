import threading
from concurrent.futures import TimeoutError

import mido
import pytest

from streammuse.infrastructure.input.midi_file import MidiFileInput
from streammuse.infrastructure.inference.lekai_prompt_continuation.prompt_extension_scheduler import (
    LekaiPromptExtensionContinuationScheduler,
)
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

    def generate_prompt_accompaniment(
        self, melody_events, prompt_start_tick, prompt_length_ticks, bpm=None
    ):
        self.calls.append(
            {
                "melody_events": melody_events,
                "prompt_start_tick": prompt_start_tick,
                "prompt_length_ticks": prompt_length_ticks,
                "bpm": bpm,
            }
        )
        self.started.set()
        assert self.release.wait(timeout=2.0)
        return [_note_on(48, 0)]


class _ImmediatePromptEngine:
    def __init__(self):
        self.calls = []

    def generate_prompt_accompaniment(
        self, melody_events, prompt_start_tick, prompt_length_ticks, bpm=None
    ):
        self.calls.append(
            {
                "melody_events": melody_events,
                "prompt_start_tick": prompt_start_tick,
                "prompt_length_ticks": prompt_length_ticks,
                "bpm": bpm,
            }
        )
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


class _BlockingRecordingContinuationEngine(_RecordingContinuationEngine):
    def __init__(self):
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()

    def generate(self, **kwargs):
        self.generate_calls.append(kwargs)
        if len(self.generate_calls) == 1:
            self.started.set()
            assert self.release.wait(timeout=2.0)
        tick = int(kwargs["generation_start_tick"])
        return [_note_on(55, tick)], {"response_output_time": 1.0}


class _ShortPromptEngine(_ImmediatePromptEngine):
    def last_generated_acc_beats(self):
        return 7


class _GeneratedBeatPromptEngine(_ImmediatePromptEngine):
    def __init__(self, generated_acc_beats: int):
        super().__init__()
        self._generated_acc_beats = int(generated_acc_beats)

    def last_generated_acc_beats(self):
        return self._generated_acc_beats



def _run_fifo_packet_scenario(*, block_prompt: bool):
    prompt_engine = _BlockingPromptEngine() if block_prompt else _ImmediatePromptEngine()
    continuation_engine = _RecordingContinuationEngine()
    scheduler = LekaiPromptContinuationScheduler(
        prompt_engine=prompt_engine,
        continuation_engine=continuation_engine,
    )
    packets = [
        (36, [_note_on(62, 32)]),
        (40, []),
        (44, [_note_on(64, 40)]),
    ]

    scheduler.start(
        melody_events=[_note_on(60, 0)],
        prompt_length_ticks=32,
        generation_interval_ticks=4,
        inference_mode="sliding_window",
        model_name="lekai_prompt_continuation",
        checkpoint_path=None,
        bpm=96,
        observed_until_tick=32,
    )

    if block_prompt:
        assert prompt_engine.started.wait(timeout=2.0)
        for observed_until_tick, melody_events in packets:
            scheduler.append_melody(
                melody_events,
                observed_until_tick=observed_until_tick,
            )
        prompt_engine.release.set()
        ready_status = scheduler.wait(timeout=2.0)
    else:
        scheduler.wait(timeout=2.0)
        for observed_until_tick, melody_events in packets:
            scheduler.append_melody(
                melody_events,
                observed_until_tick=observed_until_tick,
            )
            ready_status = scheduler.wait(timeout=2.0)

    trace = [
        (call["generation_start_tick"], call["melody_events"])
        for call in continuation_engine.generate_calls
    ]
    return trace, continuation_engine.inject_calls, ready_status


def test_scheduler_fifo_requests_match_fast_and_blocked_prompt_paths():
    fast_trace, fast_inject_calls, fast_status = _run_fifo_packet_scenario(
        block_prompt=False
    )
    blocked_trace, blocked_inject_calls, blocked_status = _run_fifo_packet_scenario(
        block_prompt=True
    )

    expected_trace = [
        (32, []),
        (36, [_note_on(62, 32)]),
        (40, []),
        (44, [_note_on(64, 40)]),
    ]
    assert fast_trace == expected_trace
    assert blocked_trace == expected_trace
    assert fast_trace == blocked_trace
    assert fast_inject_calls[0]["melody_events"] == [_note_on(60, 0)]
    assert blocked_inject_calls[0]["melody_events"] == [_note_on(60, 0)]
    assert fast_status["melody_observed_until_tick"] == 44
    assert blocked_status["melody_observed_until_tick"] == 44
    assert fast_status["pending_melody_event_count"] == 0
    assert blocked_status["pending_melody_event_count"] == 0
    for generation_start_tick, melody_events in fast_trace + blocked_trace:
        assert all(int(event["tick"]) < generation_start_tick for event in melody_events)


def test_scheduler_boundaries_remain_exact_while_continuation_is_blocked():
    prompt_engine = _ImmediatePromptEngine()
    continuation_engine = _BlockingRecordingContinuationEngine()
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
        bpm=120,
        observed_until_tick=32,
    )
    assert continuation_engine.started.wait(timeout=2.0)
    scheduler.append_melody([_note_on(62, 32)], observed_until_tick=36)
    scheduler.append_melody([], observed_until_tick=40)
    scheduler.append_melody([_note_on(64, 40)], observed_until_tick=44)
    continuation_engine.release.set()
    scheduler.wait(timeout=2.0)

    assert [
        (call["generation_start_tick"], call["melody_events"])
        for call in continuation_engine.generate_calls
    ] == [
        (32, []),
        (36, [_note_on(62, 32)]),
        (40, []),
        (44, [_note_on(64, 40)]),
    ]


def test_scheduler_partial_watermark_does_not_authorize_next_boundary():
    prompt_engine = _ImmediatePromptEngine()
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
        bpm=120,
        observed_until_tick=32,
    )
    scheduler.wait(timeout=2.0)
    scheduler.append_melody([_note_on(62, 32)], observed_until_tick=34)

    with pytest.raises(TimeoutError):
        scheduler.wait(timeout=0.05)
    assert [call["generation_start_tick"] for call in continuation_engine.generate_calls] == [32]

    scheduler.append_melody([], observed_until_tick=36)
    scheduler.wait(timeout=2.0)
    assert [call["generation_start_tick"] for call in continuation_engine.generate_calls] == [
        32,
        36,
    ]
    assert continuation_engine.generate_calls[-1]["melody_events"] == [_note_on(62, 32)]


def test_scheduler_shutdown_wakes_partial_watermark_wait():
    prompt_engine = _ImmediatePromptEngine()
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
        bpm=120,
        observed_until_tick=32,
    )
    scheduler.wait(timeout=2.0)
    scheduler.append_melody([_note_on(62, 32)], observed_until_tick=34)
    with scheduler._lock:
        worker = scheduler._future
    assert worker is not None
    with pytest.raises(TimeoutError):
        worker.result(timeout=0.05)

    shutdown_thread = threading.Thread(target=scheduler.shutdown)
    shutdown_thread.start()
    shutdown_thread.join(timeout=2.0)

    assert not shutdown_thread.is_alive()
    worker.result(timeout=2.0)
    assert worker.done()


def test_scheduler_defers_frozen_events_after_short_actual_prompt():
    prompt_engine = _ShortPromptEngine()
    continuation_engine = _RecordingContinuationEngine()
    scheduler = LekaiPromptContinuationScheduler(
        prompt_engine=prompt_engine,
        continuation_engine=continuation_engine,
    )
    scheduler.start(
        melody_events=[_note_on(60, 0), _note_on(62, 28)],
        prompt_length_ticks=32,
        generation_interval_ticks=4,
        inference_mode="sliding_window",
        model_name="lekai_prompt_continuation",
        checkpoint_path=None,
        bpm=120,
        observed_until_tick=32,
    )
    scheduler.wait(timeout=2.0)

    assert continuation_engine.inject_calls[0]["injection_length_ticks"] == 28
    assert continuation_engine.inject_calls[0]["melody_events"] == [_note_on(60, 0)]
    assert [
        (call["generation_start_tick"], call["melody_events"])
        for call in continuation_engine.generate_calls
    ] == [
        (28, []),
        (32, [_note_on(62, 28)]),
    ]


@pytest.mark.parametrize(
    ("generated_acc_beats", "expected_first_tick"),
    [(9, 36), (8, 32)],
)
def test_prompt_extension_preserves_success_and_fallback_start_boundaries(
    generated_acc_beats, expected_first_tick
):
    prompt_engine = _GeneratedBeatPromptEngine(generated_acc_beats)
    continuation_engine = _RecordingContinuationEngine()
    scheduler = LekaiPromptExtensionContinuationScheduler(
        prompt_engine=prompt_engine,
        continuation_engine=continuation_engine,
        prompt_extension_ticks=4,
    )
    scheduler.start(
        melody_events=[_note_on(60, 0)],
        prompt_length_ticks=32,
        generation_interval_ticks=4,
        inference_mode="sliding_window",
        model_name="lekai_prompt_continuation",
        checkpoint_path=None,
        bpm=120,
        observed_until_tick=44,
    )
    scheduler.wait(timeout=2.0)

    assert continuation_engine.inject_calls[0]["melody_events"] == [_note_on(60, 0)]
    assert continuation_engine.generate_calls[0]["generation_start_tick"] == expected_first_tick


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
        bpm=96,
        observed_until_tick=32,
    )

    assert prompt_engine.started.wait(timeout=2.0)
    running_status = scheduler.append_melody(
        [_note_on(62, 40)],
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
    assert ready_status["last_continuation_event_count"] == 1
    assert ready_status["last_continuation_note_on_count"] == 1
    assert ready_status["last_continuation_min_tick"] == 44
    assert ready_status["last_continuation_max_tick"] == 44
    assert ready_status["empty_continuation_output_streak"] == 0
    assert [call["generation_start_tick"] for call in continuation_engine.generate_calls] == [
        32,
        36,
        40,
        44,
    ]
    assert [call["melody_events"] for call in continuation_engine.generate_calls] == [
        [],
        [],
        [],
        [_note_on(62, 40)],
    ]
    assert prompt_engine.calls[0]["bpm"] == 96
    assert {call["bpm"] for call in continuation_engine.generate_calls} == {96}
    assert ready_status["effective_bpm"] == 96
    assert continuation_engine.inject_calls[0]["melody_events"] == [
        _note_on(60, 0),
    ]
    assert scheduler.playable_accompaniment()


def test_scheduler_restarts_catchup_when_append_arrives_after_prompt_ready():
    prompt_engine = _ImmediatePromptEngine()
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
        bpm=120,
        observed_until_tick=32,
    )
    initial_ready = scheduler.wait(timeout=2.0)
    assert initial_ready["phase"] == "ready"
    assert initial_ready["melody_history_beats"] == 8
    assert initial_ready["accompaniment_history_beats"] == 9
    assert initial_ready["is_playback_ready"] is True

    appended = scheduler.append_melody(
        [_note_on(62, 40)],
        observed_until_tick=44,
    )
    assert appended["phase"] == "catchup_running"
    assert appended["is_playback_ready"] is False

    final_ready = scheduler.wait(timeout=2.0)
    assert final_ready["phase"] == "ready"
    assert final_ready["melody_history_beats"] == 11
    assert final_ready["accompaniment_history_beats"] == 12
    assert final_ready["continuation_calls"] == 4
    assert final_ready["is_playback_ready"] is True
    assert [call["generation_start_tick"] for call in continuation_engine.generate_calls] == [
        32,
        36,
        40,
        44,
    ]


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
    # Current MidiFileInput uses the shared MidiConverter path, whose returned
    # resolution is the configured output beat division.
    assert resolution == output_ticks_per_beat
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
        bpm=120,
        observed_until_tick=32,
    )
    assert prompt_engine.started.wait(timeout=2.0)

    status_after_append = scheduler.append_melody(
        append_events,
        observed_until_tick=48,
    )
    assert status_after_append["melody_observed_until_tick"] == 48
    assert status_after_append["melody_history_beats"] == 12

    prompt_engine.release.set()
    ready_status = scheduler.wait(timeout=2.0)

    assert ready_status["phase"] == "ready"
    assert ready_status["accompaniment_history_beats"] == 13
    assert ready_status["continuation_calls"] == 5
    assert [call["generation_start_tick"] for call in continuation_engine.generate_calls] == [
        32,
        36,
        40,
        44,
        48,
    ]
    assert continuation_engine.inject_calls[0]["melody_events"] == prompt_events
    assert [call["melody_events"] for call in continuation_engine.generate_calls] == [
        [],
        [],
        [],
        [],
        append_events,
    ]



def test_scheduler_rejects_contract_violating_append_without_mutating_history():
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
        bpm=120,
        observed_until_tick=32,
    )
    assert prompt_engine.started.wait(timeout=2.0)

    accepted = scheduler.append_melody([], observed_until_tick=36)
    assert accepted["melody_observed_until_tick"] == 36
    assert accepted["pending_melody_event_count"] == 0
    before = scheduler.status()

    with pytest.raises(ValueError, match="previous_observed_until_tick"):
        scheduler.append_melody([_note_on(63, 35)], observed_until_tick=40)
    assert scheduler.status() == before

    with pytest.raises(ValueError, match="tick < observed_until_tick"):
        scheduler.append_melody([_note_on(64, 40)], observed_until_tick=40)
    assert scheduler.status() == before

    with pytest.raises(ValueError, match="must be monotonic"):
        scheduler.append_melody([], observed_until_tick=34)
    assert scheduler.status() == before

    prompt_engine.release.set()
    scheduler.wait(timeout=2.0)


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
        bpm=120,
        observed_until_tick=32,
    )
    assert prompt_engine.started.wait(timeout=2.0)
    scheduler.append_melody([], observed_until_tick=36)

    cleared = scheduler.clear()
    prompt_engine.release.set()

    assert cleared["phase"] == "idle"
    assert cleared["melody_observed_until_tick"] == 0
    assert cleared["pending_melody_event_count"] == 0
    assert scheduler.status()["phase"] == "idle"
    assert continuation_engine.generate_calls == []


def test_scheduler_drain_and_clear_waits_for_running_worker():
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
        bpm=120,
        observed_until_tick=32,
    )
    assert prompt_engine.started.wait(timeout=2.0)

    finished = threading.Event()
    result = {}

    def reset_worker():
        result.update(scheduler.drain_and_clear())
        finished.set()

    thread = threading.Thread(target=reset_worker, daemon=True)
    thread.start()
    assert not finished.wait(timeout=0.05)

    prompt_engine.release.set()
    thread.join(timeout=2.0)

    assert finished.is_set()
    assert result["phase"] == "idle"
    assert result["is_running"] is False
    assert result["melody_event_count"] == 0
    assert result["accompaniment_event_count"] == 0
    assert continuation_engine.generate_calls == []
