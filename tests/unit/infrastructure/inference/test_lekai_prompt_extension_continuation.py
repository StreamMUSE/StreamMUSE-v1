from streammuse.infrastructure.inference.lekai_prompt_continuation.backend import (
    LekaiPromptContinuationBackend,
)
from streammuse.infrastructure.inference.lekai_prompt_continuation.prompt_extension_scheduler import (
    LekaiPromptExtensionContinuationScheduler,
)


def _note_on(pitch: int, tick: int) -> dict:
    return {"type": "note_on", "pitch": pitch, "tick": tick, "velocity": 100}


class _ExtensionPromptEngine:
    def __init__(self, generated_acc_beats: int = 9):
        self.calls = []
        self._generated_acc_beats = int(generated_acc_beats)

    def generate_prompt_accompaniment(self, melody_events, prompt_start_tick, prompt_length_ticks):
        self.calls.append(
            {
                "melody_events": melody_events,
                "prompt_start_tick": prompt_start_tick,
                "prompt_length_ticks": prompt_length_ticks,
            }
        )
        return [_note_on(48, 0), _note_on(50, 32)]

    def last_generated_acc_beats(self):
        return self._generated_acc_beats


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


def test_prompt_extension_scheduler_starts_continuation_after_prompt_generated_extension():
    prompt_engine = _ExtensionPromptEngine(generated_acc_beats=9)
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
        observed_until_tick=44,
    )
    ready_status = scheduler.wait(timeout=2.0)

    assert ready_status["phase"] == "ready"
    assert ready_status["prompt_extension_ticks"] == 4
    assert ready_status["accompaniment_history_beats"] == 12
    assert prompt_engine.calls[0]["prompt_length_ticks"] == 36
    assert continuation_engine.inject_calls[0]["injection_length_ticks"] == 36
    assert [call["generation_start_tick"] for call in continuation_engine.generate_calls] == [
        36,
        40,
        44,
    ]


def test_prompt_extension_scheduler_falls_back_to_prompt_boundary_when_prompt_has_no_extension():
    prompt_engine = _ExtensionPromptEngine(generated_acc_beats=8)
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
        observed_until_tick=44,
    )
    scheduler.wait(timeout=2.0)

    assert continuation_engine.inject_calls[0]["injection_length_ticks"] == 32
    assert [call["generation_start_tick"] for call in continuation_engine.generate_calls][0] == 32


def test_backend_can_select_prompt_extension_engine_with_env(monkeypatch):
    monkeypatch.setenv("LEKAI_PROMPT_CONTINUATION_ENGINE", "prompt_extension")
    monkeypatch.setenv("LEKAI_PROMPT_CONTINUATION_EXTENSION_TICKS", "4")

    backend = LekaiPromptContinuationBackend()
    info = backend.runtime_info()

    assert info["prompt_continuation_variant"] == "prompt_extension"
    assert info["prompt_extension_ticks"] == 4
