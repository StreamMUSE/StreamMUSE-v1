import torch

from streammuse.infrastructure.inference.lekai_prompt_continuation import (
    LekaiPromptContinuationBackend,
)
from streammuse.infrastructure.inference.lekai_http_backend import LekaiHttpBackend
from streammuse.infrastructure.inference.lekai_prompt_continuation.continuation_engine import (
    LekaiContinuationEngine,
)
from streammuse.infrastructure.inference.lekai_prompt_continuation.engine import (
    LekaiPromptContinuationEngine,
)


def _note_on(pitch: int, tick: int) -> dict:
    return {"type": "note_on", "pitch": pitch, "tick": tick, "velocity": 100}


class _FakePromptEngine:
    def __init__(self):
        self.calls = []
        self.reset_calls = []

    def runtime_info(self):
        return {
            "mode": "fake_prompt",
            "has_real_model": True,
            "checkpoint_path": "/tmp/prompt.safetensors",
            "fallback_reason": None,
            "load_time_ms": 1.5,
            "warmup_time_ms": 9.0,
            "warmup_error": None,
            "is_warmed_up": True,
        }

    def generate_prompt_accompaniment(self, melody_events, prompt_start_tick, prompt_length_ticks):
        self.calls.append(
            {
                "melody_events": melody_events,
                "prompt_start_tick": prompt_start_tick,
                "prompt_length_ticks": prompt_length_ticks,
            }
        )
        return [_note_on(48, 0)]

    def reset_session(self, seed):
        self.reset_calls.append(int(seed))
        return int(seed)


class _FakeContinuationEngine:
    def __init__(self):
        self.generate_calls = []
        self.inject_calls = []
        self.reset_calls = []
        self.session_epoch = 0

    def configure(self, config):
        self.config = config

    def runtime_info(self):
        return {
            "mode": "fake_continuation",
            "has_real_model": True,
            "resolved_device": "cpu",
            "resolved_dtype": "float32",
            "checkpoint_path": "/tmp/continuation.safetensors",
            "checkpoint_format": "safetensors",
            "fallback_reason": None,
            "load_time_ms": 1.0,
            "warmup_time_ms": 2.0,
            "use_cache": True,
            "runtime_model_name": "lekai",
            "runtime_inference_mode": "sliding_window",
        }

    def generate(self, **kwargs):
        self.generate_calls.append(kwargs)
        return [_note_on(55, kwargs["generation_start_tick"])], {"response_output_time": 1.0}

    def inject_history(self, melody_events, accompaniment_events, injection_length_ticks):
        self.inject_calls.append(
            {
                "melody_events": melody_events,
                "accompaniment_events": accompaniment_events,
                "injection_length_ticks": injection_length_ticks,
            }
        )
        return {
            "success": True,
            "message": "ok",
            "melody_notes_injected": len(melody_events),
            "accompaniment_notes_injected": len(accompaniment_events),
            "injection_length_ticks": injection_length_ticks,
        }

    def clear_history(self):
        return {"success": True, "message": "ok"}

    def reset_session(self, seed):
        self.reset_calls.append(int(seed))
        self.session_epoch += 1
        return {
            "success": True,
            "session_id": f"session-{self.session_epoch}",
            "session_epoch": self.session_epoch,
            "effective_seed": int(seed),
            "pending_boundary_generations": 0,
        }

    def injection_status(self):
        return {
            "is_injected": False,
            "injection_length_ticks": 0,
            "runtime_model_name": "lekai",
            "runtime_inference_mode": "",
        }


def test_prompt_continuation_backend_runtime_info_uses_model_name():
    backend = LekaiPromptContinuationBackend()

    info = backend.runtime_info()

    assert info["runtime_model_name"] == "lekai_prompt_continuation"
    assert info["mode"] == "rule_stub"


def test_prompt_continuation_backend_delegates_runtime_to_engine():
    engine = LekaiPromptContinuationEngine()
    backend = LekaiPromptContinuationBackend(engine=engine)

    assert backend.runtime_info()["runtime_model_name"] == "lekai_prompt_continuation"


def test_prompt_continuation_engine_calls_prompt_hook_when_prompt_window_exists():
    prompt_engine = _FakePromptEngine()
    continuation_engine = _FakeContinuationEngine()
    engine = LekaiPromptContinuationEngine(
        prompt_engine=prompt_engine,
        continuation_engine=continuation_engine,
    )

    accompaniment, timings = engine.generate(
        melody_events=[_note_on(60, 0)],
        generation_start_tick=32,
        generation_length_frames=8,
        generation_interval_ticks=4,
        prompt_length_ticks=32,
        inference_mode="sliding_window",
        model_name="lekai_prompt_continuation",
        checkpoint_path=None,
    )

    assert prompt_engine.calls[0]["prompt_length_ticks"] == 32
    assert continuation_engine.inject_calls[0]["accompaniment_events"] == [_note_on(48, 0)]
    assert continuation_engine.inject_calls[0]["injection_length_ticks"] == 32
    assert continuation_engine.generate_calls[0]["generation_start_tick"] == 32
    assert accompaniment == [_note_on(55, 32)]
    assert timings["response_output_time"] == 1.0


def test_prompt_continuation_engine_tracks_catchup_after_prompt_and_continuation():
    prompt_engine = _FakePromptEngine()
    continuation_engine = _FakeContinuationEngine()
    engine = LekaiPromptContinuationEngine(
        prompt_engine=prompt_engine,
        continuation_engine=continuation_engine,
    )

    engine.generate(
        melody_events=[_note_on(60, 0), _note_on(62, 44)],
        generation_start_tick=44,
        generation_length_frames=4,
        generation_interval_ticks=4,
        prompt_length_ticks=32,
        inference_mode="sliding_window",
        model_name="lekai_prompt_continuation",
        checkpoint_path=None,
    )

    status = engine.catchup_status()

    assert status["melody_history_beats"] == 11
    assert status["accompaniment_history_beats"] == 12
    assert status["beats_needed_for_playback"] == 0
    assert status["is_playback_ready"] is True


def test_prompt_continuation_engine_reports_not_ready_when_only_prompt_exists():
    prompt_engine = _FakePromptEngine()
    continuation_engine = _FakeContinuationEngine()
    engine = LekaiPromptContinuationEngine(
        prompt_engine=prompt_engine,
        continuation_engine=continuation_engine,
    )

    engine.generate(
        melody_events=[_note_on(60, 0), _note_on(62, 44)],
        generation_start_tick=44,
        generation_length_frames=0,
        generation_interval_ticks=4,
        prompt_length_ticks=32,
        inference_mode="sliding_window",
        model_name="lekai_prompt_continuation",
        checkpoint_path=None,
    )

    status = engine.catchup_status()

    assert status["melody_history_beats"] == 11
    assert status["accompaniment_history_beats"] == 8
    assert status["beats_needed_for_playback"] == 4
    assert status["is_playback_ready"] is False


def test_prompt_continuation_engine_runtime_info_includes_subengines():
    prompt_engine = _FakePromptEngine()
    continuation_engine = _FakeContinuationEngine()
    engine = LekaiPromptContinuationEngine(
        prompt_engine=prompt_engine,
        continuation_engine=continuation_engine,
    )

    info = engine.runtime_info()

    assert info["runtime_model_name"] == "lekai_prompt_continuation"
    assert info["prompt_mode"] == "fake_prompt"
    assert info["prompt_has_real_model"] is True
    assert info["prompt_checkpoint_path"] == "/tmp/prompt.safetensors"
    assert info["prompt_warmup_time_ms"] == 9.0
    assert info["prompt_is_warmed_up"] is True
    assert info["catchup_melody_history_beats"] == 0
    assert info["catchup_beats_needed_for_playback"] == 1


def test_prompt_continuation_backend_generates_with_fallback_contract():
    backend = LekaiPromptContinuationBackend()

    accompaniment, timings = backend.generate(
        melody_events=[_note_on(60, 0)],
        generation_start_tick=4,
        generation_length_frames=8,
        generation_interval_ticks=4,
        prompt_length_ticks=None,
        inference_mode="sliding_window",
        model_name="lekai_prompt_continuation",
        checkpoint_path=None,
    )

    assert isinstance(accompaniment, list)
    assert "response_output_time" in timings


def test_prompt_continuation_backend_inject_clear_status_contract():
    backend = LekaiPromptContinuationBackend()

    inject_result = backend.inject_history(
        melody_events=[_note_on(60, 0)],
        accompaniment_events=[_note_on(48, 0)],
        injection_length_ticks=16,
    )
    assert inject_result["success"] is True

    status = backend.injection_status()
    assert status["is_injected"] is True
    assert status["runtime_model_name"] == "lekai_prompt_continuation"
    assert status["catchup_melody_history_beats"] == 4
    assert status["catchup_accompaniment_history_beats"] == 4

    clear_result = backend.clear_history()
    assert clear_result["success"] is True
    assert len(clear_result["melody_history"]) == 1
    assert len(clear_result["accompaniment_history"]) == 1

    assert backend.injection_status()["is_injected"] is False
    assert backend.catchup_status()["melody_history_beats"] == 0


def test_prompt_continuation_reset_clears_history_and_passes_both_seeds():
    prompt_engine = _FakePromptEngine()
    continuation_engine = _FakeContinuationEngine()
    engine = LekaiPromptContinuationEngine(
        prompt_engine=prompt_engine,
        continuation_engine=continuation_engine,
    )
    backend = LekaiPromptContinuationBackend(engine=engine)
    backend.inject_history(
        melody_events=[_note_on(60, 0)],
        accompaniment_events=[_note_on(48, 0)],
        injection_length_ticks=16,
    )

    result = backend.reset_session(prompt_seed=111, continuation_seed=222)

    assert result == {
        "success": True,
        "prompt_seed": 111,
        "continuation_effective_seed": 222,
        "session_id": "session-1",
        "session_epoch": 1,
        "pending_boundary_generations": 0,
        "scheduler_phase": "idle",
        "scheduler_is_running": False,
    }
    assert prompt_engine.reset_calls == [111]
    assert continuation_engine.reset_calls == [222]
    assert backend.catchup_status()["melody_history_beats"] == 0
    assert backend.raw_accompaniment_history() == []


def test_prompt_continuation_reset_replays_continuation_rng_sequence():
    prompt_engine = _FakePromptEngine()
    continuation_backend = LekaiHttpBackend()
    engine = LekaiPromptContinuationEngine(
        prompt_engine=prompt_engine,
        continuation_engine=LekaiContinuationEngine(backend=continuation_backend),
    )

    engine.reset_session(prompt_seed=7, continuation_seed=31415)
    first = torch.rand(8, generator=continuation_backend._sample_generator)
    engine.reset_session(prompt_seed=7, continuation_seed=31415)
    second = torch.rand(8, generator=continuation_backend._sample_generator)

    assert torch.equal(first, second)
