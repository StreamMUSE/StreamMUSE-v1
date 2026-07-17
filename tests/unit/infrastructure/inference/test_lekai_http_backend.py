from concurrent.futures import Future
import threading
import time

import numpy as np
import pytest
import torch

from streammuse.infrastructure.inference.lekai_http_backend import (
    LekaiHttpBackend,
    SessionStateError,
)


def _note_on(pitch: int, tick: int) -> dict:
    return {"type": "note_on", "pitch": pitch, "tick": tick, "velocity": 100}


def test_generate_extends_melody_history_instead_of_replacing():
    backend = LekaiHttpBackend()
    backend.inject_history(
        melody_events=[_note_on(60, 0)],
        accompaniment_events=[],
        injection_length_ticks=16,
    )

    backend.generate(
        melody_events=[_note_on(64, 4)],
        generation_start_tick=8,
        generation_length_frames=20,
        generation_interval_ticks=4,
        prompt_length_ticks=None,
        inference_mode="sliding_window",
        model_name="lekai",
        checkpoint_path=None,
    )

    assert any(e["pitch"] == 60 for e in backend._melody_history)
    assert any(e["pitch"] == 64 for e in backend._melody_history)


def test_generate_rule_based_respects_generation_length_frames():
    backend = LekaiHttpBackend()

    accompaniment, _ = backend.generate(
        melody_events=[_note_on(60, 0)],
        generation_start_tick=4,
        generation_length_frames=12,
        generation_interval_ticks=4,
        prompt_length_ticks=None,
        inference_mode="sliding_window",
        model_name="lekai",
        checkpoint_path=None,
    )

    # 12 frames with interval 4 -> 3 intervals; one pitch -> 3 note_on + 3 note_off
    assert len(accompaniment) == 6


def test_generate_rule_based_note_off_velocity_is_zero():
    backend = LekaiHttpBackend()

    accompaniment, _ = backend.generate(
        melody_events=[_note_on(60, 0)],
        generation_start_tick=4,
        generation_length_frames=8,
        generation_interval_ticks=4,
        prompt_length_ticks=None,
        inference_mode="sliding_window",
        model_name="lekai",
        checkpoint_path=None,
    )

    note_offs = [e for e in accompaniment if e["type"] == "note_off"]
    assert note_offs
    assert all(e.get("velocity") == 0 for e in note_offs)


def test_generate_rule_based_empty_melody_returns_empty():
    backend = LekaiHttpBackend()

    accompaniment, _ = backend.generate(
        melody_events=[],
        generation_start_tick=20,
        generation_length_frames=16,
        generation_interval_ticks=4,
        prompt_length_ticks=None,
        inference_mode="sliding_window",
        model_name="lekai",
        checkpoint_path=None,
    )

    assert accompaniment == []


def test_generate_rule_based_length_is_independent_of_interval():
    backend = LekaiHttpBackend()
    generation_start_tick = 100
    generation_length_frames = 20

    for interval in [2, 4, 8]:
        accompaniment, _ = backend.generate(
            melody_events=[_note_on(60, generation_start_tick)],
            generation_start_tick=generation_start_tick,
            generation_length_frames=generation_length_frames,
            generation_interval_ticks=interval,
            prompt_length_ticks=None,
            inference_mode="sliding_window",
            model_name="lekai",
            checkpoint_path=None,
        )

        note_ons = [e for e in accompaniment if e["type"] == "note_on"]
        note_offs = [e for e in accompaniment if e["type"] == "note_off"]

        assert note_ons
        assert note_offs
        assert min(int(e["tick"]) for e in note_ons) == generation_start_tick
        assert max(int(e["tick"]) for e in note_offs) == generation_start_tick + generation_length_frames

        expected_on_ticks = [generation_start_tick + i * 4 for i in range(generation_length_frames // 4)]
        assert sorted({int(e["tick"]) for e in note_ons}) == expected_on_ticks


def test_trim_histories_keeps_recent_window_only(monkeypatch):
    monkeypatch.setenv("LEKAI_HISTORY_MAX_TICKS", "20")
    backend = LekaiHttpBackend()
    backend._melody_history = [_note_on(60, 0), _note_on(62, 40)]
    backend._accompaniment_history = [
        {"type": "note_off", "pitch": 48, "tick": 1, "velocity": 0},
        {"type": "note_off", "pitch": 50, "tick": 41, "velocity": 0},
    ]
    backend._accompaniment_token_history = {0: [255], 10: [169]}
    backend._accompaniment_bar_token_history = {0: [255], 10: [255]}

    # max_history_ticks=20 (from env), cutoff=30 -> remove tick < 30
    backend._trim_histories(generation_start_tick=50, generation_length_frames=10)

    assert all(int(e["tick"]) >= 30 for e in backend._melody_history)
    assert all(int(e["tick"]) >= 30 for e in backend._accompaniment_history)
    assert backend._accompaniment_token_history == {10: [169]}
    assert backend._accompaniment_bar_token_history == {10: [255]}


def test_runtime_info_contract_default_stub():
    backend = LekaiHttpBackend()
    info = backend.runtime_info()
    assert info["mode"] == "rule_stub"
    assert info["has_real_model"] is False
    assert "resolved_device" in info
    assert "resolved_dtype" in info


def test_prompt_context_defaults_to_retained_history_window(monkeypatch):
    monkeypatch.delenv("LEKAI_PROMPT_CONTEXT_BEATS", raising=False)
    backend = LekaiHttpBackend()

    assert backend._prompt_context_beats() == 128

    monkeypatch.setenv("LEKAI_PROMPT_CONTEXT_BEATS", "64")
    assert backend._prompt_context_beats() == 64


def test_generate_part1_tokens_preserves_generated_bar_token(monkeypatch):
    backend = LekaiHttpBackend()

    class _DummyModel:
        def __call__(self, input_ids, past_key_values=None, use_cache=True):
            _ = past_key_values, use_cache

            class _Output:
                logits = torch.zeros((1, input_ids.shape[1], 300), dtype=torch.float32)
                past_key_values = None

            return _Output()

    class _DummyAdapter:
        BAR_TOKEN = 255
        device = "cpu"
        use_cache = True
        model = _DummyModel()

    backend._model_adapter = _DummyAdapter()
    monkeypatch.setattr(
        "streammuse.infrastructure.inference.lekai_model.generation_utils.sample_token",
        lambda *args, **kwargs: torch.tensor([[255]], dtype=torch.long),
    )

    generated = backend._generate_part1_tokens_from_prompt(
        torch.tensor([257, 263, 265, 173, 255, 173, 169, 143, 6, 83, 2, 170]),
        temperature=0.0,
        top_k=1,
        top_p=0.0,
        repetition_penalty=1.2,
    )

    assert generated == [255]


def test_interleaved_prompt_reuses_exact_tokens_across_requests(monkeypatch):
    backend = LekaiHttpBackend()

    class _DummyAdapter:
        BAR_TOKEN = 255
        BOS_TOKEN = 257
        BPM_OFFSET_ID = 264
        TIME_SIG_OFFSET_ID = 259
        PAD_MARKER = 173
        model = object()
        tokenizer = object()

    class _DummyConverter:
        def events_to_pianoroll(
            self, events, start_tick, end_tick, active_pitches=None
        ):
            _ = events, active_pitches
            return np.zeros((2, 88, end_tick - start_tick), dtype=np.float32)

        def pianoroll_to_events(
            self, pianoroll, start_tick, close_at_end=False, active_pitches=None
        ):
            _ = pianoroll, start_tick, close_at_end
            return [], set(active_pitches or set())

    backend._model_adapter = _DummyAdapter()
    backend._converter = _DummyConverter()
    backend._tokenizer = object()
    monkeypatch.setattr(backend._logger, "log_generation", lambda **kwargs: None)
    monkeypatch.setattr(
        "streammuse.infrastructure.inference.lekai_model.inference_adapter.beats_to_pianoroll",
        lambda *args, **kwargs: np.zeros((2, 88, 4), dtype=np.float32),
    )

    encoded_calls = []

    def _encode(events, beat_start_tick, active_pitches, end_marker):
        _ = events
        encoded_calls.append((beat_start_tick, end_marker))
        return torch.tensor([169], dtype=torch.long), set(active_pitches)

    generated_beats = iter(([255], [169]))
    prompts = []

    def _generate(prompt_tokens, **kwargs):
        _ = kwargs
        prompts.append(prompt_tokens.tolist())
        return list(next(generated_beats))

    monkeypatch.setattr(backend, "_encode_beat_tokens", _encode)
    monkeypatch.setattr(backend, "_generate_part1_tokens_from_prompt", _generate)

    backend._generate_with_interleaved_prompt(
        generation_start_tick=4,
        generation_interval_ticks=4,
        generation_length_frames=4,
    )
    assert backend._accompaniment_token_history[1] == [255]

    encoded_calls.clear()
    backend._generate_with_interleaved_prompt(
        generation_start_tick=8,
        generation_interval_ticks=4,
        generation_length_frames=4,
    )

    assert (4, 171) not in encoded_calls
    assert (4, 170) in encoded_calls
    assert prompts[1][-2:] == [255, 169]


def test_measure_boundary_slot_is_generated_in_offline_order(monkeypatch):
    backend = LekaiHttpBackend()

    class _DummyAdapter:
        BAR_TOKEN = 255
        BOS_TOKEN = 257
        BPM_OFFSET_ID = 264
        TIME_SIG_OFFSET_ID = 259
        PAD_MARKER = 173
        model = object()
        tokenizer = object()

    class _DummyConverter:
        def events_to_pianoroll(
            self, events, start_tick, end_tick, active_pitches=None
        ):
            _ = events, active_pitches
            return np.zeros((2, 88, end_tick - start_tick), dtype=np.float32)

        def pianoroll_to_events(
            self, pianoroll, start_tick, close_at_end=False, active_pitches=None
        ):
            _ = pianoroll, start_tick, close_at_end
            return [], set(active_pitches or set())

    backend._model_adapter = _DummyAdapter()
    backend._converter = _DummyConverter()
    backend._tokenizer = object()
    monkeypatch.setattr(backend._logger, "log_generation", lambda **kwargs: None)

    decoded_beats = []

    def _decode(beats, *args, **kwargs):
        _ = args, kwargs
        decoded_beats.append(beats)
        return np.zeros((2, 88, 4), dtype=np.float32)

    monkeypatch.setattr(
        "streammuse.infrastructure.inference.lekai_model.inference_adapter.beats_to_pianoroll",
        _decode,
    )
    monkeypatch.setattr(
        backend,
        "_encode_beat_tokens",
        lambda events, beat_start_tick, active_pitches, end_marker: (
            torch.tensor([169], dtype=torch.long),
            set(active_pitches),
        ),
    )

    prompts = []
    generated_slots = iter(
        ([255], [258, 140, 7, 84, 63, 171])
    )

    def _generate(prompt_tokens, **kwargs):
        _ = kwargs
        prompts.append(prompt_tokens.tolist())
        return list(next(generated_slots))

    monkeypatch.setattr(backend, "_generate_part1_tokens_from_prompt", _generate)

    backend._generate_with_interleaved_prompt(
        generation_start_tick=16,
        generation_interval_ticks=4,
        generation_length_frames=4,
    )

    assert len(prompts) == 2
    assert prompts[1][-2:] == [255, 255]
    assert backend._accompaniment_token_history[4] == [255]
    assert backend._accompaniment_bar_token_history[4] == [258, 140, 7, 84, 63, 171]
    assert decoded_beats == [[[140, 7, 84, 63, 171]]]


def test_non_bar_boundary_slot_is_played_while_structure_runs_deferred(monkeypatch):
    backend = LekaiHttpBackend()

    class _DummyAdapter:
        BAR_TOKEN = 255
        BOS_TOKEN = 257
        BPM_OFFSET_ID = 264
        TIME_SIG_OFFSET_ID = 259
        PAD_MARKER = 173
        model = object()
        tokenizer = object()

    class _DummyConverter:
        def events_to_pianoroll(
            self, events, start_tick, end_tick, active_pitches=None
        ):
            _ = events, active_pitches
            return np.zeros((2, 88, end_tick - start_tick), dtype=np.float32)

        def pianoroll_to_events(
            self, pianoroll, start_tick, close_at_end=False, active_pitches=None
        ):
            _ = pianoroll, start_tick, close_at_end
            return [], set(active_pitches or set())

    backend._model_adapter = _DummyAdapter()
    backend._converter = _DummyConverter()
    backend._tokenizer = object()
    monkeypatch.setattr(backend._logger, "log_generation", lambda **kwargs: None)
    monkeypatch.setattr(
        backend,
        "_encode_beat_tokens",
        lambda events, beat_start_tick, active_pitches, end_marker: (
            torch.tensor([169], dtype=torch.long),
            set(active_pitches),
        ),
    )

    decoded_beats = []
    monkeypatch.setattr(
        "streammuse.infrastructure.inference.lekai_model.inference_adapter.beats_to_pianoroll",
        lambda beats, *args, **kwargs: (
            decoded_beats.append(beats)
            or np.zeros((2, 88, 4), dtype=np.float32)
        ),
    )
    generated = [140, 7, 84, 63, 171]
    monkeypatch.setattr(
        backend,
        "_generate_part1_tokens_from_prompt",
        lambda prompt_tokens, **kwargs: list(generated),
    )

    submitted_prompts = []

    def _submit(beat, prompt_tokens, **kwargs):
        _ = kwargs
        submitted_prompts.append((beat, prompt_tokens.tolist()))
        future = Future()
        future.set_result([169])
        backend._pending_boundary_generations[beat] = future

    monkeypatch.setattr(backend, "_submit_boundary_generation", _submit)

    backend._generate_with_interleaved_prompt(
        generation_start_tick=16,
        generation_interval_ticks=4,
        generation_length_frames=4,
    )

    assert decoded_beats == [[generated]]
    assert submitted_prompts[0][0] == 4
    assert submitted_prompts[0][1][-1] == 255
    assert backend._accompaniment_token_history[4] == generated
    assert 4 not in backend._accompaniment_bar_token_history

    backend._resolve_pending_boundary_generations(through_beat=4)

    assert backend._accompaniment_bar_token_history[4] == [169]
    assert backend._pending_boundary_generations == {}


def test_generate_respects_generation_length_cap(monkeypatch):
    monkeypatch.setenv("LEKAI_MAX_GENERATION_LENGTH_FRAMES", "8")
    backend = LekaiHttpBackend()

    accompaniment, _ = backend.generate(
        melody_events=[_note_on(60, 0)],
        generation_start_tick=4,
        generation_length_frames=20,
        generation_interval_ticks=4,
        prompt_length_ticks=None,
        inference_mode="sliding_window",
        model_name="lekai",
        checkpoint_path=None,
    )

    note_offs = [e for e in accompaniment if e["type"] == "note_off"]
    assert note_offs
    assert max(int(e["tick"]) for e in note_offs) == 12


def test_load_model_mps_failure_falls_back_to_cpu(monkeypatch, tmp_path):
    ckpt = tmp_path / "model.safetensors"
    ckpt.write_bytes(b"dummy")

    monkeypatch.setenv("LEKAI_DEVICE", "mps")
    monkeypatch.setenv("LEKAI_DTYPE", "auto")
    monkeypatch.setenv("LEKAI_ENABLE_MPS_FALLBACK", "true")
    monkeypatch.setenv("LEKAI_WARMUP_STEPS", "1")
    monkeypatch.setattr(
        "streammuse.infrastructure.inference.lekai_http_backend.resolve_device",
        lambda preference: "mps" if preference == "mps" else "cpu",
    )
    monkeypatch.setattr(
        "streammuse.infrastructure.inference.lekai_http_backend.resolve_dtype",
        lambda device, preference: torch.float16 if device == "mps" else torch.float32,
    )

    calls: list[str] = []

    class _DummyAdapter:
        BAR_TOKEN = 255

        def generate_from_beats(self, *args, **kwargs):
            return [[169]]

    def _fake_from_checkpoint(checkpoint_path: str, device: str, dtype=None, use_cache: bool = True):
        _ = checkpoint_path, dtype, use_cache
        calls.append(device)
        if device == "mps":
            raise RuntimeError("mps unsupported op")
        return _DummyAdapter()

    monkeypatch.setattr(
        "streammuse.infrastructure.inference.lekai_model.inference_adapter.PianoLLaMAAdapter.from_checkpoint",
        _fake_from_checkpoint,
    )
    monkeypatch.setattr(
        "streammuse.infrastructure.inference.lekai_model.MidiConverter.MidiConverter",
        lambda ticks_per_beat: object(),
    )
    monkeypatch.setattr(
        "streammuse.infrastructure.inference.lekai_model.my_tokenizer.PianoRollTokenizer",
        lambda patch_h, patch_w: object(),
    )

    backend = LekaiHttpBackend()
    backend._load_model(str(ckpt))

    info = backend.runtime_info()
    assert calls == ["mps", "cpu"]
    assert info["mode"] == "real_model"
    assert info["resolved_device"] == "cpu"
    assert str(info["fallback_reason"]).startswith("mps_load_failed:")


def test_generate_zero_prompt_window_with_model_path_falls_back_without_error():
    backend = LekaiHttpBackend()
    backend._model_adapter = object()
    backend._converter = object()
    backend._tokenizer = object()

    accompaniment, timings = backend.generate(
        melody_events=[_note_on(60, 0)],
        generation_start_tick=0,
        generation_length_frames=8,
        generation_interval_ticks=4,
        prompt_length_ticks=None,
        inference_mode="sliding_window",
        model_name="lekai",
        checkpoint_path=None,
    )

    assert isinstance(accompaniment, list)
    assert "response_output_time" in timings


def test_generate_recoverable_shape_mismatch_falls_back_to_rule_based(monkeypatch):
    backend = LekaiHttpBackend()

    class _DummyConverter:
        def events_to_pianoroll(self, events, start_tick, end_tick, active_pitches=None):
            _ = events, start_tick, end_tick, active_pitches
            return np.zeros((2, 88, 16), dtype=np.float32)

    class _DummyAdapter:
        BAR_TOKEN = 255

        def generate_from_beats(self, *args, **kwargs):
            _ = args, kwargs
            return [[169], [169]]

    backend._converter = _DummyConverter()
    backend._model_adapter = _DummyAdapter()
    backend._tokenizer = object()

    monkeypatch.setattr(
        "streammuse.infrastructure.inference.lekai_model.PianoDataset.process_measure_with_beat_interleaving",
        lambda *args, **kwargs: ([np.array([255], dtype=np.int64)], []),
    )
    monkeypatch.setattr(
        "streammuse.infrastructure.inference.lekai_model.inference_adapter.beats_to_pianoroll",
        lambda *args, **kwargs: np.zeros((2, 88, 0), dtype=np.float32),
    )

    accompaniment, _ = backend.generate(
        melody_events=[_note_on(60, 4)],
        generation_start_tick=8,
        generation_length_frames=8,
        generation_interval_ticks=4,
        prompt_length_ticks=None,
        inference_mode="sliding_window",
        model_name="lekai",
        checkpoint_path=None,
    )

    assert isinstance(accompaniment, list)
    assert accompaniment


def test_clear_history_returns_previous_histories_before_clearing():
    backend = LekaiHttpBackend()
    backend._melody_history = [_note_on(60, 0)]
    backend._accompaniment_history = [{"type": "note_on", "pitch": 48, "tick": 0, "velocity": 80}]
    backend._accompaniment_token_history = {0: [255]}
    backend._accompaniment_bar_token_history = {4: [255]}
    pending = Future()
    pending.set_result([169])
    backend._pending_boundary_generations = {8: pending}

    payload = backend.clear_history()

    assert payload["melody_history"][0]["pitch"] == 60
    assert payload["accompaniment_history"][0]["pitch"] == 48
    assert backend._melody_history == []
    assert backend._accompaniment_history == []
    assert backend._accompaniment_token_history == {}
    assert backend._accompaniment_bar_token_history == {}
    assert backend._pending_boundary_generations == {}


def test_reset_session_increments_epoch_applies_seed_and_retires_old_session():
    backend = LekaiHttpBackend()
    backend._melody_history = [_note_on(60, 0)]
    backend._input_digest_history = [_note_on(60, 0)]

    first = backend.reset_session(seed=123)
    second = backend.reset_session(seed=456)

    assert first["session_epoch"] == 1
    assert second["session_epoch"] == 2
    assert first["session_id"] != second["session_id"]
    assert second["effective_seed"] == 456
    assert backend._sample_generator.initial_seed() == 456
    assert backend._melody_history == []
    assert backend._input_digest_history == []
    with pytest.raises(SessionStateError, match="stale session epoch"):
        backend._validate_session(
            session_id=str(first["session_id"]),
            session_epoch=int(first["session_epoch"]),
        )


def test_reset_session_waits_for_pending_boundary_without_lock_inversion():
    backend = LekaiHttpBackend()
    future_started = threading.Event()
    reset_finished = threading.Event()
    result = {}

    backend._model_generation_lock.acquire()
    try:
        def boundary_work():
            future_started.set()
            with backend._model_generation_lock:
                return [169]

        backend._pending_boundary_generations[4] = backend._boundary_executor.submit(
            boundary_work
        )
        assert future_started.wait(timeout=1.0)

        def reset_work():
            result.update(backend.reset_session(seed=7))
            reset_finished.set()

        reset_thread = threading.Thread(target=reset_work, daemon=True)
        reset_thread.start()
        time.sleep(0.02)
        assert not reset_finished.is_set()
    finally:
        backend._model_generation_lock.release()

    reset_thread.join(timeout=1.0)
    assert reset_finished.is_set(), "reset deadlocked with pending boundary generation"
    assert result["effective_seed"] == 7
    assert result["pending_boundary_generations"] == 0
    assert backend._pending_boundary_generations == {}


def test_resetting_same_seed_replays_identical_sampling_sequence():
    backend = LekaiHttpBackend()

    class _UniformModel:
        def __call__(self, input_ids, past_key_values=None, use_cache=True):
            _ = past_key_values, use_cache

            class _Output:
                logits = torch.zeros((1, input_ids.shape[1], 300), dtype=torch.float32)
                past_key_values = None

            return _Output()

    class _Adapter:
        BAR_TOKEN = 255
        device = "cpu"
        use_cache = True
        model = _UniformModel()

    backend._model_adapter = _Adapter()
    prompt = torch.tensor([257, 263, 265, 173], dtype=torch.long)

    backend.reset_session(seed=20260710)
    first = backend._generate_part1_tokens_from_prompt(
        prompt,
        temperature=1.0,
        top_k=0,
        top_p=1.0,
        repetition_penalty=1.0,
    )
    backend.reset_session(seed=20260710)
    second = backend._generate_part1_tokens_from_prompt(
        prompt,
        temperature=1.0,
        top_k=0,
        top_p=1.0,
        repetition_penalty=1.0,
    )

    assert first == second


def test_generation_metadata_keeps_full_input_digest_after_prompt_history_trim(monkeypatch):
    monkeypatch.setenv("LEKAI_HISTORY_MAX_TICKS", "4")
    backend = LekaiHttpBackend()
    session = backend.reset_session(seed=99)
    cumulative = []
    metadata_rows = []

    increments = [
        [_note_on(60, 0)],
        [_note_on(62, 16)],
        [_note_on(64, 20)],
    ]
    generation_ticks = [4, 20, 24]
    for index, (increment, generation_tick) in enumerate(
        zip(increments, generation_ticks),
        start=1,
    ):
        request_id = f"req-{index}"
        backend.generate(
            melody_events=increment,
            generation_start_tick=generation_tick,
            generation_length_frames=4,
            generation_interval_ticks=4,
            prompt_length_ticks=None,
            inference_mode="sliding_window",
            model_name="lekai",
            checkpoint_path=None,
            session_id=str(session["session_id"]),
            session_epoch=int(session["session_epoch"]),
            request_id=request_id,
        )
        cumulative.extend(increment)
        metadata = backend.consume_generation_metadata(request_id)
        metadata_rows.append(metadata)
        assert metadata["input_increment_digest"] == backend._canonical_sha256(increment)
        assert metadata["input_cumulative_digest"] == backend._canonical_sha256(cumulative)

    required = {
        "request_id",
        "session_id",
        "session_epoch",
        "effective_seed",
        "generation_start_tick",
        "raw_tokens",
        "structural_tokens",
        "raw_token_digest",
        "prompt_token_digest",
        "part0_token_digest",
        "input_increment_digest",
        "input_cumulative_digest",
        "part0_roll_digest",
        "output_event_digest",
        "empty_success",
        "context_start_tick",
    }
    assert required <= metadata_rows[-1].keys()
    assert backend._input_digest_history == cumulative
    assert _note_on(60, 0) not in backend._melody_history
    # The rule-based fallback never supplies a model input roll.  It must not
    # substitute an event-table digest and pretend that it passed the formal
    # part0 converter gate.
    assert metadata_rows[-1]["part0_roll_digest"] is None
    assert metadata_rows[-1]["part0_roll_shape"] == []
    assert metadata_rows[-1]["part0_roll_digest"] != metadata_rows[-1][
        "input_cumulative_digest"
    ]
