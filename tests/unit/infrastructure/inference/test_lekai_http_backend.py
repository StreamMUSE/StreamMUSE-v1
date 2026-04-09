import numpy as np

from streammuse.infrastructure.inference.lekai_http_backend import LekaiHttpBackend


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


def test_trim_histories_keeps_recent_window_only():
    backend = LekaiHttpBackend()
    backend._melody_history = [_note_on(60, 0), _note_on(62, 40)]
    backend._accompaniment_history = [
        {"type": "note_off", "pitch": 48, "tick": 1, "velocity": 0},
        {"type": "note_off", "pitch": 50, "tick": 41, "velocity": 0},
    ]

    # max_history_ticks=20, cutoff=30 -> remove tick < 30
    backend._trim_histories(generation_start_tick=50, generation_length_frames=10)

    assert all(int(e["tick"]) >= 30 for e in backend._melody_history)
    assert all(int(e["tick"]) >= 30 for e in backend._accompaniment_history)


def test_runtime_info_contract_default_stub():
    backend = LekaiHttpBackend()
    info = backend.runtime_info()
    assert info["mode"] == "rule_stub"
    assert info["has_real_model"] is False
    assert "resolved_device" in info
    assert "resolved_dtype" in info


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

    payload = backend.clear_history()

    assert payload["melody_history"][0]["pitch"] == 60
    assert payload["accompaniment_history"][0]["pitch"] == 48
    assert backend._melody_history == []
    assert backend._accompaniment_history == []
