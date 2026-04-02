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
