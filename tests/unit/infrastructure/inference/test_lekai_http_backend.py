from concurrent.futures import Future
import threading
import time

import numpy as np
import pytest
import torch

from streammuse.infrastructure.inference.lekai_continuation_model.my_tokenizer import (
    PianoMusicTokenizer,
)
from streammuse.infrastructure.inference.lekai_model.MidiConverter import MidiConverter
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
    melody_anchor = _note_on(60, 12)
    accompaniment_anchor = _note_on(48, 10)
    backend._melody_history = [
        _note_on(55, 0),
        {"type": "note_off", "pitch": 55, "tick": 4, "velocity": 0},
        _note_on(60, 8),
        melody_anchor,
        _note_on(62, 40),
        {"type": "note_off", "pitch": 62, "tick": 42, "velocity": 0},
        {"type": "note_off", "pitch": 60, "tick": 45, "velocity": 0},
    ]
    backend._accompaniment_history = [
        _note_on(46, 1),
        {"type": "note_off", "pitch": 46, "tick": 2, "velocity": 0},
        _note_on(48, 5),
        accompaniment_anchor,
        _note_on(50, 40),
        {"type": "note_off", "pitch": 48, "tick": 41, "velocity": 0},
        {"type": "note_off", "pitch": 50, "tick": 43, "velocity": 0},
    ]
    backend._accompaniment_token_history = {0: [255], 10: [169]}
    backend._accompaniment_bar_token_history = {0: [255], 10: [255]}

    # cutoff=30: keep recent events plus one original anchor per active pitch.
    backend._trim_histories(generation_start_tick=50, generation_length_frames=10)

    assert backend._melody_history == [
        melody_anchor,
        _note_on(62, 40),
        {"type": "note_off", "pitch": 62, "tick": 42, "velocity": 0},
        {"type": "note_off", "pitch": 60, "tick": 45, "velocity": 0},
    ]
    assert backend._melody_history[0] is melody_anchor
    assert backend._accompaniment_history == [
        accompaniment_anchor,
        _note_on(50, 40),
        {"type": "note_off", "pitch": 48, "tick": 41, "velocity": 0},
        {"type": "note_off", "pitch": 50, "tick": 43, "velocity": 0},
    ]
    assert backend._accompaniment_history[0] is accompaniment_anchor
    assert backend._active_pitches_before_tick(backend._melody_history, 30) == {60}
    assert backend._active_pitches_before_tick(
        backend._accompaniment_history, 30
    ) == {48}
    assert all(event["pitch"] != 55 for event in backend._melody_history)
    assert all(event["pitch"] != 46 for event in backend._accompaniment_history)
    assert backend._accompaniment_token_history == {10: [169]}
    assert backend._accompaniment_bar_token_history == {10: [255]}

    # cutoff=50: retained note_off events close both anchors before this cutoff.
    backend._trim_histories(generation_start_tick=70, generation_length_frames=10)

    assert backend._melody_history == []
    assert backend._accompaniment_history == []
    assert backend._accompaniment_token_history == {}
    assert backend._accompaniment_bar_token_history == {}


def test_runtime_info_contract_default_stub():
    backend = LekaiHttpBackend()
    info = backend.runtime_info()
    assert info["mode"] == "rule_stub"
    assert info["has_real_model"] is False
    assert "resolved_device" in info
    assert "resolved_dtype" in info


def test_session_sampling_overrides_environment_without_mutating_it(monkeypatch):
    monkeypatch.setenv("LEKAI_RT_TEMPERATURE", "0.7")
    monkeypatch.setenv("LEKAI_RT_TOP_P", "0.8")
    monkeypatch.setenv("LEKAI_RT_TOP_K", "12")
    monkeypatch.setenv("LEKAI_RT_REPETITION_PENALTY", "1.2")
    backend = LekaiHttpBackend()

    backend.set_session_generation_config(
        temperature=1.1,
        top_p=0.95,
        top_k=50,
        repetition_penalty=1.0,
    )

    assert backend._sampling_config() == {
        "temperature": 1.1,
        "top_p": 0.95,
        "top_k": 50,
        "repetition_penalty": 1.0,
    }
    assert backend.runtime_info()["temperature"] == 1.1
    assert backend.runtime_info()["top_k"] == 50
    assert backend.runtime_info()["top_p"] == 0.95
    assert backend.runtime_info()["repetition_penalty"] == 1.0


def test_reset_session_clears_sampling_overrides_to_environment(monkeypatch):
    monkeypatch.setenv("LEKAI_RT_TEMPERATURE", "0.7")
    monkeypatch.setenv("LEKAI_RT_TOP_P", "0.8")
    monkeypatch.setenv("LEKAI_RT_TOP_K", "12")
    monkeypatch.setenv("LEKAI_RT_REPETITION_PENALTY", "1.2")
    backend = LekaiHttpBackend()
    backend.set_session_generation_config(
        temperature=1.1,
        top_p=0.95,
        top_k=50,
        repetition_penalty=1.0,
    )

    backend.reset_session(seed=7)

    assert backend._sampling_config() == {
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 12,
        "repetition_penalty": 1.2,
    }


def test_prompt_context_defaults_to_retained_history_window(monkeypatch):
    monkeypatch.delenv("LEKAI_PROMPT_CONTEXT_BEATS", raising=False)
    backend = LekaiHttpBackend()

    assert backend._prompt_context_beats() == 128

    monkeypatch.setenv("LEKAI_PROMPT_CONTEXT_BEATS", "64")
    assert backend._prompt_context_beats() == 64


@pytest.mark.parametrize(
    ("sampled_tokens", "expected"),
    [
        ([170, 99, 255], [170]),
        ([172, 99, 255], [172]),
        ([255, 99], [255]),
        ([169, 170, 255], [169, 170]),
        ([171, 170, 255], [171, 170]),
        ([258, 170, 255], [258, 170]),
    ],
    ids=["acc-end", "beat", "bar", "empty", "mel-end", "pad"],
)
def test_generate_part1_tokens_uses_acc_structural_stops(
    monkeypatch, sampled_tokens, expected
):
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
    sampled = iter(sampled_tokens)
    monkeypatch.setattr(
        "streammuse.infrastructure.inference.lekai_model.generation_utils.sample_token",
        lambda *args, **kwargs: torch.tensor([[next(sampled)]], dtype=torch.long),
    )

    generated = backend._generate_part1_tokens_from_prompt(
        torch.tensor([257, 263, 265, 255, 172]),
        temperature=0.0,
        top_k=1,
        top_p=0.0,
        repetition_penalty=1.2,
    )

    assert generated == expected


@pytest.mark.parametrize(
    ("raw_tokens", "expected"),
    [
        ([258, 140, 170], [140, 170]),
        ([258, 172], [169, 170]),
        ([258, 255], [169, 170]),
        ([258, 170], [169, 170]),
        ([140, 255], [140, 170]),
        ([169, 170], [169, 170]),
    ],
)
def test_playable_part1_tokens_filters_pad_and_structural_stop(raw_tokens, expected):
    assert LekaiHttpBackend._playable_part1_tokens(raw_tokens) == expected


class _RollConverter:
    def __init__(self, roll):
        self.roll = roll

    def events_to_pianoroll(
        self, events, start_tick, end_tick, active_pitches=None
    ):
        _ = events, start_tick, end_tick, active_pitches
        return self.roll.copy()


@pytest.mark.parametrize("track_marker", [170, 171])
def test_encode_empty_beat_uses_real_continuation_codec(track_marker):
    backend = LekaiHttpBackend()
    backend._tokenizer = PianoMusicTokenizer()
    backend._converter = _RollConverter(np.zeros((2, 88, 4), dtype=np.float32))

    tokens, active = backend._encode_beat_tokens(
        events=[],
        beat_start_tick=0,
        active_pitches={48},
        end_marker=track_marker,
    )

    assert tokens.tolist() == [169, track_marker]
    assert active == {48}


def test_nonempty_beat_round_trips_through_real_continuation_codec():
    roll = np.zeros((2, 88, 4), dtype=np.float32)
    roll[0, 0, :] = 1
    backend = LekaiHttpBackend()
    backend._tokenizer = PianoMusicTokenizer()
    backend._converter = _RollConverter(roll)

    tokens, _ = backend._encode_beat_tokens(
        events=[],
        beat_start_tick=0,
        active_pitches=set(),
        end_marker=170,
    )
    decoded = backend._decode_acc_beat_tokens(tokens.tolist())

    assert tokens.tolist() == [81, 40, 170]
    np.testing.assert_array_equal(decoded, roll)


def _install_interleaved_test_runtime(backend, monkeypatch):
    class _DummyAdapter:
        model = object()
        tokenizer = PianoMusicTokenizer()

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
    backend._tokenizer = backend._model_adapter.tokenizer
    monkeypatch.setenv("LEKAI_DEFAULT_BPM", "120")
    monkeypatch.setenv("LEKAI_TIME_SIGNATURE_INDEX", "4")
    monkeypatch.setenv("LEKAI_PROMPT_CONTEXT_BEATS", "128")
    monkeypatch.delenv("LEKAI_MEASURE_BEATS", raising=False)

    decoded_beats = []
    monkeypatch.setattr(
        backend,
        "_decode_acc_beat_tokens",
        lambda tokens: (
            decoded_beats.append([tokens])
            or np.zeros((2, 88, 4), dtype=np.float32)
        ),
    )
    return decoded_beats


def test_interleaved_generation_uses_session_sampling_overrides(monkeypatch):
    backend = LekaiHttpBackend()
    _install_interleaved_test_runtime(backend, monkeypatch)
    backend.set_session_generation_config(
        temperature=1.1,
        top_p=0.95,
        top_k=50,
        repetition_penalty=1.0,
    )
    monkeypatch.setattr(backend._logger, "log_generation", lambda **kwargs: None)
    generation_kwargs = []

    def _generate(prompt_tokens, **kwargs):
        _ = prompt_tokens
        generation_kwargs.append(dict(kwargs))
        return [169, 170]

    monkeypatch.setattr(backend, "_generate_part1_tokens_from_prompt", _generate)

    backend._generate_with_interleaved_prompt(
        generation_start_tick=4,
        generation_interval_ticks=4,
        generation_length_frames=4,
    )

    assert generation_kwargs == [
        {
            "temperature": 1.1,
            "top_k": 50,
            "top_p": 0.95,
            "repetition_penalty": 1.0,
        }
    ]


def test_trimmed_active_anchors_preserve_sustain_in_continuation_prompt(monkeypatch):
    backend = LekaiHttpBackend()
    _install_interleaved_test_runtime(backend, monkeypatch)
    backend._converter = MidiConverter(ticks_per_beat=4)
    monkeypatch.setenv("LEKAI_HISTORY_MAX_TICKS", "20")
    monkeypatch.setenv("LEKAI_PROMPT_CONTEXT_BEATS", "5")
    monkeypatch.setattr(backend._logger, "log_generation", lambda **kwargs: None)

    melody_anchor = _note_on(60, 4)
    accompaniment_anchor = _note_on(48, 8)
    backend._melody_history = [
        melody_anchor,
        {"type": "note_off", "pitch": 60, "tick": 44, "velocity": 0},
    ]
    backend._accompaniment_history = [
        accompaniment_anchor,
        {"type": "note_off", "pitch": 48, "tick": 44, "velocity": 0},
    ]
    backend._trim_histories(generation_start_tick=40, generation_length_frames=4)

    assert backend._melody_history[0] is melody_anchor
    assert backend._accompaniment_history[0] is accompaniment_anchor
    assert all(
        not (event["type"] == "note_on" and int(event["tick"]) == 20)
        for event in backend._melody_history + backend._accompaniment_history
    )

    prompts = []
    monkeypatch.setattr(
        backend,
        "_generate_part1_tokens_from_prompt",
        lambda prompt_tokens, **kwargs: (
            prompts.append(prompt_tokens.tolist()) or [169, 170]
        ),
    )

    backend._generate_with_interleaved_prompt(
        generation_start_tick=40,
        generation_interval_ticks=4,
        generation_length_frames=4,
    )

    # Sustain-only patch 40 is retained for both tracks; onset patch 67 is absent.
    assert prompts[0][:11] == [
        257,
        263,
        265,
        255,
        172,
        108,
        40,
        170,
        120,
        40,
        171,
    ]
    assert 67 not in prompts[0]
    assert backend._current_generation_trace["context_start_tick"] == 20
    assert backend._current_generation_trace[
        "token_decode_initial_active_pitches"
    ] == [48]
    part0_roll = backend._current_generation_trace["part0_roll"]
    assert part0_roll[0, 39, 0] == 1
    assert part0_roll[1, 39, 0] == 0


def test_interleaved_prompt_rebuilds_history_from_events_across_requests(monkeypatch):
    backend = LekaiHttpBackend()
    _install_interleaved_test_runtime(backend, monkeypatch)
    generation_logs = []
    monkeypatch.setattr(
        backend._logger,
        "log_generation",
        lambda **kwargs: generation_logs.append(kwargs),
    )

    encoded_calls = []

    def _encode(events, beat_start_tick, active_pitches, end_marker):
        _ = events
        encoded_calls.append((beat_start_tick, end_marker))
        token = (100 if end_marker == 170 else 200) + beat_start_tick // 4
        return torch.tensor([token, end_marker], dtype=torch.long), set(active_pitches)

    generated_beats = iter(([140, 170], [141, 170]))
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

    assert encoded_calls == [(0, 170), (0, 171)]
    assert prompts[0] == [257, 263, 265, 255, 172, 100, 170, 200, 171, 172]
    assert backend._accompaniment_token_history[1] == [140, 170]
    assert generation_logs[0]["prompt_tokens"] == prompts[0]
    assert generation_logs[0]["diagnostics"] == {
        "context_start_tick": 0,
        "current_beat": 1,
        "start_beat": 0,
        "num_beats_to_generate": 1,
        "beat_diagnostics": [
            {
                "target_beat": 1,
                "beat_start_tick": 4,
                "prompt_token_count": 10,
                "generated_tokens": [140, 170],
                "generated_token_count": 2,
                "pianoroll_nonzero": 0,
                "event_count": 0,
                "note_on_count": 0,
                "min_event_tick": None,
                "max_event_tick": None,
            }
        ],
    }

    backend._accompaniment_token_history[1] = [258, 99, 170]
    encoded_calls.clear()
    backend._generate_with_interleaved_prompt(
        generation_start_tick=8,
        generation_interval_ticks=4,
        generation_length_frames=4,
    )

    assert encoded_calls == [
        (0, 170),
        (0, 171),
        (4, 170),
        (4, 171),
    ]
    assert prompts[1] == [
        257,
        263,
        265,
        255,
        172,
        100,
        170,
        200,
        171,
        172,
        101,
        170,
        201,
        171,
        172,
    ]
    assert backend._accompaniment_token_history[1] == [99, 170]
    assert 258 not in backend._accompaniment_token_history[1]


def test_interleaved_prompt_matches_stable_grammar_across_bar_boundary(monkeypatch):
    backend = LekaiHttpBackend()
    decoded_beats = _install_interleaved_test_runtime(backend, monkeypatch)
    backend._converter = MidiConverter(ticks_per_beat=4)
    generation_logs = []
    monkeypatch.setattr(
        backend._logger,
        "log_generation",
        lambda **kwargs: generation_logs.append(kwargs),
    )

    prompts = []
    generated_beats = iter(
        ([258, 140, 170], [258, 141, 170], [258, 142, 170])
    )

    def _generate(prompt_tokens, **kwargs):
        _ = kwargs
        prompts.append(prompt_tokens.tolist())
        return list(next(generated_beats))

    monkeypatch.setattr(backend, "_generate_part1_tokens_from_prompt", _generate)
    monkeypatch.setattr(
        backend,
        "_submit_boundary_generation",
        lambda *args, **kwargs: pytest.fail("boundary generation must not be submitted"),
    )

    empty_roll = np.zeros((2, 88, 4), dtype=np.float32)
    note_roll = empty_roll.copy()
    note_roll[0, 39, :] = 1
    note_roll[1, 39, 0] = 1
    decoded_rolls = iter((empty_roll, note_roll, empty_roll))

    def _decode(tokens):
        decoded_beats.append([tokens])
        return next(decoded_rolls)

    monkeypatch.setattr(backend, "_decode_acc_beat_tokens", _decode)

    generated_events = backend._generate_with_interleaved_prompt(
        generation_start_tick=12,
        generation_interval_ticks=4,
        generation_length_frames=12,
    )

    first_prompt = [
        257,
        263,
        265,
        255,
        172,
        169,
        170,
        169,
        171,
        172,
        169,
        170,
        169,
        171,
        172,
        169,
        170,
        169,
        171,
        172,
    ]
    second_prompt = first_prompt + [169, 170, 169, 171, 255, 172]
    third_prompt = second_prompt + [120, 67, 170, 169, 171, 172]
    assert prompts == [first_prompt, second_prompt, third_prompt]
    assert generated_events == [
        {"type": "note_on", "pitch": 60, "tick": 16},
        {"type": "note_off", "pitch": 60, "tick": 20},
    ]
    assert 140 not in second_prompt
    assert second_prompt[-6:-2] == [169, 170, 169, 171]
    assert 141 not in third_prompt
    assert third_prompt[-6:-3] == [120, 67, 170]
    assert len(prompts) == 3
    assert [prompt.count(172) for prompt in prompts] == [4, 5, 6]
    assert [prompt.count(170) for prompt in prompts] == [3, 4, 5]
    assert [prompt.count(171) for prompt in prompts] == [3, 4, 5]
    assert all(173 not in prompt for prompt in prompts)
    assert prompts[1][-2:] == [255, 172]
    assert decoded_beats == [[[140, 170]], [[141, 170]], [[142, 170]]]
    assert backend._accompaniment_token_history == {
        3: [140, 170],
        4: [141, 170],
        5: [142, 170],
    }
    assert all(
        258 not in tokens for tokens in backend._accompaniment_token_history.values()
    )
    assert backend._accompaniment_bar_token_history == {}
    assert backend._pending_boundary_generations == {}
    assert backend._current_generation_trace["raw_tokens"] == [
        258,
        140,
        170,
        258,
        141,
        170,
        258,
        142,
        170,
    ]
    assert backend._current_generation_trace["token_decode_beats"] == [
        {
            "target_beat": 3,
            "start_tick": 12,
            "raw_tokens": [258, 140, 170],
            "boundary_tokens": [],
        },
        {
            "target_beat": 4,
            "start_tick": 16,
            "raw_tokens": [258, 141, 170],
            "boundary_tokens": [],
        },
        {
            "target_beat": 5,
            "start_tick": 20,
            "raw_tokens": [258, 142, 170],
            "boundary_tokens": [],
        },
    ]
    assert backend._current_generation_trace["prompt_tokens"] == prompts[-1]
    assert backend._current_generation_trace["part0_tokens"] == [169, 171] * 5
    assert generation_logs[0]["prompt_tokens"] == prompts[-1]
    assert backend._current_generation_trace["structural_tokens"] == []


@pytest.mark.parametrize(
    ("time_signature_idx", "expected_beats"),
    [(0, 4), (1, 3), (2, 2), (3, 3), (4, 4), (6, 6), (9, 4), (99, 4)],
)
def test_measure_beats_matches_stable_time_signature_mapping(
    monkeypatch, time_signature_idx, expected_beats
):
    monkeypatch.delenv("LEKAI_MEASURE_BEATS", raising=False)
    backend = LekaiHttpBackend()

    assert (
        backend._measure_beats_from_time_signature_idx(time_signature_idx)
        == expected_beats
    )


def test_measure_override_controls_bars_from_context_window_start(monkeypatch):
    backend = LekaiHttpBackend()
    _install_interleaved_test_runtime(backend, monkeypatch)
    monkeypatch.setenv("LEKAI_PROMPT_CONTEXT_BEATS", "3")
    monkeypatch.setenv("LEKAI_MEASURE_BEATS", "3")
    monkeypatch.setattr(backend._logger, "log_generation", lambda **kwargs: None)

    def _encode(events, beat_start_tick, active_pitches, end_marker):
        _ = events
        token = (100 if end_marker == 170 else 200) + beat_start_tick // 4
        return torch.tensor([token, end_marker], dtype=torch.long), set(active_pitches)

    prompts = []
    monkeypatch.setattr(backend, "_encode_beat_tokens", _encode)
    monkeypatch.setattr(
        backend,
        "_generate_part1_tokens_from_prompt",
        lambda prompt_tokens, **kwargs: (
            prompts.append(prompt_tokens.tolist()) or [150, 170]
        ),
    )

    backend._generate_with_interleaved_prompt(
        generation_start_tick=28,
        generation_interval_ticks=4,
        generation_length_frames=4,
    )

    assert prompts == [
        [
            257,
            263,
            265,
            255,
            172,
            104,
            170,
            204,
            171,
            172,
            105,
            170,
            205,
            171,
            255,
            172,
            106,
            170,
            206,
            171,
            172,
        ]
    ]
    assert backend._current_generation_trace["context_start_tick"] == 16


def test_target_active_pitches_are_derived_from_event_history(monkeypatch):
    backend = LekaiHttpBackend()
    _install_interleaved_test_runtime(backend, monkeypatch)
    monkeypatch.setattr(backend._logger, "log_generation", lambda **kwargs: None)
    backend._accompaniment_history = [_note_on(48, 0)]
    backend._active_pitches = {99}

    monkeypatch.setattr(
        backend,
        "_encode_beat_tokens",
        lambda events, beat_start_tick, active_pitches, end_marker: (
            torch.tensor([169, end_marker], dtype=torch.long),
            set(active_pitches),
        ),
    )
    monkeypatch.setattr(
        backend,
        "_generate_part1_tokens_from_prompt",
        lambda prompt_tokens, **kwargs: [169, 170],
    )
    active_snapshots = []

    def _convert(pianoroll, start_tick, close_at_end=False, active_pitches=None):
        _ = pianoroll, start_tick, close_at_end
        active_snapshots.append(set(active_pitches or set()))
        return [], set(active_pitches or set())

    monkeypatch.setattr(backend._converter, "pianoroll_to_events", _convert)

    backend._generate_with_interleaved_prompt(
        generation_start_tick=8,
        generation_interval_ticks=4,
        generation_length_frames=4,
    )

    assert active_snapshots == [{48}]
    assert backend._active_pitches == {48}
    assert backend._current_generation_trace[
        "token_decode_initial_active_pitches"
    ] == [48]


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
    monkeypatch.setenv("LEKAI_DEFAULT_BPM", "120")
    monkeypatch.setenv("LEKAI_TIME_SIGNATURE_INDEX", "4")
    monkeypatch.setattr(
        "streammuse.infrastructure.inference.lekai_http_backend.resolve_device",
        lambda preference: "mps" if preference == "mps" else "cpu",
    )
    monkeypatch.setattr(
        "streammuse.infrastructure.inference.lekai_http_backend.resolve_dtype",
        lambda device, preference: torch.float16 if device == "mps" else torch.float32,
    )

    calls: list[str] = []
    adapters = []
    warmup_calls = []

    class _DummyAdapter:
        def __init__(self):
            self.tokenizer = PianoMusicTokenizer()

    def _fake_from_checkpoint(
        checkpoint_path: str,
        device: str,
        dtype=None,
        use_cache: bool = True,
    ):
        _ = checkpoint_path, dtype, use_cache
        calls.append(device)
        if device == "mps":
            raise RuntimeError("mps unsupported op")
        adapter = _DummyAdapter()
        adapters.append(adapter)
        return adapter

    monkeypatch.setattr(
        "streammuse.infrastructure.inference.lekai_continuation_model.inference_adapter.PianoContinuationAdapter.from_checkpoint",
        _fake_from_checkpoint,
    )
    monkeypatch.setattr(
        "streammuse.infrastructure.inference.lekai_model.MidiConverter.MidiConverter",
        lambda ticks_per_beat: object(),
    )

    def _warmup_generate(self, prompt_tokens, **kwargs):
        _ = self
        warmup_calls.append((prompt_tokens.tolist(), kwargs))
        return [170]

    monkeypatch.setattr(
        LekaiHttpBackend,
        "_generate_part1_tokens_from_prompt",
        _warmup_generate,
    )

    backend = LekaiHttpBackend()
    backend._load_model(str(ckpt))

    info = backend.runtime_info()
    assert calls == ["mps", "cpu"]
    assert info["mode"] == "real_model"
    assert info["resolved_device"] == "cpu"
    assert str(info["fallback_reason"]).startswith("mps_load_failed:")
    assert backend._tokenizer is adapters[0].tokenizer
    assert warmup_calls == [
        (
            [257, 263, 265, 255, 172],
            {
                "temperature": 0.8,
                "top_k": 20,
                "top_p": 0.9,
                "repetition_penalty": 1.0,
            },
        )
    ]


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


def test_generation_metadata_snapshot_is_ordered_read_only_and_cleared():
    backend = LekaiHttpBackend()

    for index, generation_tick in enumerate((4, 8), start=1):
        backend.generate(
            melody_events=[_note_on(59 + index, generation_tick - 4)],
            generation_start_tick=generation_tick,
            generation_length_frames=4,
            generation_interval_ticks=4,
            prompt_length_ticks=None,
            inference_mode="sliding_window",
            model_name="lekai",
            checkpoint_path=None,
            request_id=f"audit-{index}",
        )

    first_snapshot = backend.generation_metadata_snapshot()
    second_snapshot = backend.generation_metadata_snapshot()

    assert [row["request_id"] for row in first_snapshot] == ["audit-1", "audit-2"]
    assert first_snapshot == second_snapshot
    assert {
        "raw_tokens",
        "structural_tokens",
        "raw_token_digest",
        "token_decode_digest",
        "prompt_token_digest",
        "part0_token_digest",
        "input_increment_digest",
        "input_cumulative_digest",
        "part0_roll_digest",
        "output_event_digest",
    } <= first_snapshot[0].keys()

    first_snapshot[0]["raw_tokens"].append(-1)
    assert backend.generation_metadata_snapshot()[0]["raw_tokens"] != first_snapshot[0][
        "raw_tokens"
    ]

    backend.clear_history()
    assert backend.generation_metadata_snapshot() == []


def test_generation_metadata_snapshot_does_not_wait_for_model_session_gate():
    backend = LekaiHttpBackend()
    with backend._generation_metadata_lock:
        backend._generation_metadata = {
            "completed": {"request_id": "completed", "raw_tokens": [169]}
        }

    gate_entered = threading.Event()
    release_gate = threading.Event()
    snapshot_finished = threading.Event()
    snapshot_result = []

    def hold_generation_gate():
        with backend._session_gate:
            gate_entered.set()
            release_gate.wait(timeout=2.0)

    def read_snapshot():
        snapshot_result.extend(backend.generation_metadata_snapshot())
        snapshot_finished.set()

    holder = threading.Thread(target=hold_generation_gate, daemon=True)
    reader = threading.Thread(target=read_snapshot, daemon=True)
    holder.start()
    assert gate_entered.wait(timeout=1.0)
    reader.start()
    try:
        assert snapshot_finished.wait(timeout=0.5)
        assert snapshot_result == [
            {"request_id": "completed", "raw_tokens": [169]}
        ]
    finally:
        release_gate.set()
        holder.join(timeout=1.0)
        reader.join(timeout=1.0)


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
        [
            _note_on(60, 0),
            {"type": "note_off", "pitch": 60, "tick": 1, "velocity": 0},
        ],
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
    assert metadata_rows[-1]["part0_trace_available"] is False
    assert metadata_rows[-1]["part0_roll_shape"] == []
    assert metadata_rows[-1]["part0_roll_digest"] != metadata_rows[-1][
        "input_cumulative_digest"
    ]
