import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from streammuse.infrastructure.inference.lekai_prompt_continuation import prompt_engine
from streammuse.infrastructure.inference.lekai_prompt_continuation.prompt_engine import (
    LekaiPromptEngine,
)


def _count_markers(tokens, marker):
    return tokens.tolist().count(marker)


def test_prompt_engine_uses_time_signature_bar_length(monkeypatch):
    monkeypatch.setenv("LEKAI_PROMPT_TIME_SIGNATURE_INDEX", "2")
    monkeypatch.setenv("LEKAI_PROMPT_BPM", "80")
    engine = LekaiPromptEngine()

    tokens, _bpm, window_ticks = engine._build_melody_prompt_tokens(
        melody_events=[],
        prompt_start_tick=0,
        prompt_length_ticks=32,
    )

    vocab = engine._tokenizer.vocab
    assert window_ticks == 32
    assert _count_markers(tokens, vocab.bar_token_id) == 4
    assert _count_markers(tokens, vocab.beat_marker) == 8


def test_prompt_engine_defaults_to_common_time_bars(monkeypatch):
    monkeypatch.setenv("LEKAI_PROMPT_TIME_SIGNATURE_INDEX", "4")
    engine = LekaiPromptEngine()

    tokens, _bpm, window_ticks = engine._build_melody_prompt_tokens(
        melody_events=[],
        prompt_start_tick=0,
        prompt_length_ticks=32,
    )

    vocab = engine._tokenizer.vocab
    assert window_ticks == 32
    assert _count_markers(tokens, vocab.bar_token_id) == 2
    assert _count_markers(tokens, vocab.beat_marker) == 8


def test_prompt_engine_explicit_session_bpm_overrides_environment(monkeypatch):
    monkeypatch.setenv("LEKAI_PROMPT_BPM", "80")
    monkeypatch.setenv("LEKAI_DEFAULT_BPM", "70")
    engine = LekaiPromptEngine()

    tokens, effective_bpm, _window_ticks = engine._build_melody_prompt_tokens(
        melody_events=[],
        prompt_start_tick=0,
        prompt_length_ticks=32,
        bpm=220,
    )

    assert effective_bpm == 220
    assert tokens.tolist()[2] == engine._tokenizer.encode_bpm(220)


def test_prompt_engine_omitted_session_bpm_preserves_environment_fallback(monkeypatch):
    monkeypatch.setenv("LEKAI_DEFAULT_BPM", "70")
    monkeypatch.setenv("LEKAI_PROMPT_BPM", "80")
    engine = LekaiPromptEngine()

    _tokens, effective_bpm, _window_ticks = engine._build_melody_prompt_tokens(
        melody_events=[],
        prompt_start_tick=0,
        prompt_length_ticks=32,
    )

    assert effective_bpm == 80


def test_prompt_engine_condition_length_defaults_to_prompt_beats(monkeypatch):
    monkeypatch.setenv("LEKAI_PROMPT_TIME_SIGNATURE_INDEX", "2")
    engine = LekaiPromptEngine()

    assert engine._prompt_condition_length_ticks(32) == 32


def test_prompt_engine_condition_length_can_be_overridden_by_beats(monkeypatch):
    monkeypatch.setenv("LEKAI_PROMPT_TIME_SIGNATURE_INDEX", "2")
    monkeypatch.setenv("LEKAI_PROMPT_CONDITION_BEATS", "4")
    engine = LekaiPromptEngine()

    assert engine._prompt_condition_length_ticks(32) == 16


def test_prompt_engine_defaults_to_stanley_single_candidate_path(monkeypatch):
    monkeypatch.delenv("LEKAI_PROMPT_SELECTION_MODE", raising=False)
    engine = LekaiPromptEngine()

    info = engine.runtime_info()

    assert info["selection_mode"] == "single"
    assert info["batch_candidate_count"] == 1
    assert engine._generation_parameters("single") == {
        "temperature": 1.1,
        "top_k": 0,
        "top_p": 0.95,
        "repetition_penalty": 1.0,
    }


def test_prompt_engine_exposes_paired_batch_selection_modes(monkeypatch):
    monkeypatch.setenv("LEKAI_PROMPT_SELECTION_MODE", "batch_first")
    monkeypatch.setenv("LEKAI_PROMPT_BATCH_CANDIDATES", "5")
    engine = LekaiPromptEngine()

    assert engine.runtime_info()["batch_candidate_count"] == 5
    assert engine._generation_parameters("batch_first")["top_k"] == 50

    monkeypatch.setenv("LEKAI_PROMPT_SELECTION_MODE", "rule_s")
    assert engine.runtime_info()["selection_mode"] == "rule_s"


def test_prompt_engine_preserves_existing_sampling_defaults(monkeypatch):
    for name in (
        "LEKAI_PROMPT_TEMPERATURE",
        "LEKAI_PROMPT_TOP_K",
        "LEKAI_PROMPT_TOP_P",
        "LEKAI_PROMPT_REPETITION_PENALTY",
    ):
        monkeypatch.delenv(name, raising=False)
    engine = LekaiPromptEngine()

    assert engine._generation_parameters("single") == {
        "temperature": 1.1,
        "top_k": 0,
        "top_p": 0.95,
        "repetition_penalty": 1.0,
    }
    for mode in ("batch_first", "rule_s", "rule_s_v3"):
        assert engine._generation_parameters(mode) == {
            "temperature": 0.8,
            "top_k": 50,
            "top_p": 0.95,
            "repetition_penalty": 1.0,
        }


def test_prompt_engine_session_overrides_win_without_mutating_environment(monkeypatch):
    monkeypatch.setenv("LEKAI_PROMPT_SELECTION_MODE", "single")
    monkeypatch.setenv("LEKAI_PROMPT_BATCH_CANDIDATES", "6")
    monkeypatch.setenv("LEKAI_PROMPT_TEMPERATURE", "0.7")
    monkeypatch.setenv("LEKAI_PROMPT_TOP_P", "0.8")
    monkeypatch.setenv("LEKAI_PROMPT_TOP_K", "12")
    monkeypatch.setenv("LEKAI_PROMPT_REPETITION_PENALTY", "1.2")
    engine = LekaiPromptEngine()

    engine.set_session_generation_config(
        prompt_selection_mode="rule-s-if-else",
        prompt_batch_candidates=10,
        temperature=1.1,
        top_p=0.95,
        top_k=50,
        repetition_penalty=1.0,
    )

    info = engine.runtime_info()
    assert info["selection_mode"] == "rule_s_if_else"
    assert info["batch_candidate_count"] == 10
    assert {
        key: info[key]
        for key in ("temperature", "top_p", "top_k", "repetition_penalty")
    } == {
        "temperature": 1.1,
        "top_p": 0.95,
        "top_k": 50,
        "repetition_penalty": 1.0,
    }
    assert os.environ["LEKAI_PROMPT_TEMPERATURE"] == "0.7"


def test_prompt_engine_reset_restores_environment_sampling(monkeypatch):
    monkeypatch.setenv("LEKAI_PROMPT_SELECTION_MODE", "batch_first")
    monkeypatch.setenv("LEKAI_PROMPT_BATCH_CANDIDATES", "6")
    monkeypatch.setenv("LEKAI_PROMPT_TEMPERATURE", "0.7")
    monkeypatch.setenv("LEKAI_PROMPT_TOP_P", "0.8")
    monkeypatch.setenv("LEKAI_PROMPT_TOP_K", "12")
    monkeypatch.setenv("LEKAI_PROMPT_REPETITION_PENALTY", "1.2")
    engine = LekaiPromptEngine()
    engine.set_session_generation_config(
        prompt_selection_mode="rule_s",
        prompt_batch_candidates=5,
        temperature=1.1,
        top_p=0.95,
        top_k=50,
        repetition_penalty=1.0,
    )

    engine.reset_session(7)

    info = engine.runtime_info()
    assert info["selection_mode"] == "batch_first"
    assert info["batch_candidate_count"] == 6
    assert info["temperature"] == 0.7
    assert info["top_p"] == 0.8
    assert info["top_k"] == 12
    assert info["repetition_penalty"] == 1.2


def test_prompt_engine_accepts_rule_s_v3_aliases(monkeypatch):
    engine = LekaiPromptEngine()

    monkeypatch.setenv("LEKAI_PROMPT_SELECTION_MODE", "rule_s_v3")
    assert engine.runtime_info()["selection_mode"] == "rule_s_v3"

    monkeypatch.setenv("LEKAI_PROMPT_SELECTION_MODE", "rule-s-v3")
    assert engine.runtime_info()["selection_mode"] == "rule_s_v3"


def test_prompt_engine_accepts_rule_s_if_else_aliases_with_default_n10(monkeypatch):
    monkeypatch.delenv("LEKAI_PROMPT_BATCH_CANDIDATES", raising=False)
    engine = LekaiPromptEngine()

    monkeypatch.setenv("LEKAI_PROMPT_SELECTION_MODE", "rule_s_if_else")
    assert engine.runtime_info()["selection_mode"] == "rule_s_if_else"
    assert engine.runtime_info()["batch_candidate_count"] == 10

    monkeypatch.setenv("LEKAI_PROMPT_SELECTION_MODE", "rule-s-if-else")
    assert engine.runtime_info()["selection_mode"] == "rule_s_if_else"
    assert engine.runtime_info()["batch_candidate_count"] == 10


def test_prompt_engine_extracts_if_else_candidate_features():
    engine = LekaiPromptEngine()
    engine._model = object()
    roll = np.zeros((2, 88, 8), dtype=np.uint8)
    roll[0, [39, 43], 0] = 1
    roll[1, [39, 43], 0] = 1
    roll[0, 41, 4] = 1
    roll[1, 41, 4] = 1
    engine._tokenizer = SimpleNamespace(
        vocab=SimpleNamespace(eos_token_id=99, track_marker_acc=7),
        parse_generated_sequence=lambda _tokens: ([], [object(), object()]),
        decode_beats_to_pianoroll=lambda _beats, *, track_marker_id: roll,
    )

    candidate = engine._candidate_from_tokens(
        torch.tensor([1, 2], dtype=torch.long),
        candidate_number=1,
        prompt_length_ticks=8,
        ppl_score={"available": False},
        include_rule_s_if_else_features=True,
    )

    assert sum(candidate["acc_pitch_class_note_counts"]) == 3
    assert candidate["acc_pitch_class_note_entropy"] > 0
    assert candidate["acc_pitch_change_score"] == 1


def test_prompt_engine_if_else_passes_only_rank1_to_continuation(monkeypatch):
    monkeypatch.setenv("LEKAI_PROMPT_BATCH_CANDIDATES", "4")
    engine = LekaiPromptEngine()
    engine._model = object()
    generated = torch.tensor([[101], [202], [303], [404]], dtype=torch.long)

    monkeypatch.setattr(
        engine,
        "_generate_token_batch",
        lambda _prompt_tokens, *, candidate_count, seed_offset=0: generated,
    )
    monkeypatch.setattr(
        engine,
        "_candidate_from_tokens",
        lambda _sequence, *, candidate_number, **_kwargs: {
            "candidate_number": candidate_number
        },
    )
    monkeypatch.setattr(
        engine._tokenizer,
        "parse_generated_sequence",
        lambda _tokens: ([], []),
    )
    monkeypatch.setattr(
        prompt_engine,
        "score_prompt_batch_ppl",
        lambda *_args, **_kwargs: [{} for _ in range(4)],
    )
    monkeypatch.setattr(
        prompt_engine,
        "select_rule_s_if_else_candidates",
        lambda candidates: {
            "rule_id": "rule_s_if_else_v1",
            "selected_indices": [2, 0, 1],
            "selected_candidate_numbers": [3, 1, 2],
            "stage_2_candidate_numbers": [1, 2, 3, 4],
            "fallback_reason": None,
            "candidates": candidates,
        },
    )

    selected, metadata = engine._generate_batch_selected_tokens(
        torch.tensor([1], dtype=torch.long),
        prompt_length_ticks=32,
        selection_mode="rule_s_if_else",
    )

    assert selected.tolist() == [[303]]
    assert metadata["selected_candidate_number"] == 3
    assert metadata["selected_final_rank"] == 1
    assert metadata["selection_output_policy"] == "rank1_only"
    assert metadata["selection_attempt_count"] == 1
    assert metadata["selection_attempt_fallback_reasons"] == []
    assert metadata["ranked_candidate_numbers"] == [3, 1, 2]


def test_prompt_engine_if_else_resamples_with_next_seed_when_rank1_is_missing(
    monkeypatch,
):
    monkeypatch.setenv("LEKAI_PROMPT_BATCH_CANDIDATES", "2")
    monkeypatch.setenv("LEKAI_PROMPT_SEED", "100")
    engine = LekaiPromptEngine()
    engine._model = object()
    generated = torch.tensor([[101], [202]], dtype=torch.long)
    seed_offsets = []

    def generate(_prompt_tokens, *, candidate_count, seed_offset=0):
        seed_offsets.append(seed_offset)
        return generated

    monkeypatch.setattr(engine, "_generate_token_batch", generate)
    monkeypatch.setattr(
        engine,
        "_candidate_from_tokens",
        lambda _sequence, *, candidate_number, **_kwargs: {
            "candidate_number": candidate_number
        },
    )
    monkeypatch.setattr(
        engine._tokenizer,
        "parse_generated_sequence",
        lambda _tokens: ([], []),
    )
    monkeypatch.setattr(
        prompt_engine,
        "score_prompt_batch_ppl",
        lambda *_args, **_kwargs: [{}, {}],
    )
    decisions = iter(
        [
            {
                "rule_id": "rule_s_if_else_v1",
                "selected_indices": [],
                "selected_candidate_numbers": [],
                "stage_2_candidate_numbers": [],
                "fallback_reason": "no_candidate_with_minimum_note_count",
                "candidates": [],
            },
            {
                "rule_id": "rule_s_if_else_v1",
                "selected_indices": [1],
                "selected_candidate_numbers": [2],
                "stage_2_candidate_numbers": [2],
                "fallback_reason": "fewer_than_four_candidates_in_tonal_pool",
                "candidates": [{"candidate_number": 1}, {"candidate_number": 2}],
            },
        ]
    )
    monkeypatch.setattr(
        prompt_engine,
        "select_rule_s_if_else_candidates",
        lambda _candidates: next(decisions),
    )

    selected, metadata = engine._generate_batch_selected_tokens(
        torch.tensor([1], dtype=torch.long),
        prompt_length_ticks=32,
        selection_mode="rule_s_if_else",
    )

    assert selected.tolist() == [[202]]
    assert seed_offsets == [0, 1]
    assert metadata["selection_attempt_count"] == 2
    assert metadata["selection_attempt_fallback_reasons"] == [
        "no_candidate_with_minimum_note_count"
    ]
    assert metadata["selection_seed"] == 101


def test_prompt_engine_if_else_resampling_is_bounded(monkeypatch):
    monkeypatch.setenv("LEKAI_PROMPT_BATCH_CANDIDATES", "2")
    monkeypatch.setenv("LEKAI_PROMPT_SELECTION_MAX_ATTEMPTS", "2")
    engine = LekaiPromptEngine()
    engine._model = object()
    generated = torch.tensor([[101], [202]], dtype=torch.long)
    seed_offsets = []

    def generate(_prompt_tokens, *, candidate_count, seed_offset=0):
        seed_offsets.append(seed_offset)
        return generated

    monkeypatch.setattr(engine, "_generate_token_batch", generate)
    monkeypatch.setattr(
        engine,
        "_candidate_from_tokens",
        lambda _sequence, *, candidate_number, **_kwargs: {
            "candidate_number": candidate_number
        },
    )
    monkeypatch.setattr(
        engine._tokenizer,
        "parse_generated_sequence",
        lambda _tokens: ([], []),
    )
    monkeypatch.setattr(
        prompt_engine,
        "score_prompt_batch_ppl",
        lambda *_args, **_kwargs: [{}, {}],
    )
    monkeypatch.setattr(
        prompt_engine,
        "select_rule_s_if_else_candidates",
        lambda _candidates: {
            "rule_id": "rule_s_if_else_v1",
            "selected_indices": [],
            "selected_candidate_numbers": [],
            "stage_2_candidate_numbers": [],
            "fallback_reason": "no_candidate_with_minimum_note_count",
            "candidates": [],
        },
    )

    with pytest.raises(RuntimeError, match="after 2 attempts"):
        engine._generate_batch_selected_tokens(
            torch.tensor([1], dtype=torch.long),
            prompt_length_ticks=32,
            selection_mode="rule_s_if_else",
        )

    assert seed_offsets == [0, 1]


def test_prompt_engine_session_seed_overrides_environment_and_clears_diagnostics(monkeypatch):
    monkeypatch.setenv("LEKAI_PROMPT_SEED", "12")
    engine = LekaiPromptEngine()
    engine._last_prompt_token_ids = [1, 2]
    engine._last_generated_token_ids = [1, 2, 3]
    engine._last_new_token_ids = [3]
    engine._last_generated_acc_beats = 8
    engine._last_generation_metadata = {"selection_mode": "rule_s"}

    actual_seed = engine.reset_session(34)

    assert actual_seed == 34
    assert engine.runtime_info()["sample_seed"] == 34
    assert engine.last_generation_log()["prompt_tokens"] == []
    assert engine.last_generation_log()["generated_tokens"] == []
    assert engine.last_generation_log()["generated_acc_beats"] == 0
