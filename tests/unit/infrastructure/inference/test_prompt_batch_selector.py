import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from streammuse.infrastructure.inference.lekai_prompt_continuation.prompt_batch_selector import (
    RULE_S_V3_DURATION_EPSILON,
    RULE_S_V3_DURATION_TAU,
    RULE_S_V3_WEIGHTS,
    accompaniment_features_from_pianoroll,
    duration_match_score,
    low_register_penalty_from_pianoroll,
    median_note_duration_from_pianoroll,
    pitch_change_score_from_pianoroll,
    pitch_class_note_distribution_from_pianoroll,
    score_prompt_batch_ppl,
    select_rule_s_candidate,
    select_rule_s_if_else_candidates,
    select_rule_s_v2_candidate,
    select_rule_s_v3_candidate,
    tonal_fit_score,
)


def _candidate(
    number,
    *,
    ppl,
    pitch_range,
    entropy,
    voices,
    notes=4,
    beats=8,
    mel_duration=1.0,
    acc_duration=1.0,
    mel_evidence=None,
    acc_evidence=None,
    low_penalty=0.0,
):
    mel_evidence = mel_evidence or [8.0, 0, 0, 0, 4.0, 0, 0, 4.0, 0, 0, 0, 0]
    acc_evidence = acc_evidence or [8.0, 0, 0, 0, 4.0, 0, 0, 4.0, 0, 0, 0, 0]
    return {
        "candidate_number": number,
        "generated_beats": beats,
        "required_beats": 8,
        "acc_note_count": notes,
        "prompt_ppl_available": ppl is not None,
        "prompt_ppl": ppl,
        "acc_pitch_range": pitch_range,
        "acc_pitch_class_entropy": entropy,
        "acc_average_voice_number": voices,
        "mel_median_note_duration_ticks": mel_duration,
        "acc_median_note_duration_ticks": acc_duration,
        "mel_pitch_class_duration_evidence": mel_evidence,
        "acc_pitch_class_duration_evidence": acc_evidence,
        "low_register_penalty": low_penalty,
    }


def test_accompaniment_features_use_note_duration_and_concurrency():
    roll = np.zeros((2, 88, 8), dtype=np.uint8)
    roll[0, 39, 0:4] = 1
    roll[1, 39, 0] = 1
    roll[0, 43, 2:6] = 1
    roll[1, 43, 2] = 1

    features = accompaniment_features_from_pianoroll(roll, length_ticks=8)

    assert features["acc_note_count"] == 2
    assert features["acc_pitch_range"] == 4
    assert features["acc_pitch_class_entropy"] == 1.0
    assert features["acc_average_voice_number"] == 1.0
    assert features["acc_median_note_duration_ticks"] == 4.0


def test_median_note_duration_uses_onsets_and_clips_at_window_end():
    roll = np.zeros((2, 88, 8), dtype=np.uint8)
    roll[0, 39, 0:2] = 1
    roll[1, 39, 0] = 1
    roll[0, 43, 4:8] = 1
    roll[1, 43, 4] = 1

    assert median_note_duration_from_pianoroll(roll, length_ticks=8) == 3.0


def test_rule_s_filters_empty_and_incomplete_candidates():
    candidates = [
        _candidate(1, ppl=1.0, pitch_range=30, entropy=3.0, voices=4, notes=0),
        _candidate(2, ppl=1.0, pitch_range=30, entropy=3.0, voices=4, beats=7),
        _candidate(3, ppl=3.0, pitch_range=20, entropy=2.0, voices=3),
    ]

    decision = select_rule_s_candidate(candidates)

    assert decision["selected_candidate_number"] == 3
    assert decision["eligible_count"] == 1
    assert [row["eligible"] for row in decision["candidates"]] == [False, False, True]


def test_rule_s_uses_frozen_weighted_ranks_and_lower_ppl_tiebreak():
    candidates = [
        _candidate(1, ppl=2.0, pitch_range=10, entropy=1.0, voices=1),
        _candidate(2, ppl=2.0, pitch_range=40, entropy=3.0, voices=4),
        _candidate(3, ppl=1.5, pitch_range=25, entropy=2.0, voices=2),
    ]

    decision = select_rule_s_candidate(candidates)

    assert decision["selected_candidate_number"] == 2
    assert decision["fallback_reason"] is None


def test_rule_s_falls_back_to_first_candidate_when_all_are_ineligible():
    candidates = [
        _candidate(1, ppl=None, pitch_range=0, entropy=0, voices=0, notes=0),
        _candidate(2, ppl=None, pitch_range=0, entropy=0, voices=0, notes=0),
    ]

    decision = select_rule_s_candidate(candidates)

    assert decision["selected_index"] == 0
    assert decision["eligible_count"] == 0
    assert decision["fallback_reason"] == "no_eligible_candidate"


def test_rule_s_v2_penalizes_median_duration_mismatch_without_changing_v1():
    candidates = [
        _candidate(
            1,
            ppl=1.5,
            pitch_range=20,
            entropy=2.0,
            voices=2.0,
            acc_duration=1.0,
        ),
        _candidate(
            2,
            ppl=1.5,
            pitch_range=20,
            entropy=2.0,
            voices=3.0,
            acc_duration=8.0,
        ),
    ]

    v1 = select_rule_s_candidate(candidates)
    v2 = select_rule_s_v2_candidate(v1["candidates"], duration_weight=1.0)

    assert v1["selected_candidate_number"] == 2
    assert v2["selected_candidate_number"] == 1
    assert v2["candidates"][0]["duration_ratio"] == 1.0
    assert v2["candidates"][0]["duration_mismatch"] == 0.0
    assert v2["candidates"][1]["duration_mismatch"] > 2.0


def test_rule_s_v3_exact_formula_has_no_average_voice_term_and_preserves_v1():
    candidates = [
        _candidate(
            1, ppl=2.0, pitch_range=20, entropy=2.0, voices=1, low_penalty=0.25
        ),
        _candidate(
            2, ppl=2.0, pitch_range=20, entropy=2.0, voices=99, low_penalty=0.25
        ),
    ]

    v1_before = select_rule_s_candidate(candidates)
    v3 = select_rule_s_v3_candidate(candidates)
    v1_after = select_rule_s_candidate(candidates)

    row = v3["candidates"][0]
    expected = sum(
        RULE_S_V3_WEIGHTS[metric] * 0.75
        for metric in (
            "prompt_ppl",
            "acc_pitch_range",
            "acc_pitch_class_entropy",
            "duration_match",
            "tonal_fit",
        )
    ) + RULE_S_V3_WEIGHTS["low_register_penalty"] * 0.25
    assert row["rule_s_v3_score"] == pytest.approx(expected)
    assert "acc_average_voice_number" not in row["rule_s_v3_contributions"]
    assert v3["selected_candidate_number"] == 1
    assert v1_before == v1_after
    assert v1_before["selected_candidate_number"] == 2


def test_rule_s_v3_duration_match_prefers_equal_medians():
    exact = duration_match_score(4.0, 4.0)
    double = duration_match_score(8.0, 4.0)

    assert exact == 1.0
    assert double == pytest.approx(
        math.exp(
            -abs(
                math.log(
                    (8.0 + RULE_S_V3_DURATION_EPSILON)
                    / (4.0 + RULE_S_V3_DURATION_EPSILON)
                )
            )
            / RULE_S_V3_DURATION_TAU
        )
    )
    assert exact > double


def test_rule_s_v3_tonal_fit_prefers_compatible_pitch_classes():
    melody = [8.0, 0, 0, 0, 4.0, 0, 0, 4.0, 0, 0, 0, 0]
    compatible = [4.0, 0, 0, 0, 4.0, 0, 0, 4.0, 0, 0, 0, 0]
    incompatible = [0, 4.0, 0, 4.0, 0, 0, 4.0, 0, 4.0, 0, 4.0, 0]

    compatible_fit, confidence = tonal_fit_score(melody, compatible)
    incompatible_fit, _ = tonal_fit_score(melody, incompatible)

    assert confidence is not None and 0.0 < confidence <= 1.0
    assert compatible_fit is not None and incompatible_fit is not None
    assert compatible_fit > incompatible_fit


def test_rule_s_v3_low_register_penalty_is_absolute_duration_weighted_hinge():
    roll = np.zeros((2, 88, 8), dtype=np.uint8)
    roll[0, 0, 0:4] = 1  # MIDI 21: full hinge penalty.
    roll[1, 0, 0] = 1
    roll[0, 39, 4:8] = 1  # MIDI 60: no penalty.
    roll[1, 39, 4] = 1

    penalty = low_register_penalty_from_pianoroll(roll, length_ticks=8)
    decision = select_rule_s_v3_candidate(
        [
            _candidate(
                1,
                ppl=2,
                pitch_range=10,
                entropy=1,
                voices=1,
                low_penalty=penalty,
            )
        ]
    )

    assert penalty == 0.5
    assert decision["candidates"][0]["rule_s_v3_contributions"][
        "low_register_penalty"
    ] == -0.25


def test_rule_s_v3_missing_optional_metrics_remains_eligible():
    candidate = _candidate(
        1,
        ppl=2,
        pitch_range=10,
        entropy=1,
        voices=1,
        mel_duration=0,
        mel_evidence=[0.0] * 12,
    )

    decision = select_rule_s_v3_candidate([candidate])
    row = decision["candidates"][0]

    assert decision["selected_candidate_number"] == 1
    assert decision["eligible_count"] == 1
    assert row["duration_match"] is None
    assert row["tonal_fit"] is None
    assert row["rule_s_v3_contributions"]["duration_match"] == 0.0
    assert row["rule_s_v3_contributions"]["tonal_fit"] == 0.0


def test_rule_s_v3_ineligible_fallback_and_tie_are_deterministic():
    tied = [
        _candidate(1, ppl=2, pitch_range=10, entropy=1, voices=1),
        _candidate(2, ppl=2, pitch_range=10, entropy=1, voices=1),
    ]
    unavailable = [
        _candidate(1, ppl=None, pitch_range=0, entropy=0, voices=0, notes=0),
        _candidate(2, ppl=None, pitch_range=0, entropy=0, voices=0, notes=0),
    ]

    assert select_rule_s_v3_candidate(tied)["selected_candidate_number"] == 1
    fallback = select_rule_s_v3_candidate(unavailable)
    assert fallback["selected_index"] == 0
    assert fallback["fallback_reason"] == "no_eligible_candidate"


def test_if_else_pitch_change_matches_adjacent_onset_pitch_sets():
    roll = np.zeros((2, 88, 8), dtype=np.uint8)
    roll[0, [25, 30], 0] = 1
    roll[1, [25, 30], 0] = 1
    roll[0, [23, 28], 4] = 1
    roll[1, [23, 28], 4] = 1

    counts, entropy = pitch_class_note_distribution_from_pianoroll(
        roll, length_ticks=8
    )

    assert sum(counts) == 4
    assert entropy == 2.0
    assert pitch_change_score_from_pianoroll(roll, length_ticks=8) == 1


def test_if_else_stage_one_requires_complete_prompt_and_minimum_notes():
    melody = [8.0, 0, 0, 0, 4.0, 0, 0, 4.0, 0, 0, 0, 0]
    candidates = [
        {
            "candidate_number": 1,
            "generated_beats": 5,
            "required_beats": 6,
            "acc_note_count": 8,
            "mel_pitch_class_duration_evidence": melody,
            "acc_pitch_class_note_counts": [8] + [0] * 11,
            "acc_pitch_class_note_entropy": 0.0,
            "acc_pitch_change_score": 3,
        },
        {
            "candidate_number": 2,
            "generated_beats": 6,
            "required_beats": 6,
            "acc_note_count": 3,
            "mel_pitch_class_duration_evidence": melody,
            "acc_pitch_class_note_counts": [3] + [0] * 11,
            "acc_pitch_class_note_entropy": 0.0,
            "acc_pitch_change_score": 2,
        },
        {
            "candidate_number": 3,
            "generated_beats": 6,
            "required_beats": 6,
            "acc_note_count": 4,
            "mel_pitch_class_duration_evidence": melody,
            "acc_pitch_class_note_counts": [4] + [0] * 11,
            "acc_pitch_class_note_entropy": 0.0,
            "acc_pitch_change_score": 1,
        },
    ]

    decision = select_rule_s_if_else_candidates(candidates, min_notes=4)

    assert decision["stage_1_candidate_numbers"] == [3]
    assert decision["selected_candidate_numbers"] == [3]
    rows = {row["candidate_number"]: row for row in decision["candidates"]}
    assert rows[1]["complete_prompt"] is False
    assert rows[1]["stage_1_note_count_pass"] is True
    assert rows[1]["stage_1_pass"] is False
    assert rows[2]["complete_prompt"] is True
    assert rows[2]["stage_1_note_count_pass"] is False
    assert rows[2]["stage_1_pass"] is False
    assert rows[3]["stage_1_pass"] is True


def test_if_else_selector_keeps_change_top4_then_combines_entropy_and_note_ranks():
    base = {
        "generated_beats": 8,
        "required_beats": 8,
        "acc_note_count": 4,
        "mel_pitch_class_duration_evidence": [
            8.0,
            0,
            0,
            0,
            4.0,
            0,
            0,
            4.0,
            0,
            0,
            0,
            0,
        ],
        "acc_pitch_class_note_counts": [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
    }
    candidates = [
        {
            **base,
            "candidate_number": number,
            "acc_pitch_class_note_entropy": entropy,
            "acc_pitch_change_score": variation,
        }
        for number, (entropy, variation) in enumerate(
            [(2.0, 0), (1.5, 3), (1.0, 2), (0.5, 1)], start=1
        )
    ]

    decision = select_rule_s_if_else_candidates(candidates)

    assert decision["pitch_change_ranked_candidate_numbers"] == [2, 3, 4, 1]
    assert decision["pitch_change_rank_1_to_4_candidate_numbers"] == [2, 3, 4, 1]
    assert decision["entropy_ranked_candidate_numbers"] == [1, 2, 3, 4]
    assert decision["note_count_ranked_candidate_numbers"] == [1, 2, 3, 4]
    assert decision["final_combined_ranked_candidate_numbers"] == [1, 2, 3]
    assert decision["selected_candidate_numbers"] == [1, 2, 3]


def test_if_else_uses_most_in_key_top5_when_strict_pool_is_empty():
    melody = [8.0, 0, 0, 0, 4.0, 0, 0, 4.0, 0, 0, 0, 0]
    counts = [
        [3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0],
        [2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 2, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0],
        [1, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
    candidates = [
        {
            "candidate_number": index,
            "generated_beats": 8,
            "required_beats": 8,
            "acc_note_count": 4,
            "mel_pitch_class_duration_evidence": melody,
            "acc_pitch_class_note_counts": distribution,
            "acc_pitch_class_note_entropy": float(index),
            "acc_pitch_change_score": index,
        }
        for index, distribution in enumerate(counts, start=1)
    ]

    decision = select_rule_s_if_else_candidates(candidates)

    assert decision["stage_2_strict_in_key_candidate_numbers"] == []
    assert decision["stage_2_tonal_fallback_used"] is True
    assert decision["stage_2_tonal_fallback_candidate_numbers"] == [1, 2, 3, 4, 5]
    assert decision["stage_2_candidate_numbers"] == [1, 2, 3, 4, 5]
    assert decision["pitch_change_rank_1_to_4_candidate_numbers"] == [5, 4, 3, 2]
    assert decision["final_combined_ranked_candidate_numbers"] == [5, 4, 3]


def test_if_else_never_reintroduces_candidates_above_note_cap():
    melody = [8.0, 0, 0, 0, 4.0, 0, 0, 4.0, 0, 0, 0, 0]
    candidates = [
        {
            "candidate_number": number,
            "generated_beats": 8,
            "required_beats": 8,
            "acc_note_count": note_count,
            "mel_pitch_class_duration_evidence": melody,
            "acc_pitch_class_note_counts": [note_count] + [0] * 11,
            "acc_pitch_class_note_entropy": 0.0,
            "acc_pitch_change_score": number,
        }
        for number, note_count in enumerate([4, 8, 12, 20, 26], start=1)
    ]

    decision = select_rule_s_if_else_candidates(candidates, max_notes=25)

    assert decision["stage_2_note_count_cap_candidate_numbers"] == [1, 2, 3, 4]
    assert 5 not in decision["stage_2_candidate_numbers"]
    assert 5 not in decision["selected_candidate_numbers"]


def test_prompt_ppl_scores_only_accompaniment_content_tokens():
    vocab = SimpleNamespace(
        beat_marker=1,
        bar_token_id=2,
        eos_token_id=3,
        bos_token_id=4,
        track_marker_acc=5,
        track_marker_mel=6,
    )
    tokenizer = SimpleNamespace(vocab=vocab)
    generated = torch.tensor([[4, 9, 5, 7, 5, 8, 3]], dtype=torch.long)

    class FakeModel:
        pad_token_id = 0

        def __call__(self, input_ids, attention_mask):
            logits = torch.zeros((1, input_ids.shape[1], 10), dtype=torch.float32)
            logits[0, 2, 7] = 8.0
            logits[0, 4, 8] = 8.0
            return SimpleNamespace(logits=logits)

    scores = score_prompt_batch_ppl(
        FakeModel(),
        generated,
        prompt_token_count=2,
        device="cpu",
        tokenizer=tokenizer,
        max_acc_beats=2,
    )

    assert scores[0]["available"] is True
    assert scores[0]["scored_token_count"] == 2
    assert scores[0]["ppl"] < 1.01
