from types import SimpleNamespace

import numpy as np
import torch

from streammuse.infrastructure.inference.lekai_prompt_continuation.prompt_batch_selector import (
    accompaniment_features_from_pianoroll,
    median_note_duration_from_pianoroll,
    score_prompt_batch_ppl,
    select_rule_s_candidate,
    select_rule_s_v2_candidate,
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
):
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
