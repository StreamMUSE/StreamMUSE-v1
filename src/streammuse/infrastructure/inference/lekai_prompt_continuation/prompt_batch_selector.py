"""Frozen Rule-S selector for batched Lekai Prompt Model candidates."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch


RULE_S_ID = "empty_filter_discovery_spearman_weighted_rank_v1"
RULE_S_WEIGHTS = {
    "prompt_ppl": -0.45,
    "acc_pitch_range": 0.47699162299076214,
    "acc_pitch_class_entropy": 0.4666666666666666,
    "acc_average_voice_number": 0.4958158260214505,
}


def accompaniment_features_from_pianoroll(
    pianoroll: np.ndarray,
    *,
    length_ticks: int,
) -> dict[str, float | int]:
    """Compute the four frozen Prompt metrics on the accompaniment grid."""

    if pianoroll.ndim != 3 or pianoroll.shape[0] < 2:
        raise ValueError("expected accompaniment pianoroll with shape (2, pitch, time)")
    length_ticks = int(length_ticks)
    if length_ticks <= 0:
        raise ValueError("length_ticks must be positive")

    sustain = np.asarray(pianoroll[0, :, :length_ticks] > 0, dtype=np.float64)
    onset = np.asarray(pianoroll[1, :, :length_ticks] > 0, dtype=np.float64)
    note_count = int(onset.sum())
    active_pitch_indices = np.flatnonzero(onset.any(axis=1))
    if note_count == 0 or active_pitch_indices.size == 0:
        return {
            "acc_note_count": 0,
            "acc_pitch_range": 0.0,
            "acc_pitch_class_entropy": 0.0,
            "acc_average_voice_number": 0.0,
        }

    pitch_weights = np.zeros(onset.shape[0], dtype=np.float64)
    total_note_duration = 0.0
    for pitch_index in active_pitch_indices:
        for start in np.flatnonzero(onset[pitch_index]):
            end = int(start) + 1
            while end < sustain.shape[1] and sustain[pitch_index, end] > 0:
                end += 1
            duration = float(end - int(start))
            pitch_weights[pitch_index] += duration
            total_note_duration += duration

    chroma = np.zeros(12, dtype=np.float64)
    for pitch_index, weight in enumerate(pitch_weights):
        if weight > 0:
            chroma[(pitch_index + 21) % 12] += float(weight)
    if chroma.sum() > 0:
        chroma /= chroma.sum()
    nonzero = chroma[chroma > 0]

    return {
        "acc_note_count": note_count,
        "acc_pitch_range": float(
            active_pitch_indices.max() - active_pitch_indices.min()
        ),
        "acc_pitch_class_entropy": float(-np.sum(nonzero * np.log2(nonzero))),
        "acc_average_voice_number": float(total_note_duration / length_ticks),
    }


def normalized_average_rank(
    candidates: list[dict[str, Any]],
    candidate: dict[str, Any],
    key: str,
) -> float:
    value = float(candidate[key])
    lower_count = sum(float(other[key]) < value for other in candidates)
    equal_count = sum(float(other[key]) == value for other in candidates)
    average_rank = 1.0 + lower_count + (equal_count - 1) / 2.0
    return average_rank / len(candidates)


def select_rule_s_candidate(
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    """Score eligible candidates and return a zero-based selection decision."""

    if not candidates:
        raise ValueError("at least one Prompt candidate is required")

    scored = [dict(candidate) for candidate in candidates]
    eligible = [
        candidate
        for candidate in scored
        if int(candidate["generated_beats"]) >= int(candidate["required_beats"])
        and int(candidate["acc_note_count"]) > 0
        and bool(candidate["prompt_ppl_available"])
        and math.isfinite(float(candidate["prompt_ppl"]))
    ]

    for candidate in scored:
        candidate["eligible"] = candidate in eligible
        candidate["rule_s_score"] = None

    if not eligible:
        return {
            "rule_id": RULE_S_ID,
            "selected_index": 0,
            "selected_candidate_number": int(scored[0]["candidate_number"]),
            "eligible_count": 0,
            "fallback_reason": "no_eligible_candidate",
            "candidates": scored,
        }

    for candidate in eligible:
        candidate["rule_s_score"] = sum(
            weight * normalized_average_rank(eligible, candidate, metric)
            for metric, weight in RULE_S_WEIGHTS.items()
        )

    selected = max(
        eligible,
        key=lambda candidate: (
            float(candidate["rule_s_score"]),
            -float(candidate["prompt_ppl"]),
            -int(candidate["candidate_number"]),
        ),
    )
    return {
        "rule_id": RULE_S_ID,
        "selected_index": int(selected["candidate_number"]) - 1,
        "selected_candidate_number": int(selected["candidate_number"]),
        "eligible_count": len(eligible),
        "fallback_reason": None,
        "candidates": scored,
    }


def trim_at_eos(sequence: torch.Tensor, eos_token_id: int) -> torch.Tensor:
    eos_positions = (sequence == int(eos_token_id)).nonzero(as_tuple=False)
    if len(eos_positions):
        return sequence[: int(eos_positions[0].item()) + 1]
    return sequence


def accompaniment_target_positions(
    tokens: list[int],
    *,
    prompt_token_count: int,
    tokenizer: Any,
    max_acc_beats: int,
) -> tuple[list[int], int]:
    vocab = tokenizer.vocab
    structural = {
        int(vocab.beat_marker),
        int(vocab.bar_token_id),
        int(vocab.eos_token_id),
        int(vocab.bos_token_id),
    }
    positions: list[int] = []
    acc_beat_count = 0
    in_scored_acc = False
    for token_index in range(int(prompt_token_count), len(tokens)):
        token = int(tokens[token_index])
        if token == int(vocab.track_marker_acc):
            acc_beat_count += 1
            in_scored_acc = acc_beat_count <= int(max_acc_beats)
            if not in_scored_acc:
                break
            continue
        if token == int(vocab.track_marker_mel) or token in structural:
            in_scored_acc = False
            continue
        if in_scored_acc:
            positions.append(token_index)
    return positions, min(acc_beat_count, int(max_acc_beats))


@torch.no_grad()
def score_prompt_batch_ppl(
    model: Any,
    generated: torch.Tensor,
    *,
    prompt_token_count: int,
    device: str,
    tokenizer: Any,
    max_acc_beats: int,
) -> list[dict[str, Any]]:
    """Compute raw-logit PPL for each candidate's accompaniment content."""

    sequences = generated.to(device)
    inputs = sequences[:, :-1]
    attention_mask = inputs.ne(int(model.pad_token_id)).long()
    logits = model(input_ids=inputs, attention_mask=attention_mask).logits.float()
    results: list[dict[str, Any]] = []

    for candidate_index, sequence in enumerate(sequences):
        tokens = trim_at_eos(
            sequence.detach().cpu(),
            int(tokenizer.vocab.eos_token_id),
        ).tolist()
        positions, scored_beats = accompaniment_target_positions(
            tokens,
            prompt_token_count=int(prompt_token_count),
            tokenizer=tokenizer,
            max_acc_beats=int(max_acc_beats),
        )
        if not positions:
            results.append(
                {
                    "available": False,
                    "reason": "no_scored_accompaniment_content_tokens",
                    "scored_acc_beats": scored_beats,
                    "scored_token_count": 0,
                    "ppl": None,
                }
            )
            continue

        logit_positions = torch.tensor(
            [position - 1 for position in positions],
            device=logits.device,
            dtype=torch.long,
        )
        targets = sequences[candidate_index, positions]
        selected_logits = logits[candidate_index].index_select(0, logit_positions)
        selected_logprobs = torch.log_softmax(selected_logits, dim=-1).gather(
            dim=-1,
            index=targets.unsqueeze(-1),
        ).squeeze(-1)
        mean_nll = -selected_logprobs.mean().item()
        results.append(
            {
                "available": True,
                "reason": None,
                "scored_acc_beats": scored_beats,
                "scored_token_count": int(selected_logprobs.numel()),
                "mean_nll": mean_nll,
                "ppl": math.exp(min(mean_nll, 80.0)),
            }
        )
    return results
