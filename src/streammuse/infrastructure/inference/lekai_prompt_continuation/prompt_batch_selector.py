"""Frozen Rule-S selector for batched Lekai Prompt Model candidates."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch


RULE_S_ID = "empty_filter_discovery_spearman_weighted_rank_v1"
RULE_S_V2_ID = "rule_s_v1_plus_median_duration_mismatch_v2"
RULE_S_V3_ID = "rule_s_prompt_musical_compatibility_v3"
RULE_S_WEIGHTS = {
    "prompt_ppl": -0.45,
    "acc_pitch_range": 0.47699162299076214,
    "acc_pitch_class_entropy": 0.4666666666666666,
    "acc_average_voice_number": 0.4958158260214505,
}
RULE_S_V2_DURATION_WEIGHT = 0.49
RULE_S_V2_EXPECTED_LOG_DURATION_RATIO = 0.0
RULE_S_V3_WEIGHTS = {
    "prompt_ppl": -0.45,
    "acc_pitch_range": 0.47699162299076214,
    "acc_pitch_class_entropy": 0.4666666666666666,
    "duration_match": 0.3,
    "tonal_fit": 0.5,
    "low_register_penalty": -0.5,
}
RULE_S_V3_DURATION_EPSILON = 1e-6
RULE_S_V3_DURATION_TAU = math.log(2.0)
RULE_S_V3_LOW_REGISTER_MIDI = 36
RULE_S_V3_LOW_REGISTER_SPAN = 15.0

# Soft major/minor pitch-class templates. Values are compatibility strengths,
# not probabilities; rotations provide all 24 keys.
_MAJOR_TONAL_TEMPLATE = np.asarray(
    [1.0, 0.1, 0.65, 0.1, 0.8, 0.7, 0.1, 0.9, 0.1, 0.65, 0.1, 0.6],
    dtype=np.float64,
)
_MINOR_TONAL_TEMPLATE = np.asarray(
    [1.0, 0.1, 0.6, 0.8, 0.1, 0.7, 0.1, 0.9, 0.65, 0.1, 0.6, 0.1],
    dtype=np.float64,
)


def duration_weighted_pitch_class_evidence(
    pianoroll: np.ndarray,
    *,
    length_ticks: int | None = None,
) -> np.ndarray:
    """Return unnormalized note-duration evidence for the 12 pitch classes."""

    if pianoroll.ndim != 3 or pianoroll.shape[0] < 2:
        raise ValueError("expected pianoroll with shape (2, pitch, time)")
    length_ticks = pianoroll.shape[2] if length_ticks is None else int(length_ticks)
    if length_ticks <= 0:
        raise ValueError("length_ticks must be positive")

    sustain = np.asarray(pianoroll[0, :, :length_ticks] > 0, dtype=np.bool_)
    onset = np.asarray(pianoroll[1, :, :length_ticks] > 0, dtype=np.bool_)
    evidence = np.zeros(12, dtype=np.float64)
    for pitch_index in np.flatnonzero(onset.any(axis=1)):
        for start in np.flatnonzero(onset[pitch_index]):
            end = int(start) + 1
            while end < sustain.shape[1] and sustain[pitch_index, end]:
                end += 1
            evidence[(int(pitch_index) + 21) % 12] += end - int(start)
    return evidence


def low_register_penalty_from_pianoroll(
    pianoroll: np.ndarray,
    *,
    length_ticks: int,
) -> float:
    """Duration-weighted squared hinge below MIDI 36, clamped to [0, 1]."""

    if pianoroll.ndim != 3 or pianoroll.shape[0] < 2:
        raise ValueError("expected pianoroll with shape (2, pitch, time)")
    length_ticks = int(length_ticks)
    if length_ticks <= 0:
        raise ValueError("length_ticks must be positive")

    sustain = np.asarray(pianoroll[0, :, :length_ticks] > 0, dtype=np.bool_)
    onset = np.asarray(pianoroll[1, :, :length_ticks] > 0, dtype=np.bool_)
    weighted_penalty = 0.0
    total_duration = 0.0
    for pitch_index in np.flatnonzero(onset.any(axis=1)):
        midi_pitch = int(pitch_index) + 21
        hinge = max(0.0, RULE_S_V3_LOW_REGISTER_MIDI - midi_pitch)
        for start in np.flatnonzero(onset[pitch_index]):
            end = int(start) + 1
            while end < sustain.shape[1] and sustain[pitch_index, end]:
                end += 1
            duration = float(end - int(start))
            weighted_penalty += duration * (hinge / RULE_S_V3_LOW_REGISTER_SPAN) ** 2
            total_duration += duration
    if total_duration <= 0:
        return 0.0
    return float(np.clip(weighted_penalty / total_duration, 0.0, 1.0))


def duration_match_score(
    median_acc_duration: float | None,
    median_mel_duration: float | None,
    *,
    epsilon: float = RULE_S_V3_DURATION_EPSILON,
    tau: float = RULE_S_V3_DURATION_TAU,
) -> float | None:
    """Return symmetric log-ratio duration compatibility, or None if unavailable."""

    try:
        acc = float(median_acc_duration)
        mel = float(median_mel_duration)
    except (TypeError, ValueError):
        return None
    if (
        acc <= 0
        or mel <= 0
        or not math.isfinite(acc)
        or not math.isfinite(mel)
        or epsilon <= 0
        or tau <= 0
        or not math.isfinite(epsilon)
        or not math.isfinite(tau)
    ):
        return None
    return math.exp(-abs(math.log((acc + epsilon) / (mel + epsilon))) / tau)


def tonal_fit_score(
    melody_evidence: list[float] | np.ndarray,
    accompaniment_evidence: list[float] | np.ndarray,
) -> tuple[float | None, float | None]:
    """Return soft key-template fit and melody key confidence in [0, 1]."""

    melody = np.asarray(melody_evidence, dtype=np.float64).copy()
    accompaniment = np.asarray(accompaniment_evidence, dtype=np.float64).copy()
    if melody.shape != (12,) or accompaniment.shape != (12,):
        return None, None
    if (
        not np.all(np.isfinite(melody))
        or not np.all(np.isfinite(accompaniment))
        or np.any(melody < 0)
        or np.any(accompaniment < 0)
    ):
        return None, None
    if melody.sum() <= 0 or accompaniment.sum() <= 0:
        return None, None
    melody /= melody.sum()
    accompaniment /= accompaniment.sum()

    templates = [
        np.roll(template, tonic)
        for template in (_MAJOR_TONAL_TEMPLATE, _MINOR_TONAL_TEMPLATE)
        for tonic in range(12)
    ]
    fits = np.asarray([float(np.dot(melody, template)) for template in templates])
    best_template = templates[int(np.argmax(fits))]
    uniform_fit = float(best_template.mean())
    confidence = float(
        np.clip((float(fits.max()) - uniform_fit) / (1.0 - uniform_fit), 0.0, 1.0)
    )
    compatibility = float(np.clip(np.dot(accompaniment, best_template), 0.0, 1.0))
    return confidence * compatibility, confidence


def median_note_duration_from_pianoroll(
    pianoroll: np.ndarray,
    *,
    length_ticks: int,
) -> float:
    """Return the median note duration on a sustain/onset pianoroll grid."""

    if pianoroll.ndim != 3 or pianoroll.shape[0] < 2:
        raise ValueError("expected pianoroll with shape (2, pitch, time)")
    length_ticks = int(length_ticks)
    if length_ticks <= 0:
        raise ValueError("length_ticks must be positive")

    sustain = np.asarray(pianoroll[0, :, :length_ticks] > 0, dtype=np.bool_)
    onset = np.asarray(pianoroll[1, :, :length_ticks] > 0, dtype=np.bool_)
    durations: list[float] = []
    for pitch_index in np.flatnonzero(onset.any(axis=1)):
        for start in np.flatnonzero(onset[pitch_index]):
            end = int(start) + 1
            while end < sustain.shape[1] and sustain[pitch_index, end]:
                end += 1
            durations.append(float(end - int(start)))
    return float(np.median(durations)) if durations else 0.0


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
            "acc_median_note_duration_ticks": 0.0,
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
        "acc_median_note_duration_ticks": median_note_duration_from_pianoroll(
            pianoroll,
            length_ticks=length_ticks,
        ),
    }


def melody_features_from_pianoroll(
    pianoroll: np.ndarray,
    *,
    length_ticks: int,
) -> dict[str, float | int | list[float]]:
    """Compute the prompt-melody evidence required by Rule-S v3."""

    evidence = duration_weighted_pitch_class_evidence(
        pianoroll,
        length_ticks=length_ticks,
    )
    return {
        "mel_note_count": int(np.asarray(pianoroll[1, :, :length_ticks] > 0).sum()),
        "mel_median_note_duration_ticks": median_note_duration_from_pianoroll(
            pianoroll,
            length_ticks=length_ticks,
        ),
        "mel_pitch_class_duration_evidence": evidence.tolist(),
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


def select_rule_s_v2_candidate(
    candidates: list[dict[str, Any]],
    *,
    duration_weight: float = RULE_S_V2_DURATION_WEIGHT,
    expected_log_duration_ratio: float = RULE_S_V2_EXPECTED_LOG_DURATION_RATIO,
    epsilon: float = 1e-6,
) -> dict[str, Any]:
    """Add a Melody-relative median-duration penalty to frozen Rule-S v1."""

    if not candidates:
        raise ValueError("at least one Prompt candidate is required")
    if duration_weight < 0 or not math.isfinite(duration_weight):
        raise ValueError("duration_weight must be finite and non-negative")
    if not math.isfinite(expected_log_duration_ratio):
        raise ValueError("expected_log_duration_ratio must be finite")
    if epsilon <= 0 or not math.isfinite(epsilon):
        raise ValueError("epsilon must be finite and positive")

    if all("rule_s_score" in candidate for candidate in candidates):
        scored = [dict(candidate) for candidate in candidates]
    else:
        scored = select_rule_s_candidate(candidates)["candidates"]

    eligible = []
    for candidate in scored:
        melody_duration = float(candidate.get("mel_median_note_duration_ticks", 0.0))
        accompaniment_duration = float(
            candidate.get("acc_median_note_duration_ticks", 0.0)
        )
        duration_available = (
            bool(candidate.get("eligible"))
            and melody_duration > 0
            and accompaniment_duration > 0
            and math.isfinite(melody_duration)
            and math.isfinite(accompaniment_duration)
        )
        candidate["duration_metric_available"] = duration_available
        candidate["duration_ratio"] = None
        candidate["duration_log_ratio"] = None
        candidate["duration_mismatch"] = None
        candidate["rule_s_v2_eligible"] = duration_available
        candidate["rule_s_v2_score"] = None
        if duration_available:
            ratio = (accompaniment_duration + epsilon) / (melody_duration + epsilon)
            log_ratio = math.log(ratio)
            candidate["duration_ratio"] = ratio
            candidate["duration_log_ratio"] = log_ratio
            candidate["duration_mismatch"] = abs(
                log_ratio - expected_log_duration_ratio
            )
            eligible.append(candidate)

    if not eligible:
        fallback = select_rule_s_candidate(scored)
        return {
            "rule_id": RULE_S_V2_ID,
            "selected_index": int(fallback["selected_index"]),
            "selected_candidate_number": int(fallback["selected_candidate_number"]),
            "eligible_count": 0,
            "fallback_reason": "duration_metric_unavailable_use_rule_s_v1",
            "duration_weight": duration_weight,
            "expected_log_duration_ratio": expected_log_duration_ratio,
            "candidates": scored,
        }

    for candidate in eligible:
        candidate["rule_s_v2_score"] = float(candidate["rule_s_score"]) - (
            duration_weight
            * normalized_average_rank(
                eligible,
                candidate,
                "duration_mismatch",
            )
        )

    selected = max(
        eligible,
        key=lambda candidate: (
            float(candidate["rule_s_v2_score"]),
            -float(candidate["prompt_ppl"]),
            -int(candidate["candidate_number"]),
        ),
    )
    return {
        "rule_id": RULE_S_V2_ID,
        "selected_index": int(selected["candidate_number"]) - 1,
        "selected_candidate_number": int(selected["candidate_number"]),
        "eligible_count": len(eligible),
        "fallback_reason": None,
        "duration_weight": duration_weight,
        "expected_log_duration_ratio": expected_log_duration_ratio,
        "candidates": scored,
    }


def select_rule_s_v3_candidate(
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    """Score Rule-S v3 without changing the frozen v1/v2 selectors."""

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

    ranked_metrics = (
        "prompt_ppl",
        "acc_pitch_range",
        "acc_pitch_class_entropy",
        "duration_match",
        "tonal_fit",
    )
    for candidate in scored:
        candidate["eligible"] = candidate in eligible
        candidate["rule_s_v3_score"] = None
        candidate["rule_s_v3_ranks"] = {metric: None for metric in ranked_metrics}
        candidate["rule_s_v3_contributions"] = {
            metric: 0.0 for metric in RULE_S_V3_WEIGHTS
        }
        candidate["duration_match"] = duration_match_score(
            candidate.get("acc_median_note_duration_ticks", 0.0),
            candidate.get("mel_median_note_duration_ticks", 0.0),
        )
        tonal_fit, key_confidence = tonal_fit_score(
            candidate.get("mel_pitch_class_duration_evidence", []),
            candidate.get("acc_pitch_class_duration_evidence", []),
        )
        candidate["tonal_fit"] = tonal_fit
        candidate["melody_key_confidence"] = key_confidence
        low_penalty = candidate.get("low_register_penalty")
        try:
            low_penalty_value = float(low_penalty)
        except (TypeError, ValueError):
            low_penalty_value = math.nan
        candidate["low_register_penalty"] = (
            float(np.clip(low_penalty_value, 0.0, 1.0))
            if math.isfinite(low_penalty_value)
            else None
        )
        candidate["duration_match_available"] = candidate["duration_match"] is not None
        candidate["tonal_fit_available"] = candidate["tonal_fit"] is not None
        candidate["low_register_penalty_available"] = (
            candidate["low_register_penalty"] is not None
        )

    if not eligible:
        return {
            "rule_id": RULE_S_V3_ID,
            "selected_index": 0,
            "selected_candidate_number": int(scored[0]["candidate_number"]),
            "eligible_count": 0,
            "fallback_reason": "no_eligible_candidate",
            "candidates": scored,
        }

    for metric in ranked_metrics:
        available = [
            candidate
            for candidate in eligible
            if candidate.get(metric) is not None
            and math.isfinite(float(candidate[metric]))
        ]
        for candidate in available:
            rank = normalized_average_rank(available, candidate, metric)
            contribution = RULE_S_V3_WEIGHTS[metric] * rank
            candidate["rule_s_v3_ranks"][metric] = rank
            candidate["rule_s_v3_contributions"][metric] = contribution

    for candidate in eligible:
        if candidate["low_register_penalty"] is not None:
            candidate["rule_s_v3_contributions"]["low_register_penalty"] = (
                RULE_S_V3_WEIGHTS["low_register_penalty"]
                * float(candidate["low_register_penalty"])
            )
        candidate["rule_s_v3_score"] = sum(
            candidate["rule_s_v3_contributions"].values()
        )

    selected = max(
        eligible,
        key=lambda candidate: (
            float(candidate["rule_s_v3_score"]),
            -float(candidate["prompt_ppl"]),
            -int(candidate["candidate_number"]),
        ),
    )
    return {
        "rule_id": RULE_S_V3_ID,
        "selected_index": int(selected["candidate_number"]) - 1,
        "selected_candidate_number": int(selected["candidate_number"]),
        "eligible_count": len(eligible),
        "fallback_reason": None,
        "duration_tau": RULE_S_V3_DURATION_TAU,
        "duration_epsilon": RULE_S_V3_DURATION_EPSILON,
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
