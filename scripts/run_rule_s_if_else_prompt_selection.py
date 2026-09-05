#!/usr/bin/env python
"""Run Rule-S if/else selection and export every Prompt candidate as MIDI."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from streammuse.infrastructure.inference.lekai_prompt_continuation.prompt_batch_selector import (
    accompaniment_features_from_pianoroll,
    melody_features_from_pianoroll,
    pitch_change_score_from_pianoroll,
    pitch_class_note_distribution_from_pianoroll,
    score_prompt_batch_ppl,
    select_rule_s_if_else_candidates,
    trim_at_eos,
)
from streammuse.infrastructure.inference.lekai_prompt_continuation.prompt_model.config import (
    ModelConfig,
)
from streammuse.infrastructure.inference.lekai_prompt_continuation.prompt_model.inference import (
    load_model,
)
from streammuse.infrastructure.inference.lekai_prompt_continuation.prompt_model.my_tokenizer import (
    PianoMusicTokenizer,
)
from streammuse.infrastructure.inference.lekai_prompt_continuation.prompt_model.Token2Midi import (
    MidiConverter,
)


TIMESTEPS_PER_BEAT = 4
DEFAULT_OUTPUT_ROOT = Path("prompt_selection_runs")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Prompt candidates for one prepared NPZ, select with the "
            "Rule-S if/else funnel, and export every candidate as MIDI."
        )
    )
    parser.add_argument("--npz-file", type=Path, required=True)
    parser.add_argument("--prompt-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Run output directory. Defaults to the gitignored "
            "prompt_selection_runs/<npz-stem>_<timestamp>."
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--candidate-count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--prompt-num-bars", type=int, default=2)
    parser.add_argument("--prompt-prefix-beats", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=1.1)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    return parser.parse_args(argv)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )


def load_npz(path: Path) -> tuple[list[np.ndarray], dict[str, Any]]:
    with np.load(path, allow_pickle=True) as payload:
        metadata = dict(payload["metadata"].item())
        measures = [
            np.asarray(payload[f"measure_{index}"])
            for index in range(int(metadata["num_measures"]))
        ]
    if not measures:
        raise ValueError(f"NPZ has no measures: {path}")
    if any(measure.ndim != 3 or measure.shape[0] != 4 for measure in measures):
        raise ValueError(
            "NPZ measures must have shape (4, 88, time): "
            "melody sustain/onset and accompaniment sustain/onset"
        )
    return measures, metadata


def build_prompt_tokens(
    tokenizer: PianoMusicTokenizer,
    measures: list[np.ndarray],
    metadata: dict[str, Any],
    *,
    num_bars: int,
) -> torch.Tensor:
    if num_bars <= 0:
        raise ValueError("--prompt-num-bars must be positive")
    if len(measures) < num_bars:
        raise ValueError(f"NPZ has {len(measures)} bars; prompt needs {num_bars}")

    _, _, _, measure_beats = tokenizer._encode_measures(
        measures[:num_bars], metadata
    )
    parts = [
        torch.tensor(
            [
                tokenizer.vocab.bos_token_id,
                tokenizer.encode_time_sig(int(metadata["time_signature_idx"])),
                tokenizer.encode_bpm(float(metadata.get("bpm") or 120)),
            ],
            dtype=torch.long,
        )
    ]
    for beats in measure_beats:
        parts.append(torch.tensor([tokenizer.vocab.bar_token_id], dtype=torch.long))
        for melody, _accompaniment in beats:
            parts.append(torch.tensor([tokenizer.vocab.beat_marker], dtype=torch.long))
            parts.append(melody)
    return torch.cat(parts)


def candidate_from_sequence(
    sequence: torch.Tensor,
    *,
    candidate_number: int,
    tokenizer: PianoMusicTokenizer,
    prefix_beats: int,
    ppl_score: dict[str, Any],
) -> tuple[dict[str, Any], list[Any]]:
    trimmed = trim_at_eos(sequence.detach().cpu(), tokenizer.vocab.eos_token_id)
    _melody_beats, all_acc_beats = tokenizer.parse_generated_sequence(trimmed)
    acc_beats = all_acc_beats[:prefix_beats]
    prefix_ticks = prefix_beats * TIMESTEPS_PER_BEAT
    if acc_beats:
        acc_roll = tokenizer.decode_beats_to_pianoroll(
            acc_beats,
            track_marker_id=tokenizer.vocab.track_marker_acc,
        )
    else:
        acc_roll = np.zeros((2, 88, prefix_ticks), dtype=np.uint8)

    features: dict[str, Any] = accompaniment_features_from_pianoroll(
        acc_roll,
        length_ticks=prefix_ticks,
    )
    note_counts, note_entropy = pitch_class_note_distribution_from_pianoroll(
        acc_roll,
        length_ticks=prefix_ticks,
    )
    features.update(
        {
            "acc_pitch_class_note_counts": note_counts,
            "acc_pitch_class_note_entropy": note_entropy,
            "acc_pitch_change_score": pitch_change_score_from_pianoroll(
                acc_roll,
                length_ticks=prefix_ticks,
            ),
        }
    )
    ppl_available = bool(ppl_score.get("available"))
    return (
        {
            "candidate_number": int(candidate_number),
            "generated_beats": len(all_acc_beats),
            "required_beats": int(prefix_beats),
            "prompt_ppl_available": ppl_available,
            "prompt_ppl": (
                float(ppl_score["ppl"]) if ppl_available else None
            ),
            "prompt_ppl_scored_token_count": int(
                ppl_score.get("scored_token_count", 0)
            ),
            "prompt_ppl_reason": ppl_score.get("reason"),
            "generated_token_count": int(trimmed.numel()),
            "prompt_token_hash": hashlib.sha256(
                trimmed.numpy().tobytes()
            ).hexdigest(),
            **features,
        },
        acc_beats,
    )


def ranking_rows(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for candidate in candidates:
        rows.append(
            {
                "rank": 0,
                "status": candidate["selection_status"],
                "candidate_number": int(candidate["candidate_number"]),
                "complete_prompt": candidate["complete_prompt"],
                "stage_1_note_count_pass": candidate["stage_1_note_count_pass"],
                "stage_1_pass": candidate["stage_1_pass"],
                "stage_2_note_count_cap_pass": candidate[
                    "stage_2_note_count_cap_pass"
                ],
                "stage_2_in_key_pass": candidate["stage_2_in_key_pass"],
                "stage_2_tonal_fallback_top5": candidate[
                    "stage_2_tonal_fallback_top5"
                ],
                "stage_2_tonal_top_up": candidate["stage_2_tonal_top_up"],
                "stage_2_pool_member": candidate["stage_2_pool_member"],
                "pitch_change_rank": candidate["pitch_change_rank"],
                "stage_3_pitch_change_rank_1_to_4": candidate[
                    "stage_3_pitch_change_rank_1_to_4"
                ],
                "entropy_rank": candidate["entropy_rank"],
                "entropy_top3": candidate["entropy_top3"],
                "entropy_rank_points": candidate["entropy_rank_points"],
                "note_count_rank": candidate["note_count_rank"],
                "note_count_top3": candidate["note_count_top3"],
                "note_count_rank_points": candidate["note_count_rank_points"],
                "combined_rank_score": candidate["combined_rank_score"],
                "final_rank": candidate["final_rank"],
                "prompt_ppl": candidate["prompt_ppl"],
                "note_count": candidate["acc_note_count"],
                "out_of_key_pitch_classes": " ".join(
                    str(value) for value in candidate["out_of_key_pitch_classes"]
                ),
                "out_of_key_note_count": candidate["out_of_key_note_count"],
                "in_key_note_ratio": candidate["in_key_note_ratio"],
                "pitch_class_note_entropy": candidate[
                    "acc_pitch_class_note_entropy"
                ],
                "pitch_change_score": candidate["acc_pitch_change_score"],
                "midi_file": candidate["midi_file"],
            }
        )
    rows.sort(
        key=lambda row: (
            row["status"] != "SELECTED",
            int(row["final_rank"] or 999),
            not bool(row["stage_2_pool_member"]),
            -float(row["pitch_class_note_entropy"]),
            int(row["candidate_number"]),
        )
    )
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def write_ranking_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def resolve_output_dir(npz_path: Path, requested: Path | None) -> Path:
    if requested is not None:
        return requested.expanduser().resolve()
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    return (DEFAULT_OUTPUT_ROOT / f"{npz_path.stem}_{timestamp}").resolve()


def validate_args(args: argparse.Namespace) -> None:
    if args.candidate_count < 2:
        raise ValueError("--candidate-count must be at least 2")
    if args.prompt_prefix_beats <= 0:
        raise ValueError("--prompt-prefix-beats must be positive")
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")


def main() -> None:
    args = parse_args()
    validate_args(args)
    npz_path = args.npz_file.expanduser().resolve()
    checkpoint_path = args.prompt_checkpoint.expanduser().resolve()
    if not npz_path.is_file():
        raise FileNotFoundError(f"NPZ not found: {npz_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Prompt checkpoint not found: {checkpoint_path}")

    output_dir = resolve_output_dir(npz_path, args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    selected_dir = output_dir / "selected"
    unselected_dir = output_dir / "unselected"
    selected_dir.mkdir(parents=True, exist_ok=True)
    unselected_dir.mkdir(parents=True, exist_ok=True)

    measures, metadata = load_npz(npz_path)
    config = ModelConfig()
    tokenizer = PianoMusicTokenizer(config=config)
    converter = MidiConverter(tokenizer)
    prompt_tokens = build_prompt_tokens(
        tokenizer,
        measures,
        metadata,
        num_bars=args.prompt_num_bars,
    )
    prefix_ticks = args.prompt_prefix_beats * TIMESTEPS_PER_BEAT
    melody_roll = np.concatenate(
        [measure[:2] for measure in measures[: args.prompt_num_bars]], axis=2
    )
    if melody_roll.shape[2] < prefix_ticks:
        raise ValueError(
            f"selected {args.prompt_num_bars} bars provide {melody_roll.shape[2]} ticks, "
            f"but --prompt-prefix-beats needs {prefix_ticks} ticks"
        )

    melody_beats, _unused_acc = tokenizer.parse_generated_sequence(prompt_tokens)
    melody_beats = melody_beats[: args.prompt_prefix_beats]
    melody_features = melody_features_from_pianoroll(
        melody_roll,
        length_ticks=prefix_ticks,
    )
    tempo = float(metadata.get("bpm") or 120)
    converter.beats_to_midi(
        melody_beats,
        [],
        tempo=tempo,
        save_path=str(output_dir / "00_prompt_melody.mid"),
    )

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    model = load_model(
        str(checkpoint_path),
        config,
        device=args.device,
        use_fp16=args.fp16,
    )

    started = time.perf_counter()
    generated = model.generate_music_batch(
        initial_tokens=prompt_tokens,
        batch_size=args.candidate_count,
        device=args.device,
        max_length=int(prompt_tokens.numel()) + args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    )
    generation_seconds = time.perf_counter() - started
    ppl_scores = score_prompt_batch_ppl(
        model,
        generated,
        prompt_token_count=int(prompt_tokens.numel()),
        device=args.device,
        tokenizer=tokenizer,
        max_acc_beats=args.prompt_prefix_beats,
    )

    candidates = []
    accompaniment_beats = []
    for index in range(args.candidate_count):
        candidate, beats = candidate_from_sequence(
            generated[index],
            candidate_number=index + 1,
            tokenizer=tokenizer,
            prefix_beats=args.prompt_prefix_beats,
            ppl_score=ppl_scores[index],
        )
        candidate.update(melody_features)
        candidates.append(candidate)
        accompaniment_beats.append(beats)

    decision = select_rule_s_if_else_candidates(candidates)
    selected_number_order = [
        int(value) for value in decision["selected_candidate_numbers"]
    ]
    selected_numbers = set(selected_number_order)
    scored_candidates = decision["candidates"]

    for candidate, beats in zip(scored_candidates, accompaniment_beats, strict=True):
        number = int(candidate["candidate_number"])
        is_selected = number in selected_numbers
        status = "SELECTED" if is_selected else "UNSELECTED"
        target_dir = selected_dir if is_selected else unselected_dir
        if is_selected:
            final_rank = int(candidate["final_rank"])
            midi_name = f"rank_{final_rank:02d}_candidate_{number:02d}_{status}.mid"
        else:
            midi_name = f"candidate_{number:02d}_{status}.mid"
        midi_path = target_dir / midi_name
        converter.beats_to_midi(
            melody_beats,
            beats,
            tempo=tempo,
            save_path=str(midi_path),
        )
        candidate["selection_status"] = status
        candidate["midi_file"] = midi_path.relative_to(output_dir).as_posix()

    rows = ranking_rows(scored_candidates)
    write_ranking_csv(output_dir / "ranking.csv", rows)
    write_json(
        output_dir / "selection.json",
        {
            "input": {
                "npz_file": str(npz_path),
                "prompt_checkpoint": str(checkpoint_path),
                "prompt_midi": "00_prompt_melody.mid",
                "metadata": metadata,
            },
            "generation": {
                "candidate_count": args.candidate_count,
                "seed": args.seed,
                "prompt_num_bars": args.prompt_num_bars,
                "prompt_prefix_beats": args.prompt_prefix_beats,
                "temperature": args.temperature,
                "top_k": args.top_k,
                "top_p": args.top_p,
                "repetition_penalty": args.repetition_penalty,
                "generation_seconds": generation_seconds,
            },
            "selection": {
                key: value for key, value in decision.items() if key != "candidates"
            },
            "ranking": rows,
            "candidates": scored_candidates,
        },
    )

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "selected_candidate_numbers": selected_number_order,
                "selected_midis": [
                    row["midi_file"] for row in rows if row["status"] == "SELECTED"
                ],
                "ranking_csv": "ranking.csv",
                "selection_json": "selection.json",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
