#!/usr/bin/env python
"""Compare single-sample Prompt inference with batched Rule-S Best-of-N.

The Prompt Model receives one real batch containing N copies of the same
Melody condition. Candidate 1 represents ordinary single-sample inference.
Rule-S selects one candidate from the full batch, and only those two
conditions are passed to the Continuation Model.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import random
import shutil
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import torch

from apply_frozen_selector import (
    DISCOVERY_PIECE_CENTERED_SPEARMAN,
    accompaniment_features,
    normalized_average_rank,
)
from run_q1_pilot import (
    DEFAULT_CHECKPOINT,
    DEFAULT_DATA_DIR,
    DEFAULT_PROMPT_CHECKPOINT,
    SCRIPT_DIR,
    convert_prompt_acc_to_offline,
    inject_prompt_prefix,
    load_offline_components,
    load_prompt_components,
    parse_npz_ids,
    parse_seeds,
    resolve_device,
    set_seed,
    write_csv,
    write_json,
)


DEFAULT_OLD3 = "5889045,5472153,3410421"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Single-sample versus batched Rule-S Best-of-N inference."
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--prompt-checkpoint", default=DEFAULT_PROMPT_CHECKPOINT)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", default=str(SCRIPT_DIR / "outputs"))
    parser.add_argument("--run-prefix", default="rule_s_best_of_n")
    parser.add_argument("--npz-ids", default=DEFAULT_OLD3)
    parser.add_argument("--candidate-count", type=int, default=5)
    parser.add_argument("--prompt-batch-seed", type=int, default=20260727)
    parser.add_argument("--continuation-seeds", default="0,1")
    parser.add_argument("--prompt-prefix-beats", type=int, default=8)
    parser.add_argument("--prompt-num-bars", type=int, default=2)
    parser.add_argument("--prompt-max-new-tokens", type=int, default=1024)
    parser.add_argument("--prompt-temperature", type=float, default=0.8)
    parser.add_argument("--prompt-top-k", type=int, default=50)
    parser.add_argument("--prompt-top-p", type=float, default=0.95)
    parser.add_argument("--prompt-repetition-penalty", type=float, default=1.0)
    parser.add_argument("--continuation-temperature", type=float, default=1.1)
    parser.add_argument("--continuation-top-k", type=int, default=0)
    parser.add_argument("--continuation-top-p", type=float, default=0.95)
    parser.add_argument("--continuation-repetition-penalty", type=float, default=1.0)
    parser.add_argument("--blind-seed", type=int, default=20260727)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--mode", choices=("train", "test"), default="test")
    parser.add_argument("--max-schedule-steps", type=int, default=0)
    return parser.parse_args()


def trim_at_eos(sequence: torch.Tensor, eos_token_id: int) -> torch.Tensor:
    eos_positions = (sequence == eos_token_id).nonzero(as_tuple=False)
    if len(eos_positions):
        return sequence[: int(eos_positions[0].item()) + 1]
    return sequence


@torch.no_grad()
def generate_prompt_batch(
    model,
    initial_tokens: torch.Tensor,
    *,
    candidate_count: int,
    device: str,
    max_length: int,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
) -> tuple[torch.Tensor, float, float]:
    """Generate N candidates in one model.generate call."""
    input_ids = (
        initial_tokens.unsqueeze(0)
        .repeat(candidate_count, 1)
        .contiguous()
        .to(device)
    )
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    started = time.perf_counter()
    generated = model.model.generate(
        input_ids=input_ids,
        max_length=max_length,
        do_sample=True,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        eos_token_id=model.eos_token_id,
        pad_token_id=model.pad_token_id,
    )
    if device.startswith("cuda"):
        torch.cuda.synchronize()
        peak_memory_mib = torch.cuda.max_memory_allocated() / 1024**2
    else:
        peak_memory_mib = 0.0
    return generated.cpu(), time.perf_counter() - started, peak_memory_mib


def accompaniment_target_positions(
    tokens: list[int],
    *,
    prompt_token_count: int,
    tokenizer,
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
    for token_index in range(prompt_token_count, len(tokens)):
        token = int(tokens[token_index])
        if token == int(vocab.track_marker_acc):
            acc_beat_count += 1
            in_scored_acc = acc_beat_count <= max_acc_beats
            if not in_scored_acc:
                break
            continue
        if token == int(vocab.track_marker_mel) or token in structural:
            in_scored_acc = False
            continue
        if in_scored_acc:
            positions.append(token_index)
    return positions, min(acc_beat_count, max_acc_beats)


@torch.no_grad()
def score_prompt_batch(
    model,
    generated: torch.Tensor,
    *,
    prompt_token_count: int,
    device: str,
    tokenizer,
    max_acc_beats: int,
) -> list[dict]:
    """Compute per-candidate PPL with one raw-logit batch forward."""
    sequences = generated.to(device)
    inputs = sequences[:, :-1]
    attention_mask = inputs.ne(int(model.pad_token_id)).long()
    logits = model(input_ids=inputs, attention_mask=attention_mask).logits.float()
    results: list[dict] = []
    for candidate_index, sequence in enumerate(sequences):
        tokens = trim_at_eos(
            sequence.detach().cpu(),
            int(tokenizer.vocab.eos_token_id),
        ).tolist()
        positions, scored_beats = accompaniment_target_positions(
            tokens,
            prompt_token_count=prompt_token_count,
            tokenizer=tokenizer,
            max_acc_beats=max_acc_beats,
        )
        if not positions:
            results.append(
                {
                    "available": False,
                    "reason": "no_scored_accompaniment_content_tokens",
                    "all_generated_token_count": max(
                        0, len(tokens) - prompt_token_count
                    ),
                    "scored_acc_beats": scored_beats,
                    "scored_token_count": 0,
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
                "definition": (
                    "exp(mean NLL of accompaniment content tokens in the first "
                    f"{max_acc_beats} generated beats under raw Prompt Model logits)"
                ),
                "all_generated_token_count": max(
                    0, len(tokens) - prompt_token_count
                ),
                "scored_acc_beats": scored_beats,
                "scored_token_count": int(selected_logprobs.numel()),
                "mean_nll": mean_nll,
                "ppl": math.exp(min(mean_nll, 80.0)),
                "scored_token_ids": [
                    int(value) for value in targets.detach().cpu().tolist()
                ],
                "scored_token_logprobs": [
                    float(value)
                    for value in selected_logprobs.detach().cpu().tolist()
                ],
            }
        )
    return results


def apply_rule_s(candidates: list[dict]) -> dict:
    eligible = [
        row
        for row in candidates
        if row["prompt_generated_beats"] == row["required_prompt_beats"]
        and not row["is_empty_prompt"]
        and row["prompt_ppl_available"]
    ]
    if not eligible:
        raise RuntimeError("no eligible Prompt candidate for Rule-S")
    for row in eligible:
        row["rule_s_score"] = sum(
            correlation * normalized_average_rank(eligible, row, metric)
            for metric, correlation in DISCOVERY_PIECE_CENTERED_SPEARMAN.items()
        )
    selected = max(
        eligible,
        key=lambda row: (
            row["rule_s_score"],
            -row["prompt_ppl_numeric"],
        ),
    )
    for row in candidates:
        row.setdefault("rule_s_score", math.nan)
        row["selected_by_rule_s"] = row is selected
    return selected


def sequence_hash(sequence: torch.Tensor, eos_token_id: int) -> str:
    trimmed = trim_at_eos(sequence, eos_token_id)
    payload = ",".join(str(int(token)) for token in trimmed.tolist())
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def main() -> None:
    args = parse_args()
    if args.candidate_count < 2:
        raise ValueError("candidate-count must be at least 2")
    device = resolve_device(args.device)
    checkpoint = Path(args.checkpoint)
    prompt_checkpoint = Path(args.prompt_checkpoint)
    data_dir = Path(args.data_dir)
    requested_files = parse_npz_ids(args.npz_ids)
    continuation_seeds = parse_seeds(
        args.continuation_seeds, "continuation-seeds"
    )
    if not requested_files:
        raise ValueError("npz-ids must contain at least one item")
    for path, label in (
        (checkpoint, "continuation checkpoint"),
        (prompt_checkpoint, "prompt checkpoint"),
        (data_dir, "data directory"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")
    missing = [name for name in requested_files if not (data_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"missing npz files: {', '.join(missing)}")

    offline = load_offline_components()
    offline_config = offline.ModelConfig()
    dataset = offline.PianoDataset(
        str(data_dir),
        config=offline_config,
        cache_lengths=False,
        mode=args.mode,
    )
    dataset.data_files = requested_files
    converter = offline.MidiConverter(dataset.tokenizer)

    prompt = load_prompt_components()
    prompt_config = prompt.ModelConfig()
    prompt_tokenizer = prompt.PianoMusicTokenizer(config=prompt_config)
    prompt_dataset = SimpleNamespace(
        root_dir=dataset.root_dir,
        data_files=dataset.data_files,
        tokenizer=prompt_tokenizer,
    )

    print(f"Using device: {device}", flush=True)
    print(f"Continuation checkpoint: {checkpoint}", flush=True)
    continuation_model = offline.load_model(
        model_path=str(checkpoint),
        model_config=offline_config,
        device=device,
        use_fp16=args.fp16,
    )
    print(f"Prompt checkpoint: {prompt_checkpoint}", flush=True)
    prompt_model = prompt.load_model(
        model_path=str(prompt_checkpoint),
        model_config=prompt_config,
        device=device,
        use_fp16=args.fp16,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / f"{args.run_prefix}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    config_payload = {
        "created_at": timestamp,
        "experiment": "single_sample_vs_actual_batch_rule_s_best_of_n",
        "npz_ids": requested_files,
        "candidate_count": args.candidate_count,
        "baseline_definition": "candidate_01 from the same Prompt batch",
        "rule_s_definition": {
            "empty_prompt_filter": True,
            "incomplete_prompt_filter": True,
            "weights": DISCOVERY_PIECE_CENTERED_SPEARMAN,
            "selection": "maximum within-batch weighted normalized rank score",
        },
        "prompt_generation": {
            "actual_batch_generate_call": True,
            "batch_seed": args.prompt_batch_seed,
            "temperature": args.prompt_temperature,
            "top_k": args.prompt_top_k,
            "top_p": args.prompt_top_p,
            "repetition_penalty": args.prompt_repetition_penalty,
        },
        "continuation_seeds": continuation_seeds,
        "prompt_prefix_beats": args.prompt_prefix_beats,
        "prompt_num_bars": args.prompt_num_bars,
        "prompt_checkpoint": str(prompt_checkpoint),
        "continuation_checkpoint": str(checkpoint),
        "continuation_sampling": {
            "temperature": args.continuation_temperature,
            "top_k": args.continuation_top_k,
            "top_p": args.continuation_top_p,
            "repetition_penalty": args.continuation_repetition_penalty,
        },
    }
    write_json(run_dir / "run_config.json", config_payload)

    generator = torch.Generator(
        device=device if device.startswith("cuda") else "cpu"
    )
    prompt_rows: list[dict] = []
    selection_rows: list[dict] = []
    output_rows: list[dict] = []

    for piece_index, npz_name in enumerate(requested_files, start=1):
        idx = piece_index - 1
        piece_id = Path(npz_name).stem
        piece_dir = run_dir / f"piece_{piece_index:02d}_{piece_id}"
        prompt_dir = piece_dir / "prompts"
        output_dir = piece_dir / "conditions"
        prompt_dir.mkdir(parents=True)
        output_dir.mkdir()

        base_prep = offline.prepare_generation(dataset, idx, gt_prefix_beats=0)
        tempo = base_prep["metadata"]["bpm"] or 120
        converter.gt_to_midi(base_prep["gt_path"], str(piece_dir / "00_GT.mid"))
        prompt_prep = prompt.prepare_condition(
            prompt_dataset,
            idx,
            num_bars=args.prompt_num_bars,
        )
        if prompt_prep is None:
            raise RuntimeError(f"not enough bars for prompt stage: {npz_name}")

        batch_seed = args.prompt_batch_seed + piece_index - 1
        set_seed(batch_seed)
        prompt_max_length = (
            len(prompt_prep["prompt_tokens"]) + args.prompt_max_new_tokens
        )
        generated, batch_seconds, peak_memory_mib = generate_prompt_batch(
            prompt_model,
            prompt_prep["prompt_tokens"],
            candidate_count=args.candidate_count,
            device=device,
            max_length=prompt_max_length,
            temperature=args.prompt_temperature,
            top_k=args.prompt_top_k,
            top_p=args.prompt_top_p,
            repetition_penalty=args.prompt_repetition_penalty,
        )
        prompt_scores = score_prompt_batch(
            prompt_model,
            generated,
            prompt_token_count=len(prompt_prep["prompt_tokens"]),
            device=device,
            tokenizer=prompt_tokenizer,
            max_acc_beats=args.prompt_prefix_beats,
        )

        candidates: list[dict] = []
        offline_prefixes: dict[int, list[torch.Tensor]] = {}
        for candidate_index in range(args.candidate_count):
            sequence = trim_at_eos(
                generated[candidate_index],
                int(prompt_tokenizer.vocab.eos_token_id),
            )
            _, acc_beats = prompt_tokenizer.parse_generated_sequence(sequence)
            acc_beats = acc_beats[: args.prompt_prefix_beats]
            offline_prefix = [
                convert_prompt_acc_to_offline(
                    prompt_tokenizer,
                    dataset.tokenizer,
                    beat,
                )
                for beat in acc_beats
            ]
            offline_prefixes[candidate_index] = offline_prefix
            prompt_midi = (
                prompt_dir / f"candidate_{candidate_index + 1:02d}.mid"
            )
            converter.beats_to_midi(
                mel_beats=base_prep["mel_beats"][: len(offline_prefix)],
                acc_beats=offline_prefix,
                tempo=tempo,
                save_path=str(prompt_midi),
            )
            score_path = (
                prompt_dir / f"candidate_{candidate_index + 1:02d}_scores.json"
            )
            write_json(score_path, prompt_scores[candidate_index])
            features = accompaniment_features(
                prompt_midi,
                beats=args.prompt_prefix_beats,
            )
            score = prompt_scores[candidate_index]
            candidate = {
                "piece_id": piece_id,
                "npz_file": npz_name,
                "candidate_index": candidate_index + 1,
                "batch_seed": batch_seed,
                "prompt_midi": str(prompt_midi.relative_to(run_dir)),
                "prompt_token_scores": str(score_path.relative_to(run_dir)),
                "prompt_token_hash": sequence_hash(
                    sequence,
                    int(prompt_tokenizer.vocab.eos_token_id),
                ),
                "prompt_generated_beats": len(acc_beats),
                "required_prompt_beats": args.prompt_prefix_beats,
                "prompt_ppl_available": bool(score.get("available")),
                "prompt_ppl_numeric": float(
                    score.get("ppl", math.inf)
                ),
                "prompt_mean_nll": score.get("mean_nll", ""),
                "acc_note_count": features["acc_note_count"],
                "acc_pitch_range": features["acc_pitch_range"],
                "acc_pitch_class_entropy": features[
                    "acc_pitch_class_entropy"
                ],
                "acc_average_voice_number": features[
                    "acc_average_voice_number"
                ],
                "is_empty_prompt": features["acc_note_count"] == 0,
                "is_baseline_candidate": candidate_index == 0,
                "prompt_batch_seconds": batch_seconds,
                "prompt_batch_peak_memory_mib": peak_memory_mib,
            }
            candidates.append(candidate)

        selected = apply_rule_s(candidates)
        unique_prompt_count = len(
            {row["prompt_token_hash"] for row in candidates}
        )
        same_prompt_for_both_conditions = (
            selected["prompt_token_hash"]
            == candidates[0]["prompt_token_hash"]
        )
        selection_rows.append(
            {
                "piece_id": piece_id,
                "npz_file": npz_name,
                "candidate_count": args.candidate_count,
                "unique_prompt_count": unique_prompt_count,
                "unique_prompt_rate": unique_prompt_count
                / args.candidate_count,
                "baseline_candidate_index": 1,
                "rule_s_candidate_index": selected["candidate_index"],
                "same_prompt_for_both_conditions": (
                    same_prompt_for_both_conditions
                ),
                "rule_s_score": selected["rule_s_score"],
                "prompt_batch_seconds": batch_seconds,
                "prompt_batch_peak_memory_mib": peak_memory_mib,
            }
        )
        prompt_rows.extend(candidates)

        conditions = [
            ("single_sample", candidates[0]),
            (f"best_of_{args.candidate_count:02d}_rule_s", selected),
        ]
        for condition_name, candidate in conditions:
            candidate_zero_index = int(candidate["candidate_index"]) - 1
            offline_prefix = offline_prefixes[candidate_zero_index]
            if len(offline_prefix) != args.prompt_prefix_beats:
                raise RuntimeError(
                    f"{piece_id} candidate {candidate['candidate_index']} has "
                    f"{len(offline_prefix)} beats; expected "
                    f"{args.prompt_prefix_beats}"
                )
            for continuation_index, continuation_seed in enumerate(
                continuation_seeds,
                start=1,
            ):
                set_seed(continuation_seed)
                generator.manual_seed(continuation_seed)
                prep = offline.prepare_generation(
                    dataset,
                    idx,
                    gt_prefix_beats=0,
                )
                prep["schedule"], injected = inject_prompt_prefix(
                    prep["schedule"],
                    offline_prefix,
                )
                if injected != args.prompt_prefix_beats:
                    raise RuntimeError(
                        f"{piece_id}: injected {injected} Prompt beats"
                    )
                if args.max_schedule_steps > 0:
                    prep["schedule"] = prep["schedule"][
                        : args.max_schedule_steps
                    ]
                started = time.perf_counter()
                acc_beats, _ = continuation_model.generate_accompaniment(
                    initial_tokens=prep["initial_tokens"],
                    schedule=prep["schedule"],
                    vocab=prep["vocab"],
                    device=device,
                    temperature=args.continuation_temperature,
                    top_k=args.continuation_top_k,
                    top_p=args.continuation_top_p,
                    repetition_penalty=args.continuation_repetition_penalty,
                    generator=generator,
                )
                continuation_seconds = time.perf_counter() - started
                condition_dir = output_dir / condition_name
                condition_dir.mkdir(exist_ok=True)
                output_midi = condition_dir / (
                    f"continuation_seed_{continuation_seed}.mid"
                )
                converter.beats_to_midi(
                    mel_beats=prep["mel_beats"],
                    acc_beats=acc_beats,
                    tempo=tempo,
                    save_path=str(output_midi),
                )
                output_rows.append(
                    {
                        "piece_id": piece_id,
                        "npz_file": npz_name,
                        "condition": condition_name,
                        "candidate_index": candidate["candidate_index"],
                        "continuation_index": continuation_index,
                        "continuation_seed": continuation_seed,
                        "prompt_midi": candidate["prompt_midi"],
                        "output_midi": str(output_midi.relative_to(run_dir)),
                        "same_prompt_for_both_conditions": (
                            same_prompt_for_both_conditions
                        ),
                        "continuation_seconds": continuation_seconds,
                    }
                )
                print(
                    f"{piece_id}: {condition_name}, candidate="
                    f"{candidate['candidate_index']}, continuation_seed="
                    f"{continuation_seed}",
                    flush=True,
                )

    write_csv(
        run_dir / "prompt_candidates.csv",
        list(prompt_rows[0]),
        prompt_rows,
    )
    write_csv(
        run_dir / "selection_summary.csv",
        list(selection_rows[0]),
        selection_rows,
    )
    write_csv(
        run_dir / "condition_outputs.csv",
        list(output_rows[0]),
        output_rows,
    )

    blind_dir = run_dir / "blind_listening"
    blind_dir.mkdir()
    shuffled = output_rows.copy()
    random.Random(args.blind_seed).shuffle(shuffled)
    blind_rows: list[dict] = []
    mapping_rows: list[dict] = []
    for blind_index, row in enumerate(shuffled, start=1):
        blind_id = f"sample_{blind_index:03d}"
        shutil.copy2(
            run_dir / row["output_midi"],
            blind_dir / f"{blind_id}.mid",
        )
        blind_rows.append(
            {
                "blind_id": blind_id,
                "overall_score_1_to_5": "",
                "melody_fit_1_to_5": "",
                "accompaniment_quality_1_to_5": "",
                "preferred_in_piece_seed_pair": "",
                "comments": "",
            }
        )
        mapping_rows.append({"blind_id": blind_id, **row})
    write_csv(
        run_dir / "blind_ratings.csv",
        list(blind_rows[0]),
        blind_rows,
    )
    write_csv(
        run_dir / "blind_mapping_private.csv",
        list(mapping_rows[0]),
        mapping_rows,
    )
    print(f"Best-of-N experiment complete: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
