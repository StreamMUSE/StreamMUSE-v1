#!/usr/bin/env python
"""Compare 8-beat and 9-beat Prompt prefixes with paired continuations."""

from __future__ import annotations

import argparse
import random
import shutil
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import torch

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


DEFAULT_SEVEN = (
    "5889045,5817083,332891,43038,5472153,3410421,2581346"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paired offline comparison of 8-beat and 9-beat Prompts."
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--prompt-checkpoint", default=DEFAULT_PROMPT_CHECKPOINT)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", default=str(SCRIPT_DIR / "outputs"))
    parser.add_argument("--run-prefix", default="prompt_8_vs_9")
    parser.add_argument("--npz-ids", default=DEFAULT_SEVEN)
    parser.add_argument("--prompt-num-bars", type=int, default=2)
    parser.add_argument("--prompt-max-new-tokens", type=int, default=1024)
    parser.add_argument("--prompt-seed-base", type=int, default=20260727)
    parser.add_argument("--max-prompt-attempts", type=int, default=20)
    parser.add_argument("--continuation-seeds", default="0,1,2")
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


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint = Path(args.checkpoint)
    prompt_checkpoint = Path(args.prompt_checkpoint)
    data_dir = Path(args.data_dir)
    requested_files = parse_npz_ids(args.npz_ids)
    continuation_seeds = parse_seeds(
        args.continuation_seeds,
        "continuation-seeds",
    )
    if not requested_files:
        raise ValueError("npz-ids must contain at least one item")
    if args.max_prompt_attempts < 1:
        raise ValueError("max-prompt-attempts must be positive")
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
    generator = torch.Generator(
        device=device if device.startswith("cuda") else "cpu"
    )
    attempt_rows: list[dict] = []
    prompt_rows: list[dict] = []
    output_rows: list[dict] = []

    for piece_index, npz_name in enumerate(requested_files, start=1):
        idx = piece_index - 1
        piece_id = Path(npz_name).stem
        base_prep = offline.prepare_generation(dataset, idx, gt_prefix_beats=0)
        if len(base_prep["mel_beats"]) < 9:
            raise RuntimeError(f"{piece_id}: fewer than 9 Melody beats")
        tempo = base_prep["metadata"]["bpm"] or 120
        prompt_prep = prompt.prepare_condition(
            prompt_dataset,
            idx,
            num_bars=args.prompt_num_bars,
        )
        if prompt_prep is None:
            raise RuntimeError(f"{piece_id}: not enough bars for Prompt Model")

        accepted_sequence = None
        accepted_acc_beats = None
        accepted_seed = None
        accepted_seconds = None
        for attempt_index in range(args.max_prompt_attempts):
            prompt_seed = (
                args.prompt_seed_base
                + (piece_index - 1) * args.max_prompt_attempts
                + attempt_index
            )
            set_seed(prompt_seed)
            prompt_max_length = (
                len(prompt_prep["prompt_tokens"])
                + args.prompt_max_new_tokens
            )
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            started = time.perf_counter()
            generated = prompt_model.generate_music(
                initial_tokens=prompt_prep["prompt_tokens"],
                device=device,
                max_length=prompt_max_length,
                temperature=args.prompt_temperature,
                top_k=args.prompt_top_k,
                top_p=args.prompt_top_p,
                repetition_penalty=args.prompt_repetition_penalty,
            )
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            generation_seconds = time.perf_counter() - started
            sequence = trim_at_eos(
                generated.squeeze(0),
                int(prompt_tokenizer.vocab.eos_token_id),
            )
            _, acc_beats = prompt_tokenizer.parse_generated_sequence(sequence)
            attempt_rows.append(
                {
                    "piece_id": piece_id,
                    "npz_file": npz_name,
                    "attempt_index": attempt_index + 1,
                    "prompt_seed": prompt_seed,
                    "generated_acc_beats": len(acc_beats),
                    "generation_seconds": generation_seconds,
                    "accepted": len(acc_beats) >= 9,
                }
            )
            print(
                f"{piece_id}: Prompt attempt {attempt_index + 1}, "
                f"seed={prompt_seed}, acc_beats={len(acc_beats)}",
                flush=True,
            )
            if len(acc_beats) >= 9:
                accepted_sequence = sequence
                accepted_acc_beats = acc_beats[:9]
                accepted_seed = prompt_seed
                accepted_seconds = generation_seconds
                break
        if accepted_sequence is None or accepted_acc_beats is None:
            raise RuntimeError(
                f"{piece_id}: no Prompt with at least 9 accompaniment beats "
                f"after {args.max_prompt_attempts} attempts"
            )

        piece_dir = run_dir / f"piece_{piece_index:02d}_{piece_id}"
        prompt_dir = piece_dir / "prompts"
        condition_root = piece_dir / "conditions"
        prompt_dir.mkdir(parents=True)
        condition_root.mkdir()
        converter.gt_to_midi(
            base_prep["gt_path"],
            str(piece_dir / "00_GT.mid"),
        )

        offline_prefixes: dict[int, list[torch.Tensor]] = {}
        for prefix_beats in (8, 9):
            offline_prefix = [
                convert_prompt_acc_to_offline(
                    prompt_tokenizer,
                    dataset.tokenizer,
                    beat,
                )
                for beat in accepted_acc_beats[:prefix_beats]
            ]
            offline_prefixes[prefix_beats] = offline_prefix
            prompt_midi = prompt_dir / f"prompt_{prefix_beats:02d}_beats.mid"
            converter.beats_to_midi(
                mel_beats=base_prep["mel_beats"][:prefix_beats],
                acc_beats=offline_prefix,
                tempo=tempo,
                save_path=str(prompt_midi),
            )
            prompt_rows.append(
                {
                    "piece_id": piece_id,
                    "npz_file": npz_name,
                    "prompt_seed": accepted_seed,
                    "prompt_generation_seconds": accepted_seconds,
                    "generated_acc_beats_before_truncation": len(
                        prompt_tokenizer.parse_generated_sequence(
                            accepted_sequence
                        )[1]
                    ),
                    "prefix_beats": prefix_beats,
                    "prompt_midi": str(prompt_midi.relative_to(run_dir)),
                }
            )

        for prefix_beats in (8, 9):
            condition_name = f"prompt_{prefix_beats:02d}_beats"
            condition_dir = condition_root / condition_name
            condition_dir.mkdir()
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
                    offline_prefixes[prefix_beats],
                )
                if injected != prefix_beats:
                    raise RuntimeError(
                        f"{piece_id}: injected {injected}/{prefix_beats} beats"
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
                        "prefix_beats": prefix_beats,
                        "prompt_seed": accepted_seed,
                        "continuation_index": continuation_index,
                        "continuation_seed": continuation_seed,
                        "output_midi": str(output_midi.relative_to(run_dir)),
                        "continuation_seconds": continuation_seconds,
                    }
                )
                print(
                    f"{piece_id}: prefix={prefix_beats}, "
                    f"continuation_seed={continuation_seed}",
                    flush=True,
                )

    config_payload = {
        "created_at": timestamp,
        "experiment": "paired_prompt_8_beats_vs_9_beats",
        "npz_ids": requested_files,
        "prompt_generation_retries_only_require_at_least_9_acc_beats": True,
        "prompt_seed_base": args.prompt_seed_base,
        "max_prompt_attempts": args.max_prompt_attempts,
        "prompt_sampling": {
            "temperature": args.prompt_temperature,
            "top_k": args.prompt_top_k,
            "top_p": args.prompt_top_p,
            "repetition_penalty": args.prompt_repetition_penalty,
        },
        "conditions": [
            "first 8 Prompt Model accompaniment beats",
            "first 9 Prompt Model accompaniment beats",
        ],
        "continuation_seeds": continuation_seeds,
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
    write_csv(
        run_dir / "prompt_attempts.csv",
        list(attempt_rows[0]),
        attempt_rows,
    )
    write_csv(
        run_dir / "prompt_manifest.csv",
        list(prompt_rows[0]),
        prompt_rows,
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
    rating_rows: list[dict] = []
    mapping_rows: list[dict] = []
    for blind_index, row in enumerate(shuffled, start=1):
        blind_id = f"sample_{blind_index:03d}"
        shutil.copy2(
            run_dir / row["output_midi"],
            blind_dir / f"{blind_id}.mid",
        )
        rating_rows.append(
            {
                "rater_id": "user_prompt_8_vs_9",
                "blind_id": blind_id,
                "overall_score_1_to_5": "",
                "beat9_transition_1_to_5": "",
                "melody_fit_1_to_5": "",
                "accompaniment_quality_1_to_5": "",
                "failure_reason": "",
                "comments": "",
            }
        )
        mapping_rows.append({"blind_id": blind_id, **row})
    write_csv(
        run_dir / "blind_ratings.csv",
        list(rating_rows[0]),
        rating_rows,
    )
    write_csv(
        run_dir / "blind_mapping_private.csv",
        list(mapping_rows[0]),
        mapping_rows,
    )
    print(f"8-vs-9 experiment complete: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
