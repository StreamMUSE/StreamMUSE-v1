#!/usr/bin/env python
"""Rescore an existing Q1 run using only the injected accompaniment beats."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from run_q1_pilot import (
    DEFAULT_DATA_DIR,
    LEKAI_ROOT,
    load_offline_components,
    load_prompt_components,
    score_generated_prompt,
    write_csv,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fp16", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    prompt_prefix_beats = int(config["prompt_prefix_beats"])

    offline = load_offline_components()
    offline_config = offline.ModelConfig()
    dataset = offline.PianoDataset(
        args.data_dir,
        config=offline_config,
        cache_lengths=False,
        mode="test",
    )
    dataset.data_files = list(config["npz_ids"])

    prompt = load_prompt_components()
    prompt_config = prompt.ModelConfig()
    prompt_tokenizer = prompt.PianoMusicTokenizer(config=prompt_config)
    prompt_dataset = SimpleNamespace(
        root_dir=dataset.root_dir,
        data_files=dataset.data_files,
        tokenizer=prompt_tokenizer,
    )
    prompt_model = prompt.load_model(
        model_path=config["prompt_checkpoint"],
        model_config=prompt_config,
        device=args.device,
        use_fp16=args.fp16,
    )

    prompt_manifest_path = run_dir / "prompt_manifest.csv"
    with prompt_manifest_path.open(encoding="utf-8", newline="") as handle:
        prompt_rows = list(csv.DictReader(handle))

    ppl_by_prompt: dict[tuple[str, str], float] = {}
    for row in prompt_rows:
        piece_index = config["npz_ids"].index(row["npz_file"])
        prompt_prep = prompt.prepare_condition(
            prompt_dataset,
            piece_index,
            num_bars=int(config["prompt_num_bars"]),
        )
        if prompt_prep is None:
            raise RuntimeError(f"not enough bars: {row['npz_file']}")

        score_path = run_dir / row["prompt_token_scores"]
        old_scores = json.loads(score_path.read_text(encoding="utf-8"))
        generated_ids = old_scores.get("all_generated_token_ids")
        if generated_ids is None:
            generated_ids = old_scores.get("selected_token_ids")
        if generated_ids is None:
            raise RuntimeError(f"generated token ids missing: {score_path}")
        generated = torch.cat(
            [
                prompt_prep["prompt_tokens"].long(),
                torch.tensor(generated_ids, dtype=torch.long),
            ]
        ).unsqueeze(0)
        scores = score_generated_prompt(
            prompt_model,
            generated,
            len(prompt_prep["prompt_tokens"]),
            args.device,
            prompt_tokenizer,
            prompt_prefix_beats,
        )
        write_json(score_path, scores)
        row["prompt_generated_tokens"] = scores.get("all_generated_token_count", "")
        row["prompt_scored_tokens"] = scores.get("scored_token_count", "")
        row["prompt_mean_nll"] = scores.get("mean_nll", "")
        row["prompt_ppl"] = scores.get("ppl", "")
        ppl_by_prompt[(row["piece_id"], row["prompt_seed"])] = float(scores["ppl"])

    prompt_fields = list(prompt_rows[0])
    if "prompt_scored_tokens" not in prompt_fields:
        insert_at = prompt_fields.index("prompt_generated_tokens") + 1
        prompt_fields.insert(insert_at, "prompt_scored_tokens")
    write_csv(prompt_manifest_path, prompt_fields, prompt_rows)

    for name in ("continuation_manifest.csv", "blind_mapping_private.csv"):
        path = run_dir / name
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            row["prompt_ppl"] = ppl_by_prompt[(row["piece_id"], row["prompt_seed"])]
        write_csv(path, list(rows[0]), rows)

    config["prompt_ppl_definition"] = (
        "Raw Prompt Model PPL over accompaniment content tokens in only the "
        f"first {prompt_prefix_beats} generated beats; structural tokens excluded."
    )
    write_json(run_dir / "run_config.json", config)
    print(f"Rescored {len(prompt_rows)} prompts in {run_dir}")


if __name__ == "__main__":
    main()
