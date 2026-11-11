"""Run a real-model, text-only eight-turn character dialogue.

No services or GPU processes are launched. Run with PYTHONPATH=src and the
existing .venv Python. Failed/cancelled runs retain their evidence artifacts.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path

from streammuse.application.rap.characters import CharacterExperiment, ExperimentConfig


def add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-url", help="Override STREAMMUSE_RESEARCH_MODEL_URL")
    parser.add_argument("--model", help="Override STREAMMUSE_RESEARCH_MODEL (default qwen-rap)")
    parser.add_argument("--timeout-s", type=float, default=60.0)
    parser.add_argument("--max-calls", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)


def config_from_args(args: argparse.Namespace, **overrides) -> ExperimentConfig:
    return ExperimentConfig(
        model_url=args.model_url, model=args.model, timeout_s=args.timeout_s,
        max_calls=args.max_calls, seed=overrides.pop("seed", args.seed), **overrides,
    )


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


async def run_to_directory(config: ExperimentConfig, output_dir: Path) -> dict:
    experiment = CharacterExperiment(config)
    destination = output_dir.resolve() / f"{config.policy}-{experiment.run_id}"
    destination.mkdir(parents=True, exist_ok=False)
    _write_json(destination / "manifest.json", {
        "run_id": experiment.run_id, "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": asdict(config), "model_url": experiment.model_url, "model": experiment.model,
        "actors": experiment.snapshot()["actors"], "real_model_required": True,
        "clock_domain": "client_monotonic", "correctness_source": "host_referee",
        "prompt_budget": "1400 UTF-8 content bytes; not a measured tokenizer count",
        "sampling": {"temperature": 0.0, "seed": config.seed, "n": 1, "enable_thinking": False},
        "scope": "text only; no audio; shared model weights, separate per-actor prompts and clients",
    })
    try:
        await experiment.run()
    finally:
        result = experiment.snapshot()
        result["artifact_path"] = str(destination)
        _write_json(destination / "summary.json", result)
        _write_jsonl(destination / "events.jsonl", experiment.event_page()["events"])
        _write_jsonl(destination / "proposals.jsonl", result["proposals"])
        _write_jsonl(destination / "accepted.jsonl", result["accepted"])
        _write_jsonl(destination / "publication_attempts.jsonl", result["publication_attempts"])
    return result


def print_summary(result: dict) -> None:
    print(json.dumps({key: result[key] for key in (
        "run_id", "state", "mode", "policy", "seed", "metrics", "error", "artifact_path",
    )}, ensure_ascii=False))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_arguments(parser)
    parser.add_argument("--turns", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=Path("output/research_day_20261105/characters"))
    args = parser.parse_args(argv)
    try:
        config = config_from_args(args, mode="dialogue", policy="fixed_turn", turns=args.turns)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        result = asyncio.run(run_to_directory(config, args.output_dir))
    except KeyboardInterrupt:
        return 130
    print_summary(result)
    return 0 if result["state"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
