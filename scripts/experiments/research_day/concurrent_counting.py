"""Compare real-model fixed-turn, naive, and version-checked two-actor counting.

Each run is capped at 60 seconds and counts 1..10. Policies run sequentially so
only one experiment occupies the shared endpoint at a time. Naive failures are
retained, not repaired. Host arbitration is not evidence of model intelligence.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

if __package__:
    from .characters import add_model_arguments, config_from_args, print_summary, run_to_directory
else:
    from characters import add_model_arguments, config_from_args, print_summary, run_to_directory


async def run_comparison(configs, output_dir: Path) -> list[dict]:
    results = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for config in configs:
        result = await run_to_directory(config, output_dir)
        results.append({key: result[key] for key in (
            "run_id", "state", "policy", "seed", "metrics", "error", "artifact_path",
        )})
        print_summary(result)
        (output_dir / "comparison.json").write_text(json.dumps({
            "runs": results,
            "correctness_source": "host_referee",
            "interpretation": "Accepted-prefix correctness is enforced/audited by code, not proof of unaided model coordination.",
        }, indent=2) + "\n", encoding="utf-8")
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_arguments(parser)
    parser.add_argument("--policy", choices=("all", "fixed_turn", "naive", "coordinated"), default="all")
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=Path("output/research_day_20261105/counting"))
    args = parser.parse_args(argv)
    if not 1 <= args.trials <= 100:
        parser.error("trials must be in 1..100")
    policies = ("fixed_turn", "naive", "coordinated") if args.policy == "all" else (args.policy,)
    try:
        configs = [config_from_args(args, mode="counting", policy=policy, seed=args.seed + trial)
                   for trial in range(args.trials) for policy in policies]
    except ValueError as exc:
        parser.error(str(exc))
    try:
        results = asyncio.run(run_comparison(configs, args.output_dir))
    except KeyboardInterrupt:
        return 130
    return 0 if all(result["state"] == "completed" for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
