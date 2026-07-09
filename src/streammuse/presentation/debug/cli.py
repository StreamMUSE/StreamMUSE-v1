"""CLI entrypoint for StreamMUSE replay debugger."""

from __future__ import annotations

import argparse
import json
import sys

from streammuse.application.debug.replay import ReplayConfig, run_replay


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="streammuse-debug")
    subcommands = parser.add_subparsers(dest="command", required=True)
    replay = subcommands.add_parser("replay", help="Run a debug replay and emit trace artifacts")
    replay.add_argument("--scenario", default="lekai-prompt-continuation")
    replay.add_argument("--midi-file", required=True)
    replay.add_argument("--compare", default="offline,sim")
    replay.add_argument("--output-dir", required=True)
    replay.add_argument("--prompt-checkpoint")
    replay.add_argument("--continuation-checkpoint")
    replay.add_argument("--device", default="auto")
    replay.add_argument("--prompt-length-ticks", type=int, default=32)
    replay.add_argument("--generation-interval-ticks", type=int, default=4)
    replay.add_argument("--timeout-s", type=float, default=120.0)
    replay.add_argument("--bpm", type=int, default=120)
    replay.add_argument("--ticks-per-beat", type=int, default=4)
    replay.add_argument("--max-tick", type=int)
    replay.add_argument("--trim-leading-rest", action="store_true")
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    if args.command == "replay":
        result = run_replay(
            ReplayConfig(
                scenario=args.scenario,
                midi_file=args.midi_file,
                compare=tuple(part.strip() for part in args.compare.split(",") if part.strip()),
                output_dir=args.output_dir,
                prompt_checkpoint=args.prompt_checkpoint,
                continuation_checkpoint=args.continuation_checkpoint,
                device=args.device,
                prompt_length_ticks=args.prompt_length_ticks,
                generation_interval_ticks=args.generation_interval_ticks,
                timeout_s=args.timeout_s,
                bpm=args.bpm,
                ticks_per_beat=args.ticks_per_beat,
                max_tick=args.max_tick,
                trim_leading_rest=args.trim_leading_rest,
            )
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
