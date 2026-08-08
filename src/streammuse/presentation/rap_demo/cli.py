"""CLI assembly for the terminal-only live rap showcase."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
from uuid import uuid4

from streammuse.application.rap.monitoring import RapEventDispatcher, RapEventPublisher
from streammuse.application.rap.realtime import RollingRapController
from streammuse.application.rap.runtime import RapDemoDependencies, RapTickLoop
from streammuse.domain.rap import ScoreWeights
from streammuse.domain.timing import Tempo
from streammuse.infrastructure.inference.local_chat_client import LocalChatModelClient, LocalChatModelClientConfig
from streammuse.infrastructure.rap.fallback import PrevalidatedFallbackCatalog
from streammuse.infrastructure.rap.generators import (
    LocalChatCandidateGenerator,
    PhraseBankGenerator,
    ScriptedFailureGenerator,
)
from streammuse.infrastructure.rap.prosody import CmuProsodyAnalyzer
from streammuse.infrastructure.rap.scenarios import default_scenario, load_scenario
from streammuse.infrastructure.rap.templates import BUILTIN_TEMPLATES
from streammuse.presentation.rap_demo.terminal import TerminalRapSink


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="streammuse-rap-demo",
        description="Run the terminal-only realtime rap showcase.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--scenario", type=Path)
    parser.add_argument("--generator", choices=("phrase_bank", "local_chat", "scripted_failure"), default="phrase_bank")
    parser.add_argument("--model-url", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--model", default="qwen-rap")
    parser.add_argument("--timeout-s", type=float, default=5.0)
    parser.add_argument("--candidate-count", type=int, default=8)
    parser.add_argument("--lookahead-bars", type=int, default=2)
    parser.add_argument("--minimum-score", type=float, default=0.55)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--max-bars", type=int, default=12, help="Zero runs until interrupted")
    parser.add_argument("--log-dir", type=Path, default=Path("logs/rap"))
    parser.add_argument("--terminal-detail", choices=("summary", "candidates", "full"), default="full")
    parser.add_argument("--host", default="127.0.0.1", help=argparse.SUPPRESS)
    parser.add_argument("--port", type=int, default=8012, help=argparse.SUPPRESS)
    parser.add_argument("--no-web", action="store_true", help="Accepted for terminal-only compatibility")
    return parser


def build_demo(
    args: argparse.Namespace,
    *,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> RapDemoDependencies:
    if args.max_bars < 0:
        raise ValueError("max-bars must not be negative")
    if args.candidate_count <= 0 or args.lookahead_bars <= 0:
        raise ValueError("candidate-count and lookahead-bars must be positive")
    if not 0.0 <= args.minimum_score <= 1.0:
        raise ValueError("minimum-score must be between zero and one")

    scenario = load_scenario(args.scenario) if args.scenario else default_scenario()
    tempo = Tempo(scenario.tempo_bpm, 4, 4)
    analyzer = CmuProsodyAnalyzer()
    fallbacks = PrevalidatedFallbackCatalog.build(scenario, BUILTIN_TEMPLATES, analyzer)
    generator, close_primary = _build_generator(args)
    session_id = f"rap-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}"
    session_dir = args.log_dir / session_id
    session_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "session_id": session_id,
        "scenario_id": scenario.scenario_id,
        "tempo_bpm": scenario.tempo_bpm,
        "generator": args.generator,
        "model_url": args.model_url,
        "model": args.model,
        "candidate_count": args.candidate_count,
        "lookahead_bars": args.lookahead_bars,
        "minimum_score": args.minimum_score,
        "seed": args.seed,
        "max_bars": args.max_bars,
        "score_weights": asdict(ScoreWeights()),
    }
    (session_dir / "session.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    publisher = RapEventPublisher(session_id)
    dispatcher = RapEventDispatcher(publisher.queue, sinks=(TerminalRapSink(args.terminal_detail),))
    dispatcher.start()
    controller = RollingRapController(
        tempo=tempo,
        scenario=scenario,
        templates=BUILTIN_TEMPLATES,
        fallback_catalog=fallbacks,
        analyzer=analyzer,
        weights=ScoreWeights(),
        publisher=publisher,
        primary_generator=generator,
        candidate_count=args.candidate_count,
        lookahead_bars=args.lookahead_bars,
        minimum_score=args.minimum_score,
        seed=args.seed,
        planning_bar_limit=args.max_bars or None,
        close_primary=close_primary,
        monotonic=clock,
    )
    tick_loop = RapTickLoop(tempo, on_tick=controller.on_tick, clock=clock, sleep=sleep)
    session_metadata = {
        key: manifest[key]
        for key in (
            "scenario_id",
            "generator",
            "model_url",
            "model",
            "candidate_count",
            "lookahead_bars",
            "minimum_score",
            "seed",
            "score_weights",
        )
    }
    return RapDemoDependencies(
        tempo,
        controller,
        publisher,
        dispatcher,
        tick_loop,
        session_dir,
        session_metadata=session_metadata,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        demo = build_demo(args)
        demo.run(max_bars=args.max_bars)
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}")
        return 2
    return 0


def _build_generator(args: argparse.Namespace):
    if args.generator == "phrase_bank":
        return PhraseBankGenerator(), None
    if args.generator == "scripted_failure":
        return ScriptedFailureGenerator(), None
    client = LocalChatModelClient(
        LocalChatModelClientConfig(
            base_url=args.model_url,
            model=args.model,
            timeout_s=args.timeout_s,
        )
    )
    return LocalChatCandidateGenerator(client), client.close


if __name__ == "__main__":
    raise SystemExit(main())
