"""CLI assembly for the terminal-only live rap showcase."""

from __future__ import annotations

import argparse
import platform
import subprocess
import time
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Callable
from uuid import uuid4

from streammuse.application.rap.monitoring import RapEventDispatcher, RapEventPublisher, RapStateProjector
from streammuse.application.rap.monitoring_payloads import flow_template_payload
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
from streammuse.infrastructure.rap.recorder import RapSessionRecorder, build_session_manifest
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
    if args.timeout_s <= 0:
        raise ValueError("timeout-s must be positive")
    if not 0.0 <= args.minimum_score <= 1.0:
        raise ValueError("minimum-score must be between zero and one")

    scenario = load_scenario(args.scenario) if args.scenario else default_scenario()
    if not scenario.loop and args.max_bars == 0:
        raise ValueError("max-bars 0 requires a looping scenario")
    if not scenario.loop and args.max_bars > scenario.total_bars:
        raise ValueError("max-bars exceeds the non-looping scenario length")
    tempo = Tempo(scenario.tempo_bpm, 4, 4)
    analyzer = CmuProsodyAnalyzer()
    fallbacks = PrevalidatedFallbackCatalog.build(scenario, BUILTIN_TEMPLATES, analyzer)
    generator, close_primary = _build_generator(args)
    session_id = f"rap-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}"
    session_dir = args.log_dir / session_id
    manifest = _build_manifest(args, scenario, tempo)
    manifest.update({"session_id": session_id, "max_bars": args.max_bars})
    recorder = RapSessionRecorder(session_dir, manifest)

    publisher = RapEventPublisher(session_id)
    projector = RapStateProjector()
    dispatcher = RapEventDispatcher(
        publisher.queue,
        sinks=(recorder, projector, TerminalRapSink(args.terminal_detail)),
    )
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
        planning_bar_limit=args.max_bars or (None if scenario.loop else scenario.total_bars),
        close_primary=close_primary,
        monotonic=clock,
    )
    tick_loop = RapTickLoop(tempo, on_tick=controller.on_tick, clock=clock, sleep=sleep)
    session_metadata = {
        "scenario_id": scenario.scenario_id,
        "generator": args.generator,
        "model_url": args.model_url,
        "model": args.model,
        "candidate_count": args.candidate_count,
        "lookahead_bars": args.lookahead_bars,
        "minimum_score": args.minimum_score,
        "seed": args.seed,
        "score_weights": manifest["score_weights"],
    }
    return RapDemoDependencies(
        tempo,
        controller,
        publisher,
        dispatcher,
        tick_loop,
        session_dir,
        session_metadata=session_metadata,
        recorder=recorder,
        projector=projector,
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


def _build_manifest(args: argparse.Namespace, scenario, tempo: Tempo) -> dict[str, object]:
    revision, dirty = _repository_state()
    templates = []
    seen: set[str] = set()
    for segment in scenario.segments:
        if segment.template_id in seen:
            continue
        seen.add(segment.template_id)
        payload = flow_template_payload(BUILTIN_TEMPLATES.get(segment.template_id))
        templates.append(
            {
                "template_id": payload["template_id"],
                "name": payload["name"],
                "definition": {
                    "ticks_per_beat": payload["ticks_per_beat"],
                    "beats_per_bar": payload["beats_per_bar"],
                    "slots": [
                        {
                            "tick_in_bar": slot["tick_in_bar"],
                            "duration_ticks": slot["duration_ticks"],
                            "target_stress": slot["target_stress"],
                            "boundary_strength": slot["boundary_strength"],
                            "rhyme_group": slot["rhyme_group"],
                        }
                        for slot in payload["slots"]
                    ],
                },
                "provenance": dict(payload["provenance"]),
            }
        )
    scenario_payload = {
        "scenario_id": scenario.scenario_id,
        "tempo_bpm": scenario.tempo_bpm,
        "loop": scenario.loop,
        "segments": [
            {
                "start_bar": segment.start_bar,
                "bars": segment.bars,
                "topic": segment.topic,
                "template_id": segment.template_id,
                "fallback_lines": list(segment.fallback_lines),
            }
            for segment in scenario.segments
        ],
    }
    try:
        package_version = version("streammuse")
    except PackageNotFoundError:
        package_version = "0.1.0"
    return build_session_manifest(
        scenario_id=scenario.scenario_id,
        scenario=scenario_payload,
        seed=args.seed,
        tempo={
            "bpm": tempo.bpm,
            "ticks_per_beat": tempo.ticks_per_beat,
            "beats_per_bar": tempo.beats_per_bar,
        },
        templates=templates,
        generator_config={"name": args.generator, "candidate_count": args.candidate_count},
        model_config={
            "name": args.model if args.generator == "local_chat" else "none",
            "base_url": args.model_url if args.generator == "local_chat" else None,
        },
        score_weights=asdict(ScoreWeights()),
        minimum_score=args.minimum_score,
        timeout_seconds=args.timeout_s,
        lookahead_bars=args.lookahead_bars,
        python_version=platform.python_version(),
        platform=platform.platform(),
        package_version=package_version,
        git_revision=revision,
        git_dirty=dirty,
    )


def _repository_state() -> tuple[str, bool]:
    root = Path(__file__).resolve().parents[4]
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True, timeout=2.0
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"], cwd=root, check=True, capture_output=True, text=True, timeout=2.0
            ).stdout.strip()
        )
        return revision or "unknown", dirty
    except (OSError, subprocess.SubprocessError):
        return "unknown", False


if __name__ == "__main__":
    raise SystemExit(main())
