"""Run the beat-aligned rap-text prototype."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from streammuse.application.rap.rhythm import available_patterns
from streammuse.application.rap.service import CandidateGenerator, RapPrototypeService
from streammuse.domain.rap import RapPlan, ScheduledSyllable
from streammuse.domain.timing import Tempo
from streammuse.infrastructure.inference.local_chat_client import LocalChatModelClient, LocalChatModelClientConfig
from streammuse.infrastructure.rap.generators import LocalChatCandidateGenerator, PhraseBankGenerator


def build_parser() -> argparse.ArgumentParser:
    """Build the standalone prototype command parser."""
    parser = argparse.ArgumentParser(
        prog="streammuse-rap",
        description="Generate and inspect beat-aligned rap-text schedules.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--topic", default="the moment")
    parser.add_argument("--tempo", type=float, default=90.0, help="Tempo in beats per minute")
    parser.add_argument("--pattern", choices=available_patterns(), default="boom_bap")
    parser.add_argument("--bars", type=int, default=4)
    parser.add_argument("--candidate-count", type=int, default=12)
    parser.add_argument("--generator", choices=("phrase_bank", "local_chat"), default="phrase_bank")
    parser.add_argument("--model-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="local-model")
    parser.add_argument("--timeout-s", type=float, default=5.0)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--play", action="store_true", help="Emit scheduled syllables at the selected tempo")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Create, print, optionally export, and optionally play a rap-text plan."""
    args = build_parser().parse_args(argv)
    client: LocalChatModelClient | None = None
    try:
        generator, client = _build_generator(args)
        tempo = Tempo(bpm=args.tempo, ticks_per_beat=4, beats_per_bar=4)
        plan = RapPrototypeService(tempo, args.pattern, generator).build_plan(
            args.topic,
            bars=args.bars,
            candidate_count=args.candidate_count,
        )
    except ValueError as exc:
        print(f"Error: {exc}")
        return 2
    finally:
        if client is not None:
            client.close()

    print(render_plan(plan))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(plan_to_dict(plan), indent=2) + "\n", encoding="utf-8")
        print(f"Wrote schedule JSON: {args.output_json}")
    if args.play:
        play_plan(plan)
    return 0


def render_plan(plan: RapPlan) -> str:
    """Render each scheduled syllable with its musical and wall-clock position."""
    lines = [
        f"Topic: {plan.topic}",
        f"Tempo: {plan.tempo.bpm:g} BPM | Pattern: {plan.pattern} | Source: {plan.candidate_source}",
    ]
    if plan.warning:
        lines.append(f"Warning: {plan.warning}")

    for index, line in enumerate(plan.lines, start=1):
        lines.extend(
            [
                "",
                f"Bar {index}: {line.text}",
                f"  score={line.score:.2f} syllables={len(line.syllables)}",
                "  tick  position  seconds  accent  syllable",
            ]
        )
        for event in line.events:
            lines.append(_render_event_row(event, plan.tempo))
    return "\n".join(lines)


def plan_to_dict(plan: RapPlan) -> dict[str, Any]:
    """Serialize the plan in the shape a future voice adapter can consume."""
    return {
        "topic": plan.topic,
        "tempo": {
            "bpm": plan.tempo.bpm,
            "ticks_per_beat": plan.tempo.ticks_per_beat,
            "beats_per_bar": plan.tempo.beats_per_bar,
        },
        "pattern": plan.pattern,
        "candidate_source": plan.candidate_source,
        "warning": plan.warning,
        "lines": [
            {
                "text": line.text,
                "score": line.score,
                "syllable_count": len(line.syllables),
                "overflow_count": line.overflow_count,
                "events": [_event_to_dict(event, plan.tempo) for event in line.events],
            }
            for line in plan.lines
        ],
    }


def play_plan(
    plan: RapPlan,
    *,
    write: Callable[[str], None] = print,
    clock: Callable[[], float] = time.perf_counter,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Emit the already-aligned schedule in real time using a monotonic clock."""
    start = clock()
    for event in plan.events:
        target_offset = plan.tempo.tick_to_seconds(event.slot.tick)
        remaining = target_offset - (clock() - start)
        if remaining > 0:
            sleep(remaining)
        position = _position(event)
        suffix = "*" if event.syllable.stressed else ""
        write(f"[{position}] {event.syllable.label}{suffix}")


def _build_generator(args: argparse.Namespace) -> tuple[CandidateGenerator, LocalChatModelClient | None]:
    phrase_bank = PhraseBankGenerator()
    if args.generator == "phrase_bank":
        return phrase_bank, None

    client = LocalChatModelClient(
        LocalChatModelClientConfig(
            base_url=args.model_url,
            model=args.model,
            timeout_s=args.timeout_s,
        )
    )
    return LocalChatCandidateGenerator(client), client


def _render_event_row(event: ScheduledSyllable, tempo: Tempo) -> str:
    return (
        f"  {event.slot.tick:>4}  {_position(event):>7}  "
        f"{tempo.tick_to_seconds(event.slot.tick):>7.3f}  {event.slot.accent:>6.2f}  {event.syllable.label}"
    )


def _event_to_dict(event: ScheduledSyllable, tempo: Tempo) -> dict[str, Any]:
    return {
        "tick": event.slot.tick,
        "bar": event.slot.bar,
        "beat": event.slot.beat,
        "tick_in_beat": event.slot.tick_in_beat,
        "seconds": tempo.tick_to_seconds(event.slot.tick),
        "accent": event.slot.accent,
        "word": event.syllable.word,
        "label": event.syllable.label,
        "index_in_word": event.syllable.index_in_word,
        "syllable_count": event.syllable.syllable_count,
        "stressed": event.syllable.stressed,
    }


def _position(event: ScheduledSyllable) -> str:
    return f"B{event.slot.bar + 1} {event.slot.beat + 1}.{event.slot.tick_in_beat + 1}"
