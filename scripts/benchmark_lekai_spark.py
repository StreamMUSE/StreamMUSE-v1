from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from pathlib import Path
from typing import Any


TIMESTEPS_PER_BEAT = 4


def percentile(samples: list[float], q: float) -> float:
    if not samples:
        return 0.0
    ordered = sorted(samples)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * q
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    weight = rank - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def summarize_ms(samples: list[float]) -> dict[str, float | int | None]:
    if not samples:
        return {
            "count": 0,
            "min_ms": None,
            "mean_ms": None,
            "median_ms": None,
            "p90_ms": None,
            "p95_ms": None,
            "max_ms": None,
        }
    return {
        "count": len(samples),
        "min_ms": min(samples),
        "mean_ms": statistics.mean(samples),
        "median_ms": statistics.median(samples),
        "p90_ms": percentile(samples, 0.90),
        "p95_ms": percentile(samples, 0.95),
        "max_ms": max(samples),
    }


def sync_cuda() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        return


def cuda_memory() -> dict[str, int | None]:
    try:
        import torch

        if not torch.cuda.is_available():
            return {"cuda_allocated": None, "cuda_reserved": None}
        return {
            "cuda_allocated": int(torch.cuda.memory_allocated()),
            "cuda_reserved": int(torch.cuda.memory_reserved()),
        }
    except Exception:
        return {"cuda_allocated": None, "cuda_reserved": None}


def parse_int_list(raw: str) -> list[int]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("integer list cannot be empty")
    return [int(item) for item in values]


def load_midi_events(
    midi_path: str,
    *,
    max_tick: int | None,
    trim_leading_rest: bool,
    velocity: int = 80,
) -> tuple[list[dict[str, int | str]], dict[str, int | bool]]:
    from streammuse.infrastructure.input.midi_file import MidiFileInput

    notes, resolution, actual_max_tick = MidiFileInput._midi_to_notes(
        midi_path,
        beat_div=TIMESTEPS_PER_BEAT,
        min_pitch=0,
        max_pitch=127,
        program=None,
        max_tick=max_tick,
    )
    first_tick = min((int(note["tick"]) for note in notes), default=0)
    offset = first_tick if trim_leading_rest else 0

    events: list[dict[str, int | str]] = []
    for note in notes:
        tick = int(note["tick"]) - offset
        if tick < 0:
            continue
        pitch = int(note["pitch"])
        duration = max(1, int(note["duration"]))
        events.append(
            {
                "type": "note_on",
                "pitch": pitch,
                "tick": tick,
                "velocity": int(velocity),
            }
        )
        events.append({"type": "note_off", "pitch": pitch, "tick": tick + duration})
    events.sort(
        key=lambda event: (
            int(event["tick"]),
            0 if str(event["type"]) == "note_off" else 1,
            int(event["pitch"]),
        )
    )
    return events, {
        "note_count": len(notes),
        "event_count": len(events),
        "resolution": int(resolution),
        "actual_max_tick": int(actual_max_tick),
        "first_note_tick_original": int(first_tick),
        "trim_leading_rest": bool(trim_leading_rest),
    }


def events_until(events: list[dict[str, int | str]], start_tick: int, end_tick: int) -> list[dict[str, int | str]]:
    return [
        dict(event)
        for event in events
        if int(start_tick) <= int(event["tick"]) < int(end_tick)
    ]


def benchmark_prompt(
    *,
    checkpoint_path: str,
    melody_events: list[dict[str, int | str]],
    prompt_length_ticks: int,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    from streammuse.infrastructure.inference.lekai_prompt_continuation.prompt_engine import (
        LekaiPromptEngine,
    )

    load_start = time.perf_counter()
    engine = LekaiPromptEngine(checkpoint_path=checkpoint_path)
    sync_cuda()
    load_ms = (time.perf_counter() - load_start) * 1000

    for _ in range(max(0, warmup)):
        engine.generate_prompt_accompaniment(
            melody_events=melody_events,
            prompt_start_tick=0,
            prompt_length_ticks=prompt_length_ticks,
        )
        sync_cuda()

    samples: list[float] = []
    event_counts: list[int] = []
    acc_beats: list[int] = []
    for _ in range(max(1, repeats)):
        start = time.perf_counter()
        accompaniment = engine.generate_prompt_accompaniment(
            melody_events=melody_events,
            prompt_start_tick=0,
            prompt_length_ticks=prompt_length_ticks,
        )
        sync_cuda()
        samples.append((time.perf_counter() - start) * 1000)
        event_counts.append(len(accompaniment))
        acc_beats.append(int(engine.last_generated_acc_beats()))

    return {
        "load_ms_measured": load_ms,
        "runtime_info": engine.runtime_info(),
        "generate_ms": summarize_ms(samples),
        "raw_generate_ms": samples,
        "accompaniment_event_count_last": event_counts[-1] if event_counts else 0,
        "accompaniment_event_count_mean": statistics.mean(event_counts) if event_counts else 0,
        "generated_acc_beats_last": acc_beats[-1] if acc_beats else 0,
        **cuda_memory(),
    }


def benchmark_continuation(
    *,
    checkpoint_path: str,
    melody_events: list[dict[str, int | str]],
    warmup_requests: int,
    requests: int,
    generation_interval_ticks: int,
    generation_length_frames: int,
) -> dict[str, Any]:
    from streammuse.infrastructure.inference.lekai_http_backend import LekaiHttpBackend

    load_start = time.perf_counter()
    backend = LekaiHttpBackend(checkpoint_path=checkpoint_path)
    sync_cuda()
    load_ms = (time.perf_counter() - load_start) * 1000

    total = max(0, warmup_requests) + max(1, requests)
    sent_index = 0
    measured_round_trip_ms: list[float] = []
    measured_server_ms: list[float] = []
    event_counts: list[int] = []

    for idx in range(total):
        generation_start_tick = (idx + 1) * int(generation_interval_ticks)
        increment: list[dict[str, int | str]] = []
        while sent_index < len(melody_events) and int(melody_events[sent_index]["tick"]) < generation_start_tick:
            increment.append(dict(melody_events[sent_index]))
            sent_index += 1

        start = time.perf_counter()
        accompaniment, timings = backend.generate(
            melody_events=increment,
            generation_start_tick=generation_start_tick,
            generation_length_frames=generation_length_frames,
            generation_interval_ticks=generation_interval_ticks,
            prompt_length_ticks=None,
            inference_mode="sliding_window",
            model_name="lekai",
            checkpoint_path=None,
        )
        sync_cuda()
        elapsed_ms = (time.perf_counter() - start) * 1000
        server_ms = (
            float(timings["inference_end_time"]) - float(timings["inference_start_time"])
        ) * 1000

        if idx >= warmup_requests:
            measured_round_trip_ms.append(elapsed_ms)
            measured_server_ms.append(server_ms)
            event_counts.append(len(accompaniment))

    return {
        "load_ms_measured": load_ms,
        "runtime_info": backend.runtime_info(),
        "round_trip_ms_per_beat": summarize_ms(measured_round_trip_ms),
        "server_inference_ms_per_beat": summarize_ms(measured_server_ms),
        "raw_round_trip_ms": measured_round_trip_ms,
        "raw_server_inference_ms": measured_server_ms,
        "accompaniment_event_count_last": event_counts[-1] if event_counts else 0,
        "accompaniment_event_count_mean": statistics.mean(event_counts) if event_counts else 0,
        **cuda_memory(),
    }


def benchmark_scheduler(
    *,
    prompt_checkpoint_path: str,
    continuation_checkpoint_path: str,
    melody_events: list[dict[str, int | str]],
    prompt_length_ticks: int,
    generation_interval_ticks: int,
    observed_until_ticks: list[int],
    timeout_s: float,
) -> list[dict[str, Any]]:
    from streammuse.infrastructure.inference.lekai_prompt_continuation import (
        LekaiPromptContinuationBackend,
    )

    results: list[dict[str, Any]] = []
    for observed_until_tick in observed_until_ticks:
        backend = LekaiPromptContinuationBackend(
            prompt_checkpoint_path=prompt_checkpoint_path,
            continuation_checkpoint_path=continuation_checkpoint_path,
        )
        sync_cuda()
        prompt_events = events_until(melody_events, 0, prompt_length_ticks)
        append_events = events_until(melody_events, prompt_length_ticks, observed_until_tick)

        start = time.perf_counter()
        start_status = backend.start_prompt_catchup(
            melody_events=prompt_events,
            prompt_length_ticks=prompt_length_ticks,
            generation_interval_ticks=generation_interval_ticks,
            inference_mode="sliding_window",
            model_name="lekai_prompt_continuation",
            checkpoint_path=None,
            observed_until_tick=prompt_length_ticks,
        )
        append_status = backend.append_melody_events(
            append_events,
            observed_until_tick=observed_until_tick,
        )

        samples: list[dict[str, Any]] = []
        last_marker: tuple[Any, Any] | None = None
        deadline = time.perf_counter() + timeout_s
        final_status: dict[str, Any]
        while True:
            status = backend.scheduler_status()
            marker = (status.get("phase"), status.get("continuation_calls"))
            if marker != last_marker:
                samples.append({"elapsed_s": round(time.perf_counter() - start, 3), **status})
                last_marker = marker
            if status.get("is_playback_ready") or status.get("is_failed") or time.perf_counter() >= deadline:
                final_status = dict(status)
                break
            time.sleep(0.05)
        sync_cuda()

        results.append(
            {
                "observed_until_tick": int(observed_until_tick),
                "prompt_event_count": len(prompt_events),
                "append_event_count": len(append_events),
                "start_status": start_status,
                "append_status": append_status,
                "total_scheduler_ms": (time.perf_counter() - start) * 1000,
                "final_status": final_status,
                "samples": samples,
                "playable_event_count": len(backend.playable_accompaniment()),
                "prompt_history_event_count": len(backend.prompt_accompaniment_history()),
                "raw_history_event_count": len(backend.raw_accompaniment_history()),
            }
        )
        backend.clear_history()
    return results


def analyze_console_log(path: Path) -> dict[str, Any]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    current_tick: int | None = None
    first_user_tick: int | None = None
    first_model_tick: int | None = None
    first_model_line: str | None = None
    first_ready_tick: int | None = None
    model_event_count = 0
    schedule_reports: list[dict[str, Any]] = []

    for line in lines:
        tick_match = re.search(r"\[tick\] tick=(\d+)", line)
        if tick_match:
            current_tick = int(tick_match.group(1))
        if first_ready_tick is None and "Prompt-continuation accompaniment is playable" in line:
            first_ready_tick = current_tick
        if first_user_tick is None and "[event] source=user" in line:
            first_user_tick = current_tick
        if "[event] source=model" in line:
            model_event_count += 1
            if first_model_tick is None:
                first_model_tick = current_tick
                first_model_line = line
        if "Scheduled " in line and "playable accompaniment event(s)" in line:
            match = re.search(
                r"Scheduled (\d+) playable accompaniment event\(s\); "
                r"dropped (\d+) past event\(s\);(?: clipped (\d+) sustaining note\(s\);)?",
                line,
            )
            if match:
                schedule_reports.append(
                    {
                        "tick_context": current_tick,
                        "scheduled": int(match.group(1)),
                        "dropped": int(match.group(2)),
                        "clipped": int(match.group(3) or 0),
                        "line": line,
                    }
                )

    scheduled_zero = [row for row in schedule_reports if int(row["scheduled"]) == 0]
    return {
        "path": str(path),
        "first_user_tick": first_user_tick,
        "first_user_seconds_at_120bpm": None if first_user_tick is None else first_user_tick * 0.125,
        "first_ready_tick": first_ready_tick,
        "first_ready_seconds_at_120bpm": None if first_ready_tick is None else first_ready_tick * 0.125,
        "first_model_tick": first_model_tick,
        "first_model_seconds_at_120bpm": None if first_model_tick is None else first_model_tick * 0.125,
        "first_model_line": first_model_line,
        "model_event_count": model_event_count,
        "schedule_report_count": len(schedule_reports),
        "scheduled_zero_count": len(scheduled_zero),
        "scheduled_zero_before_first_model": sum(
            1
            for row in scheduled_zero
            if first_model_tick is not None
            and row["tick_context"] is not None
            and int(row["tick_context"]) < first_model_tick
        ),
        "dropped_before_first_model": sum(
            int(row["dropped"])
            for row in schedule_reports
            if first_model_tick is not None
            and row["tick_context"] is not None
            and int(row["tick_context"]) < first_model_tick
        ),
        "total_dropped_in_schedule_reports": sum(int(row["dropped"]) for row in schedule_reports),
        "first_8_schedule_reports": schedule_reports[:8],
    }


def command_micro(args: argparse.Namespace) -> None:
    events, midi_info = load_midi_events(
        args.midi_file,
        max_tick=args.midi_max_tick,
        trim_leading_rest=args.trim_leading_rest,
    )
    prompt_events = events_until(events, 0, args.prompt_length_ticks)

    payload: dict[str, Any] = {
        "timestamp": time.time(),
        "midi": midi_info,
        "config": {
            "prompt_length_ticks": args.prompt_length_ticks,
            "generation_interval_ticks": args.generation_interval_ticks,
            "generation_length_frames": args.generation_length_frames,
            "prompt_repeats": args.prompt_repeats,
            "continuation_requests": args.continuation_requests,
            "observed_until_ticks": args.observed_until_ticks,
        },
    }

    payload["prompt_model"] = benchmark_prompt(
        checkpoint_path=args.prompt_checkpoint,
        melody_events=prompt_events,
        prompt_length_ticks=args.prompt_length_ticks,
        warmup=args.prompt_warmup,
        repeats=args.prompt_repeats,
    )

    payload["continuation_model"] = benchmark_continuation(
        checkpoint_path=args.continuation_checkpoint,
        melody_events=events,
        warmup_requests=args.continuation_warmup,
        requests=args.continuation_requests,
        generation_interval_ticks=args.generation_interval_ticks,
        generation_length_frames=args.generation_length_frames,
    )

    if not args.skip_scheduler:
        payload["prompt_continuation_scheduler"] = benchmark_scheduler(
            prompt_checkpoint_path=args.prompt_checkpoint,
            continuation_checkpoint_path=args.continuation_checkpoint,
            melody_events=events,
            prompt_length_ticks=args.prompt_length_ticks,
            generation_interval_ticks=args.generation_interval_ticks,
            observed_until_ticks=parse_int_list(args.observed_until_ticks),
            timeout_s=args.scheduler_timeout_s,
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"[spark-bench] wrote {output}")


def command_analyze_console(args: argparse.Namespace) -> None:
    results = {}
    for item in args.case:
        if "=" not in item:
            raise ValueError(f"--case must use name=path, got {item}")
        name, raw_path = item.split("=", 1)
        results[name] = analyze_console_log(Path(raw_path))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"[spark-bench] wrote {output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Spark/H200 Lekai benchmark helpers.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    micro = subparsers.add_parser("micro", help="Benchmark prompt, continuation, and scheduler directly.")
    micro.add_argument("--prompt-checkpoint", required=True)
    micro.add_argument("--continuation-checkpoint", required=True)
    micro.add_argument("--midi-file", required=True)
    micro.add_argument("--output", required=True)
    micro.add_argument("--midi-max-tick", type=int, default=320)
    micro.add_argument("--trim-leading-rest", action="store_true")
    micro.add_argument("--prompt-length-ticks", type=int, default=32)
    micro.add_argument("--generation-interval-ticks", type=int, default=4)
    micro.add_argument("--generation-length-frames", type=int, default=4)
    micro.add_argument("--prompt-warmup", type=int, default=0)
    micro.add_argument("--prompt-repeats", type=int, default=3)
    micro.add_argument("--continuation-warmup", type=int, default=5)
    micro.add_argument("--continuation-requests", type=int, default=50)
    micro.add_argument("--observed-until-ticks", type=str, default="48,64")
    micro.add_argument("--scheduler-timeout-s", type=float, default=120.0)
    micro.add_argument("--skip-scheduler", action="store_true")
    micro.set_defaults(func=command_micro)

    analyze = subparsers.add_parser("analyze-console", help="Analyze public CLI console logs.")
    analyze.add_argument("--case", action="append", required=True, help="Case in name=path form.")
    analyze.add_argument("--output", required=True)
    analyze.set_defaults(func=command_analyze_console)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
