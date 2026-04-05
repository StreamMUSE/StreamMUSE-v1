from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import requests


def parse_int_list(raw: str) -> list[int]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("List argument cannot be empty")
    return [int(v) for v in values]


def percentile(samples: list[float], p: float) -> float:
    if not samples:
        return 0.0
    ordered = sorted(samples)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * p
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    weight = rank - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def benchmark_one_case(
    url: str,
    timeout_s: float,
    num_requests: int,
    warmup_requests: int,
    generation_length_frames: int,
    generation_interval_ticks: int,
) -> dict[str, float | int]:
    payload = {
        "melody_notes": [
            {"type": "note_on", "pitch": 60, "tick": 0},
            {"type": "note_off", "pitch": 60, "tick": 4},
        ],
        "generation_start_tick": 8,
        "generation_length_frames": int(generation_length_frames),
        "generation_interval_ticks": int(generation_interval_ticks),
        "prompt_length_ticks": None,
        "inference_mode": "sliding_window",
        "model_name": "lekai",
        "checkpoint_path": None,
    }

    for _ in range(max(0, warmup_requests)):
        resp = requests.post(url, json=payload, timeout=timeout_s)
        resp.raise_for_status()

    latencies_ms: list[float] = []
    status_codes: list[int] = []

    for _ in range(max(1, num_requests)):
        start = time.perf_counter()
        resp = requests.post(url, json=payload, timeout=timeout_s)
        elapsed_ms = (time.perf_counter() - start) * 1000
        status_codes.append(resp.status_code)
        resp.raise_for_status()
        latencies_ms.append(elapsed_ms)

    return {
        "generation_length_frames": generation_length_frames,
        "generation_interval_ticks": generation_interval_ticks,
        "count": len(latencies_ms),
        "min_ms": min(latencies_ms),
        "max_ms": max(latencies_ms),
        "mean_ms": statistics.mean(latencies_ms),
        "p50_ms": percentile(latencies_ms, 0.50),
        "p95_ms": percentile(latencies_ms, 0.95),
        "p99_ms": percentile(latencies_ms, 0.99),
        "status_code": status_codes[-1],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Lekai HTTP endpoint latency matrix.")
    parser.add_argument(
        "--url",
        type=str,
        default="http://127.0.0.1:8000/generate_accompaniment",
        help="Lekai generate endpoint URL",
    )
    parser.add_argument("--timeout-s", type=float, default=30.0)
    parser.add_argument("--num-requests", type=int, default=20)
    parser.add_argument("--warmup-requests", type=int, default=3)
    parser.add_argument("--generation-length-frames-list", type=str, default="8,12,16,20")
    parser.add_argument("--generation-interval-ticks-list", type=str, default="2,4,8")
    parser.add_argument("--output", type=str, default=None, help="Optional JSON output path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lengths = parse_int_list(args.generation_length_frames_list)
    intervals = parse_int_list(args.generation_interval_ticks_list)

    matrix_results: list[dict[str, float | int]] = []
    run_started = time.time()

    print(f"[benchmark] url={args.url}")
    print(f"[benchmark] lengths={lengths}, intervals={intervals}")

    for length in lengths:
        for interval in intervals:
            print(f"[benchmark] running length={length}, interval={interval}")
            result = benchmark_one_case(
                url=args.url,
                timeout_s=float(args.timeout_s),
                num_requests=int(args.num_requests),
                warmup_requests=int(args.warmup_requests),
                generation_length_frames=length,
                generation_interval_ticks=interval,
            )
            matrix_results.append(result)
            print(
                f"[benchmark] length={length}, interval={interval}, "
                f"p50={result['p50_ms']:.1f}ms, p95={result['p95_ms']:.1f}ms, p99={result['p99_ms']:.1f}ms"
            )

    payload = {
        "timestamp": run_started,
        "url": args.url,
        "num_requests": int(args.num_requests),
        "warmup_requests": int(args.warmup_requests),
        "results": matrix_results,
    }

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[benchmark] wrote report to {output_path}")
    else:
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
