"""Make timed steering edits on the running Mac app and save the trace.

The app must be stopped when this starts. It switches the inference page to
steering mode, starts playback, and makes ``--edits`` edits: the first half of
them in the first half of a bar, the rest in the second half, alternating
between two topics and two flow templates. After each edit it waits until the
new version is performed. It then stops playback and writes ``trace.json``,
``final.json`` and ``edits.json`` to ``--output``.

    uv run python scripts/experiments/inference_opt/steering_edits.py \\
        --url http://127.0.0.1:8018 --bpm 60 --edits 6 --output output/inference_opt/steering-60
    uv run python scripts/analyze_inference_trace.py output/inference_opt/steering-60/trace.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import urllib.request


def call(base: str, path: str, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(base + path, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=15) as response:
        return json.load(response)


def wait_for(base: str, predicate, timeout: float):
    deadline = time.monotonic() + timeout
    while True:
        state = call(base, "/api/inference/state")
        if predicate(state):
            return state
        if state.get("state") == "error" or (state.get("error") and state.get("state") != "running"):
            raise RuntimeError(f"inference stopped: {state.get('error')}")
        if time.monotonic() > deadline:
            raise TimeoutError(f"not reached within {timeout:.0f} s; phase {state.get('phase')}, "
                               f"tick {state.get('current', {}).get('tick')}")
        time.sleep(0.03)


def run(base: str, output: Path, *, bpm: float, edits: int, topics, templates, settle_bars: int, timeout: float):
    output.mkdir(parents=True, exist_ok=True)
    initial = call(base, "/api/inference/state")
    if initial.get("state") != "stopped":
        raise RuntimeError("stop the current performance before running this")
    call(base, "/api/inference/config", {"mode": "steering", "bpm": bpm, "topic": topics[0],
                                          "template_id": templates[0]})
    call(base, "/api/inference/start", {})
    records = []
    try:
        state = wait_for(base, lambda s: s["phase"] == "MODEL", timeout)
        next_bar = int(state["current"]["tick"] // 16) + settle_bars
        for index in range(edits):
            first_half = index < (edits + 1) // 2
            low, high = (1.0, 6.0) if first_half else (9.0, 14.0)
            before = wait_for(base, lambda s: s["phase"] == "MODEL" and s["current"]["tick"] // 16 >= next_bar
                              and low <= s["current"]["tick"] % 16 <= high, timeout)
            payload = {"topic": topics[(index + 1) % len(topics)],
                       "template_id": templates[(index + 1) % len(templates)]}
            sent = time.monotonic()
            accepted = call(base, "/api/inference/steer", payload)
            target = accepted["current"]["requested_version"]
            performed = wait_for(base, lambda s: s["current"]["playing_version"] == target, timeout)
            records.append({
                "edit": index + 1, "half": "first" if first_half else "second", "payload": payload,
                "tick_before_edit": before["current"]["tick"], "target_version": target,
                "tick_when_performed_observed": performed["current"]["tick"],
                "observed_seconds": time.monotonic() - sent,
                "steering_playback_latency_ms": performed["metrics"].get("steering_playback_latency_ms"),
            })
            print(json.dumps(records[-1]), flush=True)
            # Let the new plan play for a while so the next edit starts from a settled window.
            next_bar = int(performed["current"]["tick"] // 16) + settle_bars
    finally:
        call(base, "/api/inference/stop", {})
        deadline = time.monotonic() + 30
        final = call(base, "/api/inference/state")
        while final.get("state") not in ("stopped", "error") and time.monotonic() < deadline:
            time.sleep(0.1)
            final = call(base, "/api/inference/state")
        trace = call(base, "/api/inference/export?format=json")
        for name, value in (("trace", trace), ("final", final),
                            ("edits", {"bpm": bpm, "edits": records, "acoustic_verification": False})):
            (output / f"{name}.json").write_text(json.dumps(value, indent=2) + "\n")
    return {"edits_performed": len(records), "missing_duration_ms": final["metrics"]["missing_duration_ms"],
            "stale_duration_ms": final["metrics"]["stale_duration_ms"],
            "steering_adapter": final["capabilities"].get("steering_adapter"), "output": str(output)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://127.0.0.1:8018")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bpm", type=float, default=60)
    parser.add_argument("--edits", type=int, default=6)
    parser.add_argument("--topics", nargs=2, default=["night market", "robot orchestra"])
    parser.add_argument("--templates", nargs=2,
                        default=["mcflow_4b52a40dac67813b8d53", "mcflow_4fa0466611a111afb52f"])
    parser.add_argument("--settle-bars", type=int, default=3,
                        help="bars to let play between one edit being performed and the next edit")
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args(argv)
    print(json.dumps(run(args.url, args.output, bpm=args.bpm, edits=args.edits, topics=args.topics,
                         templates=args.templates, settle_bars=args.settle_bars, timeout=args.timeout)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
