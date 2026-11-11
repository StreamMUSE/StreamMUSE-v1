"""Exercise real Mac/H200 cycles and save their observable trace, without resets."""

import argparse
import json
from pathlib import Path
import time
import urllib.error
import urllib.request
import uuid


def call(base, path, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(base + path, data=data,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=15) as response:
        return json.load(response)


def run(base, output, cycles, timeout):
    output.mkdir(parents=True, exist_ok=True)
    initial = call(base, "/api/inference/state")
    if initial.get("state") not in ("stopped", "STOPPED"):
        raise RuntimeError("Stop the existing performance before running acceptance")
    owner = "acceptance-" + uuid.uuid4().hex[:12]
    call(base, "/api/inference/config", {"mode": "cycle", "bpm": 60,
                                          "user_beats": 4, "model_beats": 8})
    call(base, "/api/inference/start", {"owner_id": owner})
    started = time.monotonic()
    last_heartbeat = 0.0
    transitions, samples, seen, tap_counts = [], [], set(), {}
    completed = 0
    previous = None
    user_started = started
    try:
        while time.monotonic() - started < timeout:
            now = time.monotonic()
            if now - last_heartbeat >= 1:
                call(base, "/api/inference/heartbeat", {"owner_id": owner})
                last_heartbeat = now
            state = call(base, "/api/inference/state")
            phase = state.get("phase")
            cycle = state.get("cycle_id")
            key = (phase, cycle)
            if key != previous:
                transitions.append({"elapsed_s": now - started, "phase": phase, "cycle_id": cycle})
                if phase == "USER":
                    user_started = now
                    if previous and previous[0] == "MODEL":
                        completed += 1
                previous = key
            if not samples or now - started - samples[-1]["elapsed_s"] >= 0.2:
                samples.append({"elapsed_s": now - started, "state": {
                    name: state.get(name) for name in ("session_id", "epoch", "state", "phase",
                                                       "cycle_id", "current", "metrics", "error")}})
            if state.get("error"):
                raise RuntimeError(str(state["error"]))
            if completed >= cycles:
                break
            if phase == "USER":
                elapsed_ms = state.get("phase_elapsed_ms", (now - user_started) * 1000)
                for tick in (2, 3, 4, 6, 8, 10, 12, 14):
                    event = (cycle, tick)
                    if tick * 250 <= elapsed_ms and event not in seen:
                        call(base, "/api/inference/tap", {
                            "owner_id": owner, "session_id": state["session_id"],
                            "epoch": state["epoch"], "cycle_id": cycle,
                            "event_id": f"{owner}-{cycle}-{tick}",
                            "phase_elapsed_ms": tick * 250,
                        })
                        seen.add(event)
                        tap_counts[str(cycle)] = tap_counts.get(str(cycle), 0) + 1
            time.sleep(0.04)
        if completed < cycles:
            raise RuntimeError(f"Only {completed}/{cycles} exchanges completed before timeout")
    finally:
        call(base, "/api/inference/stop", {})
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            final = call(base, "/api/inference/state")
            if final.get("state") in ("stopped", "STOPPED"):
                break
            time.sleep(0.1)
        trace = call(base, "/api/inference/export?format=json")
        for name, value in (("samples", samples), ("trace", trace), ("final", final),
                            ("summary", {"completed_cycles": completed, "transitions": transitions,
                                         "tap_counts": tap_counts, "acoustic_verification": False})):
            (output / f"{name}.json").write_text(json.dumps(value, indent=2) + "\n")
    return {"completed_cycles": completed, "output": str(output)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8018")
    parser.add_argument("--output", type=Path, default=Path("output/inference_20261111/acceptance"))
    parser.add_argument("--cycles", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()
    print(json.dumps(run(args.url, args.output, args.cycles, args.timeout)))


if __name__ == "__main__":
    main()
