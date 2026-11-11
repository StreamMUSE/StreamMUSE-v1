"""Live steering, restart, and cross-page ownership acceptance against one app."""

import argparse
import json
from pathlib import Path
import time
import urllib.error

from live_acceptance import call


def wait_for(base, predicate, timeout=120, path="/api/inference/state"):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = call(base, path)
        if predicate(state):
            return state
        if state.get("error"):
            raise RuntimeError(state["error"])
        time.sleep(0.1)
    raise TimeoutError(f"Condition not reached: {json.dumps(state)[:1500]}")


def run(base, output):
    output.mkdir(parents=True, exist_ok=True)
    initial = call(base, "/api/inference/state")
    assert initial["state"] == "stopped"
    session = initial["session_id"]
    evidence = {}
    call(base, "/api/inference/config", {"mode": "steering", "bpm": 60,
        "topic": "night market", "template_id": "mcflow_4b52a40dac67813b8d53"})
    call(base, "/api/inference/start", {})
    try:
        before = wait_for(base, lambda s: s["phase"] == "MODEL" and
                          2 <= s["current"]["tick"] % 16 <= 8)
        version = before["current"]["playing_version"]
        accepted = call(base, "/api/inference/steer", {
            "topic": "robot orchestra", "template_id": "mcflow_4fa0466611a111afb52f"})
        target = accepted["current"]["requested_version"]
        assert target > version
        assert accepted["current"]["playing_version"] == version
        changed = wait_for(base, lambda s: s["current"]["playing_version"] == target)
        assert changed["session_id"] == session
        evidence["steering"] = {"before": before, "accepted": accepted, "performed": changed}
        try:
            call(base, "/api/control/plan", {"topic": "must not mutate legacy"})
        except urllib.error.HTTPError as error:
            assert error.code == 409
        else:
            raise AssertionError("Legacy mode-specific write was accepted during inference")
        call(base, "/api/control/stop", {})
        stopped = wait_for(base, lambda s: s["state"] == "stopped", timeout=10)
        epoch = stopped["epoch"]
        call(base, "/api/inference/start", {})
        restarted = wait_for(base, lambda s: s["phase"] == "MODEL")
        assert restarted["epoch"] > epoch
        assert restarted["config"]["topic"] == "robot orchestra"
        newest = restarted["requests"][-1]
        assert newest["context"]["history"], "Stop/Start lost performed lyrics"
        evidence["restart"] = restarted
    finally:
        try:
            call(base, "/api/inference/stop", {})
            wait_for(base, lambda s: s["state"] == "stopped", timeout=10)
        finally:
            (output / "inference-trace.json").write_text(json.dumps(
                call(base, "/api/inference/export?format=json"), indent=2) + "\n")

    call(base, "/api/control/start", {})
    try:
        legacy = wait_for(base, lambda s: s.get("audio", {}).get("current_bar", -1) is not None
                          and s.get("audio", {}).get("current_bar", -1) >= 1
                          and s.get("audio", {}).get("state") == "running",
                          path="/api/state")
        assert legacy["session_id"] == session
        monitor = call(base, "/api/inference/state")
        assert monitor["legacy_active"] is True
        try:
            call(base, "/api/inference/start", {})
        except urllib.error.HTTPError as error:
            assert error.code == 409
        else:
            raise AssertionError("Competing inference playback was accepted")
        evidence["legacy"] = {"audio": legacy["audio"], "fallbacks": legacy["fallbacks"],
                               "monitor": monitor}
    finally:
        call(base, "/api/inference/stop", {})
        wait_for(base, lambda s: s["state"] == "stopped", timeout=10)
    call(base, "/api/control/reset", {})
    final = call(base, "/api/inference/state")
    assert final["session_id"] == session
    assert final["state"] == "stopped"
    evidence["final"] = final
    (output / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    return {"session_id": session, "steering_latency_ms": changed["metrics"].get("steering_playback_latency_ms"),
            "restart_without_reset": True, "cross_page_stop_both_directions": True,
            "legacy_mode_retested": True, "output": str(output)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8018")
    parser.add_argument("--output", type=Path, default=Path("output/inference_20261111/steering"))
    args = parser.parse_args()
    print(json.dumps(run(args.url, args.output)))
