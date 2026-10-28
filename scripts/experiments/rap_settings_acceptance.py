"""Exercise independent controls against the running Mac client and H200 models."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
import urllib.request

from rap_interaction_acceptance import Acceptance, require_device_output


def timing(state):
    return state.get("audio", {}).get("timing_evidence") or {}


def policy(state):
    return (timing(state).get("warp") or {}).get("policy")


def early_bar(state):
    tick = state.get("current_tick")
    return isinstance(tick, int) and tick % 16 <= 3 and policy(state) is not None


def post_timed(url, patch):
    started = time.monotonic()
    request = urllib.request.Request(url, data=json.dumps(patch).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=35) as response:
        body = json.load(response)
    return {"elapsed_ms": (time.monotonic() - started) * 1000, "body": body}


def change_policy(test, selected):
    before = test.wait("/api/state", early_bar)
    current = timing(before)["bar"]
    original = before["frozen_bars"][str(current)]
    started = time.monotonic()
    ack = test.request("/api/control/settings/local", {"alignment_policy": selected})
    elapsed = (time.monotonic() - started) * 1000
    assert ack["coordinator_epoch"] == before["coordinator_epoch"], "policy restarted session"
    assert ack["effective_from_bar"] > current, "policy targeted already playing audio"
    played = test.wait("/api/state", lambda state: policy(state) == selected and timing(state)["bar"] >= ack["effective_from_bar"])
    retained = played["frozen_bars"][str(current)]
    assert retained["text"] == original["text"]
    assert retained["timing_evidence"]["warp"] == original["timing_evidence"]["warp"]
    assert played["coordinator_epoch"] == before["coordinator_epoch"]
    return {"policy": selected, "accepted_ms": elapsed, "current_bar_preserved": current,
            "effective_from_bar": ack["effective_from_bar"], "observed_bar": timing(played)["bar"], "ack": ack}


def run(args):
    test = Acceptance(args.url, args.output)
    summary = {"status": "running", "alignment_changes": []}
    try:
        test.stop()
        test.request("/api/control/settings", {"mode": "continuous", "tempo_bpm": 60,
                     "alignment_policy": "gentle_sparse_r3", "chord_loop": "minor_cycle"})
        test.request("/api/control/reset", {})
        test.request("/api/control/start", {})
        test.wait("/api/control/settings", lambda state: state["state"] == "running", timeout=40)
        require_device_output(test.request("/api/control/settings")["audio_output"])
        for selected in ("all_onsets_r3", "all_onsets_stress_r3", "gentle_sparse_r3"):
            summary["alignment_changes"].append(change_policy(test, selected))

        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(post_timed, args.url + "/api/control/settings/generation", {"tempo_bpm": 55})
            time.sleep(0.05)
            remote_pending = not future.done()
            local = post_timed(args.url + "/api/control/settings/local", {"drum_gain": 0.3, "vocal_gain": 0.65})
            remote_still_pending = not future.done()
            remote = future.result()
        summary["overlap"] = {"remote_pending_before_local": remote_pending,
                              "remote_pending_after_local": remote_still_pending,
                              "local_ms": local["elapsed_ms"], "generation_ms": remote["elapsed_ms"],
                              "local_ack": local["body"], "generation_ack": remote["body"]}
        test.wait("/api/state", lambda state: (timing(state).get("timing") or {}).get("bpm") == 55)
        test.request("/api/control/settings/local", {"chord_loop": "major_cycle"})
        test.wait("/api/state", lambda state: (timing(state).get("mix") or {}).get("chord_loop") == "major_cycle")
        options = test.request("/api/control/plan")
        template = next(item for item in options["templates"] if item["template_id"] == "mcflow_4fa0466611a111afb52f")
        test.request("/api/control/plan", {"topic": "quiet trains at sunrise", "template_id": template["template_id"]})
        time.sleep(10)
        state = test.request("/api/state")
        assert state["audio"]["underruns"] == 0
        summary["continuous"] = {"underruns": state["audio"]["underruns"], "bars": state["frozen_bars"]}
        test.stop()
        test.request("/api/control/start", {})
        test.wait("/api/control/settings", lambda state: state["state"] == "running", timeout=35)
        require_device_output(test.request("/api/control/settings")["audio_output"])
        test.stop()

        test.request("/api/control/settings", {"mode": "alternating", "tempo_bpm": 60,
                     "response_bars": 2, "alignment_policy": "gentle_sparse_r3"})
        test.request("/api/control/start", {})
        test.wait("/api/control/settings", lambda state: state["state"] == "running", timeout=35)
        with ThreadPoolExecutor(max_workers=1) as pool:
            turn = pool.submit(test.turn, "live-policy", [2, 4, 5, 7, 8, 10, 12, 14])
            test.wait("/api/control/interaction", lambda state: state["phase"] == "MODEL_PLAY", timeout=65)
            before = test.request("/api/state")
            ack = test.request("/api/control/settings/local", {"alignment_policy": "all_onsets_stress_r3"})
            summary["exchange"] = {"turn": turn.result(), "ack": ack,
                                   "before_policy": policy(before), "before_bar": timing(before).get("bar")}
        after = test.request("/api/state")
        responses = [bar for bar in after["frozen_bars"].values() if bar.get("text")]
        assert len(responses) >= 2
        assert responses[-1]["timing_evidence"]["warp"]["policy"] == "all_onsets_stress_r3"
        summary["exchange"]["responses"] = responses
        summary["status"] = "passed"
    except Exception as error:
        summary.update(status="failed", error=repr(error))
        raise
    finally:
        test.stop()
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        test.evidence.close()
    print(json.dumps({"status": summary["status"], "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8018")
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
