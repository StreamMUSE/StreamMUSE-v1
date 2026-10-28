"""Live delayed-capture acceptance against the Mac GUI and H200 models."""

import argparse
import json
from pathlib import Path

from rap_interaction_acceptance import Acceptance, require_device_output


def run(args):
    test = Acceptance(args.url, args.output)
    results = []
    try:
        for index, policy in enumerate(("gentle_sparse_r3", "all_onsets_r3", "all_onsets_stress_r3")):
            test.stop()
            test.request("/api/control/settings", {
                "mode": "alternating", "tempo_bpm": 60, "response_bars": 2,
                "alignment_policy": policy, "chord_loop": "minor_cycle",
            })
            test.request("/api/control/start", {})
            test.wait("/api/control/settings", lambda s: s["state"] == "running", timeout=35)
            ticks = [2, 4, 5, 7, 8, 10, 12, 14] if index != 1 else [4, 6, 8, 9, 10, 12, 14, 15]
            result = test.turn(index, ticks)
            state = test.request("/api/state")
            bars = [bar for bar in state["frozen_bars"].values() if bar.get("text")]
            assert len(bars) >= 2, "Expected two completed model responses"
            response = bars[-2:]
            assert all(bar["flow"]["slots"][0]["tick_in_bar"] == ticks[0] for bar in response)
            assert all(bar["source"] == "moss_lyric_stream" for bar in response), response
            result.update(policy=policy, responses=response)
            results.append(result)
            (args.output / f"{index}_{policy}_state.json").write_text(json.dumps(state, indent=2))
            test.stop()
            # Starting again without Reset is an existing workflow we preserve.
            test.request("/api/control/start", {})
            test.wait("/api/control/settings", lambda s: s["state"] == "running", timeout=35)
            require_device_output(test.request("/api/control/settings")["audio_output"])
            test.stop()
        (args.output / "summary.json").write_text(json.dumps({"status": "passed", "turns": results}, indent=2))
    finally:
        test.stop()
        test.evidence.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8018")
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
