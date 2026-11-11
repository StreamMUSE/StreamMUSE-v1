"""Check live tempo edits using a running Mac client and real render server.

This takes ownership of playback, preserves completed callback PCM, and restores
the original tempo while stopped. It does not verify acoustic output.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

from rap_interaction_acceptance import Acceptance, require_device_output


def run(url: str, output: Path) -> None:
    test = Acceptance(url, output)
    original = test.request("/api/control/settings")["values"]
    results = []
    summary = {"status": "interrupted", "edits": results}
    try:
        if test.request("/api/control/settings")["state"] == "stopped":
            test.request("/api/control/start", {})
        test.wait("/api/control/settings", lambda s: s["state"] == "running")
        previous_bar = -1
        for bpm, phase in [(75, "early"), (90, "late"), (60, "early"), (90, "early")]:
            def ready(s):
                tick = s.get("current_tick")
                return (isinstance(tick, int) and tick // 16 > previous_bar
                        and (tick % 16 <= 1 if phase == "early" else tick % 16 >= 13)
                        and str(tick // 16) in s["frozen_bars"])

            before = test.wait("/api/state", ready, timeout=30)
            bar = before["current_tick"] // 16
            frozen = before["frozen_bars"][str(bar)]
            old_timing = before["audio"]["timing_evidence"]["timing"]
            sent = time.monotonic()
            response = test.request("/api/control/settings/generation", {"tempo_bpm": bpm})
            (output / f"edit-{len(results)}-ack.json").write_text(json.dumps(response, indent=2))
            ack = response["groups"]["generation"]
            effective = ack["effective_from_bar"]
            assert effective > bar, (bar, ack)
            played = test.wait("/api/state", lambda s: any(
                e["event_type"] == "bar_playback_completed" and e["bar"] == effective
                for e in s["recent_events"]), timeout=40)
            assert played["frozen_bars"][str(bar)] == frozen, "Current bar was changed"
            assert played["audio"]["underruns"] == 0, played["audio"]
            require_device_output(test.request("/api/control/settings")["audio_output"])
            events = played["recent_events"]
            start = next(e for e in events if e["event_type"] == "bar_playback_started" and e["bar"] == effective)
            end = next(e for e in events if e["event_type"] == "bar_playback_completed" and e["bar"] == effective)
            commit = next(e for e in events if e["event_type"] == "bar_audio_committed" and e["bar"] == effective)
            timing = commit["payload"]["timing_evidence"]["timing"]
            assert timing["bpm"] == bpm, timing
            frames = end["payload"]["absolute_frame"] - start["payload"]["absolute_frame"]
            assert frames == timing["end_frame"] - timing["start_frame"], (frames, timing)
            result = {"bpm": bpm, "phase": phase, "request_bar": bar,
                      "request_tick_in_bar": before["current_tick"] % 16,
                      "effective_bar": effective, "bars_ahead": effective - bar,
                      "ack": ack, "playback_after_seconds": start["monotonic_ns"] / 1e9 - sent,
                      "old_timing": old_timing, "new_timing": timing,
                      "current_bar_unchanged": True, "callback_frames": frames,
                      "source": played["frozen_bars"][str(effective)]["source"],
                      "underruns": played["audio"]["underruns"]}
            results.append(result)
            print(json.dumps(result), flush=True)
            previous_bar = effective
        summary["status"] = "passed"
    except Exception as error:
        summary.update(status="failed", error=repr(error))
        raise
    finally:
        test.stop()
        state = test.request("/api/state")
        (output / "final-state.json").write_text(json.dumps(state, indent=2))
        recording = state["audio"].get("recording_path")
        if recording and Path(recording).is_file():
            shutil.copy2(recording, output / "tempo_changes_mix.wav")
        settings = test.request("/api/control/settings")
        if settings["values"]["tempo_bpm"] != original["tempo_bpm"]:
            test.request("/api/control/settings/generation", {"tempo_bpm": original["tempo_bpm"]})
        (output / "summary.json").write_text(json.dumps(summary, indent=2))
        test.evidence.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8018")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.url, args.output)
