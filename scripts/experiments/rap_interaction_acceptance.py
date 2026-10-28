"""Exercise the running Mac demo against real models and preserve HTTP evidence.

This is an explicit live acceptance tool, not a mocked test or a listening study.
It takes ownership of the selected demo's transport controls for its duration.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path


def require_device_output(status):
    assert (status.get("backend") == "sounddevice" and status.get("audible") is True
            and not status.get("error")), f"Expected active device output, got {status}"


class Acceptance:
    def __init__(self, url: str, output: Path, *, require_audible: bool = True) -> None:
        self.url = url.rstrip("/")
        output.mkdir(parents=True, exist_ok=True)
        self.output = output
        self.require_audible = require_audible
        self.evidence = (output / "http.jsonl").open("a")

    def request(self, path, payload=None):
        started = time.monotonic()
        data = None if payload is None else json.dumps(payload).encode()
        request = urllib.request.Request(self.url + path, data=data, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=15) as response:
                status, body = response.status, json.load(response)
        except urllib.error.HTTPError as error:
            status, body = error.code, json.load(error)
        self.evidence.write(json.dumps({"at": started, "elapsed_ms": (time.monotonic() - started) * 1000,
                                       "path": path, "input": payload, "status": status, "body": body}) + "\n")
        self.evidence.flush()
        if status >= 400:
            raise RuntimeError(f"{path}: {status}: {body}")
        return body

    def wait(self, path, predicate, timeout=65):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            body = self.request(path)
            if predicate(body):
                return body
            time.sleep(0.15)
        raise TimeoutError(f"Timed out waiting for {path}: {body}")

    def stop(self):
        state = self.request("/api/control/settings")
        if state["state"] == "stopped":
            return state
        self.request("/api/control/stop", {})
        return self.wait("/api/control/settings", lambda s: s["state"] == "stopped", timeout=15)

    def turn(self, index, ticks):
        self.wait("/api/control/interaction", lambda s: s["phase"] == "DRUMS_IDLE" and s["state"] == "running")
        armed = self.request("/api/control/interaction/arm", {})
        start, end = armed["capture_start_seconds"], armed["capture_end_seconds"]
        for tick in ticks:
            due = start + tick * (end - start) / 16
            time.sleep(max(0, due - time.monotonic()))
            result = self.request("/api/control/interaction/tap", {
                "server_time_seconds": time.monotonic(), "uncertainty_ms": 0.0,
            })
            if not result["accepted"]:
                raise AssertionError(f"tap {tick} rejected: {result['reason']}")
        if not ticks:
            final = self.wait("/api/control/interaction", lambda s: s["phase"] == "DRUMS_IDLE" and s["last_error"] == "empty_capture")
        else:
            final = self.wait("/api/control/interaction", lambda s: s["phase"] == "DRUMS_IDLE" and s["turn_id"] != armed["turn_id"])
            actual = [slot["tick_in_bar"] for slot in final["flow"]["slots"]]
            expected = sorted(set(min(15, int(tick + 0.499999)) for tick in ticks))
            assert actual == expected, (expected, actual)
        state = self.request("/api/state")
        assert state["audio"]["underruns"] == 0, state["audio"]
        output = self.request("/api/control/settings").get("audio_output", {})
        if self.require_audible:
            require_device_output(output)
        result = {"index": index, "input_ticks": ticks, "capture_bar": final["capture_bar"],
                  "turn_id": final["turn_id"], "delivered": final["delivered_response_bars"],
                  "replays": final["replay_count"], "elapsed_seconds": time.monotonic() - end,
                  "error": final["last_error"], "generation_error": final["generation_error"],
                  "complete": not ticks or final["delivered_response_bars"] == final["response_bars"],
                  "underruns": state["audio"]["underruns"], "audio_output": output}
        print(json.dumps(result), flush=True)
        return result

    def run(self, count):
        patterns = [[0.15, 4, 8, 12], [0.15, 2, 4, 6, 8, 10, 12, 14, 15.8],
                    [1, 3, 5, 7, 9, 11, 13, 15], [0.15, 3, 6, 8, 11, 14],
                    [0.15, 8], [0.15, 2, 3, 5, 7, 8, 10, 13, 15],
                    [2, 4, 7, 10, 12, 15.8], [0.15, 4, 8, 12, 15.8],
                    [0.15, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                    [0.15, 3, 7, 8, 12, 15.8]]
        self.stop()
        self.request("/api/control/reset", {})
        self.request("/api/control/settings", {"mode": "alternating", "tempo_bpm": 60, "response_bars": 2})
        self.request("/api/control/start", {})
        self.wait("/api/control/settings", lambda s: s["state"] == "running", timeout=35)
        results = []
        summary = {"status": "interrupted", "turns": results}
        try:
            results.append(self.turn("empty", []))
            for index in range(count):
                results.append(self.turn(index, patterns[index % len(patterns)]))
            summary = {"status": "passed" if all(item["complete"] for item in results) else "incomplete_turns",
                       "turns": results,
                       "scope": ("real HTTP, real H200 models; "
                                 + ("active Mac device stream" if self.require_audible else "timed/no-device output permitted")
                                 + "; no acoustic listening judgment")}
        except Exception as error:
            summary = {"status": "failed", "error": repr(error), "turns": results}
            raise
        finally:
            self.stop()
            (self.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            self.evidence.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8018")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--turns", type=int, default=10)
    parser.add_argument("--allow-silent", action="store_true", help="Allow timed/null output; not a speaker acceptance test")
    args = parser.parse_args()
    Acceptance(args.url, args.output, require_audible=not args.allow_silent).run(args.turns)


if __name__ == "__main__":
    main()
