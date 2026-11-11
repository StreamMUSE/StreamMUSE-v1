"""Launch one Mac app exposing both the legacy dashboard and inference page."""

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8018)
    parser.add_argument("--renderer", default="http://127.0.0.1:18721")
    parser.add_argument("--device", default="MacBook Pro Speakers")
    parser.add_argument("--output", type=Path, default=Path("output/inference_20261111/client"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    record = output / "client.pid.json"
    if record.exists():
        previous = json.loads(record.read_text())
        try:
            os.kill(previous["pid"], 0)
        except ProcessLookupError:
            pass
        else:
            raise SystemExit("Recorded client still exists; inspect it before restarting")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", args.port))
    argv = [str(root / ".venv/bin/python"), "-m", "streammuse.presentation.rap_demo.cli",
            "--rap-audio-renderer", "moss_aligned_remote", "--rap-protocol", "v3",
            "--rap-audio-transport", "opus", "--rap-render-url", args.renderer,
            "--tempo", "60", "--audio-output", "composite", "--audio-device", args.device,
            "--max-bars", "0", "--port", str(args.port), "--log-dir", str(output / "sessions"),
            "--terminal-layout", "stream", "--terminal-detail", "summary",
            "--interaction-mode", "continuous", "--response-bars", "2",
            "--rap-warp-policy", "all_onsets_r3", "--drum-gain", "0.5",
            "--vocal-gain", "0.8", "--chord-loop", "off", "--chord-gain", "0"]
    environment = {"PYTHONPATH": str(root / "src"),
                   "STREAMMUSE_RESEARCH_MODEL_URL": "http://127.0.0.1:18711/v1"}
    log = output / "client.log"
    with log.open("ab", buffering=0) as handle:
        process = subprocess.Popen(argv, cwd=root, env={**os.environ, **environment},
                                   stdin=subprocess.DEVNULL, stdout=handle,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    record.write_text(json.dumps({"pid": process.pid, "argv": argv, "cwd": str(root),
                                 "environment": environment, "log": str(log),
                                 "started_at": time.time()}, indent=2) + "\n")
    print(json.dumps({"pid": process.pid, "page": f"http://127.0.0.1:{args.port}/inference",
                      "log": str(log)}))


if __name__ == "__main__":
    main()
