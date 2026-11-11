"""Start the research demo beside the existing Mac demo, without stopping it."""

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=Path("output/rap_demo_20261104/client.pid.json"))
    parser.add_argument("--output", type=Path, default=Path("output/research_day_20261105/client"))
    parser.add_argument("--port", type=int, default=8019)
    parser.add_argument("--renderer", default="http://127.0.0.1:18721")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    record_path = output / "client.pid.json"
    if record_path.exists():
        raise SystemExit("Existing launch record; inspect it before launching another client")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", args.port))
    baseline = json.loads(args.baseline.read_text())
    argv = list(baseline["argv"])
    argv[0] = str(root / ".venv/bin/python")
    for option, value in {
        "--port": str(args.port), "--rap-render-url": args.renderer,
        "--tempo": "60", "--log-dir": str(output / "sessions"),
    }.items():
        argv[argv.index(option) + 1] = value
    environment = {**baseline["environment"], "PYTHONPATH": str(root / "src"),
                   "STREAMMUSE_RESEARCH_MODEL_URL": "http://127.0.0.1:18711/v1"}
    log = output / "client.log"
    with log.open("ab", buffering=0) as handle:
        process = subprocess.Popen(argv, cwd=root, env={**os.environ, **environment},
                                   stdin=subprocess.DEVNULL, stdout=handle,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    record_path.write_text(json.dumps({
        "pid": process.pid, "argv": argv, "cwd": str(root), "environment": environment,
        "log": str(log), "started_at": time.time(),
        "ssh_control_socket": "/tmp/rap-research-20261105.sock",
    }, indent=2) + "\n")
    print(json.dumps({"pid": process.pid, "url": f"http://127.0.0.1:{args.port}/research", "log": str(log)}))


if __name__ == "__main__":
    main()
