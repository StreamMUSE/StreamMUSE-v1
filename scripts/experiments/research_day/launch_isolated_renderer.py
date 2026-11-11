"""Launch an isolated renderer using an existing optimized deployment manifest.

Does not stop or modify the baseline process, model servers, or model assets.
Run this script on H200 after syncing the new source tree.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import uuid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18721)
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text())
    root = args.root.resolve()
    repo = root / "repo"
    record_path = root / "render.pid.json"
    if record_path.exists():
        raise SystemExit("Existing launch record: inspect it before starting another process")
    if not (repo / "src/streammuse/presentation/rap_render_server.py").is_file():
        raise SystemExit("Sync the source tree before launching")
    if not 1024 <= args.port <= 65535:
        raise SystemExit("Invalid nonprivileged port")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", args.port))
    argv = list(baseline["argv"])
    if "streammuse.presentation.rap_render_server" not in argv:
        raise SystemExit("Baseline is not a rap renderer")
    for option, value in {
        "--port": str(args.port),
        "--artifact-root": str(root / "render_artifacts"),
    }.items():
        argv[argv.index(option) + 1] = value
    environment = dict(baseline["environment"])
    old_repo = baseline["cwd"]
    environment["PYTHONPATH"] = environment["PYTHONPATH"].replace(old_repo, str(repo))
    environment["STREAMMUSE_DEPLOYMENT_ROOT"] = str(root)
    environment["STREAMMUSE_DEPLOYMENT_TOKEN"] = uuid.uuid4().hex
    root.mkdir(parents=True, exist_ok=True)
    log = root / "render.log"
    with log.open("ab", buffering=0) as output:
        process = subprocess.Popen(
            argv, cwd=repo, env={**os.environ, **environment},
            stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    record = {
        "service": "research-render", "pid": process.pid, "argv": argv,
        "environment": environment, "cwd": str(repo), "port": args.port,
        "log": str(log), "baseline": str(args.baseline), "started_at": time.time(),
    }
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"pid": process.pid, "port": args.port, "log": str(log)}))


if __name__ == "__main__":
    main()
