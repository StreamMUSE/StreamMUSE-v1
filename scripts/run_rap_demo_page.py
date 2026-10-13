"""Run the demo-page matrix: versions x samples x {fake_realtime, realtime}.

For each sample, each listed version runs in turn (so network drift spreads
over the versions), each in both modes. Before every run the tunnel round trip
is measured (5 x ``/health``); above ``--max-rtt`` seconds (median) the run
waits and measures again. A run whose process fails is retried once; fallbacks
inside a run are results and are never retried. Every run appends one line to
``<out>/runs.jsonl``.

    uv run python scripts/run_rap_demo_page.py --versions v0.0,v1.0 --out output/rap_demo_page_2026-10

Plan: developing-logs/plans/2026-10-12-rap-version-demo-page-plan.md.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

import httpx

ROOT = Path(__file__).resolve().parents[1]
PROJECTS = ROOT.parent

VERSIONS = {
    "v0.0": {"tree": PROJECTS / "StreamMUSE-rap-v0.0", "tag": "rap-v0.0", "url": "http://127.0.0.1:18120", "extra": []},
    "v0.1": {"tree": PROJECTS / "StreamMUSE-rap-v0.1", "tag": "rap-v0.1", "url": "http://127.0.0.1:18220", "extra": []},
    "v1.0": {"tree": ROOT, "tag": "rap-v1.0", "url": "http://127.0.0.1:18020", "extra": ["--rap-protocol", "v3"]},
}
FAKE_TIMEOUTS = ["--rap-render-startup-timeout", "600", "--rap-render-rolling-timeout", "60"]


def _rtt(url: str) -> list[float]:
    values = []
    with httpx.Client(timeout=10.0) as client:
        for _ in range(5):
            started = time.perf_counter()
            try:
                client.get(f"{url}/health").raise_for_status()
                values.append(round(time.perf_counter() - started, 3))
            except httpx.HTTPError:
                values.append(float("inf"))
    return values


def _command(version: str, mode: str, sample: dict, log_dir: Path, bars: int) -> list[str]:
    config = VERSIONS[version]
    venv = config["tree"] / ".venv" / "bin"
    demo = [
        "--scenario", str(ROOT / "demo_page" / "scenarios" / f"{sample['sample']}.json"),
        "--seed", str(sample["seed"]),
        "--max-bars", str(bars),
        "--rap-audio-renderer", "moss_aligned_remote",
        "--rap-audio-transport", "opus",
        "--rap-render-url", config["url"],
        "--audio-output", "wav",
        "--log-dir", str(log_dir),
        "--terminal-detail", "summary",
        "--no-web",
        *config["extra"],
    ]
    if mode == "fake_realtime":
        return [str(venv / "python"), str(ROOT / "scripts" / "rap_fake_realtime.py"), "--", *demo, *FAKE_TIMEOUTS]
    return [str(venv / "streammuse-rap-demo"), *demo]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--versions", required=True, help="comma-separated, from " + ", ".join(VERSIONS))
    parser.add_argument("--modes", default="fake_realtime,realtime")
    parser.add_argument("--samples", default="", help="comma-separated sample ids (default: all)")
    parser.add_argument("--bars", type=int, default=16)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-rtt", type=float, default=2.0)
    parser.add_argument("--warmup", action="store_true", help="first run S01 (4 bars, realtime) per version, unrecorded")
    args = parser.parse_args()
    versions = [v.strip() for v in args.versions.split(",") if v.strip()]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    samples = json.loads((ROOT / "demo_page" / "samples.json").read_text())
    if args.samples:
        wanted = set(args.samples.split(","))
        samples = [s for s in samples if s["sample"] in wanted]
    env = {**os.environ, "PATH": f"/opt/homebrew/bin:{os.environ.get('PATH', '')}"}
    ledger = args.out / "runs.jsonl"
    args.out.mkdir(parents=True, exist_ok=True)

    if args.warmup:
        for version in versions:
            log_dir = args.out / "warmup" / version
            subprocess.run(_command(version, "realtime", samples[0], log_dir, 4), env=env, cwd=ROOT,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=600)

    for sample in samples:
        for version in versions:
            for mode in modes:
                log_dir = args.out / "runs" / version / mode / sample["sample"]
                for attempt in (1, 2):
                    while True:
                        rtt = _rtt(VERSIONS[version]["url"])
                        if statistics.median(rtt) <= args.max_rtt:
                            break
                        print(f"rtt {rtt} too slow; waiting", flush=True)
                        time.sleep(30)
                    before = set(log_dir.glob("*")) if log_dir.exists() else set()
                    started = datetime.now(timezone.utc).isoformat()
                    out_path = log_dir.parent / f"{sample['sample']}.attempt{attempt}.out"
                    log_dir.mkdir(parents=True, exist_ok=True)
                    with out_path.open("w") as output:
                        try:
                            code = subprocess.run(
                                _command(version, mode, sample, log_dir, args.bars),
                                env=env, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT, timeout=900,
                            ).returncode
                        except subprocess.TimeoutExpired:
                            code = "timeout"
                    sessions = sorted(str(p) for p in set(log_dir.glob("*")) - before if p.is_dir())
                    record = {
                        "version": version, "tag": VERSIONS[version]["tag"], "mode": mode, **sample,
                        "attempt": attempt, "exit": code, "session_dirs": sessions, "console": str(out_path),
                        "rtt_seconds": rtt, "started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(),
                    }
                    with ledger.open("a") as handle:
                        handle.write(json.dumps(record) + "\n")
                    print(json.dumps({k: record[k] for k in ("version", "mode", "sample", "attempt", "exit")}), flush=True)
                    if code == 0 and len(sessions) == 1:
                        break


if __name__ == "__main__":
    main()
