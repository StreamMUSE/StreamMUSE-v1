import os
import sys
import argparse
import subprocess
from pathlib import Path
import shutil
import time
import shlex


def find_uv_command():
    # prefer system `uv`, fallback to `python -m uv`
    if shutil.which("uv"):
        return ["uv", "run"]
    # try module invocation
    return [sys.executable, "-m", "uv", "run"]


def run_one(
    mel_path: Path,
    injection_length: int,
    generation_length: int,
    extra_args: list,
    workdir: Path,
    out_dir: Path,
    generation_interval_ticks: int,
    generation_length_per_request: int,
    url: str = "http://localhost:8988/generate_accompaniment",
    timeout: int = None,
):
    cmd_base = find_uv_command()
    cmd = cmd_base + [
        "app/client.py",
        "--injection-file",
        str(mel_path),
        "--injection-length",
        str(injection_length),
        "--midi-file-input",
        str(mel_path),
        "--generation_length",
        str(generation_length),
        "--server_url",
        url,
        "--generation_interval_ticks",
        str(generation_interval_ticks),
        "--generation_length_per_request",
        str(generation_length_per_request),
    ]

    if extra_args:
        # 将传入的 extra-arg 字符串拆成独立 token 再追加
        cmd += shlex.split(extra_args)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(workdir)

    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"{mel_path.stem}.log"

    print(f"Running: {' '.join(cmd)}")
    print(f"Log -> {log_path}")
    with open(log_path, "wb") as lf:
        proc = subprocess.run(
            cmd,
            cwd=str(workdir),
            env=env,
            stdout=lf,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
    return proc.returncode


def iter_dataset(
    dataset_dir: Path,
    injection_length: int,
    generation_length: int,
    workdir: Path,
    out_root: Path,
    generation_interval_ticks: int,
    generation_length_per_request: int,
    url: str = "http://localhost:8988/generate_accompaniment",
    pattern: str = "*.mid",
    delay_between_runs: float = 1.0,
    extra_args: list = [],
):
    files = sorted(dataset_dir.glob(pattern))
    if not files:
        print(f"No files found in {dataset_dir} matching {pattern}")
        return 1

    out_root.mkdir(parents=True, exist_ok=True)

    failures = []
    for mel in files:
        run_out = out_root / mel.stem
        run_out.mkdir(parents=True, exist_ok=True)
        rc = run_one(
            mel,
            injection_length,
            generation_length,
            extra_args,
            workdir,
            run_out,
            generation_interval_ticks=generation_interval_ticks,
            generation_length_per_request=generation_length_per_request,
            url=url,
        )
        if rc != 0:
            print(f"[!] Run failed for {mel} (returncode {rc})")
            failures.append((mel, rc))
        else:
            print(f"[✓] Completed {mel}")
        time.sleep(delay_between_runs)

    if failures:
        print("\nFailures:")
        for f, rc in failures:
            print(f" - {f}: rc={rc}")
        return 2
    return 0


def main():
    p = argparse.ArgumentParser(
        description="Batch-run StreamMUSE client over a dataset"
    )
    p.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("input/mel"),
        help="Directory with mel .mid files",
    )
    p.add_argument(
        "--injection-length", type=int, default=75, help="Injection length (ticks)"
    )
    p.add_argument(
        "--generation-length", type=int, default=384, help="Generation length (ticks)"
    )
    p.add_argument(
        "--workdir",
        type=Path,
        default=Path.cwd(),
        help="Project root (sets PYTHONPATH)",
    )
    p.add_argument(
        "--out-root",
        type=Path,
        default=Path("experiments/batch_runs"),
        help="Base output/log directory",
    )
    p.add_argument(
        "--pattern", type=str, default="*.mid", help="Glob pattern for files"
    )
    p.add_argument(
        "--delay", type=float, default=0.01, help="Seconds to wait between runs"
    )
    p.add_argument(
        "--generation-interval-ticks",
        type=int,
        default=None,
        help="Generation interval (ticks)",
    )
    p.add_argument(
        "--generation-length-per-request",
        type=int,
        default=None,
        help="Generation length per request (ticks)",
    )
    p.add_argument(
        "--server-url",
        type=str,
        default="http://localhost:8988/generate_accompaniment",
        help="URL of the StreamMUSE server",
    )
    p.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="Extra args to append to client command",
    )
    args = p.parse_args()

    # quick check: server reachable?
    # skip check here; assume user manages server lifecycle

    if (
        args.generation_interval_ticks is None
        or args.generation_length_per_request is None
    ):
        print(
            "Error: --generation-interval-ticks and --generation-length-per-request must be specified."
        )
        sys.exit(1)

    rc = iter_dataset(
        dataset_dir=args.dataset_dir,
        injection_length=args.injection_length,
        generation_length=args.generation_length,
        workdir=args.workdir,
        out_root=args.out_root,
        generation_interval_ticks=args.generation_interval_ticks,
        generation_length_per_request=args.generation_length_per_request,
        url=args.server_url,
        pattern=args.pattern,
        delay_between_runs=args.delay,
        extra_args=args.extra_arg,
    )
    sys.exit(rc)


if __name__ == "__main__":
    main()
