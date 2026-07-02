from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import requests


CLI_OUTPUT_TYPES = ["audio", "console", "midi_file", "websocket", "composite", "json_log", "session"]
CLI_INFERENCE_TYPES = ["http", "stanley"]
CLI_MODEL_NAMES = ["stanley", "lekai"]
CLI_INFERENCE_MODES = ["sliding_window", "stateful"]
CLI_INJECTION_MODES = ["none", "fixed", "self"]


@dataclass
class RunResult:
    melody_path: str
    output_dir: str
    return_code: int
    duration_s: float
    command: str
    log_file: str
    status: str
    message: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-run streammuse-cli over all melody MIDI files in a folder.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--mel-dir", type=str, required=True, help="Folder containing melody MIDI files")
    parser.add_argument("--output-root", type=str, required=True, help="Root output folder for batch results")
    parser.add_argument("--extensions", type=str, default=".mid,.midi", help="Comma-separated melody file suffixes")
    parser.add_argument("--recursive", action="store_true", help="Recursively scan mel-dir")
    parser.add_argument("--offset", type=int, default=0, help="Start index after sorting files")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of files to run")
    parser.add_argument("--dry-run", action="store_true", help="Print commands only, do not execute")
    parser.add_argument("--fail-fast", action="store_true", help="Stop immediately when one run fails")
    parser.add_argument(
        "--sleep-between-runs-s",
        type=float,
        default=0.0,
        help="Sleep seconds between two runs",
    )

    parser.add_argument("--tempo", type=float, default=120.0)
    parser.add_argument("--ticks-per-beat", type=int, default=4)
    parser.add_argument("--beats-per-bar", type=int, default=4)
    parser.add_argument("--midi-file-delay-ticks", type=int, default=0)

    parser.add_argument("--inference-type", type=str, choices=CLI_INFERENCE_TYPES, default="http")
    parser.add_argument("--model-name", type=str, choices=CLI_MODEL_NAMES, default="lekai")
    parser.add_argument("--inference-mode", type=str, choices=CLI_INFERENCE_MODES, default="sliding_window")
    parser.add_argument(
        "--server-url",
        type=str,
        default="http://127.0.0.1:8000/generate_accompaniment",
        help="Server endpoint for --inference-type http",
    )
    parser.add_argument("--timeout-s", type=float, default=30.0)
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--model-size", type=str, default="0.12B")
    parser.add_argument("--model-max-seq-len-frames", type=int, default=96)
    parser.add_argument("--generation-interval-ticks", type=int, default=4)
    parser.add_argument("--generation-length-frames", type=int, default=16)
    parser.add_argument("--max-ticks", type=int, default=None)
    parser.add_argument(
        "--skip-server-check",
        action="store_true",
        help="Skip calling /health before batch run (http only)",
    )

    parser.add_argument("--output-type", type=str, choices=CLI_OUTPUT_TYPES, default="composite")
    parser.add_argument("--midi-out-port", type=str, default=None)
    parser.add_argument("--inference-log-detail", type=str, choices=["summary", "full"], default="summary")
    parser.add_argument(
        "--session-artifact-tier",
        type=str,
        choices=["normal", "debug"],
        default="normal",
        help="Session artifact tier passed to streammuse-cli; batch sweeps default to normal",
    )

    parser.add_argument("--injection-mode", type=str, choices=CLI_INJECTION_MODES, default="none")
    parser.add_argument("--injection-file", type=str, default=None, help="Used when --injection-mode fixed")
    parser.add_argument(
        "--injection-length",
        type=int,
        default=0,
        help="Injection ticks, must be > 0 when injection-mode is fixed/self",
    )
    parser.add_argument("--inject-acc-file", type=str, default=None, help="Used when --injection-mode fixed")
    parser.add_argument(
        "--inject-acc-dir",
        type=str,
        default=None,
        help="Used when --injection-mode self; maps relative melody path to acc path",
    )

    return parser.parse_args()


def parse_extensions(raw: str) -> set[str]:
    suffixes: set[str] = set()
    for part in raw.split(","):
        value = part.strip().lower()
        if not value:
            continue
        if not value.startswith("."):
            value = f".{value}"
        suffixes.add(value)
    if not suffixes:
        raise ValueError("No valid extensions parsed from --extensions")
    return suffixes


def discover_melody_files(mel_dir: Path, extensions: set[str], recursive: bool) -> list[Path]:
    iterator = mel_dir.rglob("*") if recursive else mel_dir.glob("*")
    files = [p for p in iterator if p.is_file() and p.suffix.lower() in extensions]
    files.sort(key=lambda p: str(p.relative_to(mel_dir)).lower())
    return files


def health_url_from_generate_url(generate_url: str) -> str:
    parsed = urlsplit(generate_url)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"Invalid --server-url: {generate_url}")
    return urlunsplit((parsed.scheme, parsed.netloc, "/health", "", ""))


def validate_args(args: argparse.Namespace, mel_dir: Path) -> None:
    if not mel_dir.exists() or not mel_dir.is_dir():
        raise FileNotFoundError(f"mel-dir not found or not a directory: {mel_dir}")

    if args.offset < 0:
        raise ValueError("--offset must be >= 0")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be > 0")
    if args.sleep_between_runs_s < 0:
        raise ValueError("--sleep-between-runs-s must be >= 0")

    if args.inference_type == "http" and args.model_name == "lekai":
        if int(args.generation_length_frames) % 4 != 0:
            raise ValueError("For Lekai, --generation-length-frames must be a multiple of 4")

    if args.injection_mode == "none":
        if args.injection_length != 0:
            raise ValueError("--injection-length must be 0 when --injection-mode none")
        if args.injection_file:
            raise ValueError("--injection-file is only valid when --injection-mode fixed")
        if args.inject_acc_file:
            raise ValueError("--inject-acc-file is only valid when --injection-mode fixed")
        if args.inject_acc_dir:
            raise ValueError("--inject-acc-dir is only valid when --injection-mode self")
    elif args.injection_mode == "fixed":
        if not args.injection_file:
            raise ValueError("--injection-file is required when --injection-mode fixed")
        if args.injection_length <= 0:
            raise ValueError("--injection-length must be > 0 when --injection-mode fixed")
        if args.inject_acc_dir:
            raise ValueError("--inject-acc-dir is not valid when --injection-mode fixed")
    elif args.injection_mode == "self":
        if args.injection_file:
            raise ValueError("--injection-file is not used when --injection-mode self")
        if args.inject_acc_file:
            raise ValueError("--inject-acc-file is not used when --injection-mode self")
        if args.injection_length <= 0:
            raise ValueError("--injection-length must be > 0 when --injection-mode self")


def build_injection_args(args: argparse.Namespace, mel_dir: Path, melody_file: Path) -> tuple[list[str], str]:
    if args.injection_mode == "none":
        return [], ""

    if args.injection_mode == "fixed":
        injection_file = str(Path(args.injection_file).expanduser().resolve())
        cmd = ["--injection-file", injection_file, "--injection-length", str(int(args.injection_length))]
        message = f"fixed:{injection_file}"
        if args.inject_acc_file:
            inject_acc_file = str(Path(args.inject_acc_file).expanduser().resolve())
            cmd.extend(["--inject-acc-file", inject_acc_file])
            message += f",acc:{inject_acc_file}"
        return cmd, message

    relative_path = melody_file.relative_to(mel_dir)
    cmd = ["--injection-file", str(melody_file), "--injection-length", str(int(args.injection_length))]
    message = f"self:{melody_file}"

    if args.inject_acc_dir:
        inject_acc_path = Path(args.inject_acc_dir).expanduser().resolve() / relative_path
        if inject_acc_path.exists():
            cmd.extend(["--inject-acc-file", str(inject_acc_path)])
            message += f",acc:{inject_acc_path}"
        else:
            message += f",acc_missing:{inject_acc_path}"

    return cmd, message


def build_cli_command(
    args: argparse.Namespace,
    mel_dir: Path,
    melody_file: Path,
    run_dir: Path,
) -> tuple[list[str], str]:
    cmd = [
        "uv",
        "run",
        "streammuse-cli",
        "--input-mode",
        "midi_file",
        "--midi-file-path",
        str(melody_file),
        "--tempo",
        str(float(args.tempo)),
        "--ticks-per-beat",
        str(int(args.ticks_per_beat)),
        "--beats-per-bar",
        str(int(args.beats_per_bar)),
        "--midi-file-delay-ticks",
        str(int(args.midi_file_delay_ticks)),
        "--inference-type",
        str(args.inference_type),
        "--model-name",
        str(args.model_name),
        "--inference-mode",
        str(args.inference_mode),
        "--timeout-s",
        str(float(args.timeout_s)),
        "--model-size",
        str(args.model_size),
        "--model-max-seq-len-frames",
        str(int(args.model_max_seq_len_frames)),
        "--generation-interval-ticks",
        str(int(args.generation_interval_ticks)),
        "--generation-length-frames",
        str(int(args.generation_length_frames)),
        "--output-type",
        str(args.output_type),
        "--inference-log-detail",
        str(args.inference_log_detail),
        "--session-artifact-tier",
        str(args.session_artifact_tier),
    ]

    if args.inference_type == "http":
        cmd.extend(["--server-url", str(args.server_url)])

    if args.checkpoint_path:
        cmd.extend(["--checkpoint-path", str(Path(args.checkpoint_path).expanduser().resolve())])

    if args.midi_out_port:
        cmd.extend(["--midi-out-port", str(args.midi_out_port)])

    if args.max_ticks is not None:
        cmd.extend(["--max-ticks", str(int(args.max_ticks))])

    if args.output_type == "midi_file":
        midi_output = run_dir / f"{melody_file.stem}_combined.mid"
        cmd.extend(["--midi-file-output-path", str(midi_output)])
    else:
        cmd.extend(["--log-dir", str(run_dir / "logs")])

    injection_args, injection_message = build_injection_args(args, mel_dir, melody_file)
    cmd.extend(injection_args)
    return cmd, injection_message


def command_to_text(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def run_one_command(cmd: list[str], cwd: Path, log_path: Path) -> tuple[int, float]:
    start = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"$ {command_to_text(cmd)}\n\n")
        log_file.flush()
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    duration_s = time.perf_counter() - start
    return int(proc.returncode), duration_s


def to_serializable_args(args: argparse.Namespace) -> dict[str, object]:
    data = vars(args).copy()
    return {key: value for key, value in data.items()}


def main() -> int:
    args = parse_args()

    project_root = Path(__file__).resolve().parents[1]
    mel_dir = Path(args.mel_dir).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()

    try:
        validate_args(args, mel_dir)
        extensions = parse_extensions(args.extensions)
    except Exception as exc:
        print(f"[batch] argument error: {exc}")
        return 2

    melody_files = discover_melody_files(mel_dir, extensions, bool(args.recursive))
    if args.offset:
        melody_files = melody_files[args.offset :]
    if args.limit is not None:
        melody_files = melody_files[: args.limit]

    if not melody_files:
        print("[batch] no melody files matched")
        return 1

    if args.inference_type == "http" and not args.skip_server_check and not args.dry_run:
        try:
            health_url = health_url_from_generate_url(args.server_url)
            response = requests.get(health_url, timeout=float(args.timeout_s))
            response.raise_for_status()
            print(f"[batch] server health check ok: {health_url}")
        except Exception as exc:
            print(f"[batch] server health check failed: {exc}")
            return 1

    output_root.mkdir(parents=True, exist_ok=True)

    print(f"[batch] mel_dir={mel_dir}")
    print(f"[batch] output_root={output_root}")
    print(f"[batch] files={len(melody_files)}")
    if args.dry_run:
        print("[batch] dry-run mode enabled")
    elif args.max_ticks is None:
        print(
            "[batch] warning: --max-ticks is not set. "
            "In midi_file mode each run may keep ticking and not auto-exit."
        )

    results: list[RunResult] = []
    started_at = time.time()
    total = len(melody_files)

    for index, melody_file in enumerate(melody_files, start=1):
        rel = melody_file.relative_to(mel_dir)
        run_dir = output_root / rel.parent / rel.stem
        run_dir.mkdir(parents=True, exist_ok=True)
        log_path = run_dir / "batch_runner.log"

        cmd, injection_message = build_cli_command(args, mel_dir, melody_file, run_dir)
        cmd_text = command_to_text(cmd)
        print(f"[batch] ({index}/{total}) {rel}")
        if injection_message:
            print(f"[batch]   injection={injection_message}")

        if args.dry_run:
            print(f"[batch]   cmd={cmd_text}")
            results.append(
                RunResult(
                    melody_path=str(melody_file),
                    output_dir=str(run_dir),
                    return_code=0,
                    duration_s=0.0,
                    command=cmd_text,
                    log_file=str(log_path),
                    status="dry_run",
                )
            )
            continue

        print(f"[batch]   running... log={log_path}")
        return_code, duration_s = run_one_command(cmd, project_root, log_path)
        status = "ok" if return_code == 0 else "failed"
        message = ""
        if return_code != 0:
            message = f"command exited with code {return_code}"
        print(f"[batch]   status={status}, duration_s={duration_s:.2f}, log={log_path}")

        results.append(
            RunResult(
                melody_path=str(melody_file),
                output_dir=str(run_dir),
                return_code=return_code,
                duration_s=duration_s,
                command=cmd_text,
                log_file=str(log_path),
                status=status,
                message=message,
            )
        )

        if return_code != 0 and args.fail_fast:
            print("[batch] fail-fast enabled, stopping")
            break

        if args.sleep_between_runs_s > 0:
            time.sleep(float(args.sleep_between_runs_s))

    finished_at = time.time()
    ok_count = sum(1 for item in results if item.status in {"ok", "dry_run"})
    failed_count = sum(1 for item in results if item.status == "failed")

    summary = {
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_s": finished_at - started_at,
        "config": to_serializable_args(args),
        "mel_dir": str(mel_dir),
        "output_root": str(output_root),
        "processed": len(results),
        "ok": ok_count,
        "failed": failed_count,
        "results": [item.__dict__ for item in results],
    }

    summary_path = output_root / "batch_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[batch] summary={summary_path}")
    print(f"[batch] done: processed={len(results)}, ok={ok_count}, failed={failed_count}")

    if failed_count > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())