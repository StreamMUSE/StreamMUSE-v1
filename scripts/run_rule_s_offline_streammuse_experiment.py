#!/usr/bin/env python
"""Run the paired Rule-S experiment through offline and StreamMUSE paths."""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from streammuse.infrastructure.input.midi_file import MidiFileInput


REPO_ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = ("batch_first", "rule_s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare candidate-1 and Rule-S Best-of-N through both the offline "
            "and actual StreamMUSE MIDI-file simulation paths."
        )
    )
    parser.add_argument("--midi-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt-checkpoint", type=Path, required=True)
    parser.add_argument("--continuation-checkpoint", type=Path, required=True)
    parser.add_argument("--candidate-count", type=int, default=5)
    parser.add_argument("--continuation-seeds", default="0,1")
    parser.add_argument("--prompt-seed-base", type=int, default=20260813)
    parser.add_argument("--condition-bpm", type=int, default=120)
    parser.add_argument("--playback-tempo", type=int, default=15)
    parser.add_argument("--ticks-per-beat", type=int, default=4)
    parser.add_argument("--prompt-length-ticks", type=int, default=32)
    parser.add_argument("--generation-interval-ticks", type=int, default=4)
    parser.add_argument("--generation-length-frames", type=int, default=4)
    parser.add_argument("--prompt-context-beats", type=int, default=32)
    parser.add_argument("--history-max-ticks", type=int, default=128)
    parser.add_argument("--max-eval-beats", type=int, default=32)
    parser.add_argument("--final-catchup-grace-ticks", type=int, default=3)
    parser.add_argument("--tail-beats", type=int, default=0)
    parser.add_argument(
        "--trim-leading-rest",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Shift the first retained Melody note to tick 0 in both paths",
    )
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--prompt-temperature", type=float, default=0.8)
    parser.add_argument("--prompt-top-k", type=int, default=50)
    parser.add_argument("--prompt-top-p", type=float, default=0.95)
    parser.add_argument("--prompt-repetition-penalty", type=float, default=1.0)
    parser.add_argument("--rt-temperature", type=float, default=1.1)
    parser.add_argument("--rt-top-k", type=int, default=0)
    parser.add_argument("--rt-top-p", type=float, default=0.95)
    parser.add_argument("--rt-repetition-penalty", type=float, default=1.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )


def parse_seeds(raw: str) -> list[int]:
    seeds = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("--continuation-seeds must contain unique integers")
    return seeds


def require_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    return resolved


def collect_midis(midi_dir: Path) -> list[Path]:
    resolved = midi_dir.expanduser().resolve()
    files = sorted(resolved.glob("*.mid"))
    if not files:
        raise FileNotFoundError(f"no MIDI files found under {resolved}")
    return files


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def request_json(url: str, *, timeout: float = 3.0) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def wait_for_server(
    base_url: str,
    process: subprocess.Popen,
    log_path: Path,
    *,
    timeout_s: float = 300.0,
) -> dict[str, Any]:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if process.poll() is not None:
            tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-40:])
            raise RuntimeError(f"server exited with {process.returncode}:\n{tail}")
        try:
            health = request_json(f"{base_url}/health")
            if health.get("status") == "ok":
                return request_json(
                    f"{base_url}/prompt_continuation/runtime_info",
                    timeout=30.0,
                )
        except (OSError, urllib.error.URLError, json.JSONDecodeError):
            pass
        time.sleep(1.0)
    raise TimeoutError(f"server did not become healthy within {timeout_s}s")


def midi_max_tick(
    path: Path,
    ticks_per_beat: int,
    tail_beats: int,
    max_eval_beats: int,
    trim_leading_rest: bool,
) -> int:
    notes, _resolution, actual_max_tick = MidiFileInput._midi_to_notes(
        str(path),
        beat_div=int(ticks_per_beat),
        min_pitch=0,
        max_pitch=127,
        program=None,
        max_tick=None,
    )
    first_note_tick = min((int(note["tick"]) for note in notes), default=0)
    retained_max_tick = int(actual_max_tick)
    if trim_leading_rest:
        retained_max_tick = max(0, retained_max_tick - first_note_tick)
    requested_max_tick = retained_max_tick + int(tail_beats) * int(ticks_per_beat)
    eval_max_tick = int(max_eval_beats) * int(ticks_per_beat)
    return min(requested_max_tick, eval_max_tick)


def experiment_env(
    args: argparse.Namespace,
    *,
    condition: str,
    prompt_seed: int,
    continuation_seed: int,
    port: int,
) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(args.gpu),
            "LEKAI_DEVICE": "cuda",
            "LEKAI_PROMPT_DEVICE": "cuda",
            "LEKAI_DTYPE": "float16",
            "LEKAI_PROMPT_DTYPE": "float16",
            "LEKAI_PROMPT_CHECKPOINT_PATH": str(args.prompt_checkpoint),
            "LEKAI_CONTINUATION_CHECKPOINT_PATH": str(
                args.continuation_checkpoint
            ),
            "LEKAI_PROMPT_CONTINUATION_REQUIRE_REAL_MODELS": "1",
            "LEKAI_PROMPT_CONTINUATION_ENGINE": "standard",
            "LEKAI_PROMPT_CONTINUATION_RECOVER_LATE_EVENTS": "0",
            "LEKAI_PROMPT_CONTINUATION_REHYDRATE_ACTIVE_NOTES": "0",
            "LEKAI_PROMPT_CONTINUATION_STRICT_REPRESENTATION_LOOP": "1",
            "LEKAI_PROMPT_SELECTION_MODE": condition,
            "LEKAI_PROMPT_BATCH_CANDIDATES": str(args.candidate_count),
            "LEKAI_PROMPT_SEED": str(prompt_seed),
            "LEKAI_PROMPT_BPM": str(args.condition_bpm),
            "LEKAI_DEFAULT_BPM": str(args.condition_bpm),
            "LEKAI_PROMPT_TEMPERATURE": str(args.prompt_temperature),
            "LEKAI_PROMPT_TOP_K": str(args.prompt_top_k),
            "LEKAI_PROMPT_TOP_P": str(args.prompt_top_p),
            "LEKAI_PROMPT_REPETITION_PENALTY": str(
                args.prompt_repetition_penalty
            ),
            "LEKAI_RT_TEMPERATURE": str(args.rt_temperature),
            "LEKAI_RT_TOP_K": str(args.rt_top_k),
            "LEKAI_RT_TOP_P": str(args.rt_top_p),
            "LEKAI_RT_REPETITION_PENALTY": str(args.rt_repetition_penalty),
            "LEKAI_RT_SEED": str(continuation_seed),
            "LEKAI_PROMPT_CONTEXT_BEATS": str(args.prompt_context_beats),
            "LEKAI_HISTORY_MAX_TICKS": str(args.history_max_ticks),
            "LEKAI_SERVER_HOST": "127.0.0.1",
            "LEKAI_SERVER_PORT": str(port),
        }
    )
    return env


def run_logged(
    command: list[str],
    *,
    env: dict[str, str],
    stdout_path: Path,
    stderr_path: Path,
    timeout_s: float,
) -> None:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        command,
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    stdout_path.write_text(result.stdout, encoding="utf-8")
    stderr_path.write_text(result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(command)}\n"
            f"stdout tail:\n{result.stdout[-2000:]}\n"
            f"stderr tail:\n{result.stderr[-2000:]}"
        )


def validate_streammuse_session(
    streammuse_dir: Path,
    *,
    prompt_length_ticks: int,
    evaluation_max_tick: int,
) -> dict[str, Any]:
    sessions = sorted(path for path in streammuse_dir.rglob("session_*") if path.is_dir())
    if not sessions:
        raise RuntimeError(f"no StreamMUSE session found under {streammuse_dir}")
    session_dir = sessions[-1]
    status_path = session_dir / "prompt_continuation_history_status.json"
    raw_history_path = session_dir / "prompt_continuation_raw_history.json"
    if not status_path.is_file():
        raise RuntimeError(f"missing StreamMUSE history status: {status_path}")
    if not raw_history_path.is_file():
        raise RuntimeError(f"missing StreamMUSE raw history: {raw_history_path}")

    history_status = json.loads(status_path.read_text(encoding="utf-8"))
    not_ready = []
    for key in ("prompt_status", "raw_status"):
        status = history_status.get(key)
        if not isinstance(status, dict) or not status.get("is_playback_ready"):
            not_ready.append(
                {
                    "stream": key,
                    "phase": status.get("phase") if isinstance(status, dict) else None,
                    "melody_history_beats": (
                        status.get("melody_history_beats")
                        if isinstance(status, dict)
                        else None
                    ),
                    "accompaniment_history_beats": (
                        status.get("accompaniment_history_beats")
                        if isinstance(status, dict)
                        else None
                    ),
                    "beats_needed_for_playback": (
                        status.get("beats_needed_for_playback")
                        if isinstance(status, dict)
                        else None
                    ),
                }
            )
    if not_ready:
        raise RuntimeError(
            "StreamMUSE continuation did not become playback-ready: "
            + json.dumps(not_ready, sort_keys=True)
        )

    raw_history = json.loads(raw_history_path.read_text(encoding="utf-8"))
    continuation_note_ons = [
        event
        for event in raw_history
        if event.get("type") == "note_on"
        and int(prompt_length_ticks) <= int(event.get("tick", -1))
        < int(evaluation_max_tick)
    ]
    if not continuation_note_ons:
        raise RuntimeError(
            "StreamMUSE became playback-ready but produced no continuation "
            f"note_on in [{prompt_length_ticks}, {evaluation_max_tick})"
        )
    return {
        "session_dir": str(session_dir),
        "history_status": str(status_path),
        "raw_history": str(raw_history_path),
        "continuation_note_on_count": len(continuation_note_ons),
        "continuation_note_on_max_tick": max(
            int(event["tick"]) for event in continuation_note_ons
        ),
        "prompt_status": history_status["prompt_status"],
        "raw_status": history_status["raw_status"],
    }


def run_case(
    args: argparse.Namespace,
    *,
    midi_path: Path,
    piece_index: int,
    condition: str,
    continuation_seed: int,
    run_root: Path,
) -> dict[str, Any]:
    piece_dir = run_root / f"{piece_index:02d}_{midi_path.stem}"
    case_dir = piece_dir / condition / f"seed{continuation_seed}"
    status_path = case_dir / "case_status.json"
    if not args.no_resume and status_path.exists():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("status") == "complete":
            print(f"[skip] {piece_dir.name} {condition} seed={continuation_seed}")
            return status

    case_dir.mkdir(parents=True, exist_ok=True)
    prompt_seed = int(args.prompt_seed_base) + piece_index - 1
    evaluation_max_tick = midi_max_tick(
        midi_path,
        args.ticks_per_beat,
        args.tail_beats,
        args.max_eval_beats,
        args.trim_leading_rest,
    )
    streammuse_run_stop_tick = (
        evaluation_max_tick + args.final_catchup_grace_ticks
    )
    port = free_port()
    env = experiment_env(
        args,
        condition=condition,
        prompt_seed=prompt_seed,
        continuation_seed=continuation_seed,
        port=port,
    )

    offline_dir = case_dir / "offline"
    streammuse_dir = case_dir / "streammuse"
    offline_cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_lekai_prompt_continuation_offline.py"),
        "--midi-file", str(midi_path),
        "--output-dir", str(offline_dir),
        "--prompt-checkpoint", str(args.prompt_checkpoint),
        "--continuation-checkpoint", str(args.continuation_checkpoint),
        "--device", "cuda",
        "--dtype", "float16",
        "--prompt-selection-mode", condition,
        "--prompt-batch-candidates", str(args.candidate_count),
        "--prompt-length-ticks", str(args.prompt_length_ticks),
        "--generation-interval-ticks", str(args.generation_interval_ticks),
        "--bpm", str(args.condition_bpm),
        "--max-tick", str(evaluation_max_tick),
        "--prompt-seed", str(prompt_seed),
        "--prompt-temperature", str(args.prompt_temperature),
        "--prompt-top-k", str(args.prompt_top_k),
        "--prompt-top-p", str(args.prompt_top_p),
        "--prompt-repetition-penalty", str(args.prompt_repetition_penalty),
        "--rt-seed", str(continuation_seed),
        "--rt-temperature", str(args.rt_temperature),
        "--rt-top-k", str(args.rt_top_k),
        "--rt-top-p", str(args.rt_top_p),
        "--rt-repetition-penalty", str(args.rt_repetition_penalty),
    ]
    if args.trim_leading_rest:
        offline_cmd.append("--trim-leading-rest")
    cli_cmd = [
        sys.executable,
        "-m", "streammuse.presentation.cli.cli",
        "--input-mode", "midi_file",
        "--midi-file-path", str(midi_path),
        "--inference-type", "http",
        "--continuation-mode", "prompt_continuation",
        "--prompt-length-ticks", str(args.prompt_length_ticks),
        "--server-url", f"http://127.0.0.1:{port}",
        "--generation-interval-ticks", str(args.generation_interval_ticks),
        "--generation-length-frames", str(args.generation_length_frames),
        "--run-stop-tick", str(streammuse_run_stop_tick),
        "--tempo", str(args.playback_tempo),
        "--model-condition-bpm", str(args.condition_bpm),
        "--output-type", "session",
        "--log-dir", str(streammuse_dir),
    ]
    if args.trim_leading_rest:
        cli_cmd.append("--midi-file-trim-leading-rest")

    if args.dry_run:
        status = {
            "status": "dry_run",
            "midi": str(midi_path),
            "condition": condition,
            "prompt_seed": prompt_seed,
            "continuation_seed": continuation_seed,
            "evaluation_max_tick": evaluation_max_tick,
            "streammuse_run_stop_tick": streammuse_run_stop_tick,
            "offline_command": offline_cmd,
            "streammuse_command": cli_cmd,
        }
        write_json(status_path, status)
        return status

    started = time.perf_counter()
    try:
        offline_reused = (
            not args.no_resume and (offline_dir / "batch_summary.json").is_file()
        )
        if not offline_reused:
            run_logged(
                offline_cmd,
                env=env,
                stdout_path=offline_dir / "stdout.log",
                stderr_path=offline_dir / "stderr.log",
                timeout_s=1800.0,
            )

        server_log_path = streammuse_dir / "server.log"
        server_log_path.parent.mkdir(parents=True, exist_ok=True)
        with server_log_path.open("w", encoding="utf-8") as server_log:
            server = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "streammuse.infrastructure.inference.server_lekai",
                ],
                cwd=str(REPO_ROOT),
                env=env,
                stdout=server_log,
                stderr=subprocess.STDOUT,
                text=True,
            )
            try:
                runtime_info = wait_for_server(
                    f"http://127.0.0.1:{port}", server, server_log_path
                )
                if runtime_info.get("prompt_selection_mode") != condition:
                    raise RuntimeError(
                        "server selection mode mismatch: "
                        f"{runtime_info.get('prompt_selection_mode')} != {condition}"
                    )
                write_json(streammuse_dir / "runtime_info.json", runtime_info)
                wallclock_s = streammuse_run_stop_tick * 60.0 / (
                    args.playback_tempo * args.ticks_per_beat
                )
                run_logged(
                    cli_cmd,
                    env=env,
                    stdout_path=streammuse_dir / "stdout.log",
                    stderr_path=streammuse_dir / "stderr.log",
                    timeout_s=max(600.0, wallclock_s * 2.0 + 300.0),
                )
                prompt_generation_log = request_json(
                    f"http://127.0.0.1:{port}/prompt_continuation/"
                    "prompt_generation_log",
                    timeout=30.0,
                )
                write_json(
                    streammuse_dir / "prompt_generation_log.json",
                    prompt_generation_log,
                )
                streammuse_validation = validate_streammuse_session(
                    streammuse_dir,
                    prompt_length_ticks=args.prompt_length_ticks,
                    evaluation_max_tick=evaluation_max_tick,
                )
                write_json(
                    streammuse_dir / "validation.json",
                    streammuse_validation,
                )
            finally:
                server.terminate()
                try:
                    server.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait(timeout=30)

        status = {
            "status": "complete",
            "midi": str(midi_path),
            "condition": condition,
            "prompt_seed": prompt_seed,
            "continuation_seed": continuation_seed,
            "evaluation_max_tick": evaluation_max_tick,
            "streammuse_run_stop_tick": streammuse_run_stop_tick,
            "elapsed_s": time.perf_counter() - started,
            "offline_reused": offline_reused,
            "offline_dir": str(offline_dir),
            "streammuse_dir": str(streammuse_dir),
            "streammuse_validation": streammuse_validation,
        }
        write_json(status_path, status)
        return status
    except Exception as exc:
        write_json(
            status_path,
            {
                "status": "failed",
                "midi": str(midi_path),
                "condition": condition,
                "prompt_seed": prompt_seed,
                "continuation_seed": continuation_seed,
                "error": repr(exc),
            },
        )
        raise


def main() -> None:
    args = parse_args()
    if args.candidate_count < 2:
        raise ValueError("--candidate-count must be at least 2")
    if args.condition_bpm < 1 or args.playback_tempo < 1:
        raise ValueError("condition and playback tempos must be positive")
    if args.prompt_context_beats < 1 or args.history_max_ticks < 1:
        raise ValueError("context and history limits must be positive")
    if not 1 <= args.final_catchup_grace_ticks < args.generation_interval_ticks:
        raise ValueError(
            "final catch-up grace must be at least one tick and remain below "
            "the generation interval"
        )
    if args.max_eval_beats * args.ticks_per_beat <= args.prompt_length_ticks:
        raise ValueError("evaluation window must extend beyond the Prompt")
    args.prompt_checkpoint = require_file(args.prompt_checkpoint, "prompt checkpoint")
    args.continuation_checkpoint = require_file(
        args.continuation_checkpoint, "continuation checkpoint"
    )
    midi_files = collect_midis(args.midi_dir)
    seeds = parse_seeds(args.continuation_seeds)
    run_root = args.output_dir.expanduser().resolve()
    run_root.mkdir(parents=True, exist_ok=True)

    config = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment": "legacy_human_offline_vs_streammuse_rule_s_best_of_n",
        "midi_files": [str(path) for path in midi_files],
        "conditions": list(CONDITIONS),
        "continuation_seeds": seeds,
        "candidate_count": args.candidate_count,
        "baseline": "candidate 1 from the same Batch=N Prompt generation",
        "rule_s": "highest frozen Rule-S score from the same Batch=N",
        "prompt_checkpoint": str(args.prompt_checkpoint),
        "continuation_checkpoint": str(args.continuation_checkpoint),
        "gpu": str(args.gpu),
        "prompt_seed_base": args.prompt_seed_base,
        "prompt_sampling": {
            "temperature": args.prompt_temperature,
            "top_k": args.prompt_top_k,
            "top_p": args.prompt_top_p,
            "repetition_penalty": args.prompt_repetition_penalty,
        },
        "continuation_sampling": {
            "temperature": args.rt_temperature,
            "top_k": args.rt_top_k,
            "top_p": args.rt_top_p,
            "repetition_penalty": args.rt_repetition_penalty,
        },
        "condition_bpm": args.condition_bpm,
        "playback_tempo": args.playback_tempo,
        "ticks_per_beat": args.ticks_per_beat,
        "prompt_length_ticks": args.prompt_length_ticks,
        "generation_interval_ticks": args.generation_interval_ticks,
        "generation_length_frames": args.generation_length_frames,
        "prompt_context_beats": args.prompt_context_beats,
        "history_max_ticks": args.history_max_ticks,
        "max_eval_beats": args.max_eval_beats,
        "final_catchup_grace_ticks": args.final_catchup_grace_ticks,
        "tail_beats": args.tail_beats,
        "trim_leading_rest": args.trim_leading_rest,
        "dry_run": args.dry_run,
    }
    write_json(run_root / "run_config.json", config)

    statuses = []
    total = len(midi_files) * len(CONDITIONS) * len(seeds)
    completed = 0
    for piece_index, midi_path in enumerate(midi_files, start=1):
        for condition in CONDITIONS:
            for continuation_seed in seeds:
                completed += 1
                print(
                    f"[{completed}/{total}] {midi_path.stem} "
                    f"{condition} seed={continuation_seed}",
                    flush=True,
                )
                statuses.append(
                    run_case(
                        args,
                        midi_path=midi_path,
                        piece_index=piece_index,
                        condition=condition,
                        continuation_seed=continuation_seed,
                        run_root=run_root,
                    )
                )
                write_json(run_root / "run_status.json", statuses)

    write_json(
        run_root / "summary.json",
        {
            "status": "dry_run" if args.dry_run else "complete",
            "case_count": len(statuses),
            "cases": statuses,
        },
    )


if __name__ == "__main__":
    main()
