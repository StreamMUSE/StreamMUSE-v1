from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import queue
import socket
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import mido
import requests


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = Path(
    "/data/home/bowenzheng/mbzuai-projects/models/ModelLekai/"
    "epoch_4_1104_1204/model.safetensors"
)
CONDITION_BPM = 120
TICKS_PER_BEAT = 4
LATE_SCHEDULE_POLICIES = {
    "clamped_partial_note",
    "clamped_partial_note_off",
    "clamped_open_note",
    "dropped_past_note",
    "dropped_past_placeholder",
    "late_isolated_note_off",
    "late_placeholder",
    "forced_note_off",
}


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    root: Path
    expected_size: int

    @property
    def npz_dir(self) -> Path:
        return self.root / "npz"

    @property
    def mel_dir(self) -> Path:
        return self.root / "mel"


@dataclass(frozen=True)
class SampleSpec:
    dataset: str
    stem: str
    npz_path: Path
    melody_path: Path
    melody_last_beat: int
    max_ticks: int

    @property
    def sample_id(self) -> str:
        return f"{self.dataset}/{self.stem}"

    @property
    def artifact_id(self) -> str:
        return f"{self.dataset}__{self.stem}"


@dataclass(frozen=True)
class Comparison:
    actual_count: int
    offline_count: int
    matched: int
    union_count: int
    match_rate: float
    only_in_actual: list[tuple[int, int]]
    only_in_offline: list[tuple[int, int]]

    @property
    def consistent(self) -> bool:
        return not self.only_in_actual and not self.only_in_offline


@dataclass
class ServerProcess:
    gpu: int
    port: int
    process: subprocess.Popen[str]
    log_handle: Any
    log_path: Path
    runtime_info: dict[str, Any]

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run artifact-level offline vs MIDI-file realtime consistency over every "
            "inputs_lekai and old_input sample without pytest."
        )
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--tempo", type=int, default=120)
    parser.add_argument("--tail-beats", type=int, default=24)
    parser.add_argument(
        "--samples",
        default=None,
        help="Optional comma-separated sample ids such as inputs_lekai/5,old_input/nyan_cat",
    )
    parser.add_argument("--prompt-context-beats", type=int, default=None)
    parser.add_argument("--history-max-ticks", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Recompute comparisons from an existing --output-dir without running inference",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp_path.replace(path)


def tail(path: Path, lines: int = 40) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-lines:])
    except OSError:
        return "(log unavailable)"


def melody_last_beat(path: Path) -> int:
    midi = mido.MidiFile(str(path))
    last_tick = 0
    for track in midi.tracks:
        absolute_tick = 0
        for message in track:
            absolute_tick += int(message.time)
            if message.type == "note_on" and int(message.velocity) > 0:
                last_tick = max(last_tick, absolute_tick)
    return last_tick // int(midi.ticks_per_beat)


def discover_samples(dataset: DatasetSpec, tail_beats: int) -> list[SampleSpec]:
    npz_by_stem = {path.stem: path.resolve() for path in dataset.npz_dir.glob("*.npz")}
    midi_by_stem = {path.stem: path.resolve() for path in dataset.mel_dir.glob("*.mid")}
    if set(npz_by_stem) != set(midi_by_stem):
        raise RuntimeError(
            f"{dataset.name} NPZ/MIDI stems differ: "
            f"npz_only={sorted(set(npz_by_stem) - set(midi_by_stem))}, "
            f"midi_only={sorted(set(midi_by_stem) - set(npz_by_stem))}"
        )
    if len(npz_by_stem) != dataset.expected_size:
        raise RuntimeError(
            f"{dataset.name} expected {dataset.expected_size} samples, found {len(npz_by_stem)}"
        )
    samples = []
    for stem in sorted(npz_by_stem):
        last_beat = melody_last_beat(midi_by_stem[stem])
        samples.append(
            SampleSpec(
                dataset=dataset.name,
                stem=stem,
                npz_path=npz_by_stem[stem],
                melody_path=midi_by_stem[stem],
                melody_last_beat=last_beat,
                max_ticks=(last_beat + tail_beats) * TICKS_PER_BEAT,
            )
        )
    return samples


def note_intervals(path: Path) -> tuple[list[tuple[int, int, int]], int]:
    midi = mido.MidiFile(str(path))
    intervals: list[tuple[int, int, int]] = []
    for track in midi.tracks:
        name = ""
        absolute_tick = 0
        open_notes: dict[int, list[int]] = {}
        for message in track:
            absolute_tick += int(message.time)
            if message.type == "track_name":
                name = str(getattr(message, "name", ""))
                continue
            if "Accompaniment" not in name and "Part1" not in name:
                continue
            if message.type == "note_on" and int(message.velocity) > 0:
                open_notes.setdefault(int(message.note), []).append(absolute_tick)
            elif message.type == "note_off" or (
                message.type == "note_on" and int(message.velocity) == 0
            ):
                starts = open_notes.get(int(message.note))
                if starts:
                    intervals.append((starts.pop(0), absolute_tick, int(message.note)))
        for pitch, starts in open_notes.items():
            intervals.extend((start, absolute_tick, pitch) for start in starts)
    return intervals, int(midi.ticks_per_beat)


def active_cells(path: Path, max_beat: int) -> set[tuple[int, int]]:
    intervals, ticks_per_beat = note_intervals(path)
    result: set[tuple[int, int]] = set()
    max_step = max_beat * TICKS_PER_BEAT
    for start, end, pitch in intervals:
        start_step = (start * TICKS_PER_BEAT) // ticks_per_beat
        end_step = max(
            start_step + 1,
            (end * TICKS_PER_BEAT + ticks_per_beat - 1) // ticks_per_beat,
        )
        for step in range(start_step, min(end_step, max_step)):
            result.add((step, pitch))
    return result


def compare_midi(actual: Path, offline: Path, max_beat: int) -> Comparison:
    actual_cells = active_cells(actual, max_beat)
    offline_cells = active_cells(offline, max_beat)
    matched = len(actual_cells & offline_cells)
    union = len(actual_cells | offline_cells)
    return Comparison(
        actual_count=len(actual_cells),
        offline_count=len(offline_cells),
        matched=matched,
        union_count=union,
        match_rate=100.0 if union == 0 else round(matched / union * 100.0, 4),
        only_in_actual=sorted(actual_cells - offline_cells),
        only_in_offline=sorted(offline_cells - actual_cells),
    )


def subprocess_env(gpu: int) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONUNBUFFERED"] = "1"
    return env


def run_offline_dataset(
    dataset: DatasetSpec,
    checkpoint: Path,
    output_root: Path,
    gpu: int,
    resume: bool,
) -> Path:
    output_dir = output_root / "offline" / dataset.name
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = list(output_dir.glob("*_generated.mid"))
    if resume and len(existing) == dataset.expected_size:
        return output_dir

    log_path = output_root / "logs" / f"offline_{dataset.name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_lekai_offline.py"),
        "--checkpoint",
        str(checkpoint),
        "--npz-dir",
        str(dataset.npz_dir),
        "--output-dir",
        str(output_dir),
        "--device",
        "cuda",
        "--dtype",
        "auto",
        "--condition-idx",
        "all",
        "--expected-dataset-size",
        str(dataset.expected_size),
        "--source-midi-dir",
        str(dataset.mel_dir),
        "--require-source-midi",
        "--temperature",
        "0.0",
        "--top-k",
        "1",
        "--top-p",
        "0.0",
        "--repetition-penalty",
        "1.2",
        "--gt-prefix-beats",
        "0",
        "--bpm",
        str(CONDITION_BPM),
        "--seed",
        "0",
    ]
    with log_path.open("w") as log_handle:
        result = subprocess.run(
            command,
            cwd=str(REPO_ROOT),
            env=subprocess_env(gpu),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=3600,
        )
    if result.returncode != 0:
        raise RuntimeError(
            f"offline generation failed for {dataset.name} on GPU {gpu}:\n{tail(log_path)}"
        )
    generated = list(output_dir.glob("*_generated.mid"))
    if len(generated) != dataset.expected_size:
        raise RuntimeError(
            f"offline {dataset.name} produced {len(generated)} files, "
            f"expected {dataset.expected_size}"
        )
    return output_dir


def offline_path(sample: SampleSpec, output_root: Path) -> Path:
    matches = sorted(
        (output_root / "offline" / sample.dataset).glob(f"*_{sample.stem}_generated.mid")
    )
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one offline MIDI for {sample.sample_id}, found {len(matches)}: {matches}"
        )
    return matches[0]


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def wait_for_server(server: ServerProcess, timeout_seconds: float = 180.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if server.process.poll() is not None:
            raise RuntimeError(
                f"server GPU {server.gpu} exited with {server.process.returncode}:\n"
                f"{tail(server.log_path)}"
            )
        try:
            response = requests.get(f"{server.base_url}/health", timeout=2)
            if response.ok and response.json().get("status") == "ok":
                runtime = requests.get(f"{server.base_url}/runtime_info", timeout=10)
                runtime.raise_for_status()
                return dict(runtime.json())
        except requests.RequestException:
            pass
        time.sleep(1.0)
    raise RuntimeError(f"server GPU {server.gpu} did not become healthy:\n{tail(server.log_path)}")


def start_server(
    *,
    gpu: int,
    checkpoint: Path,
    output_root: Path,
    prompt_context_beats: int | None,
    history_max_ticks: int | None,
) -> ServerProcess:
    port = free_port()
    server_dir = output_root / "servers" / f"gpu_{gpu}"
    server_dir.mkdir(parents=True, exist_ok=True)
    log_path = server_dir / "server.log"
    log_handle = log_path.open("w")
    env = subprocess_env(gpu)
    env.update(
        {
            "LEKAI_CHECKPOINT_PATH": str(checkpoint),
            "LEKAI_SERVER_HOST": "127.0.0.1",
            "LEKAI_SERVER_PORT": str(port),
            "LEKAI_DEVICE": "cuda",
            "LEKAI_DTYPE": "auto",
            "LEKAI_RT_TEMPERATURE": "0.0",
            "LEKAI_RT_TOP_K": "1",
            "LEKAI_RT_TOP_P": "0.0",
            "LEKAI_RT_REPETITION_PENALTY": "1.2",
            "LEKAI_RT_SEED": "0",
            "LEKAI_DEFAULT_BPM": str(CONDITION_BPM),
            "LEKAI_RT_LOG_DIR": str(server_dir / "generation_logs"),
        }
    )
    if prompt_context_beats is not None:
        env["LEKAI_PROMPT_CONTEXT_BEATS"] = str(prompt_context_beats)
    if history_max_ticks is not None:
        env["LEKAI_HISTORY_MAX_TICKS"] = str(history_max_ticks)

    process = subprocess.Popen(
        [sys.executable, "-m", "streammuse.infrastructure.inference.server_lekai"],
        cwd=str(REPO_ROOT),
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    server = ServerProcess(
        gpu=gpu,
        port=port,
        process=process,
        log_handle=log_handle,
        log_path=log_path,
        runtime_info={},
    )
    try:
        server.runtime_info = wait_for_server(server)
        write_json(server_dir / "runtime_info.json", server.runtime_info)
        return server
    except Exception:
        stop_server(server)
        raise


def stop_server(server: ServerProcess) -> None:
    if server.process.poll() is None:
        server.process.terminate()
        try:
            server.process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            server.process.kill()
            server.process.wait(timeout=20)
    server.log_handle.close()


def count_dropped_requests(session_dir: Path) -> int:
    payload = json.loads((session_dir / "inferences.json").read_text())
    ticks = [int(row["request_data"]["generation_start_tick"]) for row in payload]
    return sum(max(0, (current - previous) // TICKS_PER_BEAT - 1) for previous, current in zip(ticks, ticks[1:]))


def late_schedule_rows(session_dir: Path) -> list[dict[str, Any]]:
    trace_path = session_dir / "model_schedule_trace.jsonl"
    if not trace_path.is_file():
        return []
    rows = []
    for line_number, line in enumerate(trace_path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("policy") in LATE_SCHEDULE_POLICIES:
            rows.append({"line_number": line_number, **row})
    return rows


def latest_session(log_dir: Path) -> Path:
    sessions = sorted(path for path in log_dir.rglob("session_*") if path.is_dir())
    if not sessions:
        raise RuntimeError(f"no session directory produced under {log_dir}")
    return sessions[-1]


def comparison_payload(comparison: Comparison) -> dict[str, Any]:
    payload = asdict(comparison)
    payload["consistent"] = comparison.consistent
    payload["only_in_actual"] = [list(cell) for cell in comparison.only_in_actual[:100]]
    payload["only_in_offline"] = [list(cell) for cell in comparison.only_in_offline[:100]]
    payload["difference_list_truncated"] = (
        len(comparison.only_in_actual) > 100 or len(comparison.only_in_offline) > 100
    )
    return payload


def run_realtime_sample(
    sample: SampleSpec,
    server: ServerProcess,
    checkpoint: Path,
    checkpoint_sha256: str,
    output_root: Path,
    tempo: int,
) -> dict[str, Any]:
    requests.post(f"{server.base_url}/clear_history", timeout=30).raise_for_status()
    log_dir = output_root / "realtime" / sample.artifact_id / f"tempo_{tempo}"
    log_dir.mkdir(parents=True, exist_ok=True)
    cli_log = log_dir / "cli.log"
    command = [
        sys.executable,
        "-m",
        "streammuse.presentation.cli.cli",
        "--input-mode",
        "midi_file",
        "--midi-file-path",
        str(sample.melody_path),
        "--model-name",
        "lekai",
        "--inference-type",
        "http",
        "--server-url",
        f"{server.base_url}/generate_accompaniment",
        "--generation-interval-ticks",
        "4",
        "--generation-length-frames",
        "4",
        "--max-ticks",
        str(sample.max_ticks),
        "--tempo",
        str(tempo),
        "--model-condition-bpm",
        str(CONDITION_BPM),
        "--output-type",
        "session",
        "--session-artifact-tier",
        "debug",
        "--log-dir",
        str(log_dir),
    ]
    timeout_seconds = max(300.0, (sample.max_ticks / TICKS_PER_BEAT / tempo) * 60 * 2 + 180)
    started = time.monotonic()
    with cli_log.open("w") as log_handle:
        result = subprocess.run(
            command,
            cwd=str(REPO_ROOT),
            env=subprocess_env(server.gpu),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_seconds,
        )
    elapsed_seconds = time.monotonic() - started
    if result.returncode != 0:
        raise RuntimeError(
            f"realtime CLI failed for {sample.sample_id} on GPU {server.gpu}:\n{tail(cli_log)}"
        )

    session_dir = latest_session(log_dir)
    offline_midi = offline_path(sample, output_root)
    combined_midi = session_dir / "combined.mid"
    theoretical_midi = session_dir / "theoretical_model.mid"
    for required in (combined_midi, theoretical_midi, session_dir / "inferences.json"):
        if not required.is_file():
            raise RuntimeError(f"missing artifact for {sample.sample_id}: {required}")

    theoretical = compare_midi(theoretical_midi, offline_midi, sample.melody_last_beat)
    combined = compare_midi(combined_midi, offline_midi, sample.melody_last_beat)
    late_rows = late_schedule_rows(session_dir)
    dropped = count_dropped_requests(session_dir)
    payload = {
        "sample_id": sample.sample_id,
        "dataset": sample.dataset,
        "stem": sample.stem,
        "gpu": server.gpu,
        "tempo": tempo,
        "condition_bpm": CONDITION_BPM,
        "melody_last_beat": sample.melody_last_beat,
        "max_ticks": sample.max_ticks,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "npz_path": str(sample.npz_path),
        "melody_path": str(sample.melody_path),
        "offline_midi": str(offline_midi),
        "session_dir": str(session_dir),
        "theoretical_midi": str(theoretical_midi),
        "combined_midi": str(combined_midi),
        "theoretical_vs_offline": comparison_payload(theoretical),
        "combined_vs_offline": comparison_payload(combined),
        "dropped_requests": dropped,
        "late_schedule_count": len(late_rows),
        "late_schedule_rows": late_rows[:100],
        "server_runtime": server.runtime_info,
        "comparison_resolution": {
            "unit": "model_timestep",
            "steps_per_beat": TICKS_PER_BEAT,
        },
    }
    payload["model_consistent"] = theoretical.consistent
    payload["playback_consistent"] = combined.consistent
    payload["valid_realtime_run"] = dropped == 0 and not late_rows
    payload["passed"] = (
        theoretical.consistent and combined.consistent and dropped == 0 and not late_rows
    )
    return payload


def failure_payload(sample: SampleSpec, gpu: int, tempo: int, exc: BaseException) -> dict[str, Any]:
    return {
        "sample_id": sample.sample_id,
        "dataset": sample.dataset,
        "stem": sample.stem,
        "gpu": gpu,
        "tempo": tempo,
        "passed": False,
        "error": f"{type(exc).__name__}: {exc}",
    }


def reanalyze_existing_results(
    samples: list[SampleSpec], output_root: Path
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for sample in samples:
        result_path = output_root / "results" / f"{sample.artifact_id}.json"
        if not result_path.is_file():
            raise FileNotFoundError(f"missing result for analysis-only: {result_path}")
        payload = json.loads(result_path.read_text())
        session_dir = Path(payload["session_dir"])
        offline_midi = Path(payload["offline_midi"])
        theoretical = compare_midi(
            Path(payload["theoretical_midi"]), offline_midi, sample.melody_last_beat
        )
        combined = compare_midi(
            Path(payload["combined_midi"]), offline_midi, sample.melody_last_beat
        )
        late_rows = late_schedule_rows(session_dir)
        dropped = count_dropped_requests(session_dir)
        payload.update(
            {
                "theoretical_vs_offline": comparison_payload(theoretical),
                "combined_vs_offline": comparison_payload(combined),
                "dropped_requests": dropped,
                "late_schedule_count": len(late_rows),
                "late_schedule_rows": late_rows[:100],
                "model_consistent": theoretical.consistent,
                "playback_consistent": combined.consistent,
                "valid_realtime_run": dropped == 0 and not late_rows,
                "comparison_resolution": {
                    "unit": "model_timestep",
                    "steps_per_beat": TICKS_PER_BEAT,
                },
            }
        )
        payload["passed"] = (
            theoretical.consistent
            and combined.consistent
            and dropped == 0
            and not late_rows
        )
        write_json(result_path, payload)
        results.append(payload)
    return results


def write_summary(
    output_root: Path,
    results: list[dict[str, Any]],
    args: argparse.Namespace,
    checkpoint_sha256: str,
) -> None:
    ordered = sorted(results, key=lambda row: str(row["sample_id"]))
    passed = sum(bool(row.get("passed")) for row in ordered)
    model_passed = sum(bool(row.get("model_consistent")) for row in ordered)
    playback_passed = sum(bool(row.get("playback_consistent")) for row in ordered)
    payload = {
        "schema_version": 1,
        "created_at": datetime.now().isoformat(),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "tempo": args.tempo,
        "condition_bpm": CONDITION_BPM,
        "comparison_resolution": {
            "unit": "model_timestep",
            "steps_per_beat": TICKS_PER_BEAT,
        },
        "gpus": [int(value) for value in args.gpus.split(",") if value.strip()],
        "sample_count": len(ordered),
        "passed_count": passed,
        "model_consistent_count": model_passed,
        "playback_consistent_count": playback_passed,
        "results": ordered,
    }
    write_json(output_root / "summary.json", payload)

    lines = [
        "# All-Sample Offline vs Realtime Consistency",
        "",
        f"- Samples: {len(ordered)}",
        f"- Passed: {passed}/{len(ordered)}",
        f"- Theoretical vs offline: {model_passed}/{len(ordered)}",
        f"- Combined vs offline: {playback_passed}/{len(ordered)}",
        f"- Realtime tempo: {args.tempo} BPM",
        f"- Model conditioning BPM: {CONDITION_BPM}",
        f"- Checkpoint SHA-256: `{payload['checkpoint_sha256']}`",
        "",
        "| Sample | Offline model-step cells | Theoretical | Combined | Dropped | Late | Pass |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in ordered:
        if row.get("error"):
            lines.append(
                f"| {row['sample_id']} | - | ERROR | ERROR | - | - | no |"
            )
            continue
        theoretical = row["theoretical_vs_offline"]
        combined = row["combined_vs_offline"]
        lines.append(
            f"| {row['sample_id']} | {theoretical['offline_count']} | "
            f"{theoretical['matched']}/{theoretical['union_count']} "
            f"({theoretical['match_rate']}%) | "
            f"{combined['matched']}/{combined['union_count']} ({combined['match_rate']}%) | "
            f"{row['dropped_requests']} | {row['late_schedule_count']} | "
            f"{'yes' if row['passed'] else 'no'} |"
        )
    errors = [row for row in ordered if row.get("error")]
    if errors:
        lines.extend(["", "## Errors", ""])
        for row in errors:
            lines.append(f"- `{row['sample_id']}`: {row['error']}")
    (output_root / "summary.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    checkpoint_sha256 = sha256_file(args.checkpoint)
    if args.tempo <= 0 or args.tail_beats < 0:
        raise ValueError("--tempo must be positive and --tail-beats must be non-negative")
    if args.analyze_only and args.output_dir is None:
        raise ValueError("--analyze-only requires --output-dir")
    gpus = [int(value) for value in args.gpus.split(",") if value.strip()]
    if not gpus:
        raise ValueError("--gpus must contain at least one GPU index")

    output_root = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else REPO_ROOT
        / "output"
        / "consistency"
        / f"all_samples_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    datasets = [
        DatasetSpec("inputs_lekai", REPO_ROOT / "prompts" / "inputs_lekai", 5),
        DatasetSpec("old_input", REPO_ROOT / "prompts" / "old_input", 10),
    ]
    samples = [
        sample
        for dataset in datasets
        for sample in discover_samples(dataset, args.tail_beats)
    ]
    if args.samples:
        selected = {value.strip() for value in args.samples.split(",") if value.strip()}
        known = {sample.sample_id for sample in samples}
        unknown = selected - known
        if unknown:
            raise ValueError(f"unknown --samples: {sorted(unknown)}")
        samples = [sample for sample in samples if sample.sample_id in selected]

    if args.analyze_only:
        results = reanalyze_existing_results(samples, output_root)
        write_summary(output_root, results, args, checkpoint_sha256)
        failed = [row for row in results if not row.get("passed")]
        print(
            f"[all-sample] analysis complete: {len(results) - len(failed)}/{len(results)} "
            f"passed; summary={output_root / 'summary.md'}",
            flush=True,
        )
        return 1 if failed else 0

    manifest = {
        "schema_version": 1,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "tempo": args.tempo,
        "condition_bpm": CONDITION_BPM,
        "tail_beats": args.tail_beats,
        "prompt_context_beats_override": args.prompt_context_beats,
        "history_max_ticks_override": args.history_max_ticks,
        "gpus": gpus,
        "samples": [
            {
                **asdict(sample),
                "npz_path": str(sample.npz_path),
                "melody_path": str(sample.melody_path),
                "sample_id": sample.sample_id,
            }
            for sample in samples
        ],
    }
    write_json(output_root / "manifest.json", manifest)
    print(f"[all-sample] output={output_root}", flush=True)
    print(f"[all-sample] samples={len(samples)}, gpus={gpus}", flush=True)

    datasets_needed = [
        dataset for dataset in datasets if any(sample.dataset == dataset.name for sample in samples)
    ]
    print("[all-sample] generating offline references", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(datasets_needed)) as executor:
        futures = {
            executor.submit(
                run_offline_dataset,
                dataset,
                args.checkpoint,
                output_root,
                gpus[index % len(gpus)],
                args.resume,
            ): dataset.name
            for index, dataset in enumerate(datasets_needed)
        }
        for future in concurrent.futures.as_completed(futures):
            dataset_name = futures[future]
            future.result()
            print(f"[all-sample] offline complete: {dataset_name}", flush=True)

    task_queue: queue.Queue[SampleSpec] = queue.Queue()
    for sample in sorted(samples, key=lambda item: item.melody_last_beat, reverse=True):
        result_path = output_root / "results" / f"{sample.artifact_id}.json"
        if args.resume and result_path.is_file():
            existing = json.loads(result_path.read_text())
            if existing.get("passed"):
                continue
        task_queue.put(sample)

    results: list[dict[str, Any]] = []
    results_lock = threading.Lock()
    progress = 0

    def worker(gpu: int) -> None:
        nonlocal progress
        server = start_server(
            gpu=gpu,
            checkpoint=args.checkpoint,
            output_root=output_root,
            prompt_context_beats=args.prompt_context_beats,
            history_max_ticks=args.history_max_ticks,
        )
        try:
            while True:
                try:
                    sample = task_queue.get_nowait()
                except queue.Empty:
                    break
                try:
                    result = run_realtime_sample(
                        sample,
                        server,
                        args.checkpoint,
                        checkpoint_sha256,
                        output_root,
                        args.tempo,
                    )
                except BaseException as exc:
                    result = failure_payload(sample, gpu, args.tempo, exc)
                write_json(output_root / "results" / f"{sample.artifact_id}.json", result)
                with results_lock:
                    results.append(result)
                    progress += 1
                    print(
                        f"[all-sample] {progress}/{len(samples)} {sample.sample_id}: "
                        f"{'PASS' if result.get('passed') else 'FAIL'} (GPU {gpu})",
                        flush=True,
                    )
                task_queue.task_done()
        finally:
            stop_server(server)

    active_gpus = gpus[: min(len(gpus), max(1, task_queue.qsize()))]
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(active_gpus)) as executor:
        futures = [executor.submit(worker, gpu) for gpu in active_gpus]
        for future in concurrent.futures.as_completed(futures):
            future.result()

    if args.resume:
        selected_sample_ids = {sample.sample_id for sample in samples}
        for result_path in sorted((output_root / "results").glob("*.json")):
            existing = json.loads(result_path.read_text())
            if (
                existing["sample_id"] in selected_sample_ids
                and existing["sample_id"] not in {row["sample_id"] for row in results}
            ):
                results.append(existing)

    write_summary(output_root, results, args, checkpoint_sha256)
    failed = [row for row in results if not row.get("passed")]
    print(
        f"[all-sample] complete: {len(results) - len(failed)}/{len(results)} passed; "
        f"summary={output_root / 'summary.md'}",
        flush=True,
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
