"""Runners for prompt+continuation realtime/offline consistency tests."""

from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from tests.consistency.conftest import (
    CONDITION_BPM,
    REPO_ROOT,
    LekaiServer,
    SongSpec,
    _subprocess_env,
)

GENERATION_INTERVAL_TICKS = 4
PROMPT_LENGTH_TICKS = 32


@dataclass(frozen=True)
class TwoStageRealtimeRun:
    session_dir: Path
    trace_path: Path
    stdout_path: Path
    stderr_path: Path
    prompt_history_path: Path
    raw_history_path: Path
    history_status_path: Path


@dataclass(frozen=True)
class TwoStageOfflineRun:
    song_dir: Path
    final_midi: Path
    summary_path: Path
    summary: dict[str, object]


def run_two_stage_realtime(
    server: LekaiServer,
    song: SongSpec,
    *,
    tempo: int,
    out_dir: Path,
    timeout_s: float | None = None,
) -> TwoStageRealtimeRun:
    log_dir = out_dir / f"two_stage_song{song.number}_tempo{tempo}"
    log_dir.mkdir(parents=True, exist_ok=True)
    trace_path = log_dir / "prompt_continuation_trace.jsonl"
    trace_path.write_text("")
    stdout_path = log_dir / "stdout.log"
    stderr_path = log_dir / "stderr.log"

    if timeout_s is None:
        beats = song.max_ticks / 4
        timeout_s = max(180.0, (beats / tempo) * 60 * 2 + 180)

    cmd = [
        sys.executable,
        "-m",
        "streammuse.presentation.cli.cli",
        "--input-mode", "midi_file",
        "--midi-file-path", str(song.mel_path),
        "--inference-type", "http",
        "--continuation-mode", "prompt_continuation",
        "--prompt-length-ticks", str(PROMPT_LENGTH_TICKS),
        "--server-url", server.base_url,
        "--generation-interval-ticks", str(GENERATION_INTERVAL_TICKS),
        "--max-ticks", str(song.max_ticks),
        "--tempo", str(tempo),
        "--output-type", "session",
        "--log-dir", str(log_dir),
    ]
    env = _subprocess_env()
    env.update(
        {
            "LEKAI_PROMPT_CONTINUATION_TRACE_PATH": str(trace_path),
            "LEKAI_PROMPT_CONTINUATION_RECOVER_LATE_EVENTS": "0",
            "LEKAI_PROMPT_CONTINUATION_REHYDRATE_ACTIVE_NOTES": "0",
            "LEKAI_PROMPT_CONTINUATION_STRICT_REPRESENTATION_LOOP": "1",
        }
    )
    result = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    stdout_path.write_text(result.stdout)
    stderr_path.write_text(result.stderr)
    if result.returncode != 0:
        raise RuntimeError(
            f"two-stage realtime CLI failed (code {result.returncode}) for song {song.number} tempo {tempo}.\n"
            f"stdout tail:\n{result.stdout[-2000:]}\nstderr tail:\n{result.stderr[-2000:]}"
        )
    sessions = sorted(p for p in log_dir.rglob("session_*") if p.is_dir())
    if not sessions:
        raise RuntimeError(f"no session dir produced under {log_dir}")
    session_dir = sessions[-1]
    prompt_history_path = session_dir / "prompt_continuation_prompt_history.json"
    raw_history_path = session_dir / "prompt_continuation_raw_history.json"
    history_status_path = session_dir / "prompt_continuation_history_status.json"
    missing = [
        str(path)
        for path in (prompt_history_path, raw_history_path, history_status_path)
        if not path.exists()
    ]
    if missing:
        raise RuntimeError(
            "two-stage realtime session did not save server-side history before clear: "
            + ", ".join(missing)
        )
    return TwoStageRealtimeRun(
        session_dir=session_dir,
        trace_path=trace_path,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        prompt_history_path=prompt_history_path,
        raw_history_path=raw_history_path,
        history_status_path=history_status_path,
    )


def load_event_history(path: Path) -> list[dict[str, int | str]]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, list):
        raise ValueError(f"event history is not a list: {path}")
    events: list[dict[str, int | str]] = []
    for event in payload:
        if not isinstance(event, dict):
            continue
        if "type" not in event or "pitch" not in event or "tick" not in event:
            continue
        events.append(
            {
                "type": str(event["type"]),
                "pitch": int(event["pitch"]),
                "tick": int(event["tick"]),
            }
        )
    return events


def event_counter(
    events: list[dict[str, int | str]],
    *,
    min_tick: int | None = None,
    max_tick: int | None = None,
) -> Counter[tuple[int, int, str]]:
    counter: Counter[tuple[int, int, str]] = Counter()
    for event in events:
        tick = int(event["tick"])
        if min_tick is not None and tick < int(min_tick):
            continue
        if max_tick is not None and tick >= int(max_tick):
            continue
        counter[(tick, int(event["pitch"]), str(event["type"]))] += 1
    return counter


def continuation_counter(
    raw_events: list[dict[str, int | str]],
    prompt_events: list[dict[str, int | str]],
    *,
    min_tick: int = PROMPT_LENGTH_TICKS,
    max_tick: int | None = None,
) -> Counter[tuple[int, int, str]]:
    # Raw history contains the prompt seed plus continuation. Subtract the prompt
    # multiset first so boundary events at prompt_length_ticks are classified by origin.
    continuation = event_counter(raw_events) - event_counter(prompt_events)
    windowed: Counter[tuple[int, int, str]] = Counter()
    for key, count in continuation.items():
        tick = key[0]
        if tick < int(min_tick):
            continue
        if max_tick is not None and tick >= int(max_tick):
            continue
        windowed[key] = count
    return windowed


def counter_summary(
    left_name: str,
    left: Counter[tuple[int, int, str]],
    right_name: str,
    right: Counter[tuple[int, int, str]],
    *,
    head: int = 20,
) -> str:
    only_left = left - right
    only_right = right - left
    left_items = sorted(only_left.items())[:head]
    right_items = sorted(only_right.items())[:head]
    return (
        f"{left_name} events={sum(left.values())}, {right_name} events={sum(right.values())}, "
        f"only_{left_name}={sum(only_left.values())}, only_{right_name}={sum(only_right.values())}\n"
        f"only {left_name} first {head}: {left_items}\n"
        f"only {right_name} first {head}: {right_items}"
    )


def count_dropped_and_clipped(trace_path: Path) -> dict[str, int]:
    dropped_past = 0
    clipped = 0
    dropped_too_late = 0
    skipped_unpaired = 0
    scheduled_events = 0
    schedule_rows = 0
    if not trace_path.exists():
        raise FileNotFoundError(f"trace file not found: {trace_path}")
    for line in trace_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("kind") != "schedule_playable":
            continue
        schedule_rows += 1
        dropped_past += int(row.get("dropped_past", 0) or 0)
        clipped += int(row.get("clipped_sustains", 0) or 0)
        dropped_too_late += int(row.get("dropped_too_late_note_on", 0) or 0)
        skipped_unpaired += int(row.get("skipped_unpaired", 0) or 0)
        scheduled_events += int(row.get("scheduled_event_count", 0) or 0)
    return {
        "dropped_past": dropped_past,
        "clipped_sustains": clipped,
        "dropped_too_late_note_on": dropped_too_late,
        "skipped_unpaired": skipped_unpaired,
        "scheduled_events": scheduled_events,
        "schedule_rows": schedule_rows,
    }


def run_two_stage_offline(
    prompt_checkpoint: Path,
    continuation_checkpoint: Path,
    song: SongSpec,
    *,
    out_dir: Path,
) -> TwoStageOfflineRun:
    offline_root = out_dir / f"two_stage_offline_song{song.number}"
    offline_root.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_lekai_prompt_continuation_offline.py"),
        "--midi-file", str(song.mel_path),
        "--output-dir", str(offline_root),
        "--prompt-checkpoint", str(prompt_checkpoint),
        "--continuation-checkpoint", str(continuation_checkpoint),
        "--device", "cuda",
        "--dtype", "auto",
        "--prompt-length-ticks", str(PROMPT_LENGTH_TICKS),
        "--generation-interval-ticks", str(GENERATION_INTERVAL_TICKS),
        "--bpm", str(CONDITION_BPM),
        "--prompt-seed", "12345",
        "--prompt-temperature", "1.1",
        "--prompt-top-k", "0",
        "--prompt-top-p", "0.95",
        "--prompt-repetition-penalty", "1.0",
        "--rt-temperature", "0.0",
        "--rt-top-k", "1",
        "--rt-top-p", "0.0",
        "--rt-repetition-penalty", "1.0",
    ]
    result = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        env=_subprocess_env(),
        capture_output=True,
        text=True,
        timeout=1200,
    )
    (offline_root / "stdout.log").write_text(result.stdout)
    (offline_root / "stderr.log").write_text(result.stderr)
    if result.returncode != 0:
        raise RuntimeError(
            f"two-stage offline run failed (code {result.returncode}) for song {song.number}.\n"
            f"stdout tail:\n{result.stdout[-2000:]}\nstderr tail:\n{result.stderr[-2000:]}"
        )
    summaries = sorted(offline_root.glob("*/summary.json"))
    if not summaries:
        raise RuntimeError(f"no summary.json produced under {offline_root}")
    summary_path = summaries[-1]
    summary = json.loads(summary_path.read_text())
    final_midi = Path(str(summary["final_midi"]))
    if not final_midi.exists():
        raise RuntimeError(f"offline final MIDI missing: {final_midi}")
    return TwoStageOfflineRun(
        song_dir=summary_path.parent,
        final_midi=final_midi,
        summary_path=summary_path,
        summary=summary,
    )
