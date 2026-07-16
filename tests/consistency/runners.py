"""Subprocess runners for the realtime (CLI) and offline (script) generation paths.

These drive the real entry points unchanged — the test only orchestrates and asserts; it
never re-implements generation. See conftest.py for the song/env constants.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tests.consistency.conftest import (
    CONDITION_BPM,
    NPZ_DIR,
    REPO_ROOT,
    LekaiServer,
    SongSpec,
    _subprocess_env,
)

GENERATION_INTERVAL_TICKS = 4
GENERATION_LENGTH_FRAMES = 4


def run_realtime(
    server: LekaiServer,
    song: SongSpec,
    *,
    tempo: int,
    out_dir: Path,
    timeout_s: float | None = None,
) -> Path:
    """Run the real RealTimeMusicService via streammuse-cli against the live server.

    Returns the session directory containing combined.mid / inferences.json / etc.
    """
    log_dir = out_dir / f"song{song.number}_tempo{tempo}"
    log_dir.mkdir(parents=True, exist_ok=True)

    if timeout_s is None:
        # Wall-clock ~ max_ticks / ticks_per_beat / tempo * 60, plus generous slack.
        beats = song.max_ticks / 4
        timeout_s = max(120.0, (beats / tempo) * 60 * 2 + 120)

    cmd = [
        sys.executable,
        "-m",
        "streammuse.presentation.cli.cli",
        "--input-mode", "midi_file",
        "--midi-file-path", str(song.mel_path),
        "--model-name", "lekai",
        "--inference-type", "http",
        "--server-url", server.generate_url,
        "--generation-interval-ticks", str(GENERATION_INTERVAL_TICKS),
        "--generation-length-frames", str(GENERATION_LENGTH_FRAMES),
        "--max-ticks", str(song.max_ticks),
        "--tempo", str(tempo),
        "--model-condition-bpm", str(CONDITION_BPM),
        "--output-type", "session",
        "--log-dir", str(log_dir),
    ]
    result = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        env=_subprocess_env(),
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"realtime CLI failed (code {result.returncode}) for song {song.number} tempo {tempo}.\n"
            f"stdout tail:\n{result.stdout[-2000:]}\nstderr tail:\n{result.stderr[-2000:]}"
        )

    sessions = sorted(p for p in log_dir.rglob("session_*") if p.is_dir())
    if not sessions:
        raise RuntimeError(f"no session dir produced under {log_dir}")
    return sessions[-1]


def count_dropped_requests(session_dir: Path, *, interval: int = GENERATION_INTERVAL_TICKS) -> int:
    """Count dropped (merged) inference requests from inferences.json.

    In a clean run the recorded generation_start_ticks step by exactly ``interval`` (one per
    beat tail). Under load the latest-only drain merges requests, so a tick is skipped and
    the step jumps. The number of skipped triggers is ``(gap // interval) - 1`` summed over
    gaps. A nonzero result means the run was too slow at this tempo, not a consistency bug.
    """
    inferences = json.loads((session_dir / "inferences.json").read_text())
    ticks = [r["request_data"]["generation_start_tick"] for r in inferences]
    dropped = 0
    for prev, cur in zip(ticks, ticks[1:]):
        step = cur - prev
        if step > interval:
            dropped += step // interval - 1
    return dropped


def run_offline(checkpoint: Path, song: SongSpec, *, out_dir: Path) -> Path:
    """Run run_lekai_offline.py for one song with greedy params, BPM pinned to CONDITION_BPM.

    Returns the generated MIDI path selected by the song's stable NPZ stem.
    """
    offline_dir = out_dir / f"offline_song{song.number}"
    offline_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_lekai_offline.py"),
        "--checkpoint", str(checkpoint),
        "--npz-dir", str(NPZ_DIR),
        "--output-dir", str(offline_dir),
        "--device", "cuda",
        "--dtype", "auto",
        "--temperature", "0.0",
        "--top-k", "1",
        "--top-p", "0.0",
        "--gt-prefix-beats", "0",
        "--bpm", str(CONDITION_BPM),
        "--seed", "0",
        "--expected-dataset-size", "5",
        "--condition-stem", song.npz_stem,
        "--source-midi", str(song.mel_path),
        "--require-source-midi",
    ]
    result = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        env=_subprocess_env(),
        capture_output=True,
        text=True,
        timeout=600,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"offline run failed (code {result.returncode}) for song {song.number}.\n"
            f"stdout tail:\n{result.stdout[-2000:]}\nstderr tail:\n{result.stderr[-2000:]}"
        )

    generated = sorted(offline_dir.glob(f"*_{song.npz_stem}_generated.mid"))
    if not generated:
        raise RuntimeError(f"no generated MIDI under {offline_dir}")
    return generated[-1]
