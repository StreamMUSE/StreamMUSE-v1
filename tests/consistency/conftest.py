"""Fixtures and helpers for the realtime/offline consistency test.

This whole package is opt-in: it needs a real Lekai checkpoint and a GPU, and each case
runs a real wall-clock realtime session (tens of seconds). It is skipped entirely unless
``LEKAI_CHECKPOINT_PATH`` points at an existing file.

Run it with::

    LEKAI_CHECKPOINT_PATH=models/ModelLekai/epoch_4_1104_1204/model.safetensors \
      uv run pytest tests/consistency/ -v

See developing-logs/plans/2026-06-12-consistency-final-test-plan.md for the design and the
Phase 0 findings that justify the comparison method (pianoroll-level, non-empty songs,
windowed to the melody length).
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import mido
import pytest

try:  # requests is already a dependency of the inference stack
    import requests
except ImportError:  # pragma: no cover - defensive
    requests = None  # type: ignore[assignment]


REPO_ROOT = Path(__file__).resolve().parents[2]
NPZ_DIR = REPO_ROOT / "prompts" / "inputs_lekai" / "npz"
MEL_DIR = REPO_ROOT / "prompts" / "inputs_lekai" / "mel"

# Conditioning BPM pinned on BOTH sides so the BPM token is identical (see plan core #2).
# 120 lands in the encode_bpm "medium" bucket; the song's native BPM is irrelevant as long
# as both sides agree.
CONDITION_BPM = 120

# Deterministic (greedy) sampling: temperature==0 short-circuits to argmax in
# generation_utils.sample_token, so top_k/top_p/repetition_penalty are inert but kept for
# parity with the historical commands.
DETERMINISTIC_SERVER_ENV = {
    "LEKAI_RT_TEMPERATURE": "0.0",
    "LEKAI_RT_TOP_K": "1",
    "LEKAI_RT_TOP_P": "0.0",
    "LEKAI_RT_REPETITION_PENALTY": "1.2",
    "LEKAI_DEFAULT_BPM": str(CONDITION_BPM),
}

# Dataset file ordering is NOT numeric: PianoDataset lists [2,5,3,1,4].npz, so the offline
# --condition-idx does not equal the song number. (Verified Phase 0.)
SONG_TO_CONDITION_IDX = {1: 3, 2: 0, 3: 2, 4: 4, 5: 1}

# Greedy collapses many songs to empty accompaniment; only these produce non-trivial,
# deterministic output worth comparing (Phase 0 appendix B). song 1/3 are (near-)empty.
NON_EMPTY_SONGS = (4, 5, 2)

# Extra beats the realtime run goes past the melody end (see SongSpec.max_ticks).
TAIL_BEATS = 24


@dataclass(frozen=True)
class SongSpec:
    number: int
    condition_idx: int
    melody_last_beat: int

    @property
    def mel_path(self) -> Path:
        return MEL_DIR / f"{self.number}.mid"

    @property
    def max_ticks(self) -> int:
        # Generous tail past the melody end: the realtime path needs headroom to GENERATE and
        # play out the accompaniment for the final in-window beats (a note conditioned on the
        # last melody beat can sustain a few beats further, and playback must reach its
        # note_off). Too tight a tail drops the last in-window cells (verified: 2-beat tail
        # missed cell (56,43) for song 4; a ~24-beat tail matches 100%). The comparison
        # window (melody_last_beat) ignores whatever the tail generates past the song.
        return (self.melody_last_beat + TAIL_BEATS) * 4


def _melody_last_beat(mel_path: Path) -> int:
    mid = mido.MidiFile(str(mel_path))
    tpb = mid.ticks_per_beat
    last_tick = 0
    for track in mid.tracks:
        abs_tick = 0
        for msg in track:
            abs_tick += msg.time
            if msg.type == "note_on" and msg.velocity > 0:
                last_tick = max(last_tick, abs_tick)
    return last_tick // tpb


def song_spec(number: int) -> SongSpec:
    return SongSpec(
        number=number,
        condition_idx=SONG_TO_CONDITION_IDX[number],
        melody_last_beat=_melody_last_beat(MEL_DIR / f"{number}.mid"),
    )


def _checkpoint_path() -> Path | None:
    raw = os.environ.get("LEKAI_CHECKPOINT_PATH")
    if not raw:
        return None
    p = Path(raw)
    return p if p.exists() else None


@pytest.fixture(scope="session")
def lekai_checkpoint() -> Path:
    ckpt = _checkpoint_path()
    if ckpt is None:
        pytest.skip("consistency test requires LEKAI_CHECKPOINT_PATH pointing at a checkpoint")
    if requests is None:
        pytest.skip("consistency test requires the 'requests' package")
    return ckpt


@pytest.fixture(scope="session")
def artifacts_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    d = REPO_ROOT / "output" / "consistency" / stamp
    d.mkdir(parents=True, exist_ok=True)
    return d


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("LEKAI_DEVICE", "cuda")
    env.setdefault("LEKAI_DTYPE", "auto")
    gpu = os.environ.get("STREAMMUSE_CONSISTENCY_GPU")
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    return env


@dataclass
class LekaiServer:
    base_url: str
    generate_url: str
    log_path: Path


@pytest.fixture(scope="session")
def lekai_server(lekai_checkpoint: Path, artifacts_dir: Path) -> LekaiServer:
    port = _free_port()
    log_path = artifacts_dir / "server.log"

    env = _subprocess_env()
    env.update(DETERMINISTIC_SERVER_ENV)
    env["LEKAI_CHECKPOINT_PATH"] = str(lekai_checkpoint)
    env["LEKAI_SERVER_PORT"] = str(port)

    base_url = f"http://127.0.0.1:{port}"
    log_file = log_path.open("w")
    proc = subprocess.Popen(
        [sys.executable, "-m", "streammuse.infrastructure.inference.server_lekai"],
        cwd=str(REPO_ROOT),
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )

    try:
        _wait_for_health(base_url, proc, log_path, timeout_s=120)
        # Warm up: first request triggers CUDA kernel compilation / allocation, which is
        # exactly when a fast tempo would otherwise drop a stale request.
        requests.post(f"{base_url}/clear_history", timeout=30)
        server = LekaiServer(
            base_url=base_url,
            generate_url=f"{base_url}/generate_accompaniment",
            log_path=log_path,
        )
        yield server
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=15)
        log_file.close()


def _wait_for_health(base_url: str, proc: subprocess.Popen, log_path: Path, *, timeout_s: int) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            tail = _tail(log_path)
            raise RuntimeError(f"server exited early (code {proc.returncode}). Log tail:\n{tail}")
        try:
            r = requests.get(f"{base_url}/health", timeout=2)
            if r.ok and r.json().get("status") == "ok":
                return
        except Exception:
            pass
        time.sleep(1.0)
    proc.terminate()
    raise RuntimeError(f"server not ready within {timeout_s}s. Log tail:\n{_tail(log_path)}")


def _tail(path: Path, n: int = 30) -> str:
    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError:
        return "(no log)"
    return "\n".join(lines[-n:])


# --- Two-stage prompt+continuation fixtures ---

DEFAULT_MODELS_DIR = Path.home() / "mbzuai-projects" / "models"
DEFAULT_PROMPT_CHECKPOINT = DEFAULT_MODELS_DIR / "lekai_prompt_model" / "model.safetensors"
DEFAULT_CONTINUATION_CHECKPOINT = DEFAULT_MODELS_DIR / "lekai_continuation_model" / "model.safetensors"

TWO_STAGE_DETERMINISTIC_ENV = {
    **DETERMINISTIC_SERVER_ENV,
    "LEKAI_PROMPT_CONTINUATION_ENGINE": "standard",
    "LEKAI_PROMPT_CONTINUATION_REQUIRE_REAL_MODELS": "1",
    "LEKAI_PROMPT_CONTINUATION_RECOVER_LATE_EVENTS": "0",
    "LEKAI_PROMPT_CONTINUATION_REHYDRATE_ACTIVE_NOTES": "0",
    "LEKAI_PROMPT_CONTINUATION_STRICT_REPRESENTATION_LOOP": "1",
    "LEKAI_PROMPT_SEED": "12345",
    "LEKAI_PROMPT_BPM": str(CONDITION_BPM),
    "LEKAI_PROMPT_TEMPERATURE": "1.1",
    "LEKAI_PROMPT_TOP_K": "0",
    "LEKAI_PROMPT_TOP_P": "0.95",
    "LEKAI_PROMPT_REPETITION_PENALTY": "1.0",
}


def _env_path_or_default(env_name: str, default: Path) -> Path | None:
    raw = os.environ.get(env_name)
    if raw:
        path = Path(raw).expanduser()
        return path if path.exists() else None
    # Keep consistency tests opt-in unless the caller explicitly allows local defaults.
    if os.environ.get("STREAMMUSE_CONSISTENCY_USE_DEFAULT_MODELS", "").lower() in {"1", "true", "yes", "on"}:
        return default if default.exists() else None
    return None


@pytest.fixture(scope="session")
def lekai_prompt_checkpoint() -> Path:
    ckpt = _env_path_or_default("LEKAI_PROMPT_CHECKPOINT_PATH", DEFAULT_PROMPT_CHECKPOINT)
    if ckpt is None:
        pytest.skip(
            "two-stage consistency requires LEKAI_PROMPT_CHECKPOINT_PATH "
            "or STREAMMUSE_CONSISTENCY_USE_DEFAULT_MODELS=1"
        )
    if requests is None:
        pytest.skip("consistency test requires the 'requests' package")
    return ckpt


@pytest.fixture(scope="session")
def lekai_continuation_checkpoint() -> Path:
    ckpt = _env_path_or_default("LEKAI_CONTINUATION_CHECKPOINT_PATH", DEFAULT_CONTINUATION_CHECKPOINT)
    if ckpt is None:
        pytest.skip(
            "two-stage consistency requires LEKAI_CONTINUATION_CHECKPOINT_PATH "
            "or STREAMMUSE_CONSISTENCY_USE_DEFAULT_MODELS=1"
        )
    if requests is None:
        pytest.skip("consistency test requires the 'requests' package")
    return ckpt


@pytest.fixture(scope="session")
def two_stage_lekai_server(
    lekai_prompt_checkpoint: Path,
    lekai_continuation_checkpoint: Path,
    artifacts_dir: Path,
) -> LekaiServer:
    port = _free_port()
    log_path = artifacts_dir / "two_stage_server.log"

    env = _subprocess_env()
    env.update(TWO_STAGE_DETERMINISTIC_ENV)
    env["LEKAI_PROMPT_CHECKPOINT_PATH"] = str(lekai_prompt_checkpoint)
    env["LEKAI_CONTINUATION_CHECKPOINT_PATH"] = str(lekai_continuation_checkpoint)
    env["LEKAI_SERVER_PORT"] = str(port)

    base_url = f"http://127.0.0.1:{port}"
    log_file = log_path.open("w")
    proc = subprocess.Popen(
        [sys.executable, "-m", "streammuse.infrastructure.inference.server_lekai"],
        cwd=str(REPO_ROOT),
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )

    try:
        _wait_for_health(base_url, proc, log_path, timeout_s=180)
        info = requests.get(f"{base_url}/prompt_continuation/runtime_info", timeout=30).json()
        if not bool(info.get("has_real_model")) or not bool(info.get("prompt_has_real_model")):
            raise RuntimeError(
                "two-stage server did not load real models. "
                f"runtime_info={info}\nLog tail:\n{_tail(log_path)}"
            )
        requests.post(f"{base_url}/clear_history", timeout=30)
        yield LekaiServer(
            base_url=base_url,
            generate_url=f"{base_url}/generate_accompaniment",
            log_path=log_path,
        )
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=15)
        log_file.close()
