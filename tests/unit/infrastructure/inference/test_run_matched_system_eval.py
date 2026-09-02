from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest


@pytest.fixture
def matched_runner() -> ModuleType:
    root = Path(__file__).resolve().parents[4]
    module_name = "_streammuse_test_run_matched_system_eval"
    spec = importlib.util.spec_from_file_location(
        module_name, root / "scripts" / "run_matched_system_eval.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load run_matched_system_eval.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop(module_name, None)


def _identity(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "sha256": "a" * 64,
        "size_bytes": 1,
    }


def test_load_cohort_manifest_resolves_paths_and_checks_hash(
    matched_runner, tmp_path: Path
) -> None:
    script = matched_runner
    midi = tmp_path / "piece.mid"
    midi.write_bytes(b"MThd-matched-input")
    manifest = tmp_path / "cohort.json"
    manifest.write_text(
        json.dumps(
            {
                "pieces": [
                    {
                        "piece_id": "piece-01",
                        "midi_path": "piece.mid",
                        "melody_input_sha256": script.file_sha256(midi),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    pieces = script.load_cohort_manifest(manifest)

    assert len(pieces) == 1
    assert pieces[0].piece_id == "piece-01"
    assert pieces[0].midi_path == midi.resolve()
    assert pieces[0].melody_input_sha256 == script.file_sha256(midi)


def test_server_environments_freeze_mode_specific_contracts(
    matched_runner, tmp_path: Path
) -> None:
    script = matched_runner
    prompt = tmp_path / "prompt.safetensors"
    continuation = tmp_path / "continuation.safetensors"
    prompt.write_bytes(b"p")
    continuation.write_bytes(b"c")
    code = {"git_commit": "b" * 40}

    standard = script.build_server_environment(
        "streammuse_v1_standard",
        port=18001,
        gpu="2",
        server_dir=tmp_path / "standard",
        code=code,
        prompt_checkpoint=_identity(prompt),
        continuation_checkpoint=_identity(continuation),
    )
    prompt_continuation = script.build_server_environment(
        "streammuse_v2_prompt_continuation",
        port=18002,
        gpu="2",
        server_dir=tmp_path / "prompt_continuation",
        code=code,
        prompt_checkpoint=_identity(prompt),
        continuation_checkpoint=_identity(continuation),
    )

    assert standard["LEKAI_CHECKPOINT_PATH"] == str(continuation.resolve())
    assert standard["LEKAI_REQUIRE_SESSION"] == "1"
    assert "LEKAI_PROMPT_CHECKPOINT_PATH" not in standard
    assert prompt_continuation["LEKAI_PROMPT_CHECKPOINT_PATH"] == str(prompt.resolve())
    assert prompt_continuation["LEKAI_CONTINUATION_CHECKPOINT_PATH"] == str(
        continuation.resolve()
    )
    assert "LEKAI_CHECKPOINT_PATH" not in prompt_continuation
    assert "LEKAI_REQUIRE_SESSION" not in prompt_continuation
    assert prompt_continuation["LEKAI_PROMPT_SELECTION_MODE"] == "rule_s"
    assert prompt_continuation["LEKAI_PROMPT_BATCH_CANDIDATES"] == "5"
    assert prompt_continuation["LEKAI_PROMPT_TEMPERATURE"] == "1.05"
    assert prompt_continuation["LEKAI_PROMPT_TOP_P"] == "0.98"
    assert prompt_continuation["LEKAI_PROMPT_TOP_K"] == "0"
    assert prompt_continuation["LEKAI_PROMPT_REPETITION_PENALTY"] == "1.0"
    assert prompt_continuation[
        "LEKAI_PROMPT_CONTINUATION_RECOVER_LATE_EVENTS"
    ] == "0"


def test_reset_trial_uses_mode_specific_atomic_endpoint(
    matched_runner, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = matched_runner
    calls: list[tuple[str, str, dict[str, int]]] = []

    def fake_request(url, *, method="GET", payload=None, timeout=30.0):
        calls.append((url, method, dict(payload or {})))
        if "prompt_continuation" in url:
            return {
                "success": True,
                "prompt_seed": 7,
                "continuation_effective_seed": 7,
                "session_id": "pc-session",
                "session_epoch": 2,
                "pending_boundary_generations": 0,
                "scheduler_phase": "idle",
                "scheduler_is_running": False,
            }
        return {
            "success": True,
            "effective_seed": 7,
            "session_id": "standard-session",
            "session_epoch": 3,
            "pending_boundary_generations": 0,
        }

    monkeypatch.setattr(script, "request_json", fake_request)

    standard = script.reset_trial(
        "streammuse_v1_standard", "http://127.0.0.1:1", 7
    )
    prompt_continuation = script.reset_trial(
        "streammuse_v2_prompt_continuation", "http://127.0.0.1:2", 7
    )

    assert standard["session_id"] == "standard-session"
    assert prompt_continuation["session_id"] == "pc-session"
    assert calls == [
        (
            "http://127.0.0.1:1/debug/reset_session",
            "POST",
            {"seed": 7},
        ),
        (
            "http://127.0.0.1:2/prompt_continuation/debug/reset_session",
            "POST",
            {"prompt_seed": 7, "continuation_seed": 7},
        ),
    ]


def _write_session(script, root: Path, *, mode: str) -> Path:
    session = root / "2026-09-02" / "session_120000"
    session.mkdir(parents=True)
    script.write_json(
        session / "session_config.json",
        {
            "tempo_bpm": 120.0,
            "ticks_per_beat": 4,
            "beats_per_bar": 4,
            "input_type": "midi_file",
            "output_type": "session",
            "session_artifact_tier": "debug",
            "continuation_mode": mode,
            "generation_interval_ticks": 4,
            "generation_length_frames": 4,
            "count_in_beats": 0,
        },
    )
    condition = "standard" if mode == "standard" else "prompt_continuation"
    rows = [
        {
            "schema_version": 2,
            "record_type": "frame_deadline",
            "condition": condition,
            "tick": tick,
        }
        for tick in range(128)
    ]
    (session / "system_trace.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    return session


def test_validate_session_requires_contiguous_schema_v2_deadlines(
    matched_runner, tmp_path: Path
) -> None:
    script = matched_runner
    session = _write_session(script, tmp_path / "valid", mode="prompt_continuation")

    result = script.validate_session(
        "streammuse_v2_prompt_continuation", tmp_path / "valid"
    )

    assert result["session_dir"] == str(session.resolve())
    assert result["deadline_count"] == 128
    assert result["deadline_ticks_contiguous"] is True

    rows = (session / "system_trace.jsonl").read_text(encoding="utf-8").splitlines()
    (session / "system_trace.jsonl").write_text(
        "\n".join(rows[:64] + rows[65:]) + "\n", encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="continuously cover ticks 0..127"):
        script.validate_session(
            "streammuse_v2_prompt_continuation", tmp_path / "valid"
        )


def test_dry_run_builds_matched_piece_seed_system_matrix(
    matched_runner, tmp_path: Path
) -> None:
    script = matched_runner
    prompt = tmp_path / "prompt.safetensors"
    continuation = tmp_path / "continuation.safetensors"
    prompt.write_bytes(b"prompt")
    continuation.write_bytes(b"continuation")
    first = tmp_path / "first.mid"
    second = tmp_path / "second.mid"
    first.write_bytes(b"MThd-first")
    second.write_bytes(b"MThd-second")
    cohort = tmp_path / "cohort.json"
    cohort.write_text(
        json.dumps(
            {
                "pieces": [
                    {"piece_id": "first", "midi_path": "first.mid"},
                    {"piece_id": "second", "midi_path": "second.mid"},
                ]
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "output"
    args = script.parse_args(
        [
            "--cohort-manifest",
            str(cohort),
            "--output-root",
            str(output),
            "--prompt-checkpoint",
            str(prompt),
            "--continuation-checkpoint",
            str(continuation),
            "--seeds",
            "3,7",
            "--smoke-limit",
            "1",
            "--dry-run",
        ]
    )

    result = script.run_evaluation(args)

    assert result["run_status"] == "dry_run"
    assert result["summary"] == {"dry_run": 4}
    assert len(result["trials"]) == 4
    assert {row["piece_id"] for row in result["trials"]} == {"first"}
    assert {row["seed"] for row in result["trials"]} == {3, 7}
    assert {row["system_id"] for row in result["trials"]} == set(script.SYSTEM_IDS)
    assert len({row["melody_input_sha256"] for row in result["trials"]}) == 1
    for row in result["trials"]:
        command = row["client_command"]
        assert command[command.index("--tempo") + 1] == "120"
        assert command[command.index("--ticks-per-beat") + 1] == "4"
        assert command[command.index("--run-stop-tick") + 1] == "128"
    assert (output / "run_manifest.json").is_file()
