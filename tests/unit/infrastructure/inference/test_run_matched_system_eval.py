from __future__ import annotations

import csv
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


def test_load_cohort_manifest_accepts_builder_samples_schema(
    matched_runner, tmp_path: Path
) -> None:
    script = matched_runner
    sample_dir = tmp_path / "001"
    sample_dir.mkdir()
    midi = sample_dir / "melody_120bpm.mid"
    midi.write_bytes(b"MThd-builder-sample")
    byte_hash = script.file_sha256(midi)
    canonical_hash = "c" * 64
    manifest = tmp_path / "cohort_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset_contract": "checkpoint-aligned legacy BEAT 192788 NPZ",
                "split": {"split_seed": 42},
                "selection": {"count": 1, "seed": 0},
                "samples": [
                    {
                        "order": 1,
                        "piece_id": "001",
                        "selection_rank": 1,
                        "test_position": 12,
                        "source_npz": "001/source.npz",
                        "melody_midi": "001/melody_120bpm.mid",
                        "gt_midi": "001/gt_120bpm.mid",
                        "num_measures": 32,
                        "num_steps": 512,
                        "melody_note_count": 80,
                        "accompaniment_note_count": 160,
                        "source_npz_sha256": "a" * 64,
                        "melody_midi_sha256": byte_hash,
                        "gt_midi_sha256": "b" * 64,
                        "canonical_melody_input_sha256": canonical_hash,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    pieces = script.load_cohort_manifest(manifest)

    assert len(pieces) == 1
    assert pieces[0].midi_path == midi.resolve()
    assert pieces[0].melody_input_sha256 == byte_hash
    assert pieces[0].canonical_melody_input_sha256 == canonical_hash

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["samples"][0]["melody_midi_sha256"] = "d" * 64
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="MIDI hash mismatch"):
        script.load_cohort_manifest(manifest)


def test_default_seed_contract_is_three_trials(matched_runner) -> None:
    script = matched_runner
    args = script.parse_args(
        [
            "--cohort-manifest",
            "cohort.json",
            "--output-root",
            "output",
            "--prompt-checkpoint",
            "prompt.safetensors",
            "--continuation-checkpoint",
            "continuation.safetensors",
        ]
    )

    assert args.seeds == "0,1,2"
    assert args.prompt_selection_mode == "rule_s"
    assert args.prompt_batch_candidates == 5


@pytest.mark.parametrize("mode", ["rule_s", "rule_s_v3"])
def test_ranked_prompt_modes_require_at_least_two_candidates(
    matched_runner, mode: str
) -> None:
    with pytest.raises(ValueError, match="must be at least 2"):
        matched_runner.effective_prompt_candidate_count(mode, 1)

    assert matched_runner.effective_prompt_candidate_count(mode, 3) == 3


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
    assert standard["LEKAI_TIME_SIGNATURE_INDEX"] == "0"
    assert standard["LEKAI_PROMPT_TIME_SIGNATURE_INDEX"] == "0"
    assert "LEKAI_PROMPT_CHECKPOINT_PATH" not in standard
    assert prompt_continuation["LEKAI_PROMPT_CHECKPOINT_PATH"] == str(prompt.resolve())
    assert prompt_continuation["LEKAI_CONTINUATION_CHECKPOINT_PATH"] == str(
        continuation.resolve()
    )
    assert "LEKAI_CHECKPOINT_PATH" not in prompt_continuation
    assert "LEKAI_REQUIRE_SESSION" not in prompt_continuation
    assert prompt_continuation["LEKAI_TIME_SIGNATURE_INDEX"] == "0"
    assert prompt_continuation["LEKAI_PROMPT_TIME_SIGNATURE_INDEX"] == "0"
    assert prompt_continuation["LEKAI_PROMPT_SELECTION_MODE"] == "rule_s"
    assert prompt_continuation["LEKAI_PROMPT_BATCH_CANDIDATES"] == "5"
    assert prompt_continuation["LEKAI_PROMPT_TEMPERATURE"] == "1.05"
    assert prompt_continuation["LEKAI_PROMPT_TOP_P"] == "0.98"
    assert prompt_continuation["LEKAI_PROMPT_TOP_K"] == "0"
    assert prompt_continuation["LEKAI_PROMPT_REPETITION_PENALTY"] == "1.0"
    assert prompt_continuation[
        "LEKAI_PROMPT_CONTINUATION_RECOVER_LATE_EVENTS"
    ] == "0"


def test_single_prompt_selection_uses_effective_n1_env_and_runtime_contract(
    matched_runner, tmp_path: Path
) -> None:
    script = matched_runner
    prompt = tmp_path / "prompt.safetensors"
    continuation = tmp_path / "continuation.safetensors"
    prompt.write_bytes(b"p")
    continuation.write_bytes(b"c")
    code = {"git_commit": "b" * 40}
    prompt_identity = _identity(prompt)
    continuation_identity = _identity(continuation)

    env = script.build_server_environment(
        "streammuse_v2_prompt_continuation",
        port=18002,
        gpu="2",
        server_dir=tmp_path / "prompt_continuation",
        code=code,
        prompt_checkpoint=prompt_identity,
        continuation_checkpoint=continuation_identity,
        prompt_selection_mode="single",
    )

    assert env["LEKAI_PROMPT_SELECTION_MODE"] == "single"
    assert env["LEKAI_PROMPT_BATCH_CANDIDATES"] == "1"

    runtime = {
        "has_real_model": True,
        "fallback_reason": None,
        "checkpoint_sha256": continuation_identity["sha256"],
        "code_identity": code["git_commit"],
        "resolved_device": "cuda:0",
        "effective_bpm": script.BPM,
        "ticks_per_beat": script.TICKS_PER_BEAT,
        "prompt_context_beats": script.PROMPT_CONTEXT_BEATS,
        "history_retention_ticks": script.HISTORY_MAX_TICKS,
        "time_signature_index": script.CHECKPOINT_TIME_SIGNATURE_INDEX,
        "prompt_has_real_model": True,
        "prompt_fallback_reason": None,
        "prompt_checkpoint_path": prompt_identity["path"],
        "prompt_selection_mode": "single",
        "prompt_batch_candidate_count": 1,
        **script.SAMPLING,
    }
    errors = script.runtime_contract_errors(
        "streammuse_v2_prompt_continuation",
        runtime,
        code=code,
        prompt_checkpoint=prompt_identity,
        continuation_checkpoint=continuation_identity,
        prompt_selection_mode="single",
    )

    assert errors == []


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


def test_eval_manifest_csv_matches_toolkit_contract(
    matched_runner, tmp_path: Path
) -> None:
    script = matched_runner
    output_root = tmp_path / "portable_results"
    complete_a = output_root / "trials" / "a"
    complete_b = output_root / "trials" / "b"
    complete_a.mkdir(parents=True)
    complete_b.mkdir(parents=True)
    melody_hash = "e" * 64
    systems = script.SYSTEM_IDS
    trials = [
        {
            "piece_id": "piece-01",
            "seed": 0,
            "system_id": systems[0],
            "session_dir": str(complete_a.resolve()),
            "run_status": "complete",
            "melody_input_sha256": melody_hash,
            "failure_reason": "must be blanked for complete rows",
        },
        {
            "piece_id": "piece-01",
            "seed": 0,
            "system_id": systems[1],
            "session_dir": None,
            "run_status": "failed",
            "melody_input_sha256": melody_hash,
            "failure_reason": "model failure",
        },
        {
            "piece_id": "piece-01",
            "seed": 1,
            "system_id": systems[0],
            "session_dir": None,
            "run_status": "failed",
            "melody_input_sha256": melody_hash,
            "failure_reason": "timeout",
        },
        {
            "piece_id": "piece-01",
            "seed": 1,
            "system_id": systems[1],
            "session_dir": str(complete_b.resolve()),
            "run_status": "complete",
            "melody_input_sha256": melody_hash,
            "failure_reason": None,
        },
    ]
    manifest_path = output_root / "eval_manifest.csv"

    script.write_eval_manifest(manifest_path, trials)

    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        assert tuple(reader.fieldnames or ()) == script.EVAL_MANIFEST_FIELDS
    assert len(rows) == 4
    assert {row["run_status"] for row in rows} == {"complete", "failed"}
    assert all(
        row["failure_reason"] == ""
        for row in rows
        if row["run_status"] == "complete"
    )
    assert all(
        row["failure_reason"]
        for row in rows
        if row["run_status"] == "failed"
    )
    assert all(
        not Path(row["session_dir"]).is_absolute()
        for row in rows
        if row["session_dir"]
    )
    assert not (output_root / ".eval_manifest.csv.tmp").exists()

    evaluator = (
        Path(__file__).resolve().parents[4].parent
        / "eval-matched-system-v2"
        / "src"
        / "eval_toolkit"
        / "system_trace_v2.py"
    )
    if evaluator.is_file():
        module_name = "_streammuse_test_eval_system_trace_v2"
        spec = importlib.util.spec_from_file_location(module_name, evaluator)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
            parsed = module.load_manifest(
                manifest_path, expected_piece_count=1, expected_seed_count=2
            )
        finally:
            sys.modules.pop(module_name, None)
        assert len(parsed) == 4


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
    assert result["evaluation_contract"]["checkpoint_conditioning"] == {
        "time_signature": "4/4",
        "continuation_time_signature_index": 0,
        "prompt_time_signature_index": 0,
    }
    assert result["evaluation_contract"]["streammuse_v2_prompt"] == {
        "selection_mode": "rule_s",
        "candidate_count": 5,
        "prompt_length_ticks": 32,
    }
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
    with (output / "eval_manifest.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        eval_rows = list(csv.DictReader(handle))
    assert len(eval_rows) == 4
    assert {row["run_status"] for row in eval_rows} == {"missing"}
    assert all(row["failure_reason"] == "dry_run_not_executed" for row in eval_rows)


def test_single_prompt_selection_manifest_records_effective_n1(
    matched_runner, tmp_path: Path
) -> None:
    script = matched_runner
    prompt = tmp_path / "prompt.safetensors"
    continuation = tmp_path / "continuation.safetensors"
    midi = tmp_path / "piece.mid"
    prompt.write_bytes(b"prompt")
    continuation.write_bytes(b"continuation")
    midi.write_bytes(b"MThd-piece")
    cohort = tmp_path / "cohort.json"
    cohort.write_text(
        json.dumps(
            {"pieces": [{"piece_id": "piece", "midi_path": "piece.mid"}]}
        ),
        encoding="utf-8",
    )
    args = script.parse_args(
        [
            "--cohort-manifest",
            str(cohort),
            "--output-root",
            str(tmp_path / "output"),
            "--prompt-checkpoint",
            str(prompt),
            "--continuation-checkpoint",
            str(continuation),
            "--systems",
            "streammuse_v2_prompt_continuation",
            "--seeds",
            "0",
            "--prompt-selection-mode",
            "single",
            "--dry-run",
        ]
    )

    result = script.run_evaluation(args)

    assert result["evaluation_contract"]["streammuse_v2_prompt"] == {
        "selection_mode": "single",
        "candidate_count": 1,
        "prompt_length_ticks": 32,
    }
