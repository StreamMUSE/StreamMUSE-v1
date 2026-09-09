"""P+C-only retest on frozen formal inputs, using the normal MIDI-file CLI."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil
import subprocess
import time

import run_formal_user10_test40 as formal
import run_matched_system_eval as matched

BASE = "69ff9fa0"
CONDITION = "pc_rule_if_else_n10"
PRODUCTION_FILES = {
    "src/streammuse/application/services/prompt_continuation_realtime_service.py",
    "src/streammuse/infrastructure/inference/lekai_prompt_continuation/scheduler.py",
}


def source_identity():
    code = matched.code_identity()
    assert code["git_clean"], code["git_status_porcelain"]
    changed = subprocess.check_output(
        ["git", "diff", "--name-only", BASE, "HEAD", "--", "src", "transformers"],
        cwd=matched.REPO_ROOT, text=True,
    ).splitlines()
    assert set(changed) == PRODUCTION_FILES, changed
    return code


def prepare(args):
    assert not args.root.exists(), args.root
    baseline_path = args.baseline / "plan.json"
    baseline = formal.read(baseline_path)
    plan = copy.deepcopy(baseline)
    plan.update(
        modern_code=source_identity(), modern_system_ref=source_identity()["git_commit"],
        root=str(args.root), conditions=[CONDITION], expected_trials=150,
        baseline_plan=str(baseline_path),
        baseline_plan_sha256=matched.file_sha256(baseline_path),
        baseline_code=baseline["modern_code"],
    )
    plan.pop("raw_gate", None)
    plan.pop("legacy", None)
    plan.pop("legacy_code", None)
    plan["checkpoints"] = {k: baseline["checkpoints"][k] for k in ("prompt", "continuation")}
    plan["modern"]["boundary_order"] = "beat_tail_snapshot_after_existing_input_buffer"
    plan["modern"]["backend_visibility"] = "request_admission_boundary_and_unsent_event_indices"
    for name, cohort in plan["cohorts"].items():
        assert matched.file_sha256(Path(cohort["manifest"])) == cohort["manifest_sha256"]
        assert len(cohort["pieces"]) == (10 if name == "user10" else 40)
        for case in cohort["pieces"]:
            assert matched.file_sha256(Path(case["melody_midi"])) == case["melody_midi_sha256"]
    for checkpoint in plan["checkpoints"].values():
        assert matched.file_sha256(Path(checkpoint["path"])) == checkpoint["sha256"]
    formal.save(args.root / "plan.json", plan)
    shutil.copy2(matched.REPO_ROOT / "docs/pc_beat_tail_retest.md", args.root / "README.md")
    print(json.dumps({"root": str(args.root), "trials": 150, "code": plan["modern_code"]}))


def worker(args):
    assert args.gpu in (0, 1, 2)
    used = subprocess.check_output(
        ["nvidia-smi", "-i", str(args.gpu), "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        text=True,
    )
    assert int(used.strip()) < 500, f"GPU {args.gpu} is busy"
    plan = formal.read(args.root / "plan.json")
    assert source_identity()["git_commit"] == plan["modern_code"]["git_commit"]
    assert matched.file_sha256(Path(plan["baseline_plan"])) == plan["baseline_plan_sha256"]
    for checkpoint in plan["checkpoints"].values():
        assert matched.file_sha256(Path(checkpoint["path"])) == checkpoint["sha256"]
    if not args.smoke:
        gate = formal.read(args.root / "_smoke/acceptance.json")
        assert gate["accepted"] and gate["git_commit"] == plan["modern_code"]["git_commit"]
        for gpu in (0, 1, 2):
            assert formal.read(args.root / "_smoke/status" / f"gpu{gpu}.json")["status"] == "completed"
    root = args.root / "_smoke" if args.smoke else args.root
    status_path = root / "status" / f"gpu{args.gpu}.json"
    assert not status_path.exists(), status_path
    status = {"status": "running", "gpu": args.gpu, "seed": 0 if args.smoke else args.gpu,
              "smoke": args.smoke, "completed": [], "started_unix": time.time()}
    formal.save(status_path, status)
    handle = None
    try:
        for name in ("user10", "test40"):
            cohort = plan["cohorts"][name]
            assert matched.file_sha256(Path(cohort["manifest"])) == cohort["manifest_sha256"]
            cases = cohort["pieces"][args.gpu:args.gpu + 1] if args.smoke else cohort["pieces"]
            handle = matched.start_server(
                formal.IDS[CONDITION], output_root=root / "_servers" / name / f"gpu{args.gpu}",
                python_bin=str(formal.PYTHON), gpu=str(args.gpu), timeout_s=180,
                code=plan["modern_code"], prompt_checkpoint=plan["checkpoints"]["prompt"],
                continuation_checkpoint=plan["checkpoints"]["continuation"], time_signature_index=4,
                prompt_selection_mode="rule_s_if_else", prompt_batch_candidates=10,
                contract=formal.contract_for(cohort, CONDITION, 128 if args.smoke else cases[0]["stop_tick"]),
            )
            for case in cases:
                assert matched.file_sha256(Path(case["melody_midi"])) == case["melody_midi_sha256"]
                for repeat in (range(2) if args.smoke else range(1)):
                    seed = status["seed"]
                    stop_tick = min(case["stop_tick"], 128) if args.smoke else case["stop_tick"]
                    status["current"] = {"cohort": name, "condition": CONDITION, "piece": case["piece_id"],
                                         "repeat": repeat, "stop_tick": stop_tick}
                    formal.save(status_path, status)
                    print("START", status["current"], seed, flush=True)
                    out = root / "runs" / name / CONDITION / case["piece_id"] / f"seed{seed}"
                    if args.smoke:
                        out = out / f"repeat{repeat}"
                    record = matched.run_trial(
                        handle=handle,
                        piece=matched.CohortPiece(case["piece_id"], Path(case["melody_midi"]), case["melody_midi_sha256"]),
                        seed=seed, trial_dir=out, python_bin=str(formal.PYTHON),
                        timeout_s=stop_tick * 60 / cohort["playback_bpm"] / 4 + 180,
                        code=plan["modern_code"], prompt_checkpoint=plan["checkpoints"]["prompt"],
                        continuation_checkpoint=plan["checkpoints"]["continuation"], time_signature_index=4,
                        prompt_selection_mode="rule_s_if_else", prompt_batch_candidates=10,
                        contract=formal.contract_for(cohort, CONDITION, stop_tick),
                    )
                    assert record["run_status"] == "complete", record.get("failure_reason")
                    result = formal.validate_modern_artifacts(record, CONDITION)
                    result.update(status["current"], seed=seed)
                    if not args.smoke:
                        result["listening_midi"] = formal.publish_playback(root, name, CONDITION, case, seed, result)
                    status["completed"].append(result)
                    formal.save(status_path, status)
                    print("DONE", status["current"], "completed", len(status["completed"]), flush=True)
            matched.stop_server(handle)
            handle = None
        status.update(status="completed", finished_unix=time.time())
    except BaseException as exc:
        status.update(status="failed", error=repr(exc), finished_unix=time.time())
        raise
    finally:
        if handle is not None:
            matched.stop_server(handle)
        formal.save(status_path, status)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "worker"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--gpu", type=int, choices=[0, 1, 2])
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.action == "prepare" and args.baseline is None:
        parser.error("prepare requires --baseline")
    if args.action == "worker" and args.gpu is None:
        parser.error("worker requires --gpu")
    globals()[args.action](args)


if __name__ == "__main__":
    main()
