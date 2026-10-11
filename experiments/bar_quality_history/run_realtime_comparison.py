"""Reuse the production MIDI CLI, scheduler, HTTP server and playback exports."""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import run_matched_system_eval as matched
import run_formal_user10_test40 as formal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--gpu", choices=["1", "2", "3"], required=True)
    parser.add_argument('--repair-policy', choices=['strict', 'preserve_cross_boundary'], default='strict')
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    cohort = plan["cohorts"]["user10"]
    case = cohort["pieces"][0]
    args.root.mkdir(parents=True, exist_ok=False)
    code = matched.code_identity()
    # Uncommitted experimental work is explicitly recorded and source-hashed.
    paths = list((ROOT / "src/streammuse/infrastructure/inference").rglob("*.py"))
    paths += list((ROOT / "experiments/bar_quality_history").glob("*.py"))
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    matched.write_json(args.root / "code_snapshot.json", {"git": code, "sha256": hashes})
    checkpoints = plan["checkpoints"]
    for item in checkpoints.values():
        assert matched.file_sha256(Path(item["path"])) == item["sha256"]
    contract = formal.contract_for(cohort, "pc_rule_if_else_n10", case["stop_tick"])
    matched.write_json(args.root / "plan.json", {"case": case, "contract": asdict(contract),
                       "checkpoints": checkpoints, "seeds": [0, 1, 2], "bar": [64, 80],
                       "conditions": ["baseline", "repair"], "repair_policy": args.repair_policy,
                       "scope": "full-song production MIDI-file realtime"})
    config = {"mode": "real", "checkpoint_path": checkpoints["continuation"]["path"],
              "device": "cuda", "dtype": "float16", "bpm": cohort["model_bpm"],
              "context_beats": 32, **cohort["sampling"], "time_signature_index": 4,
              "tonal_constraint_enabled": True, "empty_token_guard_enabled": True}
    matched.write_json(args.root / "worker_config.json", config)
    system_id = "streammuse_v2_prompt_continuation"
    summary = []
    for condition in ("baseline", "repair"):
        directory = args.root / condition
        directory.mkdir()
        port = matched.free_port()
        env = matched.build_server_environment(system_id, port=port, gpu=args.gpu,
            server_dir=directory, code=code, prompt_checkpoint=checkpoints["prompt"],
            continuation_checkpoint=checkpoints["continuation"], time_signature_index=4,
            prompt_selection_mode="rule_s_if_else", prompt_batch_candidates=10, contract=contract)
        env.update(BAR_HISTORY_MODE=condition, BAR_HISTORY_CONFIG=str(args.root / "worker_config.json"),
                   BAR_HISTORY_REPAIR_POLICY=args.repair_policy,
                   BAR_HISTORY_WORKER_AUDIT=str(directory / "worker"), LEKAI_HISTORY_REPAIR_ENABLED="1",
                   STREAMMUSE_EVAL_RUNTIME_SNAPSHOT=str(directory / "backend_startup_snapshot.json"),
                   PYTHONPATH=os.pathsep.join([str(ROOT), str(ROOT / "src"), str(ROOT / "transformers/src")]))
        command = [sys.executable, "-u", "-m", "experiments.bar_quality_history.launch_repair_server"]
        log = (directory / "server.log").open("w")
        proc = subprocess.Popen(command, env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, text=True)
        handle = matched.ServerHandle(system_id, f"http://127.0.0.1:{port}", proc, log, env, {})
        try:
            handle.startup_runtime = matched.wait_for_server(handle, timeout_s=240, code=code,
                prompt_checkpoint=checkpoints["prompt"], continuation_checkpoint=checkpoints["continuation"],
                time_signature_index=4, prompt_selection_mode="rule_s_if_else", prompt_batch_candidates=10,
                contract=contract)
            matched.write_json(directory / "server_identity.json", {"command": command, "pid": proc.pid,
                               "url": handle.base_url, "environment": matched._environment_record(env),
                               "runtime": handle.startup_runtime})
            for seed in (0, 1, 2):
                print("START", condition, seed, flush=True)
                trial = directory / f"seed{seed}"
                record = matched.run_trial(handle=handle,
                    piece=matched.CohortPiece(case["piece_id"], Path(case["melody_midi"]), case["melody_midi_sha256"]),
                    seed=seed, trial_dir=trial, python_bin=sys.executable,
                    timeout_s=case["stop_tick"] / 4 * 60 / cohort["playback_bpm"] + 180,
                    code=code, prompt_checkpoint=checkpoints["prompt"], continuation_checkpoint=checkpoints["continuation"],
                    time_signature_index=4, prompt_selection_mode="rule_s_if_else", prompt_batch_candidates=10,
                    contract=contract)
                audit = matched.request_json(handle.base_url + "/debug/history_repair", timeout=30)
                matched.write_json(trial / "history_repair_audit.json", audit)
                assert record["run_status"] == "complete", record.get("failure_reason")
                artifacts = formal.validate_modern_artifacts(record, "pc_rule_if_else_n10")
                listening = args.root / "listening"
                listening.mkdir(exist_ok=True)
                shutil.copy2(artifacts["combined"], listening / f"{condition}_s{seed}_playback.mid")
                result = {"condition": condition, "seed": seed, "session_dir": record["session_dir"],
                          "applied": len(audit["applied"]), "decisions": [
                              {k: r[k] for k in ("status", "reason", "result", "error") if k in r}
                              for r in audit["controller"]]}
                summary.append(result)
                matched.write_json(args.root / "progress.json", summary)
                print("DONE", json.dumps(result), flush=True)
        finally:
            matched.stop_server(handle)
    matched.write_json(args.root / "complete.json", summary)


if __name__ == "__main__":
    main()
