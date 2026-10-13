"""Compare frozen fixed-tick runs with the unmodified ordinary MIDI-file CLI."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
from types import SimpleNamespace
from urllib.parse import urlsplit

import plain_lekai_device_replay_acceptance as audit


CHECKPOINT_SHA = "d93139044a8614aeb66c58b5696371a575389199fbe64653b8994d3f6b056271"
FIELDS = ("part0_roll_sha256", "part0_tokens", "prompt_tokens", "raw_tokens", "structural_tokens")


def matching_rows(path, session_id):
    with path.open(encoding="utf-8") as stream:
        return [row for line in stream if line.strip()
                if (row := json.loads(line))["session_id"] == session_id]


def load_run(out, server, seed, fixed):
    status = audit.read(out / "status.json")
    assert status["status"] == "completed"
    reset = audit.read(out / "session_initialized.json")
    assert reset["effective_seed"] == seed
    calls = matching_rows(server / "backend_calls.jsonl", reset["session_id"])
    assert calls and all(c["effective_seed"] == seed for c in calls)
    mapped = {c["request"]["generation_start_tick"]: c for c in calls}
    assert len(mapped) == len(calls), "Repeated generation tick needs separate diagnosis"
    samples = matching_rows(server / "sampling_observer.jsonl", reset["session_id"])
    assert samples
    sessions = list((out / "logs").glob("*/session_*"))
    assert len(sessions) == 1
    session = sessions[0]
    if fixed:
        delivered = audit.read(out / "delivered_events.json")
        input_events = [{k: e[k] for k in ("tick", "event_type", "pitch", "velocity", "channel")}
                        for e in delivered]
    else:
        trace = audit.rows(session / "input_quantization_trace.jsonl")
        assert trace
        input_events = [{"tick": e["quantized_tick"], **{k: e[k] for k in
                         ("event_type", "pitch", "velocity", "channel")}} for e in trace]
    return {
        "calls": mapped,
        "samples": [{k: r[k] for k in ("prompt_tokens", "sampling", "samples", "returned_tokens")}
                    for r in samples],
        "input_events": input_events,
        "playback": audit.midi_notes(session / "combined.mid"),
        "session": str(session),
    }


def compare_runs(a, b):
    common = sorted(set(a["calls"]) & set(b["calls"]))
    only_a = sorted(set(a["calls"]) - set(b["calls"]))
    only_b = sorted(set(b["calls"]) - set(a["calls"]))
    mismatches = {k: [] for k in (*FIELDS, "input_events", "raw_output")}
    examples = {}
    for tick in common:
        left, right = a["calls"][tick], b["calls"][tick]
        for field in mismatches:
            if field in FIELDS:
                x, y = left["trace"][field], right["trace"][field]
            elif field == "input_events":
                x, y = left["request"]["melody_events"], right["request"]["melody_events"]
            else:
                x, y = left["raw_output"], right["raw_output"]
            if x != y:
                mismatches[field].append(tick)
                examples.setdefault(field, {"generation_start_tick": tick, "left": x, "right": y})
    timing_fields = ("tick", "event_type", "pitch", "channel")
    # This diagnostic excludes velocity explicitly; the full comparison above does not.
    a_timing = [{k: e[k] for k in timing_fields} for e in a["input_events"]]
    b_timing = [{k: e[k] for k in timing_fields} for e in b["input_events"]]
    result = {
        "calls": [len(a["calls"]), len(b["calls"])], "common_calls": len(common),
        "only_left_generation_ticks": only_a, "only_right_generation_ticks": only_b,
        "input_event_counts": [len(a["input_events"]), len(b["input_events"])],
        "delivered_events_exact": a["input_events"] == b["input_events"],
        "delivered_timing_pitch_channel_order_exact": a_timing == b_timing,
        "mismatched_generation_ticks": mismatches,
        "equal_call_counts": {k: len(common) - len(v) for k, v in mismatches.items()},
        "model_inputs_exact": not only_a and not only_b and not mismatches["prompt_tokens"],
        "raw_tokens_exact": not only_a and not only_b and not mismatches["raw_tokens"]
                            and not mismatches["structural_tokens"],
        "raw_output_exact": not only_a and not only_b and not mismatches["raw_output"],
        "sampler_exact": a["samples"] == b["samples"],
        "raw_note_on": [sum(e["type"] == "note_on" for c in r["calls"].values()
                            for e in c["raw_output"]) for r in (a, b)],
        "playback_exact": a["playback"] == b["playback"],
        "playback_notes": [len(a["playback"]), len(b["playback"])],
        "common_playback_notes": sum((Counter(a["playback"]) & Counter(b["playback"])).values()),
        "first_difference_examples": examples,
    }
    result["strict_exact"] = not only_a and not only_b and not any(mismatches.values())
    return result


def fixed_locations(reference, piece):
    if piece == "01_tong_hua":
        base = reference / "pilot_01_tong_hua"
        return base / piece, base / "server"
    return reference / "cases" / piece, reference / "server"


def validate_reference(reference):
    progress = audit.read(reference / "progress.json")
    assert progress["status"] in ("completed", "completed_with_differences")
    plan = audit.read(reference / "plan.json")
    assert len(plan["cases"]) == 10
    assert (plan["playback_bpm"], plan["model_bpm"], plan["temperature"],
            plan["top_p"], plan["top_k"], plan["repetition_penalty"]) == (90, 80, 1.1, .95, 50, 1)
    for case in plan["cases"]:
        piece = case["piece"]
        assert audit.sha((reference / "inputs" / piece / "melody.mid").read_bytes()) == case["midi_sha256"]
        root, server = fixed_locations(reference, piece)
        for repeat in (1, 2):
            cfg = audit.read(root / f"repeat_{repeat}" / "configuration.json")
            assert cfg["input"]["type"] == "list" and cfg["count_in_beats"] == 4
            assert cfg["tempo"] == {"bpm": 90, "ticks_per_beat": 4, "beats_per_bar": 4}
            assert cfg["input_snap_forward_fraction"] == .4
            for key, value in {"model_name": "lekai", "model_condition_bpm": 80,
                               "temperature": 1.1, "top_p": .95, "top_k": 50,
                               "repetition_penalty": 1, "generation_interval_ticks": 4,
                               "generation_length_frames": 4}.items():
                assert cfg["inference"][key] == value
            assert audit.read(root / f"repeat_{repeat}" / "status.json")["status"] == "completed"
        assert audit.read(server / "identity.json")["base"].startswith(plan["system_ref"])
    return plan


def run_clients(args):
    plan = audit.read(args.root / "plan.json")
    progress = {"status": "running", "pieces": [], "no_retries": True}
    audit.save(args.root / "progress.json", progress)
    for case in plan["cases"]:
        piece = case["piece"]
        entry = {"piece": piece, "status": "running", "repeats_completed": 0}
        progress["pieces"].append(entry)
        audit.save(args.root / "progress.json", progress)
        try:
            own = args.root / "cases" / piece
            own.mkdir(parents=True, exist_ok=False)
            client_args = SimpleNamespace(out=own, server=args.server, seed=plan["seed"],
                                          stop_tick=case["stop_tick"], timed_input=None)
            midi = args.root / "inputs" / piece / "melody.mid"
            assert audit.sha(midi.read_bytes()) == case["midi_sha256"]
            for repeat in (1, 2):
                print(f"START {piece} ordinary midi_file repeat {repeat}", flush=True)
                audit.one_client(client_args, f"midi_file_repeat_{repeat}", midi=midi)
                entry["repeats_completed"] += 1
                audit.save(args.root / "progress.json", progress)
            fixed_root, fixed_server = fixed_locations(args.reference, piece)
            ordinary = [load_run(own / f"midi_file_repeat_{r}", args.root / "server", plan["seed"], False)
                        for r in (1, 2)]
            results = {}
            for r in (1, 2):
                baseline = load_run(fixed_root / f"repeat_{r}", fixed_server, plan["seed"], True)
                results[f"fixed_vs_midi_repeat_{r}"] = compare_runs(baseline, ordinary[r - 1])
            results["midi_repeat_1_vs_2"] = compare_runs(*ordinary)
            audit.save(own / "comparison.json", results)
            entry.update(status="completed", comparisons={name: {k: value[k] for k in (
                "calls", "input_event_counts", "delivered_events_exact", "model_inputs_exact",
                "raw_tokens_exact", "raw_output_exact", "raw_note_on", "playback_exact", "playback_notes")}
                for name, value in results.items()})
            print(json.dumps(entry), flush=True)
        except BaseException as exc:
            entry.update(status="failed", error=repr(exc))
            progress["status"] = "failed"
            audit.save(args.root / "progress.json", progress)
            raise
        audit.save(args.root / "progress.json", progress)
    progress["status"] = "completed"
    audit.save(args.root / "progress.json", progress)


def launch(args):
    plan = validate_reference(args.reference)
    if subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip():
        raise RuntimeError("GPU is in use")
    assert audit.sha(Path(args.checkpoint).read_bytes()) == CHECKPOINT_SHA
    url = urlsplit(args.server)
    assert url.hostname == "127.0.0.1"
    with socket.socket() as check:
        check.bind((url.hostname, url.port))
    identity = audit.identity(plan["system_ref"])
    args.root.mkdir(parents=True, exist_ok=False)
    shutil.copytree(args.reference / "inputs", args.root / "inputs")
    audit.save(args.root / "plan.json", plan)
    audit.save(args.root / "identity.json", identity)
    audit.save(args.root / "comparison_contract.json", {
        "fixed_reference_root": str(args.reference), "scope": "fixed event adapter versus ordinary MIDI-file CLI",
        "historical_live_equality_claim": False, "seed": plan["seed"],
        "new_runs": 20, "frozen_fixed_runs_reused": 20,
        "input_quantization_trace_enabled_for_midi_file": True,
        "fixed_baseline_input_trace_enabled": False,
        "production_source_changed": False, "pause_clock_for_inference": False,
        "mismatch_policy": "retain differences and continue; process failure stops; no retries",
    })
    snapshot = args.root / "executed_scripts"
    snapshot.mkdir()
    for name in (Path(__file__).name, "plain_lekai_device_replay_acceptance.py", "device_replay_acceptance.py"):
        shutil.copy2(Path(__file__).with_name(name), snapshot / name)
    command = [sys.executable, "-u", str(Path(__file__).with_name("plain_lekai_device_replay_acceptance.py")),
               "server", "--out", str(args.root / "server"), "--gpu", "0", "--host", url.hostname,
               "--port", str(url.port), "--checkpoint-path", args.checkpoint,
               "--seed", str(plan["seed"]), "--system-ref", plan["system_ref"]]
    with (args.root / "server.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 180
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError("Backend exited")
                try:
                    info = audit.api(args.server, "/runtime_info")
                    assert info["has_real_model"]
                    break
                except OSError:
                    time.sleep(1)
            else:
                raise TimeoutError("Backend startup timeout")
            # Verify the retained and new model environments, excluding only run-specific paths/ports.
            old = audit.read(args.reference / "server/environment.json")
            new = audit.read(args.root / "server/environment.json")
            ignored = {"LEKAI_RT_LOG_DIR", "LEKAI_SERVER_PORT"}
            assert {k: v for k, v in old.items() if k not in ignored} == {k: v for k, v in new.items() if k not in ignored}
            assert audit.read(args.reference / "server/libraries.json") == audit.read(args.root / "server/libraries.json")
            print("Real backend ready; environment and libraries match fixed-tick baseline", flush=True)
            run_clients(args)
        finally:
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--server", default="http://127.0.0.1:18848")
    parser.add_argument("--checkpoint", default="/home/xiaosongma/models/streammuse_prompt_continuation/continuation/model.safetensors")
    launch(parser.parse_args())
