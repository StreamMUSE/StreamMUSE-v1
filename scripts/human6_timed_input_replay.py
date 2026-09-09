"""Prepare and run six preserved human arrival traces through ordinary Realtime."""

from __future__ import annotations

import argparse
from collections import Counter
import contextlib
import io
import json
import math
from pathlib import Path
import shutil
from types import SimpleNamespace

import plain_lekai_device_replay_acceptance as audit


CASES = [
    ("01_tong_hua", "today_20260906", "session_170902"),
    ("02_night_dancer", "today_20260906", "session_171130"),
    ("03_qun_qing", "today_20260906", "session_171416"),
    ("04_lemon", "today_20260906", "session_171734"),
    ("05_cruel_angel", "today_20260906", "session_172010"),
    ("10_castle_in_the_sky", "yesterday_20260905", "session_181017"),
]


def prepare(args):
    args.root.mkdir(parents=True, exist_ok=False)
    cases = []
    for piece, date, session in CASES:
        source = args.source_root / date / session
        trace = source / "input_quantization_trace.jsonl"
        records, _ = audit.timed_messages(trace)
        cfg = audit.read(source / "session_config.json")
        assert cfg["tempo_bpm"] == 90 and cfg["model_condition_bpm"] == 80
        assert cfg["count_in_beats"] == 4
        dest = args.root / "inputs" / piece
        dest.mkdir(parents=True)
        names = ["input_quantization_trace.jsonl", "session_config.json", "prompt_continuation_replay_melody.mid"]
        for name in names:
            shutil.copy2(source / name, dest / name)
        active, orphan_offs, retriggers = set(), [], []
        for r in records:
            key = (r["channel"], r["pitch"])
            if r["event_type"] == "note_on" and r["velocity"] > 0:
                if key in active:
                    retriggers.append(r["event_sequence"])
                active.add(key)
            else:
                if key not in active:
                    orphan_offs.append(r["event_sequence"])
                active.discard(key)
        cases.append({
            "piece": piece, "source_date": date, "source_session": session,
            "trace": (Path("inputs") / piece / names[0]).as_posix(),
            "source_directory": str(source), "source_sha256": {name: audit.sha((source / name).read_bytes()) for name in names},
            "event_count": len(records), "first_seconds": records[0]["elapsed_seconds"],
            "last_seconds": records[-1]["elapsed_seconds"],
            "stop_tick": math.ceil(records[-1]["elapsed_seconds"] * 6 / 4) * 4 + 16,
            "orphan_note_off_sequences": orphan_offs, "retrigger_sequences": retriggers,
            "active_pitches_at_trace_end": sorted(active),
        })
    audit.save(args.root / "plan.json", {
        "system_ref": "ec843a79", "condition": "plain_lekai_standard_no_prompt_no_constraints",
        "seed": 1051154023138951872, "playback_bpm": 90, "model_bpm": 80,
        "temperature": 1.1, "top_p": 0.95, "top_k": 50, "repetition_penalty": 1,
        "stages": ["A_recorded_application_arrival_via_virtual_MIDI", "B_exported_Melody_MIDI_replay"],
        "trim_leading_silence": False, "rescale_input_time": False,
        "source_clock": "application received time, not physical key-down time",
        "intent": "New-system reprocessing of old human inputs, not reproduction of old accompaniment",
        "stop_policy": "Next beat boundary after last input event, plus four beats of playback tail",
        "cases": cases,
    })
    print(json.dumps(cases, indent=2))


def run(args):
    plan = audit.read(args.root / "plan.json")
    cases = plan["cases"]
    if args.pieces:
        unknown = set(args.pieces) - {c["piece"] for c in cases}
        if unknown:
            raise ValueError(f"Unknown pieces: {sorted(unknown)}")
        cases = [c for c in cases if c["piece"] in args.pieces]
    audit.save(args.root / "run_contract.json", {
        "server": args.server, "input_only": args.input_only,
        "pieces": [c["piece"] for c in cases],
        "source_plan_sha256": audit.sha((args.root / "plan.json").read_bytes()),
        "identity": audit.identity(plan["system_ref"]),
        "scope": "Input delivery diagnostic with real inference and sampling observers; not a latency benchmark",
    })
    progress = {"status": "running", "condition": plan["condition"], "cases": []}
    audit.save(args.root / "progress.json", progress)
    for case in cases:
        item = {"piece": case["piece"], "status": "running"}
        progress["cases"].append(item)
        audit.save(args.root / "progress.json", progress)
        print(f"START {case['piece']}", flush=True)
        try:
            trace = args.root / case["trace"]
            assert audit.sha(trace.read_bytes()) == case["source_sha256"]["input_quantization_trace.jsonl"]
            client_args = SimpleNamespace(
                out=args.root / "cases" / case["piece"] / "client", server=args.server,
                system_ref=plan["system_ref"], seed=plan["seed"], timed_input=trace,
                stop_tick=case["stop_tick"], skip_repeat=True,
            )
            if args.input_only:
                audit.one_client(client_args, "A_virtual_device")
            else:
                audit.clients(client_args)
            item["status"] = "completed"
        except KeyboardInterrupt:
            item["status"] = "interrupted"
            progress["status"] = "interrupted"
            audit.save(args.root / "progress.json", progress)
            raise
        except Exception as exc:
            item.update(status="failed", error=repr(exc))
            print(f"FAILED {case['piece']}: {exc!r}", flush=True)
        audit.save(args.root / "progress.json", progress)
    progress["status"] = "completed" if all(c["status"] == "completed" for c in progress["cases"]) else "failed"
    audit.save(args.root / "progress.json", progress)
    print(json.dumps(progress), flush=True)


def percentile(values, fraction):
    if not values:
        return None
    values = sorted(values)
    pos = (len(values) - 1) * fraction
    low, high = math.floor(pos), math.ceil(pos)
    return values[low] + (values[high] - values[low]) * (pos - low)


def delivery_check(source, actual, sent):
    expected = [(r["event_type"], r["pitch"], r["velocity"]) for r in source]
    received = [(r["event_type"], r["pitch"], r["velocity"]) for r in actual]
    check = {"source_events": len(source), "received_events": len(actual),
             "sequence_content_exact": expected == received, "sent_events": len(sent)}
    if expected != received:
        return check
    if len(sent) == len(source):
        send_errors = [(b["actual_relative_seconds"] - a["elapsed_seconds"]) * 1000
                       for a, b in zip(source, sent)]
        receive_after_send = [(a["elapsed_seconds"] - b["actual_relative_seconds"]) * 1000
                              for a, b in zip(actual, sent)]
        check.update({
            "send_lateness_ms_median": percentile(send_errors, 0.5),
            "send_lateness_ms_p95": percentile(send_errors, 0.95),
            "send_lateness_ms_max": max(send_errors),
            "receive_after_send_ms_median": percentile(receive_after_send, 0.5),
            "receive_after_send_ms_p95": percentile(receive_after_send, 0.95),
            "receive_after_send_ms_max": max(receive_after_send),
        })
    errors_ms = [(b["elapsed_seconds"] - a["elapsed_seconds"]) * 1000 for a, b in zip(source, actual)]
    tick_differences = [b["quantized_tick"] - a["quantized_tick"] for a, b in zip(source, actual)]
    check.update({
        "arrival_error_ms_median": percentile(errors_ms, 0.5),
        "arrival_abs_error_ms_p95": percentile([abs(v) for v in errors_ms], 0.95),
        "arrival_abs_error_ms_max": max(abs(v) for v in errors_ms),
        "quantized_tick_exact_events": tick_differences.count(0),
        "quantized_tick_difference_histogram": dict(Counter(tick_differences)),
        "different_tick_sequences": [a["event_sequence"] for a, delta in zip(source, tick_differences) if delta],
    })
    return check


def check_inputs(args):
    plan = audit.read(args.root / "plan.json")
    results = []
    for case in plan["cases"]:
        root = args.root / "cases" / case["piece"]
        a = root / "client/A_virtual_device"
        if not (a / "status.json").exists():
            continue
        session = next((a / "logs").glob("*/session_*"))
        check = delivery_check(audit.rows(args.root / case["trace"]),
            audit.rows(session / "input_quantization_trace.jsonl"), audit.rows(a / "sent_messages.jsonl"))
        audit.save(root / "input_delivery_check.json", check)
        results.append({"piece": case["piece"], **check})
    print(json.dumps(results, indent=2))


def analyze(args):
    plan = audit.read(args.root / "plan.json")
    results = []
    for case in plan["cases"]:
        root = args.root / "cases" / case["piece"]
        if not (root / "client/status.json").exists():
            results.append({"piece": case["piece"], "status": "incomplete"})
            continue
        with contextlib.redirect_stdout(io.StringIO()):
            audit.analyze(SimpleNamespace(out=root, backend_dir=args.root / "server"))
        comparison = audit.read(root / "analysis.json")
        a = root / "client/A_virtual_device"
        session = next((a / "logs").glob("*/session_*"))
        check = delivery_check(audit.rows(args.root / case["trace"]),
            audit.rows(session / "input_quantization_trace.jsonl"), audit.rows(a / "sent_messages.jsonl"))
        audit.save(root / "input_delivery_check.json", check)
        result = {"piece": case["piece"], "status": "completed", "input_delivery": check,
                  "counts": comparison["counts"], "comparison": comparison["comparisons"]["A_vs_B"]}
        results.append(result)
        print(json.dumps({"piece": case["piece"], "input_delivery": check,
            "equal_model_calls": result["comparison"]["equal_call_counts"],
            "sampler_exact": result["comparison"]["sampler_exact"],
            "playback_exact": result["comparison"]["playback"]["exact"]}), flush=True)
    audit.save(args.root / "summary.json", results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["prepare", "run", "analyze", "check-inputs"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--server", default="http://10.127.30.168:18845")
    parser.add_argument("--pieces", nargs="+")
    parser.add_argument("--input-only", action="store_true")
    args = parser.parse_args()
    {"prepare": prepare, "run": run, "analyze": analyze, "check-inputs": check_inputs}[args.mode](args)
