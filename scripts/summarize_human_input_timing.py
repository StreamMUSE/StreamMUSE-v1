"""Compare preserved application-arrival traces with same-host timed input runs."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import human6_timed_input_replay as cohort
from streammuse.application.services.input_timing import diagnose_input_quantization
from streammuse.domain.timing import Tempo


MISSING_CASES = [
    ("06_gao_bai_qi_qiu", "yesterday_20260905", "session_172945"),
    ("07_qing_hua_ci", "yesterday_20260905", "session_174319"),
    ("08_xiao_xing_yun", "yesterday_20260905", "session_175259"),
    ("09_yue_liang_dai_biao_wo_de_xin", "yesterday_20260905", "session_180149"),
]


def is_on(event):
    return event["event_type"] == "note_on" and event["velocity"] > 0


def compare(source, actual):
    signature = lambda r: (r["event_sequence"], r["event_type"], r["pitch"], r["velocity"], r["channel"])
    if list(map(signature, source)) != list(map(signature, actual)):
        raise ValueError("Event sequence/content mismatch: do not pair events by approximate time")
    changes, deltas, errors, active, notes, orphans = [], [], [], {}, [], []
    changed_types, totals = Counter(), Counter()
    for i, (a, b) in enumerate(zip(source, actual)):
        for field in ("bpm", "ticks_per_beat", "snap_forward_fraction"):
            if a[field] != b[field]:
                raise ValueError(f"Quantization setting mismatch: {field}")
        tempo = Tempo(a["bpm"], a["ticks_per_beat"], 4)
        for r in (a, b):
            q = diagnose_input_quantization(r["elapsed_seconds"], tempo,
                                            snap_forward_fraction=r["snap_forward_fraction"])
            if q.quantized_tick != r["quantized_tick"]:
                raise ValueError("Stored quantized tick differs from the shared quantizer")
        delta = b["quantized_tick"] - a["quantized_tick"]
        error = (b["elapsed_seconds"] - a["elapsed_seconds"]) * 1000
        deltas.append(delta)
        errors.append(error)
        kind = "note_on" if is_on(a) else "note_off"
        totals[kind] += 1
        if delta:
            changed_types[kind] += 1
            changes.append({"sequence": a["event_sequence"], "type": kind, "pitch": a["pitch"],
                            "old_tick": a["quantized_tick"], "new_tick": b["quantized_tick"],
                            "delta_ticks": delta, "arrival_error_ms": error})
        key = (a["channel"], a["pitch"])
        if is_on(a):
            if key in active:
                raise ValueError("Overlapping same-pitch note_on requires an explicit pairing policy")
            active[key] = i
        elif key not in active:
            orphans.append(a["event_sequence"])
        else:
            j = active.pop(key)
            old_duration = a["quantized_tick"] - source[j]["quantized_tick"]
            new_duration = b["quantized_tick"] - actual[j]["quantized_tick"]
            if min(old_duration, new_duration) < 0:
                raise ValueError("Negative note duration")
            notes.append({"onset_changed": source[j]["quantized_tick"] != actual[j]["quantized_tick"],
                          "end_changed": bool(delta), "old_duration": old_duration,
                          "new_duration": new_duration})
    if active:
        raise ValueError("Trace ends with unpaired note_on")
    changed = sum(d != 0 for d in deltas)
    duration_deltas = [n["new_duration"] - n["old_duration"] for n in notes]
    summary = {
        "events": len(source), "event_content_exact": True,
        "changed_events": changed, "changed_events_percent": 100 * changed / len(source),
        "note_on_events": totals["note_on"], "changed_note_on": changed_types["note_on"],
        "changed_note_on_percent": 100 * changed_types["note_on"] / totals["note_on"],
        "note_off_events": totals["note_off"], "changed_note_off": changed_types["note_off"],
        "changed_note_off_percent": 100 * changed_types["note_off"] / totals["note_off"],
        "arrival_error_median_ms": cohort.percentile(errors, 0.5),
        "arrival_abs_error_p95_ms": cohort.percentile([abs(e) for e in errors], 0.95),
        "arrival_abs_error_max_ms": max(abs(e) for e in errors),
        "max_abs_tick_shift": max(abs(d) for d in deltas),
        "paired_notes": len(notes), "orphan_note_off_count": len(orphans),
        "changed_note_intervals": sum(n["onset_changed"] or n["end_changed"] for n in notes),
        "changed_durations": sum(d != 0 for d in duration_deltas),
        "max_abs_duration_shift_ticks": max(map(abs, duration_deltas), default=0),
        "source_zero_tick_notes": sum(n["old_duration"] == 0 for n in notes),
        "replay_zero_tick_notes": sum(n["new_duration"] == 0 for n in notes),
        "positive_to_zero_tick_notes": sum(n["old_duration"] > 0 and n["new_duration"] == 0 for n in notes),
        "zero_to_positive_tick_notes": sum(n["old_duration"] == 0 and n["new_duration"] > 0 for n in notes),
    }
    details = {"changes": changes, "tick_shift_histogram": dict(Counter(deltas)),
               "duration_shift_histogram": dict(Counter(duration_deltas)),
               "orphan_note_off_sequences": orphans}
    return summary, details


def main(args):
    summaries, details = [], {}
    for piece, date, session in sorted(cohort.CASES + MISSING_CASES):
        source_path = args.source_root / date / session / "input_quantization_trace.jsonl"
        row = {"piece": piece, "source_session": session}
        if not source_path.exists():
            summaries.append({**row, "status": "missing_original_arrival_trace"})
            continue
        completed = []
        for root in args.run_roots:
            client = root / "cases" / piece / "client/A_virtual_device"
            if (client / "status.json").exists():
                if cohort.audit.read(client / "status.json")["status"] == "completed":
                    completed.append((root, client))
        if len(completed) != 1:
            raise ValueError(f"Expected exactly one completed run for {piece}, found {len(completed)}")
        root, client = completed[0]
        traces = list((client / "logs").glob("*/session_*/input_quantization_trace.jsonl"))
        if len(traces) != 1:
            raise ValueError(f"Ambiguous input trace for {piece}")
        source, actual = cohort.audit.rows(source_path), cohort.audit.rows(traces[0])
        plan = cohort.audit.read(root / "plan.json")
        case = next(c for c in plan["cases"] if c["piece"] == piece)
        source_sha = cohort.audit.sha(source_path.read_bytes())
        if source_sha != case["source_sha256"]["input_quantization_trace.jsonl"]:
            raise ValueError("Source file differs from the run manifest")
        summary, detail = compare(source, actual)
        stored = cohort.audit.read(root / "cases" / piece / "input_delivery_check.json")
        assert summary["changed_events"] == stored["source_events"] - stored["quantized_tick_exact_events"]
        summaries.append({**row, "status": "measured", **summary})
        details[piece] = {"run_root": str(root), "source_sha256": source_sha,
                          "actual_sha256": cohort.audit.sha(traces[0].read_bytes()), **detail}
    args.out.mkdir(parents=True, exist_ok=True)
    cohort.audit.save(args.out / "summary.json", summaries)
    cohort.audit.save(args.out / "analysis/event_changes.json", details)
    fields = list(dict.fromkeys(k for row in summaries for k in row))
    with (args.out / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--run-roots", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    main(parser.parse_args())
