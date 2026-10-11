"""Audit paired real-model integration logs; never interpret budgets as ISR."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    root = args.directory
    comparisons = []
    for seed, condition in [(0, "repair"), (1, "repair"), (2, "repair"), (0, "identity_control")]:
        baseline = json.loads((root / f"baseline_s{seed}.json").read_text())
        trial = json.loads((root / f"{condition}_s{seed}.json").read_text())
        assert baseline["runtime"]["mode"] == trial["runtime"]["mode"] == "real_model"
        assert baseline["runtime"]["checkpoint_path"] == trial["runtime"]["checkpoint_path"]
        rows = []
        for a, b in zip(baseline["rows"], trial["rows"], strict=True):
            assert a["start_tick"] == b["start_tick"]
            rows.append({"tick": a["start_tick"],
                         "melody_tokens_equal": a["metadata"]["part0_tokens"] == b["metadata"]["part0_tokens"],
                         "melody_roll_equal": a["metadata"]["part0_roll_digest"] == b["metadata"]["part0_roll_digest"],
                         "full_input_equal": a["input_tokens"] == b["input_tokens"],
                         "raw_tokens_equal": a["metadata"]["raw_tokens"] == b["metadata"]["raw_tokens"],
                         "returned_events_equal": a["events"] == b["events"]})
        entry = {"seed": seed, "condition": condition, "requests": len(rows),
                 "applied": len(trial["applied_repairs"]), "by_tick": rows}
        for key in ("melody_tokens_equal", "melody_roll_equal", "full_input_equal",
                    "raw_tokens_equal", "returned_events_equal"):
            entry[key] = sum(r[key] for r in rows)
            entry[f"first_{key}_difference"] = next((r["tick"] for r in rows if not r[key]), None)
        completed = [r for r in trial["controller_audit"] if r["status"] == "completed"]
        if completed and "provenance" in completed[0]["candidate"]:
            result = completed[0]["candidate"]["provenance"]
            if "elapsed_seconds" in result:
                entry["repair_seconds"] = result["elapsed_seconds"]
                entry["candidate_initial_carry"] = result["metadata"]["token_decode_initial_active_pitches"]
                entry["decision"] = completed[0]["result"]
        comparisons.append(entry)
    failures = []
    for c in comparisons:
        if c["melody_tokens_equal"] != c["requests"] or c["melody_roll_equal"] != c["requests"]:
            failures.append("Melody changed")
        if (not c["applied"] or c["condition"] == "identity_control") and any(
                c[k] != c["requests"] for k in ("full_input_equal", "raw_tokens_equal", "returned_events_equal")):
            failures.append(f"unchanged-history path differs: {c['condition']} seed {c['seed']}")
        if any(not r[k] for r in c["by_tick"] if r["tick"] < 52
               for k in ("full_input_equal", "raw_tokens_equal", "returned_events_equal")):
            failures.append("pre-repair output differs")
    report = {"scope": "paired paced backend integration, not actual playback ISR or music quality",
              "failures": failures, "comparisons": comparisons}
    with (root / "audit_comparison.json").open("x", encoding="utf-8") as output:
        json.dump(report, output, ensure_ascii=False, indent=2)
    print(json.dumps({"failures": failures, "comparisons": [
        {k: v for k, v in c.items() if k != "by_tick"} for c in comparisons]}, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
