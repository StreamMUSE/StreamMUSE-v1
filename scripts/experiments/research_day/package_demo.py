"""Collect the verified 60 BPM story recording and its exact lyric trace."""
import json
from pathlib import Path
import shutil


def main():
    root = Path(__file__).resolve().parents[3]
    evidence = root / "output/research_day_20261105"
    source = evidence / "story_demo_60bpm/summary.json"
    summary = json.loads(source.read_text())
    if summary["status"] != "observed" or summary["audio_artifact"]["status"] != "copied":
        raise SystemExit("Only package a verified run with preserved audio")
    target = evidence / "demo"
    target.mkdir(parents=True, exist_ok=True)
    stem = "courier_map_story_60bpm"
    shutil.copyfile(summary["audio_artifact"]["path"], target / f"{stem}_mix.wav")
    lines = [f"{row['bar'] + 1:02d}. {row['text']}" for row in summary["chosen_lyrics"]]
    (target / f"{stem}_lyrics.txt").write_text("\n".join(lines) + "\n")
    (target / f"{stem}_evidence.json").write_text(json.dumps(summary, indent=2) + "\n")
    comparisons = []
    for name in ("story_prioritized", "story_demo_60bpm", "story_90bpm", "story_120bpm", "story_late_90bpm"):
        path = evidence / name / "summary.json"
        if not path.exists():
            continue
        run = json.loads(path.read_text())
        rows = run.get("chosen_lyrics", [])
        comparisons.append({
            "run": name, "status": run["status"], "bpm": run["bpm"],
            "bar_seconds": 240 / run["bpm"], "edit_phase": run.get("edit_phase", 0),
            "bars": len(rows), "remote_latest_bars": sum(x["choice"] == "remote_latest" for x in rows),
            "edits": [{"event": e["text"], "effective_from_bar": e["effective_from_bar"],
                       "first_new_context_bar": (e.get("first_playback_callback") or {}).get("bar"),
                       "request_to_new_context_ms": (e.get("first_playback_callback") or {}).get("request_to_callback_ms")}
                      for e in run["events"]],
        })
    (target / "steering_comparison.json").write_text(json.dumps({
        "measurement": "software playback callbacks, not acoustic or semantic correctness",
        "conditions": "short shared-GPU pilots; different model draws; edit phase is controlled",
        "runs": comparisons,
    }, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
