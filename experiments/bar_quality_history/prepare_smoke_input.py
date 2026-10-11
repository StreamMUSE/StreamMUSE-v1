"""Prepare a closed-note MIDI snapshot for a fixed-bar functional smoke test."""
import argparse
import hashlib
import json
from pathlib import Path

import mido


def notes(path, track):
    midi = mido.MidiFile(path)
    active, result, tick = {}, [], 0
    for event in midi.tracks[track]:
        tick += event.time
        if event.type not in ("note_on", "note_off"):
            continue
        if tick * 4 % midi.ticks_per_beat:
            raise ValueError("MIDI is not exactly on the 4-step grid")
        step = tick * 4 // midi.ticks_per_beat
        key = event.channel, event.note
        if event.type == "note_on" and event.velocity > 0:
            if key in active:
                raise ValueError("same-pitch overlap not supported")
            active[key] = step
        else:
            onset = active.pop(key)
            if onset >= step:
                raise ValueError("nonpositive note duration")
            result.append({"onset": onset, "end": step, "pitch": event.note})
    if active:
        raise ValueError("unclosed MIDI notes")
    return sorted(result, key=lambda n: (n["onset"], n["pitch"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--melody", type=Path, required=True)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--melody-track", type=int, required=True)
    parser.add_argument("--history-track", type=int, required=True)
    parser.add_argument("--bar-start", type=int, required=True)
    parser.add_argument("--bar-length", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.bar_start < 0 or args.bar_start % 4 or args.bar_length <= 0 or args.bar_length % 4:
        raise ValueError("explicit whole-beat boundaries required")
    mel = notes(args.melody, args.melody_track)
    acc = notes(args.history, args.history_track)
    end = args.bar_start + args.bar_length
    if any(n["onset"] < end < n["end"] for n in mel + acc):
        raise ValueError("notes active at snapshot boundary; select a supported bar")
    if any(n["onset"] < args.bar_start < n["end"] for n in acc):
        raise ValueError("accompaniment crosses repair start")
    doc = {
        "steps_per_beat": 4, "bar_start_step": args.bar_start,
        "bar_length_steps": args.bar_length, "observed_until": end,
        "melody_notes": [n for n in mel if n["end"] <= end],
        "history_notes": [n for n in acc if n["end"] <= end],
        "played_notes": [], "policy": "repair_history", "repair_fallback": "error",
        "decisions": [{"bar_start_step": args.bar_start, "boundary_step": end,
                       "decision": "reject", "source": "fixed functional intervention, not a quality label"}],
    }
    provenance = {
        "sources": [{"path": str(p.resolve()), "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "track": t}
                    for p, t in ((args.melody, args.melody_track), (args.history, args.history_track))],
        "selection": "fixed bar with closed notes, not selected by quality",
        "playback": "not supplied or evaluated; played_notes is an empty placeholder",
        "duration_cutoff": "no future note endpoints supplied to probe; no duration clipping",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("input.json", "input_provenance.json"):
        if (args.output_dir / name).exists():
            raise FileExistsError(name)
    for name, value in (("input.json", doc), ("input_provenance.json", provenance)):
        with (args.output_dir / name).open("x", encoding="utf-8") as file:
            json.dump(value, file, ensure_ascii=False, indent=2)
    print(json.dumps({"bar": [args.bar_start, end], "melody_notes": len(doc["melody_notes"]),
                      "history_notes": len(doc["history_notes"])}))


if __name__ == "__main__":
    main()
