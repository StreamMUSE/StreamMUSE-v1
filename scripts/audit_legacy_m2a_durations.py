"""Read-only duration audit of saved Legacy human10 requests and MIDI exports."""

import argparse
from collections import Counter, defaultdict, deque
import csv
from fractions import Fraction
import hashlib
import json
from pathlib import Path

import mido


def histogram(values):
    return dict(sorted(Counter(str(v) for v in values).items(), key=lambda item: Fraction(item[0])))


def midi_track_durations(path):
    midi = mido.MidiFile(path)
    result = []
    for track in midi.tracks:
        active = defaultdict(deque)
        durations, at, orphan, retriggers = [], 0, 0, 0
        for msg in track:
            at += msg.time
            if msg.type == "note_on" and msg.velocity:
                key = (msg.channel, msg.note)
                retriggers += bool(active[key])
                active[key].append(at)
            elif msg.type == "note_off" or (msg.type == "note_on" and not msg.velocity):
                key = (msg.channel, msg.note)
                if active[key]:
                    durations.append(Fraction((at - active[key].popleft()) * 4, midi.ticks_per_beat))
                else:
                    orphan += 1
        if durations:
            result.append({"track": track.name, "notes": len(durations), "durations_ticks": histogram(durations),
                           "unclosed": sum(map(len, active.values())), "orphan_off": orphan,
                           "overlapping_same_pitch_onsets": retriggers})
    return result


def main(args):
    args.out.mkdir(parents=True, exist_ok=False)
    raw = args.bundle / "_metadata/all_legacy_backend_raw/runs"
    paths = sorted(raw.rglob("inferences.json"))
    assert len(paths) == 30
    summaries, melody_all, acc_all, placeholder_all = [], [], [], []
    for path in paths:
        piece, seed_dir = path.relative_to(raw).parts[:2]
        seed = int(seed_dir.removeprefix("seed_"))
        records = json.loads(path.read_text(encoding="utf-8"))
        mel = [n["duration"] for r in records for n in r["request"]["melody_notes"] if n["pitch"] >= 0]
        acc = [n["duration"] for r in records for n in r["response"]["accompaniment"]
               if n["pitch"] >= 0 and not n.get("is_placeholder", False)]
        placeholders = [n["duration"] for r in records for n in r["response"]["accompaniment"]
                        if n["pitch"] < 0 or n.get("is_placeholder", False)]
        folders = [p for p in args.bundle.glob(f"{piece}_*") if p.is_dir() and not p.name.endswith("_raw")]
        assert len(folders) == 1
        midi = folders[0] / f"{7 + seed:02d}_legacy_s{seed}_full.mid"
        item = {"piece": piece, "seed": seed, "requests": len(records),
                "request_melody_note_records": len(mel), "request_melody_duration_ticks": histogram(mel),
                "response_acc_note_records": len(acc), "response_acc_duration_ticks": histogram(acc),
                "placeholder_duration_ticks": histogram(placeholders),
                "source_log": str(path), "source_log_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "output_midi": str(midi), "output_tracks": midi_track_durations(midi)}
        summaries.append(item)
        melody_all.extend(mel)
        acc_all.extend(acc)
        placeholder_all.extend(placeholders)
    report = {"scope": "Existing human10 Legacy I=4 GL=7, ten pieces x three seeds; no new inference",
              "units": "4 steps per beat; native MIDI durations converted using MIDI PPQ",
              "response_counts_include_repeated_candidates_across_requests": True,
              "midi_pairing": "FIFO by channel/pitch; overlaps counted, not assumed unambiguous",
              "trials": len(summaries),
              "request_melody_note_records": len(melody_all),
              "request_melody_duration_ticks": histogram(melody_all),
              "response_acc_note_records": len(acc_all), "response_acc_duration_ticks": histogram(acc_all),
              "placeholder_duration_ticks": histogram(placeholder_all), "per_trial": summaries}
    (args.out / "duration_audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    with (args.out / "duration_audit.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=["piece", "seed", "melody_note_records", "melody_duration_values",
                                                   "acc_note_records", "acc_duration_values"])
        writer.writeheader()
        for row in summaries:
            writer.writerow({"piece": row["piece"], "seed": row["seed"],
                             "melody_note_records": row["request_melody_note_records"],
                             "melody_duration_values": ",".join(row["request_melody_duration_ticks"]),
                             "acc_note_records": row["response_acc_note_records"],
                             "acc_duration_values": ",".join(row["response_acc_duration_ticks"])})
    print(json.dumps({k: v for k, v in report.items() if k != "per_trial"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    main(parser.parse_args())
