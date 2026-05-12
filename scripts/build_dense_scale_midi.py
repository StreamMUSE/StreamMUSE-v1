"""Build a controlled dense-scale benchmark MIDI file.

One note per quarter (no leading rest); ascending C major scale repeated for
the requested number of beats. Used as a content-neutral input for lateness
benchmarks; see docs/benchmarks/methodology.md.
"""

from __future__ import annotations

import argparse
import os
import sys

import mido


SCALE_DEGREES = (60, 62, 64, 65, 67, 69, 71, 72)  # C4..C5


def build(path: str, beats: int, bpm: int, velocity: int, ticks_per_beat: int) -> None:
    mid = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm), time=0))
    for i in range(beats):
        pitch = SCALE_DEGREES[i % len(SCALE_DEGREES)]
        # Note duration: one full quarter so adjacent notes do not overlap.
        track.append(mido.Message("note_on",  note=pitch, velocity=velocity, time=0))
        track.append(mido.Message("note_off", note=pitch, velocity=0,        time=ticks_per_beat))
    track.append(mido.MetaMessage("end_of_track", time=0))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    mid.save(path)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="prompts/bench/dense_scale.mid")
    p.add_argument("--beats", type=int, default=32)
    p.add_argument("--bpm", type=int, default=120)
    p.add_argument("--velocity", type=int, default=80)
    p.add_argument("--ticks-per-beat", type=int, default=480)
    args = p.parse_args()
    build(args.out, args.beats, args.bpm, args.velocity, args.ticks_per_beat)
    print(f"wrote {args.out} ({args.beats} beats, {args.bpm} bpm)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
