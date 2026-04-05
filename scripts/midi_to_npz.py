"""Convert paired melody/accompaniment MIDI files to NPZ format for PianoLLaMA.

Usage:
    uv run python scripts/midi_to_npz.py \
        --mel-dir prompts/inputs_lekai/mel \
        --acc-dir prompts/inputs_lekai/acc \
        --output-dir prompts/inputs_lekai/npz

NPZ format (expected by PianoDataset):
    metadata: dict with time_signature_idx, bpm, num_measures, is_continuation
    measure_0, measure_1, ...: np.ndarray of shape (4, 88, timesteps_per_measure)
        channels [0:2] = melody (sustain, onset)
        channels [2:4] = accompaniment (sustain, onset)
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.streammuse.infrastructure.inference.lekai_model.MidiConverter import MidiConverter


# Time signature (numerator, denominator) → index used by PianoDataset.
# 2/2 is stored as 9; at load time PianoDataset maps 9 → 4.
TIME_SIG_MAP = {
    (2, 4): 2,
    (3, 4): 3,
    (4, 4): 4,
    (6, 8): 6,
    (2, 2): 9,
}

TICKS_PER_BEAT = 4  # Must match model's tokenization resolution


def midi_pair_to_npz(
    mel_path: str,
    acc_path: str,
    output_path: str,
    ticks_per_beat: int = TICKS_PER_BEAT,
) -> bool:
    """Convert one (melody, accompaniment) MIDI pair to a single NPZ file."""
    converter = MidiConverter(ticks_per_beat=ticks_per_beat)

    # --- Load melody ---
    mel_pm, mel_meta = converter.load_midi(mel_path)
    if mel_pm is None:
        print(f"  [SKIP] Cannot load melody: {mel_path}")
        return False

    mel_notes, mel_max_tick = converter.midi_to_notes(mel_pm)
    if not mel_notes:
        print(f"  [SKIP] Melody has no notes: {mel_path}")
        return False

    # --- Load accompaniment ---
    acc_pm, acc_meta = converter.load_midi(acc_path)
    if acc_pm is None:
        print(f"  [SKIP] Cannot load accompaniment: {acc_path}")
        return False

    acc_notes, acc_max_tick = converter.midi_to_notes(acc_pm)

    # --- Determine metadata ---
    bpm = mel_meta["bpm"]
    time_sig = mel_meta["time_sig"]  # (numerator, denominator)

    time_sig_idx = TIME_SIG_MAP.get(time_sig)
    if time_sig_idx is None:
        print(f"  [WARN] Unknown time signature {time_sig}, defaulting to 4/4")
        time_sig_idx = 4
        time_sig = (4, 4)

    beats_per_measure = time_sig[0]
    ticks_per_measure = beats_per_measure * ticks_per_beat

    # --- Build pianorolls ---
    max_tick = max(mel_max_tick, acc_max_tick)
    # Pad to full measure boundary
    if max_tick % ticks_per_measure != 0:
        max_tick = ((max_tick // ticks_per_measure) + 1) * ticks_per_measure

    mel_pr = converter.notes_to_pianoroll(mel_notes, max_tick=max_tick)  # (2, 88, T)
    acc_pr = converter.notes_to_pianoroll(acc_notes, max_tick=max_tick)  # (2, 88, T)

    num_measures = max_tick // ticks_per_measure

    # --- Build save dict ---
    save_dict: dict = {
        "metadata": {
            "time_signature_idx": time_sig_idx,
            "bpm": int(round(bpm)),
            "num_measures": num_measures,
            "is_continuation": False,
        },
    }

    for i in range(num_measures):
        start = i * ticks_per_measure
        end = start + ticks_per_measure

        # (4, 88, ticks_per_measure): [mel_sustain, mel_onset, acc_sustain, acc_onset]
        measure = np.zeros((4, 88, ticks_per_measure), dtype=np.float32)
        measure[0] = mel_pr[0, :, start:end]  # melody sustain
        measure[1] = mel_pr[1, :, start:end]  # melody onset
        measure[2] = acc_pr[0, :, start:end]  # acc sustain
        measure[3] = acc_pr[1, :, start:end]  # acc onset

        save_dict[f"measure_{i}"] = measure

    np.savez(output_path, **save_dict)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert paired MIDI files to NPZ for PianoLLaMA")
    parser.add_argument("--mel-dir", required=True, help="Directory containing melody MIDI files")
    parser.add_argument("--acc-dir", required=True, help="Directory containing accompaniment MIDI files")
    parser.add_argument("--output-dir", required=True, help="Output directory for NPZ files")
    parser.add_argument("--ticks-per-beat", type=int, default=TICKS_PER_BEAT)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    mel_files = sorted(f for f in os.listdir(args.mel_dir) if f.endswith(".mid"))

    converted = 0
    skipped = 0

    for mel_file in mel_files:
        acc_file = mel_file  # Same filename in acc dir
        mel_path = os.path.join(args.mel_dir, mel_file)
        acc_path = os.path.join(args.acc_dir, acc_file)

        if not os.path.exists(acc_path):
            print(f"  [SKIP] No matching acc file for {mel_file}")
            skipped += 1
            continue

        stem = os.path.splitext(mel_file)[0]
        output_path = os.path.join(args.output_dir, f"{stem}.npz")

        print(f"Converting {mel_file} ...", end=" ")
        ok = midi_pair_to_npz(mel_path, acc_path, output_path, ticks_per_beat=args.ticks_per_beat)
        if ok:
            print("OK")
            converted += 1
        else:
            skipped += 1

    print(f"\nDone: {converted} converted, {skipped} skipped")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
