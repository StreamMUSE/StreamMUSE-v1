import argparse
import numpy as np
import os
import sys
import pretty_midi
from lekai_model.MidiConverter import MidiConverter


def get_time_signature_idx(time_signature_str):
    """Map time signature string to index"""
    ts_map = {
        "4/4": 0,
        "3/4": 1,
        "2/4": 2,
        "6/8": 3,
        "3/8": 4,
        "5/4": 5,
        "7/4": 6,
        "9/8": 7,
        "12/8": 8,
        "2/2": 9,
        "6/4": 10,
        "1/4": 11,
        "5/8": 12,
        "7/8": 13,
    }
    return ts_map.get(time_signature_str, -1)


def get_key_signature_idx(key_signature):
    """Map key signature to index (sharps count)"""
    if key_signature is None:
        return -1
    # pretty_midi key_signature is KeySignature object
    # But pm.key_signature_changes returns KeySignature objects with .key_number
    # We need to convert key number to sharps/flats count?
    # pretty_midi.key_number_to_key_name(key_number) gives string like 'C Major'
    # We need to map that to sharps count.

    # Simplified: Just return -1 if we can't easily map, or try to implement logic.
    # For now, let's try to get it right if possible, or default to -1.
    return -1  # Placeholder as pretty_midi doesn't give sharps count directly easily without logic


def mid2pianoroll(midi_path, output_path, acc_midi_path=None):
    converter = MidiConverter()

    # Load original MIDI to get metadata and tempo map
    try:
        pm = pretty_midi.PrettyMIDI(midi_path)
    except Exception as e:
        print(f"Error loading MIDI: {e}")
        return

    # If acc_midi_path is provided, load it too
    pm_acc = None
    if acc_midi_path:
        try:
            pm_acc = pretty_midi.PrettyMIDI(acc_midi_path)
        except Exception as e:
            print(f"Error loading Acc MIDI: {e}")
            return

    # Extract Metadata
    bpm = 120.0
    if pm.get_tempo_changes()[1].size > 0:
        bpm = pm.get_tempo_changes()[1][0]

    time_sig = (4, 4)
    if pm.time_signature_changes:
        ts = pm.time_signature_changes[0]
        time_sig = (ts.numerator, ts.denominator)

    time_sig_str = f"{time_sig[0]}/{time_sig[1]}"

    # Key Signature
    key_sig_name = None
    key_sig_idx = -1
    if pm.key_signature_changes:
        ks = pm.key_signature_changes[0]
        key_sig_name = pretty_midi.key_number_to_key_name(ks.key_number)

    metadata = {
        "bpm": bpm,
        "time_sig": time_sig,  # Keep tuple for internal use if needed
        "time_signature": time_sig_str,
        "time_signature_idx": get_time_signature_idx(time_sig_str),
        "key_signature": key_sig_name,
        "key_signature_idx": key_sig_idx,
        "tempo_text": None,  # MIDI doesn't usually have this text easily accessible
        "resolution": 16,
        "num_parts": 2,
        "num_channels": 4,
    }

    # Separate Melody and Accompaniment
    melody_notes = []
    acc_notes = []

    if acc_midi_path:
        # Case 1: Separate files provided
        print(f"Using separate files: Mel={midi_path}, Acc={acc_midi_path}")
        # Melody from pm
        for inst in pm.instruments:
            if not inst.is_drum:
                melody_notes.extend(inst.notes)
        # Acc from pm_acc
        for inst in pm_acc.instruments:
            if not inst.is_drum:
                acc_notes.extend(inst.notes)
    else:
        # Case 2: Single file, try to separate by instruments
        if len(pm.instruments) > 1:
            print(f"Found {len(pm.instruments)} instruments. Assuming Inst 0 is Melody, others Acc.")
            melody_notes.extend(pm.instruments[0].notes)
            for inst in pm.instruments[1:]:
                if not inst.is_drum:
                    acc_notes.extend(inst.notes)
        elif len(pm.instruments) == 1:
            print("Warning: Only 1 instrument found. Cannot separate Melody/Acc. Putting all in Melody.")
            melody_notes.extend(pm.instruments[0].notes)
        else:
            print("No instruments found.")
            return

    # Helper to convert notes to pianoroll using MidiConverter logic
    # We need to preserve the tempo map for accurate conversion
    def convert_notes(notes, original_pm):
        # Create a temp PM with the same tempo map
        temp_pm = pretty_midi.PrettyMIDI()
        # Copy tempo changes
        times, tempos = original_pm.get_tempo_changes()
        for t, tmp in zip(times, tempos):
            temp_pm._tick_scales.append((original_pm.time_to_tick(t), 60.0 / (tmp * original_pm.resolution)))
            # pretty_midi doesn't expose easy way to set tempo map directly without internal access
            # But we can use the 'initial_tempo' for simple cases.
            # For complex cases, we might need to rely on MidiConverter.midi_to_notes handling it.

        # Actually, MidiConverter.midi_to_notes uses pm.get_tempo_changes().
        # So we just need a PM that returns the same tempo changes.
        # But pretty_midi.PrettyMIDI(initial_tempo=...) only sets one tempo.
        # If we want to support variable tempo, we need to add them.
        # But pretty_midi doesn't have add_tempo_change method easily exposed?
        # It does! pm.instruments... no.
        # It's not easy.

        # ALTERNATIVE:
        # We can modify MidiConverter.midi_to_notes to accept a list of notes AND a tempo map source.
        # OR, we can just pass the original PM to midi_to_notes, but filter the notes inside?
        # MidiConverter.midi_to_notes iterates over pm.instruments.

        # Let's create a temp PM and add instruments with the specific notes.
        # And copy the tempo changes?
        # Actually, if we just use the original PM's tempo map logic in our own function, it's better.
        pass

    # Let's use the original PM but modify instruments temporarily
    original_instruments = pm.instruments

    # 1. Melody
    pm.instruments = []
    inst_mel = pretty_midi.Instrument(program=0, name="Melody")
    inst_mel.notes = melody_notes
    pm.instruments.append(inst_mel)
    mel_list, max_tick_mel = converter.midi_to_notes(pm)

    # 2. Acc
    pm.instruments = []
    inst_acc = pretty_midi.Instrument(program=0, name="Accompaniment")
    inst_acc.notes = acc_notes
    pm.instruments.append(inst_acc)
    acc_list, max_tick_acc = converter.midi_to_notes(pm)

    # Restore
    pm.instruments = original_instruments

    # Calculate dimensions
    max_tick = max(max_tick_mel, max_tick_acc)
    ticks_per_beat = converter.ticks_per_beat  # 4
    beats_per_measure = time_sig[0]
    ticks_per_measure = ticks_per_beat * beats_per_measure

    num_measures = (max_tick + ticks_per_measure - 1) // ticks_per_measure
    if num_measures == 0:
        num_measures = 1
    total_length = num_measures * ticks_per_measure

    # Convert to Pianoroll
    pr_mel = converter.notes_to_pianoroll(mel_list, max_tick=total_length)
    pr_acc = converter.notes_to_pianoroll(acc_list, max_tick=total_length)

    # Stack: (4, 88, T)
    # Ch 0-1: Mel, Ch 2-3: Acc
    full_pr = np.concatenate([pr_mel, pr_acc], axis=0)

    # Segment
    segments = {}
    valid_measures = []

    for i in range(num_measures):
        start = i * ticks_per_measure
        end = (i + 1) * ticks_per_measure
        segment = full_pr[:, :, start:end]
        segments[f"measure_{i}"] = segment
        valid_measures.append(i)

    metadata["num_measures"] = num_measures
    metadata["total_length"] = total_length
    metadata["valid_measures"] = valid_measures
    metadata["original_measures"] = num_measures  # Assuming all are valid

    # Save
    np.savez_compressed(output_path, **segments, metadata=metadata)
    print(f"Converted {midi_path} -> {output_path}")
    print(f"Measures: {num_measures}, Shape per measure: {segment.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_midi", help="Path to input MIDI file (Melody)")
    parser.add_argument("output_npz", help="Path to output NPZ file")
    parser.add_argument("--input_acc", help="Path to input Accompaniment MIDI file (Optional)", default=None)
    args = parser.parse_args()

    mid2pianoroll(args.input_midi, args.output_npz, args.input_acc)
