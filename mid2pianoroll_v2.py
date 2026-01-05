import argparse
import numpy as np
import os
import sys
import pretty_midi
from lekai_model.MidiConverter import MidiConverter


class MidiToNpzConverter:
    def __init__(self, resolution=16):
        self.resolution = resolution
        self.midi_converter = MidiConverter(ticks_per_beat=resolution // 4)

    def get_time_signature_idx(self, time_signature_str):
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

    def fill_gaps(self, notes, max_gap=1):
        """
        Fill small gaps (<= max_gap ticks) between consecutive notes of the same pitch.
        This helps avoid staccato artifacts from quantization.
        """
        notes_by_pitch = {}
        for note in notes:
            p = note["pitch"]
            if p not in notes_by_pitch:
                notes_by_pitch[p] = []
            notes_by_pitch[p].append(note)

        count = 0
        for p in notes_by_pitch:
            p_notes = notes_by_pitch[p]
            p_notes.sort(key=lambda x: x["tick"])

            for i in range(len(p_notes) - 1):
                n1 = p_notes[i]
                n2 = p_notes[i + 1]

                end_n1 = n1["tick"] + n1["duration"]
                start_n2 = n2["tick"]

                gap = start_n2 - end_n1

                # If gap is small (e.g. 1 tick), close it
                if 0 < gap <= max_gap:
                    n1["duration"] += gap
                    count += 1
        return count

    def get_aligned_notes(self, midi_path, acc_midi_path=None):
        """
        Load and align Melody and Accompaniment notes, handling BPM mismatches.
        Returns:
            tuple: (mel_list, acc_list, metadata) or (None, None, None) if failed.
            mel_list, acc_list are lists of note dicts with 'tick' aligned to musical time.
        """
        # Load original MIDI to get metadata and tempo map
        try:
            pm = pretty_midi.PrettyMIDI(midi_path)
        except Exception as e:
            print(f"Error loading MIDI {midi_path}: {e}")
            return None, None, None

        # If acc_midi_path is provided, load it too
        pm_acc = None
        if acc_midi_path:
            try:
                pm_acc = pretty_midi.PrettyMIDI(acc_midi_path)
                # Debug: Check BPM and Start Time
                acc_bpm = 120.0
                if pm_acc.get_tempo_changes()[1].size > 0:
                    acc_bpm = pm_acc.get_tempo_changes()[1][0]

                first_note_time = float("inf")
                for inst in pm_acc.instruments:
                    if not inst.is_drum:
                        for note in inst.notes:
                            if note.start < first_note_time:
                                first_note_time = note.start

                print(f"  [Debug] Acc BPM: {acc_bpm:.2f}, First Note Time: {first_note_time:.4f}s")

            except Exception as e:
                print(f"Error loading Acc MIDI {acc_midi_path}: {e}")
                return None, None, None

        # Extract Metadata
        bpm = 120.0
        if pm.get_tempo_changes()[1].size > 0:
            bpm = pm.get_tempo_changes()[1][0]

        # Debug Mel
        mel_first_note = float("inf")
        for inst in pm.instruments:
            if not inst.is_drum:
                for note in inst.notes:
                    if note.start < mel_first_note:
                        mel_first_note = note.start
        print(f"  [Debug] Mel BPM: {bpm:.2f}, First Note Time: {mel_first_note:.4f}s")

        # FIX: Use Acc BPM if Mel is default (120) and Acc is different
        # This handles cases where Mel lost tempo info but Acc kept it.
        use_acc_tempo = False
        if pm_acc and abs(bpm - 120.0) < 0.01:
            acc_bpm = 120.0
            if pm_acc.get_tempo_changes()[1].size > 0:
                acc_bpm = pm_acc.get_tempo_changes()[1][0]

            if abs(acc_bpm - 120.0) > 0.01:
                print(f"  [Info] Overriding Mel BPM ({bpm}) with Acc BPM ({acc_bpm})")
                bpm = acc_bpm
                use_acc_tempo = True
                pass

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
            "time_signature_idx": self.get_time_signature_idx(time_sig_str),
            "key_signature": key_sig_name,
            "key_signature_idx": key_sig_idx,
            "tempo_text": None,  # MIDI doesn't usually have this text easily accessible
            "resolution": self.resolution,
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
                return None, None, None

        # Use the original PM but modify instruments temporarily to use MidiConverter logic
        # This ensures we use the tempo map from the Melody MIDI
        original_instruments = pm.instruments

        # STRATEGY: Process Mel and Acc SEPARATELY using their OWN tempo maps.
        # Assumption: Mel and Acc are aligned in "Musical Time" (Beats/Measures),
        # even if their "Physical Time" (Seconds) differs due to wrong BPM in Mel.
        # By converting both to Ticks (which are proportional to Beats), they will align.

        # 1. Melody
        # Use pm (which likely has default 120 BPM)
        pm.instruments = []
        inst_mel = pretty_midi.Instrument(program=0, name="Melody")
        inst_mel.notes = melody_notes
        pm.instruments.append(inst_mel)
        mel_list, max_tick_mel = self.midi_converter.midi_to_notes(pm)

        # 2. Acc
        # Use pm_acc if available (which has real BPM), otherwise pm
        target_pm_acc = pm_acc if pm_acc else pm
        original_acc_instruments = target_pm_acc.instruments

        target_pm_acc.instruments = []
        inst_acc = pretty_midi.Instrument(program=0, name="Accompaniment")
        inst_acc.notes = acc_notes
        target_pm_acc.instruments.append(inst_acc)
        acc_list, max_tick_acc = self.midi_converter.midi_to_notes(target_pm_acc)

        # Restore instruments
        pm.instruments = original_instruments
        if pm_acc:
            pm_acc.instruments = original_acc_instruments

        # FIX: Fill gaps in Acc
        filled = self.fill_gaps(acc_list, max_gap=1)
        if filled > 0:
            print(f"  Filled {filled} gaps in Accompaniment.")

        return mel_list, acc_list, metadata

    def get_numpy_data(self, midi_path, acc_midi_path=None):
        """
        Convert MIDI file(s) to Numpy data (segments and metadata).
        Returns:
            tuple: (segments, metadata) or (None, None) if failed.
        """
        mel_list, acc_list, metadata = self.get_aligned_notes(midi_path, acc_midi_path)
        if mel_list is None:
            return None, None

        # Calculate dimensions
        # We need to recalculate max_tick because get_aligned_notes doesn't return it directly
        # But we can compute it from the lists
        max_tick_mel = max([n["tick"] + n["duration"] for n in mel_list]) if mel_list else 0
        max_tick_acc = max([n["tick"] + n["duration"] for n in acc_list]) if acc_list else 0

        max_tick = max(max_tick_mel, max_tick_acc)
        ticks_per_beat = self.midi_converter.ticks_per_beat  # 4
        beats_per_measure = metadata["time_sig"][0]
        ticks_per_measure = ticks_per_beat * beats_per_measure

        num_measures = (max_tick + ticks_per_measure - 1) // ticks_per_measure
        if num_measures == 0:
            num_measures = 1
        total_length = num_measures * ticks_per_measure

        # Convert to Pianoroll
        pr_mel = self.midi_converter.notes_to_pianoroll(mel_list, max_tick=total_length)
        pr_acc = self.midi_converter.notes_to_pianoroll(acc_list, max_tick=total_length)

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

        return segments, metadata

    def convert(self, midi_path, output_path, acc_midi_path=None):
        """
        Convert MIDI file(s) to NPZ format.

        Args:
            midi_path (str): Path to the main MIDI file (usually Melody).
            output_path (str): Path to save the output NPZ file.
            acc_midi_path (str, optional): Path to the accompaniment MIDI file.
                                           If provided, it will be merged with midi_path.
        """
        segments, metadata = self.get_numpy_data(midi_path, acc_midi_path)
        if segments is None:
            return False

        # Save
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        np.savez_compressed(output_path, **segments, metadata=metadata)
        print(f"Converted {midi_path} -> {output_path}")
        print(f"Measures: {metadata['num_measures']}")
        return True


def process_batch(input_dir, output_dir, acc_dir=None):
    """
    Batch process MIDI files in a directory.
    """
    converter = MidiToNpzConverter()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    files = [f for f in os.listdir(input_dir) if f.endswith(".mid") or f.endswith(".midi")]

    for f in files:
        midi_path = os.path.join(input_dir, f)
        output_path = os.path.join(output_dir, os.path.splitext(f)[0] + ".npz")

        acc_midi_path = None
        if acc_dir:
            # Try to find matching acc file
            # Assumption: same filename in acc_dir
            acc_candidate = os.path.join(acc_dir, f)
            if os.path.exists(acc_candidate):
                acc_midi_path = acc_candidate
            else:
                print(f"Warning: Acc file not found for {f} in {acc_dir}")

        converter.convert(midi_path, output_path, acc_midi_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert MIDI to NPZ for training/inference.")
    parser.add_argument("input_midi", help="Path to input MIDI file (Melody) or Directory")
    parser.add_argument("output_npz", help="Path to output NPZ file or Directory")
    parser.add_argument(
        "--input_acc", help="Path to input Accompaniment MIDI file or Directory (Optional)", default=None
    )
    args = parser.parse_args()

    if os.path.isdir(args.input_midi):
        print(f"Batch processing from {args.input_midi}...")
        process_batch(args.input_midi, args.output_npz, args.input_acc)
    else:
        converter = MidiToNpzConverter()
        converter.convert(args.input_midi, args.output_npz, args.input_acc)
