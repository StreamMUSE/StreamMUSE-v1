import os
import argparse
import pretty_midi
import torch
import time
import sys
import numpy as np

# Add the current directory to sys.path to ensure imports work
sys.path.append(os.getcwd())

from app.inference_engines.transformer_engine_lekai import InferenceEngineLekai
from lekai_model.MidiConverter import MidiConverter
from mid2pianoroll_v2 import MidiToNpzConverter


def save_midi(acc_notes, melody_notes, output_path, bpm=120, mel_program=0):
    pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)

    # Melody Track
    inst_mel = pretty_midi.Instrument(program=mel_program, name="Melody")
    seconds_per_beat = 60.0 / bpm
    seconds_per_tick = seconds_per_beat / 4

    for note in melody_notes:
        start_time = note["tick"] * seconds_per_tick
        duration_time = note["duration"] * seconds_per_tick
        end_time = start_time + duration_time
        pm_note = pretty_midi.Note(velocity=80, pitch=int(note["pitch"]), start=start_time, end=end_time)
        inst_mel.notes.append(pm_note)
    pm.instruments.append(inst_mel)

    # Accompaniment Track
    inst_acc = pretty_midi.Instrument(program=0, name="Accompaniment")
    for note in acc_notes:
        start_time = note["tick"] * seconds_per_tick
        duration_time = note["duration"] * seconds_per_tick
        end_time = start_time + duration_time
        pm_note = pretty_midi.Note(velocity=70, pitch=int(note["pitch"]), start=start_time, end=end_time)
        inst_acc.notes.append(pm_note)
    pm.instruments.append(inst_acc)

    pm.write(output_path)
    print(f"Saved MIDI to {output_path}")


def fake_offline_lekai(
    checkpoint_path,
    input_midi_path,
    output_path,
    input_acc_midi_path=None,
    prompt_beats=0,
):
    print(f"Processing {input_midi_path}...")

    # 1. Load Melody and Accompaniment using MidiToNpzConverter (handles BPM mismatch)
    converter = MidiToNpzConverter()
    melody_notes, acc_notes, metadata = converter.get_aligned_notes(input_midi_path, input_acc_midi_path)

    if melody_notes is None:
        print("Failed to load MIDI files.")
        return

    bpm = metadata.get("bpm", 120)
    time_sig = metadata.get("time_sig", (4, 4))

    print(f"Detected BPM: {bpm}, Time Sig: {time_sig}")

    # Calculate max_tick from notes
    max_tick_mel = max([n["tick"] + n["duration"] for n in melody_notes]) if melody_notes else 0
    max_tick_acc = max([n["tick"] + n["duration"] for n in acc_notes]) if acc_notes else 0
    max_tick = max(max_tick_mel, max_tick_acc)

    mel_program = 0
    try:
        pm_tmp = pretty_midi.PrettyMIDI(input_midi_path)
        for inst in pm_tmp.instruments:
            if not inst.is_drum:
                mel_program = inst.program
                break
    except:
        pass

    if not melody_notes:
        print("No notes found in MIDI.")
        return

    print(f"Loaded {len(melody_notes)} melody notes. Max tick: {max_tick}")
    if acc_notes:
        print(f"Loaded {len(acc_notes)} acc notes.")

    # 2. Initialize Engine
    try:
        engine = InferenceEngineLekai(
            checkpoint_path=checkpoint_path,
            model_size="llama",  # Default
            inference_mode="stateful",  # Use stateful to simulate streaming
        )
    except Exception as e:
        print(f"Failed to initialize engine: {e}")
        return

    # 3. Simulation Loop
    ticks_per_beat = 4
    total_beats = (max_tick + ticks_per_beat - 1) // ticks_per_beat + 4  # Add some buffer at end

    all_generated_notes = []

    print(f"Starting generation for {total_beats} beats (Prompt: {prompt_beats} beats)...")

    start_time = time.time()

    for beat in range(total_beats):
        start_tick = beat * ticks_per_beat
        end_tick = (beat + 1) * ticks_per_beat

        # Identify new notes for this beat
        new_mel_notes = [n for n in melody_notes if start_tick <= n["tick"] < end_tick]

        if beat < prompt_beats and input_acc_midi_path:
            # Prompt Phase: Use existing accompaniment
            new_acc_notes = [n for n in acc_notes if start_tick <= n["tick"] < end_tick]

            # Manually update history in engine
            # Note: We assume injection_offset_ticks is 0
            engine.melody_history.extend(new_mel_notes)
            engine.accompaniment_history.extend(new_acc_notes)

            # Add to output
            all_generated_notes.extend(new_acc_notes)
        else:
            # Generation Phase: Let model generate
            try:
                generated_notes, _, _, _, _ = engine.generate_accompaniment(
                    melody_notes=new_mel_notes, generation_start_tick=start_tick, bpm=bpm, time_sig=time_sig
                )

                if generated_notes:
                    all_generated_notes.extend(generated_notes)

            except Exception as e:
                print(f"Error at beat {beat}: {e}")
                import traceback

                traceback.print_exc()
                break

    total_time = time.time() - start_time
    print(f"Generation finished in {total_time:.2f}s")

    # 4. Save Result
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        save_midi(all_generated_notes, melody_notes, output_path, mel_program=mel_program)

        # Save Original (Melody + Original Acc) for comparison
        if input_acc_midi_path and acc_notes:
            dirname = os.path.dirname(output_path)
            basename = os.path.basename(output_path)
            # Try to replace 'gen_' with 'origin_' if present, otherwise prepend
            if basename.startswith("gen_"):
                origin_basename = "origin_" + basename[4:]
            else:
                origin_basename = "origin_" + basename

            origin_path = os.path.join(dirname, origin_basename)
            save_midi(acc_notes, melody_notes, origin_path, mel_program=mel_program)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--input_midi", type=str, required=True, help="Path to input melody MIDI")
    parser.add_argument("--input_acc", type=str, default=None, help="Path to input accompaniment MIDI (for prompting)")
    parser.add_argument("--prompt_beats", type=int, default=0, help="Number of beats to use from input_acc as prompt")
    parser.add_argument("--output_dir", type=str, default="output_lekai", help="Directory to save outputs")

    args = parser.parse_args()

    if os.path.isdir(args.input_midi):
        # Process directory
        for filename in os.listdir(args.input_midi):
            if filename.endswith(".mid") or filename.endswith(".midi"):
                input_path = os.path.join(args.input_midi, filename)

                # Try to find matching acc file if input_acc is a directory
                input_acc_path = None
                if args.input_acc:
                    if os.path.isdir(args.input_acc):
                        input_acc_path = os.path.join(args.input_acc, filename)
                        if not os.path.exists(input_acc_path):
                            print(f"Warning: Matching acc file not found for {filename}")
                            input_acc_path = None
                    else:
                        input_acc_path = args.input_acc

                output_path = os.path.join(args.output_dir, f"gen_{filename}")
                fake_offline_lekai(args.checkpoint_path, input_path, output_path, input_acc_path, args.prompt_beats)
    else:
        # Process single file
        filename = os.path.basename(args.input_midi)
        output_path = os.path.join(args.output_dir, f"gen_{filename}")
        fake_offline_lekai(args.checkpoint_path, args.input_midi, output_path, args.input_acc, args.prompt_beats)
