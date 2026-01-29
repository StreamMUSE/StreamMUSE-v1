#!/usr/bin/env python3
"""
Test tokenization round-trip:
MIDI → notes → events → pianoroll → tokens → decompress → pianoroll → events → notes → MIDI

Uses functions directly from transformer_engine_lekai.py to test the actual engine flow.
"""

import sys
import os
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lekai_model.config import ModelConfig
from lekai_model.my_tokenizer import PianoRollTokenizer
from lekai_model.MidiConverter import MidiConverter

# Import functions from the actual engine
from app.inference_engines.transformer_engine_lekai import (
    midi_to_note,
    notes_to_events,
    events_to_notes,
    save_to_midi,
)


def main():
    parser = argparse.ArgumentParser(description="Test tokenization round-trip")
    parser.add_argument("--midi", type=str, default="input/acc/001.mid",
                        help="Path to test MIDI file")
    parser.add_argument("--max-beats", type=int, default=32,
                        help="Maximum beats to test")
    parser.add_argument("--output-dir", type=str, default="output/roundtrip_test",
                        help="Directory for output files")
    args = parser.parse_args()
    
    # Initialize components (same as engine)
    config = ModelConfig()
    tokenizer = PianoRollTokenizer(
        patch_h=config.patch_h,
        patch_w=config.patch_w,
        marker_offset=81,
        measures_length=88,
        end_marker_part0=170,
        end_marker_part1=171,
        empty_marker=169,
        img_h=88,
    )
    converter = MidiConverter(ticks_per_beat=4)
    ticks_per_beat = 4
    
    print("="*60)
    print("TOKENIZATION ROUND-TRIP TEST")
    print("="*60)
    print(f"MIDI file: {args.midi}")
    print(f"Max beats: {args.max_beats}")
    
    # Step 1: Load MIDI using engine's function
    if not os.path.exists(args.midi):
        print(f"Error: MIDI file not found: {args.midi}")
        sys.exit(1)
    
    notes, metadata, max_tick = midi_to_note(args.midi)
    print(f"\n[Step 1] midi_to_note: {len(notes)} notes, max_tick={max_tick}")
    print(f"  Metadata: {metadata}")
    
    if not notes:
        print("Error: No notes in MIDI file")
        sys.exit(1)
    
    # Limit notes to max_beats
    max_tick_limit = args.max_beats * ticks_per_beat
    filtered_notes = [n for n in notes if n["tick"] < max_tick_limit]
    # Truncate duration if needed
    for n in filtered_notes:
        if n["tick"] + n.get("duration", 1) > max_tick_limit:
            n["duration"] = max(1, max_tick_limit - n["tick"])
    
    print(f"  Filtered to {len(filtered_notes)} notes within first {args.max_beats} beats")
    
    # Step 2: notes → events (using engine's function)
    events = notes_to_events(filtered_notes)
    print(f"\n[Step 2] notes_to_events: {len(events)} events")
    
    # Step 3: Process beat by beat (like engine does)
    # events → pianoroll → tokens → decompress → pianoroll → events
    print(f"\n[Step 3] Beat-by-beat: events → pianoroll → tokens → pianoroll → events")
    
    num_beats = (max_tick_limit + ticks_per_beat - 1) // ticks_per_beat
    all_reconstructed_events = []
    
    for beat_idx in range(num_beats):
        beat_start = beat_idx * ticks_per_beat
        beat_end = (beat_idx + 1) * ticks_per_beat
        
        # Filter notes for this beat
        beat_notes = []
        for n in filtered_notes:
            note_start = n["tick"]
            note_end = n["tick"] + n.get("duration", 1)
            if note_start < beat_end and note_end > beat_start:
                new_n = n.copy()
                new_n["tick"] = max(0, note_start - beat_start)
                end_rel = min(ticks_per_beat, note_end - beat_start)
                new_n["duration"] = max(1, end_rel - new_n["tick"])
                beat_notes.append(new_n)
        
        # notes → pianoroll (using converter like engine)
        pr = converter.notes_to_pianoroll(beat_notes, ticks_per_beat)
        
        # pianoroll → tokens (using tokenizer like engine)
        patch_tokens = tokenizer.image_to_patch_tokens(pr, strict_mode=True)
        compressed = tokenizer.compress_tokens(patch_tokens, end_marker=tokenizer.end_marker_part1)
        
        # tokens → pianoroll (using tokenizer like engine)
        decompressed = tokenizer.decompress_tokens(compressed, end_marker_id=tokenizer.end_marker_part1)
        pr_recon = tokenizer.patch_tokens_to_image(decompressed)
        
        # pianoroll → events (using converter like engine)
        beat_events = converter.pianoroll_to_events(pr_recon, start_tick=beat_start)
        all_reconstructed_events.extend(beat_events)
    
    print(f"  Total reconstructed events: {len(all_reconstructed_events)}")
    
    # Step 4: events → notes (using engine's function)
    reconstructed_notes = events_to_notes(all_reconstructed_events)
    print(f"\n[Step 4] events_to_notes: {len(reconstructed_notes)} notes")
    
    # Step 5: Save MIDI files
    os.makedirs(args.output_dir, exist_ok=True)
    bpm = metadata.get("bpm", 120)
    
    original_path = os.path.join(args.output_dir, "1_original.mid")
    roundtrip_path = os.path.join(args.output_dir, "2_roundtrip.mid")
    
    print(f"\n[Step 5] Saving MIDI files...")
    
    # Save original (using engine's save_to_midi, but we need a dummy melody)
    # Since save_to_midi expects both melody and acc, we'll use pretty_midi directly for simplicity
    import pretty_midi
    
    def save_notes_to_midi(notes_list, path, bpm_val):
        midi = pretty_midi.PrettyMIDI(initial_tempo=bpm_val)
        seconds_per_tick = 60.0 / (bpm_val * ticks_per_beat)
        inst = pretty_midi.Instrument(program=0, name="Piano")
        for n in notes_list:
            start_time = n["tick"] * seconds_per_tick
            end_time = (n["tick"] + n.get("duration", 1)) * seconds_per_tick
            velocity = n.get("velocity", 80)
            if end_time <= start_time:
                end_time = start_time + 0.05
            note = pretty_midi.Note(velocity=velocity, pitch=n["pitch"], start=start_time, end=end_time)
            inst.notes.append(note)
        midi.instruments.append(inst)
        midi.write(path)
    
    save_notes_to_midi(filtered_notes, original_path, bpm)
    save_notes_to_midi(reconstructed_notes, roundtrip_path, bpm)
    
    print(f"\n{'='*60}")
    print("OUTPUT FILES")
    print(f"{'='*60}")
    print(f"1. Original:  {original_path} ({len(filtered_notes)} notes)")
    print(f"2. Roundtrip: {roundtrip_path} ({len(reconstructed_notes)} notes)")
    print(f"\n🎵 Compare these MIDI files to verify the tokenization pipeline!")
    print("   If they sound the same, the pipeline is correct.")


if __name__ == "__main__":
    main()
