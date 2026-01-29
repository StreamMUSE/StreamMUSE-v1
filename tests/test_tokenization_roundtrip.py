#!/usr/bin/env python3
"""
Test tokenization round-trip:
MIDI → notes → events → pianoroll → tokens → decompress → pianoroll → events → notes → MIDI

Uses functions directly from transformer_engine_lekai.py to test the actual engine flow.
"""

import sys
import os
import argparse
import json
import numpy as np

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


def pianoroll_to_str(pr, beat_idx=None):
    """Convert pianoroll to readable string for logging."""
    if pr is None:
        return "None"
    
    sustain = pr[0]  # (88, T)
    onset = pr[1]    # (88, T)
    T = pr.shape[2]
    
    lines = []
    if beat_idx is not None:
        lines.append(f"Beat {beat_idx} pianoroll (shape={pr.shape}):")
    
    # Find active pitches
    active_mask = np.any(sustain > 0, axis=1) | np.any(onset > 0, axis=1)
    active_pitches = np.where(active_mask)[0]
    
    if len(active_pitches) == 0:
        lines.append("  (empty)")
        return "\n".join(lines)
    
    lines.append(f"  tick:   " + "".join([str(t) for t in range(T)]))
    for pitch_idx in reversed(active_pitches):
        midi_pitch = pitch_idx + 21
        row = f"  P{midi_pitch:3d}: "
        for t in range(T):
            s = sustain[pitch_idx, t]
            o = onset[pitch_idx, t]
            if o > 0 and s > 0:
                row += "O"  # onset + sustain
            elif s > 0:
                row += "S"  # sustain only
            elif o > 0:
                row += "o"  # onset only (shouldn't happen normally)
            else:
                row += "."
        lines.append(row)
    
    return "\n".join(lines)


def compare_pianorolls(pr1, pr2, beat_idx):
    """Compare two pianorolls and return diff info."""
    if pr1 is None or pr2 is None:
        return "One or both pianorolls are None"
    
    if pr1.shape != pr2.shape:
        return f"Shape mismatch: {pr1.shape} vs {pr2.shape}"
    
    if np.array_equal(pr1, pr2):
        return "MATCH"
    
    # Find differences
    diff = pr1 != pr2
    diff_count = np.sum(diff)
    
    lines = [f"DIFF: {diff_count} elements differ"]
    
    # Show which pitches differ
    for ch, ch_name in [(0, "sustain"), (1, "onset")]:
        for pitch in range(88):
            if not np.array_equal(pr1[ch, pitch], pr2[ch, pitch]):
                midi_pitch = pitch + 21
                lines.append(f"  ch={ch_name}, P{midi_pitch}: orig={pr1[ch, pitch].tolist()} -> recon={pr2[ch, pitch].tolist()}")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Test tokenization round-trip")
    parser.add_argument("--midi", type=str, default="input/acc/001.mid",
                        help="Path to test MIDI file")
    parser.add_argument("--max-beats", type=int, default=32,
                        help="Maximum beats to test")
    parser.add_argument("--output-dir", type=str, default="output/roundtrip_test",
                        help="Directory for output files")
    parser.add_argument("--debug-beats", type=int, default=4,
                        help="Number of beats to show detailed debug info")
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
    
    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "debug_log.txt")
    log_file = open(log_path, "w")
    
    def log(msg):
        print(msg)
        log_file.write(msg + "\n")
    
    log("="*80)
    log("TOKENIZATION ROUND-TRIP TEST (DETAILED DEBUG)")
    log("="*80)
    log(f"MIDI file: {args.midi}")
    log(f"Max beats: {args.max_beats}")
    log(f"Debug beats: {args.debug_beats}")
    
    # Step 1: Load MIDI using engine's function
    if not os.path.exists(args.midi):
        log(f"Error: MIDI file not found: {args.midi}")
        sys.exit(1)
    
    notes, metadata, max_tick = midi_to_note(args.midi)
    log(f"\n[Step 1] midi_to_note: {len(notes)} notes, max_tick={max_tick}")
    log(f"  Metadata: {metadata}")
    
    if not notes:
        log("Error: No notes in MIDI file")
        sys.exit(1)
    
    # Limit notes to max_beats
    max_tick_limit = args.max_beats * ticks_per_beat
    filtered_notes = [n for n in notes if n["tick"] < max_tick_limit]
    # Truncate duration if needed
    for n in filtered_notes:
        if n["tick"] + n.get("duration", 1) > max_tick_limit:
            n["duration"] = max(1, max_tick_limit - n["tick"])
    
    log(f"  Filtered to {len(filtered_notes)} notes within first {args.max_beats} beats")
    
    # Save original notes list
    log(f"\n--- Original Notes (first 20) ---")
    for i, n in enumerate(filtered_notes[:20]):
        log(f"  [{i}] pitch={n['pitch']}, tick={n['tick']}, duration={n.get('duration', 1)}")
    if len(filtered_notes) > 20:
        log(f"  ... and {len(filtered_notes) - 20} more")
    
    # Step 2: notes → events (using engine's function)
    events = notes_to_events(filtered_notes)
    log(f"\n[Step 2] notes_to_events: {len(events)} events")
    
    log(f"\n--- Original Events (first 40) ---")
    for i, e in enumerate(events[:40]):
        log(f"  [{i}] type={e['type']}, pitch={e['pitch']}, tick={e['tick']}")
    if len(events) > 40:
        log(f"  ... and {len(events) - 40} more")
    
    # Step 3: Process beat by beat (like engine does)
    log(f"\n[Step 3] Beat-by-beat processing")
    log("="*80)
    
    num_beats = (max_tick_limit + ticks_per_beat - 1) // ticks_per_beat
    all_reconstructed_events = []
    
    issues_found = []
    
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
        pr_orig = converter.notes_to_pianoroll(beat_notes, ticks_per_beat)
        
        # pianoroll → tokens (using tokenizer like engine)
        patch_tokens = tokenizer.image_to_patch_tokens(pr_orig, strict_mode=True)
        compressed = tokenizer.compress_tokens(patch_tokens, end_marker=tokenizer.end_marker_part1)
        
        # tokens → pianoroll (using tokenizer like engine)
        decompressed = tokenizer.decompress_tokens(compressed, end_marker_id=tokenizer.end_marker_part1)
        pr_recon = tokenizer.patch_tokens_to_image(decompressed)
        
        # pianoroll → events (using converter like engine)
        beat_events = converter.pianoroll_to_events(pr_recon, start_tick=beat_start)
        all_reconstructed_events.extend(beat_events)
        
        # Compare pianorolls
        pr_match = np.array_equal(pr_orig, pr_recon)
        
        # Detailed logging for first N beats or if there's an issue
        if beat_idx < args.debug_beats or not pr_match:
            log(f"\n{'='*40}")
            log(f"BEAT {beat_idx} (ticks {beat_start}-{beat_end})")
            log(f"{'='*40}")
            
            log(f"\n  Beat notes ({len(beat_notes)}):")
            for n in beat_notes:
                log(f"    pitch={n['pitch']}, tick={n['tick']}, duration={n['duration']}")
            
            log(f"\n  Original pianoroll:")
            log(pianoroll_to_str(pr_orig, beat_idx))
            
            log(f"\n  Tokens: {compressed}")
            
            log(f"\n  Reconstructed pianoroll:")
            log(pianoroll_to_str(pr_recon, beat_idx))
            
            log(f"\n  Pianoroll comparison: {compare_pianorolls(pr_orig, pr_recon, beat_idx)}")
            
            log(f"\n  Reconstructed events ({len(beat_events)}):")
            for e in beat_events:
                log(f"    type={e['type']}, pitch={e['pitch']}, tick={e['tick']}")
            
            if not pr_match:
                issues_found.append(beat_idx)
    
    log(f"\n{'='*80}")
    log(f"Total reconstructed events: {len(all_reconstructed_events)}")
    
    if issues_found:
        log(f"\n⚠️ Pianoroll mismatch found in beats: {issues_found}")
    else:
        log(f"\n✅ All pianorolls match after tokenize/detokenize")
    
    # Step 4: events → notes (using engine's function)
    reconstructed_notes = events_to_notes(all_reconstructed_events)
    log(f"\n[Step 4] events_to_notes: {len(reconstructed_notes)} notes")
    
    log(f"\n--- Reconstructed Notes (first 20) ---")
    for i, n in enumerate(reconstructed_notes[:20]):
        log(f"  [{i}] pitch={n['pitch']}, tick={n['tick']}, duration={n.get('duration', 1)}")
    if len(reconstructed_notes) > 20:
        log(f"  ... and {len(reconstructed_notes) - 20} more")
    
    # Compare original vs reconstructed notes
    log(f"\n{'='*80}")
    log("NOTES COMPARISON")
    log(f"{'='*80}")
    log(f"Original: {len(filtered_notes)} notes")
    log(f"Reconstructed: {len(reconstructed_notes)} notes")
    
    # Build lookup
    orig_dict = {}
    for n in filtered_notes:
        key = (n["tick"], n["pitch"])
        orig_dict[key] = n.get("duration", 1)
    
    recon_dict = {}
    for n in reconstructed_notes:
        key = (n["tick"], n["pitch"])
        recon_dict[key] = n.get("duration", 1)
    
    missing = []
    extra = []
    duration_diff = []
    
    for key, dur in orig_dict.items():
        if key not in recon_dict:
            missing.append((key, dur))
        elif recon_dict[key] != dur:
            duration_diff.append((key, dur, recon_dict[key]))
    
    for key, dur in recon_dict.items():
        if key not in orig_dict:
            extra.append((key, dur))
    
    if missing:
        log(f"\n❌ Missing in reconstructed ({len(missing)}):")
        for (tick, pitch), dur in missing[:20]:
            log(f"   tick={tick}, pitch={pitch}, duration={dur}")
    
    if extra:
        log(f"\n❌ Extra in reconstructed ({len(extra)}):")
        for (tick, pitch), dur in extra[:20]:
            log(f"   tick={tick}, pitch={pitch}, duration={dur}")
    
    if duration_diff:
        log(f"\n⚠️ Duration differences ({len(duration_diff)}):")
        for (tick, pitch), orig_dur, recon_dur in duration_diff[:20]:
            log(f"   tick={tick}, pitch={pitch}: {orig_dur} -> {recon_dur}")
    
    if not missing and not extra and not duration_diff:
        log(f"\n✅ All notes match perfectly!")
    
    # Step 5: Save MIDI files
    bpm = metadata.get("bpm", 120)
    
    original_path = os.path.join(args.output_dir, "1_original.mid")
    roundtrip_path = os.path.join(args.output_dir, "2_roundtrip.mid")
    
    log(f"\n[Step 5] Saving MIDI files...")
    
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
    
    log(f"\n{'='*80}")
    log("OUTPUT FILES")
    log(f"{'='*80}")
    log(f"1. Original:  {original_path} ({len(filtered_notes)} notes)")
    log(f"2. Roundtrip: {roundtrip_path} ({len(reconstructed_notes)} notes)")
    log(f"3. Debug log: {log_path}")
    
    log_file.close()
    print(f"\n🔍 Detailed log saved to: {log_path}")


if __name__ == "__main__":
    main()
