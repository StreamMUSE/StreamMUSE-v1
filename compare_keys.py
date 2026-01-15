#!/usr/bin/env python3
"""
Compare melody and accompaniment keys using music21 method
"""

import os
import sys

# Change to project directory
os.chdir('/home/andrew/stanleyz')

# Add app directory to path for imports
sys.path.insert(0, '/home/andrew/stanleyz/app')
sys.path.insert(0, '/home/andrew/stanleyz')

from key_detection import detect_key_music21
from midi_input_script import midi_to_note


def compare_keys(file_number):
    """Compare melody and accompaniment keys for a given file number"""
    mel_path = f'/home/andrew/stanleyz/input/mel/{file_number}.mid'
    acc_path = f'/home/andrew/stanleyz/input/acc/{file_number}.mid'

    if not os.path.exists(mel_path) or not os.path.exists(acc_path):
        return None

    # Load and detect keys
    mel_notes, _, _ = midi_to_note(mel_path)
    acc_notes, _, _ = midi_to_note(acc_path)

    if not mel_notes or not acc_notes:
        return None

    mel_key = detect_key_music21(mel_notes)
    acc_key = detect_key_music21(acc_notes)

    # Check if keys match
    match = mel_key == acc_key

    # Check if relative major/minor
    relative = False
    if not match:
        # Extract note name and mode
        mel_parts = mel_key.split('_')
        acc_parts = acc_key.split('_')
        if len(mel_parts) == 2 and len(acc_parts) == 2:
            mel_note, mel_mode = mel_parts
            acc_note, acc_mode = acc_parts
            # Simple relative check (not perfect but good enough)
            if mel_mode != acc_mode:
                relative = True

    return {
        'file': file_number,
        'mel_key': mel_key,
        'acc_key': acc_key,
        'match': match,
        'relative': relative,
        'mel_notes_count': len(mel_notes),
        'acc_notes_count': len(acc_notes)
    }


def main():
    """Compare all numbered MIDI files"""

    print("="*80)
    print("MELODY vs ACCOMPANIMENT KEY COMPARISON (using music21)")
    print("="*80)
    print()

    # Check files 001-010
    results = []
    for i in range(1, 11):
        file_num = f"{i:03d}"
        result = compare_keys(file_num)
        if result:
            results.append(result)

    # Print results
    print(f"{'File':<8} {'Melody Key':<15} {'Acc Key':<15} {'Status':<15} {'Mel Notes':<10} {'Acc Notes':<10}")
    print("-"*80)

    matches = []
    relatives = []
    mismatches = []

    for r in results:
        if r['match']:
            status = "✅ MATCH"
            matches.append(r)
        elif r['relative']:
            status = "⚠️  RELATIVE"
            relatives.append(r)
        else:
            status = "❌ MISMATCH"
            mismatches.append(r)

        print(f"{r['file']:<8} {r['mel_key']:<15} {r['acc_key']:<15} {status:<15} {r['mel_notes_count']:<10} {r['acc_notes_count']:<10}")

    print()
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Perfect Matches:  {len(matches)} files")
    print(f"Relative Keys:    {len(relatives)} files")
    print(f"Mismatches:       {len(mismatches)} files")
    print()

    if matches:
        print("RECOMMENDED FOR PROMPT LIBRARY (Perfect Matches):")
        print("-"*80)
        for r in matches:
            print(f"  {r['file']}.mid - {r['mel_key']}")
        print()

    if relatives:
        print("MIGHT WORK (Relative Major/Minor):")
        print("-"*80)
        for r in relatives:
            print(f"  {r['file']}.mid - Melody: {r['mel_key']}, Acc: {r['acc_key']}")
        print()

    if mismatches:
        print("NOT RECOMMENDED (Key Mismatches):")
        print("-"*80)
        for r in mismatches:
            print(f"  {r['file']}.mid - Melody: {r['mel_key']}, Acc: {r['acc_key']}")
        print()

    # Key distribution of matches
    if matches:
        print("KEY DISTRIBUTION (Matches Only):")
        print("-"*80)
        key_counts = {}
        for r in matches:
            key = r['mel_key']
            if key not in key_counts:
                key_counts[key] = []
            key_counts[key].append(r['file'])

        for key in sorted(key_counts.keys()):
            files = ', '.join(key_counts[key])
            print(f"  {key:<20} {len(key_counts[key])} file(s): {files}")
        print()


if __name__ == "__main__":
    main()
