#!/usr/bin/env python3
"""
Scan POP909 dataset and generate key map for all songs
"""

import os
import sys
import json
import glob
from tqdm import tqdm

# Change to project directory
os.chdir('/home/andrew/stanleyz')

# Add app directory to path for imports
sys.path.insert(0, '/home/andrew/stanleyz/app')
sys.path.insert(0, '/home/andrew/stanleyz')

from key_detection import detect_key_lightweight, detect_key_music21
from midi_utils import midi_to_note


def analyze_song(song_number: str, dataset_path: str, use_music21: bool = False):
    """
    Analyze a single song pair (melody + accompaniment)

    Args:
        song_number: Song number (e.g., "001", "002")
        dataset_path: Path to dataset directory
        use_music21: Use music21 method (slower but more accurate)

    Returns:
        dict with analysis results or None if error
    """
    mel_path = os.path.join(dataset_path, 'mel', f'{song_number}.mid')
    acc_path = os.path.join(dataset_path, 'acc', f'{song_number}.mid')

    if not os.path.exists(mel_path) or not os.path.exists(acc_path):
        return None

    try:
        # Load notes
        mel_notes, mel_resolution, mel_ticks = midi_to_note(mel_path)
        acc_notes, acc_resolution, acc_ticks = midi_to_note(acc_path)

        if not mel_notes or not acc_notes:
            return {
                'song_number': song_number,
                'error': 'No notes found'
            }

        # Detect keys
        if use_music21:
            mel_key = detect_key_music21(mel_notes)
            acc_key = detect_key_music21(acc_notes)
            method = 'music21'
        else:
            mel_key = detect_key_lightweight(mel_notes)
            acc_key = detect_key_lightweight(acc_notes)
            method = 'lightweight'

        # Check if keys match
        match = mel_key == acc_key

        return {
            'song_number': song_number,
            'melody_key': mel_key,
            'accompaniment_key': acc_key,
            'keys_match': match,
            'melody_notes_count': len(mel_notes),
            'accompaniment_notes_count': len(acc_notes),
            'melody_ticks': mel_ticks,
            'accompaniment_ticks': acc_ticks,
            'detection_method': method
        }

    except Exception as e:
        return {
            'song_number': song_number,
            'error': str(e)
        }


def main():
    """Scan all songs in POP909 dataset"""

    dataset_path = '/home/andrew/stanleyz/input/pop909_dataset'

    # Find all melody files
    mel_files = sorted(glob.glob(os.path.join(dataset_path, 'mel', '*.mid')))
    song_numbers = [os.path.basename(f).replace('.mid', '') for f in mel_files]

    print("="*80)
    print("POP909 DATASET KEY ANALYSIS")
    print("="*80)
    print(f"Dataset path: {dataset_path}")
    print(f"Total songs found: {len(song_numbers)}")
    print(f"Detection method: music21 (more accurate)")
    print()

    # Ask user for method choice
    use_music21 = True  # Default to music21 for accuracy

    # Analyze all songs
    results = []
    errors = []

    print("Analyzing songs...")
    for song_num in tqdm(song_numbers, desc="Processing"):
        result = analyze_song(song_num, dataset_path, use_music21=use_music21)
        if result:
            if 'error' in result:
                errors.append(result)
            else:
                results.append(result)

    print(f"\nCompleted: {len(results)} songs analyzed, {len(errors)} errors")
    print()

    # Save to JSON
    output_file = 'pop909_key_map.json'
    with open(output_file, 'w') as f:
        json.dump({
            'dataset': 'POP909',
            'total_songs': len(results),
            'detection_method': 'music21',
            'songs': results,
            'errors': errors
        }, f, indent=2)

    print(f"✓ Saved key map to: {output_file}")
    print()

    # Generate statistics
    matches = [r for r in results if r['keys_match']]
    mismatches = [r for r in results if not r['keys_match']]

    print("="*80)
    print("STATISTICS")
    print("="*80)
    print(f"Total songs analyzed:     {len(results)}")
    print(f"Keys match (mel = acc):   {len(matches)} ({len(matches)/len(results)*100:.1f}%)")
    print(f"Keys mismatch:            {len(mismatches)} ({len(mismatches)/len(results)*100:.1f}%)")
    print(f"Errors:                   {len(errors)}")
    print()

    # Key distribution (melody)
    key_counts = {}
    for r in results:
        key = r['melody_key']
        key_counts[key] = key_counts.get(key, 0) + 1

    print("KEY DISTRIBUTION (Melody Keys):")
    print("-"*80)
    for key in sorted(key_counts.keys(), key=lambda k: key_counts[k], reverse=True):
        count = key_counts[key]
        percentage = count / len(results) * 100
        bar = '█' * int(percentage / 2)
        print(f"  {key:<15} {count:3d} songs ({percentage:5.1f}%) {bar}")
    print()

    # Find songs for common keys
    print("SONGS BY COMMON KEYS (Perfect Matches Only):")
    print("-"*80)
    common_keys = ['C_major', 'G_major', 'D_major', 'A_major', 'F_major',
                   'A_minor', 'E_minor', 'D_minor', 'B_minor']

    for key in common_keys:
        matching_songs = [r['song_number'] for r in matches if r['melody_key'] == key]
        if matching_songs:
            print(f"\n{key} ({len(matching_songs)} songs):")
            print(f"  {', '.join(matching_songs[:20])}", end='')
            if len(matching_songs) > 20:
                print(f" ... and {len(matching_songs) - 20} more")
            else:
                print()

    print()
    print("="*80)
    print(f"✓ Analysis complete! Key map saved to: {output_file}")
    print("="*80)


if __name__ == "__main__":
    main()
