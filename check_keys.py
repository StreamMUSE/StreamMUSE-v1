#!/usr/bin/env python3
"""
Key Detection Script for MIDI Files

Analyzes MIDI files and detects their musical keys using both lightweight
and music21-based detection algorithms.
"""

import glob
import os
import sys

# Change to project directory
os.chdir('/home/andrew/stanleyz')

# Add app directory to path for imports
sys.path.insert(0, '/home/andrew/stanleyz/app')
sys.path.insert(0, '/home/andrew/stanleyz')

from key_detection import detect_key_lightweight, detect_key_music21
from midi_input_script import midi_to_note


def analyze_midi_file(filepath: str, max_ticks: int = None):
    """
    Analyze a single MIDI file and detect its key.

    Args:
        filepath: Path to MIDI file
        max_ticks: Optional maximum ticks to analyze (None = all)

    Returns:
        dict with analysis results
    """
    try:
        # Read MIDI file
        notes, resolution, max_tick = midi_to_note(filepath, max_tick=max_ticks)

        if not notes:
            return {
                'success': False,
                'error': 'No notes found in file'
            }

        # Detect key using both methods
        key_lightweight = detect_key_lightweight(notes)

        try:
            key_music21 = detect_key_music21(notes)
        except Exception as e:
            key_music21 = f"Error: {e}"

        # Calculate statistics
        total_ticks = max_tick if max_ticks is None else min(max_tick, max_ticks)
        pitch_range = (min(n['pitch'] for n in notes), max(n['pitch'] for n in notes))

        return {
            'success': True,
            'key_lightweight': key_lightweight,
            'key_music21': key_music21,
            'num_notes': len(notes),
            'total_ticks': total_ticks,
            'pitch_range': pitch_range,
            'resolution': resolution
        }

    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


def main():
    """Main function to analyze multiple MIDI files"""

    # Get file pattern from command line or use default
    if len(sys.argv) > 1:
        pattern = sys.argv[1]
    else:
        pattern = '/home/andrew/stanleyz/input/acc/00*.mid'

    print("="*80)
    print("MIDI KEY DETECTION ANALYSIS")
    print("="*80)
    print(f"Pattern: {pattern}\n")

    # Find all matching files
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"No files found matching pattern: {pattern}")
        return

    print(f"Found {len(files)} file(s)\n")

    # Analyze each file
    results = []
    for filepath in files:
        filename = os.path.basename(filepath)
        print(f"Analyzing: {filename}")
        print("-" * 80)

        result = analyze_midi_file(filepath)
        result['filename'] = filename
        result['filepath'] = filepath
        results.append(result)

        if result['success']:
            print(f"  Key (Lightweight):    {result['key_lightweight']}")
            print(f"  Key (music21):        {result['key_music21']}")
            print(f"  Number of notes:      {result['num_notes']}")
            print(f"  Total ticks:          {result['total_ticks']}")
            print(f"  Pitch range:          {result['pitch_range'][0]} - {result['pitch_range'][1]}")
            print(f"  Resolution:           {result['resolution']} ticks/beat")

            # Check if methods agree
            if result['key_lightweight'] == result['key_music21']:
                print(f"  ✓ Both methods agree!")
            else:
                print(f"  ⚠ Methods disagree - may need manual verification")
        else:
            print(f"  ✗ Error: {result['error']}")

        print()

    # Summary table
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print(f"{'Filename':<20} {'Lightweight':<15} {'music21':<15} {'Notes':<8} {'Ticks':<8}")
    print("-"*80)

    for result in results:
        if result['success']:
            print(f"{result['filename']:<20} "
                  f"{result['key_lightweight']:<15} "
                  f"{result['key_music21']:<15} "
                  f"{result['num_notes']:<8} "
                  f"{result['total_ticks']:<8}")
        else:
            print(f"{result['filename']:<20} ERROR: {result['error']}")

    print("="*80)

    # Key distribution
    if any(r['success'] for r in results):
        print("\nKEY DISTRIBUTION (Lightweight Method):")
        print("-"*80)
        key_counts = {}
        for result in results:
            if result['success']:
                key = result['key_lightweight']
                key_counts[key] = key_counts.get(key, 0) + 1

        for key, count in sorted(key_counts.items()):
            print(f"  {key:<20} {count} file(s)")
        print()


if __name__ == "__main__":
    main()
