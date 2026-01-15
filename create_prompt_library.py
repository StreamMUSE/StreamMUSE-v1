#!/usr/bin/env python3
"""
Create prompt library structure from POP909 dataset
"""

import os
import json
import shutil

# Load the key map
with open('pop909_key_map.json', 'r') as f:
    key_map = json.load(f)

# Define the keys we want to include
TOP_PRIORITY_KEYS = {
    'C_major': {'description': 'C major - Most common key, no sharps/flats'},
    'G_major': {'description': 'G major - Very common, 1 sharp'},
    'D_major': {'description': 'D major - Common, 2 sharps'},
    'F_major': {'description': 'F major - Common, 1 flat'},
    'A_major': {'description': 'A major - Fairly common, 3 sharps'},
    'A_minor': {'description': 'A minor - Most common minor key'}
}

SECONDARY_KEYS = {
    'E_minor': {'description': 'E minor - Common, relative to G major'},
    'D_minor': {'description': 'D minor - Common, relative to F major'},
    'B_minor': {'description': 'B minor - Fairly common, 2 sharps'}
}

# Combine all keys
ALL_KEYS = {**TOP_PRIORITY_KEYS, **SECONDARY_KEYS}

# Base paths
POP909_PATH = '/home/andrew/stanleyz/input/pop909_dataset'
PROMPT_LIB_PATH = '/home/andrew/stanleyz/prompts'

def create_prompt_library():
    """Create the prompt library structure"""

    print("="*80)
    print("CREATING PROMPT LIBRARY")
    print("="*80)
    print()

    # Create base prompts directory
    os.makedirs(PROMPT_LIB_PATH, exist_ok=True)
    print(f"✓ Created base directory: {PROMPT_LIB_PATH}")

    # Build metadata structure
    metadata = {}

    # Process each song from key map
    songs = key_map['songs']

    for key_name, key_info in ALL_KEYS.items():
        # Find all songs with perfect matches for this key
        matching_songs = [
            s for s in songs
            if s['melody_key'] == key_name and s['keys_match']
        ]

        if not matching_songs:
            print(f"⚠ Warning: No matching songs found for {key_name}")
            continue

        # Create key directory
        key_dir = os.path.join(PROMPT_LIB_PATH, key_name)
        os.makedirs(key_dir, exist_ok=True)

        print(f"\nProcessing {key_name}:")
        print(f"  Found {len(matching_songs)} matching songs")

        # Initialize metadata for this key
        metadata[key_name] = []

        # Copy MIDI files for each matching song
        for i, song in enumerate(matching_songs):
            song_num = song['song_number']

            # Source paths
            mel_src = os.path.join(POP909_PATH, 'mel', f'{song_num}.mid')
            acc_src = os.path.join(POP909_PATH, 'acc', f'{song_num}.mid')

            # Destination paths
            mel_dst = os.path.join(key_dir, f'pop909_{song_num}_mel.mid')
            acc_dst = os.path.join(key_dir, f'pop909_{song_num}_acc.mid')

            # Copy files
            if os.path.exists(mel_src) and os.path.exists(acc_src):
                shutil.copy2(mel_src, mel_dst)
                shutil.copy2(acc_src, acc_dst)

                # Add to metadata
                metadata[key_name].append({
                    'name': f'pop909_{song_num}',
                    'melody_path': mel_dst,
                    'accompaniment_path': acc_dst,
                    'duration_ticks': song['melody_ticks'],
                    'source': 'POP909',
                    'song_number': song_num,
                    'melody_notes_count': song['melody_notes_count'],
                    'accompaniment_notes_count': song['accompaniment_notes_count']
                })

                print(f"  ✓ Copied {song_num}.mid")
            else:
                print(f"  ✗ Error: Source files not found for {song_num}")

    # Save metadata.json
    metadata_path = os.path.join(PROMPT_LIB_PATH, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print()
    print("="*80)
    print("SUMMARY")
    print("="*80)

    total_prompts = sum(len(v) for v in metadata.values())
    print(f"Total keys: {len(metadata)}")
    print(f"Total prompts: {total_prompts}")
    print()

    print("Prompts per key:")
    for key_name in sorted(metadata.keys()):
        count = len(metadata[key_name])
        priority = "TOP" if key_name in TOP_PRIORITY_KEYS else "SECONDARY"
        print(f"  {key_name:<15} {count} prompts [{priority}]")

    print()
    print(f"✓ Metadata saved to: {metadata_path}")
    print(f"✓ Library created at: {PROMPT_LIB_PATH}")
    print()
    print("="*80)
    print("PROMPT LIBRARY READY!")
    print("="*80)
    print()
    print("Usage:")
    print(f"  python app/client.py \\")
    print(f"      --listening-duration-ticks 64 \\")
    print(f"      --prompt-library-path prompts \\")
    print(f"      --key-detection-method lightweight \\")
    print(f"      --tempo 120")
    print()


if __name__ == "__main__":
    create_prompt_library()
