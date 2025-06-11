import os
from mido import MidiFile, MidiTrack, Message
from tqdm import tqdm
# Input and output directories
INPUT_DIR = 'dataset/POP909-Dataset'          # Replace with actual path
OUTPUT_DIR = 'dataset/POP909-Dataset/acc'          # Where to save the melody-only MIDI files
os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_acc_track(mid_path, output_path):
    midi = MidiFile(mid_path)
    for i, track in enumerate(midi.tracks):
        if 'piano' in track.name.lower() or 'bridge' in track.name.lower():
            new_midi = MidiFile()
            new_track = MidiTrack()
            new_midi.tracks.append(new_track)
            for msg in track:
                new_track.append(msg)
            new_midi.save(output_path)
            print(f"Saved acc track to: {output_path}")
            return
    print(f"No acc track found in: {mid_path}")