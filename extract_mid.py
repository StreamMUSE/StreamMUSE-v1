import os
from mido import MidiFile, MidiTrack, Message
from tqdm import tqdm
# Input and output directories
INPUT_DIR = './POP909-Dataset/POP909/'          # Replace with actual path
OUTPUT_DIR = './dataset/POP909-Dataset/mel'          # Where to save the melody-only MIDI files
os.makedirs(OUTPUT_DIR, exist_ok=True)
start = "001"

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

def extract_mel_track(mid_path, output_path):
    """
    Extracts the first track identified as a melody track (e.g., 'melody', 'lead', 'vocal')
    from a MIDI file and saves it to a new file.
    Includes basic error handling.
    """
    try:
        midi = MidiFile(mid_path)
        found_mel_track = False
        for track in midi.tracks:
            # Check for common melody track names (case-insensitive)
            # You might need to adjust these keywords based on POP909's specific naming conventions
            track_name_lower = track.name.lower()
            if 'melody' in track_name_lower or \
               'lead' in track_name_lower or \
               'vocal' in track_name_lower or \
               'main' in track_name_lower: # 'main' might catch a general main instrument melody
                
                new_midi = MidiFile()
                new_track = MidiTrack()
                new_midi.tracks.append(new_track)
                for msg in track:
                    new_track.append(msg)
                new_midi.save(output_path)
                print(f"  Saved melody track from '{os.path.basename(mid_path)}' to: {output_path}")
                found_mel_track = True
                return # Exit the function after finding and saving the first relevant track
        
        if not found_mel_track:
            print(f"  No explicit melody track found in: {os.path.basename(mid_path)}")

    except Exception as e:
        print(f"  Error processing {os.path.basename(mid_path)}: {e}")



if __name__ == "__main__":
    print(f"Starting extraction from subdirectories within: {INPUT_DIR}")
    print(f"Saving extracted tracks to: {OUTPUT_DIR}\n")

    all_midi_files_to_process = []

    # Iterate through numbered subdirectories (001 to 909)
    # This assumes the subdirectories are named numerically from "001" to "909"
    # Or, you can dynamically list directories if names are not strictly sequential
    for i in range(1, 910): # Loop from 1 up to and including 909
        subdir_name = f"{i:03d}" # Formats number as 3 digits (e.g., 1 becomes "001")
        current_subdir_path = os.path.join(INPUT_DIR, subdir_name)

        if os.path.isdir(current_subdir_path): # Check if the directory actually exists
            # Find MIDI files within this subdirectory
            for filename in os.listdir(current_subdir_path):
                if filename.lower().endswith(('.mid', '.midi')):
                    full_midi_path = os.path.join(current_subdir_path, filename)
                    all_midi_files_to_process.append(full_midi_path)
        # else:
        #     print(f"Warning: Directory '{current_subdir_path}' not found.")

    if not all_midi_files_to_process:
        print(f"No MIDI files found in any subdirectories of: {INPUT_DIR}")
    else:
        # Process each collected MIDI file with a progress bar
        for input_midi_path in tqdm(all_midi_files_to_process, desc="Processing MIDI files"):
            # Extract the base filename (e.g., "001.mid")
            midi_filename = os.path.basename(input_midi_path)
            output_midi_path = os.path.join(OUTPUT_DIR, midi_filename) # Save to the flat output directory

            extract_mel_track(input_midi_path, output_midi_path)

    print("\nExtraction process complete!")