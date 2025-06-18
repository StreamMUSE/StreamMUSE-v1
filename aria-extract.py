from symusic import Score, Note, Track
from typing import List, Tuple
import argparse
import os
import mido
from tqdm import tqdm
import glob
import random

INPUT_DIR = './POP909-Dataset/POP909/' 

def extract_midi(midi_file_path: str) -> Tuple[Score, List[Note], List[Note]]:
    """
    Extracts notes from a MIDI file, assigning them to right or left hand
    based on a simple pitch threshold.

    - Any note with a pitch >= C4 (MIDI pitch 60) is considered a "right-hand" note.
    - Any note with a pitch < C4 (MIDI pitch 60) is considered a "left-hand" note.

    :param midi_file_path: Path to the MIDI file.
    :return: A tuple containing:
             - The original symusic.Score object.
             - List of right-hand symusic.Note objects.
             - List of left-hand symusic.Note objects.
    """
    try:
        with open(midi_file_path, 'rb') as f:
            midi_data = f.read()
        score = Score.from_midi(midi_data, strict_mode=False)
    except Exception as e:
        print(f"Error loading MIDI file '{midi_file_path}': {e}")
        return Score(ticks_per_quarter=480), [], []

    right_hand_notes: List[Note] = []
    left_hand_notes: List[Note] = []

    # C4's MIDI pitch is 60.
    C4_PITCH = 60

    for track in score.tracks:
        for note in track.notes:
            if note.pitch >= C4_PITCH:
                right_hand_notes.append(note)
            else:
                left_hand_notes.append(note)

    return score, right_hand_notes, left_hand_notes

def save_notes_to_midi(
    notes: list[Note],
    original_score: Score,
    output_path: str,
    track_name: str
):
    """
    Saves a list of symusic.Note objects to a new MIDI file using the symusic library.
    This is the definitive version that correctly creates a Score and saves it.
    """
    # 1. Create a score object using the Score() factory function.
    new_score = Score()
    
    # 2. Set the 'ticks_per_quarter' attribute on the newly created object.
    new_score.ticks_per_quarter = original_score.ticks_per_quarter

    # 3. Create a new track for the notes.
    track = Track(name=track_name, program=0, is_drum=False)

    # 4. Add the notes to the track's note list.
    track.notes.extend(notes)
    
    # 5. IMPORTANT: Sort notes by start time for a valid MIDI file.
    track.notes.sort(key=lambda note: note.start)
    
    # 6. Add the completed track to the score.
    new_score.tracks.append(track)
    
    try:
        # Create the directory if it doesn't exist.
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 7. Use the correct .dump_midi() method to save the file.
        new_score.dump_midi(output_path)
        
    except Exception as e:
        print(f"Error saving MIDI file with symusic to '{output_path}': {e}")


def calculate_silence_percentage(midi_file_path: str) -> float:
    """
    Calculates the percentage of silence (no notes playing) in a MIDI file.

    Silence is defined as any period within the total duration of the score
    where no notes are currently active.

    :param midi_file_path: Path to the MIDI file.
    :return: The percentage of the MIDI file's duration that is silent (0.0 to 100.0).
             Returns 0.0 if the file cannot be loaded or has no duration.
    """
    try:
        with open(midi_file_path, 'rb') as f:
            midi_data = f.read()
        score = Score.from_midi(midi_data, strict_mode=False)
    except FileNotFoundError:
        print(f"Error: MIDI file not found at '{midi_file_path}'", file=sys.stderr)
        return 0.0
    except Exception as e:
        print(f"Error loading MIDI file '{midi_file_path}': {e}", file=sys.stderr)
        return 0.0

    # Collect all note intervals (start, end)
    all_note_intervals: List[Tuple[int, int]] = []
    max_note_end_tick = 0 # To determine the actual end of the music based on notes

    for track in score.tracks:
        for note in track.notes:
            # Ensure note duration is positive; ignore notes with zero or negative duration
            if note.duration > 0:
                note_end = note.start + note.duration
                all_note_intervals.append((note.start, note_end))
                max_note_end_tick = max(max_note_end_tick, note_end)

    # Determine the total score duration.
    # The 'score.end()' method (if it exists and is callable) typically provides
    # the tick of the last event in the entire score.
    # We prioritize score.end() if it's available and callable, otherwise fall back to max_note_end_tick.
    total_score_duration_ticks = 0
    if hasattr(score, 'end') and callable(score.end):
        try:
            total_score_duration_ticks = score.end()
        except Exception as e:
            # Fallback if calling score.end() fails for some reason
            print(f"Warning: Calling score.end() failed ({e}), falling back to max_note_end_tick.", file=sys.stderr)
            total_score_duration_ticks = max_note_end_tick
    else:
        total_score_duration_ticks = max_note_end_tick

    # Ensure total_score_duration_ticks is at least 1 if there are any notes,
    # to avoid division by zero later if max_note_end_tick was 0 but notes exist.
    if total_score_duration_ticks == 0 and all_note_intervals:
        # If notes exist but calculated duration is 0, this implies notes start and end at 0.
        # This is an unusual scenario for 'duration', so we might make it 1 to proceed
        # or indicate an error. For percentage, it's better to ensure a non-zero divisor.
        # However, it's more robust to let the logic below handle it.
        pass # The 0.0 return at the end will cover this if needed.

    # If there are no notes, the entire duration is silent.
    # If total_score_duration_ticks is 0, and no notes, 0% silent.
    if not all_note_intervals:
        return 100.0 if total_score_duration_ticks > 0 else 0.0

    # If total_score_duration_ticks is still 0 after considering all notes,
    # it means either the file is truly empty or symusic couldn't extract a meaningful duration.
    # In this case, 0% silence (as there's no duration to be silent within).
    if total_score_duration_ticks == 0:
        return 0.0


    # Sort intervals by their start time
    all_note_intervals.sort(key=lambda x: x[0])

    # Merge overlapping intervals to find the total active time
    merged_active_intervals: List[Tuple[int, int]] = []

    # Initialize with the first interval
    current_merged_start = all_note_intervals[0][0]
    current_merged_end = all_note_intervals[0][1]

    for i in range(1, len(all_note_intervals)):
        next_start, next_end = all_note_intervals[i]

        # If the next interval overlaps or touches the current merged interval
        if next_start <= current_merged_end:
            current_merged_end = max(current_merged_end, next_end)
        else:
            # No overlap, so close the current merged interval and start a new one
            merged_active_intervals.append((current_merged_start, current_merged_end))
            current_merged_start = next_start
            current_merged_end = next_end

    # Add the last merged interval
    merged_active_intervals.append((current_merged_start, current_merged_end))

    # Calculate total active duration from merged intervals
    total_active_duration_ticks = 0
    for start, end in merged_active_intervals:
        # Cap the end of the interval at the total score duration.
        # This handles cases where notes might extend beyond the official score.end
        effective_end = min(end, total_score_duration_ticks)
        if start < effective_end: # Ensure the interval actually contributes positive duration
            total_active_duration_ticks += (effective_end - start)

    # Ensure active duration does not exceed total score duration (double check)
    total_active_duration_ticks = min(total_active_duration_ticks, total_score_duration_ticks)

    # Calculate silence duration
    silence_duration_ticks = total_score_duration_ticks - total_active_duration_ticks

    # Calculate percentage
    # Avoid division by zero if total_score_duration_ticks is 0.
    if total_score_duration_ticks == 0:
        return 0.0

    percentage_silent = (silence_duration_ticks / total_score_duration_ticks) * 100.0

    return percentage_silent

def calculate_silence_percentage_measure(midi_file_path: str) -> float:
    """
    Calculates the percentage of silence in a MIDI file, where silence is defined
    as any continuous period of at least 4 beats with no notes playing.

    :param midi_file_path: Path to the MIDI file.
    :return: The percentage of the MIDI file's duration that is silent (0.0 to 100.0).
             Returns 0.0 if the file cannot be loaded or has no duration.
    """
    try:
        with open(midi_file_path, 'rb') as f:
            midi_data = f.read()
        score = Score.from_midi(midi_data, strict_mode=False)
    except FileNotFoundError:
        print(f"Error: MIDI file not found at '{midi_file_path}'", file=sys.stderr)
        return 0.0
    except Exception as e:
        print(f"Error loading MIDI file '{midi_file_path}': {e}", file=sys.stderr)
        return 0.0

    # Collect all note intervals (start, end)
    all_note_intervals: List[Tuple[int, int]] = []
    max_note_end_tick = 0

    for track in score.tracks:
        for note in track.notes:
            if note.duration > 0:
                note_end = note.start + note.duration
                all_note_intervals.append((note.start, note_end))
                max_note_end_tick = max(max_note_end_tick, note_end)

    # Determine the total score duration
    total_score_duration_ticks = 0
    if hasattr(score, 'end') and callable(score.end):
        try:
            total_score_duration_ticks = score.end()
        except Exception:
            total_score_duration_ticks = max_note_end_tick
    else:
        total_score_duration_ticks = max_note_end_tick

    # If there's no duration, there's no silence to measure.
    if total_score_duration_ticks == 0:
        return 0.0

    # If there are no notes, the entire duration is silent.
    # We must check if this total silence meets the 4-beat criteria.
    if not all_note_intervals:
        ticks_per_beat = score.ticks_per_quarter or 480 # Use a default if not present
        min_silence_duration_ticks = 4 * ticks_per_beat
        return 100.0 if total_score_duration_ticks >= min_silence_duration_ticks else 0.0

    # Sort intervals by their start time
    all_note_intervals.sort(key=lambda x: x[0])

    # Merge overlapping intervals to find continuous active periods
    merged_active_intervals: List[Tuple[int, int]] = []
    if all_note_intervals:
        current_merged_start, current_merged_end = all_note_intervals[0]
        for i in range(1, len(all_note_intervals)):
            next_start, next_end = all_note_intervals[i]
            if next_start <= current_merged_end:
                current_merged_end = max(current_merged_end, next_end)
            else:
                merged_active_intervals.append((current_merged_start, current_merged_end))
                current_merged_start, current_merged_end = next_start, next_end
        merged_active_intervals.append((current_merged_start, current_merged_end))

    # --- NEW LOGIC: Calculate silence based on a 4-beat threshold ---

    # A "beat" is a quarter note. Ticks per quarter note gives us ticks per beat.
    ticks_per_beat = score.ticks_per_quarter
    # Handle cases with invalid TPQ in the MIDI file
    if not isinstance(ticks_per_beat, int) or ticks_per_beat <= 0:
        print(f"Warning: Invalid ticks_per_quarter ({ticks_per_beat}) in {midi_file_path}. Using 480.", file=sys.stderr)
        ticks_per_beat = 480

    # Define the minimum silence duration in ticks for it to be counted
    min_silence_duration_ticks = 4 * ticks_per_beat

    total_qualified_silence_ticks = 0
    last_event_end = 0

    # Iterate through the merged active intervals to find the silent gaps between them
    for active_start, active_end in merged_active_intervals:
        # Calculate the duration of the silence *before* this active interval
        silence_duration = active_start - last_event_end

        # Check if this silence meets the 4-beat threshold
        if silence_duration >= min_silence_duration_ticks:
            total_qualified_silence_ticks += silence_duration

        # Update the end of the last event to the end of the current active interval
        last_event_end = active_end

    # After the loop, check for silence from the end of the last note to the end of the score
    final_silence_duration = total_score_duration_ticks - last_event_end
    if final_silence_duration >= min_silence_duration_ticks:
        total_qualified_silence_ticks += final_silence_duration

    # Calculate percentage based on the qualified silence
    percentage_silent = (total_qualified_silence_ticks / total_score_duration_ticks) * 100.0

    return percentage_silent

if __name__ == "__main__":
    
    ##CODE TO GET SILENCE OF MIDI FILES IN ENTIRE DIRECTORY
    parser = argparse.ArgumentParser(description="Calculate the average silence percentage for all MIDI files in a directory.")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to the directory containing MIDI files.")
    args = parser.parse_args()

    input_directory = args.input_dir

    if not os.path.isdir(input_directory):
        print(f"Error: Input directory not found at '{input_directory}'")
    else:
        total_percentage = 0
        midi_file_count = 0

        # Walk through the directory
        for root, dirs, files in os.walk(input_directory):
            for file in sorted(files):  # Sorting ensures a consistent order (e.g., 001, 002, ...)
                if file.endswith(".mid"):
                    input_midi_path = os.path.join(root, file)
                    try:
                        percentage = calculate_silence_percentage_measure(input_midi_path)
                        print(f"Processing {input_midi_path}: {percentage:.2f}% silence")
                        total_percentage += percentage
                        midi_file_count += 1
                    except Exception as e:
                        print(f"Could not process {input_midi_path}. Error: {e}")

        if midi_file_count > 0:
            average_silence = total_percentage / midi_file_count
            print("\n--------------------------------------------------")
            print(f"Processed {midi_file_count} MIDI files.")
            print(f"Average silence percentage across all files: {average_silence:.2f}%")
        else:
            print("No MIDI files found in the specified directory.")

    ## CODE FOR EXTRACTING RANDOM LEFT AND RIGHT HAND FROM MIDI
    # parser = argparse.ArgumentParser(description="Extract right-hand (>=C4) and left-hand (<C4) notes from a random sample of MIDI files in a directory.")
    # parser.add_argument("--input_dir", type=str, required=True, help="Path to the root directory containing input MIDI files (e.g., ./aria-dataset/data/).")
    # parser.add_argument("--output_dir", type=str, required=True, help="Path to the directory where separated MIDI files will be saved.")
    # parser.add_argument("--sample_size", type=int, default=80000, help="The number of MIDI files to randomly sample for processing.")
    # args = parser.parse_args()

    # left_hand_dir = os.path.join(args.output_dir, "left_hand")
    # right_hand_dir = os.path.join(args.output_dir, "right_hand")

    # os.makedirs(left_hand_dir, exist_ok=True)
    # os.makedirs(right_hand_dir, exist_ok=True)

    # # --- MODIFIED LOGIC: Use glob to recursively find all MIDI files ---
    # print(f"Searching for MIDI files in '{args.input_dir}'...")
    # # Construct a platform-independent search pattern for .mid files
    # search_pattern_mid = os.path.join(args.input_dir, '**', '*.mid')
    # all_midi_files = glob.glob(search_pattern_mid, recursive=True)

    # # Also search for .midi files
    # search_pattern_midi = os.path.join(args.input_dir, '**', '*.midi')
    # all_midi_files.extend(glob.glob(search_pattern_midi, recursive=True))

    # if not all_midi_files:
    #     print(f"No MIDI files found in '{args.input_dir}'.")
    # else:

    #     print(f"Found {len(all_midi_files)} total MIDI files.")

    #     # --- NEW LOGIC: Randomly sample a subset of the files ---
    #     num_to_sample = args.sample_size
    #     if len(all_midi_files) > num_to_sample:
    #         print(f"Randomly sampling {num_to_sample} files for processing...")
    #         files_to_process = random.sample(all_midi_files, num_to_sample)
    #     else:
    #         print(f"Found {len(all_midi_files)} files, which is less than the sample size. Processing all available files.")
    #         files_to_process = all_midi_files

    #     # Process each sampled MIDI file
    #     print(f"\nProcessing {len(files_to_process)} files...")
    #     for input_path in tqdm(files_to_process, desc="Processing MIDI files"):
    #         # Assuming extract_midi and save_notes_to_midi are defined elsewhere
    #         original_score, right_hand_notes, left_hand_notes = extract_midi(input_path)

    #         if original_score is None:
    #             continue

    #         relative_path_no_ext = os.path.splitext(os.path.relpath(input_path, args.input_dir))[0]
    #         # Replace directory separators with an underscore for a flat, unique filename
    #         output_filename = relative_path_no_ext.replace(os.sep, '_') + '.mid'

    #         right_hand_output_path = os.path.join(right_hand_dir, output_filename)
    #         left_hand_output_path = os.path.join(left_hand_dir, output_filename)
            
    #         if right_hand_notes:
    #             save_notes_to_midi(right_hand_notes, original_score, right_hand_output_path, "Right Hand")
    #         if left_hand_notes:
    #             save_notes_to_midi(left_hand_notes, original_score, left_hand_output_path, "Left Hand")

    #     print("\nExtraction process complete!")
    #     print(f"Left hand files saved in: '{left_hand_dir}'")
    #     print(f"Right hand files saved in: '{right_hand_dir}'")

    ## CODE FOR EXTRACTING LEFT AND RIGHT HAND FROM MIDI
    # parser = argparse.ArgumentParser(description="Extract right-hand (>=C4) and left-hand (<C4) notes from all MIDI files in a directory.")
    # parser.add_argument("--input_dir", type=str, required=True, help="Path to the root directory containing input MIDI files.")
    # parser.add_argument("--output_dir", type=str, required=True, help="Path to the directory where separated MIDI files will be saved.")
    # args = parser.parse_args()

    # # --- NEW LOGIC: Define the specific output folders ---
    # left_hand_dir = os.path.join(args.output_dir, "left_hand")
    # right_hand_dir = os.path.join(args.output_dir, "right_hand")

    # # --- NEW LOGIC: Create these folders at the start ---
    # os.makedirs(left_hand_dir, exist_ok=True)
    # os.makedirs(right_hand_dir, exist_ok=True)

    # # Find all MIDI files in the input directory and its subdirectories
    # all_midi_files = []
    # for root, _, files in os.walk(args.input_dir):
    #     for file in files:
    #         if file.lower().endswith(('.mid', '.midi')):
    #             all_midi_files.append(os.path.join(root, file))

    # if not all_midi_files:
    #     print(f"No MIDI files found in '{args.input_dir}'.")
    # else:

    #     print(f"Found {len(all_midi_files)} MIDI files to process.")

    #     # Process each MIDI file
    #     for input_path in tqdm(all_midi_files, desc="Processing MIDI files"):
    #         original_score, right_hand_notes, left_hand_notes = extract_midi(input_path)

    #         if original_score is None:
    #             continue

    #         # --- MODIFIED LOGIC: Create a unique flat filename ---
    #         # This prevents collisions if you have files with the same name in different subfolders.
    #         # For example, 'subdir/song.mid' becomes 'subdir_song.mid'
    #         relative_path_no_ext = os.path.splitext(os.path.relpath(input_path, args.input_dir))[0]
    #         output_filename = relative_path_no_ext.replace(os.sep, '_') + '.mid'

    #         # --- MODIFIED LOGIC: Define output paths for the new structure ---
    #         right_hand_output_path = os.path.join(right_hand_dir, output_filename)
    #         left_hand_output_path = os.path.join(left_hand_dir, output_filename)
            
    #         # Save the new MIDI files to their respective folders
    #         if right_hand_notes:
    #             save_notes_to_midi(right_hand_notes, original_score, right_hand_output_path, "Right Hand")
    #         if left_hand_notes:
    #             save_notes_to_midi(left_hand_notes, original_score, left_hand_output_path, "Left Hand")

    #     print("\nExtraction process complete!")
    #     print(f"Left hand files saved in: '{left_hand_dir}'")
    #     print(f"Right hand files saved in: '{right_hand_dir}'")




