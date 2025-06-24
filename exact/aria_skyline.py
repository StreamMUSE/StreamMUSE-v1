from symusic import Score, Note, Track
from typing import List, Tuple
import argparse
import os
import glob
import concurrent.futures
from tqdm import tqdm
import multiprocessing

def extract_midi_skyline(
    midi_file_path: str, 
    max_skyline_drop: int = 12,
    max_rest_duration_sec: float = 2.0
) -> Tuple[Score, List[Note], List[Note]]:
    """
    Separates a MIDI file into a 'skyline' melody and the 'accompaniment' notes.

    - The skyline is the highest active pitch, but avoids sudden large drops.
    - The accompaniment consists of ALL other notes that are not part of the skyline.

    :param midi_file_path: Path to the MIDI file.
    :param max_skyline_drop: Max semitones the skyline can drop between consecutive notes.
    :param max_rest_duration_sec: Max duration of a rest in seconds before the skyline
                                  pitch restriction is reset.
    :return: A tuple of (original Score, skyline notes, accompaniment notes).
    """
    try:
        score = Score(midi_file_path)
    except Exception as e:
        print(f"Error loading MIDI file '{midi_file_path}': {e}")
        return Score(ticks_per_quarter=480), [], []

    all_notes = [note for track in score.tracks for note in track.notes]
    if not all_notes:
        return score, [], []

    # Convert max rest duration from seconds to MIDI ticks
    # This uses the first tempo marking as a reference.
    qpm = score.tempos[0].qpm if score.tempos else 120.0
    ticks_per_second = score.ticks_per_quarter * (qpm / 60)
    max_rest_duration_ticks = max_rest_duration_sec * ticks_per_second

    event_times = sorted(list(set(t for note in all_notes for t in (note.start, note.end))))

    skyline_notes: List[Note] = []
    accompaniment_notes: List[Note] = []
    last_accomp_note_for_pitch = {}

    for i in range(len(event_times) - 1):
        start_time = event_times[i]
        end_time = event_times[i+1]
        
        if start_time >= end_time:
            continue

        active_notes_in_interval = [
            note for note in all_notes 
            if note.start <= start_time and note.end > start_time
        ]
        
        if not active_notes_in_interval:
            continue

        highest_note = max(active_notes_in_interval, key=lambda note: note.pitch)
        
        # --- Check for significant skyline drops, but only after short rests ---
        is_valid_skyline_note = True
        if skyline_notes:
            time_since_last_skyline = start_time - skyline_notes[-1].end
            # Only apply the drop rule if the rest was shorter than the threshold
            if time_since_last_skyline < max_rest_duration_ticks:
                last_skyline_pitch = skyline_notes[-1].pitch
                if last_skyline_pitch - highest_note.pitch > max_skyline_drop:
                    is_valid_skyline_note = False
        
        # --- Process each note in the interval ---
        for note in active_notes_in_interval:
            is_the_skyline_note = (note is highest_note and is_valid_skyline_note)

            if is_the_skyline_note:
                # --- 1. Add to Skyline (with merging) ---
                if (skyline_notes and 
                    skyline_notes[-1].pitch == note.pitch and
                    skyline_notes[-1].end == start_time):
                    skyline_notes[-1].duration += (end_time - start_time)
                else:
                    skyline_notes.append(Note(
                        time=start_time,
                        duration=end_time - start_time,
                        pitch=note.pitch,
                        velocity=note.velocity
                    ))
            else:
                # --- 2. Add to Accompaniment (with merging) ---
                last_accomp_note = last_accomp_note_for_pitch.get(note.pitch)
                if (last_accomp_note and last_accomp_note.end == start_time):
                    last_accomp_note.duration += (end_time - start_time)
                else:
                    new_accomp_note = Note(
                        time=start_time,
                        duration=end_time - start_time,
                        pitch=note.pitch,
                        velocity=note.velocity
                    )
                    accompaniment_notes.append(new_accomp_note)
                    last_accomp_note_for_pitch[note.pitch] = new_accomp_note

    return score, skyline_notes, accompaniment_notes

def separate_midi_by_pitch(
    midi_file_path: str,
    num_top_notes: int = 1,
    max_melody_drop: int = 12,
    max_interval_spread: int = 24, # New parameter: max semitones between top note and others in a chord
    max_rest_duration_sec: float = 2.0
) -> Tuple[Score, List[Note], List[Note]]:
    """
    Separates a MIDI file into a 'melody' (top N notes) and 'accompaniment'.

    Args:
        midi_file_path: Path to the MIDI file.
        num_top_notes: The number of top notes to attempt to extract as melody.
        max_melody_drop: The maximum allowed pitch drop (in semitones) for the highest
                         melodic line to maintain continuity.
        max_interval_spread: Maximum pitch difference between the highest note and other
                             notes in a chord for them to be considered 'melody'.
        max_rest_duration_sec: Maximum rest in the melody before the drop check resets.

    Returns:
        A tuple containing the original score, a list of melody notes,
        and a list of accompaniment notes.
    """
    try:
        # FIX for newer symusic versions: Create Score, then set TPQ.
        score = Score(midi_file_path)
    except Exception as e:
        print(f"\n[ERROR] Error loading MIDI file '{midi_file_path}': {e}", flush=True)
        empty_score = Score()
        empty_score.ticks_per_quarter = 480
        return empty_score, [], []

    all_notes = [note for track in score.tracks for note in track.notes]
    if not all_notes:
        return score, [], []

    qpm = score.tempos[0].qpm if score.tempos else 120.0
    ticks_per_second = score.ticks_per_quarter * (qpm / 60)
    max_rest_duration_ticks = max_rest_duration_sec * ticks_per_second

    event_times = sorted(list(set(t for note in all_notes for t in (note.start, note.end))))

    melody_notes: List[Note] = []
    accompaniment_notes: List[Note] = []
    # Use dictionaries to track the last note of a given pitch for merging
    last_melody_note_for_pitch: Dict[int, Note] = {}
    last_accomp_note_for_pitch: Dict[int, Note] = {}

    last_highest_melody_note = None

    for i in range(len(event_times) - 1):
        start_time = event_times[i]
        end_time = event_times[i+1]
        if start_time >= end_time: continue

        active_notes_in_interval = [note for note in all_notes if note.start <= start_time and note.end > start_time]
        if not active_notes_in_interval:
            continue

        # --- NEW LOGIC START ---

        # 1. Sort active notes by pitch, highest first
        sorted_notes = sorted(active_notes_in_interval, key=lambda note: note.pitch, reverse=True)
        highest_note = sorted_notes[0]

        # 2. Check for melodic continuity based on the absolute highest note
        is_melodically_valid = True
        if last_highest_melody_note:
            time_since_last_melody = start_time - last_highest_melody_note.end
            if time_since_last_melody < max_rest_duration_ticks:
                if last_highest_melody_note.pitch - highest_note.pitch > max_melody_drop:
                    is_melodically_valid = False

        # 3. Determine which notes belong to melody vs. accompaniment
        notes_for_melody = set()
        if is_melodically_valid:
            # The highest note is part of the melody if valid
            notes_for_melody.add(highest_note)
            # Add up to (num_top_notes - 1) more notes if they are within the interval spread
            for j in range(1, min(num_top_notes, len(sorted_notes))):
                note = sorted_notes[j]
                if highest_note.pitch - note.pitch <= max_interval_spread:
                    notes_for_melody.add(note)
                else:
                    # Since notes are sorted, we can stop early
                    break

        # --- NEW LOGIC END ---

        # 4. Process and merge all active notes
        for note in active_notes_in_interval:
            is_melody_note = note in notes_for_melody

            # Select the target list and dictionary for merging
            target_notes = melody_notes if is_melody_note else accompaniment_notes
            last_note_map = last_melody_note_for_pitch if is_melody_note else last_accomp_note_for_pitch

            last_note = last_note_map.get(note.pitch)
            if last_note and last_note.end == start_time:
                # Extend the duration of the last note
                last_note.duration += (end_time - start_time)
            else:
                # Create a new note
                new_note = Note(time=start_time, duration=end_time - start_time, pitch=note.pitch, velocity=note.velocity)
                target_notes.append(new_note)
                last_note_map[note.pitch] = new_note

        # Update the last highest note for the next iteration's continuity check
        if notes_for_melody:
            last_highest_melody_note = max(notes_for_melody, key=lambda n: n.pitch)

    return score, melody_notes, accompaniment_notes

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

def printTypes():
    message = """Types:
    Use as type argument to specify the type
    skyline_basic - basic skyline algorithm

    """
    return message

# --- 封装的“工作函数” ---
def process_single_file(input_path: str, args):
    """
    处理单个文件的完整逻辑，方便并行调用。
    """
    try:
        original_score = None
        right_hand_notes = None
        left_hand_notes = None
        if args.type == "skyline_basic":
            original_score, right_hand_notes, left_hand_notes = extract_midi_skyline(input_path)
        elif args.type == "skyline_topk":
            original_score, right_hand_notes, left_hand_notes = separate_midi_by_pitch(input_path, args.k)
        else:
            print(printTypes())
            return
        
        if original_score is None or (not right_hand_notes and not left_hand_notes):
            return f"Skipped (no data): {input_path}"

        # --- 计算输出路径 ---
        left_hand_dir = os.path.join(args.output_dir, "acc")
        right_hand_dir = os.path.join(args.output_dir, "mel")
        
        relative_path_no_ext = os.path.splitext(os.path.relpath(input_path, args.input_dir))[0]
        output_filename = relative_path_no_ext.replace(os.sep, '_') + '.mid'

        right_hand_output_path = os.path.join(right_hand_dir, output_filename)
        left_hand_output_path = os.path.join(left_hand_dir, output_filename)
        
        # --- 保存文件 ---
        if right_hand_notes:
            save_notes_to_midi(right_hand_notes, original_score, right_hand_output_path, "Right Hand")
        if left_hand_notes:
            save_notes_to_midi(left_hand_notes, original_score, left_hand_output_path, "Left Hand")
            
        return f"{input_path} is done. Good job"
    except Exception as e:
        return f"Failed: {input_path} with error: {e}"

def process_file_wrapper(input_path: str, input_dir: str, right_hand_dir: str, left_hand_dir: str, args) -> str:
    """
    Orchestrates the processing for a single file and returns a status string.
    This is called by the main worker function.
    """
    # CHANGE: Use a flushed print for reliable diagnostic output from child processes.
    #print(f"[PID: {os.getpid()}] Starting: {os.path.relpath(input_path, input_dir)}", flush=True)

    try:
        original_score = None
        right_hand_notes = None
        left_hand_notes = None
        if args.type == "skyline_basic":
            original_score, right_hand_notes, left_hand_notes = extract_midi_skyline(input_path)
        elif args.type == "skyline_topk":
            original_score, right_hand_notes, left_hand_notes = separate_midi_by_pitch(input_path, args.k)
        else:
            print("Doing nothing LOLLLLLLLL")
            print(printTypes())
            return
        
        if original_score is None or (not right_hand_notes and not left_hand_notes):
            return f"Skipped (no notes or load error): {os.path.relpath(input_path, input_dir)}"

        relative_path_no_ext = os.path.splitext(os.path.relpath(input_path, input_dir))[0]
        output_filename = relative_path_no_ext.replace(os.sep, '_') + '.mid'
        right_hand_output_path = os.path.join(right_hand_dir, output_filename)
        left_hand_output_path = os.path.join(left_hand_dir, output_filename)

        if right_hand_notes:
            save_notes_to_midi(right_hand_notes, original_score, right_hand_output_path, "Melody")
        if left_hand_notes:
            save_notes_to_midi(left_hand_notes, original_score, left_hand_output_path, "Accompaniment")

        return f"{os.path.relpath(input_path, input_dir)} is done. OMG YOUR GOOD"
    except Exception as e:
        return f"Failed: {os.path.relpath(input_path, input_dir)} with error: {e}"


def worker(task_queue: multiprocessing.Queue, result_queue: multiprocessing.Queue, input_dir: str, right_hand_dir: str, left_hand_dir: str, args):
    """
    The main function for each worker process.
    """
    # The worker loop continues until it receives a `None` sentinel value
    for file_path in iter(task_queue.get, None):
        try:
            result = process_file_wrapper(file_path, input_dir, right_hand_dir, left_hand_dir, args)
            result_queue.put(result)
        except Exception as e:
            result_queue.put(f"WORKER CRASH on {file_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract right-hand (>=C4) and left-hand (<C4) notes from MIDI files concurrently."
        )
    parser.add_argument("--input_dir", type=str, required=True, help="Path to the root directory containing input MIDI files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the directory where separated MIDI files will be saved.")
    # 添加一个控制进程数的参数
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="Number of worker processes to use. Defaults to the number of CPU cores.")
    parser.add_argument("--type", type=str, required=True, help="Type of algorithm used to extract melody and accompanement.")
    parser.add_argument("--k", type=int, default = 1, help="Type of algorithm used to extract melody and accompanement.")
    args = parser.parse_args()


    VALID_TYPES = ["skyline_basic", "skyline_topk"] 
    
    if args.type not in VALID_TYPES:
        print(f"Error: Invalid type '{args.type}'.")
        print(printTypes()) # Now this prints in the main program
        # Exit the script immediately
        import sys
        sys.exit(1) 
        
    left_hand_dir = os.path.join(args.output_dir, "acc")
    right_hand_dir = os.path.join(args.output_dir, "mel")

    os.makedirs(left_hand_dir, exist_ok=True)
    os.makedirs(right_hand_dir, exist_ok=True)

    all_midi_files = []
    print(f"Searching for MIDI files in '{args.input_dir}'...")
    for root, _, files in os.walk(args.input_dir):
        for file in files:
            if file.lower().endswith(('.mid', '.midi')):
                all_midi_files.append(os.path.join(root, file))

    if not all_midi_files:
        print(f"No MIDI files found in '{args.input_dir}'.")
    else:
        num_files = len(all_midi_files)
        num_workers = min(args.workers, num_files) if num_files > 0 else 0
        print(f"Found {num_files} MIDI files to process using {num_workers} workers.")

        if num_workers > 0:
            task_queue = multiprocessing.Queue()
            result_queue = multiprocessing.Queue()

            processes = []
            for _ in range(num_workers):
                p = multiprocessing.Process(
                    target=worker,
                    args=(task_queue, result_queue, args.input_dir, right_hand_dir, left_hand_dir, args)
                )
                p.start()
                processes.append(p)

            for path in all_midi_files:
                task_queue.put(path)

            for _ in range(num_workers):
                task_queue.put(None)

            with tqdm(total=num_files, desc="This is a progres bar --->") as pbar:
                for _ in range(num_files):
                    result = result_queue.get()
                    # CHANGE: Use tqdm.write to print results without breaking the progress bar
                    tqdm.write(result)
                    pbar.update(1)

            for p in processes:
                p.join()

        print("\n--- File extracting process completed YIPEEEEEEEEEEE! ---")
        print(f"Acompanement files saved in: '{left_hand_dir}'")
        print(f"Melodie files saved in: '{right_hand_dir}'")