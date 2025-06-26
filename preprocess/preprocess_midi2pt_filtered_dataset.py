import numpy as np
import xf_midi
from settings import RWC_DATASET_PATH, LA_DATASET_PATH, NOTTINGHAM_DATASET_PATH
import os
from joblib import Parallel, delayed
import torch
import shutil
import json
import pretty_midi
import argparse

tokenize_dict = {'<sos>': 0, '<eos>': 1, '<pad>': 2}
tokenize_count = [-1, -1, -1]

DURATION_TEMPLATES = np.array([1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096])

def update_token_dict(midi_path, beat_div=4):
    """
    Updates the global tokenize_dict with unique (program, pitch, duration) tuples
    found in the MIDI file.
    """
    try:
        # Initialize XFMidi object with a constant tempo based on beat division
        midi = xf_midi.XFMidi(midi_path, constant_tempo=60.0 / beat_div)
    except Exception:
        # Return None if the MIDI file cannot be processed
        return None
    
    # Get the end time of the MIDI, rounded to an integer
    midi_end_time = int(midi.get_end_time())
    if midi_end_time <= 0:
        return
    
    # Define duration boundaries for quantization
    duration_boundaries = (DURATION_TEMPLATES[1:] + DURATION_TEMPLATES[:-1]) / 2
    
    # Iterate through each instrument in the MIDI file
    for ins in midi.instruments:
        program = ins.program
        # If it's a drum track, set program to 127
        if ins.is_drum:
            program = 127
        
        # Iterate through each note in the instrument
        for note in ins.notes:
            # Round start and end times to integers
            start_time = int(round(note.start))
            end_time = int(round(note.end))
            
            # Process note only if it's within valid time range
            if start_time >= 0 and end_time < midi_end_time:
                if ins.is_drum:
                    duration = 0 # Drums have a duration of 0 in this context
                else:
                    # Quantize duration to the nearest template value
                    duration = np.searchsorted(duration_boundaries, end_time - start_time)
                
                # Create a unique key for the note
                key = (program, note.pitch, duration)
                
                # Add key to tokenize_dict if it's new
                if key not in tokenize_dict:
                    tokenize_dict[key] = len(tokenize_dict)
                    tokenize_count.append(0)
                
                # Increment count for the key
                tokenize_count[tokenize_dict[key]] += 1

def preprocess_midi(midi_path, max_polyphony, beat_div=4, ins_ids='all'):
    """
    Preprocesses a MIDI file into a numerical tensor representation,
    trimming leading and trailing empty time steps.

    Args:
        midi_path (str): Path to the MIDI file.
        max_polyphony (int): Maximum number of notes allowed per time step.
        beat_div (int): Beat division for tempo calculation.
        ins_ids (str or list): Instrument filter. 'all' for all instruments,
                                'drum' for drums, 'nondrum' for non-drums,
                                or specific track selections.

    Returns:
        tuple: A tuple containing the processed tensor [T, 3*max_polyphony]
               and a pitch shift range tensor [pitch_shift_min, pitch_shift_max],
               or None if the MIDI file is invalid or empty.
    """
    try:
        # Initialize XFMidi object
        midi = xf_midi.XFMidi(midi_path, constant_tempo=60.0 / beat_div)
    except Exception as e:
        print(f"Error processing {midi_path}: Invalid MIDI file. Error: {e}")
        return None
    
    # Get the rounded end time of the MIDI
    midi_end_time = int(midi.get_end_time())
    if midi_end_time <= 0:
        return None # Return None if MIDI is empty or has no duration
    
    # Ensure ins_ids is a list for consistent iteration
    if not isinstance(ins_ids, list):
        ins_ids = [ins_ids]
    
    # Define duration boundaries for quantizing note durations
    duration_boundaries = (DURATION_TEMPLATES[1:] + DURATION_TEMPLATES[:-1]) / 2
    
    min_pitch = 127 # Initialize min_pitch to highest possible MIDI pitch
    max_pitch = 0   # Initialize max_pitch to lowest possible MIDI pitch
    
    result_rolls = [] # List to store processed rolls for different instrument filters
    
    # Process for each specified instrument ID filter
    for ins_id in ins_ids:
        has_any_note = False # Flag to check if any note was processed for current ins_id
        # Initialize rolls array: [time, polyphony_slot, (program, pitch, duration)]
        # Filled with 255 (padding value)
        rolls = np.full((midi_end_time, max_polyphony, 3), dtype=np.uint8, fill_value=255)
        # polyphony_counts tracks how many notes are at each time step
        polyphony_counts = np.zeros(midi_end_time, dtype=np.uint8)
        
        # Iterate through instruments in the MIDI file
        for i, ins in enumerate(midi.instruments):
            program = ins.program
            if ins.is_drum:
                program = 127 # Special program for drums
            
            # Iterate through notes of the current instrument
            for note in ins.notes:
                start_time = int(round(note.start))
                end_time = int(round(note.end))
                
                # Check if note is within bounds and polyphony limit
                if start_time >= 0 and end_time < midi_end_time and polyphony_counts[start_time] < max_polyphony:
                    if ins.is_drum:
                        duration = 0 # Drum notes have zero duration in this representation
                    else:
                        # Quantize duration
                        duration = np.searchsorted(duration_boundaries, end_time - start_time)
                        min_pitch = min(min_pitch, note.pitch) # Update min_pitch
                        max_pitch = max(max_pitch, note.pitch) # Update max_pitch
                    
                    add_note = False # Flag to determine if the note should be added based on ins_id
                    
                    # Logic to filter notes based on ins_id
                    if ins_id == 'all':
                        add_note = True
                    elif isinstance(ins_id, int):
                        raise NotImplementedError # Not implemented for integer ins_ids
                    elif isinstance(ins_id, str):
                        if '-' in ins_id: # For 'track-N', 'upto-N', 'from-N', 'notrack-N'
                            task, num = ins_id.split('-')
                            num = int(num)
                            if task == 'track':
                                add_note = i == num
                            elif task == 'upto':
                                add_note = i <= num
                            elif task == 'from':
                                add_note = i >= num
                            elif task == 'notrack':
                                add_note = i != num
                            else:
                                raise NotImplementedError
                        elif ins_id == 'drum':
                            add_note = ins.is_drum
                        elif ins_id == 'nondrum':
                            add_note = not ins.is_drum
                        elif ins_id == 'empty':
                            add_note = False # Explicitly do not add notes for 'empty'
                        else:
                            raise NotImplementedError
                    else:
                        raise NotImplementedError
                    
                    # If note should be added, update rolls and polyphony_counts
                    if add_note:
                        has_any_note = True
                        rolls[start_time, polyphony_counts[start_time]] = [program, note.pitch, duration]
                        polyphony_counts[start_time] += 1
        
        # If no notes were processed for a non-empty instrument ID, return None (invalid MIDI)
        if not has_any_note and ins_id != 'empty':
            return None
        
        # After processing all notes for an instrument ID:
        # Sort notes within each time step and add EOS token for empty slots
        for t in range(midi_end_time):
            # Sort notes by program, then pitch, then duration
            if polyphony_counts[t] > 0:
                rolls[t, :polyphony_counts[t]] = rolls[t, :polyphony_counts[t]][
                    np.lexsort((rolls[t, :polyphony_counts[t], 2], # Duration
                                rolls[t, :polyphony_counts[t], 1], # Pitch
                                rolls[t, :polyphony_counts[t], 0]))] # Program
            
            # If there's space in the polyphony slots, mark the end of notes for this time step
            if polyphony_counts[t] < max_polyphony:
                rolls[t, polyphony_counts[t], 0] = 254 # EOS token for slot (program 254)
        
        result_rolls.append(rolls) # Add processed roll to the list
    
    # Concatenate all processed rolls along the polyphony dimension
    result_rolls = np.concatenate(result_rolls, axis=1)

    # --- New logic to trim leading and trailing empty time steps ---
    first_active_frame = -1
    last_active_frame = -1

    # Find the first and last time steps that contain any notes
    # A time step is considered 'active' if its polyphony_count > 0, meaning it has at least one note.
    for t in range(midi_end_time):
        # Check if the current time step has any notes
        if polyphony_counts[t] > 0:
            if first_active_frame == -1:
                first_active_frame = t # Set the first active frame
            last_active_frame = t     # Continuously update the last active frame

    # If no active frames are found (entire MIDI is empty after filtering)
    if first_active_frame == -1:
        return None # Return None if the MIDI results in no active notes

    # Slice the result_rolls to keep only the active portion
    result_rolls = result_rolls[first_active_frame : last_active_frame + 1]
    
    # Get song-level pitch shift range (only considers non-drum pitches)
    # min_pitch and max_pitch were updated only for non-drum notes
    pitch_shift_max = 127 - max_pitch
    pitch_shift_min = -min_pitch
    
    # Convert numpy array to torch tensor and return
    return torch.tensor(result_rolls), torch.tensor([pitch_shift_min, pitch_shift_max], dtype=torch.int8)

def tensor_to_midi(
    rolls: torch.LongTensor,
    save_path: str,
    tempo: float = 120.0,
    instrument_program: int = 0,
):
    """
    Converts a tensor representation back into a MIDI file.

    Args:
        rolls (torch.LongTensor): [T, 3*max_polyphony] tensor, where each row is
                                  [prog0, pitch0, dur0, prog1, pitch1, dur1, ...].
                                  prog==254 -> EOS for slot, pitch==255 -> PAD.
        save_path (str): The path where the MIDI file will be saved.
        tempo (float): Playback tempo in BPM.
        instrument_program (int): General MIDI program number for the instrument (0=Steinway Grand Piano).
    """
    # 1) Set up PrettyMIDI object and add one instrument (e.g., piano)
    pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    piano = pretty_midi.Instrument(program=instrument_program, is_drum=False)
    pm.instruments.append(piano)

    # 2) Calculate real-time duration per time-step.
    # A time step here corresponds to a 16th note if beat_div=4 and tempo is in BPM.
    time_step = 60.0 / tempo / 4 

    T, C = rolls.shape # T = number of time steps, C = 3 * max_polyphony
    max_poly = C // 3  # Calculate max_polyphony from the tensor shape

    # 3) Iterate through each time step of the tensor
    for t in range(T):
        row = rolls[t] # Get the current time step's data
        for slot in range(max_poly):
            base = slot * 3 # Calculate the starting index for the current note slot
            prog  = int(row[base].item())     # Get program
            pitch = int(row[base + 1].item()) # Get pitch
            dur_i = int(row[base + 2].item()) # Get duration index

            # If program is 254, it's an EOS token for this slot, meaning no more notes in this time step.
            # We can break early if we encounter an EOS token in a slot.
            if prog == 254:
                 break
            
            # If pitch is 255, it's a padding value, skip this slot.
            if pitch == 255:
                continue
            
            # Sanity checks for valid pitch and duration index
            if not (0 <= pitch < 128):
                continue
            if not (0 <= dur_i < len(DURATION_TEMPLATES)):
                continue

            # Calculate start and end times for the note in seconds
            start = t * time_step
            end   = start + DURATION_TEMPLATES[dur_i] * time_step
            
            # Append the note to the piano instrument
            piano.notes.append(
                pretty_midi.Note(
                    velocity=100, # Fixed velocity
                    pitch=pitch,
                    start=start,
                    end=end
                )
            )

    # 4) Write the PrettyMIDI object to a MIDI file
    os.makedirs(os.path.dirname(save_path), exist_ok=True) # Ensure output directory exists
    pm.write(save_path)

def create_tokenize_dict(folder):
    """
    Creates a tokenization dictionary by scanning all MIDI files in a folder.
    """
    midi_files = []
    # Recursively find all .mid and .MID files
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith('.mid') or file.endswith('.MID'):
                midi_files.append(os.path.join(root, file))
    
    from tqdm import tqdm
    # Process each MIDI file to update the token dictionary
    for i, midi_file in enumerate(tqdm(midi_files)):
        update_token_dict(midi_file)
        if i % 100 == 0:
            print('Tokenize dict size:', len(tokenize_dict))
    print('Final Tokenize dict size:', len(tokenize_dict))

def create_npy_dataset_from_midi(folders,
                                 max_polyphony,
                                 dataset_name,
                                 ins_ids='all',
                                 scan_subfolders=True, # This parameter is not used in the current implementation.
                                 max_idx=None):
    """
    Creates a dataset of processed MIDI tensors (and corresponding lengths/pitch shifts)
    from a list of folders, saving them as .pt files.
    
    Args:
        folders (list): List of paths to directories containing MIDI files.
        max_polyphony (int): Maximum polyphony per time step.
        dataset_name (str): Base name for the output .pt files.
        ins_ids (str or list): Instrument filter for preprocessing.
        scan_subfolders (bool): Whether to scan subfolders (currently ignored).
        max_idx (int): Maximum number of MIDI files to process.

    Returns:
        list: Paths to the saved tensor and length files.
    """

    # 1. Collect all .mid/.MID files from the specified folders
    midi_files = []
    for folder in folders:
        # Check if the folder exists
        if not os.path.exists(folder):
            print(f"Warning: Folder not found: {folder}. Skipping.")
            continue
        for filename in os.listdir(folder):
            if filename.lower().endswith('.mid'):
                midi_files.append(os.path.join(folder, filename))
    midi_files = sorted(midi_files) # Sort files for consistent processing order
    
    # Apply max_idx limit if specified
    if max_idx is not None:
        midi_files = midi_files[:max_idx]
    print(f'Processing {len(midi_files)} files...')

    # 2. Define a worker function for parallel processing
    def safe_preprocess(midi_path):
        try:
            # Call preprocess_midi which returns (data, shift) or None
            data, shift = preprocess_midi(midi_path, max_polyphony, ins_ids=ins_ids)
            return (midi_path, data, shift, None) # Success: return data and None for error
        except Exception as e:
            # Failure: return None for data and the error message
            return (midi_path, None, None, str(e))

    # 3. Execute preprocessing in parallel
    results = Parallel(n_jobs=-1, verbose=10)(
        delayed(safe_preprocess)(midi_path) for midi_path in midi_files
    )

    # 4. Separate successful and failed results
    successful_data = []
    successful_shifts = []
    lengths = []
    failed_list = [] # To record failed files and their error messages

    for midi_path, data, shift, error in results:
        if error is None and data is not None: # Check if data is not None (i.e., not an empty MIDI after trimming)
            successful_data.append(data)
            successful_shifts.append(shift)
            lengths.append(len(data)) # Store the length of the processed tensor
        else:
            failed_list.append((midi_path, error if error else "MIDI became empty after trimming."))

    # 5. Report failed files
    if failed_list:
        print("\n----- The following files failed processing or became empty after trimming: -----")
        for path, err_msg in failed_list:
            print(f"  - {path}  →  Error: {err_msg}")
        print("---------------------------------------------------------------------------------\n")

    # 6. Concatenate and save successful data if any exists
    if successful_data:
        all_data = torch.cat(successful_data, dim=0)
        all_shifts = torch.cat(successful_shifts, dim=0)
        
        # Ensure 'data' directory exists
        os.makedirs('data', exist_ok=True)

        # Save to disk
        data_path = f'data/{dataset_name}.pt'
        shift_path = f'data/{dataset_name}.pitch_shift_range.pt'
        length_path = f'data/{dataset_name}.length.pt'

        torch.save(all_data, data_path)
        torch.save(all_shifts, shift_path)
        torch.save(torch.tensor(lengths), length_path)
        
        print(f"Successfully processed {len(successful_data)} files. Data saved to 'data/' directory.")
    else:
        print("No successful MIDI files processed. Skipping data saving.")
        return None # Return None if no data was successfully processed

    return [data_path, length_path]

def sync(path_m, path_a):
    """
    Synchronizes two datasets (e.g., melody and accompaniment) by trimming
    each pair of corresponding sequences to their minimum length.
    
    Args:
        path_m (list): [path_to_melody_tensor, path_to_melody_lengths]
        path_a (list): [path_to_accompaniment_tensor, path_to_accompaniment_lengths]
    """
    if path_m is None or path_a is None:
        print("Cannot sync: one or both datasets are empty or failed to process.")
        return

    tensor_m_path = path_m[0]
    length_m_path = path_m[1]

    tensor_a_path = path_a[0]
    length_a_path = path_a[1]

    mel = torch.load(tensor_m_path)
    acc = torch.load(tensor_a_path)

    length_mel = torch.load(length_m_path)
    length_acc = torch.load(length_a_path)

    # Calculate the minimum length for each corresponding pair of sequences
    min_lengths = torch.minimum(length_mel, length_acc)

    adjusted_tensor1 = []
    adjusted_tensor2 = []
    start_idx1 = 0
    start_idx2 = 0

    # Iterate through each sequence pair and trim them
    for i in range(len(min_lengths)):
        # Trim subtensors to the calculated minimum length
        adjusted_tensor1.append(mel[start_idx1 : start_idx1 + min_lengths[i]])
        adjusted_tensor2.append(acc[start_idx2 : start_idx2 + min_lengths[i]])

        # Update start indices for the next sequence
        start_idx1 += length_mel[i]
        start_idx2 += length_acc[i]

    # Concatenate the adjusted subtensors back into single tensors
    adjusted_tensor1 = torch.cat(adjusted_tensor1)
    adjusted_tensor2 = torch.cat(adjusted_tensor2)
    
    adjusted_tensor1 = adjusted_tensor1.reshape(adjusted_tensor1.shape[0],adjusted_tensor1.shape[1]*adjusted_tensor1.shape[2])
    adjusted_tensor2 = adjusted_tensor2.view(adjusted_tensor2.shape[0],adjusted_tensor2.shape[1]*adjusted_tensor2.shape[2])

    # Save the updated lengths and trimmed tensors back to their original paths
    torch.save(min_lengths, length_m_path)
    torch.save(min_lengths, length_a_path)
   
    torch.save(adjusted_tensor1, tensor_m_path)
    torch.save(adjusted_tensor2, tensor_a_path)
    
    print(f'Melody tensor shape after sync: {adjusted_tensor1.shape}')
    print(f'Accompaniment tensor shape after sync: {adjusted_tensor2.shape}')
    print('Tensor matching complete.')

def create_tensor(dataset_name, folders, max_polyphony=4):
    """
    Wrapper function to create processed tensors from MIDI folders.
    """
    paths = create_npy_dataset_from_midi(folders, max_polyphony, f'{dataset_name}_cp{max_polyphony}', scan_subfolders=False)
    return paths

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process midi folder(s) into usable tensors for the task")
    
    # Positional arguments for dataset naming and folder paths
    parser.add_argument("--name", type=str, help="Name your dataset")
    parser.add_argument("--folders", nargs='+', type=str, help="Paths to midi folders (e.g., 'path/to/RWC/dataset' 'path/to/LA/dataset')")
    parser.add_argument("--polyphony", type=int, default=4, help="Maximum number of notes allowed in one timestep (default: 4)")

    args = parser.parse_args()

    # Construct paths for melody and accompaniment subfolders
    melody_f = [os.path.join(f, 'mel') for f in args.folders]
    accomp_f = [os.path.join(f, 'acc') for f in args.folders]
    
    # Define dataset names for melody and accompaniment
    melody_n = args.name + '_mel'
    accomp_n = args.name + '_acc'

    print(f"Creating melody tensor from: {melody_f}")
    paths_m = create_tensor(melody_n, melody_f, max_polyphony=args.polyphony)
    
    print(f"Creating accompaniment tensor from: {accomp_f}")
    paths_a = create_tensor(accomp_n, accomp_f, max_polyphony=args.polyphony)
    
    # Synchronize the created melody and accompaniment tensors
    sync(paths_m, paths_a)