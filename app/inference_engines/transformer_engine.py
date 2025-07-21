import torch
import os
import numpy as np
import time
import sys
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt

# Add the project root to the Python path to allow for absolute imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from m2a_transformer import RoFormerSymbolicTransformer, EOS_TOKEN, PAD_TOKEN
from preprocess.preprocess_midi2pt_dataset import DURATION_TEMPLATES

class TransformerInferenceEngine:
    """
    This class wraps the RoFormerSymbolicTransformer model and handles all the
    data processing required for real-time interactive music generation. It manages
    the performance history, prepares input tensors for the model, and decodes
    the model's output back into playable musical notes.
    """
    def __init__(self, checkpoint_path: str, max_polyphony=4, generation_length_frames=20, model_max_seq_len_frames=384):
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        # for debugging purposes
        self.save_x_counter = 0  # 新增计数器
        self.x_save_dir = "app/logs/server/x_distribution"
        os.makedirs(self.x_save_dir, exist_ok=True)

        print(f"Loading model from: {checkpoint_path}")
        # Determine model size from checkpoint path name
        if 'small' in checkpoint_path:
            self.model = RoFormerSymbolicTransformer.load_from_checkpoint(checkpoint_path, 
                                                                          large=False, 
                                                                          map_location=lambda storage, loc: storage.cuda(0) if torch.cuda.is_available() else 'cpu'
                                                                          )
        else:
            self.model = RoFormerSymbolicTransformer.load_from_checkpoint(checkpoint_path, 
                                                                          large=True, 
                                                                          map_location=lambda storage, loc: storage.cuda(0) if torch.cuda.is_available() else 'cpu'
                                                                          )
        
        if torch.cuda.is_available():
            self.model.to('cuda:0')
        self.model.eval()
        print("Model loaded successfully.")
        
        # The model's max sequence length (the total number of tokens it can process at once).
        # This is measured in "frames", where one tick of music contains two frames:
        # one for the melody and one for the accompaniment.
        self.model_max_seq_len_frames = model_max_seq_len_frames
        
        # The effective history length, in musical ticks, that the model uses as a prompt.
        # This must be half the frame length because ticks are interleaved (mel, acc, mel, acc...).
        self.prompt_length_ticks = self.model_max_seq_len_frames // 2
        
        # The maximum number of simultaneous notes allowed at a single tick.
        self.max_polyphony = max_polyphony
        
        # The number of frames (melody + accompaniment) to generate in one go.
        self.generation_length_frames = generation_length_frames
        
        # Stateful history of all notes played during the session.
        # These are stored with absolute ticks relative to the start of the performance.
        self.melody_history = []
        self.accompaniment_history = []

    def clear_history(self):
        """Clears the performance history."""
        self.melody_history = []
        self.accompaniment_history = []

    def _notes_to_rolls(self, notes: list, max_tick: int, max_polyphony=4, program=0):
        """
        Converts a list of musical notes into a tensor representation (a "piano roll").

        This function creates a tensor where each row represents a tick and each column
        contains information about the notes starting at that tick.

        Args:
            notes (list): A list of note dictionaries, each with 'pitch', 'tick', 'duration'.
                          The ticks in this list are expected to be relative to the start
                          of the desired tensor, not absolute performance ticks.
            max_tick (int): The total number of ticks for the resulting tensor (its length).
            max_polyphony (int): The maximum number of simultaneous notes allowed per tick.
            program (int): The instrument program number (0 for melody, 1 for accompaniment).

        Returns:
            torch.Tensor: A tensor of shape `(max_tick, max_polyphony * 3)` ready for
                          the model's `preprocess` step. Each note is represented by
                          3 numbers: (program, pitch, duration_idx). The tensor is
                          padded with 255 for empty slots.
        """
        # DURATION_TEMPLATES contains the quantized durations the model understands.
        # We find the midpoints between these templates to decide which template a
        # given note duration is closest to.
        duration_boundaries = (DURATION_TEMPLATES[1:] + DURATION_TEMPLATES[:-1]) / 2.0
        
        # Initialize the piano roll tensor.
        # Shape: (time, polyphony, features)
        # Features are (instrument_program, pitch, duration_index).
        # We use 255 as a special padding value, which is later ignored by the model.
        rolls = np.full((max_tick, max_polyphony, 3), dtype=np.uint8, fill_value=255)
        
        # Keep track of how many notes are at each tick to handle polyphony.
        polyphony_counts = np.zeros(max_tick, dtype=np.uint8)

        for note in notes:
            # TEEHEE CHECKPOINT | SHOULD BE OK
            tick = note['tick']
            if tick >= max_tick:
                continue

            # Drop notes that exceed the maximum polyphony for a given tick.
            if polyphony_counts[tick] >= max_polyphony:
                print(f"Warning: Exceeded max polyphony at tick {tick}. Note with pitch {note['pitch']} dropped.")
                continue
            
            # Find the index of the closest duration template.
            duration_idx = np.searchsorted(duration_boundaries, note['duration'])

            # Place the note's data into the correct slot in the tensor.
            slot = polyphony_counts[tick]
            rolls[tick, slot, 0] = program
            rolls[tick, slot, 1] = note['pitch']
            rolls[tick, slot, 2] = duration_idx
            
            polyphony_counts[tick] += 1

        # For polyphonic ticks, sort the notes by pitch. This creates a canonical
        # representation, which helps the model learn more effectively.
        for i in range(max_tick):
            if polyphony_counts[i] > 1:
                sorted_indices = np.argsort(rolls[i, :polyphony_counts[i], 1])
                rolls[i, :polyphony_counts[i]] = rolls[i, :polyphony_counts[i]][sorted_indices]
            if polyphony_counts[i] < max_polyphony:
                rolls[i, polyphony_counts[i], 0] = 254 
                
        # Reshape the tensor to the final format expected by the model's preprocess function.
        # Shape becomes (max_tick, max_polyphony * 3), e.g., (48, 12).
        return torch.tensor(rolls.reshape(max_tick, -1))

    def _tensors_to_notes(self, output_tensors: list, start_frame_offset=0, single=False):
        """
        Decodes the symbolic output from the model back into a list of note events.

        This is the reverse of `_notes_to_rolls`, translating the model's tensor
        predictions into a human-readable and playable format.

        Args:
            output_tensors (list): A list of tensors from the model's output. Each tensor
                                   in the list represents one frame of music.
            start_frame_offset (int): The frame number where the generation began. This is
                                      used to calculate the ticks relative to the start of
                                      the *newly generated* music, not the start of the prompt.
            single (bool): If True, assumes each frame is one tick (not interleaved).

        Returns:
            list[list[dict]]: A list containing one list of note dictionaries for each generated sample.
        """
        notes_per_sample = []
        num_timesteps = len(output_tensors)
        if num_timesteps == 0:
            return []
        
        n_samples = output_tensors[0].shape[0]

        for i in range(n_samples):
            sample_notes = []
            # We only iterate from the start of the generated content.
            for time_step in range(start_frame_offset, num_timesteps):
                data = output_tensors[time_step][i]
                content = data.flatten()
                
                # Calculate the tick relative to the start of the *generation*.
                relative_time_step = time_step - start_frame_offset
                tick = relative_time_step if single else relative_time_step // 2
                
                # Each note is encoded as two tokens: (program, pitch_duration_combo).
                for j in range(0, len(content), 2):
                    program = int(content[j].item())
                    # Stop decoding this frame if we hit a special token.
                    if program == EOS_TOKEN or program == PAD_TOKEN:
                        continue
                    if j + 1 >= len(content):
                        break
                    
                    # This is the core decoding step.
                    # The model predicts a single number that combines pitch and duration.
                    # We reverse the formula: pitch_duration = (duration_idx * 128) + pitch
                    pitch_duration = int(content[j+1].item()) - 2 # Subtract special token offset
                    if pitch_duration < 0:
                        continue

                    pitch = pitch_duration % 128
                    duration_idx = pitch_duration // 128

                    # Validate the decoded duration index.
                    if not (0 <= duration_idx < len(DURATION_TEMPLATES)):
                        continue

                    duration = DURATION_TEMPLATES[duration_idx]

                    sample_notes.append({
                        'tick': tick*4,
                        'pitch': pitch,
                        'duration': int(duration),
                        'program': program
                    })
            notes_per_sample.append(sample_notes)
        return notes_per_sample

    def generate_accompaniment(self, melody_notes, generation_start_tick: int, accompaniment_notes=[]):
        """
        Generates musical accompaniment based on a history of melody and accompaniment notes.
        This is the main public method for the engine.
        """
        preprocess_start_time = time.perf_counter()

        # Step 1: Update the internal melody history with the new user notes for this turn.
        self.melody_history.extend(melody_notes)
        # Note: We do NOT update the accompaniment history until after generation.

        # Step 2: Create a prompt for the model from history occurring BEFORE the generation start tick.
        # This logic ensures the model always gets a fixed-size input and that the most
        # recent history is right-aligned (padded at the front).
        prompt_end_tick = generation_start_tick
        prompt_start_tick = max(0, prompt_end_tick - self.prompt_length_ticks)
        
        # This is the fixed context length the model expects.
        model_context_len = self.prompt_length_ticks
        # This is the actual duration of the history we have available for the prompt.
        actual_history_duration = prompt_end_tick - prompt_start_tick
        
        # If there's no history to use, there's nothing to generate from.
        if actual_history_duration <= 0:
            return ([], preprocess_start_time, time.perf_counter(), time.perf_counter(), time.perf_counter())

        # Calculate the padding needed at the beginning of the context window.
        padding_duration = model_context_len - actual_history_duration

        # Filter notes and make their ticks relative to the *padded* window.
        # This aligns the existing history to the END of the context window.
        # For example, if padding_duration is 10, the first note's tick will be 10, not 0.
        prompt_melody = [
            {**n, 'tick': (n['tick'] - prompt_start_tick) + padding_duration}
            for n in self.melody_history if prompt_start_tick <= n['tick'] < prompt_end_tick
        ]
        prompt_acc = [
            {**n, 'tick': (n['tick'] - prompt_start_tick) + padding_duration}
            for n in self.accompaniment_history if prompt_start_tick <= n['tick'] < prompt_end_tick
        ]
        # # Log the prompt_acc to a file for debugging purposes.
        # with open(self.log_filename_promt, "a") as f:
        #     f.write(f"\n==== PROMPT_ACC {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ====\n")
        #     for note in prompt_acc:
        #         f.write(json.dumps(note, ensure_ascii=False) + "\n")
        
        # Step 3: Convert note events into tensor 'rolls' using the fixed model context length.
        # The resulting tensors will have shape (48, 12).
        x_mel_raw = self._notes_to_rolls(prompt_melody, model_context_len, self.max_polyphony, program=0)
        x_acc_raw = self._notes_to_rolls(prompt_acc, model_context_len, self.max_polyphony, program=1)

        # Step 4: Preprocess the rolls and prepare them for the model (e.g., move to GPU).
        # `preprocess` combines (program, pitch, duration) into the single combined token
        # that the model was trained on.
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        x_mel_raw = x_mel_raw.unsqueeze(0).to(device)
        x_acc_raw = x_acc_raw.unsqueeze(0).to(device)
        x_mel, x_acc = self.model.preprocess(x_mel_raw.long(), pitch_shift=torch.zeros(1, dtype=torch.int8).to(device), y=x_acc_raw.long())

        # The model expects an interleaved sequence of melody and accompaniment frames.
        # We stack them and reshape to create this interleaved format.
        # Final shape `x` is (batch_size, num_frames, subseq_len), e.g., (1, 96, 8).
        batch_size, seq_len, subseq_len = x_mel.shape
        stacked = torch.stack([x_acc, x_mel], dim=2)
        x = stacked.view(batch_size, seq_len * 2, subseq_len)

        # Step 5: Run the core inference.
        inference_start_time = time.perf_counter()

        # # Debugging: Save the x tensor distribution to visualize PAD_TOKEN (3204) distribution.
        # print(f"x, {x.shape}, {x}")
        # self.save_x_counter += 1
        # if self.save_x_counter % 10 == 0:
        #     x_2d = x[0].detach().cpu().numpy()  # shape: (num_frames, subseq_len)
        #     mask_3204 = (x_2d == 3204).astype(int)
        #     plt.figure(figsize=(12, 6))
        #     plt.imshow(mask_3204.T, aspect='auto', cmap='gray_r', interpolation='nearest')
        #     plt.xlabel('Frame')
        #     plt.ylabel('Token Index')
        #     plt.title(f'3204 (PAD_TOKEN) Distribution, call {self.save_x_counter}')
        #     plt.colorbar(label='Is 3204')
        #     save_path = os.path.join(self.x_save_dir, f"x_pad_{self.save_x_counter}.png")
        #     plt.savefig(save_path)
        #     plt.close()
        with torch.no_grad():
            output_tensors = self.model.global_sampling(x, x_mel_gt=None, temperature=1, max_seq_len=self.generation_length_frames)
        inference_end_time = time.perf_counter()

        postprocess_start_time = time.perf_counter()
        
        # Step 6: Decode the model's full tensor output (prompt + generation) into note events.
        # The resulting note ticks are relative to the beginning of the prompt tensor.
        all_notes_relative_to_prompt = self._tensors_to_notes(output_tensors, start_frame_offset=0)
        all_notes_relative_to_prompt = all_notes_relative_to_prompt[0] if all_notes_relative_to_prompt else []

        # Step 7: Post-process by first converting all decoded notes to have absolute ticks.
        # This makes subsequent filtering logic cleaner.
        all_notes_absolute = []
        for note in all_notes_relative_to_prompt:
            # Convert tick from relative-to-prompt to absolute performance time
            tick_relative_to_generation = note['tick'] - self.prompt_length_ticks
            absolute_tick = tick_relative_to_generation + generation_start_tick
            all_notes_absolute.append({**note, 'tick': absolute_tick})
            
        # Now, filter the notes using their absolute ticks.
        # - Filter for notes that occur at or after the generation_start_tick.
        # - Filter for accompaniment notes only (program == 1).
        # - Filter for unique notes to prevent duplicates. (Temporarily Disabled)
        # unique_notes_tracker = set()
        final_generated_notes = []
        for note in all_notes_absolute:
            if note['tick'] >= generation_start_tick and note['program'] == 1:
                # The uniqueness filter below is temporarily disabled.
                # A note is unique based on its absolute tick, pitch, and duration.
                # note_signature = (note['tick'], note['pitch'], note['duration'])
                # if note_signature not in unique_notes_tracker:
                #     unique_notes_tracker.add(note_signature)
                final_generated_notes.append(note)

        # Step 8: Update the history with the newly generated accompaniment notes.
        # This ensures they become part of the context for the next turn.
        # 先收集新生成的所有 tick
        new_ticks = set(note['tick'] for note in final_generated_notes)
        # print(f"final generated notes:{final_generated_notes}")
        # print(f"acc history 1:{self.accompaniment_history}")

        tem = self.accompaniment_history.copy()
        # 从历史中移除 tick 与新生成重复的 note
        self.accompaniment_history = [n for n in self.accompaniment_history if n['tick'] not in new_ticks]
        tem = set(json.dumps(n, sort_keys=True) for n in tem) - set(json.dumps(n, sort_keys=True) for n in self.accompaniment_history)
        print(f"difference{tem}")
        self.accompaniment_history.extend(final_generated_notes)

        # Prune history to prevent memory leaks in long-running sessions.
        # We can safely remove any notes that are older than the prompt window we just used.

        self.melody_history = [n for n in self.melody_history if n['tick'] >= prompt_start_tick]
        self.accompaniment_history = [n for n in self.accompaniment_history if n['tick'] >= prompt_start_tick]
        # print(f"acc history 2:{self.accompaniment_history}")

        # # Log the accompaniment history to a file for debugging purposes.
        # with open(self.log_filename_history, "w") as f:
        #     for note in self.accompaniment_history:
        #         f.write(json.dumps(note, ensure_ascii=False) + "\n")

        return (final_generated_notes, preprocess_start_time, inference_start_time, inference_end_time, postprocess_start_time)

    