import torch
import os
import numpy as np
import time
import sys

# Add the project root to the Python path to allow for absolute imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from m2a_transformer import RoFormerSymbolicTransformer, EOS_TOKEN, PAD_TOKEN
from preprocess_large_midi_dataset import DURATION_TEMPLATES

class TransformerInferenceEngine:
    def __init__(self, checkpoint_path: str, max_polyphony=4, generation_length_frames=32):
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        print(f"Loading model from: {checkpoint_path}")
        # Determine model size from checkpoint path name
        if 'small' in checkpoint_path:
            self.model = RoFormerSymbolicTransformer.load_from_checkpoint(checkpoint_path, large=False)
        else:
            self.model = RoFormerSymbolicTransformer.load_from_checkpoint(checkpoint_path, large=True)
        
        if torch.cuda.is_available():
            self.model.to('cuda:7')
        self.model.eval()
        print("Model loaded successfully.")
        
        # The model's max sequence length is defined in interleaved frames (melody, accompaniment).
        # The effective prompt length in ticks is half of that, as ticks are our musical time unit.
        self.model_max_seq_len_frames = 96 # Derived from m2a_transformer.py
        self.prompt_length_ticks = self.model_max_seq_len_frames // 2
        
        self.max_polyphony = max_polyphony
        self.generation_length_frames = generation_length_frames
        
        self.melody_history = []
        self.accompaniment_history = []

    def clear_history(self):
        self.melody_history = []
        self.accompaniment_history = []

    def _notes_to_rolls(self, notes, max_tick, max_polyphony=4, program=0):
        duration_boundaries = (DURATION_TEMPLATES[1:] + DURATION_TEMPLATES[:-1]) / 2.0
        rolls = np.full((max_tick, max_polyphony, 3), dtype=np.uint8, fill_value=255)
        polyphony_counts = np.zeros(max_tick, dtype=np.uint8)

        for note in notes:
            tick = note['tick']
            if tick >= max_tick:
                continue

            if polyphony_counts[tick] >= max_polyphony:
                print(f"Warning: Exceeded max polyphony at tick {tick}. Note with pitch {note['pitch']} dropped.")
                continue
            
            duration_idx = np.searchsorted(duration_boundaries, note['duration'])

            slot = polyphony_counts[tick]
            rolls[tick, slot, 0] = program
            rolls[tick, slot, 1] = note['pitch']
            rolls[tick, slot, 2] = duration_idx
            
            polyphony_counts[tick] += 1

        for i in range(max_tick):
            if polyphony_counts[i] > 1:
                sorted_indices = np.argsort(rolls[i, :polyphony_counts[i], 1])
                rolls[i, :polyphony_counts[i]] = rolls[i, :polyphony_counts[i]][sorted_indices]

        return torch.tensor(rolls.reshape(max_tick, -1))

    def _tensors_to_notes(self, output_tensors, single=False):
        notes_per_sample = []
        num_timesteps = len(output_tensors)
        if num_timesteps == 0:
            return []
        
        n_samples = output_tensors[0].shape[0]

        for i in range(n_samples):
            sample_notes = []
            for time_step in range(num_timesteps):
                data = output_tensors[time_step][i]
                content = data.flatten()
                tick = time_step if single else time_step // 2
                
                for j in range(0, len(content), 2):
                    program = int(content[j].item())
                    if program == EOS_TOKEN or program == PAD_TOKEN:
                        continue
                    if j + 1 >= len(content):
                        break
                    
                    pitch_duration = int(content[j+1].item()) - 2
                    if pitch_duration < 0:
                        continue

                    pitch = pitch_duration % 128
                    duration_idx = pitch_duration // 128

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

        This method is stateful and maintains a history of all notes played. It uses the most
        recent segment of this history as a prompt for the transformer model to generate the
        next sequence of accompaniment notes.

        Args:
            melody_notes (list[dict]): A list of new melody note events from the user.
            generation_start_tick (int): The absolute tick value from which the generation should start.
            accompaniment_notes (list[dict], optional): A list of new accompaniment note events.

        Returns:
            tuple: A tuple containing the list of newly generated accompaniment notes and timing
                   information for different stages of the process (preprocessing, inference, postprocessing).
        """
        preprocess_start_time = time.perf_counter()

        # Step 1: Update the internal history with the new notes from this turn.
        self.melody_history.extend(melody_notes)
        self.accompaniment_history.extend(accompaniment_notes)

        # Step 2: Create a prompt for the model from the recent history.
        # The prompt is a rolling window of the last `prompt_length_ticks` of the performance.
        all_history = self.melody_history + self.accompaniment_history
        if not all_history:
             # If history is empty, ensure generation starts from the client's requested tick.
             max_tick_in_history = generation_start_tick -1
        else:
             max_tick_in_history = max(n['tick'] for n in all_history)
        
        # Trim the history to the model's maximum context window (prompt length).
        prompt_start_tick = max(0, max_tick_in_history - self.prompt_length_ticks + 1)
        
        # Filter notes to include only those within the prompt window and make their ticks relative to the start of the prompt.
        prompt_melody = [
            {**n, 'tick': n['tick'] - prompt_start_tick}
            for n in self.melody_history if n['tick'] >= prompt_start_tick
        ]
        prompt_acc = [
            {**n, 'tick': n['tick'] - prompt_start_tick}
            for n in self.accompaniment_history if n['tick'] >= prompt_start_tick
        ]
        
        prompt_duration_ticks = max_tick_in_history - prompt_start_tick + 1

        # Step 3: Convert the note events into tensor 'rolls' that the model can understand.
        x_mel_raw = self._notes_to_rolls(prompt_melody, prompt_duration_ticks, self.max_polyphony, program=0)
        x_acc_raw = self._notes_to_rolls(prompt_acc, prompt_duration_ticks, self.max_polyphony, program=1)

        # Step 4: Preprocess the rolls and prepare them for the model (e.g., move to GPU).
        device = 'cuda:7' if torch.cuda.is_available() else 'cpu'
        x_mel_raw = x_mel_raw.unsqueeze(0).to(device)
        x_acc_raw = x_acc_raw.unsqueeze(0).to(device)

        x_mel, x_acc = self.model.preprocess(x_mel_raw.long(), pitch_shift=torch.zeros(1, dtype=torch.int8).to(device), y=x_acc_raw.long())

        # The model expects an interleaved sequence of melody and accompaniment.
        batch_size, seq_len, subseq_len = x_mel.shape
        stacked = torch.stack([x_acc, x_mel], dim=2)
        x = stacked.view(batch_size, seq_len * 2, subseq_len)

        # Step 5: Run the core inference.
        inference_start_time = time.perf_counter()
        with torch.no_grad():
            output_tensors = self.model.global_sampling(x, x_mel_gt=None, temperature=1.0, max_seq_len=self.generation_length_frames)
        inference_end_time = time.perf_counter()

        postprocess_start_time = time.perf_counter()
        
        # Step 6: Decode the model's tensor output back into note events.
        generated_notes_relative = self._tensors_to_notes(output_tensors)
        generated_notes_relative = generated_notes_relative[0] if generated_notes_relative else []

        # Step 7: Post-process the generated notes.
        # The model's output is relative to the start of the generation, so we need to
        # offset the ticks to place them correctly in the absolute performance timeline.
        # The client is the source of truth for the timeline.
        generated_notes_absolute = []
        for note in generated_notes_relative:
            if note['program'] == 1: # We only want to return and store the accompaniment.
                note['tick'] += generation_start_tick
                generated_notes_absolute.append(note)

        # Step 8: Update the history with the newly generated accompaniment notes.
        # This ensures they become part of the context for the next turn.
        self.accompaniment_history.extend(generated_notes_absolute)
        
        return (generated_notes_absolute, preprocess_start_time, inference_start_time, inference_end_time, postprocess_start_time)

    