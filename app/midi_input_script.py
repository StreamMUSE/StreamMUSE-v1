# This is the fake real time

import pretty_midi
from inference_engines.transformer_engine import TransformerInferenceEngine
import os
import json
import torch
import numpy as np
from preprocess.xf_midi import XFMidi
from preprocess.preprocess_midi2pt_dataset import DURATION_TEMPLATES
from m2a_transformer import RoFormerSymbolicTransformer, EOS_TOKEN, PAD_TOKEN
from m2a_transformer_inference import decode_output
import pdb

def midi_to_note(midi_path, min_pitch=0, max_pitch=127, beat_div=4, program=None):
    """
    用 XFMidi 读取 midi 文件，返回 notes 列表，每个元素是 {'pitch', 'tick', 'duration'} 字典。
    """
    midi = XFMidi(midi_path, constant_tempo=60.0 / beat_div)
    max_tick = int(midi.get_end_time())
    notes = []
    for inst in midi.instruments:
        # 如果只想要某种 program，可以加判断
        if program is not None and inst.program != program:
            continue
        for note in inst.notes:
            if min_pitch <= note.pitch <= max_pitch:
                # tick = round(time * resolution)
                start_tick = int(round(note.start))
                end_tick = int(round(note.end))
                if start_tick >= 0 and end_tick < max_tick:
                    notes.append({
                        'pitch': note.pitch,
                        'tick': start_tick,
                        'duration': end_tick - start_tick
                    })
    # 按tick排序
    notes.sort(key=lambda x: x['tick'])
    
    return notes, midi.resolution, max_tick

def note_list_to_pretty_midi(notes, tempo=90, program=0, name="track"):
    time_step_length = 60.0 / tempo / 4
    
    inst = None
    if program == 0:
        inst = pretty_midi.Instrument(program=24, name='Guitar')
    else:  # program == 1
        inst = pretty_midi.Instrument(program=0, name='Piano')
        
    for note in notes:
        start = note['tick'] * time_step_length
        end = DURATION_TEMPLATES[note['duration']] * time_step_length + start
        midi_note = pretty_midi.Note(
            velocity=100,
            pitch=note['pitch'],
            start=start,
            end=end
        )
        inst.notes.append(midi_note)
    return inst

class MyInferenceEngine():
    def __init__(self, checkpoint_path: str, max_polyphony=4, generation_length_frames=2, model_max_seq_len_frames=384):
        # Check if the checkpoint file exists
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
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
        
        # Move model to GPU if available
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

    def _tensors_to_notes(self, output_tensors: list, generation_start_tick=0, single=False):
        """
        Decodes the symbolic output from the model back into a list of note events.

        This is the reverse of `_notes_to_rolls`, translating the model's tensor
        predictions into a human-readable and playable format.

        Args:
            output_tensors (list): A list of tensors from the model's output. Each tensor
                                   in the list represents one frame of music.
            start_frame_offset (int): The frame number where the generation began.
                                    We take the input as the first position, and the frame number is relative to this.
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
            for time_step in range(generation_start_tick*2, num_timesteps):
                data = output_tensors[time_step][i]
                content = data.flatten()
                
                # Calculate the tick relative to the start of the *generation*.
                tick = time_step // 2
                
                # Each note is encoded as two tokens: (program, pitch_duration_combo).
                for j in range(0, len(content), 2):
                    program = int(content[j].item())
                    # Stop decoding this frame if we hit a special token.
                    if program == EOS_TOKEN or program == PAD_TOKEN:
                        continue

                    # 因为目前所有的输出都是 1，也就是所有的都是 acc，导致 acc会变得特别多，
                    # 现在确定了 program 不是 EOS_TOKEN 或 PAD_TOKEN 之后，我们根据位置改一下
                    if time_step % 2 == 0:
                        program = 1
                    else:
                        program = 0

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
                        'tick': tick,
                        'pitch': pitch,
                        'duration': int(duration),
                        'program': program
                    })
            notes_per_sample.append(sample_notes)
        return notes_per_sample

    def generate_accompaniment(self, melody_notes, generation_start_tick, acc_notes=None):

        # update the melody and accompaniment history
        if len(self.accompaniment_history) == 0 and acc_notes is not None:
            # If this is the first call, we need to initialize the accompaniment history
            self.accompaniment_history.extend(acc_notes)
        self.melody_history.extend(melody_notes)

        # Get the length of our actually prompt 
        prompt_end_tick = generation_start_tick
        prompt_start_tick = max(0, prompt_end_tick - self.prompt_length_ticks)
        actual_prompt_length_ticks = prompt_end_tick - prompt_start_tick
        if actual_prompt_length_ticks <= 0: # check if prompt length is valid
            raise ValueError("Prompt length must be greater than zero.")
        
        # Filter notes and make their ticks relative to the *padded* window.
        # This aligns the existing history to the END of the context window.
        # For example, if padding_duration is 10, the first note's tick will be 10, not 0.
        prompt_melody = [
            n for n in self.melody_history if prompt_start_tick <= n['tick'] < prompt_end_tick
        ]
        prompt_acc = [
            n for n in self.accompaniment_history if prompt_start_tick <= n['tick'] < prompt_end_tick
        ]

        # Convert note events into tensor 'rolls' using the fixed model context length.
        # The resulting tensors will have shape (48, 12).
        x_mel_raw = self._notes_to_rolls(prompt_melody, actual_prompt_length_ticks, self.max_polyphony, program=0)
        x_acc_raw = self._notes_to_rolls(prompt_acc, actual_prompt_length_ticks, self.max_polyphony, program=1)

        # Preprocess the rolls and prepare them for the model (e.g., move to GPU).
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
        # pdb.set_trace()
        np.set_printoptions(threshold=np.inf)
        with open("tensor_dump.txt", "w") as f:
            f.write(str(x.cpu().numpy()))

        with torch.no_grad():
            output_tensors = self.model.global_sampling(x, x_mel_gt=None, temperature=1, max_seq_len=self.generation_length_frames)

        output_0 = [output_tensors[j][0 : 0 + 1, :] for j in range(len(output_tensors))]
        with open("tensor_output_dump.txt", "w") as f:
            f.write(str(output_0))
        decode_output(output_0, "temp1/testTensorToNotes.mid", tempo=90.0)

        # Decode the model's full tensor output (prompt + generation) into note events.
        # The resulting note ticks are relative to the beginning of the prompt tensor.
        notes_output = self._tensors_to_notes(output_tensors, generation_start_tick=generation_start_tick)
        notes_output = notes_output[0] if notes_output else []

        # Now, filter the notes using their absolute ticks.
        # - Filter for notes that occur at or after the generation_start_tick.
        # - Filter for accompaniment notes only (program == 1).
        # - Filter for unique notes to prevent duplicates. (Temporarily Disabled)
        # unique_notes_tracker = set()
        final_generated_notes = []
        for note in notes_output:
            if note['tick'] >= generation_start_tick and note['program'] == 1:
                # The uniqueness filter below is temporarily disabled.
                # A note is unique based on its absolute tick, pitch, and duration.
                # note_signature = (note['tick'], note['pitch'], note['duration'])
                # if note_signature not in unique_notes_tracker:
                #     unique_notes_tracker.add(note_signature)
                final_generated_notes.append(note)

        # Update the history with the newly generated accompaniment notes.
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

        return final_generated_notes

if __name__ == "__main__":
    # seed = 42
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)

    # 1. 初始化 engine
    checkpoint_path = os.getenv('CHECKPOINT_PATH', "/home/ubuntu/ugrip/stanleyz/StreamMUSE/results/ModelBaseline/cp_transformer_909+ac+1k7_trackemb_interleavepos_v0.2_large_batch_40_schedule.epoch=00.val_loss=0.90296.ckpt")
    if not checkpoint_path:
        print('Fatal Error: CHECKPOINT_PATH environment variable is not set')
        print("Please run the server like: CHECKPOINT_PATH=path/to/model.ckpt uvicorn ...")
        exit()

    # Get model parameters from environment variables with defaults
    try:
        model_max_seq_len_frames = int(os.getenv('MODEL_MAX_SEQ_LEN_FRAMES', 400))
        generation_length_frames = int(os.getenv('GENERATION_LENGTH_FRAMES', 2))
    except ValueError:
        print("Fatal Error: Invalid integer value for model parameters in environment variables.")
        exit()

    try:
        print(f"Loading model from {checkpoint_path}...")
        print(f"Using Model Max Sequence Length (Frames): {model_max_seq_len_frames}")
        print(f"Using Generation Length (Frames): {generation_length_frames}")
        # inference_engine = TransformerInferenceEngine(
        #     checkpoint_path=checkpoint_path,
        #     model_max_seq_len_frames=model_max_seq_len_frames,
        #     generation_length_frames=generation_length_frames
        # )
        inference_engine = MyInferenceEngine(
            checkpoint_path=checkpoint_path,
            model_max_seq_len_frames=model_max_seq_len_frames,
            generation_length_frames=generation_length_frames
        )
    except FileNotFoundError as e:
            print(f"Fatal Error: {e}")
            exit()

    # 2. 读取 melody midi，转为 note list
    melody_notes, resolution, max_tick = midi_to_note("input/mel/001.mid")
    acc_notes, _, _ = midi_to_note("input/acc/001.mid")
    melody_notes = sorted(melody_notes, key=lambda n: n['tick'])
    acc_notes = sorted(acc_notes, key=lambda n: n['tick'])

    # 3. 获取所有 tick 的范围
    min_tick = 0
    max_tick = max_tick
    max_steps = 200  # prompt + generate 最多多少个 ticks

    # 收集伴奏
    acc_history = []

    # initialize with prompt notes
    generation_start_tick = 100 # 从第100个tick开始生成伴奏
    current_mel = [n for n in melody_notes if n['tick'] < generation_start_tick]
    current_acc = [n for n in acc_notes if n['tick'] < generation_start_tick]
    acc_history.extend(current_acc)
    # pdb.set_trace()
    acc_notes = inference_engine.generate_accompaniment(current_mel, generation_start_tick=generation_start_tick, acc_notes=current_acc)
    acc_history.extend(acc_notes)
    for i in range(generation_start_tick+1, max_tick):
        # if i > 175:
        #     pdb.set_trace()
        if i >= max_steps:
            break
        # 当前 tick 的所有 note
        current_mel = [n for n in melody_notes if n['tick'] == i-1]


        # 关键：传入完整的 melody_history
        acc_notes = inference_engine.generate_accompaniment(current_mel, generation_start_tick=i)
        acc_history.extend(acc_notes)

        print(f"Step {i}, tick={i}, melody={current_mel}, generated acc={acc_notes}")
        # time.sleep(0.05)  # 如需模拟实时可加延迟

    # 5. 输出为两个track的midi文件
    midi_out = pretty_midi.PrettyMIDI(initial_tempo=90.0)
    melody_instr = note_list_to_pretty_midi(melody_notes, resolution, program=0, name="melody")
    acc_instr = note_list_to_pretty_midi(acc_history, resolution, program=1, name="accompaniment")
    midi_out.instruments.append(melody_instr)
    midi_out.instruments.append(acc_instr)
    midi_out.write("temp/fake_client/fake_client_output1.mid")
    print("已保存为 temp/fake_client/fake_client_output1.mid")