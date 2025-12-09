import sys
import os
import time
import json
import torch
import numpy as np
import pretty_midi
import safetensors.torch
from transformers import LlamaConfig
from lekai_model.config import ModelConfig
from lekai_model.model import PianoLLaMA
from lekai_model.my_tokenizer import PianoRollTokenizer
from lekai_model.PianoDataset import encode_bpm
from lekai_model.generation_utils import sample_token


def midi_to_note(midi_path, beat_div=4):
    """
    Simple MIDI to note list converter using pretty_midi.
    Quantizes to fixed ticks per beat (default 4).
    Assuming 120 BPM for quantization if not specified.
    """
    try:
        pm = pretty_midi.PrettyMIDI(midi_path)
    except Exception as e:
        print(f"Error loading MIDI: {e}")
        return [], 0, 0

    # Assume 120 BPM for simple testing quantization
    bpm = 120
    seconds_per_beat = 60.0 / bpm
    seconds_per_tick = seconds_per_beat / beat_div

    notes = []
    max_tick = 0

    for inst in pm.instruments:
        if inst.is_drum:
            continue
        for note in inst.notes:
            start_tick = int(round(note.start / seconds_per_tick))
            end_tick = int(round(note.end / seconds_per_tick))
            duration = max(1, end_tick - start_tick)

            if start_tick < 0:
                continue

            notes.append({"pitch": note.pitch, "tick": start_tick, "duration": duration})
            max_tick = max(max_tick, end_tick)

    notes.sort(key=lambda x: x["tick"])
    return notes, None, max_tick


class InferenceEngineLekai:
    def __init__(
        self,
        checkpoint_path: str,
        model_size: str,  # Unused for LLaMA config loading in this version, but kept for API compatibility
        max_polyphony=4,  # Unused
        generation_length_frames=2,  # Unused
        model_max_seq_len_frames=384,  # Unused
        inference_mode="sliding_window",  # "sliding_window" or "stateful"
    ):
        # Check if the checkpoint file exists
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        print(f"Loading LLaMA model from: {checkpoint_path} (Mode: {inference_mode})")

        # 1. Initialize Config
        self.config = ModelConfig()

        # 2. Initialize Tokenizer
        self.tokenizer = PianoRollTokenizer(patch_h=self.config.patch_h, patch_w=self.config.patch_w, img_h=88)

        # 3. Initialize Model Structure
        llama_config = LlamaConfig(
            vocab_size=self.config.vocab_size,
            hidden_size=self.config.hidden_size,
            num_hidden_layers=self.config.num_hidden_layers,
            num_attention_heads=self.config.num_attention_heads,
            intermediate_size=self.config.intermediate_size,
            max_position_embeddings=self.config.max_position_embeddings,
            pad_token_id=self.config.pad_token_id,
            bos_token_id=self.config.bos_token_id,
            eos_token_id=self.config.eos_token_id,
            rope_theta=self.config.rope_theta,
            attention_dropout=self.config.dropout,
            use_cache=True,
            initializer_range=0.02,
        )
        self.model = PianoLLaMA(llama_config)

        # 4. Load Weights
        state_dict = safetensors.torch.load_file(checkpoint_path)
        self.model.load_state_dict(state_dict, strict=False)

        # Move model to GPU if available
        if torch.cuda.is_available():
            self.model.cuda()
        self.model.eval()
        print("Model loaded successfully.")

        # Parameters
        self.ticks_per_beat = 4  # Assuming 4 ticks per beat for the tokenizer/model
        self.melody_history = []
        self.accompaniment_history = []
        self.counter = 0
        self.injection_offset_ticks = 0

        # Inference Mode State
        self.inference_mode = inference_mode
        self.past_key_values = None
        self.last_generated_beat = -1

    def set_injection_offset(self, offset_ticks: int):
        self.injection_offset_ticks = offset_ticks
        print(f"设置注入偏移: {offset_ticks} ticks")

    def clear_history(self):
        self.melody_history = []
        self.accompaniment_history = []
        self.past_key_values = None
        self.last_generated_beat = -1

    def _notes_to_pianoroll(self, notes, max_tick):
        """
        Convert notes list to (2, 88, max_tick) piano roll.
        Channel 0: Sustain
        Channel 1: Onset
        """
        pianoroll = np.zeros((2, 88, max_tick), dtype=np.uint8)

        for note in notes:
            pitch_idx = note["pitch"] - 21  # MIDI 21-108
            if not (0 <= pitch_idx < 88):
                continue

            start = int(note["tick"])
            duration = int(note["duration"])
            end = min(start + duration, max_tick)

            if start >= max_tick:
                continue

            # Channel 0: Sustain
            pianoroll[0, pitch_idx, start:end] = 1
            # Channel 1: Onset
            pianoroll[1, pitch_idx, start] = 1

        return pianoroll

    def _pianoroll_to_notes(self, pianoroll, start_tick):
        """
        Convert (2, 88, T) piano roll back to notes list.
        """
        notes = []
        sustain = pianoroll[0]
        onset = pianoroll[1]

        for pitch_idx in range(88):
            pitch = pitch_idx + 21
            # Find onsets
            onset_indices = np.where(onset[pitch_idx] > 0)[0]
            for start in onset_indices:
                # Find end (sustain end)
                end = start + 1
                while end < pianoroll.shape[2] and sustain[pitch_idx, end] > 0:
                    end += 1

                notes.append(
                    {
                        "pitch": pitch,
                        "tick": start + start_tick,
                        "duration": end - start,
                        "program": 1,  # Accompaniment
                    }
                )
        return notes

    def _get_tokens_for_beat(self, notes, beat_idx):
        """Helper to get tokens for a specific beat range"""
        beat_start_tick = beat_idx * self.ticks_per_beat
        beat_end_tick = (beat_idx + 1) * self.ticks_per_beat

        # Filter notes overlapping this beat
        beat_notes = []
        for n in notes:
            # Shift to relative to beat start
            if n["tick"] < beat_end_tick and (n["tick"] + n["duration"]) > beat_start_tick:
                new_n = n.copy()
                new_n["tick"] = max(0, n["tick"] - beat_start_tick)
                # Clamp duration
                end_rel = min(self.ticks_per_beat, (n["tick"] + n["duration"]) - beat_start_tick)
                new_n["duration"] = end_rel - new_n["tick"]
                beat_notes.append(new_n)

        pr = self._notes_to_pianoroll(beat_notes, self.ticks_per_beat)
        # Tokenize
        tokens = self.tokenizer.encode(pr, use_strict_mode=True)
        return torch.tensor(tokens, dtype=torch.long)

    def _generate_tokens(self, input_ids, past_key_values=None):
        """Core generation loop"""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        generated_tokens = []

        # Add bar token for Acc start
        bar_token = torch.tensor([[self.config.bar_token_id]], device=device)
        input_ids = torch.cat([input_ids, bar_token], dim=1)

        current_input = input_ids
        current_past = past_key_values

        with torch.no_grad():
            max_new_tokens = 100
            for _ in range(max_new_tokens):
                outputs = self.model(
                    input_ids=current_input,
                    past_key_values=current_past,
                    use_cache=True,
                )

                next_token_logits = outputs.logits[:, -1, :]
                current_past = outputs.past_key_values

                next_token = sample_token(
                    next_token_logits,
                    generated_tokens=input_ids,  # Note: this is just for context, might be inaccurate if we only pass partial input
                    temperature=0.8,
                    top_k=50,
                    top_p=0.95,
                    repetition_penalty=1.2,
                )

                # For next iteration
                current_input = next_token

                token_val = next_token.item()
                generated_tokens.append(token_val)

                if token_val in [self.config.end_marker_part1, self.config.bar_token_id]:
                    break

        return generated_tokens, current_past

    def generate_accompaniment(
        self,
        melody_notes,
        generation_start_tick,
        acc_notes=None,
        generation_length_frames=None,  # Unused, kept for API
        prompt_length_ticks=None,  # Unused, kept for API
    ):
        """
        Generate accompaniment using LLaMA model with beat-interleaving.
        Supports 'sliding_window' (stateless) and 'stateful' modes.
        """
        torch.cuda.synchronize()
        preprocess_start_time = time.perf_counter()

        # 1. Update History
        absolute_melody_notes = []
        for note in melody_notes:
            absolute_note = note.copy()
            absolute_note["tick"] = note["tick"] + self.injection_offset_ticks
            absolute_melody_notes.append(absolute_note)
        self.melody_history.extend(absolute_melody_notes)

        absolute_generation_start_tick = generation_start_tick + self.injection_offset_ticks

        if len(self.accompaniment_history) == 0 and acc_notes is not None:
            absolute_acc_notes = []
            for note in acc_notes:
                absolute_note = note.copy()
                absolute_note["tick"] = note["tick"] + self.injection_offset_ticks
                absolute_acc_notes.append(absolute_note)
            self.accompaniment_history.extend(absolute_acc_notes)

        # 2. Prepare Context & Prompt
        current_beat = absolute_generation_start_tick // self.ticks_per_beat
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Determine if we need a full context rebuild (Reset)
        need_reset = True
        if self.inference_mode == "stateful":
            if self.past_key_values is not None and current_beat == self.last_generated_beat + 1:
                need_reset = False

        input_ids = None
        past_key_values_to_use = None

        if need_reset:
            # --- Sliding Window / Warmup Logic ---
            context_beats = 32
            start_beat = max(0, current_beat - context_beats)
            start_tick = start_beat * self.ticks_per_beat

            # Filter history
            context_melody = [
                n for n in self.melody_history if start_tick <= n["tick"] < absolute_generation_start_tick
            ]
            context_acc = [
                n for n in self.accompaniment_history if start_tick <= n["tick"] < absolute_generation_start_tick
            ]

            # Construct Prompt Sequence: [BOS, Sig, BPM] + [Mel, Acc] pairs
            bpm_val = 120
            bpm_token = encode_bpm(bpm_val) + self.config.bpm_offset_id

            seq = [
                torch.tensor([self.config.bos_token_id], dtype=torch.long),
                torch.tensor([4 + self.config.time_sig_offset_id], dtype=torch.long),
                torch.tensor([bpm_token], dtype=torch.long),
            ]

            for b in range(start_beat, current_beat):
                # Mel
                mel_tokens = self._get_tokens_for_beat(context_melody, b)
                seq.append(torch.tensor([self.config.bar_token_id], dtype=torch.long))
                seq.append(mel_tokens)
                # Acc
                acc_tokens = self._get_tokens_for_beat(context_acc, b)
                seq.append(torch.tensor([self.config.bar_token_id], dtype=torch.long))
                seq.append(acc_tokens)

            # Add Target Melody for current beat
            target_melody = [
                n
                for n in self.melody_history
                if absolute_generation_start_tick <= n["tick"] < (current_beat + 1) * self.ticks_per_beat
            ]
            target_mel_tokens = self._get_tokens_for_beat(target_melody, current_beat)
            seq.append(torch.tensor([self.config.bar_token_id], dtype=torch.long))
            seq.append(target_mel_tokens)

            input_ids = torch.cat(seq).unsqueeze(0).to(device)
            past_key_values_to_use = None  # Reset cache

        else:
            # --- Stateful Incremental Logic ---
            # We only need to append the NEW Melody beat
            target_melody = [
                n
                for n in self.melody_history
                if absolute_generation_start_tick <= n["tick"] < (current_beat + 1) * self.ticks_per_beat
            ]
            target_mel_tokens = self._get_tokens_for_beat(target_melody, current_beat)

            # [Bar] + [Mel]
            seq = [torch.tensor([self.config.bar_token_id], dtype=torch.long), target_mel_tokens]
            input_ids = torch.cat(seq).unsqueeze(0).to(device)
            past_key_values_to_use = self.past_key_values

        torch.cuda.synchronize()
        inference_start_time = time.perf_counter()

        # 3. Generate
        generated_tokens, new_past_key_values = self._generate_tokens(input_ids, past_key_values_to_use)

        # Update State if Stateful
        if self.inference_mode == "stateful":
            self.past_key_values = new_past_key_values
            self.last_generated_beat = current_beat

        torch.cuda.synchronize()
        inference_end_time = time.perf_counter()

        # 4. Decode Generated Tokens
        # generated_tokens contains the compressed tokens for the Acc beat
        # We need to decode them back to piano roll

        # Remove the end marker if present for decoding?
        # The tokenizer.decode handles it.

        try:
            pr_generated = self.tokenizer.decode(generated_tokens, end_marker_id=self.config.end_marker_part1)
            # pr_generated shape: (2, 88, T) where T should be ticks_per_beat (4)

            # Convert to notes
            # Note: these notes are relative to the start of the beat (0-4)
            notes_relative = self._pianoroll_to_notes(pr_generated, start_tick=0)

            # Convert to absolute ticks
            absolute_generated_notes = []
            beat_abs_start = current_beat * self.ticks_per_beat

            for n in notes_relative:
                n["tick"] += beat_abs_start
                absolute_generated_notes.append(n)

        except Exception as e:
            print(f"Decoding error: {e}")
            absolute_generated_notes = []

        torch.cuda.synchronize()
        postprocess_start_time = time.perf_counter()

        # 6. Update History & Return
        self.accompaniment_history.extend(absolute_generated_notes)

        # Return relative notes (hiding injection offset)
        relative_generated_notes = []
        for note in absolute_generated_notes:
            relative_note = note.copy()
            relative_note["tick"] = note["tick"] - self.injection_offset_ticks
            relative_generated_notes.append(relative_note)

        return (
            relative_generated_notes,
            preprocess_start_time,
            inference_start_time,
            inference_end_time,
            postprocess_start_time,
        )


if __name__ == "__main__":
    # Test code
    checkpoint_path = os.getenv(
        "CHECKPOINT_PATH", "rt/RT_Accompaniment/checkpoints/epoch_4_1104_1204/model.safetensors"
    )

    try:
        inference_engine = InferenceEngineLekai(
            checkpoint_path=checkpoint_path,
            model_size="llama",
        )
    except Exception as e:
        print(f"Init failed: {e}")
        exit()

    # Load test midi
    # Assuming input/mel/001.mid exists
    try:
        melody_notes, resolution, max_tick = midi_to_note("input/mel/001.mid")
        print(f"Loaded melody: {len(melody_notes)} notes")

        # Generate for a few beats
        start_tick = 0
        for i in range(0, 16, 4):  # 4 beats
            print(f"Generating beat {i // 4}...")
            current_mel = [n for n in melody_notes if n["tick"] < i + 4]  # Feed up to current beat end

            notes, _, _, _, _ = inference_engine.generate_accompaniment(current_mel, generation_start_tick=i)
            print(f"Generated {len(notes)} notes")

    except Exception as e:
        print(f"Test run failed: {e}")
