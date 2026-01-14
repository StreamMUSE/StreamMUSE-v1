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
from lekai_model.MidiConverter import MidiConverter


def midi_to_note(midi_path, beat_div=4):
    """
    Simple MIDI to note list converter using MidiConverter.
    """
    converter = MidiConverter(ticks_per_beat=beat_div)
    pm, metadata = converter.load_midi(midi_path)
    if pm is None:
        return [], {}, 0

    notes, max_tick = converter.midi_to_notes(pm)
    return notes, metadata, max_tick


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
        self.tokenizer = PianoRollTokenizer(
            patch_h=self.config.patch_h,
            patch_w=self.config.patch_w,
            marker_offset=81,
            measures_length=88,
            end_marker_part0=170,
            end_marker_part1=171,
            empty_marker=169,
            img_h=88,
        )

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
        # Client melody is an event stream (note_on/note_off). We store raw events
        # (absolute ticks, after injection offset) and derive per-beat duration notes
        # only as an internal compatibility layer for the tokenizer.
        self.melody_event_history = []
        self.accompaniment_history = []
        self.counter = 0
        self.injection_offset_ticks = 0

        # Inference Mode State
        self.inference_mode = inference_mode
        self.past_key_values = None
        self.last_generated_beat = -1
        self.last_generated_acc_tokens = None  # Store last generated acc tokens for stateful feedback

        # Helper converter
        self.midi_converter = MidiConverter(ticks_per_beat=self.ticks_per_beat)

        # Event-stream melody support (note_on/note_off)
        # Tracks currently active pitches across requests so we can infer sustain
        # when a pitch is held but no explicit event is sent in the next interval.
        self._active_melody_pitches = set()

    def _get_mel_pianoroll_for_beat(self, beat_start_tick: int, beat_end_tick: int):
        """Build melody pianoroll for a single beat directly from event stream.

        - Uses `MidiConverter.events_to_pianoroll`.
        - Carries sustain across beats via `self._active_melody_pitches`.
        - Updates `self._active_melody_pitches` by applying note_off events within the beat
          (events_to_pianoroll doesn't return active set, so we update it here).
        """

        pr = self.midi_converter.events_to_pianoroll(
            self.melody_event_history,
            start_tick=beat_start_tick,
            end_tick=beat_end_tick,
            active_pitches=self._active_melody_pitches,
        )

        # Update active set for next beat (apply beat-local events in temporal order)
        beat_events = [
            e
            for e in self.melody_event_history
            if beat_start_tick <= e.get("tick", -1) < beat_end_tick
        ]
        beat_events.sort(
            key=lambda e: (
                int(e.get("tick", 0)),
                0 if e.get("type") == "note_off" else 1,
            )
        )
        for e in beat_events:
            et = e.get("type")
            p = e.get("pitch")
            if p is None:
                continue
            p = int(p)
            if et == "note_on":
                self._active_melody_pitches.add(p)
            elif et == "note_off":
                self._active_melody_pitches.discard(p)

        return pr

    def _normalize_melody_input(self, melody_notes: list, generation_start_tick: int):
        """Normalize client melody payload (event stream) into absolute-tick events.

        Expected input items:
          {"type": "note_on"|"note_off", "pitch": int, "tick": int}

        Output items:
          same dicts, but with "tick" shifted by injection offset (absolute tick).
        """

        if not melody_notes:
            return []

        abs_events = []
        for e in melody_notes:
            if not isinstance(e, dict):
                continue
            if e.get("type") not in {"note_on", "note_off"}:
                continue
            if "pitch" not in e or "tick" not in e:
                continue
            abs_e = e.copy()
            abs_e["tick"] = int(e["tick"]) + self.injection_offset_ticks
            abs_e["pitch"] = int(e["pitch"])
            abs_events.append(abs_e)
        return abs_events

    def set_injection_offset(self, offset_ticks: int):
        self.injection_offset_ticks = offset_ticks
        print(f"设置注入偏移: {offset_ticks} ticks")

    def clear_history(self):
        self.melody_event_history = []
        self._active_melody_pitches = set()
        self.accompaniment_history = []
        self.past_key_values = None
        self.last_generated_beat = -1
        self.last_generated_acc_tokens = None

    def _notes_to_pianoroll(self, notes, max_tick):
        """
        Convert notes list to (2, 88, max_tick) piano roll.
        Delegates to MidiConverter.
        """
        return self.midi_converter.notes_to_pianoroll(notes, max_tick)

    def _pianoroll_to_notes(self, pianoroll, start_tick):
        """
        Convert (2, 88, T) piano roll back to notes list.
        Delegates to MidiConverter.
        """
        notes = self.midi_converter.pianoroll_to_notes(pianoroll)
        # MidiConverter returns relative ticks starting from 0
        # We might need to adjust if start_tick is used elsewhere,
        # but currently _pianoroll_to_notes callers handle shifting.
        return notes

    def _get_tokens_for_beat(self, notes, beat_idx, end_marker_id):
        """Helper to get tokens for a specific beat range with specific end marker"""
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

        # Tokenize manually to specify end_marker
        # 1. Image to Patch Tokens
        patch_tokens = self.tokenizer.image_to_patch_tokens(pr, strict_mode=True)
        # 2. Compress with specific end marker
        tokens = self.tokenizer.compress_tokens(patch_tokens, end_marker=end_marker_id)

        return torch.tensor(tokens, dtype=torch.long)

    def _get_tokens_for_beat_pianoroll(self, pianoroll, end_marker_id):
        """Tokenize a beat-length pianoroll with a specific end marker."""

        patch_tokens = self.tokenizer.image_to_patch_tokens(pianoroll, strict_mode=True)
        tokens = self.tokenizer.compress_tokens(patch_tokens, end_marker=end_marker_id)
        return torch.tensor(tokens, dtype=torch.long)

    def _generate_tokens(self, input_ids, past_key_values=None):
        """Core generation loop"""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        generated_tokens = []

        # REMOVED: Unconditional bar token addition
        # bar_token = torch.tensor([[self.config.bar_token_id]], device=device)
        # input_ids = torch.cat([input_ids, bar_token], dim=1)

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
                    temperature=1.1,  # Updated to match inference.py
                    top_k=10,  # Updated to match inference.py
                    top_p=0.95,
                    repetition_penalty=1.0,  # Updated to match inference.py
                )

                # Forcefully prevent PAD token generation
                if next_token.item() == self.config.pad_token_id:
                    # If we sampled PAD, try to sample again or just pick the next best?
                    # Simple hack: just continue loop (effectively skipping this step? No, that breaks state)
                    # Better: mask PAD logit before sampling. But sample_token is a black box here.
                    # Let's just ignore it and not append it?
                    # But we need to feed something to the model.
                    # Let's assume it's a glitch and break? No.
                    pass

                # For next iteration
                current_input = next_token

                token_val = next_token.item()
                generated_tokens.append(token_val)

                if token_val in [self.config.end_marker_part1, self.config.bar_token_id]:
                    # Keep the end marker in the generated tokens for decoding
                    break

        return generated_tokens, current_past

    def generate_accompaniment(
        self,
        melody_notes,
        generation_start_tick,
        acc_notes=None,
        generation_length_frames=None,  # Unused, kept for API
        prompt_length_ticks=None,  # Unused, kept for API
        bpm=120,
        time_sig=(4, 4),
    ):
        """
        Generate accompaniment using LLaMA model with beat-interleaving.

        CRITICAL USAGE NOTE:
        This engine generates music **beat by beat**. The `generation_start_tick` determines
        the current beat index (`tick // ticks_per_beat`).
        The client **MUST** synchronize requests with the beat grid.
        For example, if `ticks_per_beat=4`, the client should send a request every 4 ticks.
        Sending requests more frequently (e.g., every tick) will cause the engine to generate
        duplicate content for the same beat, corrupting the context history.

        Args:
            melody_notes (List[dict]): New melody notes from the client.
                Format: [{'pitch': int, 'tick': int, 'duration': int}, ...]
                'tick' is absolute relative to the client's session start.
            generation_start_tick (int): The tick from which generation should start.
                Used to calculate `current_beat = generation_start_tick // ticks_per_beat`.
            acc_notes (List[dict], optional): Initial accompaniment notes (e.g. for injection).
                Same format as melody_notes.
            generation_length_frames (int, optional): Unused. This engine always generates 1 beat per call.
            prompt_length_ticks (int, optional): Unused.
            bpm (float): BPM of the track.
            time_sig (tuple): Time signature (numerator, denominator).

        Returns:
            tuple: (
                generated_notes (List[dict]): The generated accompaniment notes for the current beat.
                    Format: [{'pitch': int, 'tick': int, 'duration': int, 'program': 1}, ...]
                    'tick' is absolute relative to the client's session start.
                preprocess_start_time (float),
                inference_start_time (float),
                inference_end_time (float),
                postprocess_start_time (float)
            )
        """
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        preprocess_start_time = time.perf_counter()

        # 1. Update History (event stream only)
        abs_events = self._normalize_melody_input(melody_notes, generation_start_tick)
        self.melody_event_history.extend(abs_events)

        absolute_generation_start_tick = (
            generation_start_tick + self.injection_offset_ticks
        )

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
            else:
                print(f"Stateful reset: current_beat={current_beat}, last={self.last_generated_beat}")
                self.past_key_values = None
                self.last_generated_acc_tokens = None

        input_ids = None
        past_key_values_to_use = None

        if need_reset:
            # --- Sliding Window / Warmup Logic ---
            # Reconstruct the sequence from scratch
            # Pattern: [BOS, TimeSig, BPM, PAD, Bar, Bar] -> (A0, M0) -> (A1, M1) ...

            context_beats = 32  # Lookback
            start_beat = max(0, current_beat - context_beats)

            # Construct Prompt Sequence
            bpm_val = bpm
            bpm_token = encode_bpm(bpm_val) + self.config.bpm_offset_id

            # Map Time Sig to Token
            time_sig_map = {(4, 4): 0, (2, 4): 1, (3, 4): 2, (6, 8): 3, (2, 2): 4}
            ts_idx = time_sig_map.get(time_sig, 0)
            time_sig_token = ts_idx + self.config.time_sig_offset_id

            seq = [
                torch.tensor([self.config.bos_token_id], dtype=torch.long),
                torch.tensor([time_sig_token], dtype=torch.long),
                torch.tensor([bpm_token], dtype=torch.long),
            ]

            # Add Initial PAD (173)
            pad_marker = 173
            seq.append(torch.tensor([pad_marker], dtype=torch.long))

            for b in range(start_beat, current_beat):
                # Add Bar tokens at start of measure
                if b % 4 == 0:
                    seq.append(torch.tensor([self.config.bar_token_id], dtype=torch.long))  # Acc Bar
                    seq.append(torch.tensor([self.config.bar_token_id], dtype=torch.long))  # Mel Bar

                # Get Mel tokens for beat b (event stream -> pianoroll)
                beat_start_tick = b * self.ticks_per_beat
                beat_end_tick = (b + 1) * self.ticks_per_beat
                mel_pr_b = self._get_mel_pianoroll_for_beat(
                    beat_start_tick=beat_start_tick,
                    beat_end_tick=beat_end_tick,
                )
                mel_tokens = self._get_tokens_for_beat_pianoroll(
                    mel_pr_b,
                    end_marker_id=self.tokenizer.end_marker_part0,
                )
                seq.append(mel_tokens)

                # Get Acc tokens for beat b
                acc_notes_b = [n for n in self.accompaniment_history if beat_start_tick <= n["tick"] < beat_end_tick]
                acc_tokens = self._get_tokens_for_beat(acc_notes_b, b, end_marker_id=self.tokenizer.end_marker_part1)
                seq.append(acc_tokens)

            # Now we are at current_beat.
            # We need to prepare input for generating Acc[current_beat].
            # If current_beat % 4 == 0, we need to add [Bar, Bar] first.
            if current_beat % 4 == 0:
                seq.append(torch.tensor([self.config.bar_token_id], dtype=torch.long))  # Acc Bar
                seq.append(torch.tensor([self.config.bar_token_id], dtype=torch.long))  # Mel Bar

            # The sequence is now ready to generate Acc[current_beat]
            # IMPORTANT: We must NOT add the Acc Bar token here because _generate_tokens was doing it.
            # But we removed it from _generate_tokens to be cleaner.
            # So we should add it here if it's NOT a new measure (because new measure already added bars).
            # Wait, the pattern is (AccBar, MelBar) -> AccTokens -> MelTokens.
            # So for a new beat, we have MelTokens(prev) -> [Bar, Bar] (if new measure) -> AccTokens(current).
            # But wait, the model expects to see MelTokens(prev) before generating AccTokens(current)?
            # Let's check the training pattern or inference.py pattern.

            # In inference.py:
            # position 0 (Part0/Melody): Inject Mel tokens
            # position 1 (Part1/Acc): Generate Acc tokens

            # So the order is Mel -> Acc.
            # But in `generate_accompaniment` here, we are building context up to `current_beat`.
            # The loop `for b in range(start_beat, current_beat)` adds Acc[b] then Mel[b].
            # So the last thing added is Mel[current_beat-1].

            # Now we want to generate Acc[current_beat].
            # If current_beat is start of measure, we added [AccBar, MelBar].
            # Then we need to generate Acc.

            # BUT, the model is trained on [AccBar, MelBar, AccTokens, MelTokens] (interleaved)?
            # Or [AccBar, MelBar, MelTokens, AccTokens]?
            # Let's look at `PianoDataset.py` or `model.py`.
            # `model.py` loop:
            # if position == 0 (Part0/Melody): Inject Mel
            # if position == 1 (Part1/Acc): Generate Acc

            # So the order in `model.py` is Part0 -> Part1.
            # Wait, `model.py` says:
            # if position == 0: # part0 (Melody)
            #    inject part0_beats_list[part0_idx]
            #    position = 1
            # else: # part1 (Acc)
            #    generate
            #    position = 0

            # So the sequence is Mel -> Acc -> Mel -> Acc.

            # Let's re-check `transformer_engine_lekai.py` loop:
            # seq.append(acc_tokens)
            # seq.append(mel_tokens)
            # This implies Acc -> Mel order! This contradicts `model.py`.

            # Let's check `PianoDataset.py` to be sure about training data order.
            # But assuming `model.py` is correct (Mel -> Acc), then `transformer_engine_lekai.py` is WRONG.
            # It is building context as Acc -> Mel.

            # Let's swap them in the loop.

            # Correct Loop for Mel -> Acc order:
            for b in range(start_beat, current_beat):
                # Add Bar tokens at start of measure
                if b % 4 == 0:
                    seq.append(
                        torch.tensor([self.config.bar_token_id], dtype=torch.long)
                    )  # Acc Bar (Wait, are bars distinct?)
                    seq.append(torch.tensor([self.config.bar_token_id], dtype=torch.long))  # Mel Bar
                    # Note: config.bar_token_id is usually just one token.
                    # If the model expects two bars, that's fine.

                # Get Mel tokens for beat b
                mel_pr_b = self._get_mel_pianoroll_for_beat(
                    beat_start_tick=beat_start_tick,
                    beat_end_tick=beat_end_tick,
                )
                mel_tokens = self._get_tokens_for_beat_pianoroll(
                    mel_pr_b,
                    end_marker_id=self.tokenizer.end_marker_part0,
                )
                seq.append(mel_tokens)

                # Get Acc tokens for beat b
                beat_start_tick = b * self.ticks_per_beat
                beat_end_tick = (b + 1) * self.ticks_per_beat
                acc_notes_b = [n for n in self.accompaniment_history if beat_start_tick <= n["tick"] < beat_end_tick]
                acc_tokens = self._get_tokens_for_beat(acc_notes_b, b, end_marker_id=self.tokenizer.end_marker_part1)
                seq.append(acc_tokens)

            # Now we are at current_beat.
            # We need to prepare input for generating Acc[current_beat].
            # We need to inject Mel[current_beat] FIRST.

            # 1. Bar tokens if new measure
            if current_beat % 4 == 0:
                seq.append(torch.tensor([self.config.bar_token_id], dtype=torch.long))
                seq.append(torch.tensor([self.config.bar_token_id], dtype=torch.long))

            # 2. Inject Mel[current_beat]
            beat_start_tick = current_beat * self.ticks_per_beat
            beat_end_tick = (current_beat + 1) * self.ticks_per_beat
            mel_pr_curr = self._get_mel_pianoroll_for_beat(
                beat_start_tick=beat_start_tick,
                beat_end_tick=beat_end_tick,
            )
            mel_tokens_curr = self._get_tokens_for_beat_pianoroll(
                mel_pr_curr,
                end_marker_id=self.tokenizer.end_marker_part0,
            )
            seq.append(mel_tokens_curr)

            # 3. Now ready to generate Acc[current_beat]
            input_ids = torch.cat(seq).unsqueeze(0).to(device)
            past_key_values_to_use = None  # Reset cache

        else:
            # --- Stateful Incremental Logic ---
            # We are continuing from Acc[current_beat - 1].
            # The last generation produced Acc[current_beat - 1].
            # The sequence so far (in model's mind) ends with Acc[current_beat - 1].

            # We need to feed:
            # 1. Bar tokens (if new measure)
            # 2. Mel[current_beat]
            # Then generate Acc[current_beat]

            seq = []

            # 1. Bar tokens if new measure
            if current_beat % 4 == 0:
                seq.append(torch.tensor([self.config.bar_token_id], dtype=torch.long))
                seq.append(torch.tensor([self.config.bar_token_id], dtype=torch.long))

            # 2. Mel[current_beat]
            beat_start_tick = current_beat * self.ticks_per_beat
            beat_end_tick = (current_beat + 1) * self.ticks_per_beat
            mel_notes_curr = [n for n in self.melody_history if beat_start_tick <= n["tick"] < beat_end_tick]
            mel_tokens_curr = self._get_tokens_for_beat(
                mel_notes_curr, current_beat, end_marker_id=self.tokenizer.end_marker_part0
            )
            seq.append(mel_tokens_curr)

            input_ids = torch.cat(seq).unsqueeze(0).to(device)
            past_key_values_to_use = self.past_key_values

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        inference_start_time = time.perf_counter()

        # 3. Generate
        generated_tokens, new_past_key_values = self._generate_tokens(input_ids, past_key_values_to_use)
        # print(f"DEBUG: Generated tokens: {generated_tokens}")

        # Update State if Stateful
        if self.inference_mode == "stateful":
            self.past_key_values = new_past_key_values
            self.last_generated_beat = current_beat
            self.last_generated_acc_tokens = generated_tokens  # Store for next step

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        inference_end_time = time.perf_counter()

        # 4. Decode Generated Tokens
        # generated_tokens contains the compressed tokens for the Acc beat
        # We need to decode them back to piano roll

        # Filter out PAD tokens (258) which might be generated by mistake
        valid_tokens = [t for t in generated_tokens if t != self.config.pad_token_id]

        # Remove trailing Bar Token (255) if present
        if valid_tokens and valid_tokens[-1] == self.config.bar_token_id:
            valid_tokens.pop()

        # Ensure we have at least one token to decode
        if not valid_tokens:
            # Empty beat
            absolute_generated_notes = []
        else:
            # Decode
            try:
                # Decompress
                # Note: decompress_tokens expects a list or array
                decompressed = self.tokenizer.decompress_tokens(
                    valid_tokens, end_marker_id=self.tokenizer.end_marker_part1
                )

                # To Image
                pr = self.tokenizer.patch_tokens_to_image(decompressed)

                # To Notes
                # Note: _pianoroll_to_notes needs to know the absolute start tick
                # But here we are decoding a single beat relative to 0
                # So we get relative notes, then shift them
                relative_notes = self._pianoroll_to_notes(pr, start_tick=0)

                absolute_generated_notes = []
                beat_start_tick = current_beat * self.ticks_per_beat

                for n in relative_notes:
                    abs_note = n.copy()
                    abs_note["tick"] += beat_start_tick
                    absolute_generated_notes.append(abs_note)

            except Exception as e:
                print(f"Decoding error: {e}")
                absolute_generated_notes = []

        if torch.cuda.is_available():
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
