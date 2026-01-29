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
        
        # Track active accompaniment pitches across beats for proper note_off generation
        # at beat boundaries (when a note ends exactly at the beat boundary)
        self._active_acc_pitches = set()

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
        """Helper to get tokens for a specific beat range with specific end marker.
        
        Correctly handles cross-beat notes:
        - Notes starting in this beat: onset + sustain
        - Notes continuing from previous beat: sustain only (no onset)
        """
        beat_start_tick = beat_idx * self.ticks_per_beat
        beat_end_tick = (beat_idx + 1) * self.ticks_per_beat

        # Build pianoroll directly to handle cross-beat notes correctly
        pr = np.zeros((2, 88, self.ticks_per_beat), dtype=np.uint8)
        
        for n in notes:
            note_start = n["tick"]
            note_end = n["tick"] + n.get("duration", 1)
            pitch = n["pitch"]
            
            # Check if note overlaps with this beat
            if note_start < beat_end_tick and note_end > beat_start_tick:
                pitch_idx = pitch - 21  # MIDI pitch to pianoroll index
                if 0 <= pitch_idx < 88:
                    # Calculate overlap within this beat
                    overlap_start = max(0, note_start - beat_start_tick)
                    overlap_end = min(self.ticks_per_beat, note_end - beat_start_tick)
                    
                    # Fill sustain channel for the overlap
                    for t in range(overlap_start, overlap_end):
                        pr[0, pitch_idx, t] = 1  # sustain
                    
                    # Only set onset if note actually STARTS in this beat
                    if note_start >= beat_start_tick and note_start < beat_end_tick:
                        onset_tick = note_start - beat_start_tick
                        pr[1, pitch_idx, onset_tick] = 1  # onset

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
            # Training data order from PianoDataset.py: Mel -> Acc (part0 -> part1)
            # So we inject Mel[current_beat] FIRST, then generate Acc[current_beat].

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
            mel_pr_curr = self._get_mel_pianoroll_for_beat(
                beat_start_tick=beat_start_tick,
                beat_end_tick=beat_end_tick,
            )
            mel_tokens_curr = self._get_tokens_for_beat_pianoroll(
                mel_pr_curr,
                end_marker_id=self.tokenizer.end_marker_part0,
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
        pr = None
        if not valid_tokens:
            # Empty beat - but still need to check if any active pitches need note_off
            # since this beat has no sustain, all active pitches should be closed
            absolute_generated_events = []
            for pitch in self._active_acc_pitches:
                absolute_generated_events.append({
                    "type": "note_off",
                    "pitch": int(pitch),
                    "tick": int(beat_start_tick)
                })
            self._active_acc_pitches = set()
        else:
            # Decode
            try:
                decompressed = self.tokenizer.decompress_tokens(
                    valid_tokens, end_marker_id=self.tokenizer.end_marker_part1
                )
                pr = self.tokenizer.patch_tokens_to_image(decompressed)

                beat_start_tick = current_beat * self.ticks_per_beat
                # Convert pianoroll (beat-local) to absolute-tick events
                # Pass active_acc_pitches to correctly detect notes that ended at beat boundary
                absolute_generated_events, self._active_acc_pitches = self.midi_converter.pianoroll_to_events(
                    pr,
                    start_tick=beat_start_tick,
                    active_pitches=self._active_acc_pitches,
                )
                
                # DEBUG: Check sustain at beat end (helps diagnose cross-beat note issues)
                if pr is not None and pr.shape[2] >= 1:
                    sustain_at_end = pr[0, :, -1]  # sustain channel, all pitches, last tick
                    active_pitches_at_end = np.where(sustain_at_end > 0)[0] + 21  # Convert to MIDI pitch
                    if len(active_pitches_at_end) > 0:
                        print(f"    [DEBUG] Beat {current_beat}: sustain active at beat end for pitches: {list(active_pitches_at_end)}")
                        
            except Exception as e:
                print(f"Decoding error: {e}")
                absolute_generated_events = []

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        postprocess_start_time = time.perf_counter()

        # 6. Update History & Return
        # NOTE: accompaniment_history is still note-based elsewhere in this engine.
        # We keep updating it with decoded notes for internal context, but these are
        # not returned to the client anymore.
        try:
            # Derive notes for history/context only
            rel_notes_for_history = self._pianoroll_to_notes(pr, start_tick=0) if pr is not None else []
            beat_start_tick = current_beat * self.ticks_per_beat
            abs_notes_for_history = []
            for n in rel_notes_for_history:
                abs_n = n.copy()
                abs_n["tick"] += beat_start_tick
                abs_notes_for_history.append(abs_n)
            self.accompaniment_history.extend(abs_notes_for_history)
        except Exception:
            # If decode-to-notes fails, we can still return events
            pass

        # Return relative events (hiding injection offset)
        relative_generated_events = []
        for e in absolute_generated_events:
            re = e.copy()
            re["tick"] = int(e["tick"]) - self.injection_offset_ticks
            relative_generated_events.append(re)

        return (
            relative_generated_events,
            preprocess_start_time,
            inference_start_time,
            inference_end_time,
            postprocess_start_time,
            pr,  # Return the pianoroll for debugging
        )

# ================================================================================
# Fake real time
# ================================================================================

def visualize_pianoroll(full_pr, title="Generated Accompaniment Pianoroll", max_pitches=20):
    """
    Visualize a pianoroll as ASCII art.
    
    Args:
        full_pr: np.array of shape (2, 88, T) where channel 0=sustain, 1=onset
        title: Title to print
        max_pitches: Only show pitches that have activity (up to this many)
    """
    if full_pr is None:
        print(f"\n{title}: No pianoroll data")
        return
    
    sustain = full_pr[0]  # (88, T)
    onset = full_pr[1]    # (88, T)
    T = full_pr.shape[2]
    
    # Find pitches with any activity
    active_pitch_mask = np.any(sustain > 0, axis=1) | np.any(onset > 0, axis=1)
    active_pitches = np.where(active_pitch_mask)[0]
    
    if len(active_pitches) == 0:
        print(f"\n{title}: Empty pianoroll (no notes)")
        return
    
    # Limit to max_pitches (show highest and lowest)
    if len(active_pitches) > max_pitches:
        # Take top and bottom pitches
        half = max_pitches // 2
        active_pitches = np.concatenate([active_pitches[:half], active_pitches[-half:]])
    
    print(f"\n{title}")
    print(f"Shape: (2, 88, {T}) = {T // 4} beats")
    print("Legend: '.' = empty, 'O' = onset+sustain, 'S' = sustain only, 'o' = onset only")
    print("-" * (T + 15))
    
    # Print beat markers
    beat_header = "Pitch     |"
    for t in range(T):
        if t % 4 == 0:
            beat_header += str((t // 4) % 10)
        else:
            beat_header += " "
    print(beat_header)
    print("-" * (T + 15))
    
    for pitch_idx in reversed(active_pitches):  # High to low
        midi_pitch = pitch_idx + 21
        row = f"P{midi_pitch:3d} ({pitch_idx:2d})|"
        for t in range(T):
            s = sustain[pitch_idx, t]
            o = onset[pitch_idx, t]
            if o > 0 and s > 0:
                row += "O"
            elif s > 0:
                row += "S"
            elif o > 0:
                row += "o"
            else:
                row += "."
        print(row)
    
    print("-" * (T + 15))
    
    # Summary stats
    total_sustain = np.sum(sustain > 0)
    total_onset = np.sum(onset > 0)
    print(f"Total: {total_onset} onsets, {total_sustain} sustain ticks")
    
    # Check for sustain-only (continuation) patterns
    sustain_only = np.sum((sustain > 0) & (onset == 0))
    print(f"Sustain-only ticks (continuations): {sustain_only}")


def notes_to_events(notes):
    """
    Convert note list (pitch, tick, duration) to event stream (note_on/note_off).
    This is needed because generate_accompaniment expects event-stream format.
    """
    events = []
    for n in notes:
        pitch = n["pitch"]
        tick = n["tick"]
        duration = n.get("duration", 1)
        events.append({"type": "note_on", "pitch": pitch, "tick": tick})
        events.append({"type": "note_off", "pitch": pitch, "tick": tick + duration})
    # Sort by tick, note_off before note_on if same tick
    events.sort(key=lambda e: (e["tick"], 0 if e["type"] == "note_off" else 1))
    return events


def events_to_notes(events, debug=False):
    """
    Convert event stream (note_on/note_off) back to note list (pitch, tick, duration).
    """
    # Track active notes: pitch -> onset_tick
    active = {}
    notes = []

    # Sort events by tick, note_off before note_on
    sorted_events = sorted(events, key=lambda e: (e["tick"], 0 if e["type"] == "note_off" else 1))

    for e in sorted_events:
        pitch = e["pitch"]
        tick = e["tick"]
        if e["type"] == "note_on":
            active[pitch] = tick
        elif e["type"] == "note_off":
            if pitch in active:
                onset = active.pop(pitch)
                duration = max(1, tick - onset)
                notes.append({"pitch": pitch, "tick": onset, "duration": duration, "velocity": 80})

    # Close any remaining active notes at the last tick
    if active and sorted_events:
        last_tick = max(e["tick"] for e in sorted_events)
        if debug:
            print(f"\n[DEBUG events_to_notes] Closing {len(active)} unclosed notes at last_tick={last_tick}:")
        for pitch, onset in active.items():
            duration = max(1, last_tick - onset)
            if debug:
                print(f"  - pitch={pitch}, onset={onset}, duration={duration} (ticks)")
            notes.append({"pitch": pitch, "tick": onset, "duration": duration, "velocity": 80})

    notes.sort(key=lambda n: n["tick"])
    return notes


def save_to_midi(melody_notes, acc_notes, output_path, bpm=120, ticks_per_beat=4):
    """
    Save melody and accompaniment notes to a MIDI file with two tracks.

    Args:
        melody_notes: List of {pitch, tick, duration, ...} for melody
        acc_notes: List of {pitch, tick, duration, ...} for accompaniment
        output_path: Path to save the MIDI file
        bpm: Tempo in BPM
        ticks_per_beat: Ticks per beat used in the engine (for time conversion)
    """
    midi = pretty_midi.PrettyMIDI(initial_tempo=bpm)

    # Time conversion: 1 tick = 1/ticks_per_beat beat = 60/(bpm*ticks_per_beat) seconds
    seconds_per_tick = 60.0 / (bpm * ticks_per_beat)

    # Track 1: Melody
    melody_inst = pretty_midi.Instrument(program=0, name="Melody")
    for n in melody_notes:
        start_time = n["tick"] * seconds_per_tick
        end_time = (n["tick"] + n.get("duration", 1)) * seconds_per_tick
        velocity = n.get("velocity", 80)
        if end_time <= start_time:
            end_time = start_time + 0.05
        note = pretty_midi.Note(
            velocity=velocity,
            pitch=n["pitch"],
            start=start_time,
            end=end_time
        )
        melody_inst.notes.append(note)
    midi.instruments.append(melody_inst)

    # Track 2: Accompaniment
    acc_inst = pretty_midi.Instrument(program=0, name="Accompaniment")
    for n in acc_notes:
        start_time = n["tick"] * seconds_per_tick
        end_time = (n["tick"] + n.get("duration", 1)) * seconds_per_tick
        velocity = n.get("velocity", 80)
        if end_time <= start_time:
            end_time = start_time + 0.05
        note = pretty_midi.Note(
            velocity=velocity,
            pitch=n["pitch"],
            start=start_time,
            end=end_time
        )
        acc_inst.notes.append(note)
    midi.instruments.append(acc_inst)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    midi.write(output_path)
    print(f"✓ Saved MIDI to: {output_path}")
    print(f"  - Melody: {len(melody_notes)} notes")
    print(f"  - Accompaniment: {len(acc_notes)} notes")
    print(f"  - BPM: {bpm}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test Lekai Inference Engine")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint (safetensors)")
    parser.add_argument("--midi", type=str, default="input/mel/001.mid",
                        help="Path to melody MIDI file")
    parser.add_argument("--prompt-beats", type=int, default=0,
                        help="Number of beats to use as prompt (from ground truth accompaniment)")
    parser.add_argument("--max-beats", type=int, default=96,
                        help="Maximum number of beats to generate")
    parser.add_argument("--output", type=str, default="output/generated_accompaniment.mid",
                        help="Output MIDI file path")
    parser.add_argument("--bpm", type=float, default=None,
                        help="Override BPM (default: read from MIDI file)")
    parser.add_argument("--beats-per-gen", type=int, default=1,
                        help="Number of beats to generate per inference call (1=beat-by-beat, 4=measure-by-measure)")
    parser.add_argument("--inference-mode", type=str, default="stateful", choices=["stateful", "sliding_window"],
                        help="Inference mode: 'stateful' uses KV cache for efficiency, 'sliding_window' rebuilds context each time")
    args = parser.parse_args()

    # Checkpoint path
    checkpoint_path = args.checkpoint or os.getenv(
        "CHECKPOINT_PATH", "/home/xiaosongma/M2A_checkpoints/ModelLekai/epoch_4_1104_1204/model.safetensors"
    )

    try:
        inference_engine = InferenceEngineLekai(
            checkpoint_path=checkpoint_path,
            model_size="llama",
            inference_mode=args.inference_mode,
        )
    except Exception as e:
        print(f"Init failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

    # Load melody MIDI
    midi_path = args.midi
    if not os.path.exists(midi_path):
        print(f"Test MIDI file not found: {midi_path}")
        print("Please provide a valid MIDI file path.")
        exit(1)

    try:
        # midi_to_note returns notes with {pitch, tick, duration, ...}
        melody_notes, metadata, max_tick = midi_to_note(midi_path)
        print(f"Loaded melody: {len(melody_notes)} notes, max_tick={max_tick}")
        print(f"Metadata: {metadata}")

        if not melody_notes:
            print("Warning: No notes found in MIDI file!")
            exit(1)

        # Load accompaniment MIDI for prompt (if prompt_beats > 0)
        prompt_beats = args.prompt_beats
        acc_prompt_notes = []
        acc_prompt_events = []

        if prompt_beats > 0:
            # Derive acc path from melody path (replace /mel/ with /acc/)
            acc_path = midi_path.replace("/mel/", "/acc/")
            if not os.path.exists(acc_path):
                print(f"Warning: Accompaniment file not found for prompt: {acc_path}")
                print("Proceeding without prompt.")
                prompt_beats = 0
            else:
                acc_notes_full, _, _ = midi_to_note(acc_path)
                prompt_end_tick = prompt_beats * 4  # ticks_per_beat = 4
                # Filter notes within prompt range, truncate duration if needed
                for n in acc_notes_full:
                    if n["tick"] < prompt_end_tick:
                        note = n.copy()
                        # Truncate duration if note extends beyond prompt range
                        note_end_tick = n["tick"] + n.get("duration", 1)
                        if note_end_tick > prompt_end_tick:
                            note["duration"] = max(1, prompt_end_tick - n["tick"])
                        acc_prompt_notes.append(note)
                acc_prompt_events = notes_to_events(acc_prompt_notes)
                print(f"Loaded prompt: {len(acc_prompt_notes)} acc notes for first {prompt_beats} beats")

        # Convert melody notes to event stream (note_on/note_off) for the engine
        all_events = notes_to_events(melody_notes)
        print(f"Converted to {len(all_events)} events (note_on/note_off)")

        # Track which events we've already sent (incremental feeding)
        last_sent_idx = 0
        last_acc_prompt_idx = 0

        # Generate beat by beat (engine expects ticks_per_beat=4)
        ticks_per_beat = 4
        num_beats = min((max_tick + ticks_per_beat - 1) // ticks_per_beat, args.max_beats)
        
        # Determine BPM to use (command line > MIDI file > default)
        bpm = args.bpm if args.bpm is not None else metadata.get("bpm", 120)
        print(f"Using BPM: {bpm}" + (" (from command line)" if args.bpm else " (from MIDI file)"))
        
        beats_per_gen = args.beats_per_gen
        print(f"Beats per generation: {beats_per_gen}")

        print(f"\n--- Starting generation for {num_beats} beats (prompt: {prompt_beats} beats) ---\n")

        # Collect all generated/prompt accompaniment events
        all_acc_events = []
        
        # Collect all pianorolls for visualization
        all_pianorolls = []  # List of (beat_idx, pr)
        
        # For prompt injection: track active acc notes across beats to get correct duration
        acc_active_notes = {}  # pitch -> onset_tick
        
        # Track active generated notes (for multi-beat generation)
        gen_active_notes = {}  # pitch -> onset_tick

        beat_idx = 0
        while beat_idx < num_beats:
            generation_start_tick = beat_idx * ticks_per_beat
            beat_end_tick = (beat_idx + 1) * ticks_per_beat

            # Find new melody events to send for this beat
            new_events = []
            while last_sent_idx < len(all_events) and all_events[last_sent_idx]["tick"] < beat_end_tick:
                new_events.append(all_events[last_sent_idx])
                last_sent_idx += 1

            if beat_idx < prompt_beats:
                # Use ground truth accompaniment as prompt
                # Find acc events for this beat
                beat_acc_events = []
                while (last_acc_prompt_idx < len(acc_prompt_events) and
                       acc_prompt_events[last_acc_prompt_idx]["tick"] < beat_end_tick):
                    beat_acc_events.append(acc_prompt_events[last_acc_prompt_idx])
                    last_acc_prompt_idx += 1

                # Inject melody events into engine history
                # Note: In fake real time test, injection_offset_ticks = 0, so tick values stay the same
                abs_mel_events = inference_engine._normalize_melody_input(new_events, generation_start_tick)
                inference_engine.melody_event_history.extend(abs_mel_events)

                # Update _active_melody_pitches for proper sustain tracking
                for e in abs_mel_events:
                    et = e.get("type")
                    p = e.get("pitch")
                    if p is None:
                        continue
                    p = int(p)
                    if et == "note_on":
                        inference_engine._active_melody_pitches.add(p)
                    elif et == "note_off":
                        inference_engine._active_melody_pitches.discard(p)

                # Process acc events and track active notes for proper duration calculation
                beat_acc_notes = []
                for e in beat_acc_events:
                    pitch = e["pitch"]
                    tick = e["tick"]
                    if e["type"] == "note_on":
                        acc_active_notes[pitch] = tick
                    elif e["type"] == "note_off":
                        if pitch in acc_active_notes:
                            onset = acc_active_notes.pop(pitch)
                            duration = max(1, tick - onset)
                            beat_acc_notes.append({
                                "pitch": pitch, 
                                "tick": onset, 
                                "duration": duration, 
                                "velocity": 80
                            })
                
                # Inject acc notes into engine history
                inference_engine.accompaniment_history.extend(beat_acc_notes)

                # Collect for output
                all_acc_events.extend(beat_acc_events)

                # Update engine state (important for stateful mode continuity)
                inference_engine.last_generated_beat = beat_idx
                # Reset past_key_values since we're manually injecting (will rebuild context on first generation)
                inference_engine.past_key_values = None

                print(f"Beat {beat_idx:3d} | gen_tick={generation_start_tick:4d} | "
                      f"[PROMPT] mel={len(new_events):2d} | acc={len(beat_acc_events):2d}")
                beat_idx += 1
            else:
                # Multi-beat generation mode
                # Determine how many beats to generate in this batch
                remaining_beats = num_beats - beat_idx
                batch_size = min(beats_per_gen, remaining_beats)
                
                batch_acc_events = []
                batch_start = beat_idx
                batch_total_time = 0.0
                batch_pianorolls = []  # Collect pianorolls for visualization
                
                for b_offset in range(batch_size):
                    curr_beat = beat_idx + b_offset
                    gen_start_tick = curr_beat * ticks_per_beat
                    gen_end_tick = (curr_beat + 1) * ticks_per_beat
                    
                    # Collect melody events for this beat
                    beat_mel_events = []
                    while last_sent_idx < len(all_events) and all_events[last_sent_idx]["tick"] < gen_end_tick:
                        beat_mel_events.append(all_events[last_sent_idx])
                        last_sent_idx += 1
                    
                    # For first beat in batch, we already collected events above
                    if b_offset == 0:
                        beat_mel_events = new_events
                    
                    # Generate accompaniment for this beat
                    acc_events, pre_t, inf_start, inf_end, post_t, beat_pr = inference_engine.generate_accompaniment(
                        melody_notes=beat_mel_events,
                        generation_start_tick=gen_start_tick,
                        bpm=bpm,
                    )
                    
                    # Collect pianoroll
                    if beat_pr is not None:
                        batch_pianorolls.append((curr_beat, beat_pr))
                    
                    batch_total_time += (inf_end - inf_start)
                    
                    # Determine if this is the last beat in the batch
                    is_last_in_batch = (b_offset == batch_size - 1)
                    
                    # Process events: track note_on/note_off
                    for e in acc_events:
                        pitch = e["pitch"]
                        tick = e["tick"]
                        if e["type"] == "note_on":
                            # If there's already an active note at this pitch, close it first
                            if pitch in gen_active_notes:
                                onset = gen_active_notes.pop(pitch)
                                # Emit note_off at current tick
                                batch_acc_events.append({
                                    "type": "note_off", "pitch": pitch, "tick": tick
                                })
                            gen_active_notes[pitch] = tick
                            batch_acc_events.append(e)
                        elif e["type"] == "note_off":
                            if pitch in gen_active_notes:
                                gen_active_notes.pop(pitch)
                            batch_acc_events.append(e)
                    
                    # If last beat in batch, close all remaining active notes
                    if is_last_in_batch and gen_active_notes:
                        close_tick = gen_end_tick
                        for pitch in list(gen_active_notes.keys()):
                            batch_acc_events.append({
                                "type": "note_off", "pitch": pitch, "tick": close_tick
                            })
                            gen_active_notes.pop(pitch)
                
                # Collect batch events
                all_acc_events.extend(batch_acc_events)
                
                # Collect pianorolls for visualization
                all_pianorolls.extend(batch_pianorolls)
                
                # Stats
                batch_on = sum(1 for e in batch_acc_events if e["type"] == "note_on")
                batch_off = sum(1 for e in batch_acc_events if e["type"] == "note_off")
                balance_info = f"on={batch_on},off={batch_off}" if batch_on != batch_off else f"balanced={batch_on}"
                
                inf_time_ms = batch_total_time * 1000
                print(f"Beat {batch_start:3d}-{batch_start + batch_size - 1:3d} | "
                      f"gen_tick={batch_start * ticks_per_beat:4d}-{(batch_start + batch_size) * ticks_per_beat:4d} | "
                      f"generated_acc={len(batch_acc_events):3d} | "
                      f"inf_time={inf_time_ms:.1f}ms | {balance_info}")
                
                beat_idx += batch_size

        print("\n--- Generation complete ---")
        
        # Visualize the full generated pianoroll
        if all_pianorolls:
            # Concatenate all beat pianorolls into one full pianoroll
            # Each pr is shape (2, 88, 4) for 4 ticks per beat
            all_prs_sorted = sorted(all_pianorolls, key=lambda x: x[0])
            pr_list = [pr for _, pr in all_prs_sorted]
            full_pr = np.concatenate(pr_list, axis=2)  # (2, 88, total_ticks)
            visualize_pianoroll(full_pr, title=f"Generated Accompaniment (Beats {all_prs_sorted[0][0]} - {all_prs_sorted[-1][0]})")

        # Analyze event balance before conversion
        note_on_count = sum(1 for e in all_acc_events if e["type"] == "note_on")
        note_off_count = sum(1 for e in all_acc_events if e["type"] == "note_off")
        print(f"\n[DEBUG] Event balance: note_on={note_on_count}, note_off={note_off_count}, diff={note_on_count - note_off_count}")
        
        if note_on_count != note_off_count:
            print("[DEBUG] WARNING: note_on/note_off count mismatch! Some notes may have missing note_off.")

        # Convert accompaniment events back to notes (with debug=True)
        acc_notes = events_to_notes(all_acc_events, debug=True)
        print(f"\nTotal accompaniment: {len(all_acc_events)} events -> {len(acc_notes)} notes")
        print(f"  - Prompt beats: {prompt_beats}")
        print(f"  - Generated beats: {num_beats - prompt_beats}")
        
        # Detect abnormally long notes
        long_notes = [n for n in acc_notes if n["duration"] > 16]  # > 4 beats
        if long_notes:
            print(f"\n[DEBUG] Found {len(long_notes)} abnormally long notes (duration > 16 ticks = 4 beats):")
            for n in long_notes[:10]:  # Show first 10
                print(f"  - pitch={n['pitch']}, tick={n['tick']}, duration={n['duration']} ticks ({n['duration']/4:.1f} beats)")

        # Save to MIDI (bpm already determined above)
        output_path = args.output
        save_to_midi(
            melody_notes=melody_notes,
            acc_notes=acc_notes,
            output_path=output_path,
            bpm=bpm,
            ticks_per_beat=ticks_per_beat
        )

        print(f"\n🎵 You can now listen to the generated MIDI at: {output_path}")

    except Exception as e:
        print(f"Test run failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
