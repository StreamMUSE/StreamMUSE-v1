"""Adapter for PianoLLaMA model to work with beat tokens directly."""

from __future__ import annotations

import os
from typing import List, Optional

import numpy as np
import torch

from .model import PianoLLaMA
from .my_tokenizer import PianoRollTokenizer
from .PianoDataset import process_measure_with_beat_interleaving, encode_bpm


class PianoLLaMAAdapter:
    """
    Adapter to run PianoLLaMA inference without requiring a Dataset.
    
    Takes beat tokens directly as input and generates accompaniment.
    """
    
    # Token constants (from PianoDataset)
    BOS_TOKEN = 1
    BAR_TOKEN = 255
    BPM_OFFSET_ID = 5
    TIME_SIG_OFFSET_ID = 0
    PAD_MARKER = 173
    
    def __init__(
        self,
        model: PianoLLaMA,
        tokenizer: PianoRollTokenizer,
        device: str = "cpu",
        use_cache: bool = True,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.use_cache = use_cache
        self.model.eval()
        
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        device: str = "cpu",
        dtype: Optional[torch.dtype] = None,
        use_cache: bool = True,
    ) -> "PianoLLaMAAdapter":
        """Load model from checkpoint and create adapter."""
        from .config import ModelConfig
        from .inference import load_model
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        model_config = ModelConfig()
        model = load_model(
            model_path=checkpoint_path,
            model_config=model_config,
            device=device,
            dtype=dtype,
            use_cache=use_cache,
        )
        
        tokenizer = PianoRollTokenizer(patch_h=1, patch_w=4)
        
        return cls(model=model, tokenizer=tokenizer, device=device, use_cache=use_cache)
    
    def generate_from_beats(
        self,
        part0_beats: List[torch.Tensor],
        num_beats_to_generate: int,
        bpm: int = 120,
        time_signature_idx: int = 4,  # 4/4
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        repetition_penalty: float = 1.2,
        verbose: bool = False,
    ) -> List[List[int]]:
        """
        Generate accompaniment (part1) from melody beats (part0).
        
        Args:
            part0_beats: List of beat token tensors for melody (part0)
            num_beats_to_generate: Number of beats to generate
            bpm: BPM value
            time_signature_idx: Time signature index (4 for 4/4)
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Top-p sampling
            repetition_penalty: Repetition penalty
            verbose: Print debug info
            
        Returns:
            List of part1 beat tokens (each beat is a list of token IDs)
        """
        from .generation_utils import sample_token
        
        # Construct initial prompt (similar to generate_accompaniment)
        bpm_token = encode_bpm(bpm) + self.BPM_OFFSET_ID
        
        initial_tokens = [
            torch.tensor([self.BOS_TOKEN], dtype=torch.long),
            torch.tensor([time_signature_idx + self.TIME_SIG_OFFSET_ID], dtype=torch.long),
            torch.tensor([bpm_token], dtype=torch.long),
        ]
        
        generated = torch.cat(initial_tokens, dim=0).unsqueeze(0).to(self.device)
        
        # Generate interleaved part0/part1
        part1_beats_generated: List[List[int]] = []
        part0_idx = 0
        part1_idx = 0
        position = 0  # 0=part0, 1=part1
        
        part1_end_marker = 171
        part1_empty_marker = 169
        max_iterations = num_beats_to_generate * 100  # Safety limit
        
        current_part1_beat: List[int] = []
        past_key_values = None
        
        with torch.no_grad():
            for iteration in range(max_iterations):
                if position == 0:  # Inject part0 (melody)
                    if part0_idx < len(part0_beats):
                        next_part0_beat = part0_beats[part0_idx].to(self.device)
                        generated = torch.cat([generated, next_part0_beat.unsqueeze(0)], dim=1)
                        part0_idx += 1
                        position = 1
                        past_key_values = None
                        continue
                    else:
                        # No more part0 to inject, we're done with prompt
                        # Now generate part1 autoregressively
                        break
                        
                else:  # Generate part1 (accompaniment)
                    if part1_idx >= num_beats_to_generate:
                        break
                    
                    # Generate one token
                    outputs = self.model(
                        input_ids=generated[:, -1:] if past_key_values else generated,
                        past_key_values=past_key_values,
                          use_cache=self.use_cache,
                    )
                    
                    logits = outputs.logits[:, -1, :]
                    past_key_values = outputs.past_key_values
                    
                    next_token = sample_token(
                        logits,
                        generated_tokens=generated,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        repetition_penalty=repetition_penalty,
                    )
                    
                    generated = torch.cat([generated, next_token], dim=1)
                    current_part1_beat.append(next_token.item())
                    
                    # Check if beat is complete
                    if next_token.item() in [part1_end_marker, part1_empty_marker, self.BAR_TOKEN]:
                        part1_beats_generated.append(current_part1_beat.copy())
                        current_part1_beat = []
                        part1_idx += 1
                        position = 0  # Switch back to part0
        
        return part1_beats_generated


def beats_to_pianoroll(
    beat_tokens: List[List[int]],
    tokenizer: PianoRollTokenizer,
    timesteps_per_beat: int = 4,
) -> np.ndarray:
    """
    Convert beat tokens back to pianoroll.
    
    Uses PianoRollTokenizer's decode methods:
    1. decompress_tokens: compressed tokens -> token matrix
    2. patch_tokens_to_image: token matrix -> pianoroll
    
    Args:
        beat_tokens: List of beat token lists (each is compressed tokens for one beat)
        tokenizer: PianoRollTokenizer instance
        timesteps_per_beat: Number of timesteps per beat (default 4)
        
    Returns:
        Pianoroll array of shape (2, 88, T) where T = len(beat_tokens) * timesteps_per_beat
    """
    if not beat_tokens:
        return np.zeros((2, 88, 0), dtype=np.float32)
    
    beat_pianorolls = []
    
    for beat_idx, beat_compressed in enumerate(beat_tokens):
        if not beat_compressed:
            print(f"[beats_to_pianoroll] Warning: empty beat at index {beat_idx}, using empty pianoroll")
            empty_beat = np.zeros((2, 88, timesteps_per_beat), dtype=np.float32)
            beat_pianorolls.append(empty_beat)
            continue

        # Handle bar token (single 255)
        if len(beat_compressed) == 1 and beat_compressed[0] == 255:
            # Empty beat (just bar separator)
            empty_beat = np.zeros((2, 88, timesteps_per_beat), dtype=np.float32)
            beat_pianorolls.append(empty_beat)
            continue
        
        # Handle empty beat marker (169)
        if len(beat_compressed) == 1 and beat_compressed[0] == 169:
            empty_beat = np.zeros((2, 88, timesteps_per_beat), dtype=np.float32)
            beat_pianorolls.append(empty_beat)
            continue
        
        try:
            # Convert list to numpy array for tokenizer
            compressed_array = np.array(beat_compressed, dtype=np.int64)

            # Step 1: Decompress tokens
            # end_marker_id=171 for part1 (accompaniment)
            tokens_matrix = tokenizer.decompress_tokens(
                compressed_array,
                end_marker_id=171,
            )
            if tokens_matrix.size == 0:
                print(f"[beats_to_pianoroll] Warning: empty decompressed tokens at beat {beat_idx}")
                empty_beat = np.zeros((2, 88, timesteps_per_beat), dtype=np.float32)
                beat_pianorolls.append(empty_beat)
                continue

            # Step 2: Convert tokens to pianoroll image
            beat_pianoroll = tokenizer.patch_tokens_to_image(tokens_matrix)

            # Step 3: Ensure correct timestep dimension
            # beat_pianoroll shape should be (2, 88, timesteps_per_beat)
            if beat_pianoroll.shape[2] != timesteps_per_beat:
                beat_pianoroll = _adjust_timesteps(beat_pianoroll, timesteps_per_beat)

            beat_pianorolls.append(beat_pianoroll)
        except Exception as exc:
            print(f"[beats_to_pianoroll] Error at beat {beat_idx}: {exc}")
            empty_beat = np.zeros((2, 88, timesteps_per_beat), dtype=np.float32)
            beat_pianorolls.append(empty_beat)
    
    # Concatenate all beats along time dimension
    if beat_pianorolls:
        full_pianoroll = np.concatenate(beat_pianorolls, axis=2)
    else:
        full_pianoroll = np.zeros((2, 88, 0), dtype=np.float32)
    
    return full_pianoroll


def _adjust_timesteps(pianoroll: np.ndarray, target_timesteps: int) -> np.ndarray:
    """
    Adjust pianoroll timestep dimension to match target.
    
    Args:
        pianoroll: Input pianoroll of shape (2, 88, T)
        target_timesteps: Desired timestep dimension
        
    Returns:
        Adjusted pianoroll of shape (2, 88, target_timesteps)
    """
    current_timesteps = pianoroll.shape[2]
    
    if current_timesteps == target_timesteps:
        return pianoroll
    elif current_timesteps < target_timesteps:
        # Pad with zeros
        pad_width = ((0, 0), (0, 0), (0, target_timesteps - current_timesteps))
        return np.pad(pianoroll, pad_width, mode='constant', constant_values=0)
    else:
        # Truncate
        return pianoroll[:, :, :target_timesteps]
