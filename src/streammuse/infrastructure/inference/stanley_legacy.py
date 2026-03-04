"""Legacy Stanley inference engine implementation (duration-note based).

This is restored/contained under infrastructure so application/domain do not
depend on model code.
"""

from __future__ import annotations

import os
import time
from typing import Optional

import numpy as np
import torch

from streammuse.infrastructure.inference.stanley_stack.m2a_transformer import (
    EOS_TOKEN,
    PAD_TOKEN,
    RoFormerSymbolicTransformer,
)
from streammuse.infrastructure.inference.stanley_stack.preprocess.preprocess_midi2pt_dataset import (
    DURATION_TEMPLATES,
)


class LegacyInferenceEngineStanley:
    def __init__(
        self,
        *,
        checkpoint_path: str,
        model_size: str,
        generation_length_frames: int,
        max_polyphony: int = 4,
        model_max_seq_len_frames: int = 96,
    ) -> None:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        self.model = RoFormerSymbolicTransformer.load_from_checkpoint(
            checkpoint_path,
            model_size=model_size,
            map_location=lambda storage, loc: storage.cuda(0) if torch.cuda.is_available() else "cpu",
        )
        if torch.cuda.is_available():
            self.model.to("cuda:0")
        self.model.eval()

        self.model_max_seq_len_frames = int(model_max_seq_len_frames)
        self.prompt_length_ticks = self.model_max_seq_len_frames // 2
        self.generation_length_frames = int(generation_length_frames)
        self.max_polyphony = int(max_polyphony)

        self.melody_history: list[dict] = []
        self.accompaniment_history: list[dict] = []
        self.injection_offset_ticks = 0

    def set_injection_offset(self, offset_ticks: int) -> None:
        self.injection_offset_ticks = int(offset_ticks)

    def clear_history(self) -> None:
        self.melody_history = []
        self.accompaniment_history = []

    def _notes_to_rolls(self, notes: list, max_tick: int, max_polyphony: int, program: int) -> torch.Tensor:
        duration_boundaries = (DURATION_TEMPLATES[1:] + DURATION_TEMPLATES[:-1]) / 2.0
        rolls = np.full((max_tick, max_polyphony, 3), dtype=np.uint8, fill_value=255)
        polyphony_counts = np.zeros(max_tick, dtype=np.uint8)

        for note in notes:
            tick = int(note["tick"])
            if tick >= max_tick:
                continue
            if polyphony_counts[tick] >= max_polyphony:
                continue
            duration_idx = int(np.searchsorted(duration_boundaries, int(note["duration"])))
            slot = int(polyphony_counts[tick])
            rolls[tick, slot, 0] = int(program)
            rolls[tick, slot, 1] = int(note["pitch"])
            rolls[tick, slot, 2] = int(duration_idx)
            polyphony_counts[tick] += 1

        for i in range(max_tick):
            if polyphony_counts[i] > 1:
                sorted_indices = np.argsort(rolls[i, : polyphony_counts[i], 1])
                rolls[i, : polyphony_counts[i]] = rolls[i, : polyphony_counts[i]][sorted_indices]
            if polyphony_counts[i] < max_polyphony:
                rolls[i, polyphony_counts[i], 0] = 254

        return torch.tensor(rolls.reshape(max_tick, -1))

    def _tensors_to_notes(self, output_tensors: list, generation_start_tick: int) -> list[list[dict]]:
        notes_per_sample: list[list[dict]] = []
        num_timesteps = len(output_tensors)
        if num_timesteps == 0:
            return []

        n_samples = int(output_tensors[0].shape[0])
        for i in range(n_samples):
            sample_notes: list[dict] = []
            for time_step in range(int(generation_start_tick) * 2, num_timesteps):
                data = output_tensors[time_step][i]
                content = data.flatten()
                tick = time_step // 2

                for j in range(0, len(content), 2):
                    program = int(content[j].item())
                    if program == EOS_TOKEN or program == PAD_TOKEN:
                        continue
                    program = 1 if (time_step % 2 == 0) else 0
                    if j + 1 >= len(content):
                        break

                    pitch_duration = int(content[j + 1].item()) - 2
                    if pitch_duration < 0:
                        continue
                    pitch = int(pitch_duration % 128)
                    duration_idx = int(pitch_duration // 128)
                    if not (0 <= duration_idx < len(DURATION_TEMPLATES)):
                        continue
                    duration = int(DURATION_TEMPLATES[duration_idx])

                    sample_notes.append({"tick": int(tick), "pitch": pitch, "duration": duration, "program": program})
            notes_per_sample.append(sample_notes)
        return notes_per_sample

    def generate_accompaniment(
        self,
        melody_notes: list[dict],
        generation_start_tick: int,
        acc_notes: Optional[list[dict]] = None,
        generation_length_frames: Optional[int] = None,
        prompt_length_ticks: Optional[int] = None,
    ) -> tuple[list[dict], float, float, float, float]:
        effective_generation_length = int(generation_length_frames or self.generation_length_frames)
        effective_prompt_length = int(prompt_length_ticks or self.prompt_length_ticks)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        preprocess_start_time = time.perf_counter()

        # Apply injection offset to internal history.
        absolute_melody = []
        for n in melody_notes:
            nn = dict(n)
            nn["tick"] = int(nn["tick"]) + int(self.injection_offset_ticks)
            absolute_melody.append(nn)
        self.melody_history.extend(absolute_melody)
        absolute_generation_start = int(generation_start_tick) + int(self.injection_offset_ticks)

        if len(self.accompaniment_history) == 0 and acc_notes is not None:
            absolute_acc = []
            for n in acc_notes:
                nn = dict(n)
                nn["tick"] = int(nn["tick"]) + int(self.injection_offset_ticks)
                absolute_acc.append(nn)
            self.accompaniment_history.extend(absolute_acc)

        prompt_end = absolute_generation_start
        prompt_start = max(0, prompt_end - effective_prompt_length)
        actual_prompt_len = prompt_end - prompt_start
        if actual_prompt_len <= 0:
            raise ValueError("Prompt length must be > 0")

        prompt_mel = [n for n in self.melody_history if prompt_start <= int(n["tick"]) < prompt_end]
        prompt_acc = [n for n in self.accompaniment_history if prompt_start <= int(n["tick"]) < prompt_end]

        rel_mel = [dict(n, tick=int(n["tick"]) - prompt_start) for n in prompt_mel]
        rel_acc = [dict(n, tick=int(n["tick"]) - prompt_start) for n in prompt_acc]

        x_mel_raw = self._notes_to_rolls(rel_mel, actual_prompt_len, self.max_polyphony, program=0)
        x_acc_raw = self._notes_to_rolls(rel_acc, actual_prompt_len, self.max_polyphony, program=1)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        x_mel_raw = x_mel_raw.unsqueeze(0).to(device)
        x_acc_raw = x_acc_raw.unsqueeze(0).to(device)

        x_mel, x_acc = self.model.preprocess(
            x_mel_raw.long(),
            pitch_shift=torch.zeros(1, dtype=torch.int8).to(device),
            y=x_acc_raw.long(),
        )

        batch_size, seq_len, subseq_len = x_mel.shape
        stacked = torch.stack([x_acc, x_mel], dim=2)
        x = stacked.view(batch_size, seq_len * 2, subseq_len)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        inference_start_time = time.perf_counter()

        with torch.no_grad():
            output_tensors = self.model.global_sampling(
                x,
                x_mel_gt=None,
                temperature=1,
                max_seq_len=effective_generation_length,
            )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        inference_end_time = time.perf_counter()
        postprocess_start_time = time.perf_counter()

        rel_generation_start = absolute_generation_start - prompt_start
        notes_output = self._tensors_to_notes(output_tensors, generation_start_tick=rel_generation_start)
        notes_output = notes_output[0] if notes_output else []

        absolute_generated = []
        for n in notes_output:
            abs_tick = int(n["tick"]) + int(prompt_start)
            if abs_tick >= absolute_generation_start and int(n.get("program", 1)) == 1:
                nn = dict(n, tick=abs_tick)
                absolute_generated.append(nn)

        new_ticks = set(int(n["tick"]) for n in absolute_generated)
        self.accompaniment_history = [n for n in self.accompaniment_history if int(n["tick"]) not in new_ticks]
        self.accompaniment_history.extend(absolute_generated)

        self.melody_history = [n for n in self.melody_history if int(n["tick"]) >= prompt_start]
        self.accompaniment_history = [n for n in self.accompaniment_history if int(n["tick"]) >= prompt_start]

        relative_generated = []
        for n in absolute_generated:
            nn = dict(n)
            nn["tick"] = int(nn["tick"]) - int(self.injection_offset_ticks)
            relative_generated.append(nn)

        return relative_generated, preprocess_start_time, inference_start_time, inference_end_time, postprocess_start_time

