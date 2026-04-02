from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

import numpy as np
import torch


EventPayload = Dict[str, int | str]
TimingPayload = Dict[str, float]

# PianoLLaMA tokenization is fixed: 4 timesteps per beat.
TIMESTEPS_PER_BEAT = 4


@dataclass(frozen=True)
class BackendRuntimeConfig:
    model_name: str
    inference_mode: str
    generation_interval_ticks: int
    generation_length_frames: int
    prompt_length_ticks: Optional[int]
    checkpoint_path: Optional[str]


class LekaiHttpBackend:
    """
    HTTP Backend for Lekai model inference.
    
    Supports two modes:
    1. Rule-based stub (when no checkpoint or model fails to load)
    2. Real PianoLLaMA model (when checkpoint is available)
    """
    
    def __init__(self, checkpoint_path: Optional[str] = None) -> None:
        self._melody_history: List[EventPayload] = []
        self._accompaniment_history: List[EventPayload] = []
        self._injection_length_ticks: int = 0
        self._runtime_config: Optional[BackendRuntimeConfig] = None
        
        # Model components (Phase 3: real model integration)
        self._model_adapter = None
        self._converter = None
        self._tokenizer = None
        self._active_pitches: Set[int] = set()  # For tracking sustained notes across calls
        
        # Try to load real model if checkpoint provided
        if checkpoint_path:
            self._load_model(checkpoint_path)
        
    def _load_model(self, checkpoint_path: str) -> None:
        """Load PianoLLaMA model if available."""
        try:
            from streammuse.infrastructure.inference.lekai_model.MidiConverter import MidiConverter
            from streammuse.infrastructure.inference.lekai_model.my_tokenizer import PianoRollTokenizer
            from streammuse.infrastructure.inference.lekai_model.inference_adapter import PianoLLaMAAdapter
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            if os.path.exists(checkpoint_path):
                self._model_adapter = PianoLLaMAAdapter.from_checkpoint(checkpoint_path, device=device)
                self._converter = MidiConverter(ticks_per_beat=4)
                self._tokenizer = PianoRollTokenizer(patch_h=1, patch_w=4)
                print(f"[LekaiHttpBackend] Loaded model from {checkpoint_path} (device: {device})")
            else:
                print(f"[LekaiHttpBackend] Checkpoint not found: {checkpoint_path}, using rule-based stub")
        except Exception as e:
            print(f"[LekaiHttpBackend] Failed to load model: {e}, using rule-based stub")
            self._model_adapter = None
    
    def _has_real_model(self) -> bool:
        """Check if real model is loaded and available."""
        return self._model_adapter is not None and self._converter is not None

    def configure(self, config: BackendRuntimeConfig) -> None:
        self._runtime_config = config
        # Try to load model if checkpoint_path provided and not already loaded
        if config.checkpoint_path and not self._has_real_model():
            self._load_model(config.checkpoint_path)

    def generate(
        self,
        melody_events: List[EventPayload],
        generation_start_tick: int,
        generation_length_frames: int,
        generation_interval_ticks: int,
        prompt_length_ticks: Optional[int],
        inference_mode: str,
        model_name: str,
        checkpoint_path: Optional[str],
    ) -> tuple[List[EventPayload], TimingPayload]:
        request_arrival = time.perf_counter()
        preprocess_start = request_arrival

        self.configure(
            BackendRuntimeConfig(
                model_name=model_name,
                inference_mode=inference_mode,
                generation_interval_ticks=int(generation_interval_ticks),
                generation_length_frames=int(generation_length_frames),
                prompt_length_ticks=prompt_length_ticks,
                checkpoint_path=checkpoint_path,
            )
        )

        # Bug #4 fix: extend 而非替换
        self._melody_history.extend(melody_events)

        inference_start = time.perf_counter()
        
        # Phase 3: Use real model if available, otherwise fallback to rule-based
        if self._has_real_model():
            accompaniment = self._generate_with_model(
                generation_start_tick=int(generation_start_tick),
                generation_interval_ticks=int(generation_interval_ticks),
                generation_length_frames=int(generation_length_frames),
            )
        else:
            # Bug #5 fix: 传入 generation_length_frames
            accompaniment = self._generate_rule_based(
                generation_start_tick=int(generation_start_tick),
                generation_interval_ticks=int(generation_interval_ticks),
                generation_length_frames=int(generation_length_frames),
            )
            
        inference_end = time.perf_counter()

        self._accompaniment_history.extend(accompaniment)
        
        # Bug #6 fix: 裁剪历史防止无限增长
        self._trim_histories(generation_start_tick, generation_length_frames)
        
        response_output = time.perf_counter()

        timings: TimingPayload = {
            "request_arrival_time": request_arrival,
            "response_output_time": response_output,
            "preprocess_start_time": preprocess_start,
            "inference_start_time": inference_start,
            "inference_end_time": inference_end,
            "postprocess_start_time": inference_end,
        }
        return accompaniment, timings

    def _generate_with_model(
        self,
        generation_start_tick: int,
        generation_interval_ticks: int,
        generation_length_frames: int,
    ) -> List[EventPayload]:
        """
        Phase 3: Generate accompaniment using real PianoLLaMA model.
        
        Pipeline:
        events -> pianoroll -> beat tokens -> model -> beat tokens -> pianoroll -> events
        """
        assert self._converter is not None and self._model_adapter is not None
        
        # 1. Determine prompt window
        prompt_start = max(0, generation_start_tick - generation_length_frames)
        
        # 2. Convert melody events to pianoroll
        melody_pianoroll = self._converter.events_to_pianoroll(
            events=self._melody_history,
            start_tick=prompt_start,
            end_tick=generation_start_tick,
            active_pitches=None,  # Don't carry over from previous calls for simplicity
        )
        
        # melody_pianoroll shape: (2, 88, T)
        
        # 3. Convert pianoroll to beat tokens (part0)
        # For simplicity, we treat the entire melody as part0 (high voice)
        # In real usage, you might want to separate tracks
        from streammuse.infrastructure.inference.lekai_model.PianoDataset import process_measure_with_beat_interleaving
        
        # Pad to measure boundary (4 beats = 16 timesteps with ticks_per_beat=4)
        T = melody_pianoroll.shape[2]
        measure_size = 16  # 4 beats * 4 ticks per beat
        if T % measure_size != 0:
            pad_size = measure_size - (T % measure_size)
            melody_pianoroll = np.pad(
                melody_pianoroll, 
                ((0, 0), (0, 0), (0, pad_size)), 
                mode='constant', 
                constant_values=0
            )
        
        num_measures = melody_pianoroll.shape[2] // measure_size
        
        # Build part0 beats from melody
        part0_beats = []
        for i in range(num_measures):
            measure = melody_pianoroll[:, :, i * measure_size:(i + 1) * measure_size]
            # Stack to create 4-channel (2 for part0 + 2 for part1, but part1 is empty)
            full_measure = np.zeros((4, 88, measure_size), dtype=np.float32)
            full_measure[:2] = measure  # Part0 = melody
            
            p0_beats, _ = process_measure_with_beat_interleaving(
                full_measure, 
                tokenizer=self._tokenizer,
                timesteps_per_beat=4
            )
            
            # Add bar token
            part0_beats.append(torch.tensor([self._model_adapter.BAR_TOKEN], dtype=torch.long))
            part0_beats.extend(p0_beats)
        
        # 4. Calculate number of beats from generation length only.
        # generation_interval_ticks is a client scheduling parameter and should not
        # affect server-side generation duration.
        _ = generation_interval_ticks
        num_beats_to_generate = max(1, generation_length_frames // TIMESTEPS_PER_BEAT)
        if generation_length_frames % TIMESTEPS_PER_BEAT != 0:
            num_beats_to_generate += 1
            print(
                f"[LekaiHttpBackend] Warning: generation_length_frames ({generation_length_frames}) "
                f"not divisible by {TIMESTEPS_PER_BEAT}, rounding up beats to {num_beats_to_generate}."
            )
        
        # 5. Generate part1 using model
        part1_beats = self._model_adapter.generate_from_beats(
            part0_beats=part0_beats,
            num_beats_to_generate=num_beats_to_generate,
            bpm=int(os.environ.get("LEKAI_DEFAULT_BPM", "120")),
            temperature=0.8,
            top_k=50,
            top_p=0.95,
            verbose=False,
        )
        
        # 6. Convert part1 beats to pianoroll
        # Use the correct beats_to_pianoroll function with tokenizer decode
        from streammuse.infrastructure.inference.lekai_model.inference_adapter import beats_to_pianoroll
        part1_pianoroll = beats_to_pianoroll(
            part1_beats,
            tokenizer=self._tokenizer,
            timesteps_per_beat=TIMESTEPS_PER_BEAT,
        )
        
        expected_timesteps = num_beats_to_generate * TIMESTEPS_PER_BEAT
        expected_shape = (2, 88, expected_timesteps)
        if part1_pianoroll.shape != expected_shape:
            raise RuntimeError(
                f"Pianoroll shape mismatch: expected {expected_shape}, "
                f"got {part1_pianoroll.shape}. "
                "This indicates a bug in beats_to_pianoroll."
            )
        
        # 7. Convert pianoroll to events
        events, self._active_pitches = self._converter.pianoroll_to_events(
            pianoroll=part1_pianoroll,
            start_tick=generation_start_tick,
            close_at_end=False,
            active_pitches=self._active_pitches,
        )
        
        # Convert events to EventPayload format
        accompaniment: List[EventPayload] = []
        for ev in events:
            payload: EventPayload = {
                "type": str(ev["type"]),
                "pitch": int(ev["pitch"]),
                "tick": int(ev["tick"]),
            }
            if "velocity" in ev:
                payload["velocity"] = int(ev["velocity"])
            accompaniment.append(payload)
        
        return accompaniment

    def inject_history(
        self,
        melody_events: List[EventPayload],
        accompaniment_events: List[EventPayload],
        injection_length_ticks: int,
    ) -> Dict[str, int | bool | str]:
        self._melody_history = list(melody_events)
        self._accompaniment_history = list(accompaniment_events)
        self._injection_length_ticks = int(injection_length_ticks)
        # Reset active pitches on injection
        self._active_pitches = set()

        return {
            "success": True,
            "message": "Notes injected successfully",
            "melody_notes_injected": len(melody_events),
            "accompaniment_notes_injected": len(accompaniment_events),
            "injection_length_ticks": int(injection_length_ticks),
        }

    def clear_history(self) -> Dict[str, bool | str]:
        self._melody_history = []
        self._accompaniment_history = []
        self._injection_length_ticks = 0
        self._active_pitches = set()
        return {"success": True, "message": "History cleared"}

    def injection_status(self) -> Dict[str, bool | int | str]:
        return {
            "is_injected": self._injection_length_ticks > 0,
            "injection_length_ticks": self._injection_length_ticks,
            "runtime_model_name": self._runtime_config.model_name if self._runtime_config else "",
            "runtime_inference_mode": self._runtime_config.inference_mode if self._runtime_config else "",
        }

    def _trim_histories(self, generation_start_tick: int, generation_length_frames: int) -> None:
        """Bug #6: 裁剪历史防止无限增长"""
        max_history_ticks = int(generation_length_frames) * 2  # 保留 2 倍生成长度的历史
        cutoff_tick = int(generation_start_tick) - max_history_ticks
        
        if cutoff_tick > 0:
            self._accompaniment_history = [
                e for e in self._accompaniment_history
                if int(e.get("tick", 0)) >= cutoff_tick
            ]
            self._melody_history = [
                e for e in self._melody_history
                if int(e.get("tick", 0)) >= cutoff_tick
            ]

    def _generate_rule_based(
        self, 
        generation_start_tick: int, 
        generation_interval_ticks: int,
        generation_length_frames: int,  # Bug #5: 新增参数
    ) -> List[EventPayload]:
        """Fallback rule-based generation when model is not available."""
        _ = generation_interval_ticks
        chord_duration_ticks = TIMESTEPS_PER_BEAT

        # Generation length is independent from client interval.
        num_intervals = max(1, int(generation_length_frames) // chord_duration_ticks)
        if int(generation_length_frames) % chord_duration_ticks != 0:
            num_intervals += 1

        window_start = generation_start_tick - chord_duration_ticks * 2
        recent_on_pitches: List[int] = []

        for event in self._melody_history:
            event_type = str(event.get("type", ""))
            tick = int(event.get("tick", 0))
            pitch = int(event.get("pitch", -1))
            if pitch < 0:
                continue
            if event_type == "note_on" and window_start <= tick <= generation_start_tick:
                recent_on_pitches.append(pitch)

        unique: List[int] = []
        seen: Set[int] = set()
        for pitch in reversed(recent_on_pitches):
            if pitch in seen:
                continue
            seen.add(pitch)
            unique.append(pitch)
            if len(unique) >= 4:
                break

        if not unique:
            return []

        accompaniment: List[EventPayload] = []
        
        # Bug #5: 生成多个 interval
        for i in range(num_intervals):
            offset = i * chord_duration_ticks
            chord_tick = int(generation_start_tick + offset)
            note_off_tick = chord_tick + chord_duration_ticks

            for pitch in unique:
                model_pitch = max(36, pitch - 12)
                accompaniment.append(
                    {
                        "type": "note_on",
                        "pitch": int(model_pitch),
                        "tick": chord_tick,
                        "velocity": 80,
                    }
                )
                # Bug #1 fix: note_off 也带 velocity
                accompaniment.append(
                    {
                        "type": "note_off",
                        "pitch": int(model_pitch),
                        "tick": note_off_tick,
                        "velocity": 0,  # MIDI 标准: note_off velocity = 0
                    }
                )

        accompaniment.sort(key=lambda e: (int(e["tick"]), 0 if str(e["type"]) == "note_off" else 1))
        return accompaniment
