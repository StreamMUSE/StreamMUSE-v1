from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np
import torch

from streammuse.infrastructure.inference.runtime_device import (
    dtype_to_name,
    parse_env_bool,
    resolve_device,
    resolve_dtype,
)


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


@dataclass
class BackendRuntimeStatus:
    mode: str = "rule_stub"
    resolved_device: str = "cpu"
    resolved_dtype: str = "float32"
    checkpoint_path: Optional[str] = None
    checkpoint_format: Optional[str] = None
    fallback_reason: Optional[str] = None
    load_time_ms: Optional[float] = None
    warmup_time_ms: Optional[float] = None
    use_cache: bool = True


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
        self._runtime_status = BackendRuntimeStatus()
        
        # Model components (Phase 3: real model integration)
        self._model_adapter = None
        self._converter = None
        self._tokenizer = None
        self._active_pitches: Set[int] = set()  # For tracking sustained notes across calls
        
        # Try to load real model if checkpoint provided
        if checkpoint_path:
            self._load_model(checkpoint_path)

    @staticmethod
    def _checkpoint_format(path: str) -> str:
        suffix = Path(path).suffix.lower()
        if suffix == ".safetensors":
            return "safetensors"
        if suffix in {".pt", ".pth", ".ckpt"}:
            return suffix[1:]
        return "unknown"

    @staticmethod
    def _env_positive_int(name: str) -> Optional[int]:
        raw = os.environ.get(name)
        if raw is None or raw.strip() == "":
            return None
        value = int(raw)
        return value if value > 0 else None

    def _runtime_bool(self, name: str, default: bool) -> bool:
        return parse_env_bool(os.environ.get(name), default=default)

    def _set_stub_mode(self, *, checkpoint_path: Optional[str], fallback_reason: Optional[str]) -> None:
        self._model_adapter = None
        self._converter = None
        self._tokenizer = None
        self._runtime_status.mode = "rule_stub"
        self._runtime_status.resolved_device = "cpu"
        self._runtime_status.resolved_dtype = "float32"
        self._runtime_status.checkpoint_path = checkpoint_path
        self._runtime_status.checkpoint_format = (
            self._checkpoint_format(checkpoint_path) if checkpoint_path else None
        )
        self._runtime_status.fallback_reason = fallback_reason
        self._runtime_status.load_time_ms = None
        self._runtime_status.warmup_time_ms = None

    def _warmup_model(self, warmup_steps: int) -> float:
        assert self._model_adapter is not None
        warmup_start = time.perf_counter()
        part0 = [torch.tensor([self._model_adapter.BAR_TOKEN], dtype=torch.long)]
        _ = self._model_adapter.generate_from_beats(
            part0_beats=part0,
            num_beats_to_generate=max(1, int(warmup_steps)),
            bpm=int(os.environ.get("LEKAI_DEFAULT_BPM", "120")),
            temperature=0.8,
            top_k=20,
            top_p=0.9,
            verbose=False,
        )
        return (time.perf_counter() - warmup_start) * 1000

    def _load_model_once(
        self,
        checkpoint_path: str,
        *,
        device: str,
        dtype: torch.dtype,
        use_cache: bool,
        warmup_steps: int,
    ) -> tuple[float, float]:
        from streammuse.infrastructure.inference.lekai_model.MidiConverter import MidiConverter
        from streammuse.infrastructure.inference.lekai_model.inference_adapter import PianoLLaMAAdapter
        from streammuse.infrastructure.inference.lekai_model.my_tokenizer import PianoRollTokenizer

        load_start = time.perf_counter()
        self._model_adapter = PianoLLaMAAdapter.from_checkpoint(
            checkpoint_path,
            device=device,
            dtype=dtype,
            use_cache=use_cache,
        )
        self._converter = MidiConverter(ticks_per_beat=4)
        self._tokenizer = PianoRollTokenizer(patch_h=1, patch_w=4)
        load_time_ms = (time.perf_counter() - load_start) * 1000
        warmup_time_ms = self._warmup_model(warmup_steps)
        return load_time_ms, warmup_time_ms

    def _load_model(self, checkpoint_path: str) -> None:
        """Load PianoLLaMA model if available."""
        self._runtime_status.use_cache = self._runtime_bool("LEKAI_USE_CACHE", True)
        self._runtime_status.checkpoint_path = checkpoint_path
        self._runtime_status.checkpoint_format = self._checkpoint_format(checkpoint_path)
        self._runtime_status.fallback_reason = None

        if not os.path.exists(checkpoint_path):
            reason = f"checkpoint_not_found:{checkpoint_path}"
            print(f"[LekaiHttpBackend] Checkpoint not found: {checkpoint_path}, using rule-based stub")
            self._set_stub_mode(checkpoint_path=checkpoint_path, fallback_reason=reason)
            return

        device_preference = os.environ.get("LEKAI_DEVICE", "auto")
        dtype_preference = os.environ.get("LEKAI_DTYPE", "auto")
        warmup_steps = self._env_positive_int("LEKAI_WARMUP_STEPS") or 1
        enable_mps_fallback = self._runtime_bool("LEKAI_ENABLE_MPS_FALLBACK", True)

        try:
            primary_device = resolve_device(device_preference)
            primary_dtype = resolve_dtype(primary_device, dtype_preference)
        except Exception as exc:
            reason = f"invalid_runtime_preference:{exc}"
            print(f"[LekaiHttpBackend] {reason}, using rule-based stub")
            self._set_stub_mode(checkpoint_path=checkpoint_path, fallback_reason=reason)
            return

        try:
            load_time_ms, warmup_time_ms = self._load_model_once(
                checkpoint_path,
                device=primary_device,
                dtype=primary_dtype,
                use_cache=self._runtime_status.use_cache,
                warmup_steps=warmup_steps,
            )
            self._runtime_status.mode = "real_model"
            self._runtime_status.resolved_device = primary_device
            self._runtime_status.resolved_dtype = dtype_to_name(primary_dtype)
            self._runtime_status.load_time_ms = load_time_ms
            self._runtime_status.warmup_time_ms = warmup_time_ms
            print(
                f"[LekaiHttpBackend] Loaded model from {checkpoint_path} "
                f"(device: {primary_device}, dtype: {dtype_to_name(primary_dtype)}, "
                f"load_ms: {load_time_ms:.1f}, warmup_ms: {warmup_time_ms:.1f})"
            )
            return
        except Exception as primary_error:
            if primary_device == "mps" and enable_mps_fallback:
                fallback_reason = f"mps_load_failed:{primary_error}"
                print(f"[LekaiHttpBackend] {fallback_reason}, fallback to CPU")
                try:
                    fallback_dtype = resolve_dtype("cpu", "auto")
                    load_time_ms, warmup_time_ms = self._load_model_once(
                        checkpoint_path,
                        device="cpu",
                        dtype=fallback_dtype,
                        use_cache=self._runtime_status.use_cache,
                        warmup_steps=warmup_steps,
                    )
                    self._runtime_status.mode = "real_model"
                    self._runtime_status.resolved_device = "cpu"
                    self._runtime_status.resolved_dtype = dtype_to_name(fallback_dtype)
                    self._runtime_status.load_time_ms = load_time_ms
                    self._runtime_status.warmup_time_ms = warmup_time_ms
                    self._runtime_status.fallback_reason = fallback_reason
                    print(
                        f"[LekaiHttpBackend] CPU fallback succeeded "
                        f"(load_ms: {load_time_ms:.1f}, warmup_ms: {warmup_time_ms:.1f})"
                    )
                    return
                except Exception as fallback_error:
                    reason = f"cpu_fallback_failed:{fallback_error}"
                    print(f"[LekaiHttpBackend] {reason}, using rule-based stub")
                    self._set_stub_mode(checkpoint_path=checkpoint_path, fallback_reason=reason)
                    return

            reason = f"model_load_failed:{primary_error}"
            print(f"[LekaiHttpBackend] {reason}, using rule-based stub")
            self._set_stub_mode(checkpoint_path=checkpoint_path, fallback_reason=reason)
    
    def _has_real_model(self) -> bool:
        """Check if real model is loaded and available."""
        return self._model_adapter is not None and self._converter is not None

    def configure(self, config: BackendRuntimeConfig) -> None:
        self._runtime_config = config
        # Try to load model if checkpoint_path provided and not already loaded
        if config.checkpoint_path and not self._has_real_model():
            self._load_model(config.checkpoint_path)

    def _apply_runtime_limits(
        self,
        generation_length_frames: int,
        prompt_length_ticks: Optional[int],
    ) -> tuple[int, Optional[int]]:
        adjusted_length = int(generation_length_frames)
        adjusted_prompt = int(prompt_length_ticks) if prompt_length_ticks is not None else None

        max_generation = self._env_positive_int("LEKAI_MAX_GENERATION_LENGTH_FRAMES")
        if max_generation is not None and adjusted_length > max_generation:
            print(
                f"[LekaiHttpBackend] Cap generation_length_frames from {adjusted_length} to {max_generation} "
                "by LEKAI_MAX_GENERATION_LENGTH_FRAMES"
            )
            adjusted_length = max_generation

        max_prompt = self._env_positive_int("LEKAI_MAX_PROMPT_TICKS")
        if adjusted_prompt is not None and max_prompt is not None and adjusted_prompt > max_prompt:
            print(
                f"[LekaiHttpBackend] Cap prompt_length_ticks from {adjusted_prompt} to {max_prompt} "
                "by LEKAI_MAX_PROMPT_TICKS"
            )
            adjusted_prompt = max_prompt

        return adjusted_length, adjusted_prompt

    def runtime_info(self) -> Dict[str, str | float | bool | None]:
        return {
            "mode": "real_model" if self._has_real_model() else "rule_stub",
            "has_real_model": self._has_real_model(),
            "resolved_device": self._runtime_status.resolved_device,
            "resolved_dtype": self._runtime_status.resolved_dtype,
            "checkpoint_path": self._runtime_status.checkpoint_path,
            "checkpoint_format": self._runtime_status.checkpoint_format,
            "fallback_reason": self._runtime_status.fallback_reason,
            "load_time_ms": self._runtime_status.load_time_ms,
            "warmup_time_ms": self._runtime_status.warmup_time_ms,
            "use_cache": self._runtime_status.use_cache,
            "runtime_model_name": self._runtime_config.model_name if self._runtime_config else "",
            "runtime_inference_mode": self._runtime_config.inference_mode if self._runtime_config else "",
        }

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

        effective_generation_length_frames, effective_prompt_length_ticks = self._apply_runtime_limits(
            generation_length_frames=int(generation_length_frames),
            prompt_length_ticks=prompt_length_ticks,
        )

        self.configure(
            BackendRuntimeConfig(
                model_name=model_name,
                inference_mode=inference_mode,
                generation_interval_ticks=int(generation_interval_ticks),
                generation_length_frames=effective_generation_length_frames,
                prompt_length_ticks=effective_prompt_length_ticks,
                checkpoint_path=checkpoint_path,
            )
        )

        # Keep full melody history on the server; requests carry increments.
        self._melody_history.extend(melody_events)

        inference_start = time.perf_counter()
        
        # Phase 3: Use real model if available, otherwise fallback to rule-based
        if self._has_real_model():
            accompaniment = self._generate_with_model(
                generation_start_tick=int(generation_start_tick),
                generation_interval_ticks=int(generation_interval_ticks),
                generation_length_frames=effective_generation_length_frames,
            )
        else:
            # Rule-based fallback path.
            accompaniment = self._generate_rule_based(
                generation_start_tick=int(generation_start_tick),
                generation_interval_ticks=int(generation_interval_ticks),
                generation_length_frames=effective_generation_length_frames,
            )
            
        inference_end = time.perf_counter()

        self._accompaniment_history.extend(accompaniment)
        
        # Trim server-side histories to bound memory usage.
        self._trim_histories(generation_start_tick, effective_generation_length_frames)
        
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

        prompt_start = max(0, generation_start_tick - generation_length_frames)
        prompt_end = int(generation_start_tick)
        if prompt_end <= prompt_start:
            print(
                "[LekaiHttpBackend] zero prompt window "
                f"(start_tick={generation_start_tick}, gen_len={generation_length_frames}); "
                "fallback to rule-based generation"
            )
            return self._generate_rule_based(
                generation_start_tick=generation_start_tick,
                generation_interval_ticks=generation_interval_ticks,
                generation_length_frames=generation_length_frames,
            )
        
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
        got_shape = tuple(int(dim) for dim in part1_pianoroll.shape)
        if got_shape != expected_shape:
            got_time_axis = got_shape[2] if len(got_shape) >= 3 else -1
            print(
                "[LekaiHttpBackend] Pianoroll shape mismatch "
                f"(expected={expected_shape}, got={got_shape}, "
                f"start_tick={generation_start_tick}, gen_len={generation_length_frames})"
            )
            if got_time_axis == 0 and expected_timesteps > 0:
                print("[LekaiHttpBackend] Recoverable mismatch detected; fallback to rule-based generation")
                return self._generate_rule_based(
                    generation_start_tick=generation_start_tick,
                    generation_interval_ticks=generation_interval_ticks,
                    generation_length_frames=generation_length_frames,
                )
            raise RuntimeError(
                f"Pianoroll shape mismatch: expected {expected_shape}, "
                f"got {got_shape}. "
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

    def clear_history(self) -> Dict[str, Any]:
        melody_history = [dict(event) for event in self._melody_history]
        accompaniment_history = [dict(event) for event in self._accompaniment_history]

        self._melody_history = []
        self._accompaniment_history = []
        self._injection_length_ticks = 0
        self._active_pitches = set()
        return {
            "success": True,
            "message": "History cleared",
            "melody_history": melody_history,
            "accompaniment_history": accompaniment_history,
        }

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
