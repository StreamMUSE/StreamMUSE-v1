from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

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

# Standard Lekai offline_model vocabulary. These values must match
# external/lekai_real_time/offline_model/my_tokenizer.py, because the
# continuation checkpoint was trained with that schedule/token layout.
LEKAI_EMPTY_TOKEN = 169
LEKAI_ACC_END_TOKEN = 170
LEKAI_MEL_END_TOKEN = 171
LEKAI_BEAT_TOKEN = 172
LEKAI_BAR_TOKEN = 255
LEKAI_EOS_TOKEN = 256
LEKAI_BOS_TOKEN = 257
LEKAI_PAD_TOKEN = 258
LEKAI_TIME_SIG_OFFSET = 259
LEKAI_BPM_OFFSET = 264


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

    def _measure_beats_from_time_signature_idx(self, time_signature_idx: int) -> int:
        override = self._env_positive_int("LEKAI_MEASURE_BEATS")
        if override is not None:
            return override

        # Dataset time_signature_idx is not the numerator itself for every case,
        # but these are the values observed in the Lekai offline NPZ metadata.
        # Unknown values fall back to common-time behavior.
        if time_signature_idx == 9:
            time_signature_idx = 4
        return {
            0: 4,  # 4/4
            2: 2,  # 2/4
            3: 3,  # 3/4
            4: 4,
            6: 6,
        }.get(time_signature_idx, 4)

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
            if device_preference.strip().lower() == "mps" and enable_mps_fallback:
                primary_device = "mps"
                primary_dtype = resolve_dtype(primary_device, dtype_preference)
            else:
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

    @staticmethod
    def _event_sort_key(event: EventPayload) -> Tuple[int, int]:
        tick = int(event.get("tick", 0))
        priority = 0 if str(event.get("type", "")) == "note_off" else 1
        return tick, priority

    def _active_pitches_before_tick(self, events: List[EventPayload], cutoff_tick: int) -> Set[int]:
        active: Set[int] = set()
        for event in sorted(events, key=self._event_sort_key):
            tick = int(event.get("tick", 0))
            if tick >= cutoff_tick:
                break
            if "pitch" not in event:
                continue
            pitch = int(event["pitch"])
            event_type = str(event.get("type", ""))
            if event_type == "note_on":
                active.add(pitch)
            elif event_type == "note_off":
                active.discard(pitch)
        return active

    def _advance_active_pitches(
        self,
        events: List[EventPayload],
        start_tick: int,
        end_tick: int,
        active_pitches: Set[int],
    ) -> Set[int]:
        updated = set(active_pitches)
        beat_events = [
            event
            for event in events
            if int(event.get("tick", -1)) >= start_tick and int(event.get("tick", -1)) < end_tick
        ]
        beat_events.sort(key=self._event_sort_key)

        for event in beat_events:
            if "pitch" not in event:
                continue
            pitch = int(event["pitch"])
            event_type = str(event.get("type", ""))
            if event_type == "note_on":
                updated.add(pitch)
            elif event_type == "note_off":
                updated.discard(pitch)

        return updated

    def _encode_beat_tokens(
        self,
        events: List[EventPayload],
        beat_start_tick: int,
        active_pitches: Set[int],
        end_marker: int,
    ) -> tuple[torch.Tensor, Set[int]]:
        assert self._converter is not None and self._tokenizer is not None

        beat_end_tick = beat_start_tick + TIMESTEPS_PER_BEAT
        beat_pr = self._converter.events_to_pianoroll(
            events=events,
            start_tick=beat_start_tick,
            end_tick=beat_end_tick,
            active_pitches=active_pitches,
        )

        tokens_matrix = self._tokenizer.image_to_patch_tokens(beat_pr, strict_mode=True)
        compressed = self._tokenizer.compress_tokens(tokens_matrix, end_marker=end_marker)
        if len(compressed) == 1 and int(compressed[0]) == LEKAI_EMPTY_TOKEN:
            compressed = np.array([LEKAI_EMPTY_TOKEN, int(end_marker)], dtype=np.int64)

        next_active = self._advance_active_pitches(
            events=events,
            start_tick=beat_start_tick,
            end_tick=beat_end_tick,
            active_pitches=active_pitches,
        )
        return torch.tensor(compressed, dtype=torch.long), next_active

    def _generate_part1_tokens_from_prompt(
        self,
        prompt_tokens: torch.Tensor,
        *,
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty: float,
    ) -> List[int]:
        assert self._model_adapter is not None
        from streammuse.infrastructure.inference.lekai_model.generation_utils import sample_token

        model = getattr(self._model_adapter, "model", None)
        if model is None:
            raise RuntimeError("Model adapter does not expose `model` for interleaved decoding.")

        device = str(getattr(self._model_adapter, "device", "cpu"))
        use_cache = bool(getattr(self._model_adapter, "use_cache", True))
        pad_token_id = LEKAI_PAD_TOKEN
        part1_end_marker = LEKAI_ACC_END_TOKEN
        bar_token = LEKAI_BAR_TOKEN
        beat_token = LEKAI_BEAT_TOKEN

        generated = prompt_tokens.unsqueeze(0).to(device)
        raw_tokens: List[int] = []
        past_key_values = None

        with torch.no_grad():
            for _ in range(100):
                outputs = model(
                    input_ids=generated[:, -1:] if past_key_values is not None else generated,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
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

                token_val = int(next_token.item())
                generated = torch.cat([generated, next_token], dim=1)
                raw_tokens.append(token_val)

                if token_val in {part1_end_marker, bar_token, beat_token}:
                    break

        valid_tokens = [token for token in raw_tokens if token != pad_token_id]
        if valid_tokens and valid_tokens[-1] in {bar_token, beat_token}:
            valid_tokens.pop()
        if not valid_tokens:
            return [LEKAI_EMPTY_TOKEN]
        return valid_tokens

    def _decode_acc_beat_tokens(self, beat_tokens: List[int]) -> np.ndarray:
        assert self._tokenizer is not None

        if not beat_tokens:
            return np.zeros((2, 88, TIMESTEPS_PER_BEAT), dtype=np.float32)
        if len(beat_tokens) == 1 and beat_tokens[0] in {LEKAI_EMPTY_TOKEN, LEKAI_BAR_TOKEN, LEKAI_BEAT_TOKEN}:
            return np.zeros((2, 88, TIMESTEPS_PER_BEAT), dtype=np.float32)
        if beat_tokens == [LEKAI_EMPTY_TOKEN, LEKAI_ACC_END_TOKEN]:
            return np.zeros((2, 88, TIMESTEPS_PER_BEAT), dtype=np.float32)

        compressed = np.array(beat_tokens, dtype=np.int64)
        tokens_matrix = self._tokenizer.decompress_tokens(
            compressed,
            end_marker_id=LEKAI_ACC_END_TOKEN,
        )
        if tokens_matrix.size == 0:
            return np.zeros((2, 88, TIMESTEPS_PER_BEAT), dtype=np.float32)

        pianoroll = self._tokenizer.patch_tokens_to_image(tokens_matrix)
        if pianoroll.shape[2] < TIMESTEPS_PER_BEAT:
            pianoroll = np.pad(
                pianoroll,
                ((0, 0), (0, 0), (0, TIMESTEPS_PER_BEAT - pianoroll.shape[2])),
                mode="constant",
            )
        elif pianoroll.shape[2] > TIMESTEPS_PER_BEAT:
            pianoroll = pianoroll[:, :, :TIMESTEPS_PER_BEAT]
        return pianoroll

    def _generate_with_interleaved_prompt(
        self,
        generation_start_tick: int,
        generation_interval_ticks: int,
        generation_length_frames: int,
    ) -> List[EventPayload]:
        assert self._converter is not None and self._model_adapter is not None and self._tokenizer is not None

        _ = generation_interval_ticks

        from streammuse.infrastructure.inference.lekai_model.PianoDataset import encode_bpm

        current_beat = int(generation_start_tick) // TIMESTEPS_PER_BEAT
        num_beats_to_generate = max(1, int(generation_length_frames) // TIMESTEPS_PER_BEAT)
        if int(generation_length_frames) % TIMESTEPS_PER_BEAT != 0:
            num_beats_to_generate += 1

        context_beats = self._env_positive_int("LEKAI_PROMPT_CONTEXT_BEATS")
        start_beat = max(0, current_beat - context_beats) if context_beats is not None else 0
        context_start_tick = start_beat * TIMESTEPS_PER_BEAT

        bpm_token = encode_bpm(int(os.environ.get("LEKAI_DEFAULT_BPM", "120"))) + LEKAI_BPM_OFFSET
        time_signature_idx = int(os.environ.get("LEKAI_TIME_SIGNATURE_INDEX", "4"))
        measure_beats = self._measure_beats_from_time_signature_idx(time_signature_idx)
        if time_signature_idx == 9:
            time_signature_idx = 4
        time_sig_token = time_signature_idx + LEKAI_TIME_SIG_OFFSET

        prefix: List[torch.Tensor] = [
            torch.tensor([LEKAI_BOS_TOKEN], dtype=torch.long),
            torch.tensor([time_sig_token], dtype=torch.long),
            torch.tensor([bpm_token], dtype=torch.long),
        ]

        accompaniment_context_events: List[EventPayload] = list(self._accompaniment_history)

        generated_events: List[EventPayload] = []

        # Match offline defaults by default; keep runtime override via env vars.
        try:
            rt_temperature = float(os.environ.get("LEKAI_RT_TEMPERATURE", "1.1"))
        except Exception:
            rt_temperature = 1.1
        try:
            rt_top_k = int(os.environ.get("LEKAI_RT_TOP_K", "0"))
        except Exception:
            rt_top_k = 0
        try:
            rt_top_p = float(os.environ.get("LEKAI_RT_TOP_P", "0.95"))
        except Exception:
            rt_top_p = 0.95
        try:
            rt_repetition_penalty = float(os.environ.get("LEKAI_RT_REPETITION_PENALTY", "1.0"))
        except Exception:
            rt_repetition_penalty = 1.0

        def build_standard_offline_prompt(target_beat: int) -> torch.Tensor:
            """Build the reference offline continuation prompt before target acc.

            Reference schedule:
              [BOS, TS, BPM] -> bar -> beat -> acc/generated-or-injected -> mel
            The returned tensor ends immediately after the target beat marker,
            so the next autoregressive tokens are the target accompaniment beat.
            """

            seq: List[torch.Tensor] = list(prefix)

            melody_active = self._active_pitches_before_tick(self._melody_history, context_start_tick)
            accompaniment_active = self._active_pitches_before_tick(
                accompaniment_context_events,
                context_start_tick,
            )

            for beat in range(start_beat, target_beat):
                if beat == start_beat or beat % measure_beats == 0:
                    seq.append(torch.tensor([LEKAI_BAR_TOKEN], dtype=torch.long))
                seq.append(torch.tensor([LEKAI_BEAT_TOKEN], dtype=torch.long))

                beat_start_tick = beat * TIMESTEPS_PER_BEAT
                acc_tokens, accompaniment_active = self._encode_beat_tokens(
                    events=accompaniment_context_events,
                    beat_start_tick=beat_start_tick,
                    active_pitches=accompaniment_active,
                    end_marker=LEKAI_ACC_END_TOKEN,
                )
                seq.append(acc_tokens)

                mel_tokens, melody_active = self._encode_beat_tokens(
                    events=self._melody_history,
                    beat_start_tick=beat_start_tick,
                    active_pitches=melody_active,
                    end_marker=LEKAI_MEL_END_TOKEN,
                )
                seq.append(mel_tokens)

            if target_beat == start_beat or target_beat % measure_beats == 0:
                seq.append(torch.tensor([LEKAI_BAR_TOKEN], dtype=torch.long))
            seq.append(torch.tensor([LEKAI_BEAT_TOKEN], dtype=torch.long))

            return torch.cat(seq, dim=0)

        for beat_offset in range(num_beats_to_generate):
            target_beat = current_beat + beat_offset
            beat_start_tick = target_beat * TIMESTEPS_PER_BEAT

            prompt_tokens = build_standard_offline_prompt(target_beat)
            generated_beat_tokens = self._generate_part1_tokens_from_prompt(
                prompt_tokens,
                temperature=rt_temperature,
                top_k=rt_top_k,
                top_p=rt_top_p,
                repetition_penalty=rt_repetition_penalty,
            )

            beat_pianoroll = self._decode_acc_beat_tokens(generated_beat_tokens)

            expected_shape = (2, 88, TIMESTEPS_PER_BEAT)
            got_shape = tuple(int(dim) for dim in beat_pianoroll.shape)
            if got_shape != expected_shape:
                got_time_axis = got_shape[2] if len(got_shape) >= 3 else -1
                print(
                    "[LekaiHttpBackend] Pianoroll shape mismatch "
                    f"(expected={expected_shape}, got={got_shape}, "
                    f"start_tick={generation_start_tick}, gen_len={generation_length_frames})"
                )
                if got_time_axis == 0 and TIMESTEPS_PER_BEAT > 0:
                    print("[LekaiHttpBackend] Recoverable mismatch detected; fallback to rule-based generation")
                    return self._generate_rule_based(
                        generation_start_tick=generation_start_tick,
                        generation_interval_ticks=generation_interval_ticks,
                        generation_length_frames=generation_length_frames,
                    )
                raise RuntimeError(
                    f"Pianoroll shape mismatch: expected {expected_shape}, got {got_shape}."
                )

            active_snapshot = self._active_pitches_before_tick(
                accompaniment_context_events,
                beat_start_tick,
            )
            beat_events, next_active = self._converter.pianoroll_to_events(
                pianoroll=beat_pianoroll,
                start_tick=beat_start_tick,
                close_at_end=False,
                active_pitches=active_snapshot,
            )

            self._active_pitches = set(next_active)

            normalized_beat_events: List[EventPayload] = []
            for event in beat_events:
                payload: EventPayload = {
                    "type": str(event["type"]),
                    "pitch": int(event["pitch"]),
                    "tick": int(event["tick"]),
                }
                if "velocity" in event:
                    payload["velocity"] = int(event["velocity"])
                normalized_beat_events.append(payload)

            generated_events.extend(normalized_beat_events)
            accompaniment_context_events.extend(normalized_beat_events)

        return generated_events

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
        assert self._converter is not None and self._model_adapter is not None

        if int(generation_start_tick) <= 0:
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

        has_interleaved_runtime = (
            hasattr(self._model_adapter, "model")
            and hasattr(self._model_adapter, "tokenizer")
            and self._tokenizer is not None
        )

        if has_interleaved_runtime:
            return self._generate_with_interleaved_prompt(
                generation_start_tick=generation_start_tick,
                generation_interval_ticks=generation_interval_ticks,
                generation_length_frames=generation_length_frames,
            )

        # Compatibility path for unit tests with lightweight adapter stubs.
        return self._generate_with_model_simple(
            generation_start_tick=generation_start_tick,
            generation_interval_ticks=generation_interval_ticks,
            generation_length_frames=generation_length_frames,
        )

    def _generate_with_model_simple(
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
        """Trim histories with a stable window independent from tiny generation lengths."""
        configured_max_ticks = self._env_positive_int("LEKAI_HISTORY_MAX_TICKS")
        if configured_max_ticks is None:
            configured_max_ticks = max(512, int(generation_length_frames) * 16)

        max_history_ticks = configured_max_ticks
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
