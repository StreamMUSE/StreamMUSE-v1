"""Prompt accompaniment engine for Lekai prompt-continuation.

This class is intentionally narrow: it owns the prompt-model side only. The
request-facing backend should not load or call this model directly.
"""

from __future__ import annotations

import os
import time
from typing import Any, Optional

import numpy as np
import torch

from streammuse.infrastructure.inference.lekai_http_backend import EventPayload
from streammuse.infrastructure.inference.lekai_model.MidiConverter import MidiConverter
from streammuse.infrastructure.inference.runtime_device import (
    dtype_to_name,
    parse_env_bool,
    resolve_device,
    resolve_dtype,
)
from streammuse.infrastructure.inference.lekai_prompt_continuation.prompt_model.my_tokenizer import (
    PianoMusicTokenizer,
)
from streammuse.infrastructure.inference.lekai_prompt_continuation.token_conversion import (
    copy_events,
)


TIMESTEPS_PER_BEAT = 4
DEFAULT_BEATS_PER_BAR = 4


class LekaiPromptEngine:
    """Load and run the prompt model that generates initial accompaniment."""

    def __init__(self, checkpoint_path: Optional[str] = None) -> None:
        self._checkpoint_path = checkpoint_path
        self._model = None
        self._tokenizer = PianoMusicTokenizer()
        self._converter = MidiConverter(ticks_per_beat=TIMESTEPS_PER_BEAT)
        self._mode = "not_wired"
        self._fallback_reason: Optional[str] = None
        self._resolved_device = "cpu"
        self._resolved_dtype = "float32"
        self._load_time_ms: Optional[float] = None
        self._warmup_time_ms: Optional[float] = None
        self._warmup_error: Optional[str] = None
        self._is_warmed_up = False
        self._last_generated_acc_beats = 0
        self._last_prompt_token_ids: list[int] = []
        self._last_generated_token_ids: list[int] = []
        self._last_new_token_ids: list[int] = []
        self._last_generation_metadata: dict[str, Any] = {}

        if checkpoint_path:
            self._load_model(checkpoint_path)

    @staticmethod
    def _event_sort_key(event: EventPayload) -> tuple[int, int]:
        tick = int(event.get("tick", 0))
        priority = 0 if str(event.get("type", "")) == "note_off" else 1
        return tick, priority

    @staticmethod
    def _env_positive_int(name: str) -> Optional[int]:
        raw = os.environ.get(name)
        if raw is None or raw.strip() == "":
            return None
        value = int(raw)
        return value if value > 0 else None

    @staticmethod
    def _env_float(name: str, default: float) -> float:
        raw = os.environ.get(name)
        if raw is None or raw.strip() == "":
            return default
        try:
            return float(raw)
        except ValueError:
            return default

    @staticmethod
    def _env_int(name: str, default: int) -> int:
        raw = os.environ.get(name)
        if raw is None or raw.strip() == "":
            return default
        try:
            return int(raw)
        except ValueError:
            return default

    @staticmethod
    def _env_optional_int(name: str) -> Optional[int]:
        raw = os.environ.get(name)
        if raw is None or raw.strip() == "":
            return None
        try:
            return int(raw)
        except ValueError:
            return None

    @staticmethod
    def _env_bool(name: str, default: bool) -> bool:
        return parse_env_bool(os.environ.get(name), default=default)

    def _require_real_model(self) -> bool:
        return self._env_bool("LEKAI_PROMPT_REQUIRE_REAL_MODEL", False) or self._env_bool(
            "LEKAI_PROMPT_CONTINUATION_REQUIRE_REAL_MODELS",
            False,
        )

    def _seed_if_configured(self) -> None:
        seed = self._env_optional_int("LEKAI_PROMPT_SEED")
        if seed is None:
            seed = self._env_optional_int("LEKAI_SEED")
        if seed is None:
            return
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))

    def _prompt_condition_length_ticks(self, prompt_length_ticks: int) -> int:
        time_signature_idx = self._env_int("LEKAI_PROMPT_TIME_SIGNATURE_INDEX", 4)
        beats_per_bar = self._measure_beats_from_time_signature_idx(time_signature_idx)
        condition_beats = self._env_positive_int("LEKAI_PROMPT_CONDITION_BEATS")
        if condition_beats is not None:
            condition_ticks = int(condition_beats) * TIMESTEPS_PER_BEAT
            return max(1, min(int(prompt_length_ticks), int(condition_ticks)))

        # Compatibility escape hatch for older experiments. The default path is
        # beat-based: prompt_length_ticks=32 means an 8-beat prompt condition.
        condition_bars = self._env_positive_int("LEKAI_PROMPT_CONDITION_BARS") or self._env_positive_int(
            "LEKAI_PROMPT_NUM_BARS"
        )
        if condition_bars is not None:
            condition_ticks = int(condition_bars) * int(beats_per_bar) * TIMESTEPS_PER_BEAT
            return max(1, min(int(prompt_length_ticks), int(condition_ticks)))

        condition_ticks = int(prompt_length_ticks)
        return max(1, min(int(prompt_length_ticks), int(condition_ticks)))

    def _measure_beats_from_time_signature_idx(self, time_signature_idx: int) -> int:
        prompt_override = self._env_positive_int("LEKAI_PROMPT_MEASURE_BEATS")
        if prompt_override is not None:
            return prompt_override
        shared_override = self._env_positive_int("LEKAI_MEASURE_BEATS")
        if shared_override is not None:
            return shared_override

        # Match the Lekai offline metadata conventions. Unknown values fall back
        # to common-time behavior instead of corrupting the prompt sequence.
        if int(time_signature_idx) == 9:
            time_signature_idx = 4
        return {
            0: 4,  # 4/4
            1: 3,  # 3/4
            2: 2,  # 2/4
            3: 3,  # 3/4
            4: 4,
            6: 6,
        }.get(int(time_signature_idx), DEFAULT_BEATS_PER_BAR)

    def _sync_device(self) -> None:
        if self._resolved_device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()

    def _load_model(self, checkpoint_path: str) -> None:
        if not os.path.exists(checkpoint_path):
            self._mode = "not_wired"
            self._fallback_reason = f"checkpoint_not_found:{checkpoint_path}"
            return

        device_preference = os.environ.get("LEKAI_PROMPT_DEVICE", os.environ.get("LEKAI_DEVICE", "auto"))
        dtype_preference = os.environ.get("LEKAI_PROMPT_DTYPE", os.environ.get("LEKAI_DTYPE", "auto"))

        try:
            device = resolve_device(device_preference)
            dtype = resolve_dtype(device, dtype_preference)
        except Exception as exc:
            self._mode = "not_wired"
            self._fallback_reason = f"invalid_runtime_preference:{exc}"
            return

        use_fp16 = dtype == torch.float16
        load_start = time.perf_counter()
        try:
            from streammuse.infrastructure.inference.lekai_prompt_continuation.prompt_model.config import (
                ModelConfig,
            )
            from streammuse.infrastructure.inference.lekai_prompt_continuation.prompt_model.inference import (
                load_model,
            )

            self._model = load_model(
                model_path=checkpoint_path,
                model_config=ModelConfig(),
                device=device,
                use_fp16=use_fp16,
            )
        except Exception as exc:
            self._model = None
            self._mode = "not_wired"
            self._fallback_reason = f"model_load_failed:{exc}"
            return

        self._mode = "real_model"
        self._fallback_reason = None
        self._resolved_device = device
        self._resolved_dtype = dtype_to_name(dtype)
        self._load_time_ms = (time.perf_counter() - load_start) * 1000
        if self._env_bool("LEKAI_PROMPT_WARMUP", True):
            self.warmup()

    def _has_real_model(self) -> bool:
        return self._model is not None

    def runtime_info(self) -> dict[str, str | float | bool | None]:
        return {
            "mode": self._mode,
            "has_real_model": self._has_real_model(),
            "checkpoint_path": self._checkpoint_path,
            "fallback_reason": self._fallback_reason,
            "resolved_device": self._resolved_device,
            "resolved_dtype": self._resolved_dtype,
            "load_time_ms": self._load_time_ms,
            "warmup_time_ms": self._warmup_time_ms,
            "warmup_error": self._warmup_error,
            "is_warmed_up": self._is_warmed_up,
        }

    def _active_pitches_before_tick(
        self,
        events: list[EventPayload],
        cutoff_tick: int,
    ) -> set[int]:
        active: set[int] = set()
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

    def _build_melody_prompt_tokens(
        self,
        melody_events: list[EventPayload],
        prompt_start_tick: int,
        prompt_length_ticks: int,
    ) -> tuple[torch.Tensor, int, int]:
        time_signature_idx = self._env_int("LEKAI_PROMPT_TIME_SIGNATURE_INDEX", 4)
        beats_per_bar = self._measure_beats_from_time_signature_idx(time_signature_idx)
        timesteps_per_bar = TIMESTEPS_PER_BEAT * beats_per_bar
        num_bars = max(1, int(np.ceil(prompt_length_ticks / timesteps_per_bar)))
        window_ticks = num_bars * timesteps_per_bar

        metadata = {
            "time_signature_idx": time_signature_idx,
            "bpm": self._env_int("LEKAI_PROMPT_BPM", self._env_int("LEKAI_DEFAULT_BPM", 120)),
            "num_measures": num_bars,
        }
        prompt_end_tick = int(prompt_start_tick) + window_ticks

        active = self._active_pitches_before_tick(melody_events, int(prompt_start_tick))
        melody_pr = self._converter.events_to_pianoroll(
            events=melody_events,
            start_tick=int(prompt_start_tick),
            end_tick=prompt_end_tick,
            active_pitches=active,
        )

        measures = []
        for bar_idx in range(num_bars):
            start = bar_idx * timesteps_per_bar
            end = start + timesteps_per_bar
            measure = np.zeros((4, 88, timesteps_per_bar), dtype=np.uint8)
            measure[:2] = melody_pr[:, :, start:end]
            measures.append(measure)

        ts_token = self._tokenizer.encode_time_sig(metadata["time_signature_idx"])
        bpm_token = self._tokenizer.encode_bpm(metadata["bpm"])
        _, _, _, measure_beats = self._tokenizer._encode_measures(measures, metadata)

        v = self._tokenizer.vocab
        mel_parts = []
        for beats in measure_beats:
            mel_parts.append(torch.tensor([v.bar_token_id], dtype=torch.long))
            for mel, _acc in beats:
                mel_parts.append(torch.tensor([v.beat_marker], dtype=torch.long))
                mel_parts.append(mel)

        prefix = torch.tensor([v.bos_token_id, ts_token, bpm_token], dtype=torch.long)
        prompt_tokens = torch.cat([prefix, torch.cat(mel_parts)])
        return prompt_tokens, int(metadata["bpm"]), window_ticks

    def _decode_accompaniment_events(
        self,
        generated_tokens: torch.Tensor,
        prompt_start_tick: int,
        prompt_length_ticks: int,
    ) -> list[EventPayload]:
        _mel_beats, acc_beats = self._tokenizer.parse_generated_sequence(generated_tokens.squeeze(0))
        if not acc_beats:
            self._last_generated_acc_beats = 0
            return []
        self._last_generated_acc_beats = min(
            len(acc_beats),
            max(0, (int(prompt_length_ticks) + TIMESTEPS_PER_BEAT - 1) // TIMESTEPS_PER_BEAT),
        )

        acc_pr = self._tokenizer.decode_beats_to_pianoroll(
            acc_beats,
            track_marker_id=self._tokenizer.vocab.track_marker_acc,
        )
        if acc_pr.shape[2] > prompt_length_ticks:
            acc_pr = acc_pr[:, :, :prompt_length_ticks]

        events, _active = self._converter.pianoroll_to_events(
            pianoroll=acc_pr,
            start_tick=int(prompt_start_tick),
            close_at_end=True,
            active_pitches=None,
        )

        normalized: list[EventPayload] = []
        for event in events:
            payload: EventPayload = {
                "type": str(event["type"]),
                "pitch": int(event["pitch"]),
                "tick": int(event["tick"]),
            }
            if "velocity" in event:
                payload["velocity"] = int(event["velocity"])
            normalized.append(payload)
        return normalized

    def last_generated_acc_beats(self) -> int:
        """Return the number of accompaniment beats produced by the last prompt call."""

        return int(self._last_generated_acc_beats)

    def last_generation_log(self) -> dict[str, Any]:
        """Return token-level diagnostics for the latest prompt generation."""

        return {
            "prompt_tokens": list(self._last_prompt_token_ids),
            "prompt_token_count": len(self._last_prompt_token_ids),
            "generated_tokens": list(self._last_generated_token_ids),
            "generated_token_count": len(self._last_generated_token_ids),
            "new_tokens": list(self._last_new_token_ids),
            "new_token_count": len(self._last_new_token_ids),
            "generated_acc_beats": int(self._last_generated_acc_beats),
            **dict(self._last_generation_metadata),
        }

    def _generation_max_length(self, prompt_token_count: int, max_new_tokens: Optional[int] = None) -> int:
        if max_new_tokens is None:
            max_new_tokens = self._env_positive_int("LEKAI_PROMPT_MAX_NEW_TOKENS") or 2048
        max_length_cap = self._env_positive_int("LEKAI_PROMPT_MAX_LENGTH") or 8192
        return min(max_length_cap, int(prompt_token_count) + int(max_new_tokens))

    def _generate_tokens(
        self,
        prompt_tokens: torch.Tensor,
        *,
        max_new_tokens: Optional[int] = None,
    ) -> torch.Tensor:
        assert self._model is not None
        top_k = self._env_int("LEKAI_PROMPT_TOP_K", 0)
        if top_k < 0:
            top_k = 0

        self._seed_if_configured()
        return self._model.generate_music(
            initial_tokens=prompt_tokens,
            device=self._resolved_device,
            max_length=self._generation_max_length(
                prompt_token_count=int(prompt_tokens.numel()),
                max_new_tokens=max_new_tokens,
            ),
            temperature=self._env_float("LEKAI_PROMPT_TEMPERATURE", 1.1),
            top_k=top_k,
            top_p=self._env_float("LEKAI_PROMPT_TOP_P", 0.95),
            repetition_penalty=self._env_float("LEKAI_PROMPT_REPETITION_PENALTY", 1.0),
        )

    def warmup(self) -> dict[str, str | float | bool | None]:
        """Run one dummy two-bar prompt generation to pay first-call overhead at startup."""

        if not self._has_real_model():
            self._warmup_error = "model_not_loaded"
            self._is_warmed_up = False
            return self.runtime_info()

        warmup_time_signature_idx = self._env_int("LEKAI_PROMPT_TIME_SIGNATURE_INDEX", 4)
        warmup_beats = self._env_positive_int("LEKAI_PROMPT_WARMUP_BEATS") or (
            self._measure_beats_from_time_signature_idx(warmup_time_signature_idx) * 2
        )
        warmup_ticks = int(warmup_beats) * TIMESTEPS_PER_BEAT
        warmup_new_tokens = self._env_positive_int("LEKAI_PROMPT_WARMUP_MAX_NEW_TOKENS")

        dummy_melody: list[EventPayload] = [
            {"type": "note_on", "pitch": 60, "tick": 0, "velocity": 80},
            {"type": "note_off", "pitch": 60, "tick": TIMESTEPS_PER_BEAT},
        ]

        try:
            prompt_tokens, _bpm, _window_ticks = self._build_melody_prompt_tokens(
                melody_events=dummy_melody,
                prompt_start_tick=0,
                prompt_length_ticks=warmup_ticks,
            )
            self._sync_device()
            start = time.perf_counter()
            _ = self._generate_tokens(
                prompt_tokens,
                max_new_tokens=warmup_new_tokens,
            )
            self._sync_device()
            self._warmup_time_ms = (time.perf_counter() - start) * 1000
            self._warmup_error = None
            self._is_warmed_up = True
        except Exception as exc:
            self._warmup_time_ms = None
            self._warmup_error = str(exc)
            self._is_warmed_up = False
        return self.runtime_info()

    def generate_prompt_accompaniment(
        self,
        melody_events: list[EventPayload],
        prompt_start_tick: int,
        prompt_length_ticks: int,
    ) -> list[EventPayload]:
        """Generate prompt accompaniment for the initial prompt window."""

        copied_melody = copy_events(melody_events)
        prompt_start_tick = int(prompt_start_tick)
        prompt_length_ticks = int(prompt_length_ticks)
        if prompt_length_ticks <= 0:
            return []
        if not self._has_real_model():
            if self._require_real_model():
                reason = self._fallback_reason or "prompt_model_not_loaded"
                raise RuntimeError(f"Lekai prompt fallback disabled: {reason}")
            return []

        condition_length_ticks = self._prompt_condition_length_ticks(prompt_length_ticks)
        prompt_tokens, bpm, window_ticks = self._build_melody_prompt_tokens(
            melody_events=copied_melody,
            prompt_start_tick=prompt_start_tick,
            prompt_length_ticks=condition_length_ticks,
        )
        prompt_token_ids = [int(token) for token in prompt_tokens.detach().cpu().flatten().tolist()]
        self._last_prompt_token_ids = prompt_token_ids
        self._last_generation_metadata = {
            "bpm": int(bpm),
            "prompt_start_tick": int(prompt_start_tick),
            "requested_prompt_length_ticks": int(prompt_length_ticks),
            "condition_length_ticks": int(condition_length_ticks),
            "window_ticks": int(window_ticks),
            "temperature": self._env_float("LEKAI_PROMPT_TEMPERATURE", 1.1),
            "top_k": max(0, self._env_int("LEKAI_PROMPT_TOP_K", 0)),
            "top_p": self._env_float("LEKAI_PROMPT_TOP_P", 0.95),
            "repetition_penalty": self._env_float("LEKAI_PROMPT_REPETITION_PENALTY", 1.0),
        }
        generated = self._generate_tokens(prompt_tokens)
        generated_token_ids = [int(token) for token in generated.detach().cpu().flatten().tolist()]
        self._last_generated_token_ids = generated_token_ids
        self._last_new_token_ids = generated_token_ids[len(prompt_token_ids):]
        return self._decode_accompaniment_events(
            generated_tokens=generated,
            prompt_start_tick=prompt_start_tick,
            prompt_length_ticks=prompt_length_ticks,
        )
