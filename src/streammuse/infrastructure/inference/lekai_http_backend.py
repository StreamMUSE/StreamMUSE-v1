from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import torch

from streammuse.infrastructure.inference.runtime_device import (
    dtype_to_name,
    parse_env_bool,
    resolve_device,
    resolve_dtype,
)
from streammuse.infrastructure.inference.generation_logger import GenerationLogger


EventPayload = Dict[str, int | str]
TimingPayload = Dict[str, float]


class SessionStateError(RuntimeError):
    """A request does not belong to the currently active experiment session."""

# PianoLLaMA tokenization is fixed: 4 timesteps per beat.
TIMESTEPS_PER_BEAT = 4
DEFAULT_PROMPT_CONTEXT_BEATS = 128
DEFAULT_HISTORY_MAX_TICKS = DEFAULT_PROMPT_CONTEXT_BEATS * TIMESTEPS_PER_BEAT

# Stable continuation vocabulary used by the acc-first checkpoint.
LEKAI_EMPTY_TOKEN = 169
LEKAI_ACC_END_TOKEN = 170
LEKAI_MEL_END_TOKEN = 171
LEKAI_BEAT_TOKEN = 172
LEKAI_BAR_TOKEN = 255
LEKAI_BOS_TOKEN = 257
LEKAI_PAD_TOKEN = 258


def decode_part1_token_trace(
    beat_records: Sequence[Mapping[str, Any]],
    *,
    initial_active_pitches: Sequence[int] = (),
) -> List[EventPayload]:
    """Independently decode persisted per-beat part-1 tokens into wire events.

    The formal artifact gate calls this function on the immutable inference
    trace.  It uses only the persisted raw/structural token partitions and
    start ticks, not the server's already-decoded accompaniment payload.
    """

    from streammuse.infrastructure.inference.lekai_model.MidiConverter import (
        MidiConverter,
    )
    from streammuse.infrastructure.inference.lekai_continuation_model.my_tokenizer import (
        PianoMusicTokenizer,
    )

    tokenizer = PianoMusicTokenizer()
    converter = MidiConverter(ticks_per_beat=TIMESTEPS_PER_BEAT)
    active = {int(pitch) for pitch in initial_active_pitches}
    decoded: List[EventPayload] = []
    previous_start: int | None = None
    for index, record in enumerate(beat_records):
        if set(record) != {"target_beat", "start_tick", "raw_tokens", "boundary_tokens"}:
            raise ValueError(f"token-decode beat {index} has unexpected fields")
        target_beat = record.get("target_beat")
        start_tick = record.get("start_tick")
        raw = record.get("raw_tokens")
        boundary = record.get("boundary_tokens")
        if (
            isinstance(target_beat, bool)
            or not isinstance(target_beat, int)
            or isinstance(start_tick, bool)
            or not isinstance(start_tick, int)
            or start_tick != target_beat * TIMESTEPS_PER_BEAT
            or not isinstance(raw, list)
            or not isinstance(boundary, list)
            or any(isinstance(token, bool) or not isinstance(token, int) for token in raw + boundary)
        ):
            raise ValueError(f"token-decode beat {index} is malformed")
        if previous_start is not None and start_tick != previous_start + TIMESTEPS_PER_BEAT:
            raise ValueError("token-decode beat starts are not contiguous")
        previous_start = start_tick
        playable_raw = [token for token in raw if token != LEKAI_PAD_TOKEN]
        playback_source = (
            boundary
            if playable_raw == [LEKAI_BAR_TOKEN] and boundary
            else raw
        )
        playable = [token for token in playback_source if token != LEKAI_PAD_TOKEN]
        if playable and playable[-1] in {LEKAI_BAR_TOKEN, LEKAI_BEAT_TOKEN}:
            playable.pop()
        if not playable or playable in ([LEKAI_EMPTY_TOKEN], [LEKAI_ACC_END_TOKEN]):
            playable = [LEKAI_EMPTY_TOKEN, LEKAI_ACC_END_TOKEN]
        elif playable[-1] != LEKAI_ACC_END_TOKEN:
            playable.append(LEKAI_ACC_END_TOKEN)
        pianoroll = tokenizer.decode_beats_to_pianoroll(
            [playable],
            track_marker_id=LEKAI_ACC_END_TOKEN,
        )
        if tuple(int(value) for value in pianoroll.shape) != (2, 88, TIMESTEPS_PER_BEAT):
            raise ValueError(f"token-decode beat {index} has invalid pianoroll shape")
        events, active = converter.pianoroll_to_events(
            pianoroll=pianoroll,
            start_tick=start_tick,
            close_at_end=False,
            active_pitches=active,
        )
        decoded.extend(
            {
                "type": str(event["type"]),
                "pitch": int(event["pitch"]),
                "tick": int(event["tick"]),
                "velocity": int(event.get("velocity", 100)),
            }
            for event in events
        )
    return decoded


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
        # Requests and session resets are serialized.  The boundary worker does
        # not acquire this lock, which lets reset_session drain it without a
        # lock inversion with _model_generation_lock.
        self._session_gate = threading.RLock()
        self._accepting_requests = True
        self._session_epoch = 0
        self._session_id: Optional[str] = None
        self._effective_seed = int(os.environ.get("LEKAI_RT_SEED", "0"))
        self._sample_generator = self._make_generator(self._effective_seed, "cpu")
        self._generation_metadata: Dict[str, Dict[str, Any]] = {}
        self._current_generation_trace: Dict[str, Any] = {}
        self._checkpoint_sha256_cache: Dict[tuple[str, int, int], str] = {}
        self._melody_history: List[EventPayload] = []
        # Full session input is retained only for the cumulative audit digest;
        # model prompt history may be trimmed independently.
        self._input_digest_history: List[EventPayload] = []
        self._accompaniment_history: List[EventPayload] = []
        # Preserve the exact generated accompaniment tokens across HTTP requests.
        # Event round-tripping cannot distinguish a model-generated BAR from an empty beat.
        self._accompaniment_token_history: Dict[int, List[int]] = {}
        self._accompaniment_bar_token_history: Dict[int, List[int]] = {}
        self._model_generation_lock = threading.Lock()
        self._boundary_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="lekai-boundary",
        )
        self._pending_boundary_generations: Dict[int, Future[List[int]]] = {}
        self._injection_length_ticks: int = 0
        self._runtime_config: Optional[BackendRuntimeConfig] = None
        self._runtime_status = BackendRuntimeStatus()
        self._request_bpm: Optional[int] = None  # BPM from the current request, overrides LEKAI_DEFAULT_BPM

        # Model components (Phase 3: real model integration)
        self._model_adapter = None
        self._converter = None
        self._tokenizer = None
        self._active_pitches: Set[int] = set()  # For tracking sustained notes across calls
        
        # Logger for generation comparison
        log_dir = os.environ.get("LEKAI_RT_LOG_DIR", "logs/generation")
        self._logger = GenerationLogger(output_dir=log_dir, mode="fake_rt")
        
        # Try to load real model if checkpoint provided
        if checkpoint_path:
            self._load_model(checkpoint_path)

    @staticmethod
    def _canonical_sha256(value: Any) -> str:
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @staticmethod
    def _make_generator(seed: int, device: str) -> torch.Generator:
        # torch.Generator only accepts concrete device types.  MPS currently
        # uses the CPU generator accepted by torch.multinomial.
        generator_device = str(device) if str(device).startswith("cuda") else "cpu"
        try:
            generator = torch.Generator(device=generator_device)
        except (RuntimeError, TypeError):
            generator = torch.Generator()
        generator.manual_seed(int(seed))
        return generator

    def _checkpoint_sha256(self) -> Optional[str]:
        raw_path = self._runtime_status.checkpoint_path
        if not raw_path:
            return None
        path = Path(raw_path)
        if not path.is_file():
            return None
        stat = path.stat()
        key = (str(path.resolve()), int(stat.st_size), int(stat.st_mtime_ns))
        cached = self._checkpoint_sha256_cache.get(key)
        if cached is not None:
            return cached
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        result = digest.hexdigest()
        self._checkpoint_sha256_cache = {key: result}
        return result

    def _sampling_config(self) -> Dict[str, float | int]:
        def _float(name: str, default: float) -> float:
            try:
                return float(os.environ.get(name, str(default)))
            except (TypeError, ValueError):
                return default

        def _int(name: str, default: int) -> int:
            try:
                return int(os.environ.get(name, str(default)))
            except (TypeError, ValueError):
                return default

        return {
            "temperature": _float("LEKAI_RT_TEMPERATURE", 0.8),
            "top_k": _int("LEKAI_RT_TOP_K", 50),
            "top_p": _float("LEKAI_RT_TOP_P", 0.95),
            "repetition_penalty": _float("LEKAI_RT_REPETITION_PENALTY", 1.2),
        }

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
        if time_signature_idx == 9:
            time_signature_idx = 4
        return {
            0: 4,
            1: 3,
            2: 2,
            3: 3,
            4: 4,
            6: 6,
        }.get(time_signature_idx, 4)

    def _runtime_bool(self, name: str, default: bool) -> bool:
        return parse_env_bool(os.environ.get(name), default=default)

    def _prompt_context_beats(self) -> int:
        return (
            self._env_positive_int("LEKAI_PROMPT_CONTEXT_BEATS")
            or DEFAULT_PROMPT_CONTEXT_BEATS
        )

    def _continuation_initial_tokens(
        self,
        *,
        bpm: int,
        time_signature_idx: int,
    ) -> List[int]:
        if self._tokenizer is None:
            raise RuntimeError("Continuation tokenizer is not loaded.")
        encode_time_sig = getattr(self._tokenizer, "encode_time_sig", None)
        encode_bpm = getattr(self._tokenizer, "encode_bpm", None)
        if not callable(encode_time_sig) or not callable(encode_bpm):
            raise RuntimeError(
                "Continuation tokenizer must expose encode_time_sig() and encode_bpm()."
            )
        return [
            LEKAI_BOS_TOKEN,
            int(encode_time_sig(int(time_signature_idx))),
            int(encode_bpm(int(bpm))),
        ]

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
        assert self._model_adapter is not None and self._tokenizer is not None
        warmup_start = time.perf_counter()
        prompt = torch.tensor(
            self._continuation_initial_tokens(
                bpm=int(os.environ.get("LEKAI_DEFAULT_BPM", "120")),
                time_signature_idx=int(
                    os.environ.get("LEKAI_TIME_SIGNATURE_INDEX", "4")
                ),
            )
            + [LEKAI_BAR_TOKEN, LEKAI_BEAT_TOKEN],
            dtype=torch.long,
        )
        for _ in range(max(1, int(warmup_steps))):
            self._generate_part1_tokens_from_prompt(
                prompt,
                temperature=0.8,
                top_k=20,
                top_p=0.9,
                repetition_penalty=1.0,
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
        from streammuse.infrastructure.inference.lekai_continuation_model.inference_adapter import (
            PianoContinuationAdapter,
        )
        from streammuse.infrastructure.inference.lekai_model.MidiConverter import MidiConverter

        load_start = time.perf_counter()
        self._model_adapter = PianoContinuationAdapter.from_checkpoint(
            checkpoint_path,
            device=device,
            dtype=dtype,
            use_cache=use_cache,
        )
        self._converter = MidiConverter(ticks_per_beat=4)
        self._tokenizer = self._model_adapter.tokenizer
        codec = getattr(self._tokenizer, "_codec", None)
        required_apis = (
            getattr(codec, "image_to_patch_tokens", None),
            getattr(self._tokenizer, "compress_tokens", None),
            getattr(self._tokenizer, "decode_beats_to_pianoroll", None),
            getattr(self._tokenizer, "encode_time_sig", None),
            getattr(self._tokenizer, "encode_bpm", None),
        )
        if not all(callable(api) for api in required_apis):
            raise RuntimeError(
                "Loaded adapter does not expose the continuation tokenizer API."
            )
        load_time_ms = (time.perf_counter() - load_start) * 1000
        self._sample_generator = self._make_generator(self._effective_seed, device)
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
            self._sample_generator = self._make_generator(
                self._effective_seed, primary_device
            )
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
                    self._sample_generator = self._make_generator(
                        self._effective_seed, "cpu"
                    )
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

    def configure(self, config: BackendRuntimeConfig, input_file: Optional[str] = None) -> None:
        self._runtime_config = config
        if input_file:
            self._current_input_file = input_file
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

    def runtime_info(self) -> Dict[str, Any]:
        sampling = self._sampling_config()
        configured_history = self._env_positive_int("LEKAI_HISTORY_MAX_TICKS")
        source_path = Path(__file__)
        source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
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
            "generation_interval_ticks": (
                self._runtime_config.generation_interval_ticks
                if self._runtime_config is not None
                else None
            ),
            "generation_length_frames": (
                self._runtime_config.generation_length_frames
                if self._runtime_config is not None
                else None
            ),
            "prompt_length_ticks": (
                self._runtime_config.prompt_length_ticks
                if self._runtime_config is not None
                else None
            ),
            "ticks_per_beat": TIMESTEPS_PER_BEAT,
            "checkpoint_sha256": self._checkpoint_sha256(),
            "source_sha256": source_sha256,
            "code_identity": os.environ.get("STREAMMUSE_CODE_SHA", source_sha256),
            "effective_bpm": (
                self._request_bpm
                if self._request_bpm is not None
                else int(os.environ.get("LEKAI_DEFAULT_BPM", "120"))
            ),
            **sampling,
            "prompt_context_beats": self._prompt_context_beats(),
            "history_retention_ticks": configured_history or DEFAULT_HISTORY_MAX_TICKS,
            "max_generation_length_frames": self._env_positive_int(
                "LEKAI_MAX_GENERATION_LENGTH_FRAMES"
            ),
            "max_prompt_ticks": self._env_positive_int("LEKAI_MAX_PROMPT_TICKS"),
            "time_signature_index": int(os.environ.get("LEKAI_TIME_SIGNATURE_INDEX", "4")),
            "sample_seed": self._effective_seed,
            "session_id": self._session_id,
            "session_epoch": self._session_epoch,
            "accepting_requests": self._accepting_requests,
            "pending_boundary_generations": len(self._pending_boundary_generations),
            "boundary_generation_order": (
                "synchronous"
                if self._runtime_bool("LEKAI_DETERMINISTIC_BOUNDARY_GENERATION", False)
                else "single_executor_then_request"
            ),
        }

    def reset_session(self, seed: int) -> Dict[str, Any]:
        """Atomically retire the old session and create a seeded session epoch."""
        with self._session_gate:
            self._accepting_requests = False
            try:
                # Do not hold _model_generation_lock while waiting: boundary
                # futures need that lock in order to finish.
                self._resolve_pending_boundary_generations(store=False)
                with self._model_generation_lock:
                    self._melody_history = []
                    self._input_digest_history = []
                    self._accompaniment_history = []
                    self._accompaniment_token_history = {}
                    self._accompaniment_bar_token_history = {}
                    self._injection_length_ticks = 0
                    self._active_pitches = set()
                    self._request_bpm = None
                    self._generation_metadata = {}
                    self._current_generation_trace = {}
                    self._session_epoch += 1
                    self._session_id = uuid.uuid4().hex
                    self._effective_seed = int(seed)
                    self._sample_generator = self._make_generator(
                        self._effective_seed,
                        self._runtime_status.resolved_device,
                    )
            finally:
                self._accepting_requests = True
        return {
            "success": True,
            "session_id": self._session_id,
            "session_epoch": self._session_epoch,
            "effective_seed": self._effective_seed,
            "pending_boundary_generations": len(self._pending_boundary_generations),
        }

    def _validate_session(
        self,
        *,
        session_id: Optional[str],
        session_epoch: Optional[int],
    ) -> None:
        if not self._accepting_requests:
            raise SessionStateError("session_reset_in_progress")
        supplied = session_id is not None or session_epoch is not None
        if not supplied:
            if self._runtime_bool("LEKAI_REQUIRE_SESSION", False):
                raise SessionStateError("session_id and session_epoch are required")
            return
        if session_id is None or session_epoch is None:
            raise SessionStateError("session_id and session_epoch must be supplied together")
        if self._session_id is None:
            raise SessionStateError("no active experiment session; call /debug/reset_session first")
        if session_id != self._session_id or int(session_epoch) != self._session_epoch:
            raise SessionStateError(
                "stale session epoch: "
                f"received {session_id}/{session_epoch}, "
                f"active {self._session_id}/{self._session_epoch}"
            )

    def consume_generation_metadata(self, request_id: str) -> Dict[str, Any]:
        return dict(self._generation_metadata.pop(str(request_id), {}))

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

    def _trim_event_history(
        self,
        events: List[EventPayload],
        cutoff_tick: int,
    ) -> List[EventPayload]:
        active_anchors: Dict[int, EventPayload] = {}
        recent_events: List[EventPayload] = []

        for event in sorted(events, key=self._event_sort_key):
            if int(event.get("tick", 0)) >= cutoff_tick:
                recent_events.append(event)
                continue
            if "pitch" not in event:
                continue
            pitch = int(event["pitch"])
            if not 0 <= pitch <= 127:
                continue
            event_type = str(event.get("type", ""))
            if event_type == "note_on":
                active_anchors[pitch] = event
            elif event_type == "note_off":
                active_anchors.pop(pitch, None)

        retained = [*active_anchors.values(), *recent_events]
        retained.sort(key=self._event_sort_key)
        return retained

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

        codec = getattr(self._tokenizer, "_codec", None)
        image_to_patch_tokens = getattr(codec, "image_to_patch_tokens", None)
        compress_tokens = getattr(self._tokenizer, "compress_tokens", None)
        if not callable(image_to_patch_tokens) or not callable(compress_tokens):
            raise RuntimeError(
                "Continuation tokenizer must expose _codec.image_to_patch_tokens() "
                "and compress_tokens()."
            )
        tokens_matrix = image_to_patch_tokens(beat_pr, strict_mode=True)
        compressed = compress_tokens(tokens_matrix, track_marker=int(end_marker))

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
            raise RuntimeError("Model adapter does not expose model for interleaved decoding.")

        device = str(getattr(self._model_adapter, "device", "cpu"))
        use_cache = bool(getattr(self._model_adapter, "use_cache", True))

        generated = prompt_tokens.unsqueeze(0).to(device)
        raw_tokens: List[int] = []
        past_key_values = None

        with self._model_generation_lock, torch.no_grad():
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
                    generator=self._sample_generator,
                )

                token_val = int(next_token.item())
                generated = torch.cat([generated, next_token], dim=1)
                raw_tokens.append(token_val)

                if token_val in {
                    LEKAI_ACC_END_TOKEN,
                    LEKAI_BEAT_TOKEN,
                    LEKAI_BAR_TOKEN,
                }:
                    break

        return raw_tokens

    @staticmethod
    def _playable_part1_tokens(raw_tokens: List[int]) -> List[int]:
        playable = [token for token in raw_tokens if token != LEKAI_PAD_TOKEN]
        if playable and playable[-1] in {LEKAI_BAR_TOKEN, LEKAI_BEAT_TOKEN}:
            playable.pop()
        if not playable or playable in ([LEKAI_EMPTY_TOKEN], [LEKAI_ACC_END_TOKEN]):
            return [LEKAI_EMPTY_TOKEN, LEKAI_ACC_END_TOKEN]
        if playable[-1] != LEKAI_ACC_END_TOKEN:
            playable.append(LEKAI_ACC_END_TOKEN)
        return playable

    def _decode_acc_beat_tokens(self, beat_tokens: List[int]) -> np.ndarray:
        if self._tokenizer is None:
            raise RuntimeError("Continuation tokenizer is not loaded.")
        decode_beats = getattr(self._tokenizer, "decode_beats_to_pianoroll", None)
        if not callable(decode_beats):
            raise RuntimeError(
                "Continuation tokenizer must expose decode_beats_to_pianoroll()."
            )
        pianoroll = decode_beats(
            [beat_tokens],
            track_marker_id=LEKAI_ACC_END_TOKEN,
        )
        if pianoroll.shape[2] < TIMESTEPS_PER_BEAT:
            pianoroll = np.pad(
                pianoroll,
                ((0, 0), (0, 0), (0, TIMESTEPS_PER_BEAT - pianoroll.shape[2])),
                mode="constant",
            )
        elif pianoroll.shape[2] > TIMESTEPS_PER_BEAT:
            pianoroll = pianoroll[:, :, :TIMESTEPS_PER_BEAT]
        return pianoroll

    def _submit_boundary_generation(
        self,
        beat: int,
        prompt_tokens: torch.Tensor,
        *,
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty: float,
    ) -> None:
        if beat in self._pending_boundary_generations:
            raise RuntimeError(f"Boundary generation already pending for beat {beat}.")
        future = self._boundary_executor.submit(
            self._generate_part1_tokens_from_prompt,
            prompt_tokens.detach().cpu().clone(),
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )
        self._pending_boundary_generations[beat] = future

    def _resolve_pending_boundary_generations(
        self,
        *,
        through_beat: Optional[int] = None,
        store: bool = True,
    ) -> None:
        beats = sorted(
            beat
            for beat in self._pending_boundary_generations
            if through_beat is None or beat <= through_beat
        )
        for beat in beats:
            future = self._pending_boundary_generations.pop(beat)
            try:
                tokens = future.result()
            except Exception:
                if store:
                    raise
                continue
            if store:
                self._accompaniment_bar_token_history[beat] = list(tokens)

    def _generate_with_interleaved_prompt(
        self,
        generation_start_tick: int,
        generation_interval_ticks: int,
        generation_length_frames: int,
    ) -> List[EventPayload]:
        assert self._converter is not None and self._model_adapter is not None and self._tokenizer is not None

        _ = generation_interval_ticks
        raw_response_tokens: List[int] = []
        structural_tokens: List[int] = []
        part0_tokens: List[int] = []
        token_decode_beats: List[Dict[str, Any]] = []
        self._current_generation_trace = {}
        self._accompaniment_token_history = {
            beat: self._playable_part1_tokens(tokens)
            for beat, tokens in self._accompaniment_token_history.items()
        }

        current_beat = int(generation_start_tick) // TIMESTEPS_PER_BEAT
        num_beats_to_generate = max(1, int(generation_length_frames) // TIMESTEPS_PER_BEAT)
        if int(generation_length_frames) % TIMESTEPS_PER_BEAT != 0:
            num_beats_to_generate += 1

        context_beats = self._prompt_context_beats()
        start_beat = max(0, current_beat - context_beats)
        context_start_tick = start_beat * TIMESTEPS_PER_BEAT

        effective_bpm = self._request_bpm if self._request_bpm is not None else int(os.environ.get("LEKAI_DEFAULT_BPM", "120"))
        time_signature_idx = int(os.environ.get("LEKAI_TIME_SIGNATURE_INDEX", "4"))
        measure_beats = self._measure_beats_from_time_signature_idx(
            time_signature_idx
        )

        initial_tokens = self._continuation_initial_tokens(
            bpm=effective_bpm,
            time_signature_idx=time_signature_idx,
        )

        # Filter out any stale virtual events inserted by earlier debug code (tick < 0).
        real_accompaniment_history = [
            e for e in self._accompaniment_history if int(e.get("tick", 0)) >= 0
        ]

        accompaniment_context_events: List[EventPayload] = list(real_accompaniment_history)

        def build_standard_offline_prompt(
            target_beat: int,
        ) -> Tuple[torch.Tensor, List[int]]:
            seq: List[torch.Tensor] = [
                torch.tensor(initial_tokens, dtype=torch.long)
            ]
            prompt_part0_tokens: List[int] = []
            melody_active = self._active_pitches_before_tick(
                self._melody_history, context_start_tick
            )
            accompaniment_active = self._active_pitches_before_tick(
                accompaniment_context_events, context_start_tick
            )

            # Re-encode every completed beat from event history. This keeps
            # prompts consistent across request boundaries and within a
            # multi-beat request.
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
                prompt_part0_tokens.extend(
                    int(token) for token in mel_tokens.tolist()
                )
                seq.append(mel_tokens)

            if target_beat == start_beat or target_beat % measure_beats == 0:
                seq.append(torch.tensor([LEKAI_BAR_TOKEN], dtype=torch.long))
            seq.append(torch.tensor([LEKAI_BEAT_TOKEN], dtype=torch.long))
            return torch.cat(seq, dim=0), prompt_part0_tokens

        generated_events: List[EventPayload] = []
        token_decode_initial_active_pitches = sorted(
            self._active_pitches_before_tick(
                accompaniment_context_events,
                current_beat * TIMESTEPS_PER_BEAT,
            )
        )
        beat_diagnostics: List[Dict[str, Any]] = []
        last_prompt_tokens: List[int] = []

        try:
            rt_temperature = float(os.environ.get("LEKAI_RT_TEMPERATURE", "0.8"))
        except Exception:
            rt_temperature = 0.8
        try:
            rt_top_k = int(os.environ.get("LEKAI_RT_TOP_K", "50"))
        except Exception:
            rt_top_k = 50
        try:
            rt_top_p = float(os.environ.get("LEKAI_RT_TOP_P", "0.95"))
        except Exception:
            rt_top_p = 0.95
        try:
            rt_repetition_penalty = float(os.environ.get("LEKAI_RT_REPETITION_PENALTY", "1.2"))
        except Exception:
            rt_repetition_penalty = 1.2

        for beat_offset in range(num_beats_to_generate):
            target_beat = current_beat + beat_offset
            beat_start_tick = target_beat * TIMESTEPS_PER_BEAT

            prompt_tokens, part0_tokens = build_standard_offline_prompt(target_beat)
            last_prompt_tokens = [int(token) for token in prompt_tokens.tolist()]
            raw_generated_tokens = self._generate_part1_tokens_from_prompt(
                prompt_tokens,
                temperature=rt_temperature,
                top_k=rt_top_k,
                top_p=rt_top_p,
                repetition_penalty=rt_repetition_penalty,
            )
            playable_beat_tokens = self._playable_part1_tokens(raw_generated_tokens)
            raw_response_tokens.extend(int(token) for token in raw_generated_tokens)
            token_decode_beats.append(
                {
                    "target_beat": int(target_beat),
                    "start_tick": int(beat_start_tick),
                    "raw_tokens": [int(token) for token in raw_generated_tokens],
                    "boundary_tokens": [],
                }
            )
            beat_pianoroll = self._decode_acc_beat_tokens(playable_beat_tokens)
            pianoroll_nonzero = int(np.count_nonzero(beat_pianoroll))

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
            beat_events, next_running_active = self._converter.pianoroll_to_events(
                pianoroll=beat_pianoroll,
                start_tick=beat_start_tick,
                close_at_end=False,
                active_pitches=active_snapshot,
            )
            self._active_pitches = set(next_running_active)

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
            self._accompaniment_token_history[target_beat] = list(playable_beat_tokens)

            note_on_events = [
                event
                for event in normalized_beat_events
                if str(event.get("type", "")) == "note_on"
            ]
            event_ticks = [
                int(event.get("tick", 0)) for event in normalized_beat_events
            ]
            beat_diagnostics.append(
                {
                    "target_beat": int(target_beat),
                    "beat_start_tick": int(beat_start_tick),
                    "prompt_token_count": int(len(prompt_tokens)),
                    "generated_tokens": [int(token) for token in raw_generated_tokens],
                    "generated_token_count": int(len(raw_generated_tokens)),
                    "pianoroll_nonzero": pianoroll_nonzero,
                    "event_count": int(len(normalized_beat_events)),
                    "note_on_count": int(len(note_on_events)),
                    "min_event_tick": min(event_ticks) if event_ticks else None,
                    "max_event_tick": max(event_ticks) if event_ticks else None,
                }
            )

        final_target_beat = current_beat + num_beats_to_generate - 1
        part0_end_tick = final_target_beat * TIMESTEPS_PER_BEAT
        part0_roll = self._converter.events_to_pianoroll(
            events=self._melody_history,
            start_tick=context_start_tick,
            end_tick=part0_end_tick,
            active_pitches=self._active_pitches_before_tick(
                self._melody_history, context_start_tick
            ),
        )
        self._current_generation_trace = {
            "raw_tokens": raw_response_tokens,
            "structural_tokens": structural_tokens,
            "token_decode_beats": token_decode_beats,
            "token_decode_initial_active_pitches": token_decode_initial_active_pitches,
            "prompt_tokens": last_prompt_tokens,
            "part0_tokens": part0_tokens,
            "part0_roll": part0_roll,
            "part0_roll_start_tick": context_start_tick,
            "part0_roll_end_tick": part0_end_tick,
            "context_start_tick": context_start_tick,
        }
        
        # Get melody events in the prompt window
        prompt_melody_events = [
            e for e in self._melody_history
            if context_start_tick <= int(e.get("tick", 0)) <= generation_start_tick
        ]
        
        # Log the generation
        input_file = getattr(self, '_current_input_file', 'unknown')
        self._logger.log_generation(
            input_file=input_file,
            generation_start_tick=int(generation_start_tick),
            generation_length_frames=int(generation_length_frames),
            prompt_tokens=last_prompt_tokens,
            temperature=rt_temperature,
            top_k=rt_top_k,
            top_p=rt_top_p,
            repetition_penalty=rt_repetition_penalty,
            melody_events=prompt_melody_events,
            accompaniment_events=generated_events,
            bpm=self._request_bpm,
            notes=f"context_start_tick={context_start_tick}, current_beat={current_beat}",
            diagnostics={
                "context_start_tick": int(context_start_tick),
                "current_beat": int(current_beat),
                "start_beat": int(start_beat),
                "num_beats_to_generate": int(num_beats_to_generate),
                "beat_diagnostics": beat_diagnostics,
            },
            suffix="",
        )

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
        bpm: Optional[int] = None,
        input_file: Optional[str] = None,
        session_id: Optional[str] = None,
        session_epoch: Optional[int] = None,
        request_id: Optional[str] = None,
    ) -> tuple[List[EventPayload], TimingPayload]:
        """Generate one response and retain an auditable metadata envelope.

        The return shape remains the historical two-tuple.  The HTTP layer uses
        consume_generation_metadata(request_id) so existing direct callers do
        not need to change.
        """
        with self._session_gate:
            self._validate_session(session_id=session_id, session_epoch=session_epoch)
            effective_request_id = str(
                request_id
                or f"legacy-e{self._session_epoch}-{time.time_ns()}"
            )
            input_increment = [dict(event) for event in melody_events]
            cumulative_input = [dict(event) for event in self._input_digest_history] + input_increment
            self._current_generation_trace = {}
            accompaniment, timings = self._generate_locked(
                melody_events=melody_events,
                generation_start_tick=generation_start_tick,
                generation_length_frames=generation_length_frames,
                generation_interval_ticks=generation_interval_ticks,
                prompt_length_ticks=prompt_length_ticks,
                inference_mode=inference_mode,
                model_name=model_name,
                checkpoint_path=checkpoint_path,
                bpm=bpm,
                input_file=input_file,
            )
            self._input_digest_history.extend(input_increment)
            trace = dict(self._current_generation_trace)
            raw_tokens = [int(token) for token in trace.get("raw_tokens", [])]
            structural_tokens = [
                int(token) for token in trace.get("structural_tokens", [])
            ]
            prompt_tokens = [int(token) for token in trace.get("prompt_tokens", [])]
            part0_tokens = [int(token) for token in trace.get("part0_tokens", [])]
            token_decode_beats = [
                dict(record) for record in trace.get("token_decode_beats", [])
            ]
            token_decode_initial_active_pitches = [
                int(pitch)
                for pitch in trace.get("token_decode_initial_active_pitches", [])
            ]
            token_decode_payload = {
                "initial_active_pitches": token_decode_initial_active_pitches,
                "beats": token_decode_beats,
            }
            raw_part0_roll = trace.get("part0_roll")
            if isinstance(raw_part0_roll, np.ndarray):
                canonical_roll = np.ascontiguousarray(raw_part0_roll, dtype=np.uint8)
                part0_roll_shape = [int(value) for value in canonical_roll.shape]
                part0_roll_bytes_sha256 = hashlib.sha256(
                    canonical_roll.tobytes(order="C")
                ).hexdigest()
                part0_roll_digest = self._canonical_sha256(
                    {
                        "shape": part0_roll_shape,
                        "bytes_sha256": part0_roll_bytes_sha256,
                    }
                )
            else:
                part0_roll_shape = []
                part0_roll_bytes_sha256 = None
                part0_roll_digest = None
            metadata: Dict[str, Any] = {
                "request_id": effective_request_id,
                "session_id": self._session_id,
                "session_epoch": self._session_epoch,
                "effective_seed": self._effective_seed,
                "effective_bpm": int(
                    self._request_bpm
                    if self._request_bpm is not None
                    else os.environ.get("LEKAI_DEFAULT_BPM", "120")
                ),
                "generation_start_tick": int(generation_start_tick),
                "raw_tokens": raw_tokens,
                "structural_tokens": structural_tokens,
                "raw_token_digest": self._canonical_sha256(
                    {"raw": raw_tokens, "structural": structural_tokens}
                ),
                "token_decode_beats": token_decode_beats,
                "token_decode_initial_active_pitches": token_decode_initial_active_pitches,
                "token_decode_digest": self._canonical_sha256(token_decode_payload),
                "prompt_token_digest": self._canonical_sha256(prompt_tokens),
                "part0_tokens": part0_tokens,
                "part0_token_digest": self._canonical_sha256(part0_tokens),
                "part0_trace_available": isinstance(raw_part0_roll, np.ndarray),
                "input_increment_digest": self._canonical_sha256(input_increment),
                "input_cumulative_digest": self._canonical_sha256(cumulative_input),
                # Digest the exact 2x88xT array supplied to the part0 encoder;
                # an event-table digest is not a substitute because converter
                # window/carry semantics can change the actual model input.
                "part0_roll_digest": part0_roll_digest,
                "part0_roll_shape": part0_roll_shape,
                "part0_roll_bytes_sha256": part0_roll_bytes_sha256,
                "part0_roll_start_tick": trace.get("part0_roll_start_tick"),
                "part0_roll_end_tick": trace.get("part0_roll_end_tick"),
                # Bind the digest to the exact decoded event projection that the
                # HTTP client turns into MusicalEvent objects and the full
                # inference trace persists.  A missing wire velocity decodes to
                # 100, so normalize that default before hashing.
                "output_event_digest": self._canonical_sha256(
                    [
                        {
                            "type": str(event["type"]),
                            "pitch": int(event["pitch"]),
                            "tick": int(event["tick"]),
                            "velocity": int(event.get("velocity", 100)),
                        }
                        for event in accompaniment
                    ]
                ),
                "empty_success": len(accompaniment) == 0,
                "context_start_tick": trace.get("context_start_tick"),
            }
            self._generation_metadata[effective_request_id] = metadata
            return accompaniment, timings

    def _generate_locked(
        self,
        melody_events: List[EventPayload],
        generation_start_tick: int,
        generation_length_frames: int,
        generation_interval_ticks: int,
        prompt_length_ticks: Optional[int],
        inference_mode: str,
        model_name: str,
        checkpoint_path: Optional[str],
        bpm: Optional[int] = None,
        input_file: Optional[str] = None,
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
            ),
            input_file=input_file,
        )

        # Update request-level BPM if provided (overrides LEKAI_DEFAULT_BPM env var).
        if bpm is not None:
            self._request_bpm = int(bpm)

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

        if int(generation_start_tick) < 0:
            print(
                "[LekaiHttpBackend] negative start_tick "
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
        sampling = self._sampling_config()
        part1_beats = self._model_adapter.generate_from_beats(
            part0_beats=part0_beats,
            num_beats_to_generate=num_beats_to_generate,
            bpm=(
                self._request_bpm
                if self._request_bpm is not None
                else int(os.environ.get("LEKAI_DEFAULT_BPM", "120"))
            ),
            temperature=float(sampling["temperature"]),
            top_k=int(sampling["top_k"]),
            top_p=float(sampling["top_p"]),
            repetition_penalty=float(sampling["repetition_penalty"]),
            generator=self._sample_generator,
            verbose=False,
        )
        raw_tokens = [int(token) for beat in part1_beats for token in beat]
        prompt_tokens = [
            int(token)
            for beat in part0_beats
            for token in (beat.tolist() if hasattr(beat, "tolist") else beat)
        ]
        self._current_generation_trace = {
            "raw_tokens": raw_tokens,
            "structural_tokens": [],
            "prompt_tokens": prompt_tokens,
            "part0_tokens": prompt_tokens,
            "part0_roll": melody_pianoroll,
            "part0_roll_start_tick": prompt_start,
            "part0_roll_end_tick": prompt_start + int(melody_pianoroll.shape[2]),
            "context_start_tick": prompt_start,
        }
        
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
        with self._session_gate:
            return self._inject_history_locked(
                melody_events,
                accompaniment_events,
                injection_length_ticks,
            )

    def _inject_history_locked(
        self,
        melody_events: List[EventPayload],
        accompaniment_events: List[EventPayload],
        injection_length_ticks: int,
    ) -> Dict[str, int | bool | str]:
        self._resolve_pending_boundary_generations(store=False)
        self._melody_history = list(melody_events)
        self._input_digest_history = [dict(event) for event in melody_events]
        self._accompaniment_history = list(accompaniment_events)
        self._accompaniment_token_history = {}
        self._accompaniment_bar_token_history = {}
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
        with self._session_gate:
            return self._clear_history_locked()

    def _clear_history_locked(self) -> Dict[str, Any]:
        self._resolve_pending_boundary_generations(store=False)
        melody_history = [dict(event) for event in self._melody_history]
        accompaniment_history = [dict(event) for event in self._accompaniment_history]

        self._melody_history = []
        self._input_digest_history = []
        self._accompaniment_history = []
        self._accompaniment_token_history = {}
        self._accompaniment_bar_token_history = {}
        self._injection_length_ticks = 0
        self._active_pitches = set()
        self._request_bpm = None
        return {
            "success": True,
            "message": "History cleared",
            "melody_history": melody_history,
            "accompaniment_history": accompaniment_history,
        }

    def injection_status(self) -> Dict[str, Any]:
        return {
            "is_injected": self._injection_length_ticks > 0,
            "injection_length_ticks": self._injection_length_ticks,
            "runtime_model_name": self._runtime_config.model_name if self._runtime_config else "",
            "runtime_inference_mode": self._runtime_config.inference_mode if self._runtime_config else "",
            "generation_interval_ticks": (
                self._runtime_config.generation_interval_ticks
                if self._runtime_config is not None
                else None
            ),
            "generation_length_frames": (
                self._runtime_config.generation_length_frames
                if self._runtime_config is not None
                else None
            ),
            "prompt_length_ticks": (
                self._runtime_config.prompt_length_ticks
                if self._runtime_config is not None
                else None
            ),
            "ticks_per_beat": TIMESTEPS_PER_BEAT,
        }

    def _trim_histories(self, generation_start_tick: int, generation_length_frames: int) -> None:
        """Trim histories with a stable window independent from tiny generation lengths."""
        configured_max_ticks = self._env_positive_int("LEKAI_HISTORY_MAX_TICKS")
        if configured_max_ticks is None:
            configured_max_ticks = max(
                DEFAULT_HISTORY_MAX_TICKS,
                int(generation_length_frames) * 16,
            )

        max_history_ticks = configured_max_ticks
        cutoff_tick = int(generation_start_tick) - max_history_ticks
        
        if cutoff_tick > 0:
            self._accompaniment_history = self._trim_event_history(
                self._accompaniment_history,
                cutoff_tick,
            )
            self._melody_history = self._trim_event_history(
                self._melody_history,
                cutoff_tick,
            )
            self._accompaniment_token_history = {
                beat: tokens
                for beat, tokens in self._accompaniment_token_history.items()
                if beat * TIMESTEPS_PER_BEAT >= cutoff_tick
            }
            self._accompaniment_bar_token_history = {
                beat: tokens
                for beat, tokens in self._accompaniment_bar_token_history.items()
                if beat * TIMESTEPS_PER_BEAT >= cutoff_tick
            }

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
