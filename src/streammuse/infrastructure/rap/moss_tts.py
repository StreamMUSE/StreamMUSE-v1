"""Persistent MOSS phrase synthesis for the realtime H200 worker."""

from __future__ import annotations

import hashlib
import importlib
import os
import tempfile
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
from scipy.io import wavfile

from streammuse.experiments.rap_audio_protocols.contracts import (
    SyllableTarget,
    TwoBarRenderRequest,
)
from streammuse.experiments.rap_audio_protocols.timing import moss_token_target


RuntimeLoader = Callable[..., object]
PhraseGenerator = Callable[..., None]


class MossSynthesisFailed(RuntimeError):
    """Raised when MOSS does not produce a valid connected-phrase WAV."""


def _freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


@dataclass(frozen=True)
class MossPhraseResult:
    """Validated raw MOSS phrase plus immutable reproduction metadata."""

    output_wav: Path
    model_id: str
    model_revision: str
    reference_voice_sha256: str
    source_wav_sha256: str
    sample_rate_hz: int
    frame_count: int
    generation_time_ms: float
    resolved_generation_settings: Mapping[str, object]
    warnings: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "resolved_generation_settings",
            _freeze(self.resolved_generation_settings),
        )

    @property
    def duration_seconds(self) -> float:
        return self.frame_count / self.sample_rate_hz


class PersistentMossSynthesizer:
    """Reuse one MOSS runtime across complete-phrase synthesis calls."""

    def __init__(
        self,
        *,
        runtime: object,
        model_id: str,
        device: str,
        reference_wav: Path,
        reference_voice_sha256: str,
        phrase_generator: PhraseGenerator,
        backend_module: object,
        clock: Callable[[], float],
    ) -> None:
        self._runtime = runtime
        self._model_id = model_id
        self._device = device
        self._reference_wav = reference_wav
        self._reference_voice_sha256 = reference_voice_sha256
        self._phrase_generator = phrase_generator
        self._backend = backend_module
        self._clock = clock

    @classmethod
    def load(
        cls,
        *,
        model_id: str,
        device: str,
        reference_wav: Path,
        runtime_loader: RuntimeLoader | None = None,
        phrase_generator: PhraseGenerator | None = None,
        clock: Callable[[], float] = time.perf_counter,
        **runtime_options: Any,
    ) -> "PersistentMossSynthesizer":
        reference_path = Path(reference_wav)
        try:
            reference_bytes = reference_path.read_bytes()
        except OSError as exc:
            raise MossSynthesisFailed(
                f"unable to read MOSS reference voice: {reference_path}"
            ) from exc
        if not reference_bytes:
            raise MossSynthesisFailed("MOSS reference voice must not be empty")

        backend = importlib.import_module("scripts.rap_audio_backends.moss_backend")
        load_runtime = runtime_loader or getattr(backend, "create_runtime")
        generate = phrase_generator or getattr(backend, "_generate_chunk")
        runtime = load_runtime(
            model_id=model_id,
            device=device,
            **runtime_options,
        )
        return cls(
            runtime=runtime,
            model_id=model_id,
            device=device,
            reference_wav=reference_path,
            reference_voice_sha256=hashlib.sha256(reference_bytes).hexdigest(),
            phrase_generator=generate,
            backend_module=backend,
            clock=clock,
        )

    def warmup(self) -> Mapping[str, object]:
        """Execute one disposable phrase generation to warm model kernels."""
        started = self._clock()
        warmup = getattr(self._runtime, "warmup", None)
        if callable(warmup):
            warmup()
        with tempfile.TemporaryDirectory(prefix="streammuse-moss-warmup-") as temp_dir:
            result = self.synthesize(
                _warmup_request(),
                Path(temp_dir) / "moss-warmup.wav",
            )
        return MappingProxyType(
            {
                "model_id": self._model_id,
                "device": self._device,
                "generated": True,
                "sample_rate_hz": result.sample_rate_hz,
                "frame_count": result.frame_count,
                "source_wav_sha256": result.source_wav_sha256,
                "warmup_time_ms": max(0.0, (self._clock() - started) * 1000.0),
            }
        )

    def synthesize(
        self,
        request: TwoBarRenderRequest,
        output_wav: Path,
    ) -> MossPhraseResult:
        if not isinstance(request, TwoBarRenderRequest):
            raise MossSynthesisFailed(
                "MOSS synthesis requires a two-bar render request"
            )

        output_path = Path(output_wav)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        partial_path = output_path.with_name(
            f".{output_path.stem}.partial{output_path.suffix or '.wav'}"
        )
        output_path.unlink(missing_ok=True)
        partial_path.unlink(missing_ok=True)
        started = self._clock()
        try:
            self._phrase_generator(
                request=request,
                output_path=partial_path,
                reference_wav=self._reference_wav,
                runtime=self._runtime,
            )
            sample_rate_hz, samples = _read_valid_mono_wav(partial_path)
            frame_count = int(samples.shape[0])
            os.replace(partial_path, output_path)
        except Exception as exc:
            partial_path.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)
            if isinstance(exc, MossSynthesisFailed):
                raise
            raise MossSynthesisFailed(f"MOSS phrase synthesis failed: {exc}") from exc

        generation_time_ms = max(0.0, (self._clock() - started) * 1000.0)
        generation_kwargs = dict(getattr(self._backend, "GENERATION_KWARGS"))
        settings = {
            "language": str(getattr(self._backend, "LANGUAGE")),
            "instruction": str(getattr(self._backend, "RAP_INSTRUCTION")),
            "generation_mode": str(getattr(self._backend, "GENERATION_MODE")),
            "generation_kwargs": generation_kwargs,
            "token_target": moss_token_target(request),
        }
        warnings = tuple(
            str(value)
            for value in (
                getattr(self._backend, "STYLE_INSTRUCTION_CAVEAT", ""),
                getattr(self._backend, "DETERMINISTIC_SEED_CAVEAT", ""),
            )
            if value
        )
        return MossPhraseResult(
            output_wav=output_path,
            model_id=self._model_id,
            model_revision=_model_revision(self._runtime),
            reference_voice_sha256=self._reference_voice_sha256,
            source_wav_sha256=_file_sha256(output_path),
            sample_rate_hz=sample_rate_hz,
            frame_count=frame_count,
            generation_time_ms=generation_time_ms,
            resolved_generation_settings=settings,
            warnings=warnings,
        )


def _read_valid_mono_wav(path: Path) -> tuple[int, np.ndarray]:
    try:
        sample_rate_hz, samples = wavfile.read(path)
    except Exception as exc:
        raise MossSynthesisFailed(f"MOSS output is not a readable WAV: {path}") from exc
    array = np.asarray(samples)
    if sample_rate_hz <= 0:
        raise MossSynthesisFailed("MOSS output sample rate must be positive")
    if array.ndim == 2 and array.shape[1] == 1:
        array = array[:, 0]
    if array.ndim != 1:
        raise MossSynthesisFailed("MOSS output must be mono")
    if array.size == 0:
        raise MossSynthesisFailed("MOSS output must not be empty")
    if not np.isfinite(array).all():
        raise MossSynthesisFailed("MOSS output must contain only finite samples")
    if float(np.max(np.abs(array.astype(np.float64)))) == 0.0:
        raise MossSynthesisFailed("MOSS output must not be silent")
    return int(sample_rate_hz), array


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _model_revision(runtime: object) -> str:
    model = getattr(runtime, "model", None)
    config = getattr(model, "config", None)
    for owner in (config, model, runtime):
        revision = getattr(owner, "_commit_hash", None)
        if isinstance(revision, str) and revision:
            return revision
    return "unknown"


def _warmup_request() -> TwoBarRenderRequest:
    return TwoBarRenderRequest(
        song_id="streammuse-moss-warmup",
        chunk_index=0,
        start_bar=0,
        end_bar=2,
        text="warm voice",
        syllables=(
            SyllableTarget(
                word="warm",
                index_in_word=0,
                phonemes=("W", "AO1", "R", "M"),
                lexical_stress=1,
                target_stress=1.0,
                boundary_strength=0,
                absolute_tick=1,
                tick_in_chunk=1,
                target_seconds=0.25,
            ),
            SyllableTarget(
                word="voice",
                index_in_word=0,
                phonemes=("V", "OY1", "S"),
                lexical_stress=1,
                target_stress=1.0,
                boundary_strength=2,
                absolute_tick=4,
                tick_in_chunk=4,
                target_seconds=0.75,
            ),
        ),
    )
