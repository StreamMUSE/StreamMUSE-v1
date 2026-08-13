"""Persistent faster-whisper adapter for short independent game responses."""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from . import SpeechRecognitionError, VoiceDependencyError, _json_safe

if TYPE_CHECKING:
    from streammuse.application.tasks.human_input import VoiceInputConfig
    from streammuse.domain.tasks import SpeechContext


@dataclass(frozen=True)
class TranscriptionResult:
    """Raw ASR output and JSON-safe model diagnostics."""

    text: str
    latency_ms: float
    diagnostics: dict[str, Any] = field(default_factory=dict)
    rejection_reasons: tuple[str, ...] = ()

    @property
    def accepted(self) -> bool:
        """Whether the raw model output passed bounded quality checks."""

        return not self.rejection_reasons


_MAX_TRANSCRIPT_CHARACTERS = 512
_REJECT_AT_CONSECUTIVE_TOKEN_REPETITIONS = 16
_MAX_SEGMENT_COMPRESSION_RATIO = 10.0
_TRANSCRIPT_TOKEN_PATTERN = re.compile(r"[\w']+", flags=re.UNICODE)


def _default_model_factory(*args: Any, **kwargs: Any) -> Any:
    try:
        from faster_whisper import WhisperModel  # type: ignore[import-not-found]
    except (ImportError, OSError) as exc:
        raise VoiceDependencyError(
            "Voice transcription requires faster-whisper. Install it with "
            "`uv sync --extra voice` (or install the project's voice extra)."
        ) from exc
    return WhisperModel(*args, **kwargs)


def _default_model_downloader(*args: Any, **kwargs: Any) -> str:
    try:
        from faster_whisper.utils import download_model  # type: ignore[import-not-found]
    except (ImportError, OSError) as exc:
        raise VoiceDependencyError(
            "Voice transcription requires faster-whisper. Install it with "
            "`uv sync --extra voice` (or install the project's voice extra)."
        ) from exc
    return str(download_model(*args, **kwargs))


class FasterWhisperRecognizer:
    """Load, warm, and reuse one deterministic faster-whisper model."""

    def __init__(
        self,
        config: VoiceInputConfig,
        *,
        model_factory: Callable[..., Any] | None = None,
        model_downloader: Callable[..., str] | None = None,
        now: Callable[[], float] | None = None,
    ) -> None:
        self.config = config
        self._model_factory = model_factory or _default_model_factory
        self._model_downloader = model_downloader or _default_model_downloader
        self._now = now or time.perf_counter
        self._model: Any | None = None
        self._model_argument: str | None = None
        self._resolved_model_path: str | None = None
        self._resolved_revision: str | None = None
        self._resolution_ms: float | None = None
        self._load_ms: float | None = None
        self._warmup_ms: float | None = None
        self._closed = False

    @property
    def provenance(self) -> dict[str, Any]:
        return _json_safe(
            {
                "model": self.config.model,
                "device": self.config.device,
                "compute_type": self.config.compute_type,
                "model_cache": self.config.model_cache,
                "model_revision_requested": self.config.model_revision,
                "model_revision_resolved": self._resolved_revision,
                "model_path_resolved": self._resolved_model_path,
                "local_files_only": self.config.local_files_only,
                "decode": {
                    "language": "en",
                    "beam_size": 1,
                    "condition_on_previous_text": False,
                    "task": "transcribe",
                    "temperature": 0,
                    "vad_filter": False,
                    "word_timestamps": False,
                    "without_timestamps": True,
                },
                "transcript_quality_gate": {
                    "max_characters": _MAX_TRANSCRIPT_CHARACTERS,
                    "reject_at_consecutive_token_repetitions": (
                        _REJECT_AT_CONSECUTIVE_TOKEN_REPETITIONS
                    ),
                    "max_segment_compression_ratio": (
                        _MAX_SEGMENT_COMPRESSION_RATIO
                    ),
                },
                "model_resolution_ms": self._resolution_ms,
                "model_load_ms": self._load_ms,
                "warmup_ms": self._warmup_ms,
            }
        )

    def start(self) -> None:
        """Load the configured model once and fully consume a warm-up decode."""

        if self._model is not None:
            return
        if self._closed:
            raise SpeechRecognitionError("The speech recognizer has already been closed.")

        resolution_started = self._now()
        try:
            model_argument = self._resolve_model_argument()
        finally:
            self._resolution_ms = max(0.0, (self._now() - resolution_started) * 1000.0)
        kwargs: dict[str, Any] = {
            "device": self.config.device,
            "compute_type": self.config.compute_type,
        }
        load_started = self._now()
        try:
            model = self._model_factory(model_argument, **kwargs)
        except VoiceDependencyError:
            raise
        except Exception as exc:
            offline_hint = (
                " Check --voice-model-cache/--voice-local-files-only and ensure the requested "
                "model snapshot is present."
                if self.config.local_files_only
                else " Check the model identifier, cache, network access, and CTranslate2 installation."
            )
            raise SpeechRecognitionError(f"Could not load faster-whisper model: {exc}.{offline_hint}") from exc
        self._load_ms = max(0.0, (self._now() - load_started) * 1000.0)
        self._model = model
        self._record_resolved_model(model, model_argument)

        warmup_started = self._now()
        try:
            segments, _ = model.transcribe(
                np.zeros(16_000, dtype=np.float32),
                **self._decode_kwargs(initial_prompt=None, hotwords=None),
            )
            list(segments)
        except Exception as exc:
            try:
                close = getattr(model, "close", None)
                if callable(close):
                    close()
            except BaseException:
                pass
            self._model = None
            raise SpeechRecognitionError(f"Could not warm up faster-whisper model: {exc}") from exc
        self._warmup_ms = max(0.0, (self._now() - warmup_started) * 1000.0)

    def _resolve_model_argument(self) -> str:
        configured = str(self.config.model)
        path = Path(configured).expanduser()
        try:
            path_exists = path.exists()
        except OSError as exc:
            raise SpeechRecognitionError(
                f"Could not inspect faster-whisper model path {path}: {exc}"
            ) from exc
        if path_exists:
            try:
                if not path.is_dir():
                    raise SpeechRecognitionError(
                        f"Local faster-whisper model path must be a directory: {path}"
                    )
                resolved = str(path.resolve())
            except OSError as exc:
                raise SpeechRecognitionError(
                    f"Could not resolve faster-whisper model path {path}: {exc}"
                ) from exc
            self._model_argument = resolved
            self._resolved_model_path = resolved
            self._resolved_revision = self._revision_from_snapshot_path(path)
            return resolved

        download_kwargs: dict[str, Any] = {
            "local_files_only": bool(self.config.local_files_only),
        }
        if self.config.model_cache:
            download_kwargs["cache_dir"] = str(Path(self.config.model_cache).expanduser())
        if self.config.model_revision:
            download_kwargs["revision"] = self.config.model_revision
        try:
            downloaded = self._model_downloader(configured, **download_kwargs)
        except VoiceDependencyError:
            raise
        except Exception as exc:
            offline_hint = (
                " Check --voice-model-cache/--voice-local-files-only and ensure the requested "
                "model snapshot is present."
                if self.config.local_files_only
                else " Check the model identifier, cache, and network access."
            )
            raise SpeechRecognitionError(
                f"Could not resolve faster-whisper model snapshot: {exc}.{offline_hint}"
            ) from exc

        try:
            snapshot = Path(downloaded).expanduser()
            snapshot_is_dir = snapshot.is_dir()
        except (TypeError, OSError) as exc:
            raise SpeechRecognitionError(
                f"Could not inspect resolved faster-whisper model snapshot: {exc}"
            ) from exc
        if not snapshot_is_dir:
            raise SpeechRecognitionError(
                "faster-whisper model resolution did not return a local snapshot directory: "
                f"{snapshot}"
            )
        try:
            resolved = str(snapshot.resolve())
        except OSError as exc:
            raise SpeechRecognitionError(
                f"Could not resolve faster-whisper model snapshot path {snapshot}: {exc}"
            ) from exc
        self._model_argument = resolved
        self._resolved_model_path = resolved
        self._resolved_revision = self._revision_from_snapshot_path(snapshot)
        return resolved

    @staticmethod
    def _revision_from_snapshot_path(path: Path) -> str | None:
        parts = path.parts
        if "snapshots" in parts:
            index = len(parts) - 1 - tuple(reversed(parts)).index("snapshots")
            if index + 1 < len(parts):
                return parts[index + 1]
        return None

    def _record_resolved_model(self, model: Any, model_argument: str) -> None:
        if self._resolved_model_path is not None:
            if self._resolved_revision is None:
                self._resolved_revision = self._revision_from_snapshot_path(
                    Path(self._resolved_model_path)
                )
            return
        try:
            inner_model = getattr(model, "model", None)
        except Exception:
            inner_model = None
        holders = (model, inner_model)
        for holder in holders:
            if holder is None:
                continue
            for attribute in ("model_path", "model_dir", "download_root"):
                try:
                    value = getattr(holder, attribute, None)
                except Exception:
                    continue
                if value:
                    self._resolved_model_path = str(value)
                    break
            if self._resolved_model_path is not None:
                break
        if self._resolved_model_path is None and Path(model_argument).exists():
            self._resolved_model_path = str(Path(model_argument).resolve())
        if self._resolved_revision is None and self._resolved_model_path:
            self._resolved_revision = self._revision_from_snapshot_path(Path(self._resolved_model_path))

    @staticmethod
    def _render_hotwords(hotwords: tuple[str, ...] | None) -> str | None:
        if not hotwords:
            return None
        rendered: list[str] = []
        seen: set[str] = set()
        for value in hotwords:
            normalized = " ".join(str(value).split())
            if normalized and normalized not in seen:
                rendered.append(normalized)
                seen.add(normalized)
        return " ".join(rendered) or None

    @staticmethod
    def _decode_kwargs(*, initial_prompt: str | None, hotwords: str | None) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "language": "en",
            "beam_size": 1,
            "condition_on_previous_text": False,
            "task": "transcribe",
            "temperature": 0,
            "vad_filter": False,
            "word_timestamps": False,
            "without_timestamps": True,
        }
        if initial_prompt:
            kwargs["initial_prompt"] = initial_prompt
        if hotwords:
            kwargs["hotwords"] = hotwords
        return kwargs

    def transcribe(
        self,
        audio: np.ndarray,
        *,
        speech_context: SpeechContext | None = None,
    ) -> TranscriptionResult:
        """Transcribe one 16-kHz mono float32 waveform without rewriting it."""

        if self._model is None:
            raise SpeechRecognitionError("FasterWhisperRecognizer.start() must succeed before transcribe().")
        pipeline_started = self._now()
        waveform = np.asarray(audio)
        if waveform.ndim != 1:
            raise SpeechRecognitionError("Speech audio must be a one-dimensional mono waveform.")
        if waveform.dtype != np.float32:
            waveform = waveform.astype(np.float32)
        waveform = np.ascontiguousarray(waveform)

        initial_prompt = None
        context_hotwords: tuple[str, ...] | None = None
        if speech_context is not None:
            initial_prompt = speech_context.initial_prompt
            context_hotwords = speech_context.hotwords
        rendered_hotwords = self._render_hotwords(context_hotwords)
        kwargs = self._decode_kwargs(initial_prompt=initial_prompt, hotwords=rendered_hotwords)
        input_ready = self._now()

        started = self._now()
        try:
            segments_iter, info = self._model.transcribe(waveform, **kwargs)
            model_call_returned = self._now()
            segments = list(segments_iter)
            segments_consumed = self._now()
        except Exception as exc:
            raise SpeechRecognitionError(f"faster-whisper transcription failed: {exc}") from exc
        latency_ms = max(0.0, (segments_consumed - started) * 1000.0)
        raw_text = "".join(str(getattr(segment, "text", "")) for segment in segments).strip()
        text_assembled = self._now()
        quality = self._transcript_quality(raw_text, segments)
        quality_completed = self._now()
        timing = {
            "schema_version": 1,
            "clock": "monotonic",
            "origin": "asr_pipeline_start",
            "anchors_ms": {
                "input_ready": _elapsed_ms(pipeline_started, input_ready),
                "model_call_started": _elapsed_ms(pipeline_started, started),
                "model_call_returned": _elapsed_ms(
                    pipeline_started,
                    model_call_returned,
                ),
                "segments_consumed": _elapsed_ms(
                    pipeline_started,
                    segments_consumed,
                ),
                "text_assembled": _elapsed_ms(
                    pipeline_started,
                    text_assembled,
                ),
                "quality_gate_completed": _elapsed_ms(
                    pipeline_started,
                    quality_completed,
                ),
            },
            "durations_ms": {
                "input_prepare": _elapsed_ms(pipeline_started, input_ready),
                "model_call": _elapsed_ms(started, model_call_returned),
                "segment_iteration": _elapsed_ms(
                    model_call_returned,
                    segments_consumed,
                ),
                "text_assembly": _elapsed_ms(
                    segments_consumed,
                    text_assembled,
                ),
                "quality_gate": _elapsed_ms(
                    text_assembled,
                    quality_completed,
                ),
                "pipeline_total": _elapsed_ms(
                    pipeline_started,
                    quality_completed,
                ),
            },
        }
        diagnostics = _json_safe(
            {
                "language": getattr(info, "language", None),
                "language_probability": getattr(info, "language_probability", None),
                "duration_s": getattr(info, "duration", None),
                "duration_after_vad_s": getattr(info, "duration_after_vad", None),
                "segment_count": len(segments),
                "initial_prompt": initial_prompt,
                "hotwords": rendered_hotwords,
                "segments": [self._segment_diagnostics(segment) for segment in segments],
                "quality_gate": quality,
                "timing_breakdown": timing,
            }
        )
        return TranscriptionResult(
            text=raw_text,
            latency_ms=latency_ms,
            diagnostics=diagnostics,
            rejection_reasons=tuple(quality["reasons"]),
        )

    @staticmethod
    def _transcript_quality(raw_text: str, segments: list[Any]) -> dict[str, Any]:
        tokens = [
            match.group(0).casefold()
            for match in _TRANSCRIPT_TOKEN_PATTERN.finditer(raw_text)
        ]
        max_repetitions = 0
        run_length = 0
        previous_token: str | None = None
        for token in tokens:
            run_length = run_length + 1 if token == previous_token else 1
            previous_token = token
            max_repetitions = max(max_repetitions, run_length)

        compression_ratios: list[float] = []
        invalid_compression_ratio_count = 0
        for segment in segments:
            value = getattr(segment, "compression_ratio", None)
            if value is None:
                continue
            try:
                ratio = float(value)
            except (TypeError, ValueError):
                invalid_compression_ratio_count += 1
                continue
            if math.isfinite(ratio):
                compression_ratios.append(ratio)
            else:
                invalid_compression_ratio_count += 1
        max_compression_ratio = max(compression_ratios, default=None)

        reasons: list[str] = []
        if len(raw_text) > _MAX_TRANSCRIPT_CHARACTERS:
            reasons.append("transcript_too_long")
        if max_repetitions >= _REJECT_AT_CONSECUTIVE_TOKEN_REPETITIONS:
            reasons.append("excessive_token_repetition")
        if (
            max_compression_ratio is not None
            and max_compression_ratio > _MAX_SEGMENT_COMPRESSION_RATIO
        ):
            reasons.append("excessive_compression_ratio")
        if invalid_compression_ratio_count:
            reasons.append("invalid_compression_ratio")

        return {
            "accepted": not reasons,
            "reasons": reasons,
            "transcript_characters": len(raw_text),
            "token_count": len(tokens),
            "max_consecutive_token_repetitions": max_repetitions,
            "max_segment_compression_ratio": max_compression_ratio,
            "invalid_compression_ratio_count": invalid_compression_ratio_count,
            "thresholds": {
                "max_characters": _MAX_TRANSCRIPT_CHARACTERS,
                "reject_at_consecutive_token_repetitions": (
                    _REJECT_AT_CONSECUTIVE_TOKEN_REPETITIONS
                ),
                "max_segment_compression_ratio": (
                    _MAX_SEGMENT_COMPRESSION_RATIO
                ),
            },
        }

    @staticmethod
    def _segment_diagnostics(segment: Any) -> dict[str, Any]:
        keys = ("id", "start", "end", "avg_logprob", "no_speech_prob", "compression_ratio")
        return {key: getattr(segment, key, None) for key in keys}

    def close(self) -> None:
        """Release the resident model. Safe to call repeatedly."""

        if self._closed:
            return
        self._closed = True
        model, self._model = self._model, None
        try:
            close = getattr(model, "close", None)
            if callable(close):
                close()
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:
            raise SpeechRecognitionError(
                f"Could not close faster-whisper model: {exc}"
            ) from exc


def _elapsed_ms(start_s: float, end_s: float) -> float:
    return max(0.0, (float(end_s) - float(start_s)) * 1000.0)
