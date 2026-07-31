"""Cached synthesis and speaker playback for interactive task answers."""

from __future__ import annotations

import time
import wave
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

import numpy as np

from streammuse.application.tasks.speech_output import SpeechOutputConfig
from streammuse.domain.tasks import SpeechPlayback, SpeechRequest

from . import (
    SpeakerPlaybackError,
    SpeechArtifactError,
    SpeechSynthesisError,
    _json_safe,
)
from .speaker import SpeakerPlayer
from .synthesizer import SpeechSynthesizer, SynthesizedAudio


class AudioSpeechOutput:
    mode = "audio"

    def __init__(
        self,
        config: SpeechOutputConfig,
        *,
        synthesizer: SpeechSynthesizer,
        speaker: SpeakerPlayer,
        artifact_root: str | Path | None = None,
        now: Callable[[], float] | None = None,
    ) -> None:
        self.config = config
        self._synthesizer = synthesizer
        self._speaker = speaker
        self._artifact_root = (
            None if artifact_root is None else Path(artifact_root)
        )
        self._now = now or time.perf_counter
        self._cache: dict[str, SynthesizedAudio] = {}
        self._cache_bytes = 0
        self._prewarm_ms = 0.0
        self._prewarm_truncated = False
        self._started = False
        self._closed = False

    def start(self) -> None:
        if self._started and not self._closed:
            return
        try:
            self._synthesizer.start()
            self._speaker.start()
        except BaseException:
            try:
                self.close()
            except BaseException:
                pass
            raise
        self._started = True
        self._closed = False

    @property
    def provenance(self) -> dict[str, Any]:
        return _json_safe(
            {
                "mode": self.mode,
                "backend": self.config.backend,
                "requested_voice": self.config.voice,
                "requested_rate": float(self.config.rate),
                "requested_speaker_device": self.config.speaker_device,
                "model": self.config.model,
                "model_cache": self.config.model_cache,
                "model_revision": self.config.model_revision,
                "local_files_only": self.config.local_files_only,
                **self._synthesizer.provenance,
                **self._speaker.provenance,
                "prewarm_ms": self._prewarm_ms,
                "prewarm_entry_count": len(self._cache),
                "prewarm_truncated": self._prewarm_truncated,
                "guard_ms": float(self.config.guard_ms),
                "cache_miss": self.config.cache_miss,
                "cache_max_entries": int(self.config.cache_max_entries),
                "cache_max_bytes": int(self.config.cache_max_bytes),
                "llm_deadline_basis": self.config.llm_deadline_basis,
                "on_error": self.config.on_error,
                "save_audio": self.config.save_audio,
            }
        )

    def prepare(self, phrases: tuple[str, ...]) -> None:
        if not self.config.prewarm:
            return
        started_s = self._now()
        try:
            for phrase in dict.fromkeys(str(item) for item in phrases):
                if phrase in self._cache:
                    continue
                if len(self._cache) >= self.config.cache_max_entries:
                    self._prewarm_truncated = True
                    break
                audio = self._synthesizer.synthesize(phrase)
                if self._cache_bytes + audio.byte_count > self.config.cache_max_bytes:
                    self._prewarm_truncated = True
                    break
                self._cache[phrase] = audio
                self._cache_bytes += audio.byte_count
        finally:
            self._prewarm_ms = max(
                0.0, (self._now() - started_s) * 1000.0
            )

    def speak(self, request: SpeechRequest) -> SpeechPlayback:
        speak_started_s = self._now()
        audio = self._cache.get(request.text)
        cached = audio is not None
        synthesis_ms = 0.0
        status = "ok"
        try:
            if audio is None:
                if self.config.cache_miss == "skip":
                    return SpeechPlayback(
                        status="cache_miss_skipped",
                        spoken_text=request.text,
                    )
                synthesis_started_s = self._now()
                try:
                    audio = self._synthesizer.synthesize(request.text)
                except SpeechSynthesisError as exc:
                    return self._failure(
                        "synthesis_failed",
                        request.text,
                        exc,
                        synthesis_ms=max(
                            0.0,
                            (self._now() - synthesis_started_s) * 1000.0,
                        ),
                    )
                synthesis_ms = max(
                    0.0, (self._now() - synthesis_started_s) * 1000.0
                )
                status = "cache_miss_synthesized"

            speaker_started_s = self._now()
            try:
                played = self._speaker.play(audio)
            except SpeakerPlaybackError as exc:
                return self._failure(
                    "playback_failed",
                    request.text,
                    exc,
                    synthesis_ms=synthesis_ms,
                    audio=audio,
                )
            shift_ms = max(
                0.0, (speaker_started_s - speak_started_s) * 1000.0
            )
            playback = SpeechPlayback(
                status=(
                    "playback_failed"
                    if played.error is not None
                    else status
                ),  # type: ignore[arg-type]
                spoken_text=request.text,
                cached=cached,
                synthesis_ms=synthesis_ms,
                audio_duration_ms=audio.duration_ms,
                completed_normally=played.completed_normally,
                playback_start_offset_ms=played.playback_start_offset_ms
                + shift_ms,
                first_dac_sample_offset_ms=_shift_optional(
                    played.first_dac_sample_offset_ms, shift_ms
                ),
                playback_drained_offset_ms=_shift_optional(
                    played.playback_drained_offset_ms, shift_ms
                ),
                stream_inactive_offset_ms=_shift_optional(
                    played.stream_inactive_offset_ms, shift_ms
                ),
                metadata={
                    "sample_rate_hz": played.sample_rate_hz,
                    "device": played.device,
                },
                error=(
                    None
                    if played.error is None
                    else _error_dict(played.error)
                ),
            )
            if self.config.save_audio and played.completed_normally:
                return self._persist_audio(playback, request, audio)
            return playback
        except BaseException:
            self._speaker.abort_active()
            raise

    def _persist_audio(
        self,
        playback: SpeechPlayback,
        request: SpeechRequest,
        audio: SynthesizedAudio,
    ) -> SpeechPlayback:
        if self._artifact_root is None:
            error = SpeechArtifactError(
                "speech-save-audio requires an artifact root"
            )
            return replace(
                playback,
                status="artifact_failed",
                error=_error_dict(error),
            )
        filename = (
            f"{request.turn_id + 1:04d}_turn_{request.turn_id:04d}_llm.wav"
        )
        target = self._artifact_root / filename
        temporary = target.with_suffix(".wav.tmp")
        started_s = self._now()
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            pcm = np.clip(audio.samples, -1.0, 1.0)
            pcm16 = (pcm * 32767.0).astype("<i2", copy=False)
            with wave.open(str(temporary), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(audio.sample_rate_hz)
                handle.writeframes(pcm16.tobytes())
            temporary.replace(target)
            persistence_ms = max(
                0.0, (self._now() - started_s) * 1000.0
            )
            relative = f"artifacts/turn/{filename}"
            return replace(
                playback,
                audio_artifact=relative,
                artifact_persistence_ms=persistence_ms,
            )
        except (OSError, wave.Error) as exc:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            persistence_ms = max(
                0.0, (self._now() - started_s) * 1000.0
            )
            error = SpeechArtifactError(
                f"Could not persist synthesized audio: {exc}"
            )
            return replace(
                playback,
                status="artifact_failed",
                audio_artifact=None,
                artifact_persistence_ms=persistence_ms,
                error=_error_dict(error),
            )

    @staticmethod
    def _failure(
        status: str,
        spoken_text: str,
        error: BaseException,
        *,
        synthesis_ms: float = 0.0,
        audio: SynthesizedAudio | None = None,
    ) -> SpeechPlayback:
        return SpeechPlayback(
            status=status,  # type: ignore[arg-type]
            spoken_text=spoken_text,
            synthesis_ms=synthesis_ms,
            audio_duration_ms=0.0 if audio is None else audio.duration_ms,
            error=_error_dict(error),
        )

    def drain(self) -> None:
        self._speaker.drain()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        first_error: BaseException | None = None
        try:
            self._speaker.close()
        except BaseException as exc:
            first_error = exc
        try:
            self._synthesizer.close()
        except BaseException as exc:
            if first_error is None:
                first_error = exc
        self._cache.clear()
        self._cache_bytes = 0
        if first_error is not None:
            raise first_error


def _shift_optional(value: float | None, shift_ms: float) -> float | None:
    return None if value is None else float(value) + shift_ms


def _error_dict(error: BaseException) -> dict[str, str]:
    value = _json_safe(
        {"type": type(error).__name__, "message": str(error)}
    )
    return {"type": str(value["type"]), "message": str(value["message"])}
