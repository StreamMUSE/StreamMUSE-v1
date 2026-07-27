"""Human response source that composes microphone capture and speech-to-text."""

from __future__ import annotations

import time
import wave
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal

import numpy as np

from streammuse.domain.tasks import HumanResponse, HumanResponseRequest

from . import VoiceInfrastructureError, _json_safe
from .faster_whisper import FasterWhisperRecognizer
from .microphone import CapturedUtterance, MicrophoneCapture

if TYPE_CHECKING:
    from streammuse.application.tasks.human_input import VoiceInputConfig


WaveWriter = Callable[[Path, np.ndarray, int], None]


def _write_wav(path: Path, audio: np.ndarray, sample_rate_hz: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate_hz)
        handle.writeframes(pcm)


class VoiceHumanResponseSource:
    """Capture and transcribe exactly one utterance for each human turn."""

    mode: Literal["voice"] = "voice"

    def __init__(
        self,
        config: VoiceInputConfig,
        *,
        artifact_root: str | Path | None = None,
        microphone: MicrophoneCapture | None = None,
        recognizer: FasterWhisperRecognizer | None = None,
        wave_writer: WaveWriter | None = None,
        now: Callable[[], float] | None = None,
    ) -> None:
        self.config = config
        self.artifact_root = Path(artifact_root) if artifact_root is not None else None
        self._now = now or time.perf_counter
        self._microphone = microphone or MicrophoneCapture(config, now=self._now)
        self._recognizer = recognizer or FasterWhisperRecognizer(config, now=self._now)
        self._wave_writer = wave_writer or _write_wav
        self._started = False
        self._closed = False
        if self.config.save_audio and self.artifact_root is None:
            raise ValueError("artifact_root is required when save_audio is enabled")

    @property
    def provenance(self) -> dict[str, Any]:
        return _json_safe(
            {
                "mode": self.mode,
                "save_audio": bool(self.config.save_audio),
                "microphone": self._microphone.provenance,
                "recognizer": self._recognizer.provenance,
            }
        )

    def start(self) -> None:
        """Preflight the microphone and load/warm the resident model."""

        if self._closed:
            raise VoiceInfrastructureError("The voice response source has already been closed.")
        if self._started:
            return
        try:
            self._microphone.start()
            self._recognizer.start()
        except BaseException:
            try:
                self.close()
            except BaseException:
                pass
            raise
        self._started = True

    def read_response(self, request: HumanResponseRequest) -> HumanResponse:
        """Capture, optionally persist, and transcribe one human response."""

        if self._closed:
            raise VoiceInfrastructureError("The voice response source has already been closed.")
        if not self._started:
            raise VoiceInfrastructureError("VoiceHumanResponseSource.start() must succeed first.")
        turn_started = self._now()
        captured = self._microphone.capture(timeout_s=request.timeout_s)
        if not captured.has_speech:
            total_latency_ms = max(
                captured.endpoint_detected_offset_ms,
                (self._now() - turn_started) * 1000.0,
            )
            deadline_expired = captured.deadline_expired
            if request.timeout_s is not None:
                deadline_expired = deadline_expired or total_latency_ms >= max(
                    0.0, request.timeout_s * 1000.0
                )
            audio_artifact, artifact_persistence_ms = self._save_audio_timed(
                request.turn_id,
                captured,
            )
            base_metadata = self._capture_metadata(
                captured,
                audio_artifact=audio_artifact,
                artifact_persistence_ms=artifact_persistence_ms,
            )
            base_metadata.update(
                {
                    "status": "no_speech",
                    "raw_transcript": "",
                    "asr_start_offset_ms": None,
                    "asr_end_offset_ms": None,
                    "asr_latency_ms": 0.0,
                    "total_latency_ms": total_latency_ms,
                    "asr": None,
                }
            )
            return HumanResponse(
                text="",
                status="no_speech",
                deadline_expired=deadline_expired,
                latency_ms=total_latency_ms,
                metadata=_json_safe(base_metadata),
            )

        asr_started = self._now()
        asr_start_offset_ms = max(
            captured.endpoint_detected_offset_ms,
            (asr_started - turn_started) * 1000.0,
        )
        result = self._recognizer.transcribe(captured.audio, speech_context=request.speech_context)
        asr_ended = self._now()
        asr_end_offset_ms = max(asr_start_offset_ms + result.latency_ms, (asr_ended - turn_started) * 1000.0)
        if not result.accepted:
            status = "rejected_transcript"
        elif not result.text:
            status = "empty_transcript"
        else:
            status = "ok"
        total_latency_ms = max(asr_end_offset_ms, (asr_ended - turn_started) * 1000.0)
        deadline_expired = captured.deadline_expired
        if request.timeout_s is not None:
            deadline_expired = deadline_expired or total_latency_ms >= max(0.0, request.timeout_s * 1000.0)
        audio_artifact, artifact_persistence_ms = self._save_audio_timed(
            request.turn_id,
            captured,
        )
        base_metadata = self._capture_metadata(
            captured,
            audio_artifact=audio_artifact,
            artifact_persistence_ms=artifact_persistence_ms,
        )
        base_metadata.update(
            {
                "status": status,
                "raw_transcript": result.text,
                "transcript_rejection_reasons": list(result.rejection_reasons),
                "asr_start_offset_ms": asr_start_offset_ms,
                "asr_end_offset_ms": asr_end_offset_ms,
                "asr_latency_ms": result.latency_ms,
                "total_latency_ms": total_latency_ms,
                "asr": _json_safe(result.diagnostics),
            }
        )
        return HumanResponse(
            text=result.text,
            status=status,
            deadline_expired=deadline_expired,
            latency_ms=total_latency_ms,
            metadata=_json_safe(base_metadata),
        )

    @staticmethod
    def _capture_metadata(
        captured: CapturedUtterance,
        *,
        audio_artifact: str | None,
        artifact_persistence_ms: float,
    ) -> dict[str, Any]:
        return {
            "mode": "voice",
            "wait_for_speech_ms": captured.wait_for_speech_ms,
            "utterance_ms": captured.utterance_ms,
            "endpoint_silence_ms": captured.endpoint_silence_ms,
            "last_voiced_offset_ms": captured.last_voiced_offset_ms,
            "endpoint_detected_offset_ms": captured.endpoint_detected_offset_ms,
            "endpoint_reason": captured.endpoint_reason,
            "capture_sample_rate_hz": captured.capture_sample_rate_hz,
            "audio_overflow": captured.audio_overflow,
            "audio_artifact": audio_artifact,
            "artifact_persistence_ms": artifact_persistence_ms,
        }

    def _save_audio_timed(
        self,
        turn_id: int,
        captured: CapturedUtterance,
    ) -> tuple[str | None, float]:
        if not self.config.save_audio or self.artifact_root is None or captured.audio.size == 0:
            return None, 0.0
        started = self._now()
        artifact = self._save_audio(turn_id, captured)
        return artifact, max(0.0, (self._now() - started) * 1000.0)

    def _save_audio(self, turn_id: int, captured: CapturedUtterance) -> str | None:
        if not self.config.save_audio or self.artifact_root is None or captured.audio.size == 0:
            return None
        filename = f"{int(turn_id) + 1:04d}_turn_{int(turn_id):04d}_human.wav"
        path = self.artifact_root / filename
        try:
            self._wave_writer(path, captured.audio, captured.sample_rate_hz)
        except Exception as exc:
            raise VoiceInfrastructureError(f"Could not persist captured microphone audio: {exc}") from exc
        parts = path.parts
        if "artifacts" in parts:
            artifacts_index = max(
                index for index, part in enumerate(parts) if part == "artifacts"
            )
            return Path(*parts[artifacts_index:]).as_posix()
        return filename

    def close(self) -> None:
        """Close both components once, even if one close operation fails."""

        if self._closed:
            return
        self._closed = True
        first_error: BaseException | None = None
        try:
            self._microphone.close()
        except BaseException as exc:  # Preserve cleanup while retaining the first failure.
            first_error = exc
        try:
            self._recognizer.close()
        except BaseException as exc:
            if first_error is None:
                first_error = exc
        if first_error is not None:
            raise first_error
