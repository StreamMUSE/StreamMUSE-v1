"""Factory for optional interactive speech output."""

from __future__ import annotations

from pathlib import Path

from streammuse.application.tasks.speech_output import (
    SilentSpeechOutput,
    SpeechOutputConfig,
)
from streammuse.domain.tasks import SpeechOutputSink


class SpeechOutputFactory:
    @staticmethod
    def create(
        config: SpeechOutputConfig,
        *,
        artifact_root: str | Path | None = None,
    ) -> SpeechOutputSink:
        if config.mode == "off":
            return SilentSpeechOutput()

        from streammuse.infrastructure.voice.speaker import SpeakerPlayer
        from streammuse.infrastructure.voice.speech_sink import AudioSpeechOutput
        from streammuse.infrastructure.voice.synthesizer import create_synthesizer

        return AudioSpeechOutput(
            config,
            synthesizer=create_synthesizer(config),
            speaker=SpeakerPlayer(device=config.speaker_device),
            artifact_root=artifact_root,
        )
