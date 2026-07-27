"""Factory for interactive human-response sources."""

from __future__ import annotations

from pathlib import Path

from streammuse.application.tasks.human_input import (
    HumanInputConfig,
    TerminalHumanResponseSource,
    TerminalIO,
)
from streammuse.domain.tasks import HumanResponseSource


class HumanInputFactory:
    """Build terminal or optional voice input without eager voice imports."""

    @staticmethod
    def create(
        config: HumanInputConfig,
        *,
        terminal: TerminalIO,
        artifact_root: str | Path | None = None,
    ) -> HumanResponseSource:
        if config.mode == "terminal":
            return TerminalHumanResponseSource(terminal)

        voice_config = config.voice
        if voice_config is None:  # HumanInputConfig already validates this boundary.
            raise ValueError("voice mode requires voice configuration")

        from streammuse.infrastructure.voice import VoiceHumanResponseSource

        return VoiceHumanResponseSource(voice_config, artifact_root=artifact_root)
