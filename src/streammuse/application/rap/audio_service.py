"""Replaceable application boundaries for realtime rap audio."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from streammuse.domain.rap import (
    AudioFormat,
    AudioPlaybackNotice,
    AudioPlaybackSnapshot,
    FlowTemplate,
    PcmAudio,
    PreparedRapBar,
    RenderedSyllable,
    SyllableRenderRequest,
)
from streammuse.domain.timing import Tempo

if TYPE_CHECKING:
    from streammuse.application.rap.realtime import PlannedRapBar


class SpeechSynthesizer(Protocol):
    def synthesize(self, request: SyllableRenderRequest) -> RenderedSyllable: ...


class DrumRenderer(Protocol):
    def render(self, template: FlowTemplate, tempo: Tempo, audio_format: AudioFormat, bar: int) -> PcmAudio: ...


class RapBarRenderer(Protocol):
    def render(self, plan: PlannedRapBar) -> PreparedRapBar: ...


class RapAudioSink(Protocol):
    def start(self) -> bool | None: ...

    def enqueue(self, bar: PreparedRapBar) -> None: ...

    def request_stop_after_bar(self) -> AudioPlaybackSnapshot: ...

    def discard_pending(self) -> None: ...

    def reset(self) -> None: ...

    def snapshot(self) -> AudioPlaybackSnapshot: ...

    def drain_notices(self) -> tuple[AudioPlaybackNotice, ...]: ...

    def close(self) -> None: ...
