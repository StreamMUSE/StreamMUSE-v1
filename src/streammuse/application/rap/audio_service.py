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
    PreparedRapChunk,
    RenderedSyllable,
    RemoteRapChunkRequest,
    SyllableRenderRequest,
)
from streammuse.domain.timing import Tempo

if TYPE_CHECKING:
    from streammuse.application.rap.realtime import PlannedRapBar
    from streammuse.domain.rap import RapScenario


class RapAudioController(Protocol):
    @property
    def scenario(self) -> RapScenario: ...

    def start(self) -> None: ...

    def on_tick(self, tick: int) -> None: ...

    def request_stop(self, *, successor_bar: int | None) -> None: ...

    def resume_audio(self, bar: int) -> None: ...

    def resume_after_stop(self) -> None: ...

    def reset(self) -> int: ...

    def close(self) -> None: ...


class SpeechSynthesizer(Protocol):
    def synthesize(self, request: SyllableRenderRequest) -> RenderedSyllable: ...


class DrumRenderer(Protocol):
    def render(self, template: FlowTemplate, tempo: Tempo, audio_format: AudioFormat, bar: int) -> PcmAudio: ...


class RapBarRenderer(Protocol):
    def render(self, plan: PlannedRapBar) -> PreparedRapBar: ...


class RapChunkPreparationStrategy(Protocol):
    def prepare(self, request: RemoteRapChunkRequest, *, deadline_monotonic: float) -> PreparedRapChunk: ...

    def abort(self) -> None: ...

    def close(self) -> None: ...


class RapAudioSink(Protocol):
    def start(self) -> bool | None: ...

    def enqueue(self, bar: PreparedRapBar) -> None: ...

    def request_stop_after_bar(self) -> AudioPlaybackSnapshot: ...

    def discard_pending(self) -> None: ...

    def reset(self) -> None: ...

    def snapshot(self) -> AudioPlaybackSnapshot: ...

    def drain_notices(self) -> tuple[AudioPlaybackNotice, ...]: ...

    def close(self) -> None: ...
