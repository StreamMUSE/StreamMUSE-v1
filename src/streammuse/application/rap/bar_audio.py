"""Protocol v3 per-bar audio: the shared render request, server render and client preparation.

``bar_render_request`` is used by both sides: the server renders the MOSS line
with it, and the client warps the returned source onto the same target times,
so both compute bit-identical targets (as ``chunk_render_request`` does for v2).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import hashlib
from pathlib import Path
import time
from typing import Protocol

from streammuse.application.rap.chunk_orchestration import PhraseSourceResult
from streammuse.application.rap.lyric_session import RenderedBarAudio
from streammuse.domain.rap.lyric_stream import BarAudioManifest, BarLyricRecord
from streammuse.experiments.rap_audio_protocols.contracts import SyllableTarget, TwoBarRenderRequest


def bar_render_request(record: BarLyricRecord, tempo_bpm: float) -> TwoBarRenderRequest:
    """The one-bar render request for a lyric record."""
    bar_start_tick = record.bar * 16
    seconds_per_tick = 60.0 / tempo_bpm / 4.0
    return TwoBarRenderRequest(
        song_id=record.session_id,
        chunk_index=record.bar,
        start_bar=record.bar,
        end_bar=record.bar + 1,
        text=record.text,
        syllables=tuple(
            SyllableTarget(
                word=item.syllable.word,
                index_in_word=item.syllable.index_in_word,
                phonemes=item.syllable.phonemes,
                lexical_stress=item.syllable.stress,
                target_stress=item.slot.accent,
                boundary_strength=item.slot.boundary_strength,
                absolute_tick=item.slot.tick,
                tick_in_chunk=item.slot.tick - bar_start_tick,
                target_seconds=(item.slot.tick - bar_start_tick) * seconds_per_tick,
            )
            for item in record.scheduled
        ),
        tempo_bpm=tempo_bpm,
    )


class SourcePhraseRenderer(Protocol):
    def render_source(self, request: TwoBarRenderRequest, workspace: Path) -> PhraseSourceResult: ...


class BarPackageEncoder(Protocol):
    media_type: str

    def encode(self, manifest: BarAudioManifest, source_wav: bytes) -> bytes: ...


class MossBarAudioRenderer:
    """Server side: one MOSS line plus MMS onsets, packaged for one lyric record."""

    def __init__(
        self,
        renderer: SourcePhraseRenderer,
        encoder: BarPackageEncoder,
        *,
        workspace_root: Path,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._renderer = renderer
        self._encoder = encoder
        self._workspace_root = Path(workspace_root)
        self._clock = clock

    def render(self, record: BarLyricRecord, *, tempo_bpm: float) -> RenderedBarAudio:
        started = self._clock()
        request = bar_render_request(record, tempo_bpm)
        workspace = self._workspace_root / record.session_id / f"bar{record.bar:05d}-{record.lyric_id[:16]}"
        source = self._renderer.render_source(request, workspace)
        packaging_started = self._clock()
        manifest = BarAudioManifest(
            lyric_id=record.lyric_id,
            bar=record.bar,
            tempo_bpm=tempo_bpm,
            frame_count=int(source.audio_diagnostics["frame_count"]),
            source_sha256=hashlib.sha256(source.source_wav).hexdigest(),
            source_onsets=tuple(source.alignment_diagnostics["source_onsets"]),
            onset_confidence=tuple(source.alignment_diagnostics["onset_confidence"]),
            stage_timings_ms={
                **{name: float(value) for name, value in source.stage_timings_ms.items()},
                "render_total": max(0.0, (packaging_started - started) * 1000.0),
            },
            diagnostics={
                "fallback_counts": source.alignment_diagnostics.get("fallback_counts", {}),
                "warnings": list(source.warnings),
                "model_tool_versions": dict(source.model_tool_versions),
            },
        )
        package = self._encoder.encode(manifest, source.source_wav)
        return RenderedBarAudio(
            lyric_id=record.lyric_id,
            bar=record.bar,
            revision=record.revision,
            package=package,
            media_type=self._encoder.media_type,
            manifest=manifest,
            render_ms=max(0.0, (self._clock() - started) * 1000.0),
        )


@dataclass(frozen=True)
class PreparedBarVocal:
    """Client side: a bar's source warped onto its beat (24 kHz PCM16 WAV)."""

    vocal_wav: bytes
    warp_ms: float
