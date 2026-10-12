"""Protocol v3 per-bar audio: the shared render request, server render and client preparation.

``bar_render_request`` is used by both sides: the server renders the MOSS line
with it, and the client warps the returned source onto the same target times,
so both compute bit-identical targets (as ``chunk_render_request`` does for v2).
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
import hashlib
from pathlib import Path
import threading
import time
from typing import Protocol

import numpy as np

from streammuse.application.rap.audio_rendering import bar_frame_count, limit_peak, mix_at
from streammuse.application.rap.chunk_audio import (
    _DRUM_GAIN,
    _FINAL_PEAK,
    _VOCAL_GAIN,
    _decode_pcm16_mono_wav,
    _resample_to_output,
)
from streammuse.application.rap.chunk_orchestration import PhraseSourceResult
from streammuse.application.rap.lyric_session import RenderedBarAudio
from streammuse.domain.rap import RemoteRapChunkRequest
from streammuse.domain.rap.audio import PcmAudio, PreparedRapBar, SpeechPrefix
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
    def render_source(
        self, request: TwoBarRenderRequest, workspace: Path, *, prefix: SpeechPrefix | None = None
    ) -> PhraseSourceResult: ...


class BarPackageEncoder(Protocol):
    media_type: str

    def encode(self, manifest: BarAudioManifest, source_wav: bytes) -> bytes: ...


class MossBarAudioRenderer:
    """Server side: one MOSS line plus MMS onsets, packaged for one lyric record.

    With ``continuation``, a bar is synthesized as the continuation of the
    previous bar's latest line (its MOSS source, kept here for the last
    ``source_cache_size`` renders); without that source it is synthesized alone.
    """

    def __init__(
        self,
        renderer: SourcePhraseRenderer,
        encoder: BarPackageEncoder,
        *,
        workspace_root: Path,
        continuation: bool = True,
        source_cache_size: int = 64,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._renderer = renderer
        self._encoder = encoder
        self._workspace_root = Path(workspace_root)
        self._continuation = continuation
        self._source_cache_size = source_cache_size
        self._sources: OrderedDict[str, SpeechPrefix] = OrderedDict()
        self._sources_lock = threading.Lock()
        self._clock = clock

    def render(
        self, record: BarLyricRecord, *, tempo_bpm: float, previous: BarLyricRecord | None = None
    ) -> RenderedBarAudio:
        started = self._clock()
        request = bar_render_request(record, tempo_bpm)
        workspace = self._workspace_root / record.session_id / f"bar{record.bar:05d}-{record.lyric_id[:16]}"
        prefix, continuation = self._prefix_for(previous)
        source = (
            self._renderer.render_source(request, workspace)
            if prefix is None
            else self._renderer.render_source(request, workspace, prefix=prefix)
        )
        with self._sources_lock:
            self._sources[record.lyric_id] = SpeechPrefix(record.text, source.source_wav)
            while len(self._sources) > self._source_cache_size:
                self._sources.popitem(last=False)
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
                "continuation": continuation,
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


    def _prefix_for(self, previous: BarLyricRecord | None) -> tuple[SpeechPrefix | None, dict[str, object]]:
        if not self._continuation:
            return None, {"used": False, "reason": "disabled"}
        if previous is None:
            return None, {"used": False, "reason": "no_previous_line"}
        with self._sources_lock:
            prefix = self._sources.get(previous.lyric_id)
        if prefix is None:
            return None, {"used": False, "reason": "previous_not_rendered", "prefix_lyric_id": previous.lyric_id}
        return prefix, {"used": True, "prefix_lyric_id": previous.lyric_id}


@dataclass(frozen=True)
class PreparedBarVocal:
    """Client side: a bar's source warped onto its beat (24 kHz PCM16 WAV)."""

    vocal_wav: bytes
    warp_ms: float


class BarAudioPreparer:
    """Client side: warp one bar's MOSS line onto its beat and mix the drums."""

    def __init__(self, *, warper, drums, tempo, audio_format) -> None:
        self._warper = warper
        self._drums = drums
        self._tempo = tempo
        self._audio_format = audio_format

    def prepare(self, record: BarLyricRecord, manifest: BarAudioManifest, source_wav: bytes) -> PreparedRapBar:
        if manifest.lyric_id != record.lyric_id or manifest.bar != record.bar:
            raise ValueError("bar audio does not belong to this lyric record")
        target_frames = RemoteRapChunkRequest.frame_count_for(self._tempo.bpm, bars=1)
        warp = self._warper.warp(
            bar_render_request(record, self._tempo.bpm),
            source_wav,
            manifest.source_onsets,
            target_frame_count=target_frames,
        )
        vocals = _decode_pcm16_mono_wav(warp.vocal_wav, target_frames)
        frames = bar_frame_count(record.bar, self._tempo, self._audio_format)
        mixed = _resample_to_output(vocals, self._audio_format, frames) * np.float32(_VOCAL_GAIN)
        drum = self._drums.render(record.flow_template, self._tempo, self._audio_format, record.bar)
        if drum.format != self._audio_format or drum.frame_count != frames:
            raise ValueError("local drum render does not match the exact bar format")
        mix_at(mixed, drum, 0, _DRUM_GAIN)
        return PreparedRapBar(
            bar=record.bar,
            text=record.text,
            source="moss_lyric_stream",
            fallback_reason=None,
            scheduled=record.scheduled,
            audio=PcmAudio(self._audio_format, frames, limit_peak(mixed, _FINAL_PEAK).tobytes()),
            diagnostics=(),
            warnings=(),
            render_latency_ms=warp.warp_ms,
        )
