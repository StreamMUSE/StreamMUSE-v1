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

from streammuse.application.rap.audio_rendering import bar_frame_count, tick_frame_in_bar
from streammuse.application.rap.chunk_audio import (
    LocalPhraseWarp,
    _decode_pcm16_mono_wav,
    _resample_to_output,
    mix_audio_stems,
    remix_prepared_bar,
)
from streammuse.application.rap.chunk_orchestration import PhraseSourceResult
from streammuse.application.rap.lyric_session import RenderedBarAudio
from streammuse.domain.rap import RemoteRapChunkRequest
from streammuse.domain.rap.audio import (
    PcmAudio, PreparedRapBar, RapMixConfig, RapWarpDiagnostics,
    RenderedOnsetEstimate, SpeechPrefix,
)
from streammuse.domain.rap.lyric_stream import BarAudioManifest, BarLyricRecord
from streammuse.domain.rap.timeline import BarTimingSnapshot
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
        self._sources: OrderedDict[tuple[str, float], SpeechPrefix] = OrderedDict()
        self._sources_lock = threading.Lock()
        self._clock = clock

    def render(
        self, record: BarLyricRecord, *, tempo_bpm: float, previous: BarLyricRecord | None = None
    ) -> RenderedBarAudio:
        started = self._clock()
        request = bar_render_request(record, tempo_bpm)
        workspace = self._workspace_root / record.session_id / f"bar{record.bar:05d}-{record.lyric_id[:16]}-bpm{float(tempo_bpm).hex()}"
        prefix, continuation = self._prefix_for(previous, tempo_bpm)
        source = (
            self._renderer.render_source(request, workspace)
            if prefix is None
            else self._renderer.render_source(request, workspace, prefix=prefix)
        )
        with self._sources_lock:
            self._sources[(record.lyric_id, tempo_bpm)] = SpeechPrefix(record.text, source.source_wav)
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


    def _prefix_for(self, previous: BarLyricRecord | None, tempo_bpm: float) -> tuple[SpeechPrefix | None, dict[str, object]]:
        if not self._continuation:
            return None, {"used": False, "reason": "disabled"}
        if previous is None:
            return None, {"used": False, "reason": "no_previous_line"}
        with self._sources_lock:
            prefix = self._sources.get((previous.lyric_id, tempo_bpm))
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

    def __init__(self, *, warper, drums, tempo, audio_format, mix_config: RapMixConfig | None = None) -> None:
        self._warper = warper
        self._drums = drums
        self._tempo = tempo
        self._audio_format = audio_format
        self.set_mix_config(mix_config or RapMixConfig())

    def set_mix_config(self, config: RapMixConfig) -> None:
        if not isinstance(config, RapMixConfig):
            raise ValueError("mix config must be a RapMixConfig")
        self._mix_config = config

    def with_alignment_policy(self, policy: str) -> BarAudioPreparer:
        """Return a policy snapshot without changing any in-flight preparation."""
        switch = getattr(self._warper, "with_policy", None)
        if not callable(switch):
            raise ValueError("warper does not support alignment policy changes")
        return BarAudioPreparer(
            warper=switch(policy), drums=self._drums, tempo=self._tempo,
            audio_format=self._audio_format, mix_config=self._mix_config,
        )

    def remix(self, prepared: PreparedRapBar, config: RapMixConfig | None = None) -> PreparedRapBar:
        return remix_prepared_bar(prepared, config or self._mix_config)

    def prepare(
        self, record: BarLyricRecord, manifest: BarAudioManifest, source_wav: bytes,
        *, timing: BarTimingSnapshot | None = None,
    ) -> PreparedRapBar:
        if manifest.lyric_id != record.lyric_id or manifest.bar != record.bar:
            raise ValueError("bar audio does not belong to this lyric record")
        mix_config = self._mix_config
        tempo = timing or self._tempo
        frames = bar_frame_count(record.bar, tempo, self._audio_format)
        if manifest.tempo_bpm != tempo.bpm:
            raise ValueError("source tempo does not match destination tempo")
        if len(manifest.source_onsets) != len(record.scheduled):
            raise ValueError("source onsets must cover every scheduled syllable")
        target_frames = RemoteRapChunkRequest.frame_count_for(tempo.bpm, bars=1)
        warp = self._warper.warp(
            bar_render_request(record, tempo.bpm),
            source_wav,
            manifest.source_onsets,
            target_frame_count=target_frames,
        )
        vocals = _decode_pcm16_mono_wav(warp.vocal_wav, target_frames)
        vocal_samples = _resample_to_output(vocals, self._audio_format, frames)
        first_tick = record.scheduled[0].slot.tick - record.bar * 16
        first_sample = tick_frame_in_bar(record.bar, first_tick, tempo, self._audio_format)
        # Keep resampling-filter pre-ringing out of the explicit leading rest.
        vocal_samples[:first_sample] = 0.0
        vocal_stem = PcmAudio(self._audio_format, frames, vocal_samples.tobytes())
        drum = self._drums.render(record.flow_template, tempo, self._audio_format, record.bar)
        if drum.format != self._audio_format or drum.frame_count != frames:
            raise ValueError("local drum render does not match the exact bar format")
        warp_diagnostics = None
        if isinstance(warp, LocalPhraseWarp):
            if len(warp.target_anchors) != len(record.scheduled) or len(warp.source_anchors) != len(record.scheduled):
                raise ValueError("warp anchors must cover every scheduled syllable")
            warp_diagnostics = RapWarpDiagnostics(
                manifest.source_onsets, warp.source_anchors, warp.target_anchors,
                warp.local_warp_ratios, warp.policy, warp.warp_ms, warp.r3_attempt_deviations,
            )
        estimates = tuple(
            RenderedOnsetEstimate(
                slot_index=item.slot.slot_index,
                requested_sample=tick_frame_in_bar(record.bar, item.slot.tick - record.bar * 16, tempo, self._audio_format),
                estimated_sample=(warp.target_anchors[index] * self._audio_format.sample_rate_hz if warp_diagnostics else None),
                source_seconds=manifest.source_onsets[index],
                confidence=manifest.onset_confidence[index],
                method="source_alignment_warp_estimate" if warp_diagnostics else "unavailable",
                unavailable_reason=None if warp_diagnostics else "warper did not return applied time-map evidence",
            )
            for index, item in enumerate(record.scheduled)
        )
        prepared = PreparedRapBar(
            bar=record.bar,
            text=record.text,
            source="moss_lyric_stream",
            fallback_reason=None,
            scheduled=record.scheduled,
            audio=mix_audio_stems(vocal_stem, drum, mix_config),
            diagnostics=(),
            warnings=(),
            render_latency_ms=warp.warp_ms,
            vocal_stem=vocal_stem,
            drum_stem=drum,
            mix_config=mix_config,
            onset_estimates=estimates,
            warp_diagnostics=warp_diagnostics,
            timing=timing,
        )
        return remix_prepared_bar(prepared, mix_config)
