"""Mac-side validation, conversion, and mixing for remote MOSS chunks."""

from __future__ import annotations

import hashlib
import io
from math import gcd, isclose, isfinite
from time import monotonic
from typing import TYPE_CHECKING, Callable, Mapping, Protocol
import wave

import numpy as np
from scipy.signal import resample_poly

from streammuse.application.rap.alignment import align_exact
from streammuse.application.rap.audio_rendering import bar_frame_count, limit_peak, mix_at, tick_frame_in_bar
from streammuse.application.rap.audio_service import DrumRenderer, RapChunkPreparationStrategy
from streammuse.application.rap.monitoring_payloads import (
    bounded_chunk_event_payload,
    remote_generation_input_summary,
)
from streammuse.application.rap.service import ProsodyAnalyzer
from streammuse.domain.rap.audio import (
    AudioFormat,
    AudioWarning,
    AudioWarningCode,
    AudioWarningSeverity,
    PcmAudio,
    PreparedRapBar,
    SyllablePlacementDiagnostic,
)
from streammuse.domain.rap.flow import materialize_flow
from streammuse.domain.rap.remote_chunk import PreparedRapChunk, RemoteRapChunkManifest, RemoteRapChunkRequest
from streammuse.domain.timing import Tempo
from streammuse.infrastructure.rap.chunk_package import DecodedRapChunkPackage

if TYPE_CHECKING:
    from streammuse.infrastructure.rap.remote_chunk_client import RemoteChunkResponse


_VOCAL_GAIN = 0.80
_DRUM_GAIN = 0.55
_FINAL_PEAK = 0.95
_REMOTE_SAMPLE_RATE_HZ = 24_000


class RemoteChunkPreparationError(RuntimeError):
    """A remote chunk cannot safely replace the already-prepared fallback."""


class _RemoteChunkTransport(Protocol):
    def prepare(
        self,
        request: RemoteRapChunkRequest,
        timeout_seconds: float,
        *,
        deadline_monotonic: float | None = None,
    ) -> RemoteChunkResponse: ...

    def abort(self) -> None: ...

    def close(self) -> None: ...


class RemoteMossChunkPreparationStrategy(RapChunkPreparationStrategy):
    """Accept a remote package only after the Mac revalidates and mixes it."""

    def __init__(
        self,
        *,
        client: _RemoteChunkTransport,
        audio_format: AudioFormat,
        drums: DrumRenderer,
        prosody: ProsodyAnalyzer,
        tempo: Tempo | None = None,
        tempo_bpm: float | None = None,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        if tempo is None:
            if tempo_bpm is None or not isinstance(tempo_bpm, (int, float)) or not isfinite(tempo_bpm) or tempo_bpm <= 0:
                raise ValueError("tempo or a positive finite tempo_bpm is required")
            tempo = Tempo(float(tempo_bpm), ticks_per_beat=4, beats_per_bar=4)
        elif tempo_bpm is not None and tempo.bpm != tempo_bpm:
            raise ValueError("tempo and tempo_bpm must agree")
        if tempo.ticks_per_beat != 4 or tempo.beats_per_bar != 4:
            raise ValueError("remote MOSS chunks require 4/4 timing with four ticks per beat")
        if audio_format != AudioFormat():
            raise ValueError("remote MOSS chunks require 48 kHz stereo float32 output")
        self._client = client
        self._tempo = tempo
        self._audio_format = audio_format
        self._drums = drums
        self._prosody = prosody
        self._clock = clock
        self._closed = False

    def prepare(self, request: RemoteRapChunkRequest, *, deadline_monotonic: float) -> PreparedRapChunk:
        if self._closed:
            raise RemoteChunkPreparationError("remote MOSS preparation strategy is closed")
        if not isinstance(request, RemoteRapChunkRequest):
            raise ValueError("request must be a RemoteRapChunkRequest")
        if not isinstance(deadline_monotonic, (int, float)) or not isfinite(deadline_monotonic):
            raise ValueError("deadline_monotonic must be finite")
        now = self._clock()
        if now >= deadline_monotonic:
            raise RemoteChunkPreparationError("remote chunk deadline elapsed before request")
        if request.tempo_bpm != self._tempo.bpm:
            raise RemoteChunkPreparationError("request tempo does not match the Mac timing authority")
        try:
            response = self._client.prepare(
                request,
                deadline_monotonic - now,
                deadline_monotonic=deadline_monotonic,
            )
        except RemoteChunkPreparationError:
            raise
        except Exception as error:
            raise RemoteChunkPreparationError("remote chunk transport failed") from error
        mac_started = self._clock()
        if mac_started >= deadline_monotonic:
            raise RemoteChunkPreparationError("remote chunk arrived after its useful deadline")
        try:
            self._validate_manifest(response.package, request)
            vocal_samples = _decode_pcm16_mono_wav(response.package.vocal_wav, request.expected_frame_count)
            frame_counts = tuple(bar_frame_count(item.bar, self._tempo, self._audio_format) for item in request.bars)
            full_vocals = _resample_to_output(vocal_samples, self._audio_format, sum(frame_counts))
            observed_latency_ms = response.timing.request_ms + response.timing.first_byte_ms + response.timing.download_ms
            bars = self._prepare_bars(
                response.package.manifest,
                request,
                full_vocals,
                frame_counts,
                observed_latency_ms,
            )
        except RemoteChunkPreparationError:
            raise
        except (EOFError, ValueError, wave.Error) as error:
            raise RemoteChunkPreparationError("remote chunk package failed Mac validation") from error
        mac_completed = self._clock()
        if mac_completed >= deadline_monotonic:
            raise RemoteChunkPreparationError("remote chunk preparation missed its useful deadline")
        prepared = PreparedRapChunk(
            request_id=request.request_id,
            chunk_index=request.chunk_index,
            renderer="moss_aligned_remote",
            bars=bars,
            diagnostics=self._monitoring_evidence(
                request,
                response,
                mac_validation_mix_ms=max(0.0, (mac_completed - mac_started) * 1000.0),
            ),
        )
        if self._clock() >= deadline_monotonic:
            raise RemoteChunkPreparationError("remote chunk preparation missed its useful deadline")
        return prepared

    @staticmethod
    def _monitoring_evidence(
        request: RemoteRapChunkRequest,
        response: RemoteChunkResponse,
        *,
        mac_validation_mix_ms: float,
    ) -> dict[str, object]:
        manifest = response.package.manifest
        diagnostics = manifest.diagnostics
        transfer_ms = (
            response.timing.request_ms
            + response.timing.first_byte_ms
            + response.timing.download_ms
        )
        stage_timings = diagnostics.stage_timings_ms
        monitoring = diagnostics.monitoring_summary
        assert isinstance(monitoring, Mapping)
        selected_schedules = [
            ", ".join(
                f"t{item.slot.tick - selected.bar * 16}:{item.syllable.word}/stress{item.syllable.stress}"
                for item in selected.scheduled
            )
            for selected in manifest.selected_bars
        ]
        raw = {
            "chunk_index": request.chunk_index,
            "bars": [item.bar for item in request.bars],
            "selected_lines": [item.text for item in manifest.selected_bars],
            "flows": [
                {
                    "template_id": item.flow_template.template_id,
                    "name": item.flow_template.name,
                    "ticks_per_beat": item.flow_template.ticks_per_beat,
                    "beats_per_bar": item.flow_template.beats_per_bar,
                    "slots": [
                        {
                            "tick_in_bar": slot.tick_in_bar,
                            "duration_ticks": slot.duration_ticks,
                            "target_stress": slot.target_stress,
                            "boundary_strength": slot.boundary_strength,
                            "rhyme_group": slot.rhyme_group,
                        }
                        for slot in item.flow_template.slots
                    ],
                }
                for item in request.bars
            ],
            "selected_schedules": selected_schedules,
            "candidate_counts": {
                "requested": diagnostics.candidate_stats.requested_count,
                "parseable": diagnostics.candidate_stats.parseable_count,
                "valid": diagnostics.candidate_stats.valid_count,
                "selectable": diagnostics.candidate_stats.selectable_count,
            },
            "selected_scores": [
                {
                    "bar": item.bar,
                    "total": item.score,
                    "component_scores": item.diagnostics.get("component_scores", {}),
                }
                for item in manifest.selected_bars
            ],
            "prompt_summary": remote_generation_input_summary(request),
            "context_lines": request.context_lines,
            "stage_timings_ms": {
                "generation": stage_timings["generation"],
                "evaluation": stage_timings["evaluation"],
                "moss": stage_timings["moss"],
                "aligner": stage_timings["aligner"],
                "r3": stage_timings["warp"],
                "package": stage_timings["packaging"],
                "transfer": transfer_ms,
                "mac": mac_validation_mix_ms,
                "total": transfer_ms + mac_validation_mix_ms,
            },
            "request_budget_ms": diagnostics.accepted_request_budget_ms,
            "elapsed_ms": transfer_ms + mac_validation_mix_ms,
            "alignment": {
                "method": monitoring.get("alignment_method"),
                "confidence": monitoring.get("alignment_confidence"),
                "fallback_counts": diagnostics.alignment_diagnostics["fallback_counts"],
            },
            "warnings": diagnostics.warnings,
            "hashes": {
                "request_sha256": hashlib.sha256(request.canonical_json_bytes()).hexdigest(),
                "manifest_sha256": hashlib.sha256(manifest.canonical_json_bytes()).hexdigest(),
                "source_wav_sha256": monitoring.get("source_wav_sha256"),
                "vocal_sha256": manifest.vocal_sha256,
            },
            "artifact_refs": monitoring.get("artifact_ids"),
            "transfer": {
                "total_ms": transfer_ms,
                "response_bytes": response.timing.response_bytes,
            },
        }
        bounded = bounded_chunk_event_payload(raw)
        return {
            **bounded,
            "transfer": {
                "request_ms": response.timing.request_ms,
                "first_byte_ms": response.timing.first_byte_ms,
                "download_ms": response.timing.download_ms,
                "response_bytes": response.timing.response_bytes,
                "attempts": response.timing.attempts,
            },
        }

    def abort(self) -> None:
        if not self._closed:
            self._client.abort()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._client.close()

    def _validate_manifest(self, package: DecodedRapChunkPackage, request: RemoteRapChunkRequest) -> None:
        manifest = package.manifest
        if not isinstance(manifest, RemoteRapChunkManifest):
            raise RemoteChunkPreparationError("remote chunk manifest is invalid")
        if (
            manifest.request_id != request.request_id
            or manifest.chunk_index != request.chunk_index
            or manifest.tempo_bpm != request.tempo_bpm
            or manifest.output_sample_rate_hz != request.output_sample_rate_hz
            or manifest.expected_frame_count != request.expected_frame_count
        ):
            raise RemoteChunkPreparationError("remote chunk manifest does not match the original request")
        if (
            manifest.diagnostics.accepted_request_budget_ms != request.remaining_budget_ms
            or manifest.diagnostics.resolved_policy != request.policy
        ):
            raise RemoteChunkPreparationError("remote chunk diagnostics do not match the original request")
        if hashlib.sha256(package.vocal_wav).hexdigest() != manifest.vocal_sha256:
            raise RemoteChunkPreparationError("remote chunk vocal hash does not match its manifest")
        diagnostics = manifest.diagnostics.audio_diagnostics
        if (
            diagnostics["sample_rate_hz"] != _REMOTE_SAMPLE_RATE_HZ
            or diagnostics["frame_count"] != request.expected_frame_count
            or not isclose(float(diagnostics["duration_seconds"]), request.expected_frame_count / _REMOTE_SAMPLE_RATE_HZ, abs_tol=1 / _REMOTE_SAMPLE_RATE_HZ)
        ):
            raise RemoteChunkPreparationError("remote chunk audio diagnostics do not match the original request")
        all_scheduled = []
        for selected, bar_request in zip(manifest.selected_bars, request.bars, strict=True):
            if selected.bar != bar_request.bar or selected.flow_template_id != bar_request.flow_template.template_id:
                raise RemoteChunkPreparationError("remote selected bar does not match the original request")
            try:
                expected = align_exact(self._prosody.analyze(selected.text), materialize_flow(bar_request.flow_template, bar_request.bar))
            except ValueError as error:
                raise RemoteChunkPreparationError("remote selected text does not fit the original flow") from error
            if selected.scheduled != expected:
                raise RemoteChunkPreparationError("remote selected schedule does not match Mac reanalysis")
            all_scheduled.extend(selected.scheduled)
        alignment = manifest.diagnostics.alignment_diagnostics
        if len(alignment["source_anchors"]) != len(all_scheduled) or len(alignment["target_anchors"]) != len(all_scheduled):
            raise RemoteChunkPreparationError("remote alignment anchors do not cover every selected syllable")
        duration_seconds = request.expected_frame_count / _REMOTE_SAMPLE_RATE_HZ
        for anchors in (alignment["source_anchors"], alignment["target_anchors"]):
            values = tuple(float(item) for item in anchors)
            frame_values = tuple(round(value * _REMOTE_SAMPLE_RATE_HZ) for value in values)
            if any(not 0.0 <= value <= duration_seconds for value in values) or any(
                current <= previous for previous, current in zip(values, values[1:])
            ) or any(
                current <= previous for previous, current in zip(frame_values, frame_values[1:])
            ):
                raise RemoteChunkPreparationError("remote alignment anchors are not monotonic within the chunk")
        returned_target_frames = tuple(round(float(item) * _REMOTE_SAMPLE_RATE_HZ) for item in alignment["target_anchors"])
        expected_target_frames = _mac_target_anchor_frames(request, self._tempo, self._audio_format)
        if returned_target_frames != expected_target_frames:
            raise RemoteChunkPreparationError("remote target anchors do not match the Mac-selected syllable schedule")

    def _prepare_bars(
        self,
        manifest: RemoteRapChunkManifest,
        request: RemoteRapChunkRequest,
        vocals: np.ndarray,
        frame_counts: tuple[int, int],
        observed_latency_ms: float,
    ) -> tuple[PreparedRapBar, PreparedRapBar]:
        if vocals.shape != (sum(frame_counts), self._audio_format.channels):
            raise RemoteChunkPreparationError("resampled remote vocals do not cover both Mac bars exactly")
        result: list[PreparedRapBar] = []
        start = 0
        anchor_offset = 0
        for selected, bar_request, frames in zip(manifest.selected_bars, request.bars, frame_counts, strict=True):
            end = start + frames
            vocal_bar = vocals[start:end]
            drum = self._drums.render(bar_request.flow_template, self._tempo, self._audio_format, bar_request.bar)
            if drum.format != self._audio_format or drum.frame_count != frames:
                raise RemoteChunkPreparationError("local drum render does not match the exact Mac bar format")
            mixed = vocal_bar * np.float32(_VOCAL_GAIN)
            mix_at(mixed, drum, 0, _DRUM_GAIN)
            diagnostics = self._placement_diagnostics(manifest, selected.scheduled, bar_request.bar, frames, anchor_offset)
            warnings = tuple(
                AudioWarning(
                    code=AudioWarningCode.PRONUNCIATION_FALLBACK,
                    severity=AudioWarningSeverity.WARNING,
                    message=message,
                    bar=bar_request.bar,
                    action="remote_manifest_warning",
                )
                for message in manifest.diagnostics.warnings
            )
            result.append(
                PreparedRapBar(
                    bar=bar_request.bar,
                    text=selected.text,
                    source="moss_aligned_remote",
                    fallback_reason=None,
                    scheduled=selected.scheduled,
                    audio=PcmAudio(self._audio_format, frames, limit_peak(mixed, _FINAL_PEAK).tobytes()),
                    diagnostics=diagnostics,
                    warnings=warnings,
                    render_latency_ms=observed_latency_ms,
                )
            )
            start = end
            anchor_offset += len(selected.scheduled)
        return (result[0], result[1])

    def _placement_diagnostics(self, manifest, scheduled, bar: int, frames: int, offset: int) -> tuple[SyllablePlacementDiagnostic, ...]:
        alignment = manifest.diagnostics.alignment_diagnostics
        source_anchors = tuple(float(item) for item in alignment["source_anchors"])
        target_anchors = tuple(float(item) for item in alignment["target_anchors"])
        diagnostics = []
        for index, item in enumerate(scheduled):
            absolute_index = offset + index
            next_index = absolute_index + 1
            source_start = round(source_anchors[absolute_index] * _REMOTE_SAMPLE_RATE_HZ)
            source_end = round((source_anchors[next_index] if next_index < len(source_anchors) else manifest.expected_frame_count / _REMOTE_SAMPLE_RATE_HZ) * _REMOTE_SAMPLE_RATE_HZ)
            fitted_start = round(target_anchors[absolute_index] * self._audio_format.sample_rate_hz)
            fitted_end = round((target_anchors[next_index] if next_index < len(target_anchors) else manifest.expected_frame_count / _REMOTE_SAMPLE_RATE_HZ) * self._audio_format.sample_rate_hz)
            target_sample = tick_frame_in_bar(bar, item.slot.tick - bar * self._tempo.ticks_per_bar, self._tempo, self._audio_format)
            next_target = (
                tick_frame_in_bar(bar, scheduled[index + 1].slot.tick - bar * self._tempo.ticks_per_bar, self._tempo, self._audio_format)
                if index + 1 < len(scheduled)
                else frames
            )
            source_frames = max(0, source_end - source_start)
            fitted_frames = max(0, fitted_end - fitted_start)
            available_frames = next_target - target_sample
            diagnostics.append(
                SyllablePlacementDiagnostic(
                    bar=bar,
                    slot_index=item.slot.slot_index,
                    word=item.syllable.word,
                    target_sample=target_sample,
                    source_frames=source_frames,
                    fitted_frames=fitted_frames,
                    available_frames=available_frames,
                    compression_ratio=source_frames / fitted_frames if fitted_frames else 1.0,
                    overlap_frames=max(0, fitted_frames - available_frames),
                    pronunciation_source="moss_aligned_remote",
                    renderer_phonemes=item.syllable.phonemes,
                    rendered_frames=fitted_frames,
                )
            )
        return tuple(diagnostics)


def _decode_pcm16_mono_wav(vocal_wav: bytes, expected_frames: int) -> np.ndarray:
    try:
        with wave.open(io.BytesIO(vocal_wav), "rb") as wav:
            if wav.getcomptype() != "NONE" or wav.getsampwidth() != 2 or wav.getnchannels() != 1 or wav.getframerate() != _REMOTE_SAMPLE_RATE_HZ:
                raise RemoteChunkPreparationError("remote vocals are not 24 kHz mono PCM16")
            if wav.getnframes() != expected_frames:
                raise RemoteChunkPreparationError("remote vocals do not have the requested exact duration")
            data = wav.readframes(expected_frames)
    except (EOFError, wave.Error) as error:
        raise RemoteChunkPreparationError("remote vocals WAV is invalid") from error
    if len(data) != expected_frames * 2:
        raise RemoteChunkPreparationError("remote vocals WAV is truncated")
    return np.frombuffer(data, dtype="<i2").astype(np.float32) / np.float32(32768.0)


def _mac_target_anchor_frames(
    request: RemoteRapChunkRequest,
    tempo: Tempo,
    audio_format: AudioFormat,
) -> tuple[int, ...]:
    chunk_frame = 0
    targets = []
    for bar_request in request.bars:
        for slot in materialize_flow(bar_request.flow_template, bar_request.bar):
            local_frame = tick_frame_in_bar(
                bar_request.bar,
                slot.tick - bar_request.bar * tempo.ticks_per_bar,
                tempo,
                audio_format,
            )
            targets.append(round((chunk_frame + local_frame) * _REMOTE_SAMPLE_RATE_HZ / audio_format.sample_rate_hz))
        chunk_frame += bar_frame_count(bar_request.bar, tempo, audio_format)
    return tuple(targets)


def _resample_to_output(samples: np.ndarray, audio_format: AudioFormat, target_frames: int) -> np.ndarray:
    divisor = gcd(samples.shape[0], target_frames)
    resampled = resample_poly(
        samples,
        target_frames // divisor,
        samples.shape[0] // divisor,
    ).astype(np.float32)
    endpoint_delta = target_frames - resampled.shape[0]
    if abs(endpoint_delta) > 1:
        raise RemoteChunkPreparationError("remote resampling is inconsistent with the exact Mac frame grid")
    if endpoint_delta < 0:
        resampled = resampled[:target_frames]
    elif endpoint_delta > 0:
        resampled = np.pad(resampled, (0, endpoint_delta), mode="edge")
    if resampled.shape[0] != target_frames:
        raise RemoteChunkPreparationError("remote resampling did not produce the exact Mac frame count")
    return np.repeat(resampled[:, np.newaxis], audio_format.channels, axis=1)
