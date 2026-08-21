from __future__ import annotations

import hashlib
import io
import struct
import wave
from dataclasses import dataclass, replace

import numpy as np
import pytest
from scipy.signal import resample_poly

from streammuse.application.rap.chunk_audio import (
    RemoteChunkPreparationError,
    RemoteMossChunkPreparationStrategy,
)
from streammuse.domain.rap import (
    AudioFormat,
    FlowProvenance,
    FlowSlot,
    FlowTemplate,
    PcmAudio,
    ProsodyAnalysis,
    RemoteCandidatePolicy,
    RemoteCandidateStats,
    RemoteRapBarRequest,
    RemoteRapChunkDiagnostics,
    RemoteRapChunkManifest,
    RemoteRapChunkRequest,
    RemoteSelectedBar,
    ScheduledSyllable,
    Syllable,
    materialize_flow,
)
from streammuse.infrastructure.rap.chunk_package import DecodedRapChunkPackage
from streammuse.infrastructure.rap.remote_chunk_client import (
    RemoteChunkResponse,
    RemoteChunkTransferTiming,
)


def _flow(template_id: str) -> FlowTemplate:
    return FlowTemplate(
        template_id=template_id,
        name=template_id,
        ticks_per_beat=4,
        beats_per_bar=4,
        slots=(
            FlowSlot(tick_in_bar=0, duration_ticks=4, target_stress=1.0),
            FlowSlot(tick_in_bar=8, duration_ticks=4, target_stress=0.5),
        ),
        provenance=FlowProvenance(kind="test", source="unit-test"),
    )


def _request() -> RemoteRapChunkRequest:
    return RemoteRapChunkRequest.create(
        session_id="session-1",
        chunk_index=0,
        bars=(
            RemoteRapBarRequest(0, "space", _flow("first-flow")),
            RemoteRapBarRequest(1, "space", _flow("second-flow")),
        ),
        tempo_bpm=90.0,
        remaining_budget_ms=5_000,
        policy=RemoteCandidatePolicy.realtime_default(),
        context_lines=(),
        seed=7,
    )


def _syllables() -> tuple[Syllable, Syllable]:
    return (
        Syllable("orbit", 0, 2, 1, ("AO1",), "test-prosody"),
        Syllable("orbit", 1, 2, 0, ("B", "IH0", "T"), "test-prosody"),
    )


def _wav_bytes(samples: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(24_000)
        wav.writeframes(samples.astype("<i2", copy=False).tobytes())
    return buffer.getvalue()


def _package(request: RemoteRapChunkRequest, samples: np.ndarray | None = None) -> DecodedRapChunkPackage:
    source = samples if samples is not None else np.full(request.expected_frame_count, 1_000, dtype=np.int16)
    vocal_wav = _wav_bytes(source)
    selected = tuple(
        RemoteSelectedBar.create(
            bar_request,
            text="orbit orbit",
            scheduled=tuple(
                ScheduledSyllable(slot, syllable)
                for slot, syllable in zip(materialize_flow(bar_request.flow_template, bar_request.bar), _syllables(), strict=True)
            ),
            score=0.9,
        )
        for bar_request in request.bars
    )
    manifest = RemoteRapChunkManifest(
        request_id=request.request_id,
        chunk_index=request.chunk_index,
        tempo_bpm=request.tempo_bpm,
        output_sample_rate_hz=24_000,
        expected_frame_count=request.expected_frame_count,
        selected_bars=selected,
        diagnostics=RemoteRapChunkDiagnostics(
            accepted_request_budget_ms=request.remaining_budget_ms,
            resolved_policy=request.policy,
            candidate_stats=RemoteCandidateStats(2, 2, 2, 2, (), ()),
            stage_timings_ms={
                "generation": 1.0,
                "evaluation": 1.0,
                "moss": 1.0,
                "aligner": 1.0,
                "warp": 1.0,
                "packaging": 1.0,
                "total": 6.0,
            },
            alignment_diagnostics={
                "fallback_counts": {"word": 0},
                "source_anchors": [0.0, 1.0, 2.0, 3.0],
                "target_anchors": [0.0, 1.0, 2.0, 3.0],
                "local_warp_ratios": [1.0],
            },
            audio_diagnostics={
                "sample_rate_hz": 24_000,
                "frame_count": request.expected_frame_count,
                "duration_seconds": request.expected_frame_count / 24_000,
                "peak": 0.5,
            },
            model_tool_versions={"moss": "test", "aligner": "test", "rubberband": "test"},
            warnings=(),
        ),
        vocal_sha256=hashlib.sha256(vocal_wav).hexdigest(),
    )
    return DecodedRapChunkPackage(manifest, vocal_wav)


@dataclass
class _FakeClient:
    package: DecodedRapChunkPackage
    calls: int = 0
    aborted: int = 0
    closed: int = 0

    def prepare(self, request, timeout_seconds, *, deadline_monotonic=None):
        self.calls += 1
        return RemoteChunkResponse(
            self.package,
            RemoteChunkTransferTiming(1.0, 2.0, 3.0, len(self.package.vocal_wav), 1),
        )

    def abort(self) -> None:
        self.aborted += 1

    def close(self) -> None:
        self.closed += 1


@dataclass
class _FakeProsody:
    calls: list[str]

    def analyze(self, text: str) -> ProsodyAnalysis:
        self.calls.append(text)
        return ProsodyAnalysis(text, text, _syllables(), (), (), (), ())


@dataclass
class _FakeDrums:
    calls: list[tuple[str, int]]
    level: float = 0.0

    def render(self, template, tempo, audio_format, bar):
        self.calls.append((template.template_id, bar))
        frames = round(tempo.tick_to_seconds(tempo.ticks_per_bar) * audio_format.sample_rate_hz)
        return PcmAudio(audio_format, frames, np.full((frames, 2), self.level, dtype=np.float32).tobytes())


def test_remote_chunk_splits_and_mixes_exact_bars() -> None:
    request = _request()
    source = ((np.arange(request.expected_frame_count) % 1_000) - 500).astype(np.int16)
    client = _FakeClient(_package(request, source))
    drums = _FakeDrums([])
    prosody = _FakeProsody([])
    strategy = RemoteMossChunkPreparationStrategy(
        client=client,
        tempo_bpm=90.0,
        audio_format=AudioFormat(),
        drums=drums,
        prosody=prosody,
        clock=lambda: 0.0,
    )

    prepared = strategy.prepare(request, deadline_monotonic=10.0)

    assert tuple(bar.bar for bar in prepared.bars) == (0, 1)
    assert all(bar.audio.frame_count == 128_000 for bar in prepared.bars)
    assert all(len(bar.audio.data) == 128_000 * 2 * 4 for bar in prepared.bars)
    assert prepared.renderer == "moss_aligned_remote"
    assert drums.calls == [("first-flow", 0), ("second-flow", 1)]
    assert prosody.calls == ["orbit orbit", "orbit orbit"]
    assert all(diagnostic.pronunciation_source == "moss_aligned_remote" for bar in prepared.bars for diagnostic in bar.diagnostics)
    expected = resample_poly(source.astype(np.float32) / 32768.0, up=2, down=1).astype(np.float32) * 0.8
    actual = np.concatenate(
        [np.frombuffer(bar.audio.data, dtype=np.float32).reshape(bar.audio.frame_count, 2)[:, 0] for bar in prepared.bars]
    )
    assert actual.shape == expected.shape
    assert np.allclose(actual, expected, atol=1e-6)


def test_remote_chunk_rejects_selected_schedule_mismatch_as_preparation_failure() -> None:
    request = _request()
    package = _package(request)
    original_schedule = package.manifest.selected_bars[0].scheduled
    changed_first = replace(original_schedule[0], syllable=replace(original_schedule[0].syllable, word="wrong"))
    bad_selected = RemoteSelectedBar(
        package.manifest.selected_bars[0].bar,
        package.manifest.selected_bars[0].text,
        package.manifest.selected_bars[0].flow_template_id,
        (changed_first, original_schedule[1]),
        package.manifest.selected_bars[0].score,
        package.manifest.selected_bars[0].diagnostics,
    )
    bad_manifest = RemoteRapChunkManifest(
        request_id=package.manifest.request_id,
        chunk_index=package.manifest.chunk_index,
        tempo_bpm=package.manifest.tempo_bpm,
        output_sample_rate_hz=package.manifest.output_sample_rate_hz,
        expected_frame_count=package.manifest.expected_frame_count,
        selected_bars=(bad_selected, package.manifest.selected_bars[1]),
        diagnostics=package.manifest.diagnostics,
        vocal_sha256=package.manifest.vocal_sha256,
    )
    strategy = RemoteMossChunkPreparationStrategy(
        client=_FakeClient(DecodedRapChunkPackage(bad_manifest, package.vocal_wav)),
        tempo_bpm=90.0,
        audio_format=AudioFormat(),
        drums=_FakeDrums([]),
        prosody=_FakeProsody([]),
        clock=lambda: 0.0,
    )

    with pytest.raises(RemoteChunkPreparationError, match="selected schedule"):
        strategy.prepare(request, deadline_monotonic=10.0)


def test_remote_chunk_peak_limits_the_local_mix() -> None:
    request = _request()
    strategy = RemoteMossChunkPreparationStrategy(
        client=_FakeClient(_package(request)),
        tempo_bpm=90.0,
        audio_format=AudioFormat(),
        drums=_FakeDrums([], level=2.0),
        prosody=_FakeProsody([]),
        clock=lambda: 0.0,
    )

    prepared = strategy.prepare(request, deadline_monotonic=10.0)

    peak = max(
        float(np.max(np.abs(np.frombuffer(bar.audio.data, dtype=np.float32))))
        for bar in prepared.bars
    )
    assert peak == pytest.approx(0.95)


def test_strategy_abort_is_reusable_and_close_is_final_and_idempotent() -> None:
    request = _request()
    client = _FakeClient(_package(request))
    strategy = RemoteMossChunkPreparationStrategy(
        client=client,
        tempo_bpm=90.0,
        audio_format=AudioFormat(),
        drums=_FakeDrums([]),
        prosody=_FakeProsody([]),
        clock=lambda: 0.0,
    )

    strategy.abort()
    strategy.prepare(request, deadline_monotonic=10.0)
    strategy.close()
    strategy.close()

    assert client.aborted == 1
    assert client.calls == 1
    assert client.closed == 1
    with pytest.raises(RemoteChunkPreparationError, match="closed"):
        strategy.prepare(request, deadline_monotonic=10.0)
