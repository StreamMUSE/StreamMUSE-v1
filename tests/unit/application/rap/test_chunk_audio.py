from __future__ import annotations

import hashlib
import io
from math import gcd
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
    REMOTE_CHUNK_ARTIFACT_IDS,
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


def _request(*, tempo_bpm: float = 90.0, first_bar: int = 0) -> RemoteRapChunkRequest:
    return RemoteRapChunkRequest.create(
        session_id="session-1",
        chunk_index=first_bar // 2,
        bars=(
            RemoteRapBarRequest(first_bar, "space", _flow("first-flow")),
            RemoteRapBarRequest(first_bar + 1, "space", _flow("second-flow")),
        ),
        tempo_bpm=tempo_bpm,
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


def _wav_bytes(
    samples: np.ndarray,
    *,
    channels: int = 1,
    sample_width: int = 2,
    sample_rate_hz: int = 24_000,
) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(sample_width)
        wav.setframerate(sample_rate_hz)
        if sample_width == 2:
            frames = np.repeat(samples[:, np.newaxis], channels, axis=1).astype("<i2", copy=False).tobytes()
        else:
            frames = bytes([128]) * samples.shape[0] * channels * sample_width
        wav.writeframes(frames)
    return buffer.getvalue()


def _target_anchors(request: RemoteRapChunkRequest) -> tuple[float, ...]:
    chunk_start_frame = round(request.bars[0].bar * 4 * 60 / request.tempo_bpm * 48_000)
    anchors = []
    for bar_request in request.bars:
        for slot in bar_request.flow_template.slots:
            absolute_tick = bar_request.bar * 16 + slot.tick_in_bar
            absolute_frame = round(absolute_tick * 60 / (request.tempo_bpm * 4) * 48_000)
            package_frame = round((absolute_frame - chunk_start_frame) * 24_000 / 48_000)
            anchors.append(package_frame / 24_000)
    return tuple(anchors)


def _package(
    request: RemoteRapChunkRequest,
    samples: np.ndarray | None = None,
    *,
    warnings: tuple[str, ...] = (),
) -> DecodedRapChunkPackage:
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
                "source_anchors": _target_anchors(request),
                "target_anchors": _target_anchors(request),
                "local_warp_ratios": [1.0],
            },
            audio_diagnostics={
                "sample_rate_hz": 24_000,
                "frame_count": request.expected_frame_count,
                "duration_seconds": request.expected_frame_count / 24_000,
                "peak": 0.5,
            },
            model_tool_versions={"moss": "test", "aligner": "test", "rubberband": "test"},
            warnings=warnings,
            monitoring_summary={
                "schema_version": "streammuse.rap_chunk_monitor.v1",
                "alignment_method": "torchaudio.pipelines.MMS_FA",
                "alignment_confidence": 0.93,
                "source_wav_sha256": "c" * 64,
                "artifact_ids": {
                    "request": "request.json",
                    "candidate_ledger": "candidate_ledger.json",
                    "source_wav": "source.wav",
                    "mms_alignment": "mms_alignment.json",
                    "alignment": "alignment.json",
                    "aligned_wav": "aligned.wav",
                    "vocal_wav": "vocal.wav",
                    "manifest": "manifest.json",
                    "server_timing": "server_timing.json",
                    "response_package": "response.zip",
                },
            },
        ),
        vocal_sha256=hashlib.sha256(vocal_wav).hexdigest(),
    )
    return DecodedRapChunkPackage(manifest, vocal_wav)


def _replace_alignment(
    package: DecodedRapChunkPackage,
    *,
    source_anchors: tuple[float, ...] | None = None,
    target_anchors: tuple[float, ...] | None = None,
) -> DecodedRapChunkPackage:
    diagnostics = package.manifest.diagnostics
    alignment = dict(diagnostics.alignment_diagnostics)
    if source_anchors is not None:
        alignment["source_anchors"] = source_anchors
    if target_anchors is not None:
        alignment["target_anchors"] = target_anchors
    changed_diagnostics = replace(diagnostics, alignment_diagnostics=alignment)
    return DecodedRapChunkPackage(
        replace(package.manifest, diagnostics=changed_diagnostics),
        package.vocal_wav,
    )


def _replace_wav(package: DecodedRapChunkPackage, vocal_wav: bytes) -> DecodedRapChunkPackage:
    return DecodedRapChunkPackage(
        replace(package.manifest, vocal_sha256=hashlib.sha256(vocal_wav).hexdigest()),
        vocal_wav,
    )


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
    syllables: tuple[Syllable, ...] = _syllables()

    def analyze(self, text: str) -> ProsodyAnalysis:
        self.calls.append(text)
        return ProsodyAnalysis(text, text, self.syllables, (), (), (), ())


@dataclass
class _FakeDrums:
    calls: list[tuple[str, int]]
    level: float = 0.0

    def render(self, template, tempo, audio_format, bar):
        self.calls.append((template.template_id, bar))
        bar_seconds = tempo.beats_per_bar * 60 / tempo.bpm
        frames = round((bar + 1) * bar_seconds * audio_format.sample_rate_hz) - round(
            bar * bar_seconds * audio_format.sample_rate_hz
        )
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


def test_remote_chunk_preparation_builds_truthful_end_to_end_monitoring_evidence() -> None:
    request = _request()
    package = _package(request, warnings=("wide local stretch ratio retained: 1.250",))
    strategy = RemoteMossChunkPreparationStrategy(
        client=_FakeClient(package),
        tempo_bpm=90.0,
        audio_format=AudioFormat(),
        drums=_FakeDrums([]),
        prosody=_FakeProsody([]),
        clock=lambda: 0.0,
    )

    prepared = strategy.prepare(request, deadline_monotonic=10.0)

    diagnostics = prepared.diagnostics
    assert diagnostics["candidate_counts"] == {
        "requested": 2,
        "parseable": 2,
        "valid": 2,
        "selectable": 2,
    }
    assert [
        {
            **dict(item),
            "component_scores": dict(item["component_scores"]),
        }
        for item in diagnostics["selected_scores"]
    ] == [
        {"bar": 0, "total": 0.9, "component_scores": {"total": 0.9}},
        {"bar": 1, "total": 0.9, "component_scores": {"total": 0.9}},
    ]
    assert diagnostics["stage_timings_ms"] == {
        "generation": 1.0,
        "evaluation": 1.0,
        "moss": 1.0,
        "aligner": 1.0,
        "r3": 1.0,
        "package": 1.0,
        "transfer": 6.0,
        "mac": 0.0,
        "total": 6.0,
    }
    assert diagnostics["request_budget_ms"] == request.remaining_budget_ms
    assert diagnostics["elapsed_ms"] == 6.0
    assert diagnostics["context_lines"] == ()
    assert "deterministic generation input summary; not a verbatim provider prompt" in diagnostics["prompt_summary"]
    assert "first-flow" in diagnostics["prompt_summary"]
    assert "second-flow" in diagnostics["prompt_summary"]
    assert diagnostics["alignment"] == {
        "method": "torchaudio.pipelines.MMS_FA",
        "confidence": 0.93,
        "fallback_counts": {"word": 0},
    }
    assert diagnostics["hashes"] == {
        "request_sha256": hashlib.sha256(request.canonical_json_bytes()).hexdigest(),
        "manifest_sha256": hashlib.sha256(package.manifest.canonical_json_bytes()).hexdigest(),
        "source_wav_sha256": "c" * 64,
        "vocal_sha256": package.manifest.vocal_sha256,
    }
    assert diagnostics["artifact_refs"]["manifest"] == "manifest.json"
    assert diagnostics["artifact_refs"] == dict(REMOTE_CHUNK_ARTIFACT_IDS)
    assert diagnostics["transfer"]["response_bytes"] == len(package.vocal_wav)
    encoded = repr(diagnostics)
    assert "source_anchors" not in encoded
    assert "target_anchors" not in encoded


@pytest.mark.parametrize(
    ("tempo_bpm", "first_bar", "expected_bar_frames"),
    (
        (71.0, 0, (162_254, 162_253)),
        (83.0, 1, (138_795, 138_796)),
    ),
)
def test_remote_chunk_preserves_exact_mac_frames_at_varied_tempo_and_bar_index(
    tempo_bpm: float,
    first_bar: int,
    expected_bar_frames: tuple[int, int],
) -> None:
    request = _request(tempo_bpm=tempo_bpm, first_bar=first_bar)
    source = ((np.arange(request.expected_frame_count) % 1_000) - 500).astype(np.int16)
    strategy = RemoteMossChunkPreparationStrategy(
        client=_FakeClient(_package(request, source)),
        tempo_bpm=tempo_bpm,
        audio_format=AudioFormat(),
        drums=_FakeDrums([]),
        prosody=_FakeProsody([]),
        clock=lambda: 0.0,
    )

    prepared = strategy.prepare(request, deadline_monotonic=10.0)

    assert tuple(bar.audio.frame_count for bar in prepared.bars) == expected_bar_frames
    target_frames = sum(expected_bar_frames)
    divisor = gcd(source.shape[0], target_frames)
    expected = resample_poly(
        source.astype(np.float32) / 32768.0,
        up=target_frames // divisor,
        down=source.shape[0] // divisor,
    ).astype(np.float32)
    if expected.shape[0] > target_frames:
        expected = expected[:target_frames]
    elif expected.shape[0] < target_frames:
        expected = np.pad(expected, (0, target_frames - expected.shape[0]), mode="edge")
    actual = np.concatenate(
        [np.frombuffer(bar.audio.data, dtype=np.float32).reshape(bar.audio.frame_count, 2)[:, 0] for bar in prepared.bars]
    )
    assert actual.shape == (target_frames,)
    assert np.allclose(actual, expected * 0.8, atol=1e-6)


def test_remote_chunk_rejects_near_equal_request_tempo_before_transport() -> None:
    request = _request(tempo_bpm=90.00000001)
    client = _FakeClient(_package(request))
    strategy = RemoteMossChunkPreparationStrategy(
        client=client,
        tempo_bpm=90.0,
        audio_format=AudioFormat(),
        drums=_FakeDrums([]),
        prosody=_FakeProsody([]),
        clock=lambda: 0.0,
    )

    with pytest.raises(RemoteChunkPreparationError, match="Mac timing authority"):
        strategy.prepare(request, deadline_monotonic=10.0)
    assert client.calls == 0


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

    with pytest.raises(RemoteChunkPreparationError, match="selected schedule") as caught:
        strategy.prepare(request, deadline_monotonic=10.0)

    assert type(caught.value).__name__ == "RemoteChunkResponseRejected"
    evidence = caught.value.evidence  # type: ignore[attr-defined]
    assert evidence.request_id == request.request_id
    assert evidence.chunk_index == request.chunk_index
    assert evidence.diagnostics["candidate_counts"] == {
        "requested": 2,
        "parseable": 2,
        "valid": 2,
        "selectable": 2,
    }
    assert evidence.diagnostics["artifact_refs"] == dict(REMOTE_CHUNK_ARTIFACT_IDS)
    encoded = repr(evidence.diagnostics)
    assert "source_anchors" not in encoded
    assert "target_anchors" not in encoded
    assert "RIFF" not in encoded


def test_remote_chunk_rejects_selected_text_that_disagrees_with_mac_reanalysis() -> None:
    request = _request()
    package = _package(request)
    selected = list(package.manifest.selected_bars)
    selected[0] = replace(selected[0], text="comet comet")
    package = DecodedRapChunkPackage(
        replace(package.manifest, selected_bars=tuple(selected)),
        package.vocal_wav,
    )
    comet_syllables = tuple(replace(item, word="comet") for item in _syllables())
    drums = _FakeDrums([])
    strategy = RemoteMossChunkPreparationStrategy(
        client=_FakeClient(package),
        tempo_bpm=90.0,
        audio_format=AudioFormat(),
        drums=drums,
        prosody=_FakeProsody([], comet_syllables),
        clock=lambda: 0.0,
    )

    with pytest.raises(RemoteChunkPreparationError, match="selected schedule"):
        strategy.prepare(request, deadline_monotonic=10.0)
    assert drums.calls == []


def test_remote_chunk_rejects_selected_text_with_wrong_syllable_count() -> None:
    request = _request()
    drums = _FakeDrums([])
    strategy = RemoteMossChunkPreparationStrategy(
        client=_FakeClient(_package(request)),
        tempo_bpm=90.0,
        audio_format=AudioFormat(),
        drums=drums,
        prosody=_FakeProsody([], _syllables()[:1]),
        clock=lambda: 0.0,
    )

    with pytest.raises(RemoteChunkPreparationError, match="does not fit the original flow"):
        strategy.prepare(request, deadline_monotonic=10.0)
    assert drums.calls == []


@pytest.mark.parametrize("anchor_index", (1, 2))
def test_remote_chunk_rejects_monotonic_target_anchor_that_misses_mac_schedule(anchor_index: int) -> None:
    request = _request()
    package = _package(request)
    changed = list(package.manifest.diagnostics.alignment_diagnostics["target_anchors"])
    changed[anchor_index] = float(changed[anchor_index]) + 1 / 24_000
    package = _replace_alignment(package, target_anchors=tuple(changed))
    strategy = RemoteMossChunkPreparationStrategy(
        client=_FakeClient(package),
        tempo_bpm=90.0,
        audio_format=AudioFormat(),
        drums=_FakeDrums([]),
        prosody=_FakeProsody([]),
        clock=lambda: 0.0,
    )

    with pytest.raises(RemoteChunkPreparationError, match="target anchors do not match"):
        strategy.prepare(request, deadline_monotonic=10.0)


@pytest.mark.parametrize("anchor_name", ("source", "target"))
def test_remote_chunk_rejects_duplicate_alignment_anchor(anchor_name: str) -> None:
    request = _request()
    package = _package(request)
    anchors = list(package.manifest.diagnostics.alignment_diagnostics[f"{anchor_name}_anchors"])
    anchors[1] = anchors[0]
    package = _replace_alignment(
        package,
        source_anchors=tuple(anchors) if anchor_name == "source" else None,
        target_anchors=tuple(anchors) if anchor_name == "target" else None,
    )
    strategy = RemoteMossChunkPreparationStrategy(
        client=_FakeClient(package),
        tempo_bpm=90.0,
        audio_format=AudioFormat(),
        drums=_FakeDrums([]),
        prosody=_FakeProsody([]),
        clock=lambda: 0.0,
    )

    with pytest.raises(RemoteChunkPreparationError, match="anchors are not monotonic"):
        strategy.prepare(request, deadline_monotonic=10.0)


def test_remote_chunk_rejects_distinct_source_anchors_on_the_same_pcm_frame() -> None:
    request = _request()
    package = _package(request)
    anchors = list(package.manifest.diagnostics.alignment_diagnostics["source_anchors"])
    anchors[1] = 0.25 / 24_000
    package = _replace_alignment(package, source_anchors=tuple(anchors))
    strategy = RemoteMossChunkPreparationStrategy(
        client=_FakeClient(package),
        tempo_bpm=90.0,
        audio_format=AudioFormat(),
        drums=_FakeDrums([]),
        prosody=_FakeProsody([]),
        clock=lambda: 0.0,
    )

    with pytest.raises(RemoteChunkPreparationError, match="anchors are not monotonic"):
        strategy.prepare(request, deadline_monotonic=10.0)


def test_remote_chunk_rejects_vocal_hash_mismatch() -> None:
    request = _request()
    package = _package(request)
    corrupt = DecodedRapChunkPackage(package.manifest, package.vocal_wav + b"corrupt")
    strategy = RemoteMossChunkPreparationStrategy(
        client=_FakeClient(corrupt),
        tempo_bpm=90.0,
        audio_format=AudioFormat(),
        drums=_FakeDrums([]),
        prosody=_FakeProsody([]),
        clock=lambda: 0.0,
    )

    with pytest.raises(RemoteChunkPreparationError, match="vocal hash"):
        strategy.prepare(request, deadline_monotonic=10.0)


def test_remote_chunk_accepts_lossy_opus_wav_without_canonical_wav_hash_match() -> None:
    request = _request()
    package = _package(request)
    lossy_wav = _wav_bytes(np.full(request.expected_frame_count, 999, dtype=np.int16))
    opus_package = DecodedRapChunkPackage(package.manifest, lossy_wav, "opus")
    strategy = RemoteMossChunkPreparationStrategy(
        client=_FakeClient(opus_package),
        tempo_bpm=90.0,
        audio_format=AudioFormat(),
        drums=_FakeDrums([]),
        prosody=_FakeProsody([]),
        clock=lambda: 0.0,
    )

    prepared = strategy.prepare(request, deadline_monotonic=10.0)

    assert prepared.renderer == "moss_aligned_remote"


@pytest.mark.parametrize(
    "manifest_change",
    (
        {"request_id": "0" * 64},
        {"chunk_index": 7},
        {"tempo_bpm": 90.00000001},
    ),
)
def test_remote_chunk_rejects_manifest_identity_field_mismatch(manifest_change: dict[str, object]) -> None:
    request = _request()
    package = _package(request)
    mismatch = DecodedRapChunkPackage(replace(package.manifest, **manifest_change), package.vocal_wav)
    strategy = RemoteMossChunkPreparationStrategy(
        client=_FakeClient(mismatch),
        tempo_bpm=90.0,
        audio_format=AudioFormat(),
        drums=_FakeDrums([]),
        prosody=_FakeProsody([]),
        clock=lambda: 0.0,
    )

    with pytest.raises(RemoteChunkPreparationError, match="manifest does not match"):
        strategy.prepare(request, deadline_monotonic=10.0)


@pytest.mark.parametrize(
    "diagnostics_change",
    (
        {"accepted_request_budget_ms": 4_999},
        {"resolved_policy": replace(RemoteCandidatePolicy.realtime_default(), minimum_score=0.5)},
    ),
)
def test_remote_chunk_rejects_request_diagnostics_mismatch(diagnostics_change: dict[str, object]) -> None:
    request = _request()
    package = _package(request)
    diagnostics = replace(package.manifest.diagnostics, **diagnostics_change)
    mismatch = DecodedRapChunkPackage(
        replace(package.manifest, diagnostics=diagnostics),
        package.vocal_wav,
    )
    strategy = RemoteMossChunkPreparationStrategy(
        client=_FakeClient(mismatch),
        tempo_bpm=90.0,
        audio_format=AudioFormat(),
        drums=_FakeDrums([]),
        prosody=_FakeProsody([]),
        clock=lambda: 0.0,
    )

    with pytest.raises(RemoteChunkPreparationError, match="diagnostics do not match"):
        strategy.prepare(request, deadline_monotonic=10.0)


def test_remote_chunk_rejects_manifest_tempo_and_frame_count_mismatch() -> None:
    request = _request()
    other_package = _package(_request(tempo_bpm=91.0))
    mismatch = DecodedRapChunkPackage(
        replace(
            other_package.manifest,
            request_id=request.request_id,
            chunk_index=request.chunk_index,
        ),
        other_package.vocal_wav,
    )
    strategy = RemoteMossChunkPreparationStrategy(
        client=_FakeClient(mismatch),
        tempo_bpm=90.0,
        audio_format=AudioFormat(),
        drums=_FakeDrums([]),
        prosody=_FakeProsody([]),
        clock=lambda: 0.0,
    )

    with pytest.raises(RemoteChunkPreparationError, match="manifest does not match"):
        strategy.prepare(request, deadline_monotonic=10.0)


@pytest.mark.parametrize(
    "wav_options",
    (
        {"channels": 2},
        {"sample_width": 1},
        {"sample_rate_hz": 22_050},
    ),
)
def test_remote_chunk_rejects_invalid_pcm_wav_format(wav_options: dict[str, int]) -> None:
    request = _request()
    package = _package(request)
    samples = np.full(request.expected_frame_count, 1_000, dtype=np.int16)
    package = _replace_wav(package, _wav_bytes(samples, **wav_options))
    strategy = RemoteMossChunkPreparationStrategy(
        client=_FakeClient(package),
        tempo_bpm=90.0,
        audio_format=AudioFormat(),
        drums=_FakeDrums([]),
        prosody=_FakeProsody([]),
        clock=lambda: 0.0,
    )

    with pytest.raises(RemoteChunkPreparationError, match="24 kHz mono PCM16"):
        strategy.prepare(request, deadline_monotonic=10.0)


def test_remote_chunk_rejects_wav_duration_mismatch() -> None:
    request = _request()
    package = _package(request)
    short_samples = np.full(request.expected_frame_count - 1, 1_000, dtype=np.int16)
    package = _replace_wav(package, _wav_bytes(short_samples))
    strategy = RemoteMossChunkPreparationStrategy(
        client=_FakeClient(package),
        tempo_bpm=90.0,
        audio_format=AudioFormat(),
        drums=_FakeDrums([]),
        prosody=_FakeProsody([]),
        clock=lambda: 0.0,
    )

    with pytest.raises(RemoteChunkPreparationError, match="exact duration"):
        strategy.prepare(request, deadline_monotonic=10.0)


@pytest.mark.parametrize(
    ("vocal_wav", "message"),
    (
        (b"not a WAV package", "WAV is invalid"),
        (None, "WAV is truncated"),
    ),
)
def test_remote_chunk_rejects_malformed_and_truncated_wav(
    vocal_wav: bytes | None,
    message: str,
) -> None:
    request = _request()
    package = _package(request)
    changed_wav = package.vocal_wav[:-2] if vocal_wav is None else vocal_wav
    package = _replace_wav(package, changed_wav)
    strategy = RemoteMossChunkPreparationStrategy(
        client=_FakeClient(package),
        tempo_bpm=90.0,
        audio_format=AudioFormat(),
        drums=_FakeDrums([]),
        prosody=_FakeProsody([]),
        clock=lambda: 0.0,
    )

    with pytest.raises(RemoteChunkPreparationError, match=message):
        strategy.prepare(request, deadline_monotonic=10.0)


def test_remote_chunk_rejects_alignment_anchor_count_mismatch() -> None:
    request = _request()
    package = _package(request)
    source = tuple(package.manifest.diagnostics.alignment_diagnostics["source_anchors"][:-1])
    target = tuple(package.manifest.diagnostics.alignment_diagnostics["target_anchors"][:-1])
    package = _replace_alignment(package, source_anchors=source, target_anchors=target)
    strategy = RemoteMossChunkPreparationStrategy(
        client=_FakeClient(package),
        tempo_bpm=90.0,
        audio_format=AudioFormat(),
        drums=_FakeDrums([]),
        prosody=_FakeProsody([]),
        clock=lambda: 0.0,
    )

    with pytest.raises(RemoteChunkPreparationError, match="cover every selected syllable"):
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


def test_remote_chunk_preserves_manifest_warnings_and_observed_render_latency() -> None:
    request = _request()
    warnings = ("pronunciation fallback used", "alignment confidence degraded")
    strategy = RemoteMossChunkPreparationStrategy(
        client=_FakeClient(_package(request, warnings=warnings)),
        tempo_bpm=90.0,
        audio_format=AudioFormat(),
        drums=_FakeDrums([]),
        prosody=_FakeProsody([]),
        clock=lambda: 0.0,
    )

    prepared = strategy.prepare(request, deadline_monotonic=10.0)

    assert prepared.diagnostics["warnings"] == warnings
    assert all(tuple(warning.message for warning in bar.warnings) == warnings for bar in prepared.bars)
    assert all(bar.render_latency_ms == pytest.approx(6.0) for bar in prepared.bars)


def test_remote_chunk_rejects_deadline_expiry_during_final_value_construction() -> None:
    request = _request()
    times = iter((0.0, 0.0, 0.0, 10.0))
    strategy = RemoteMossChunkPreparationStrategy(
        client=_FakeClient(_package(request)),
        tempo_bpm=90.0,
        audio_format=AudioFormat(),
        drums=_FakeDrums([]),
        prosody=_FakeProsody([]),
        clock=lambda: next(times),
    )

    with pytest.raises(RemoteChunkPreparationError, match="missed its useful deadline"):
        strategy.prepare(request, deadline_monotonic=10.0)


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
