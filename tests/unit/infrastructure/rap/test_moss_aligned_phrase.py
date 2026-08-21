from __future__ import annotations

import hashlib
import io
import json
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest
from scipy.io import wavfile

from streammuse.application.rap.chunk_orchestration import (
    ChunkCandidatePlanner,
    PhraseRenderFailed,
    RapChunkOrchestrator,
)
from streammuse.domain.rap import (
    CandidateBatch,
    ProsodyAnalysis,
    RemoteCandidatePolicy,
    RemoteCandidateStats,
    RemoteRapBarRequest,
    RemoteRapChunkDiagnostics,
    RemoteRapChunkRequest,
    ScoreWeights,
    Syllable,
    normalize_text,
)
from streammuse.experiments.rap_audio_protocols.contracts import (
    SyllableTarget,
    TwoBarRenderRequest,
)
from streammuse.infrastructure.rap.mms_forced_alignment import (
    CharacterSpan,
    MmsAlignmentResult,
    WordSpan,
    normalize_mms_transcript,
)
from streammuse.infrastructure.rap import moss_aligned_phrase as renderer_module
from streammuse.infrastructure.rap.moss_aligned_phrase import (
    MossAlignedPhraseRenderer,
)
from streammuse.infrastructure.rap.moss_tts import MossPhraseResult
from streammuse.infrastructure.rap.templates import BUILTIN_TEMPLATES


def _syllable(
    word: str,
    index_in_word: int,
    phonemes: tuple[str, ...],
    *,
    tick: int,
    target_seconds: float,
) -> SyllableTarget:
    return SyllableTarget(
        word=word,
        index_in_word=index_in_word,
        phonemes=phonemes,
        lexical_stress=1 if index_in_word == 0 else 0,
        target_stress=1.0 if index_in_word == 0 else 0.5,
        boundary_strength=2 if tick == 12 else 0,
        absolute_tick=tick,
        tick_in_chunk=tick,
        target_seconds=target_seconds,
    )


def _request() -> TwoBarRenderRequest:
    return TwoBarRenderRequest(
        song_id="render-test",
        chunk_index=0,
        start_bar=0,
        end_bar=2,
        text="Steady motion!",
        syllables=(
            _syllable("steady", 0, ("S", "T", "EH1"), tick=1, target_seconds=0.25),
            _syllable("steady", 1, ("D", "IY0"), tick=4, target_seconds=0.75),
            _syllable("motion", 0, ("M", "OW1"), tick=8, target_seconds=1.35),
            _syllable("motion", 1, ("SH", "AH0", "N"), tick=12, target_seconds=2.00),
        ),
    )


def _word_span(
    word: str,
    word_index: int,
    *,
    start_seconds: float,
) -> WordSpan:
    characters = tuple(
        CharacterSpan(
            word=word,
            word_index=word_index,
            character=character,
            character_index=index,
            start_seconds=start_seconds + index * 0.04,
            end_seconds=start_seconds + (index + 1) * 0.04,
            score=0.9,
        )
        for index, character in enumerate(word)
    )
    return WordSpan(
        word=word,
        word_index=word_index,
        start_seconds=characters[0].start_seconds,
        end_seconds=characters[-1].end_seconds,
        score=0.9,
        characters=characters,
    )


class _FakeSynthesizer:
    def __init__(self) -> None:
        self.calls: list[tuple[TwoBarRenderRequest, Path]] = []

    def synthesize(
        self,
        request: TwoBarRenderRequest,
        output_wav: Path,
    ) -> MossPhraseResult:
        self.calls.append((request, output_wav))
        samples = np.sin(np.linspace(0.0, 80.0, 24_000, dtype=np.float32)) * 0.25
        output_wav.parent.mkdir(parents=True, exist_ok=True)
        wavfile.write(output_wav, 24_000, samples)
        source_sha256 = hashlib.sha256(output_wav.read_bytes()).hexdigest()
        return MossPhraseResult(
            output_wav=output_wav,
            model_id="OpenMOSS-Team/MOSS-TTS-v1.5",
            model_revision="revision-1",
            reference_voice_sha256="reference-sha",
            source_wav_sha256=source_sha256,
            sample_rate_hz=24_000,
            frame_count=len(samples),
            generation_time_ms=111.0,
            resolved_generation_settings=MappingProxyType({"token_target": 64}),
            warnings=("MOSS style instruction is best effort",),
        )


class _FakeAligner:
    def __init__(self) -> None:
        self.calls: list[tuple[Path, str]] = []

    def align(self, source_wav: Path, transcript: str) -> MmsAlignmentResult:
        self.calls.append((source_wav, transcript))
        words = (
            _word_span("steady", 0, start_seconds=0.10),
            _word_span("motion", 1, start_seconds=0.55),
        )
        return MmsAlignmentResult(
            normalized_transcript="steady motion",
            character_spans=tuple(
                character for word in words for character in word.characters
            ),
            word_spans=words,
            duration_seconds=1.0,
            alignment_time_ms=12.0,
            aligner_identity="torchaudio.pipelines.MMS_FA",
            aligner_version="torchaudio 2.8.0+cu128 / MMS_FA",
            warnings=("low MMS CTC score retained",),
            source_sample_rate_hz=24_000,
            source_frame_count=24_000,
            inference_sample_rate_hz=16_000,
            inference_frame_count=16_000,
            emission_frame_count=100,
        )


class _FakeFullChunkStretcher:
    def __init__(self) -> None:
        self.calls: list[tuple[int, int, tuple[tuple[int, int], ...]]] = []
        self.source_frame_counts: list[int] = []

    def __call__(
        self,
        samples: np.ndarray,
        target_frames: int,
        sample_rate_hz: int,
        time_map: tuple[tuple[int, int], ...],
    ) -> np.ndarray:
        self.source_frame_counts.append(len(samples))
        self.calls.append((target_frames, sample_rate_hz, time_map))
        return np.sin(
            np.linspace(0.0, 200.0, target_frames, dtype=np.float32)
        ) * np.float32(0.4)


def test_renders_one_continuous_r3_phrase_with_exact_pcm16_and_diagnostics(
    tmp_path: Path,
) -> None:
    synthesizer = _FakeSynthesizer()
    aligner = _FakeAligner()
    stretcher = _FakeFullChunkStretcher()
    stretcher_factory_calls: list[dict[str, object]] = []

    def stretcher_factory(**kwargs: object) -> _FakeFullChunkStretcher:
        stretcher_factory_calls.append(dict(kwargs))
        return stretcher

    renderer = MossAlignedPhraseRenderer(
        synthesizer=synthesizer,
        aligner=aligner,
        stretcher_factory=stretcher_factory,
        rubberband_version="rubberband 3.3.0 R3",
    )

    result = renderer.render(_request(), tmp_path / "request-1")

    assert stretcher_factory_calls == [{"engine": "r3", "smoothing": False}]
    assert synthesizer.calls == [(_request(), tmp_path / "request-1" / "source.wav")]
    assert aligner.calls == [(tmp_path / "request-1" / "source.wav", "Steady motion!")]
    assert len(stretcher.calls) == 1
    target_frames, sample_rate_hz, time_map = stretcher.calls[0]
    assert target_frames == 128_000
    assert sample_rate_hz == 24_000
    assert time_map[0] == (0, 0)
    assert time_map[-1] == (23_999, 127_999)

    with wave.open(io.BytesIO(result.vocal_wav), "rb") as rendered:
        assert rendered.getframerate() == 24_000
        assert rendered.getnchannels() == 1
        assert rendered.getsampwidth() == 2
        assert rendered.getnframes() == 128_000
        assert any(rendered.readframes(128_000))

    assert result.stage_timings_ms.keys() == {"moss", "aligner", "warp"}
    assert all(value >= 0.0 for value in result.stage_timings_ms.values())
    assert result.model_tool_versions == {
        "moss": "OpenMOSS-Team/MOSS-TTS-v1.5@revision-1",
        "aligner": "torchaudio.pipelines.MMS_FA; torchaudio 2.8.0+cu128 / MMS_FA",
        "rubberband": "rubberband 3.3.0 R3",
    }
    assert result.monitoring_summary == {
        "schema_version": "streammuse.rap_chunk_monitor.v1",
        "alignment_method": "torchaudio.pipelines.MMS_FA",
        "alignment_confidence": pytest.approx(0.9),
        "source_wav_sha256": hashlib.sha256(
            (tmp_path / "request-1" / "source.wav").read_bytes()
        ).hexdigest(),
    }
    assert set(result.alignment_diagnostics) == {
        "fallback_counts",
        "source_anchors",
        "target_anchors",
        "local_warp_ratios",
    }
    assert result.alignment_diagnostics["fallback_counts"] == {
        "phoneme_weighted_character": 0,
        "word_duration_proportional": 0,
    }
    assert result.alignment_diagnostics["target_anchors"] == (
        0.25,
        0.75,
        1.35,
        2.0,
    )
    assert set(result.audio_diagnostics) == {
        "sample_rate_hz",
        "frame_count",
        "duration_seconds",
        "peak",
    }
    assert result.audio_diagnostics["sample_rate_hz"] == 24_000
    assert result.audio_diagnostics["frame_count"] == 128_000
    assert result.audio_diagnostics["peak"] > 0.0
    assert any("low MMS" in warning for warning in result.warnings)
    assert any("wide local stretch ratio" in warning for warning in result.warnings)
    RemoteRapChunkDiagnostics(
        accepted_request_budget_ms=5_000,
        resolved_policy=RemoteCandidatePolicy.realtime_default(),
        candidate_stats=RemoteCandidateStats(0, 0, 0, 0, (), ()),
        stage_timings_ms={
            "generation": 0.0,
            "evaluation": 0.0,
            **result.stage_timings_ms,
            "packaging": 0.0,
            "total": sum(result.stage_timings_ms.values()),
        },
        alignment_diagnostics=result.alignment_diagnostics,
        audio_diagnostics=result.audio_diagnostics,
        model_tool_versions=result.model_tool_versions,
        warnings=result.warnings,
        monitoring_summary={
            **result.monitoring_summary,
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
    )


def test_preserves_complete_canonical_source_and_mms_alignment_artifacts(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "request"
    renderer = MossAlignedPhraseRenderer(
        synthesizer=_FakeSynthesizer(),
        aligner=_FakeAligner(),
        stretcher_factory=lambda **_: _FakeFullChunkStretcher(),
    )

    result = renderer.render(_request(), workspace)

    source_path = workspace / "source.wav"
    alignment_path = workspace / "mms_alignment.json"
    assert source_path.is_file()
    assert alignment_path.is_file()
    assert not (workspace / "moss-source.wav").exists()
    artifact = json.loads(alignment_path.read_text(encoding="utf-8"))
    assert artifact["schema_version"] == "streammuse.mms_alignment.v1"
    assert artifact["request_sha256"] == _request().sha256
    assert artifact["normalized_transcript"] == "steady motion"
    assert artifact["aligner"]["identity"] == "torchaudio.pipelines.MMS_FA"
    assert artifact["aligner"]["version"] == "torchaudio 2.8.0+cu128 / MMS_FA"
    assert artifact["aligner"]["alignment_time_ms"] == 12.0
    assert artifact["aligner"]["duration_seconds"] == 1.0
    assert artifact["aligner"]["confidence"] == pytest.approx(0.9)
    assert artifact["aligner"]["warnings"] == ["low MMS CTC score retained"]
    assert artifact["aligner"]["source_timebase"] == {
        "sample_rate_hz": 24_000,
        "frame_count": 24_000,
        "duration_seconds": 1.0,
    }
    assert artifact["aligner"]["inference_timebase"] == {
        "sample_rate_hz": 16_000,
        "frame_count": 16_000,
        "duration_seconds": 1.0,
    }
    assert artifact["aligner"]["emission_frame_count"] == 100
    assert len(artifact["character_spans"]) == len("steadymotion")
    assert artifact["character_spans"][0] == {
        "word": "steady",
        "word_index": 0,
        "character": "s",
        "character_index": 0,
        "start_seconds": 0.1,
        "end_seconds": 0.14,
        "score": 0.9,
    }
    assert len(artifact["word_spans"]) == 2
    assert len(artifact["word_spans"][0]["characters"]) == len("steady")
    assert artifact["mapping"]["coverage"] == 1.0
    assert artifact["mapping"]["method_counts"] == {"orthographic_vowel_groups": 4}
    assert len(artifact["mapping"]["anchors"]) == len(_request().syllables)
    assert [anchor["target_seconds"] for anchor in artifact["mapping"]["anchors"]] == [
        syllable.target_seconds for syllable in _request().syllables
    ]
    assert artifact["source"]["artifact"] == "source.wav"
    assert (
        artifact["source"]["sha256"]
        == hashlib.sha256(source_path.read_bytes()).hexdigest()
    )
    assert artifact["source"]["sample_rate_hz"] == 24_000
    assert artifact["source"]["frame_count"] == 24_000
    assert artifact["source"]["generation_settings"] == {"token_target": 64}
    assert artifact["output"]["sha256"] == hashlib.sha256(result.vocal_wav).hexdigest()
    assert artifact["output"]["rms"] > 0.0
    assert artifact["warp"]["engine"] == "r3"
    assert artifact["warp"]["stress_applied"] is False
    assert artifact["warp"]["timing_regularization_applied"] is False


def test_tick_zero_uses_cropped_acoustic_onset_as_single_boundary_endpoint(
    tmp_path: Path,
) -> None:
    request = replace(
        _request(),
        syllables=(
            replace(
                _request().syllables[0],
                absolute_tick=0,
                tick_in_chunk=0,
                target_seconds=0.0,
            ),
            *_request().syllables[1:],
        ),
    )
    workspace = tmp_path / "request"
    stretcher = _FakeFullChunkStretcher()
    renderer = MossAlignedPhraseRenderer(
        synthesizer=_FakeSynthesizer(),
        aligner=_FakeAligner(),
        stretcher_factory=lambda **_: stretcher,
    )

    result = renderer.render(request, workspace)

    assert stretcher.source_frame_counts == [21_600]
    assert len(stretcher.calls) == 1
    target_frames, sample_rate_hz, time_map = stretcher.calls[0]
    assert (target_frames, sample_rate_hz) == (128_000, 24_000)
    assert time_map == (
        (0, 0),
        (4_800, 18_000),
        (10_800, 32_400),
        (13_680, 48_000),
        (21_599, 127_999),
    )
    assert len({target for _, target in time_map}) == len(time_map)
    assert result.alignment_diagnostics["target_anchors"] == tuple(
        syllable.target_seconds for syllable in request.syllables
    )
    assert result.alignment_diagnostics["source_anchors"] == pytest.approx(
        (0.0, 0.2, 0.45, 0.57)
    )

    source_rate, source_samples = wavfile.read(workspace / "source.wav")
    assert source_rate == 24_000
    assert len(source_samples) == 24_000
    artifact = json.loads(
        (workspace / "mms_alignment.json").read_text(encoding="utf-8")
    )
    policy = artifact["mapping"]["endpoint_policy"]
    assert policy["name"] == "crop_first_acoustic_onset_to_target_boundary"
    assert policy["applied"] is True
    assert policy["target_zero_as_boundary"] is True
    assert policy["crop_start_source_sample"] == 2_400
    assert policy["original_frame_count"] == 24_000
    assert policy["cropped_frame_count"] == 21_600
    assert policy["original_source_wav_sha256"] == artifact["source"]["sha256"]
    assert policy["warp_input_encoding"] == "float32le"
    assert (
        policy["warp_input_float32le_sha256"]
        == hashlib.sha256(
            np.ascontiguousarray(source_samples[2_400:], dtype="<f4").tobytes()
        ).hexdigest()
    )
    assert artifact["mapping"]["anchors"][0]["source_sample"] == 2_400
    assert artifact["mapping"]["anchors"][0]["warp_source_sample"] == 0
    assert artifact["mapping"]["anchors"][0]["endpoint_role"] == "boundary"
    assert [anchor["target_seconds"] for anchor in artifact["mapping"]["anchors"]] == [
        syllable.target_seconds for syllable in request.syllables
    ]


def test_tick_zero_accepts_acoustic_onset_already_at_source_boundary(
    tmp_path: Path,
) -> None:
    class ZeroOnsetAligner(_FakeAligner):
        def align(self, source_wav: Path, transcript: str) -> MmsAlignmentResult:
            self.calls.append((source_wav, transcript))
            words = (
                _word_span("steady", 0, start_seconds=0.0),
                _word_span("motion", 1, start_seconds=0.55),
            )
            return MmsAlignmentResult(
                normalized_transcript="steady motion",
                character_spans=tuple(
                    character for word in words for character in word.characters
                ),
                word_spans=words,
                duration_seconds=1.0,
                alignment_time_ms=12.0,
                aligner_identity="torchaudio.pipelines.MMS_FA",
                aligner_version="torchaudio test",
                warnings=(),
                source_sample_rate_hz=24_000,
                source_frame_count=24_000,
                inference_sample_rate_hz=16_000,
                inference_frame_count=16_000,
                emission_frame_count=100,
            )

    request = replace(
        _request(),
        syllables=(
            replace(
                _request().syllables[0],
                absolute_tick=0,
                tick_in_chunk=0,
                target_seconds=0.0,
            ),
            *_request().syllables[1:],
        ),
    )
    workspace = tmp_path / "request"
    stretcher = _FakeFullChunkStretcher()
    renderer = MossAlignedPhraseRenderer(
        synthesizer=_FakeSynthesizer(),
        aligner=ZeroOnsetAligner(),
        stretcher_factory=lambda **_: stretcher,
    )

    result = renderer.render(request, workspace)

    assert stretcher.source_frame_counts == [24_000]
    assert stretcher.calls[0][2][0] == (0, 0)
    assert result.alignment_diagnostics["source_anchors"][0] == 0.0
    artifact = json.loads((workspace / "mms_alignment.json").read_text())
    assert artifact["mapping"]["endpoint_policy"]["crop_start_source_sample"] == 0
    assert artifact["mapping"]["anchors"][0]["endpoint_role"] == "boundary"


def test_uses_remote_contract_frame_count_at_non_90_bpm_half_frame_edge(
    tmp_path: Path,
) -> None:
    tempo_bpm = 199.99826390395916
    request = replace(_request(), tempo_bpm=tempo_bpm)
    expected_frames = RemoteRapChunkRequest.frame_count_for(tempo_bpm)
    stretcher = _FakeFullChunkStretcher()
    renderer = MossAlignedPhraseRenderer(
        synthesizer=_FakeSynthesizer(),
        aligner=_FakeAligner(),
        stretcher_factory=lambda **_: stretcher,
    )

    result = renderer.render(request, tmp_path / "request")

    assert expected_frames == 57_600
    assert stretcher.calls[0][0] == expected_frames
    assert result.audio_diagnostics["frame_count"] == expected_frames
    with wave.open(io.BytesIO(result.vocal_wav), "rb") as rendered:
        assert rendered.getnframes() == expected_frames


class _BuiltInFlowGenerator:
    def generate(self, request: object) -> CandidateBatch:
        text = " ".join("a" for _ in range(9))
        return CandidateBatch(
            request_id=getattr(request, "request_id"),
            candidates=(text,),
            source="integration-fake",
            prompt=(),
            raw_response=text,
            latency_ms=0.0,
        )


class _BuiltInFlowAnalyzer:
    def analyze(self, text: str) -> ProsodyAnalysis:
        normalized = normalize_text(text)
        words = normalized.split()
        return ProsodyAnalysis(
            text=text,
            normalized_text=normalized,
            syllables=tuple(
                Syllable(
                    word=word,
                    index_in_word=0,
                    syllable_count=1,
                    stress=1,
                    phonemes=("AH1",),
                    analysis_source="integration-fake",
                )
                for word in words
            ),
            end_rhyme_tail=("AH1",),
            oov_words=(),
            heuristic_words=(),
            punctuation_boundary_after=(),
        )


class _BuiltInFlowAligner:
    def align(self, source_wav: Path, transcript: str) -> MmsAlignmentResult:
        words = normalize_mms_transcript(transcript)
        word_spans = tuple(
            _word_span(
                word,
                index,
                start_seconds=0.10 + index * 0.05,
            )
            for index, word in enumerate(words)
        )
        return MmsAlignmentResult(
            normalized_transcript=" ".join(words),
            character_spans=tuple(
                character for word in word_spans for character in word.characters
            ),
            word_spans=word_spans,
            duration_seconds=1.0,
            alignment_time_ms=4.0,
            aligner_identity="torchaudio.pipelines.MMS_FA",
            aligner_version="torchaudio integration fake",
            warnings=(),
        )


def test_real_renderer_result_crosses_orchestrator_with_builtin_tick_zero_non_90_bpm(
    tmp_path: Path,
) -> None:
    tempo_bpm = 199.99826390395916
    remote_request = RemoteRapChunkRequest.create(
        session_id="renderer-integration",
        chunk_index=0,
        bars=(
            RemoteRapBarRequest(
                0,
                "pulse",
                BUILTIN_TEMPLATES.get("baseline_syncopated_9"),
            ),
            RemoteRapBarRequest(
                1,
                "pulse",
                BUILTIN_TEMPLATES.get("baseline_staggered_9"),
            ),
        ),
        tempo_bpm=tempo_bpm,
        remaining_budget_ms=5_000,
        policy=RemoteCandidatePolicy("integration", 1, 0, 1, 1, 0.0, 500),
        context_lines=(),
        seed=17,
    )
    planner = ChunkCandidatePlanner(
        _BuiltInFlowGenerator(),
        _BuiltInFlowAnalyzer(),
        ScoreWeights(
            stress_alignment=1.0,
            boundary_fit=0.0,
            rhyme_quality=0.0,
            topic_coverage=0.0,
            lexical_continuity=0.0,
            novelty=0.0,
        ),
    )
    stretcher = _FakeFullChunkStretcher()
    renderer = MossAlignedPhraseRenderer(
        synthesizer=_FakeSynthesizer(),
        aligner=_BuiltInFlowAligner(),
        stretcher_factory=lambda **_: stretcher,
    )
    orchestrator = RapChunkOrchestrator(
        planner,
        renderer,
        workspace_root=tmp_path,
    )

    artifact = orchestrator.render(remote_request)

    expected_frames = RemoteRapChunkRequest.frame_count_for(tempo_bpm)
    assert artifact.manifest.expected_frame_count == expected_frames == 57_600
    assert (
        artifact.manifest.diagnostics.audio_diagnostics["frame_count"]
        == expected_frames
    )
    assert set(artifact.manifest.diagnostics.alignment_diagnostics) == {
        "fallback_counts",
        "source_anchors",
        "target_anchors",
        "local_warp_ratios",
    }
    assert (
        artifact.manifest.diagnostics.alignment_diagnostics["target_anchors"][0] == 0.0
    )
    assert stretcher.calls[0][2][0] == (0, 0)
    assert len({target for _, target in stretcher.calls[0][2]}) == len(
        stretcher.calls[0][2]
    )
    assert (artifact.workspace / "source.wav").is_file()
    assert (artifact.workspace / "mms_alignment.json").is_file()
    with wave.open(io.BytesIO(artifact.vocal_wav), "rb") as rendered:
        assert rendered.getnframes() == expected_frames


def test_rejects_transcript_schedule_mismatch_before_expensive_work(
    tmp_path: Path,
) -> None:
    synthesizer = _FakeSynthesizer()
    aligner = _FakeAligner()
    renderer = MossAlignedPhraseRenderer(
        synthesizer=synthesizer,
        aligner=aligner,
        stretcher_factory=lambda **_: _FakeFullChunkStretcher(),
    )

    with pytest.raises(PhraseRenderFailed, match="disagree"):
        renderer.render(
            replace(_request(), text="different transcript"),
            tmp_path / "request",
        )

    assert synthesizer.calls == []
    assert aligner.calls == []


def test_rejects_wrong_request_type_with_typed_preflight_failure(
    tmp_path: Path,
) -> None:
    synthesizer = _FakeSynthesizer()
    aligner = _FakeAligner()
    renderer = MossAlignedPhraseRenderer(
        synthesizer=synthesizer,
        aligner=aligner,
        stretcher_factory=lambda **_: _FakeFullChunkStretcher(),
    )

    with pytest.raises(PhraseRenderFailed, match="two-bar render request"):
        renderer.render(object(), tmp_path / "request")  # type: ignore[arg-type]

    assert synthesizer.calls == []
    assert aligner.calls == []


def test_preflight_failure_removes_all_stale_renderer_owned_artifacts(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "request"
    workspace.mkdir()
    owned_paths = (
        workspace / "source.wav",
        workspace / ".source.partial.wav",
        workspace / "mms_alignment.json",
        workspace / ".mms_alignment.json.partial",
        workspace / "vocal.wav",
        workspace / ".vocal.wav.partial",
    )
    for path in owned_paths:
        path.write_bytes(b"stale successful artifact")
    renderer = MossAlignedPhraseRenderer(
        synthesizer=_FakeSynthesizer(),
        aligner=_FakeAligner(),
        stretcher_factory=lambda **_: _FakeFullChunkStretcher(),
    )

    with pytest.raises(PhraseRenderFailed, match="disagree"):
        renderer.render(replace(_request(), text="different transcript"), workspace)

    assert not any(path.exists() for path in owned_paths)
    failure = json.loads((workspace / "render_failure.json").read_text())
    assert failure["success"] is False
    assert failure["failed_stage"] == "preflight"


def test_success_publishes_vocal_last_and_removes_stale_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "request"
    workspace.mkdir()
    (workspace / "render_failure.json").write_text("stale failure")
    real_write = renderer_module._write_json_atomic
    alignment_writes = 0

    def assert_not_published(path: Path, payload: MappingProxyType | dict) -> None:
        nonlocal alignment_writes
        if path.name == "mms_alignment.json":
            alignment_writes += 1
            assert not (workspace / "vocal.wav").exists()
        real_write(path, payload)

    monkeypatch.setattr(renderer_module, "_write_json_atomic", assert_not_published)
    renderer = MossAlignedPhraseRenderer(
        synthesizer=_FakeSynthesizer(),
        aligner=_FakeAligner(),
        stretcher_factory=lambda **_: _FakeFullChunkStretcher(),
    )

    result = renderer.render(_request(), workspace)

    assert alignment_writes >= 1
    assert (workspace / "vocal.wav").read_bytes() == result.vocal_wav
    assert not (workspace / "render_failure.json").exists()


def test_base_exception_cancellation_removes_unpublished_vocal_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RenderCancelled(BaseException):
        pass

    workspace = tmp_path / "request"
    real_write = renderer_module._write_json_atomic

    def cancel_alignment_write(path: Path, payload: MappingProxyType | dict) -> None:
        if path.name == "mms_alignment.json":
            raise RenderCancelled
        real_write(path, payload)

    monkeypatch.setattr(renderer_module, "_write_json_atomic", cancel_alignment_write)
    renderer = MossAlignedPhraseRenderer(
        synthesizer=_FakeSynthesizer(),
        aligner=_FakeAligner(),
        stretcher_factory=lambda **_: _FakeFullChunkStretcher(),
    )

    with pytest.raises(RenderCancelled):
        renderer.render(_request(), workspace)

    assert (workspace / "source.wav").is_file()
    assert not (workspace / "vocal.wav").exists()
    assert not (workspace / ".vocal.wav.partial").exists()


def test_atomic_alignment_write_removes_partial_on_base_exception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class PublicationCancelled(BaseException):
        pass

    path = tmp_path / "mms_alignment.json"

    def cancel_replace(source: Path, destination: Path) -> None:
        raise PublicationCancelled

    monkeypatch.setattr(renderer_module.os, "replace", cancel_replace)

    with pytest.raises(PublicationCancelled):
        renderer_module._write_json_atomic(path, {"evidence": "complete"})

    assert not path.exists()
    assert not (tmp_path / ".mms_alignment.json.partial").exists()


def test_failed_silent_warp_cleans_stale_final_but_retains_research_artifacts(
    tmp_path: Path,
) -> None:
    class SilentStretcher(_FakeFullChunkStretcher):
        def __call__(
            self,
            samples: np.ndarray,
            target_frames: int,
            sample_rate_hz: int,
            time_map: tuple[tuple[int, int], ...],
        ) -> np.ndarray:
            super().__call__(samples, target_frames, sample_rate_hz, time_map)
            return np.zeros(target_frames, dtype=np.float32)

    workspace = tmp_path / "request"
    workspace.mkdir()
    (workspace / "vocal.wav").write_bytes(b"stale success")
    renderer = MossAlignedPhraseRenderer(
        synthesizer=_FakeSynthesizer(),
        aligner=_FakeAligner(),
        stretcher_factory=lambda **_: SilentStretcher(),
    )

    with pytest.raises(PhraseRenderFailed, match="silent"):
        renderer.render(_request(), workspace)

    assert not (workspace / "vocal.wav").exists()
    assert not (workspace / ".vocal.wav.partial").exists()
    assert (workspace / "source.wav").is_file()
    alignment = json.loads((workspace / "mms_alignment.json").read_text())
    assert len(alignment["character_spans"]) == len("steadymotion")
    assert len(alignment["mapping"]["anchors"]) == len(_request().syllables)
    failure = json.loads((workspace / "render_failure.json").read_text())
    assert failure["success"] is False
    assert failure["failed_stage"] == "warp"
    assert failure["mapped_anchor_count"] == len(_request().syllables)


def test_repeated_renders_have_deterministic_audio_hashes_and_metrics(
    tmp_path: Path,
) -> None:
    renderer = MossAlignedPhraseRenderer(
        synthesizer=_FakeSynthesizer(),
        aligner=_FakeAligner(),
        stretcher_factory=lambda **_: _FakeFullChunkStretcher(),
    )

    first = renderer.render(_request(), tmp_path / "request-a")
    second = renderer.render(_request(), tmp_path / "request-b")

    assert first.vocal_wav == second.vocal_wav
    assert first.audio_diagnostics == second.audio_diagnostics
    assert first.audio_diagnostics["peak"] == second.audio_diagnostics["peak"]
    assert first.alignment_diagnostics["source_anchors"] == pytest.approx(
        (
            0.1,
            0.3,
            0.55,
            0.67,
        )
    )


def test_concurrent_calls_serialize_resident_model_state(tmp_path: Path) -> None:
    class OverlapDetectingSynthesizer(_FakeSynthesizer):
        def __init__(self) -> None:
            super().__init__()
            self.active = False
            self.overlap_detected = False

        def synthesize(
            self,
            request: TwoBarRenderRequest,
            output_wav: Path,
        ) -> MossPhraseResult:
            if self.active:
                self.overlap_detected = True
            self.active = True
            try:
                time.sleep(0.03)
                return super().synthesize(request, output_wav)
            finally:
                self.active = False

    synthesizer = OverlapDetectingSynthesizer()
    renderer = MossAlignedPhraseRenderer(
        synthesizer=synthesizer,
        aligner=_FakeAligner(),
        stretcher_factory=lambda **_: _FakeFullChunkStretcher(),
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = (
            executor.submit(renderer.render, _request(), tmp_path / f"request-{index}")
            for index in range(2)
        )
        results = tuple(future.result() for future in futures)

    assert len(results) == 2
    assert not synthesizer.overlap_detected
    assert len(synthesizer.calls) == 2


def test_rejects_audio_that_becomes_silent_after_pcm16_quantization(
    tmp_path: Path,
) -> None:
    class SubLsbStretcher(_FakeFullChunkStretcher):
        def __call__(
            self,
            samples: np.ndarray,
            target_frames: int,
            sample_rate_hz: int,
            time_map: tuple[tuple[int, int], ...],
        ) -> np.ndarray:
            super().__call__(samples, target_frames, sample_rate_hz, time_map)
            return np.full(target_frames, 1e-8, dtype=np.float32)

    workspace = tmp_path / "request"
    renderer = MossAlignedPhraseRenderer(
        synthesizer=_FakeSynthesizer(),
        aligner=_FakeAligner(),
        stretcher_factory=lambda **_: SubLsbStretcher(),
    )

    with pytest.raises(PhraseRenderFailed, match="PCM16.*silent"):
        renderer.render(_request(), workspace)

    assert not (workspace / "vocal.wav").exists()
    assert (workspace / "source.wav").is_file()
