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

from streammuse.application.rap.chunk_orchestration import PhraseRenderFailed
from streammuse.experiments.rap_audio_protocols.contracts import (
    SyllableTarget,
    TwoBarRenderRequest,
)
from streammuse.infrastructure.rap.mms_forced_alignment import (
    CharacterSpan,
    MmsAlignmentResult,
    WordSpan,
)
from streammuse.infrastructure.rap.moss_aligned_phrase import (
    MossAlignedPhraseRenderer,
)
from streammuse.infrastructure.rap.moss_tts import MossPhraseResult


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
        )


class _FakeFullChunkStretcher:
    def __init__(self) -> None:
        self.calls: list[tuple[int, int, tuple[tuple[int, int], ...]]] = []

    def __call__(
        self,
        samples: np.ndarray,
        target_frames: int,
        sample_rate_hz: int,
        time_map: tuple[tuple[int, int], ...],
    ) -> np.ndarray:
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
    assert synthesizer.calls == [
        (_request(), tmp_path / "request-1" / "moss-source.wav")
    ]
    assert aligner.calls == [
        (tmp_path / "request-1" / "moss-source.wav", "Steady motion!")
    ]
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
    assert result.alignment_diagnostics["coverage"] == 1.0
    assert result.alignment_diagnostics["method_counts"] == {
        "orthographic_vowel_groups": 4
    }
    assert result.alignment_diagnostics["target_anchors_seconds"] == (
        0.25,
        0.75,
        1.35,
        2.0,
    )
    assert result.audio_diagnostics["sample_rate_hz"] == 24_000
    assert result.audio_diagnostics["frame_count"] == 128_000
    assert result.audio_diagnostics["channels"] == 1
    assert result.audio_diagnostics["encoding"] == "PCM16"
    assert (
        result.audio_diagnostics["sha256"]
        == hashlib.sha256(result.vocal_wav).hexdigest()
    )
    assert result.audio_diagnostics["peak"] > 0.0
    assert result.audio_diagnostics["rms"] > 0.0
    assert any("low MMS" in warning for warning in result.warnings)
    assert any("wide local stretch ratio" in warning for warning in result.warnings)

    retained = result.audio_diagnostics["retained_artifacts"]
    assert Path(retained["source_wav"]).is_file()
    assert Path(retained["alignment_json"]).is_file()
    assert Path(retained["vocal_wav"]).read_bytes() == result.vocal_wav
    diagnostics = json.loads(Path(retained["alignment_json"]).read_text())
    assert diagnostics["success"] is True
    assert diagnostics["stress_applied"] is False
    assert diagnostics["timing_regularization_applied"] is False


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
    assert (workspace / "moss-source.wav").is_file()
    failure = json.loads((workspace / "alignment.json").read_text())
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
    assert first.audio_diagnostics["sha256"] == second.audio_diagnostics["sha256"]
    assert first.audio_diagnostics["peak"] == second.audio_diagnostics["peak"]
    assert first.audio_diagnostics["rms"] == second.audio_diagnostics["rms"]
    assert first.alignment_diagnostics["source_anchors_samples"] == (
        2_400,
        7_200,
        13_200,
        16_080,
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
    assert (workspace / "moss-source.wav").is_file()
