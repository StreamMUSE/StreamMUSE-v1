from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from streammuse.experiments.rap_audio_protocols.contracts import ChunkRenderRecord, ProtocolId, SyllableTarget, TwoBarRenderRequest


def _request(chunk_index: int = 0) -> TwoBarRenderRequest:
    start_bar = chunk_index * 2
    syllables = []
    words = []
    tick_pattern = (0, 2, 3, 5, 7, 8, 10, 13, 15, 16, 18, 20, 21, 23, 24, 26, 28, 31)
    for index, tick_in_chunk in enumerate(tick_pattern):
        word = f"word{index:02d}"
        words.append(word)
        syllables.append(
            SyllableTarget(
                word=word,
                index_in_word=0,
                phonemes=("W", "ER1", "D"),
                lexical_stress=1 if index % 2 == 0 else 0,
                target_stress=1.0 if index % 2 == 0 else 0.2,
                boundary_strength=2 if index in {8, 17} else 0,
                absolute_tick=start_bar * 16 + tick_in_chunk,
                tick_in_chunk=tick_in_chunk,
                target_seconds=tick_in_chunk / 6.0,
            )
        )
    return TwoBarRenderRequest(
        song_id="01_space_exploration",
        chunk_index=chunk_index,
        start_bar=start_bar,
        end_bar=start_bar + 2,
        text=" ".join(words),
        syllables=tuple(syllables),
    )


def _write_chunk_wav(path: Path, request: TwoBarRenderRequest, *, clipped: bool = False, short_frames: int = 0) -> None:
    frame_count = 256_000 - short_frames
    samples = np.zeros(frame_count, dtype=np.float32)
    sample_rate_hz = 48_000
    for syllable in request.syllables:
        center = min(frame_count - 1, max(0, int(round(syllable.target_seconds * sample_rate_hz))))
        span = slice(max(0, center - 80), min(frame_count, center + 80))
        samples[span] = np.float32(0.12 + 0.78 * syllable.target_stress)
    if clipped:
        samples[100] = 1.0
        samples[101] = -1.0
    wavfile.write(path, sample_rate_hz, samples)


def _record(request: TwoBarRenderRequest, path: Path, *, success: bool = True) -> ChunkRenderRecord:
    return ChunkRenderRecord(
        protocol_id=ProtocolId.MOSS_GLOBAL,
        song_id=request.song_id,
        chunk_index=request.chunk_index,
        request_sha256=request.sha256,
        success=success,
        output_path=str(path) if success else None,
        output_sha256="a" * 64 if success else None,
        sample_rate_hz=48_000 if success else None,
        attempts=1,
        error=None if success else "synthetic failure",
    )


def test_compute_word_error_counts_uses_exact_levenshtein_alignment() -> None:
    evaluation = importlib.import_module("streammuse.experiments.rap_audio_protocols.evaluation")

    counts = evaluation.compute_word_error_counts(
        ("alpha", "beta", "gamma", "delta"),
        ("alpha", "theta", "gamma", "delta", "delta"),
    )

    assert counts == {
        "reference_word_count": 4,
        "substitutions": 1,
        "insertions": 1,
        "deletions": 0,
        "word_error_rate": pytest.approx(0.5),
    }


def test_estimate_syllable_timing_error_and_stress_rms_correlation_from_independent_words() -> None:
    evaluation = importlib.import_module("streammuse.experiments.rap_audio_protocols.evaluation")
    request = _request(0)
    words = tuple(
        evaluation.RecognizedWord(
            text=syllable.word,
            start_seconds=syllable.target_seconds,
            end_seconds=syllable.target_seconds + 0.12,
        )
        for syllable in request.syllables
    )
    sample_rate_hz = 48_000
    samples = np.zeros(256_000, dtype=np.float32)
    for syllable in request.syllables:
        center = int(round(syllable.target_seconds * sample_rate_hz))
        span = slice(max(0, center - 64), min(samples.shape[0], center + 64))
        samples[span] = np.float32(0.1 + 0.85 * syllable.target_stress)
    samples[10] = 1.0
    samples[11] = -1.0

    timing_errors = evaluation.estimate_syllable_timing_error_ms(request, words)
    stress_correlation = evaluation.measure_stress_rms_correlation(
        request,
        samples,
        sample_rate_hz=sample_rate_hz,
    )
    clip_counts = evaluation.compute_signal_metrics(samples)

    assert len(timing_errors) == 18
    assert max(abs(error) for error in timing_errors) < 1e-6
    assert stress_correlation == pytest.approx(1.0, abs=1e-6)
    assert clip_counts["clipped_sample_count"] == 2
    assert clip_counts["silent"] is False


def test_estimate_syllable_timing_error_skips_inserted_asr_word_without_shifting_matches() -> None:
    evaluation = importlib.import_module("streammuse.experiments.rap_audio_protocols.evaluation")
    request = _request(0)
    recognized = (
        evaluation.RecognizedWord(text="intruder", start_seconds=10.0, end_seconds=10.1),
        *(
            evaluation.RecognizedWord(
                text=syllable.word,
                start_seconds=syllable.target_seconds,
                end_seconds=syllable.target_seconds + 0.12,
            )
            for syllable in request.syllables
        ),
    )

    timing_errors = evaluation.estimate_syllable_timing_error_ms(request, recognized)

    assert len(timing_errors) == 18
    assert max(abs(error) for error in timing_errors) < 1e-6


def test_estimate_syllable_timing_error_skips_deleted_asr_word_without_shifting_matches() -> None:
    evaluation = importlib.import_module("streammuse.experiments.rap_audio_protocols.evaluation")
    request = _request(0)
    recognized = tuple(
        evaluation.RecognizedWord(
            text=syllable.word,
            start_seconds=syllable.target_seconds,
            end_seconds=syllable.target_seconds + 0.12,
        )
        for index, syllable in enumerate(request.syllables)
        if index != 7
    )

    timing_errors = evaluation.estimate_syllable_timing_error_ms(request, recognized)

    assert len(timing_errors) == 17
    assert max(abs(error) for error in timing_errors) < 1e-6


def test_evaluate_protocol_song_aggregates_duration_wer_and_failed_chunk_counts(tmp_path: Path) -> None:
    evaluation = importlib.import_module("streammuse.experiments.rap_audio_protocols.evaluation")
    first = _request(0)
    second = _request(1)
    third = _request(2)
    first_path = tmp_path / "chunk-000.wav"
    second_path = tmp_path / "chunk-001.wav"
    _write_chunk_wav(first_path, first)
    _write_chunk_wav(second_path, second, clipped=True, short_frames=960)

    transcripts = {
        0: tuple(
            evaluation.RecognizedWord(
                text=syllable.word,
                start_seconds=syllable.target_seconds,
                end_seconds=syllable.target_seconds + 0.12,
            )
            for syllable in first.syllables
        ),
        1: tuple(
            evaluation.RecognizedWord(
                text=text,
                start_seconds=index * 0.12,
                end_seconds=index * 0.12 + 0.10,
            )
            for index, text in enumerate(
                [*(f"word{index:02d}" for index in range(8)), "wrong08", *("word09", "word10", "word11", "word12", "word13", "word14", "word15", "word16", "word17", "extra17")]
            )
        ),
    }

    metrics = evaluation.evaluate_protocol_song(
        protocol_id=ProtocolId.MOSS_GLOBAL,
        song_id=first.song_id,
        requests=(first, second, third),
        chunk_records=(
            _record(first, first_path),
            _record(second, second_path),
            _record(third, tmp_path / "missing.wav", success=False),
        ),
        transcribe_chunk=lambda path: transcripts[int(Path(path).stem.split("-")[-1])],
    )

    assert metrics["protocol_id"] == ProtocolId.MOSS_GLOBAL.value
    assert metrics["song_id"] == first.song_id
    assert metrics["successful_chunk_count"] == 2
    assert metrics["failed_chunk_count"] == 1
    assert metrics["word_error_counts"] == {
        "reference_word_count": 36,
        "substitutions": 1,
        "insertions": 1,
        "deletions": 0,
    }
    assert metrics["word_error_rate"] == pytest.approx(2 / 36)
    assert metrics["duration_error_ms"] == {
        "mean": pytest.approx(10.0),
        "median": pytest.approx(10.0),
        "p95": pytest.approx(19.0),
        "max": pytest.approx(20.0),
    }
    assert metrics["clipped_sample_count"] == 2
    assert metrics["estimated_syllable_timing_error_ms"]["measured_count"] == 35
    assert metrics["stress_rms_correlation"] > 0.99


def test_build_faster_whisper_transcriber_imports_heavy_dependency_lazily(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    evaluation = importlib.import_module("streammuse.experiments.rap_audio_protocols.evaluation")
    created = []

    class _FakeSegment:
        def __init__(self) -> None:
            self.words = [type("Word", (), {"word": "alpha", "start": 0.0, "end": 0.2})()]

    class _FakeModel:
        def transcribe(self, wav_path: str, *, word_timestamps: bool, vad_filter: bool):
            assert wav_path.endswith("sample.wav")
            assert word_timestamps is True
            assert vad_filter is True
            return ([_FakeSegment()], {"language": "en"})

    def factory(*, model_size_or_path: str, device: str, compute_type: str):
        created.append((model_size_or_path, device, compute_type))
        return _FakeModel()

    transcriber = evaluation.build_faster_whisper_transcriber(
        model_size="small",
        device="cpu",
        compute_type="int8",
        whisper_model_factory=factory,
    )

    assert created == []
    words = transcriber(tmp_path / "sample.wav")

    assert created == [("small", "cpu", "int8")]
    assert [(word.text, word.start_seconds, word.end_seconds) for word in words] == [("alpha", 0.0, 0.2)]
