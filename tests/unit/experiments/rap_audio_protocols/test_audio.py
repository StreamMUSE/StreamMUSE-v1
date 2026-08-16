from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from streammuse.domain.rap import AudioFormat, PcmAudio
from streammuse.experiments.rap_audio_protocols.audio import (
    CHUNK_FRAME_COUNT,
    SONG_FRAME_COUNT,
    ChunkAssemblyDiagnostic,
    assemble_vocal_stem,
    load_wav_mono_float32,
    mix_stems,
    render_common_drums,
)
from streammuse.experiments.rap_audio_protocols.contracts import SyllableTarget, TwoBarRenderRequest


def _requests(total_chunks: int = 25) -> tuple[TwoBarRenderRequest, ...]:
    return tuple(_request(index) for index in range(total_chunks))


def _request(chunk_index: int) -> TwoBarRenderRequest:
    start_bar = chunk_index * 2
    syllables = tuple(
        SyllableTarget(
            word=f"word{index // 2}",
            index_in_word=index % 2,
            phonemes=("W", "ER1", "D"),
            lexical_stress=1 if index % 3 == 0 else 0,
            target_stress=1.0 if index % 4 == 0 else 0.4,
            boundary_strength=3 if index == 17 else 0,
            absolute_tick=start_bar * 16 + index,
            tick_in_chunk=index,
            target_seconds=index / 6,
        )
        for index in range(18)
    )
    return TwoBarRenderRequest(
        song_id="01_space_exploration",
        chunk_index=chunk_index,
        start_bar=start_bar,
        end_bar=start_bar + 2,
        text=f"chunk {chunk_index}",
        syllables=syllables,
    )


def _write_pcm16_mono(path: Path, sample_rate_hz: int, samples: np.ndarray) -> None:
    wavfile.write(path, sample_rate_hz, np.asarray(np.clip(samples, -1.0, 1.0) * 32767, dtype=np.int16))


def _audio_samples(audio: PcmAudio) -> np.ndarray:
    return np.frombuffer(audio.data, dtype=np.float32).reshape(audio.frame_count, audio.format.channels)


def test_load_wav_mono_float32_resamples_integer_sources_to_target_format(tmp_path: Path) -> None:
    first_path = tmp_path / "24khz.wav"
    second_path = tmp_path / "22050.wav"
    _write_pcm16_mono(first_path, 24_000, np.linspace(-0.75, 0.75, 2_400, dtype=np.float32))
    _write_pcm16_mono(second_path, 22_050, np.linspace(-0.5, 0.5, 2_205, dtype=np.float32))

    first = load_wav_mono_float32(first_path)
    second = load_wav_mono_float32(second_path)

    assert first.format == AudioFormat(sample_rate_hz=48_000, channels=1, sample_width_bytes=4)
    assert second.format == AudioFormat(sample_rate_hz=48_000, channels=1, sample_width_bytes=4)
    assert first.frame_count == 4_800
    assert second.frame_count == 4_800
    assert _audio_samples(first).dtype == np.float32
    assert _audio_samples(second).dtype == np.float32
    assert np.max(np.abs(_audio_samples(first))) <= 1.0
    assert np.max(np.abs(_audio_samples(second))) <= 1.0


def test_assemble_vocal_stem_pads_and_truncates_only_at_chunk_boundaries_for_50_bars(tmp_path: Path) -> None:
    requests = _requests()
    chunk_paths: list[Path] = []
    exact_chunk = np.full(CHUNK_FRAME_COUNT, 0.1, dtype=np.float32)

    for index in range(25):
        path = tmp_path / f"chunk-{index:02d}.wav"
        if index == 0:
            _write_pcm16_mono(path, 48_000, np.full(CHUNK_FRAME_COUNT - 1_000, 0.25, dtype=np.float32))
        elif index == 1:
            _write_pcm16_mono(path, 48_000, np.full(CHUNK_FRAME_COUNT + 200, 0.5, dtype=np.float32))
        else:
            _write_pcm16_mono(path, 48_000, exact_chunk)
        chunk_paths.append(path)

    assembled = assemble_vocal_stem(requests, chunk_paths_by_index={index: path for index, path in enumerate(chunk_paths)})
    samples = _audio_samples(assembled.audio)[:, 0]

    assert assembled.audio.format == AudioFormat(sample_rate_hz=48_000, channels=1, sample_width_bytes=4)
    assert assembled.audio.frame_count == SONG_FRAME_COUNT
    assert assembled.diagnostics[0] == ChunkAssemblyDiagnostic(
        chunk_index=0,
        source_frames=CHUNK_FRAME_COUNT - 1_000,
        output_frames=CHUNK_FRAME_COUNT,
        action="pad_silence",
        message="chunk shorter than two bars; padded trailing silence",
        adjusted_frames=1_000,
    )
    assert assembled.diagnostics[1].action == "truncate"
    assert "truncated" in assembled.diagnostics[1].message
    np.testing.assert_allclose(samples[CHUNK_FRAME_COUNT - 1_000 : CHUNK_FRAME_COUNT], 0.0, atol=1e-6)
    np.testing.assert_allclose(samples[CHUNK_FRAME_COUNT : (2 * CHUNK_FRAME_COUNT)], 0.5, atol=5e-5)


def test_assemble_vocal_stem_rejects_non_campaign_request_sets_without_explicit_override(tmp_path: Path) -> None:
    requests = _requests(1)
    path = tmp_path / "chunk-00.wav"
    _write_pcm16_mono(path, 48_000, np.full(CHUNK_FRAME_COUNT, 0.25, dtype=np.float32))

    with pytest.raises(ValueError, match="25 requests"):
        assemble_vocal_stem(requests, chunk_paths_by_index={0: path})


def test_smoke_override_allows_short_request_sets_for_targeted_audio_tests(tmp_path: Path) -> None:
    requests = _requests(1)
    path = tmp_path / "chunk-00.wav"
    _write_pcm16_mono(path, 48_000, np.full(CHUNK_FRAME_COUNT, 0.25, dtype=np.float32))

    assembled = assemble_vocal_stem(requests, chunk_paths_by_index={0: path}, allow_smoke_test=True)
    drums = render_common_drums(requests, song_index=3, allow_smoke_test=True)

    assert assembled.audio.frame_count == CHUNK_FRAME_COUNT
    assert drums.frame_count == CHUNK_FRAME_COUNT


def test_render_common_drums_is_deterministic_for_a_song_index_under_smoke_override() -> None:
    requests = _requests(1)

    first = render_common_drums(requests, song_index=3, allow_smoke_test=True)
    second = render_common_drums(requests, song_index=3, allow_smoke_test=True)

    assert first.format == AudioFormat(sample_rate_hz=48_000, channels=2, sample_width_bytes=4)
    assert first.frame_count == CHUNK_FRAME_COUNT
    assert first.data == second.data


def test_mix_stems_uses_shared_peak_gain_only_when_the_song_exceeds_limit() -> None:
    vocal_samples = np.full((8, 1), 0.9, dtype=np.float32)
    drum_samples = np.full((8, 2), 0.9, dtype=np.float32)
    vocals = PcmAudio(AudioFormat(sample_rate_hz=48_000, channels=1, sample_width_bytes=4), 8, vocal_samples.tobytes())
    drums = PcmAudio(AudioFormat(sample_rate_hz=48_000, channels=2, sample_width_bytes=4), 8, drum_samples.tobytes())

    mixed = mix_stems(vocals, drums)
    samples = _audio_samples(mixed.audio)

    assert mixed.peak_before_limiter == pytest.approx(1.125, rel=1e-4)
    assert mixed.applied_gain == pytest.approx(0.98 / 1.125, rel=1e-4)
    np.testing.assert_allclose(samples[:, 0], 0.98, atol=1e-6)
    np.testing.assert_allclose(samples[:, 1], 0.98, atol=1e-6)
