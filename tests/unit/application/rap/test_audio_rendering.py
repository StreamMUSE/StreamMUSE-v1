import numpy as np
import pytest

from streammuse.application.rap.audio_rendering import (
    FitContext,
    bar_frame_count,
    bar_start_frame,
    fit_syllable,
    limit_peak,
    mix_at,
    tick_frame_in_bar,
    trim_silence,
)
from streammuse.domain.rap import AudioFormat, AudioWarningCode, PcmAudio
from streammuse.domain.timing import Tempo


def mono_pcm(frames: int, value: float = 0.25, sample_rate: int = 48_000) -> PcmAudio:
    samples = np.full(frames, value, dtype=np.float32)
    return PcmAudio(AudioFormat(sample_rate, 1), frames, samples.tobytes())


def test_sixty_bpm_bar_and_tick_frames_are_exact() -> None:
    tempo = Tempo(60.0, 4, 4)
    audio_format = AudioFormat(48_000, 2)

    assert bar_start_frame(0, tempo, audio_format) == 0
    assert bar_frame_count(0, tempo, audio_format) == 192_000
    assert tick_frame_in_bar(0, 7, tempo, audio_format) == 84_000


def test_fractional_bar_lengths_do_not_accumulate_drift_at_92_bpm() -> None:
    tempo = Tempo(92.0, 4, 4)
    audio_format = AudioFormat(48_000, 2)
    lengths = [bar_frame_count(bar, tempo, audio_format) for bar in range(100)]

    assert sum(lengths) == bar_start_frame(100, tempo, audio_format)
    assert set(lengths).issubset({125_217, 125_218})


def test_timing_rejects_negative_positions_and_out_of_bar_ticks() -> None:
    tempo = Tempo(120.0, 4, 4)
    audio_format = AudioFormat()

    with pytest.raises(ValueError):
        bar_start_frame(-1, tempo, audio_format)
    with pytest.raises(ValueError):
        tick_frame_in_bar(0, -1, tempo, audio_format)
    with pytest.raises(ValueError):
        tick_frame_in_bar(0, tempo.ticks_per_bar, tempo, audio_format)


def test_timing_rejects_non_float32_audio_format() -> None:
    with pytest.raises(ValueError, match="float32"):
        bar_start_frame(0, Tempo(120.0, 4, 4), AudioFormat(sample_width_bytes=2))


def test_pcm_audio_boundaries_reject_non_float32_audio() -> None:
    audio = PcmAudio(AudioFormat(channels=1, sample_width_bytes=2), 2, bytes(4))
    context = FitContext(bar=0, slot_index=0, word="width")

    with pytest.raises(ValueError, match="float32"):
        trim_silence(audio, threshold_dbfs=-45.0, padding_ms=5.0)
    with pytest.raises(ValueError, match="float32"):
        fit_syllable(audio, available_frames=2, final_in_bar=False, context=context)
    with pytest.raises(ValueError, match="float32"):
        mix_at(np.zeros((2, 1), dtype=np.float32), audio, onset_frame=0, gain=1.0)


def test_fit_syllable_leaves_audio_unchanged_when_it_fits() -> None:
    source = mono_pcm(frames=300)
    context = FitContext(bar=1, slot_index=2, word="fits")

    fitted = fit_syllable(source, available_frames=400, final_in_bar=False, context=context)

    assert fitted.audio == source
    assert fitted.compression_ratio == pytest.approx(1.0)
    assert fitted.overlap_frames == 0
    assert fitted.warnings == ()


def test_fit_syllable_compresses_to_available_frames() -> None:
    source = mono_pcm(frames=600)
    context = FitContext(bar=1, slot_index=3, word="compress")

    fitted = fit_syllable(source, available_frames=400, final_in_bar=False, context=context)

    assert fitted.audio.frame_count == 400
    assert fitted.compression_ratio == pytest.approx(1.5)
    assert fitted.overlap_frames == 0
    assert [warning.code for warning in fitted.warnings] == [AudioWarningCode.TIMING_PRESSURE]


def test_long_nonfinal_syllable_caps_compression_and_reports_overlap() -> None:
    source = mono_pcm(frames=1_000, value=0.25)
    context = FitContext(bar=2, slot_index=4, word="timing")

    fitted = fit_syllable(source, available_frames=300, final_in_bar=False, context=context)

    assert fitted.audio.frame_count == 500
    assert fitted.compression_ratio == pytest.approx(2.0)
    assert fitted.overlap_frames == 200
    assert fitted.warnings[0].code == AudioWarningCode.TIMING_PRESSURE


def test_fit_syllable_uses_configured_compression_cap() -> None:
    source = mono_pcm(frames=900, value=0.25)
    context = FitContext(bar=2, slot_index=4, word="timing")

    fitted = fit_syllable(
        source,
        available_frames=300,
        final_in_bar=False,
        context=context,
        max_compression=3.0,
    )

    assert fitted.audio.frame_count == 300
    assert fitted.compression_ratio == pytest.approx(3.0)
    assert fitted.overlap_frames == 0


def test_final_syllable_is_forced_to_bar_boundary() -> None:
    source = mono_pcm(frames=1_000, value=0.25)
    context = FitContext(bar=2, slot_index=8, word="ending")

    fitted = fit_syllable(source, available_frames=300, final_in_bar=True, context=context)

    assert fitted.audio.frame_count == 300
    assert [warning.code for warning in fitted.warnings] == [
        AudioWarningCode.TIMING_PRESSURE,
        AudioWarningCode.FORCED_BAR_FIT,
    ]


def test_trim_silence_uses_dbfs_threshold_and_padding() -> None:
    samples = np.zeros(20, dtype=np.float32)
    samples[5:15] = 0.1
    audio = PcmAudio(AudioFormat(1_000, 1), 20, samples.tobytes())

    trimmed = trim_silence(audio, threshold_dbfs=-45.0, padding_ms=5.0)

    assert trimmed.frame_count == 20
    assert np.frombuffer(trimmed.data, dtype=np.float32).tolist() == pytest.approx(samples.tolist())


def test_mix_at_uses_exact_absolute_onset() -> None:
    destination = np.zeros((8, 1), dtype=np.float32)
    source = mono_pcm(2, value=0.5)

    mix_at(destination, source, onset_frame=3, gain=0.5)

    assert destination[:, 0].tolist() == pytest.approx([0, 0, 0, 0.25, 0.25, 0, 0, 0])


def test_limit_peak_scales_only_when_needed() -> None:
    samples = np.array([0.25, -1.0, 0.5], dtype=np.float32)

    limited = limit_peak(samples)

    assert np.max(np.abs(limited)) == pytest.approx(0.95)
    assert limited[0] == pytest.approx(0.2375)
