"""Independent retained-stem mixing, without rendering or device I/O."""

from dataclasses import replace

import numpy as np
import pytest

from streammuse.application.rap import chunk_audio
from streammuse.domain.rap import audio


def pcm(values) -> audio.PcmAudio:
    samples = np.asarray(values, dtype=np.float32)
    return audio.PcmAudio(audio.AudioFormat(), len(samples), samples.tobytes())


def test_mix_independent_levels_master_and_peak_limit() -> None:
    vocals = pcm([[0.1, -0.1], [0.4, -0.4]])
    drums = pcm([[0.2, -0.2], [0.1, -0.1]])
    config = audio.RapMixConfig(vocal_gain=0.5, drum_gain=1.5, master_gain=0.5)
    mixed = chunk_audio.mix_audio_stems(vocals, drums, config)

    assert mixed.frame_count == 2
    assert np.frombuffer(mixed.data, dtype=np.float32) == pytest.approx([0.175, -0.175, 0.175, -0.175])
    clipped = chunk_audio.mix_audio_stems(vocals, drums, audio.RapMixConfig(vocal_gain=2.0, drum_gain=2.0, peak_limit=0.4))
    assert np.max(np.abs(np.frombuffer(clipped.data, dtype=np.float32))) <= 0.400001


def test_remix_reuses_immutable_stems_and_rejects_legacy_bar_without_them() -> None:
    vocals = pcm([[0.1, -0.1]])
    drums = pcm([[0.2, -0.2]])
    original = audio.PreparedRapBar(0, "one", "test", None, (), vocals, (), (), 5.0)
    config = audio.RapMixConfig(vocal_gain=0, drum_gain=1.0, revision=1)
    with pytest.raises(ValueError, match="stems"):
        chunk_audio.remix_prepared_bar(original, config)

    retained = replace(original, vocal_stem=vocals, drum_stem=drums)
    remixed = chunk_audio.remix_prepared_bar(retained, config)
    assert remixed.audio.data == drums.data
    assert retained.audio.data == vocals.data
    assert remixed.vocal_stem is vocals and remixed.drum_stem is drums
    assert remixed.mix_config == config
    assert remixed.render_latency_ms == retained.render_latency_ms


def test_mix_rejects_mismatched_lengths_and_nonfinite_pcm() -> None:
    config = audio.RapMixConfig()
    with pytest.raises(ValueError, match="same format and frame count"):
        chunk_audio.mix_audio_stems(pcm([[0, 0]]), pcm([[0, 0], [0, 0]]), config)
    with pytest.raises(ValueError, match="finite"):
        chunk_audio.mix_audio_stems(pcm([[float("nan"), 0]]), pcm([[0, 0]]), config)


def test_finite_corrupt_stem_cannot_overflow_the_limiter_into_nan():
    config = audio.RapMixConfig(vocal_gain=2.0)
    with pytest.raises(ValueError, match="finite"):
        chunk_audio.mix_audio_stems(pcm([[3e38, -3e38]]), pcm([[0, 0]]), config)
