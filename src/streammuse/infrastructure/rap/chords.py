"""Stateless CPU-only soft-key/pad synthesis for the hand-authored chord loops."""

import numpy as np

from streammuse.domain.rap.audio import AudioFormat, PcmAudio
from streammuse.domain.rap.chords import CHORD_LOOPS, CHORD_LOOP_IDS


def render_chord_loop(
    loop_id: str,
    bar: int,
    frame_count: int,
    audio_format: AudioFormat,
) -> PcmAudio:
    """Render one exact bar of float32 PCM, or silence for ``off``.

    The absolute zero-based bar selects the progression phase. The caller's
    frame count is the sole duration authority, including after tempo changes;
    no previous render, audio device, model, or network service is needed.
    """
    if not isinstance(loop_id, str) or loop_id not in CHORD_LOOP_IDS:
        raise ValueError(f"chord_loop must be one of {CHORD_LOOP_IDS}")
    if type(bar) is not int or bar < 0:
        raise ValueError("bar must be a non-negative integer")
    if type(frame_count) is not int or frame_count < 0:
        raise ValueError("frame_count must be a non-negative integer")
    if (
        not isinstance(audio_format, AudioFormat)
        or audio_format.sample_width_bytes != 4
        or type(audio_format.sample_rate_hz) is not int
        or audio_format.sample_rate_hz <= 0
        or type(audio_format.channels) is not int
        or audio_format.channels <= 0
    ):
        raise ValueError("chords require float32 PCM with positive sample rate and channels")
    if loop_id == "off" or frame_count < 3:
        return PcmAudio(audio_format, frame_count, bytes(frame_count * audio_format.channels * 4))

    loop = CHORD_LOOPS[loop_id]
    pitches = loop.chords[bar % len(loop.chords)]
    sample_rate = audio_format.sample_rate_hz
    time = np.arange(frame_count, dtype=np.float64) / sample_rate
    samples = np.zeros(frame_count, dtype=np.float64)
    for pitch in pitches:
        frequency = 440.0 * 2 ** ((pitch - 69) / 12)
        for harmonic, gain in ((1, 0.85), (2, 0.12), (3, 0.03)):
            if frequency * harmonic < sample_rate / 2:
                samples += gain * np.sin(2 * np.pi * frequency * harmonic * time)

    # A soft key decay over a sustained pad, with zero-slope boundary fades.
    envelope = 0.55 + 0.45 * np.exp(-3.0 * np.linspace(0.0, 1.0, frame_count))
    attack = min(max(2, round(0.020 * sample_rate)), frame_count // 2)
    release = min(max(2, round(0.080 * sample_rate)), frame_count // 2)
    envelope[:attack] *= np.sin(np.linspace(0.0, np.pi / 2, attack)) ** 2
    envelope[-release:] *= np.sin(np.linspace(np.pi / 2, 0.0, release)) ** 2
    # Single-frame fades need an explicit endpoint (linspace starts at pi/2).
    envelope[0] = envelope[-1] = 0.0
    samples *= envelope * (0.6 / len(pitches))
    pcm = np.repeat(samples.astype(np.float32)[:, np.newaxis], audio_format.channels, axis=1)
    return PcmAudio(audio_format, frame_count, pcm.tobytes())
