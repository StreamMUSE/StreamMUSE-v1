from __future__ import annotations

import shutil
import warnings
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile
from scipy.io.wavfile import WavFileWarning

from streammuse.domain.rap import AudioFormat, PcmAudio
from streammuse.infrastructure.rap.time_stretch import RubberBandTimeStretcher


def _stereo_pcm(frames: int, *, sample_rate_hz: int = 48_000) -> PcmAudio:
    samples = np.linspace(-0.25, 0.25, frames * 2, dtype=np.float32).reshape(frames, 2)
    return PcmAudio(AudioFormat(sample_rate_hz, 2), frames, samples.tobytes())


def _mono_pcm(frames: int, *, sample_rate_hz: int = 48_000) -> PcmAudio:
    samples = np.linspace(-0.25, 0.25, frames, dtype=np.float32)
    return PcmAudio(AudioFormat(sample_rate_hz, 1), frames, samples.tobytes())


def _samples(audio: PcmAudio) -> np.ndarray:
    return np.frombuffer(audio.data, dtype=np.float32).reshape(
        audio.frame_count,
        audio.format.channels,
    )


def test_rubberband_stretcher_requests_r3_exact_duration_without_pitch_shift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[list[str]] = []
    expected = np.linspace(-0.4, 0.4, 120, dtype=np.float32).reshape(60, 2)

    def fake_run(command, **kwargs) -> None:
        commands.append(list(command))
        assert kwargs["check"] is True
        wavfile.write(Path(command[-1]), 48_000, expected)

    monkeypatch.setattr(
        "streammuse.infrastructure.rap.time_stretch.subprocess.run",
        fake_run,
    )

    result = RubberBandTimeStretcher(binary="rubberband").stretch(
        _stereo_pcm(100),
        60,
    )

    assert result.frame_count == 60
    assert np.array_equal(_samples(result), expected)
    assert len(commands) == 1
    command = commands[0]
    assert "--fine" in command
    assert command[command.index("--duration") + 1] == f"{60 / 48_000:.12f}"
    assert "--pitch" not in command
    assert "--frequency" not in command


def test_rubberband_time_map_stretcher_uses_one_r3_warp_and_exact_map(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from streammuse.infrastructure.rap.time_stretch import RubberBandTimeMapStretcher

    commands: list[list[str]] = []
    maps: list[str] = []
    expected = np.linspace(-0.4, 0.4, 60, dtype=np.float32)

    def fake_run(command, **kwargs) -> None:
        commands.append(list(command))
        assert kwargs["check"] is True
        map_path = Path(command[command.index("--timemap") + 1])
        maps.append(map_path.read_text(encoding="utf-8"))
        wavfile.write(Path(command[-1]), 48_000, expected)

    monkeypatch.setattr(
        "streammuse.infrastructure.rap.time_stretch.subprocess.run",
        fake_run,
    )

    result = RubberBandTimeMapStretcher().stretch(
        _mono_pcm(100),
        60,
        ((0, 0), (50, 30), (99, 59)),
    )

    assert result.frame_count == 60
    assert np.array_equal(_samples(result).reshape(-1), expected)
    assert maps == ["0 0\n50 30\n99 59\n"]
    assert len(commands) == 1
    assert "--fine" in commands[0]
    assert "--pitch" not in commands[0]
    assert "--frequency" not in commands[0]


@pytest.mark.parametrize("raw_frames", (58, 62))
def test_rubberband_stretcher_corrects_only_a_two_frame_tail_discrepancy(
    monkeypatch: pytest.MonkeyPatch,
    raw_frames: int,
) -> None:
    raw = np.arange(raw_frames * 2, dtype=np.float32).reshape(raw_frames, 2)

    def fake_run(command, **kwargs) -> None:
        del kwargs
        wavfile.write(Path(command[-1]), 48_000, raw)

    monkeypatch.setattr(
        "streammuse.infrastructure.rap.time_stretch.subprocess.run",
        fake_run,
    )

    result = RubberBandTimeStretcher(binary="rubberband").stretch(
        _stereo_pcm(100),
        60,
    )
    result_samples = _samples(result)

    assert result.frame_count == 60
    retained = min(raw_frames, 60)
    assert np.array_equal(result_samples[:retained], raw[:retained])
    if raw_frames < 60:
        assert np.array_equal(result_samples[raw_frames:], np.zeros((2, 2), dtype=np.float32))


def test_rubberband_stretcher_retries_a_material_length_error_without_resampling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[list[str]] = []
    first = np.arange(140, dtype=np.float32).reshape(70, 2)
    second = np.arange(120, dtype=np.float32).reshape(60, 2)

    def fake_run(command, **kwargs) -> None:
        del kwargs
        commands.append(list(command))
        wavfile.write(Path(command[-1]), 48_000, first if len(commands) == 1 else second)

    monkeypatch.setattr(
        "streammuse.infrastructure.rap.time_stretch.subprocess.run",
        fake_run,
    )

    result = RubberBandTimeStretcher(binary="rubberband").stretch(
        _stereo_pcm(100),
        60,
    )

    assert len(commands) == 2
    assert np.array_equal(_samples(result), second)
    first_duration = float(commands[0][commands[0].index("--duration") + 1])
    second_duration = float(commands[1][commands[1].index("--duration") + 1])
    assert second_duration == pytest.approx(first_duration * 60 / 70)


def test_rubberband_stretcher_retries_empty_output_without_dividing_by_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    expected = np.arange(120, dtype=np.float32).reshape(60, 2)

    def fake_run(command, **kwargs) -> None:
        nonlocal calls
        del kwargs
        calls += 1
        output = np.empty((0, 2), dtype=np.float32) if calls == 1 else expected
        wavfile.write(Path(command[-1]), 48_000, output)

    monkeypatch.setattr(
        "streammuse.infrastructure.rap.time_stretch.subprocess.run",
        fake_run,
    )

    result = RubberBandTimeStretcher(binary="rubberband").stretch(
        _stereo_pcm(100),
        60,
    )

    assert calls == 2
    assert np.array_equal(_samples(result), expected)


@pytest.mark.parametrize("raw_frames", (0, 50, 70))
def test_rubberband_stretcher_rejects_a_material_error_after_retry(
    monkeypatch: pytest.MonkeyPatch,
    raw_frames: int,
) -> None:
    calls = 0

    def fake_run(command, **kwargs) -> None:
        nonlocal calls
        del kwargs
        calls += 1
        wavfile.write(
            Path(command[-1]),
            48_000,
            np.zeros((raw_frames, 2), dtype=np.float32),
        )

    monkeypatch.setattr(
        "streammuse.infrastructure.rap.time_stretch.subprocess.run",
        fake_run,
    )

    with pytest.raises(RuntimeError, match=f"expected 60 frames, got {raw_frames}"):
        RubberBandTimeStretcher(binary="rubberband").stretch(_stereo_pcm(100), 60)
    assert calls == 2


@pytest.mark.skipif(shutil.which("rubberband") is None, reason="rubberband is required")
def test_real_rubberband_stretcher_preserves_a_tones_fundamental_pitch() -> None:
    sample_rate_hz = 48_000
    time = np.arange(sample_rate_hz, dtype=np.float32) / sample_rate_hz
    mono = np.sin(2.0 * np.pi * 220.0 * time).astype(np.float32)
    stereo = np.repeat(mono[:, np.newaxis], 2, axis=1)
    source = PcmAudio(AudioFormat(sample_rate_hz, 2), sample_rate_hz, stereo.tobytes())

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        stretched = RubberBandTimeStretcher().stretch(source, sample_rate_hz // 2)
    signal = _samples(stretched)[:, 0]
    spectrum = np.abs(np.fft.rfft(signal * np.hanning(len(signal))))
    frequencies = np.fft.rfftfreq(len(signal), 1.0 / sample_rate_hz)
    dominant_hz = float(frequencies[int(np.argmax(spectrum))])

    assert dominant_hz == pytest.approx(220.0, abs=2.0)
    assert not [item for item in caught if issubclass(item.category, WavFileWarning)]
