from __future__ import annotations

import numpy as np
import pytest

from streammuse.infrastructure.voice.audio import (
    PortAudioClockMapper,
    resample_float32,
)


def test_resample_float32_supports_arbitrary_rate_pairs() -> None:
    source = np.linspace(-1.0, 1.0, 480, dtype=np.float32)

    result = resample_float32(source, 48_000, 24_000)

    assert result.dtype == np.float32
    assert result.flags.c_contiguous
    assert result.size == pytest.approx(240, abs=1)


def test_portaudio_clock_mapper_uses_selected_buffer_field() -> None:
    mapper = PortAudioClockMapper("outputBufferDacTime")

    mapped = mapper.buffer_start(
        {"currentTime": 10.0, "outputBufferDacTime": 10.025},
        callback_local_s=20.0,
    )

    assert mapped == pytest.approx(20.025)


def test_portaudio_clock_mapper_falls_back_to_callback_clock() -> None:
    mapper = PortAudioClockMapper("inputBufferAdcTime")

    start, end = mapper.capture_interval(
        {},
        callback_local_s=5.0,
        frame_count=160,
        sample_rate=16_000,
    )

    assert start == pytest.approx(4.99)
    assert end == pytest.approx(5.0)
