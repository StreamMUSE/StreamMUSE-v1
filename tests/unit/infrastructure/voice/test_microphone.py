from __future__ import annotations

import threading
import time
import builtins
from typing import Any

import numpy as np
import pytest

from streammuse.application.tasks.human_input import VoiceInputConfig
from streammuse.infrastructure.voice import (
    AudioQueueOverflowError,
    MicrophoneCapture,
    MicrophoneCaptureError,
    MicrophoneDeviceError,
    VoiceDependencyError,
    enumerate_input_devices,
)
from streammuse.infrastructure.voice import microphone as microphone_module


FRAME_SAMPLES_16K = 320


def _pcm_frame(value: int) -> bytes:
    return np.full(FRAME_SAMPLES_16K, value, dtype="<i2").tobytes()


def _pcm_frame_at_rate(value: int, sample_rate: int) -> bytes:
    return np.full(sample_rate * 20 // 1000, value, dtype="<i2").tobytes()


class EnergyVad:
    def __init__(self, aggressiveness: int) -> None:
        self.aggressiveness = aggressiveness
        self.frame_sizes: list[int] = []

    def is_speech(self, frame: bytes, sample_rate: int) -> bool:
        assert len(frame) == sample_rate * 20 // 1000 * 2
        self.frame_sizes.append(len(frame))
        return bool(np.any(np.frombuffer(frame, dtype="<i2")))


class FakeStream:
    def __init__(
        self,
        callback: Any,
        chunks: list[bytes],
        statuses: list[Any] | None = None,
        time_infos: list[Any] | None = None,
    ) -> None:
        self.callback = callback
        self.chunks = chunks
        self.statuses = statuses or [None] * len(chunks)
        self.time_infos = time_infos or [{} for _ in chunks]
        self.started = 0
        self.stopped = 0
        self.closed = 0

    def start(self) -> None:
        self.started += 1
        for chunk, status, time_info in zip(self.chunks, self.statuses, self.time_infos):
            self.callback(chunk, len(chunk) // 2, time_info, status)

    def stop(self) -> None:
        self.stopped += 1

    def close(self) -> None:
        self.closed += 1


class FakeSoundDevice:
    def __init__(
        self,
        *,
        chunks: list[bytes] | None = None,
        statuses: list[Any] | None = None,
        time_infos: list[Any] | None = None,
        supported_rates: set[int] | None = None,
    ) -> None:
        self.devices = [
            {"name": "Output only", "max_input_channels": 0, "default_samplerate": 48_000.0, "hostapi": 0},
            {"name": "Test Mic", "max_input_channels": 1, "default_samplerate": 44_100.0, "hostapi": 0},
        ]
        self.default = type("Default", (), {"device": (1, 0)})()
        self.chunks = chunks or []
        self.statuses = statuses
        self.time_infos = time_infos
        self.supported_rates = supported_rates if supported_rates is not None else {16_000}
        self.checked_rates: list[int] = []
        self.streams: list[FakeStream] = []
        self.stream_kwargs: dict[str, Any] | None = None

    def query_devices(self, device: Any = ..., kind: str | None = None) -> Any:
        del kind
        if device is ...:
            return self.devices
        if device is None:
            return self.devices[1]
        if isinstance(device, int):
            return self.devices[device]
        for candidate in self.devices:
            if candidate["name"] == device:
                return candidate
        raise ValueError("unknown device")

    def check_input_settings(self, **kwargs: Any) -> None:
        rate = int(kwargs["samplerate"])
        self.checked_rates.append(rate)
        if rate not in self.supported_rates:
            raise ValueError("unsupported")

    def RawInputStream(self, **kwargs: Any) -> FakeStream:
        self.stream_kwargs = kwargs
        stream = FakeStream(
            kwargs["callback"],
            self.chunks,
            self.statuses,
            self.time_infos,
        )
        self.streams.append(stream)
        return stream


class _InputOutputPairLike:
    """Match sounddevice 0.5.5's subscriptable, non-tuple default pair."""

    def __init__(self, input_device: int, output_device: int) -> None:
        self._values = (input_device, output_device)

    def __getitem__(self, index: int) -> int:
        return self._values[index]


class BadFrameCountStream(FakeStream):
    def start(self) -> None:
        self.started += 1
        for chunk in self.chunks:
            self.callback(chunk, len(chunk) // 2 + 1, {}, None)


class BadFrameCountSoundDevice(FakeSoundDevice):
    def RawInputStream(self, **kwargs: Any) -> FakeStream:
        self.stream_kwargs = kwargs
        stream = BadFrameCountStream(kwargs["callback"], self.chunks)
        self.streams.append(stream)
        return stream


class CleanupFailureStream(FakeStream):
    def stop(self) -> None:
        self.stopped += 1
        raise RuntimeError("cleanup stop failed")


class CaptureCleanupFailureSoundDevice(FakeSoundDevice):
    """Use a healthy preflight stream and a failing per-turn stream."""

    def RawInputStream(self, **kwargs: Any) -> FakeStream:
        self.stream_kwargs = kwargs
        stream_type = FakeStream if not self.streams else CleanupFailureStream
        stream = stream_type(
            kwargs["callback"],
            self.chunks,
            self.statuses,
            self.time_infos,
        )
        self.streams.append(stream)
        return stream


def _capture(sd: FakeSoundDevice, **overrides: Any) -> MicrophoneCapture:
    values = {
        "start_timeout_ms": 100.0,
        "end_silence_ms": 40.0,
        "max_utterance_ms": 200.0,
        "pre_roll_ms": 20.0,
    }
    values.update(overrides)
    config = VoiceInputConfig(**values)
    capture = MicrophoneCapture(config, sounddevice_module=sd, vad_factory=EnergyVad)
    capture.start()
    return capture


def test_enumerate_input_devices_filters_output_only_devices() -> None:
    devices = enumerate_input_devices(sounddevice_module=FakeSoundDevice())

    assert len(devices) == 1
    assert devices[0].index == 1
    assert devices[0].name == "Test Mic"
    assert devices[0].default_sample_rate_hz == 44_100.0


def test_preflight_prefers_16khz_then_falls_back_to_supported_vad_rate() -> None:
    sd = FakeSoundDevice(supported_rates={48_000})
    capture = _capture(sd)

    assert sd.checked_rates == [16_000, 48_000]
    assert capture.provenance["capture_sample_rate_hz"] == 48_000
    assert capture.provenance["microphone_device"]["name"] == "Test Mic"
    assert len(sd.streams) == 1
    assert sd.streams[0].started == 1
    assert sd.streams[0].stopped == 1
    assert sd.streams[0].closed == 1


def test_default_device_index_is_not_guessed_from_a_duplicate_device_name() -> None:
    sd = FakeSoundDevice()
    sd.devices[0] = {
        "name": "Test Mic",
        "max_input_channels": 1,
        "default_samplerate": 48_000.0,
        "hostapi": 0,
    }
    capture = _capture(sd)

    assert capture.provenance["microphone_device"]["index"] == 1


def test_default_device_accepts_sounddevice_input_output_pair_shape() -> None:
    sd = FakeSoundDevice()
    sd.default.device = _InputOutputPairLike(1, 0)

    capture = _capture(sd)

    assert capture.provenance["microphone_device"]["index"] == 1
    assert sd.stream_kwargs is not None
    assert sd.streams[0].started == 1
    assert sd.streams[0].stopped == 1
    assert sd.streams[0].closed == 1


def test_preflight_preserves_open_error_when_cleanup_also_fails() -> None:
    class BrokenStream(FakeStream):
        def start(self) -> None:
            self.started += 1
            raise PermissionError("microphone permission denied")

        def stop(self) -> None:
            self.stopped += 1
            raise RuntimeError("stop failed")

        def close(self) -> None:
            self.closed += 1
            raise RuntimeError("close failed")

    class BrokenSoundDevice(FakeSoundDevice):
        def RawInputStream(self, **kwargs: Any) -> FakeStream:
            stream = BrokenStream(kwargs["callback"], [])
            self.streams.append(stream)
            return stream

    sd = BrokenSoundDevice()
    capture = MicrophoneCapture(VoiceInputConfig(), sounddevice_module=sd, vad_factory=EnergyVad)

    with pytest.raises(MicrophoneDeviceError, match="permission denied") as exc_info:
        capture.start()

    assert isinstance(exc_info.value.__cause__, PermissionError)
    assert sd.streams[0].stopped == 1
    assert sd.streams[0].closed == 1


def test_preflight_rejects_device_without_supported_vad_rate() -> None:
    sd = FakeSoundDevice(supported_rates=set())
    capture = MicrophoneCapture(VoiceInputConfig(), sounddevice_module=sd, vad_factory=EnergyVad)

    with pytest.raises(MicrophoneDeviceError, match="44.1 kHz-only"):
        capture.start()


def test_variable_callback_chunks_are_accumulated_into_exact_vad_frames() -> None:
    data = _pcm_frame(0) + _pcm_frame(3000) + _pcm_frame(3000) + _pcm_frame(0) + _pcm_frame(0)
    sd = FakeSoundDevice(chunks=[data[:138], data[138:902], data[902:]])
    capture = _capture(sd)

    utterance = capture.capture(timeout_s=None)

    assert utterance.endpoint_reason == "trailing_silence"
    assert utterance.deadline_expired is False
    assert utterance.wait_for_speech_ms == pytest.approx(20.0, abs=5.0)
    assert utterance.utterance_ms == 40.0
    assert utterance.endpoint_silence_ms == 40.0
    assert utterance.capture_sample_rate_hz == 16_000
    assert utterance.sample_rate_hz == 16_000
    assert utterance.audio.dtype == np.float32
    assert utterance.audio.ndim == 1
    assert utterance.audio.size == FRAME_SAMPLES_16K * 5
    assert sd.stream_kwargs["blocksize"] == 0  # type: ignore[index]
    assert sd.stream_kwargs["device"] == 1  # type: ignore[index]
    assert sd.stream_kwargs["channels"] == 1  # type: ignore[index]
    assert sd.stream_kwargs["dtype"] == "int16"  # type: ignore[index]
    assert sd.streams[-1].stopped == 1
    assert sd.streams[-1].closed == 1


def test_48khz_capture_uses_exact_20ms_vad_frames_before_resampling() -> None:
    data = b"".join(
        [
            _pcm_frame_at_rate(0, 48_000),
            _pcm_frame_at_rate(3000, 48_000),
            _pcm_frame_at_rate(3000, 48_000),
            _pcm_frame_at_rate(0, 48_000),
            _pcm_frame_at_rate(0, 48_000),
        ]
    )
    sd = FakeSoundDevice(chunks=[data], supported_rates={48_000})
    capture = _capture(sd)

    utterance = capture.capture(timeout_s=None)

    assert utterance.capture_sample_rate_hz == 48_000
    assert utterance.sample_rate_hz == 16_000
    assert utterance.audio.shape == (1_600,)


def test_continuous_speech_stops_at_max_utterance() -> None:
    sd = FakeSoundDevice(chunks=[_pcm_frame(2000) * 3])
    capture = _capture(sd, max_utterance_ms=40.0, end_silence_ms=20.0, pre_roll_ms=0.0)

    utterance = capture.capture(timeout_s=None)

    assert utterance.endpoint_reason == "max_utterance"
    assert utterance.utterance_ms == 40.0


def test_late_nonfirst_chunk_is_not_accepted_past_max_utterance() -> None:
    class SequenceClock:
        def __init__(self) -> None:
            self.values = iter([0.0, 0.02, 6.0])

        def __call__(self) -> float:
            return next(self.values, 6.0)

    sd = FakeSoundDevice(chunks=[_pcm_frame(2000), _pcm_frame(2000)])
    capture = MicrophoneCapture(
        VoiceInputConfig(
            max_utterance_ms=100.0,
            end_silence_ms=40.0,
            pre_roll_ms=0.0,
        ),
        sounddevice_module=sd,
        vad_factory=EnergyVad,
        now=SequenceClock(),
    )
    capture.start()

    utterance = capture.capture(timeout_s=None)

    assert utterance.endpoint_reason == "max_utterance"
    assert utterance.has_speech is True
    assert utterance.audio.size == FRAME_SAMPLES_16K
    assert utterance.last_voiced_offset_ms == pytest.approx(20.0)
    assert utterance.utterance_ms == pytest.approx(20.0)


def test_max_utterance_wall_timeout_handles_a_stalled_stream() -> None:
    sd = FakeSoundDevice(chunks=[_pcm_frame(2000)])
    capture = _capture(sd, max_utterance_ms=20.0, end_silence_ms=20.0, pre_roll_ms=0.0)

    utterance = capture.capture(timeout_s=None)

    assert utterance.endpoint_reason == "max_utterance"
    assert utterance.has_speech is True


def test_game_deadline_without_speech_returns_no_audio() -> None:
    capture = _capture(FakeSoundDevice())

    utterance = capture.capture(timeout_s=0.0)

    assert utterance.endpoint_reason == "deadline"
    assert utterance.deadline_expired is True
    assert utterance.has_speech is False
    assert utterance.audio.dtype == np.float32


def test_predeadline_queued_audio_is_processed_after_consumer_reaches_deadline() -> None:
    class SequenceClock:
        def __init__(self) -> None:
            self.values = iter([0.0, 0.2, 2.0])

        def __call__(self) -> float:
            return next(self.values, 2.0)

    sd = FakeSoundDevice(chunks=[_pcm_frame(2000)])
    capture = MicrophoneCapture(
        VoiceInputConfig(
            start_timeout_ms=5000.0,
            end_silence_ms=300.0,
            max_utterance_ms=5000.0,
            pre_roll_ms=0.0,
        ),
        sounddevice_module=sd,
        vad_factory=EnergyVad,
        now=SequenceClock(),
    )
    capture.start()

    utterance = capture.capture(timeout_s=1.0)

    assert utterance.endpoint_reason == "deadline"
    assert utterance.deadline_expired is True
    assert utterance.has_speech is True
    assert utterance.audio.size == FRAME_SAMPLES_16K


def test_adc_time_preserves_predeadline_audio_from_a_delayed_callback() -> None:
    class SequenceClock:
        def __init__(self) -> None:
            self.values = iter([0.0, 2.0])

        def __call__(self) -> float:
            return next(self.values, 2.0)

    sd = FakeSoundDevice(
        chunks=[_pcm_frame(2000)],
        time_infos=[{"inputBufferAdcTime": 100.9, "currentTime": 102.0}],
    )
    capture = MicrophoneCapture(
        VoiceInputConfig(
            start_timeout_ms=5000.0,
            max_utterance_ms=5000.0,
            pre_roll_ms=0.0,
        ),
        sounddevice_module=sd,
        vad_factory=EnergyVad,
        now=SequenceClock(),
    )
    capture.start()

    utterance = capture.capture(timeout_s=1.0)

    assert utterance.endpoint_reason == "deadline"
    assert utterance.deadline_expired is True
    assert utterance.has_speech is True
    assert utterance.wait_for_speech_ms == pytest.approx(900.0)
    assert utterance.audio.size == FRAME_SAMPLES_16K


def test_stream_clock_calibration_maps_adc_time_without_callback_current_time() -> None:
    class ClockedStream(FakeStream):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.stream_times = iter([200.0, 202.0])

        @property
        def time(self) -> float:
            return next(self.stream_times)

    class ClockedSoundDevice(FakeSoundDevice):
        def RawInputStream(self, **kwargs: Any) -> FakeStream:
            stream = ClockedStream(
                kwargs["callback"],
                self.chunks,
                self.statuses,
                self.time_infos,
            )
            self.streams.append(stream)
            return stream

    class SequenceClock:
        def __init__(self) -> None:
            self.values = iter([10.0, 10.0, 10.0, 12.0, 12.0, 12.0, 12.0])

        def __call__(self) -> float:
            return next(self.values, 12.0)

    sd = ClockedSoundDevice(
        chunks=[_pcm_frame(2000)],
        time_infos=[{"inputBufferAdcTime": 200.5}],
    )
    capture = MicrophoneCapture(
        VoiceInputConfig(
            start_timeout_ms=5000.0,
            max_utterance_ms=5000.0,
            pre_roll_ms=0.0,
        ),
        sounddevice_module=sd,
        vad_factory=EnergyVad,
        now=SequenceClock(),
    )
    capture.start()

    utterance = capture.capture(timeout_s=1.0)

    assert utterance.endpoint_reason == "deadline"
    assert utterance.deadline_expired is True
    assert utterance.has_speech is True
    assert utterance.wait_for_speech_ms == pytest.approx(500.0)
    assert utterance.audio.size == FRAME_SAMPLES_16K


def test_adc_time_rejects_audio_captured_after_deadline() -> None:
    class CDataTimeInfo:
        inputBufferAdcTime = 101.1
        currentTime = 102.0

    class SequenceClock:
        def __init__(self) -> None:
            self.values = iter([0.0, 2.0])

        def __call__(self) -> float:
            return next(self.values, 2.0)

    sd = FakeSoundDevice(
        chunks=[_pcm_frame(2000)],
        time_infos=[CDataTimeInfo()],
    )
    capture = MicrophoneCapture(
        VoiceInputConfig(pre_roll_ms=0.0),
        sounddevice_module=sd,
        vad_factory=EnergyVad,
        now=SequenceClock(),
    )
    capture.start()

    utterance = capture.capture(timeout_s=1.0)

    assert utterance.endpoint_reason == "deadline"
    assert utterance.deadline_expired is True
    assert utterance.has_speech is False
    assert utterance.audio.size == 0


def test_adc_time_preserves_speech_captured_before_start_timeout() -> None:
    class SequenceClock:
        def __init__(self) -> None:
            self.values = iter([0.0, 2.0])

        def __call__(self) -> float:
            return next(self.values, 2.0)

    data = _pcm_frame(2000) + _pcm_frame(0) + _pcm_frame(0)
    sd = FakeSoundDevice(
        chunks=[data],
        time_infos=[{"inputBufferAdcTime": 100.5, "currentTime": 102.0}],
    )
    capture = MicrophoneCapture(
        VoiceInputConfig(
            start_timeout_ms=1000.0,
            end_silence_ms=40.0,
            pre_roll_ms=0.0,
        ),
        sounddevice_module=sd,
        vad_factory=EnergyVad,
        now=SequenceClock(),
    )
    capture.start()

    utterance = capture.capture(timeout_s=None)

    assert utterance.endpoint_reason == "trailing_silence"
    assert utterance.has_speech is True
    assert utterance.wait_for_speech_ms == pytest.approx(500.0)


def test_adc_time_rejects_nonfirst_frame_captured_after_max_utterance() -> None:
    class SequenceClock:
        def __init__(self) -> None:
            self.values = iter([0.0, 2.0, 2.0])

        def __call__(self) -> float:
            return next(self.values, 2.0)

    sd = FakeSoundDevice(
        chunks=[_pcm_frame(2000), _pcm_frame(2000)],
        time_infos=[
            {"inputBufferAdcTime": 100.0, "currentTime": 102.0},
            {"inputBufferAdcTime": 100.2, "currentTime": 102.2},
        ],
    )
    capture = MicrophoneCapture(
        VoiceInputConfig(
            max_utterance_ms=100.0,
            end_silence_ms=40.0,
            pre_roll_ms=0.0,
        ),
        sounddevice_module=sd,
        vad_factory=EnergyVad,
        now=SequenceClock(),
    )
    capture.start()

    utterance = capture.capture(timeout_s=None)

    assert utterance.endpoint_reason == "max_utterance"
    assert utterance.audio.size == FRAME_SAMPLES_16K
    assert utterance.last_voiced_offset_ms == pytest.approx(20.0)


def test_audio_chunk_entirely_after_deadline_is_discarded() -> None:
    class SequenceClock:
        def __init__(self) -> None:
            self.values = iter([0.0, 1.2, 1.2])

        def __call__(self) -> float:
            return next(self.values, 1.2)

    sd = FakeSoundDevice(chunks=[_pcm_frame(2000)])
    capture = MicrophoneCapture(
        VoiceInputConfig(),
        sounddevice_module=sd,
        vad_factory=EnergyVad,
        now=SequenceClock(),
    )
    capture.start()

    utterance = capture.capture(timeout_s=1.0)

    assert utterance.endpoint_reason == "deadline"
    assert utterance.deadline_expired is True
    assert utterance.has_speech is False
    assert utterance.audio.size == 0


def test_audio_chunk_straddling_deadline_is_truncated() -> None:
    class SequenceClock:
        def __init__(self) -> None:
            self.values = iter([0.0, 1.03, 1.03])

        def __call__(self) -> float:
            return next(self.values, 1.03)

    frames = _pcm_frame(2000) * 4
    sd = FakeSoundDevice(chunks=[frames])
    capture = MicrophoneCapture(
        VoiceInputConfig(pre_roll_ms=0.0),
        sounddevice_module=sd,
        vad_factory=EnergyVad,
        now=SequenceClock(),
    )
    capture.start()

    utterance = capture.capture(timeout_s=1.0)

    assert utterance.endpoint_reason == "deadline"
    assert utterance.deadline_expired is True
    assert utterance.has_speech is True
    assert 0 < utterance.audio.size < FRAME_SAMPLES_16K * 4
    assert utterance.audio.size % FRAME_SAMPLES_16K == 0


def test_soft_safety_timeout_is_not_a_game_deadline() -> None:
    config = VoiceInputConfig(start_timeout_ms=10.0)
    capture = MicrophoneCapture(
        config,
        sounddevice_module=FakeSoundDevice(),
        vad_factory=EnergyVad,
    )
    capture.start()

    utterance = capture.capture(timeout_s=None)

    assert utterance.endpoint_reason == "start_timeout"
    assert utterance.deadline_expired is False


def test_first_callback_delay_is_reflected_in_wait_for_speech_offset() -> None:
    class SequenceClock:
        def __init__(self) -> None:
            self.values = iter([0.0, 0.86])

        def __call__(self) -> float:
            return next(self.values, 0.86)

    data = _pcm_frame(2000) + _pcm_frame(0) + _pcm_frame(0)
    sd = FakeSoundDevice(chunks=[data])
    capture = MicrophoneCapture(
        VoiceInputConfig(
            start_timeout_ms=1000.0,
            end_silence_ms=40.0,
            pre_roll_ms=0.0,
        ),
        sounddevice_module=sd,
        vad_factory=EnergyVad,
        now=SequenceClock(),
    )
    capture.start()

    utterance = capture.capture(timeout_s=None)

    assert utterance.endpoint_reason == "trailing_silence"
    assert utterance.wait_for_speech_ms == pytest.approx(800.0)


def test_voiced_chunk_captured_after_start_timeout_is_rejected() -> None:
    class SequenceClock:
        def __init__(self) -> None:
            self.values = iter([0.0, 1.06])

        def __call__(self) -> float:
            return next(self.values, 1.06)

    sd = FakeSoundDevice(chunks=[_pcm_frame(2000)])
    capture = MicrophoneCapture(
        VoiceInputConfig(start_timeout_ms=1000.0),
        sounddevice_module=sd,
        vad_factory=EnergyVad,
        now=SequenceClock(),
    )
    capture.start()

    utterance = capture.capture(timeout_s=None)

    assert utterance.endpoint_reason == "start_timeout"
    assert utterance.deadline_expired is False
    assert utterance.has_speech is False
    assert utterance.wait_for_speech_ms == pytest.approx(1060.0)


@pytest.mark.parametrize("first_chunk", [_pcm_frame(0), _pcm_frame(0)[:320]])
def test_late_nonfirst_voiced_chunk_is_rejected_after_start_timeout(
    first_chunk: bytes,
) -> None:
    class SequenceClock:
        def __init__(self) -> None:
            first_duration_s = len(first_chunk) / 2 / 16_000
            self.values = iter([0.0, first_duration_s, 1.2])

        def __call__(self) -> float:
            return next(self.values, 1.2)

    sd = FakeSoundDevice(chunks=[first_chunk, _pcm_frame(2000)])
    capture = MicrophoneCapture(
        VoiceInputConfig(start_timeout_ms=1000.0),
        sounddevice_module=sd,
        vad_factory=EnergyVad,
        now=SequenceClock(),
    )
    capture.start()

    utterance = capture.capture(timeout_s=None)

    assert utterance.endpoint_reason == "start_timeout"
    assert utterance.deadline_expired is False
    assert utterance.has_speech is False
    assert utterance.wait_for_speech_ms == pytest.approx(1200.0)


def test_speech_started_before_start_timeout_may_continue_after_it() -> None:
    class SequenceClock:
        def __init__(self) -> None:
            self.values = iter([0.0, 1.03])

        def __call__(self) -> float:
            return next(self.values, 1.03)

    data = _pcm_frame(0) + _pcm_frame(2000) + _pcm_frame(2000) + _pcm_frame(0)
    sd = FakeSoundDevice(chunks=[data])
    capture = MicrophoneCapture(
        VoiceInputConfig(
            start_timeout_ms=1000.0,
            end_silence_ms=20.0,
            pre_roll_ms=0.0,
        ),
        sounddevice_module=sd,
        vad_factory=EnergyVad,
        now=SequenceClock(),
    )
    capture.start()

    utterance = capture.capture(timeout_s=None)

    assert utterance.endpoint_reason == "trailing_silence"
    assert utterance.deadline_expired is False
    assert utterance.has_speech is True
    assert utterance.wait_for_speech_ms == pytest.approx(970.0)
    assert utterance.utterance_ms == 40.0


def test_callback_queue_overflow_is_raised_not_silently_dropped() -> None:
    sd = FakeSoundDevice(chunks=[_pcm_frame(1000), _pcm_frame(1000)])
    capture = _capture(sd, queue_max_chunks=1)

    with pytest.raises(AudioQueueOverflowError, match="overflowed"):
        capture.capture(timeout_s=None)


def test_callback_status_is_forwarded_to_capture_thread() -> None:
    sd = FakeSoundDevice(chunks=[_pcm_frame(1000)], statuses=["input overflow"])
    capture = _capture(sd)

    with pytest.raises(MicrophoneCaptureError, match="PortAudio callback status"):
        capture.capture(timeout_s=None)


def test_successful_capture_reports_typed_stream_cleanup_error() -> None:
    data = _pcm_frame(2000) + _pcm_frame(0) + _pcm_frame(0)
    sd = CaptureCleanupFailureSoundDevice(chunks=[data])
    capture = _capture(sd, pre_roll_ms=0.0)

    with pytest.raises(MicrophoneCaptureError, match="cleanup stop failed") as exc_info:
        capture.capture(timeout_s=None)

    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert sd.streams[-1].stopped == 1
    assert sd.streams[-1].closed == 1


def test_capture_error_is_preserved_when_stream_cleanup_also_fails() -> None:
    sd = CaptureCleanupFailureSoundDevice(
        chunks=[_pcm_frame(1000)],
        statuses=["input overflow"],
    )
    capture = _capture(sd)

    with pytest.raises(MicrophoneCaptureError, match="PortAudio callback status"):
        capture.capture(timeout_s=None)

    assert sd.streams[-1].stopped == 1
    assert sd.streams[-1].closed == 1


def test_inconsistent_portaudio_frame_count_is_rejected_on_main_thread() -> None:
    sd = BadFrameCountSoundDevice(chunks=[_pcm_frame(1000)])
    capture = _capture(sd)

    with pytest.raises(MicrophoneCaptureError, match="inconsistent mono int16 buffer"):
        capture.capture(timeout_s=None)


def test_close_unblocks_capture_and_is_idempotent() -> None:
    sd = FakeSoundDevice()
    capture = _capture(sd)
    errors: list[BaseException] = []

    def run_capture() -> None:
        try:
            capture.capture(timeout_s=None)
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=run_capture)
    thread.start()
    deadline = time.monotonic() + 1.0
    while len(sd.streams) < 2 and time.monotonic() < deadline:
        time.sleep(0.001)
    capture.close()
    capture.close()
    thread.join(timeout=1.0)

    assert not thread.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], MicrophoneCaptureError)
    assert sd.streams[-1].stopped == 1
    assert sd.streams[-1].closed == 1


def test_close_reports_typed_cleanup_error_and_remains_idempotent() -> None:
    sd = CaptureCleanupFailureSoundDevice()
    capture = _capture(sd)
    capture_errors: list[BaseException] = []

    def run_capture() -> None:
        try:
            capture.capture(timeout_s=None)
        except BaseException as exc:
            capture_errors.append(exc)

    thread = threading.Thread(target=run_capture)
    thread.start()
    deadline = time.monotonic() + 1.0
    while len(sd.streams) < 2 and time.monotonic() < deadline:
        time.sleep(0.001)

    with pytest.raises(MicrophoneCaptureError, match="cleanup stop failed") as exc_info:
        capture.close()
    capture.close()
    thread.join(timeout=1.0)

    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert not thread.is_alive()
    assert len(capture_errors) == 1
    assert isinstance(capture_errors[0], MicrophoneCaptureError)
    assert "closed while waiting" in str(capture_errors[0])
    assert sd.streams[-1].stopped == 1
    assert sd.streams[-1].closed == 1


def test_close_winning_capture_start_race_prevents_a_new_stream() -> None:
    class BlockingClock:
        def __init__(self) -> None:
            self.entered = threading.Event()
            self.release = threading.Event()

        def __call__(self) -> float:
            if not self.entered.is_set():
                self.entered.set()
                assert self.release.wait(timeout=1.0)
            return 0.0

    sd = FakeSoundDevice()
    clock = BlockingClock()
    capture = MicrophoneCapture(
        VoiceInputConfig(),
        sounddevice_module=sd,
        vad_factory=EnergyVad,
        now=clock,
    )
    capture.start()
    capture_errors: list[BaseException] = []

    def run_capture() -> None:
        try:
            capture.capture(timeout_s=None)
        except BaseException as exc:
            capture_errors.append(exc)

    thread = threading.Thread(target=run_capture)
    thread.start()
    assert clock.entered.wait(timeout=1.0)

    capture.close()
    clock.release.set()
    thread.join(timeout=1.0)

    assert not thread.is_alive()
    assert len(capture_errors) == 1
    assert isinstance(capture_errors[0], MicrophoneCaptureError)
    assert "closed before" in str(capture_errors[0])
    assert len(sd.streams) == 1
    assert sd.streams[0].closed == 1


@pytest.mark.parametrize("module_name", ["sounddevice", "webrtcvad"])
def test_optional_audio_import_oserror_is_wrapped_as_dependency_error(
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
) -> None:
    original_import = builtins.__import__

    def broken_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == module_name:
            raise OSError("shared library missing")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", broken_import)
    importer = (
        microphone_module._import_sounddevice
        if module_name == "sounddevice"
        else microphone_module._import_vad_factory
    )

    with pytest.raises(VoiceDependencyError, match="voice extra") as exc_info:
        importer()

    assert isinstance(exc_info.value.__cause__, OSError)


def test_pcm_resampling_returns_expected_length_and_float32() -> None:
    pcm = np.full(4_800, 1000, dtype="<i2").tobytes()

    audio = MicrophoneCapture._pcm_to_float32(pcm, 48_000)

    assert audio.dtype == np.float32
    assert audio.shape == (1_600,)
    assert np.all(np.isfinite(audio))
