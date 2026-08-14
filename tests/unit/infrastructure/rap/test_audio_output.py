"""Tests for realtime rap audio output adapters."""

from __future__ import annotations

import struct
import time
from pathlib import Path

import numpy as np
import pytest

from streammuse.domain.rap import (
    AudioFormat,
    AudioPlaybackNoticeKind,
    PcmAudio,
    PlaybackState,
    PreparedRapBar,
)
from streammuse.infrastructure.rap.audio_output import (
    CompositeAudioSink,
    Float32WavAudioSink,
    NullAudioSink,
    SoundDeviceAudioSink,
    TimedAudioSink,
)


class FakeOutputStream:
    def __init__(self, callback, block_frames: int) -> None:
        self.callback = callback
        self.block_frames = block_frames
        self.started = False
        self.stopped = False
        self.closed = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def close(self) -> None:
        self.closed = True


class FakeOutputStreamFactory:
    def __init__(self, *, block_frames: int) -> None:
        self.block_frames = block_frames
        self.stream: FakeOutputStream | None = None

    def __call__(self, *, audio_format: AudioFormat, callback) -> FakeOutputStream:
        self.stream = FakeOutputStream(callback, self.block_frames)
        return self.stream

    def render_frames(self, frame_count: int, *, status=0) -> list[float]:
        assert self.stream is not None
        rendered: list[float] = []
        remaining = frame_count
        while remaining:
            frames = min(self.block_frames, remaining)
            block = np.zeros((frames, 2), dtype=np.float32)
            self.stream.callback(block, frames, None, status)
            rendered.extend(block.reshape(-1).tolist())
            remaining -= frames
        return rendered


def stereo_format(sample_rate_hz: int = 48_000) -> AudioFormat:
    return AudioFormat(sample_rate_hz=sample_rate_hz, channels=2)


def prepared_bar(*, bar: int, frames: int, value: float = 0.0, audio_format: AudioFormat | None = None) -> PreparedRapBar:
    selected_format = audio_format or stereo_format()
    samples = np.full((frames, selected_format.channels), value, dtype=np.float32)
    return PreparedRapBar(
        bar=bar,
        text="test",
        source="test",
        fallback_reason=None,
        scheduled=(),
        audio=PcmAudio(selected_format, frames, samples.tobytes()),
        diagnostics=(),
        warnings=(),
        render_latency_ms=0.0,
    )


def read_float_wav(path: Path) -> tuple[tuple[object, ...], bytes]:
    raw = path.read_bytes()
    return struct.unpack("<4sI4s4sIHHIIHH4sI", raw[:44]), raw[44:]


def wait_until(predicate, *, timeout_seconds: float = 1.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("condition did not become true before timeout")
        time.sleep(0.002)


def test_sounddevice_callback_copies_bar_bytes_across_arbitrary_blocks() -> None:
    stream_factory = FakeOutputStreamFactory(block_frames=7)
    sink = SoundDeviceAudioSink(audio_format=stereo_format(), stream_factory=stream_factory)
    sink.enqueue(prepared_bar(bar=0, frames=10, value=0.25))
    sink.enqueue(prepared_bar(bar=1, frames=10, value=0.50))

    sink.start()
    rendered = stream_factory.render_frames(20)

    assert rendered[:20] == pytest.approx([0.25] * 20)
    assert rendered[20:] == pytest.approx([0.50] * 20)
    assert [notice.kind for notice in sink.drain_notices()] == [
        AudioPlaybackNoticeKind.BAR_STARTED,
        AudioPlaybackNoticeKind.BAR_COMPLETED,
        AudioPlaybackNoticeKind.BAR_STARTED,
        AudioPlaybackNoticeKind.BAR_COMPLETED,
    ]


def test_sounddevice_callback_fills_silence_and_reports_status_underrun() -> None:
    stream_factory = FakeOutputStreamFactory(block_frames=4)
    sink = SoundDeviceAudioSink(audio_format=stereo_format(), stream_factory=stream_factory)

    sink.start()
    rendered = stream_factory.render_frames(4, status="output underflow")

    assert rendered == [0.0] * 8
    assert sink.snapshot().absolute_frame == 4
    assert [notice.kind for notice in sink.drain_notices()] == [AudioPlaybackNoticeKind.UNDERRUN]


def test_stop_request_finishes_current_bar_without_dequeuing_next() -> None:
    stream_factory = FakeOutputStreamFactory(block_frames=5)
    sink = SoundDeviceAudioSink(audio_format=stereo_format(), stream_factory=stream_factory)
    sink.enqueue(prepared_bar(bar=0, frames=16))
    sink.enqueue(prepared_bar(bar=1, frames=16))
    sink.start()
    stream_factory.render_frames(1)
    sink.request_stop_after_bar()

    stream_factory.render_frames(31)

    snapshot = sink.snapshot()
    assert snapshot.state == PlaybackState.STOPPED
    assert snapshot.absolute_frame == 16
    assert snapshot.queue_depth == 1


def test_float32_wav_sink_writes_ieee_float_header_and_exact_pcm(tmp_path: Path) -> None:
    path = tmp_path / "session.wav"
    sink = Float32WavAudioSink(path, stereo_format())
    bar = prepared_bar(bar=0, frames=8, value=0.25)

    sink.enqueue(bar)
    sink.mark_completed(bar)
    sink.close()

    header, payload = read_float_wav(path)
    assert header[0] == b"RIFF"
    assert header[2] == b"WAVE"
    assert header[5] == 3
    assert header[6] == 2
    assert header[7] == 48_000
    assert header[10] == 32
    assert header[12] == len(bar.audio.data)
    assert payload == bar.audio.data


def test_composite_commits_only_completed_bar_bytes_on_stop(tmp_path: Path) -> None:
    stream_factory = FakeOutputStreamFactory(block_frames=8)
    live = SoundDeviceAudioSink(audio_format=stereo_format(), stream_factory=stream_factory)
    recorder = Float32WavAudioSink(tmp_path / "session.wav", stereo_format())
    composite = CompositeAudioSink(live, recorder)
    first = prepared_bar(bar=0, frames=8, value=0.25)
    composite.enqueue(first)
    composite.enqueue(prepared_bar(bar=1, frames=8, value=0.50))
    composite.start()
    stream_factory.render_frames(1)
    composite.request_stop_after_bar()

    stream_factory.render_frames(15)
    composite.close()

    _, payload = read_float_wav(tmp_path / "session.wav")
    assert payload == first.audio.data


def test_null_sink_records_and_completes_bars_without_opening_a_device() -> None:
    sink = NullAudioSink(audio_format=stereo_format())
    bar = prepared_bar(bar=0, frames=6)

    sink.start()
    sink.enqueue(bar)
    sink.complete_next()

    assert sink.recorded_bars == (bar,)
    assert sink.snapshot().absolute_frame == 6
    assert [notice.kind for notice in sink.drain_notices()] == [
        AudioPlaybackNoticeKind.BAR_STARTED,
        AudioPlaybackNoticeKind.BAR_COMPLETED,
    ]


def test_timed_sink_advances_realtime_clock_and_stops_after_current_bar() -> None:
    audio_format = stereo_format(sample_rate_hz=100)
    sink = TimedAudioSink(audio_format=audio_format, poll_interval_seconds=0.001)
    sink.enqueue(prepared_bar(bar=0, frames=10, audio_format=audio_format))
    sink.enqueue(prepared_bar(bar=1, frames=10, audio_format=audio_format))

    try:
        sink.start()
        wait_until(lambda: sink.snapshot().frame_in_bar >= 3)
        sink.request_stop_after_bar()
        wait_until(lambda: sink.snapshot().state == PlaybackState.STOPPED)

        snapshot = sink.snapshot()
        assert snapshot.absolute_frame == 10
        assert snapshot.queue_depth == 1

        sink.reset()
        assert sink.snapshot().state == PlaybackState.STOPPED
        assert sink.snapshot().absolute_frame == 0
        assert sink.snapshot().queue_depth == 0
    finally:
        sink.close()
