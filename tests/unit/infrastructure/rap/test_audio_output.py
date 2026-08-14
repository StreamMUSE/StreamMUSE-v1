"""Tests for realtime rap audio output adapters."""

from __future__ import annotations

import struct
import sys
import time
from pathlib import Path
from threading import Event, Thread

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
        self.stop_calls = 0
        self.callback_terminations = 0

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True
        self.stop_calls += 1

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
            try:
                self.stream.callback(block, frames, None, status)
            except BaseException as error:
                if error.__class__.__name__ not in {"CallbackStop", "CallbackTerminated"}:
                    raise
                self.stream.callback_terminations += 1
            rendered.extend(block.reshape(-1).tolist())
            remaining -= frames
        return rendered


class CallbackTerminated(Exception):
    """Injected stand-in for sounddevice.CallbackStop."""


class BlockingOutputBuffer:
    """Blocks precisely after callback PCM assignment and before state commit."""

    def __init__(self, frames: int, channels: int) -> None:
        self.samples = np.zeros((frames, channels), dtype=np.float32)
        self.copy_started = Event()
        self.allow_commit = Event()

    def fill(self, value: float) -> None:
        self.samples.fill(value)

    def __setitem__(self, key, value) -> None:
        self.copy_started.set()
        assert self.allow_commit.wait(timeout=1.0)
        self.samples[key] = value


class BlockingStartStream(FakeOutputStream):
    """Lets a lifecycle transition complete while stream.start is in flight."""

    def __init__(self, callback, block_frames: int) -> None:
        super().__init__(callback, block_frames)
        self.start_entered = Event()
        self.allow_start = Event()
        self.active = False
        self.start_calls = 0

    def start(self) -> None:
        self.start_calls += 1
        if self.start_calls == 1:
            self.start_entered.set()
            assert self.allow_start.wait(timeout=1.0)
        self.started = True
        self.active = True

    def stop(self) -> None:
        super().stop()
        self.active = False

    def close(self) -> None:
        super().close()
        self.active = False


class BlockingStartStreamFactory:
    def __init__(self) -> None:
        self.stream: BlockingStartStream | None = None

    def __call__(self, *, audio_format: AudioFormat, callback) -> BlockingStartStream:
        self.stream = BlockingStartStream(callback, block_frames=4)
        return self.stream


class ActiveOutputStream(FakeOutputStream):
    def __init__(self, callback, block_frames: int) -> None:
        super().__init__(callback, block_frames)
        self.active = False

    def start(self) -> None:
        super().start()
        self.active = True

    def stop(self) -> None:
        super().stop()
        self.active = False

    def close(self) -> None:
        super().close()
        self.active = False


class ActiveOutputStreamFactory:
    def __init__(self) -> None:
        self.streams: list[ActiveOutputStream] = []

    def __call__(self, *, audio_format: AudioFormat, callback) -> ActiveOutputStream:
        stream = ActiveOutputStream(callback, block_frames=4)
        self.streams.append(stream)
        return stream


class StartABARaceFactory:
    def __init__(self) -> None:
        self.first: BlockingStartStream | None = None
        self.second: ActiveOutputStream | None = None

    def __call__(self, *, audio_format: AudioFormat, callback) -> FakeOutputStream:
        if self.first is None:
            self.first = BlockingStartStream(callback, block_frames=4)
            return self.first
        self.second = ActiveOutputStream(callback, block_frames=4)
        return self.second


class StopDuringAcquisitionSink(SoundDeviceAudioSink):
    """Requests Stop at the old dequeue/activation race boundary."""

    def _take_next_bar(self, **kwargs):
        bar = super()._take_next_bar(**kwargs)
        self.request_stop_after_bar()
        return bar


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


def test_sounddevice_sink_passes_selected_device_to_the_lazy_portaudio_factory(monkeypatch) -> None:
    observed: dict[str, object] = {}

    class LazySoundDevice:
        class OutputStream(FakeOutputStream):
            def __init__(self, **kwargs) -> None:
                observed.update(kwargs)
                super().__init__(kwargs["callback"], block_frames=4)

    monkeypatch.setitem(sys.modules, "sounddevice", LazySoundDevice)
    sink = SoundDeviceAudioSink(audio_format=stereo_format(), device="Studio Speakers")

    sink.start()
    sink.close()

    assert observed["device"] == "Studio Speakers"


def test_sounddevice_callback_fills_silence_and_reports_status_underrun() -> None:
    stream_factory = FakeOutputStreamFactory(block_frames=4)
    sink = SoundDeviceAudioSink(audio_format=stereo_format(), stream_factory=stream_factory)

    sink.start()
    rendered = stream_factory.render_frames(4, status="output underflow")

    assert rendered == [0.0] * 8
    assert sink.snapshot().absolute_frame == 4
    assert [notice.kind for notice in sink.drain_notices()] == [
        AudioPlaybackNoticeKind.UNDERRUN,
        AudioPlaybackNoticeKind.UNDERRUN,
    ]


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
    assert snapshot.last_completed_bar == 0
    assert snapshot.absolute_frame == 16
    assert snapshot.queue_depth == 1

    sink.reset()
    assert sink.snapshot().last_completed_bar is None


def test_sounddevice_callback_uses_termination_signal_without_stream_stop() -> None:
    stream_factory = FakeOutputStreamFactory(block_frames=8)
    sink = SoundDeviceAudioSink(
        audio_format=stereo_format(),
        stream_factory=stream_factory,
        callback_stop_factory=CallbackTerminated,
    )
    sink.enqueue(prepared_bar(bar=0, frames=8))
    sink.start()
    stream_factory.render_frames(1)
    sink.request_stop_after_bar()

    stream_factory.render_frames(7)

    assert stream_factory.stream is not None
    assert stream_factory.stream.callback_terminations == 1
    assert stream_factory.stream.stop_calls == 0


def test_sounddevice_queue_underrun_is_reported_once_per_starvation_episode() -> None:
    stream_factory = FakeOutputStreamFactory(block_frames=4)
    sink = SoundDeviceAudioSink(audio_format=stereo_format(), stream_factory=stream_factory)
    sink.start()

    stream_factory.render_frames(8)
    first_episode = sink.drain_notices()

    sink.enqueue(prepared_bar(bar=0, frames=4))
    stream_factory.render_frames(4)
    sink.drain_notices()
    stream_factory.render_frames(4)
    second_episode = sink.drain_notices()

    assert [notice.kind for notice in first_episode] == [AudioPlaybackNoticeKind.UNDERRUN]
    assert [notice.kind for notice in second_episode] == [AudioPlaybackNoticeKind.UNDERRUN]
    assert sink.snapshot().underrun_count == 2


def test_stop_during_bar_acquisition_treats_that_bar_as_current() -> None:
    stream_factory = FakeOutputStreamFactory(block_frames=4)
    sink = StopDuringAcquisitionSink(audio_format=stereo_format(), stream_factory=stream_factory)
    sink.enqueue(prepared_bar(bar=0, frames=4, value=0.25))
    sink.start()

    rendered = stream_factory.render_frames(4)

    assert rendered == pytest.approx([0.25] * 8)
    assert [notice.kind for notice in sink.drain_notices()] == [
        AudioPlaybackNoticeKind.BAR_STARTED,
        AudioPlaybackNoticeKind.BAR_COMPLETED,
        AudioPlaybackNoticeKind.STOPPED,
    ]


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
    assert composite.snapshot().last_completed_bar == 0


def test_wav_recorder_reset_replaces_the_previous_audio_epoch(tmp_path: Path) -> None:
    audio_format = stereo_format()
    recorder = Float32WavAudioSink(tmp_path / "session.wav", audio_format)
    first_epoch = prepared_bar(bar=0, frames=4, value=0.25, audio_format=audio_format)
    current_epoch = prepared_bar(bar=0, frames=8, value=0.5, audio_format=audio_format)

    recorder.enqueue(first_epoch)
    recorder.mark_completed(first_epoch)
    recorder.reset()
    recorder.enqueue(current_epoch)
    recorder.mark_completed(current_epoch)
    recorder.close()

    header, payload = read_float_wav(tmp_path / "session.wav")
    assert header[-1] == len(current_epoch.audio.data)
    assert payload == current_epoch.audio.data


def test_composite_close_retains_primary_notices_after_committing_wav(tmp_path: Path) -> None:
    stream_factory = FakeOutputStreamFactory(block_frames=4)
    live = SoundDeviceAudioSink(
        audio_format=stereo_format(),
        stream_factory=stream_factory,
        callback_stop_factory=CallbackTerminated,
    )
    recorder = Float32WavAudioSink(tmp_path / "session.wav", stereo_format())
    composite = CompositeAudioSink(live, recorder)
    composite.enqueue(prepared_bar(bar=0, frames=4))
    composite.start()
    stream_factory.render_frames(1)
    composite.request_stop_after_bar()
    stream_factory.render_frames(3)
    composite.close()

    assert [notice.kind for notice in composite.drain_notices()] == [
        AudioPlaybackNoticeKind.BAR_STARTED,
        AudioPlaybackNoticeKind.BAR_COMPLETED,
        AudioPlaybackNoticeKind.STOPPED,
    ]


def test_null_sink_records_and_completes_bars_without_opening_a_device() -> None:
    sink = NullAudioSink(audio_format=stereo_format())
    bar = prepared_bar(bar=0, frames=6)

    sink.start()
    sink.enqueue(bar)
    sink.complete_next()

    assert sink.recorded_bars == (bar,)
    assert sink.snapshot().absolute_frame == 6
    assert sink.snapshot().last_completed_bar == 0
    assert [notice.kind for notice in sink.drain_notices()] == [
        AudioPlaybackNoticeKind.BAR_STARTED,
        AudioPlaybackNoticeKind.BAR_COMPLETED,
    ]

    sink.reset()
    assert sink.snapshot().last_completed_bar is None


def test_null_sink_stop_before_completion_does_not_start_a_queued_bar() -> None:
    sink = NullAudioSink(audio_format=stereo_format())
    sink.enqueue(prepared_bar(bar=0, frames=6))
    sink.start()

    sink.request_stop_after_bar()

    assert sink.complete_next() is None
    assert sink.snapshot().state == PlaybackState.STOPPED
    assert sink.snapshot().queue_depth == 1


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
        assert snapshot.last_completed_bar == 0
        assert snapshot.absolute_frame == 10
        assert snapshot.queue_depth == 1

        sink.reset()
        assert sink.snapshot().state == PlaybackState.STOPPED
        assert sink.snapshot().last_completed_bar is None
        assert sink.snapshot().absolute_frame == 0
        assert sink.snapshot().queue_depth == 0
    finally:
        sink.close()


def test_timed_sink_stop_before_clock_advances_does_not_start_a_queued_bar() -> None:
    audio_format = stereo_format(sample_rate_hz=100)
    sink = TimedAudioSink(
        audio_format=audio_format,
        poll_interval_seconds=1.0,
        clock=lambda: 0.0,
    )
    sink.enqueue(prepared_bar(bar=0, frames=10, audio_format=audio_format))

    try:
        sink.start()
        sink.request_stop_after_bar()
        wait_until(lambda: sink.snapshot().state == PlaybackState.STOPPED)

        assert sink.snapshot().absolute_frame == 0
        assert sink.snapshot().queue_depth == 1
    finally:
        sink.close()


def test_sounddevice_factory_failure_publishes_device_failed_notice() -> None:
    failure = RuntimeError("no output device")

    def failing_factory(*, audio_format: AudioFormat, callback):
        raise failure

    sink = SoundDeviceAudioSink(audio_format=stereo_format(), stream_factory=failing_factory)

    sink.start()

    assert sink.snapshot().state == PlaybackState.STOPPED
    notices = sink.drain_notices()
    assert [notice.kind for notice in notices] == [AudioPlaybackNoticeKind.DEVICE_FAILED]
    assert notices[0].message == str(failure)


def test_sounddevice_factory_failure_after_reset_publishes_device_failed_notice() -> None:
    failure = RuntimeError("no output device")

    def failing_factory(*, audio_format: AudioFormat, callback):
        raise failure

    sink = SoundDeviceAudioSink(audio_format=stereo_format(), stream_factory=failing_factory)
    sink.reset()

    sink.start()

    assert sink.snapshot().state == PlaybackState.STOPPED
    notices = sink.drain_notices()
    assert [notice.kind for notice in notices] == [AudioPlaybackNoticeKind.DEVICE_FAILED]
    assert notices[0].message == str(failure)


def test_sounddevice_stream_start_failure_publishes_device_failed_notice() -> None:
    failure = RuntimeError("stream start failed")

    class FailingStartStream(FakeOutputStream):
        def start(self) -> None:
            raise failure

    def failing_factory(*, audio_format: AudioFormat, callback) -> FailingStartStream:
        return FailingStartStream(callback, block_frames=4)

    sink = SoundDeviceAudioSink(audio_format=stereo_format(), stream_factory=failing_factory)

    sink.start()

    assert sink.snapshot().state == PlaybackState.STOPPED
    notices = sink.drain_notices()
    assert [notice.kind for notice in notices] == [AudioPlaybackNoticeKind.DEVICE_FAILED]
    assert notices[0].message == str(failure)


def test_closed_sounddevice_sink_remains_terminal_after_start_and_enqueue_attempts() -> None:
    sink = SoundDeviceAudioSink(audio_format=stereo_format(), stream_factory=FakeOutputStreamFactory(block_frames=4))
    sink.close()

    with pytest.raises(RuntimeError, match="closed"):
        sink.start()
    with pytest.raises(RuntimeError, match="closed"):
        sink.enqueue(prepared_bar(bar=0, frames=4))

    assert sink.snapshot().state == PlaybackState.CLOSED
    assert sink.drain_notices() == ()


def test_reset_invalidates_callback_state_commit_and_notices_after_pcm_copy() -> None:
    stream_factory = FakeOutputStreamFactory(block_frames=4)
    sink = SoundDeviceAudioSink(audio_format=stereo_format(), stream_factory=stream_factory)
    sink.enqueue(prepared_bar(bar=0, frames=4, value=0.25))
    sink.start()
    output = BlockingOutputBuffer(frames=4, channels=2)
    callback_thread = Thread(target=sink._callback, args=(output, 4, None, 0))

    callback_thread.start()
    assert output.copy_started.wait(timeout=1.0)
    sink.reset()
    output.allow_commit.set()
    callback_thread.join(timeout=1.0)

    assert not callback_thread.is_alive()
    assert sink.snapshot().state == PlaybackState.STOPPED
    assert sink.snapshot().absolute_frame == 0
    assert sink.snapshot().frame_in_bar == 0
    assert sink.snapshot().queue_depth == 0
    assert sink.drain_notices() == ()


def test_close_during_blocked_stream_start_cancels_physical_stream() -> None:
    stream_factory = BlockingStartStreamFactory()
    sink = SoundDeviceAudioSink(audio_format=stereo_format(), stream_factory=stream_factory)
    start_thread = Thread(target=sink.start)

    start_thread.start()
    assert stream_factory.stream is not None
    assert stream_factory.stream.start_entered.wait(timeout=1.0)
    sink.close()
    stream_factory.stream.allow_start.set()
    start_thread.join(timeout=1.0)

    assert not start_thread.is_alive()
    assert sink.snapshot().state == PlaybackState.CLOSED
    assert stream_factory.stream.closed
    assert not stream_factory.stream.active


def test_reset_during_blocked_stream_start_cancels_physical_stream() -> None:
    stream_factory = BlockingStartStreamFactory()
    sink = SoundDeviceAudioSink(audio_format=stereo_format(), stream_factory=stream_factory)
    start_thread = Thread(target=sink.start)

    start_thread.start()
    assert stream_factory.stream is not None
    assert stream_factory.stream.start_entered.wait(timeout=1.0)
    sink.reset()
    stream_factory.stream.allow_start.set()
    start_thread.join(timeout=1.0)

    assert not start_thread.is_alive()
    assert sink.snapshot().state == PlaybackState.STOPPED
    assert stream_factory.stream.closed
    assert not stream_factory.stream.active
    assert sink.drain_notices() == ()


def test_reset_then_start_b_keeps_b_active_after_stale_start_a_returns() -> None:
    stream_factory = StartABARaceFactory()
    sink = SoundDeviceAudioSink(audio_format=stereo_format(), stream_factory=stream_factory)
    start_a = Thread(target=sink.start)

    start_a.start()
    assert stream_factory.first is not None
    assert stream_factory.first.start_entered.wait(timeout=1.0)
    try:
        sink.reset()
        sink.start()
        assert stream_factory.second is not None
        assert stream_factory.second.active
        assert sink.snapshot().state == PlaybackState.RUNNING

        stream_factory.first.allow_start.set()
        start_a.join(timeout=1.0)

        assert not start_a.is_alive()
        assert stream_factory.second.active
        assert not stream_factory.second.closed
        assert sink.snapshot().state == PlaybackState.RUNNING
    finally:
        stream_factory.first.allow_start.set()
        start_a.join(timeout=1.0)


def test_running_reset_closes_detached_stream_before_starting_new_stream() -> None:
    stream_factory = ActiveOutputStreamFactory()
    sink = SoundDeviceAudioSink(audio_format=stereo_format(), stream_factory=stream_factory)
    sink.start()
    first = stream_factory.streams[0]

    sink.reset()

    assert first.stop_calls == 1
    assert first.closed
    assert not first.active

    sink.start()

    assert len(stream_factory.streams) == 2
    assert stream_factory.streams[1].active
    assert sink.snapshot().state == PlaybackState.RUNNING
