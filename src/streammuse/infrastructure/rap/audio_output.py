"""Device, recording, and test adapters for realtime rap audio."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from queue import Empty, SimpleQueue
import struct
from threading import Event, Lock, RLock, Thread, current_thread
import time
from typing import Callable, Protocol

import numpy as np

from streammuse.domain.rap import (
    AudioFormat,
    AudioPlaybackNotice,
    AudioPlaybackNoticeKind,
    AudioPlaybackSnapshot,
    PlaybackState,
    PreparedRapBar,
)


class OutputStream(Protocol):
    def start(self) -> None: ...

    def stop(self) -> None: ...

    def close(self) -> None: ...


OutputStreamFactory = Callable[..., OutputStream]
CallbackStopFactory = Callable[[], BaseException]


def _require_float32(audio_format: AudioFormat) -> None:
    if audio_format.sample_width_bytes != 4:
        raise ValueError("audio output requires 32-bit float PCM")


def _default_output_stream_factory(*, audio_format: AudioFormat, callback: Callable[..., None]) -> OutputStream:
    # Keep PortAudio and its device discovery out of text-only imports.
    import sounddevice

    return sounddevice.OutputStream(
        samplerate=audio_format.sample_rate_hz,
        channels=audio_format.channels,
        dtype="float32",
        callback=callback,
    )


def _default_callback_stop_factory() -> BaseException:
    # sounddevice is already loaded by the default stream factory in audio mode.
    import sounddevice

    return sounddevice.CallbackStop()


class SoundDeviceAudioSink:
    """PortAudio-backed sink that moves immutable bars in its output callback."""

    def __init__(
        self,
        *,
        audio_format: AudioFormat,
        stream_factory: OutputStreamFactory | None = None,
        callback_stop_factory: CallbackStopFactory | None = None,
    ) -> None:
        _require_float32(audio_format)
        self._audio_format = audio_format
        self._stream_factory = stream_factory or _default_output_stream_factory
        self._callback_stop_factory = callback_stop_factory or _default_callback_stop_factory
        self._queued: deque[PreparedRapBar] = deque()
        self._queue_lock = Lock()
        self._state_lock = RLock()
        self._notices: SimpleQueue[AudioPlaybackNotice] = SimpleQueue()
        self._stream: OutputStream | None = None
        self._state = PlaybackState.STOPPED
        self._active: PreparedRapBar | None = None
        self._frame_in_bar = 0
        self._absolute_frame = 0
        self._underrun_count = 0
        self._queue_underrun_active = False
        self._stop_requested = False
        self._epoch = 0

    def start(self) -> None:
        with self._state_lock:
            if self._state == PlaybackState.CLOSED:
                raise RuntimeError("audio sink is closed")
            if self._state in (PlaybackState.RUNNING, PlaybackState.STOP_REQUESTED):
                return
        stream: OutputStream | None = None
        try:
            with self._state_lock:
                if self._state == PlaybackState.CLOSED:
                    raise RuntimeError("audio sink is closed")
                if self._state in (PlaybackState.RUNNING, PlaybackState.STOP_REQUESTED):
                    return
                if self._stream is None:
                    self._stream = self._stream_factory(audio_format=self._audio_format, callback=self._callback)
                self._state = PlaybackState.RUNNING
                self._stop_requested = False
                stream = self._stream
            stream.start()
        except Exception as error:
            if stream is not None:
                try:
                    stream.close()
                except Exception:
                    pass
            with self._state_lock:
                if self._state == PlaybackState.CLOSED:
                    raise RuntimeError("audio sink is closed") from error
                self._stream = None
                self._active = None
                self._frame_in_bar = 0
                self._state = PlaybackState.STOPPED
                self._stop_requested = False
            self._publish(AudioPlaybackNoticeKind.DEVICE_FAILED, None, str(error))

    def enqueue(self, bar: PreparedRapBar) -> None:
        self._validate_bar(bar)
        with self._state_lock:
            if self._state == PlaybackState.CLOSED:
                raise RuntimeError("audio sink is closed")
        with self._queue_lock:
            self._queued.append(bar)

    def request_stop_after_bar(self) -> None:
        with self._state_lock:
            if self._state not in (PlaybackState.RUNNING, PlaybackState.STOP_REQUESTED):
                return
            if self._active is None:
                self._finish_stop_locked()
                stream = self._stream
            else:
                self._stop_requested = True
                self._state = PlaybackState.STOP_REQUESTED
                stream = None
        if stream is not None:
            stream.stop()

    def reset(self) -> None:
        with self._state_lock:
            stream = self._stream
            self._epoch += 1
            self._active = None
            self._frame_in_bar = 0
            self._absolute_frame = 0
            self._underrun_count = 0
            self._queue_underrun_active = False
            self._stop_requested = False
            if self._state != PlaybackState.CLOSED:
                self._state = PlaybackState.STOPPED
            self._notices = SimpleQueue()
        with self._queue_lock:
            self._queued.clear()
        if stream is not None:
            stream.stop()

    def snapshot(self) -> AudioPlaybackSnapshot:
        with self._state_lock:
            state = self._state
            current_bar = self._active.bar if self._active is not None else None
            frame_in_bar = self._frame_in_bar
            absolute_frame = self._absolute_frame
            underrun_count = self._underrun_count
        with self._queue_lock:
            queue_depth = len(self._queued)
        return AudioPlaybackSnapshot(
            state=state,
            current_bar=current_bar,
            frame_in_bar=frame_in_bar,
            absolute_frame=absolute_frame,
            queue_depth=queue_depth,
            underrun_count=underrun_count,
        )

    def drain_notices(self) -> tuple[AudioPlaybackNotice, ...]:
        notices: list[AudioPlaybackNotice] = []
        while True:
            try:
                notices.append(self._notices.get_nowait())
            except Empty:
                return tuple(notices)

    def close(self) -> None:
        with self._state_lock:
            if self._state == PlaybackState.CLOSED:
                return
            stream = self._stream
            self._epoch += 1
            self._state = PlaybackState.CLOSED
            self._active = None
            self._stop_requested = False
        if stream is not None:
            stream.stop()
            stream.close()

    def _callback(self, outdata: np.ndarray, frames: int, _time_info, status) -> None:
        outdata.fill(0.0)
        with self._state_lock:
            state = self._state
            callback_epoch = self._epoch
            if status and state in (PlaybackState.RUNNING, PlaybackState.STOP_REQUESTED):
                self._underrun_count += 1
                publish_status = True
            else:
                publish_status = False
        if publish_status:
            self._publish(AudioPlaybackNoticeKind.UNDERRUN, None, str(status), epoch=callback_epoch)

        if state not in (PlaybackState.RUNNING, PlaybackState.STOP_REQUESTED):
            return

        copied = 0
        while copied < frames:
            active = self._active
            if active is None:
                active = self._take_next_bar(epoch=callback_epoch)
                if active is None:
                    with self._state_lock:
                        if callback_epoch != self._epoch:
                            return
                        if self._state not in (PlaybackState.RUNNING, PlaybackState.STOP_REQUESTED):
                            return
                        if self._stop_requested:
                            self._finish_stop_locked()
                            stop_callback = True
                        else:
                            self._absolute_frame += frames - copied
                            stop_callback = False
                            if not self._queue_underrun_active:
                                self._queue_underrun_active = True
                                self._underrun_count += 1
                                self._publish(
                                    AudioPlaybackNoticeKind.UNDERRUN,
                                    None,
                                    "prepared bar queue empty",
                                    epoch=callback_epoch,
                                )
                    if stop_callback:
                        raise self._callback_stop_factory()
                    return

            available = active.audio.frame_count - self._frame_in_bar
            to_copy = min(frames - copied, available)
            samples = np.frombuffer(active.audio.data, dtype=np.float32).reshape(
                active.audio.frame_count, self._audio_format.channels
            )
            outdata[copied : copied + to_copy, :] = samples[self._frame_in_bar : self._frame_in_bar + to_copy, :]
            copied += to_copy
            with self._state_lock:
                if callback_epoch != self._epoch:
                    return
                self._frame_in_bar += to_copy
                self._absolute_frame += to_copy
                completed = self._frame_in_bar == active.audio.frame_count

            if not completed:
                continue

            self._publish(
                AudioPlaybackNoticeKind.BAR_COMPLETED,
                active.bar,
                "bar playback completed",
                epoch=callback_epoch,
            )
            with self._state_lock:
                if callback_epoch != self._epoch:
                    return
                self._active = None
                self._frame_in_bar = 0
                stop_requested = self._stop_requested
            if stop_requested:
                with self._state_lock:
                    if callback_epoch != self._epoch:
                        return
                    self._finish_stop_locked()
                raise self._callback_stop_factory()

    def _take_next_bar(self, *, epoch: int | None = None) -> PreparedRapBar | None:
        # A stop request must either observe the active bar or the untouched queue.
        with self._state_lock:
            expected_epoch = self._epoch if epoch is None else epoch
            if (
                expected_epoch != self._epoch
                or self._state not in (PlaybackState.RUNNING, PlaybackState.STOP_REQUESTED)
                or self._stop_requested
            ):
                return None
            with self._queue_lock:
                if not self._queued:
                    return None
                active = self._queued.popleft()
            self._active = active
            self._frame_in_bar = 0
            self._queue_underrun_active = False
            self._publish(AudioPlaybackNoticeKind.BAR_STARTED, active.bar, "bar playback started", epoch=expected_epoch)
            return active

    def _finish_stop_locked(self) -> None:
        if self._state == PlaybackState.STOPPED:
            return
        self._state = PlaybackState.STOPPED
        self._stop_requested = False
        self._publish(AudioPlaybackNoticeKind.STOPPED, None, "playback stopped")

    def _publish(
        self,
        kind: AudioPlaybackNoticeKind,
        bar: int | None,
        message: str,
        *,
        epoch: int | None = None,
    ) -> None:
        with self._state_lock:
            if epoch is not None and epoch != self._epoch:
                return
            absolute_frame = self._absolute_frame
            with self._queue_lock:
                queue_depth = len(self._queued)
            self._notices.put(
                AudioPlaybackNotice(
                    kind=kind,
                    bar=bar,
                    absolute_frame=absolute_frame,
                    queue_depth=queue_depth,
                    message=message,
                )
            )

    def _validate_bar(self, bar: PreparedRapBar) -> None:
        if bar.audio.format != self._audio_format:
            raise ValueError("prepared bar audio format does not match sink format")


class Float32WavAudioSink:
    """Streaming IEEE-float WAV recorder that writes only completed bars."""

    def __init__(self, path: Path, audio_format: AudioFormat) -> None:
        _require_float32(audio_format)
        self._path = Path(path)
        self._audio_format = audio_format
        self._queued: deque[PreparedRapBar] = deque()
        self._data_bytes = 0
        self._closed = False
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self._path.open("w+b")
        self._write_header()

    def start(self) -> None:
        self._ensure_open()

    def enqueue(self, bar: PreparedRapBar) -> None:
        self._ensure_open()
        self._validate_bar(bar)
        self._queued.append(bar)

    def request_stop_after_bar(self) -> None:
        self._ensure_open()

    def mark_completed(self, bar: PreparedRapBar) -> None:
        self._ensure_open()
        if not self._queued:
            raise ValueError("cannot complete a bar that was not enqueued")
        expected = self._queued.popleft()
        if expected != bar:
            raise ValueError("completed bar does not match the queued recorder bar")
        self._file.seek(44 + self._data_bytes)
        self._file.write(bar.audio.data)
        self._data_bytes += len(bar.audio.data)

    def reset(self) -> None:
        self._ensure_open()
        self._queued.clear()
        self._data_bytes = 0
        self._file.seek(0)
        self._file.truncate(0)
        self._write_header()

    def close(self) -> None:
        if self._closed:
            return
        self._file.seek(44 + self._data_bytes)
        self._file.truncate()
        self._write_header()
        self._file.flush()
        self._file.close()
        self._closed = True

    def _write_header(self) -> None:
        bytes_per_frame = self._audio_format.channels * self._audio_format.sample_width_bytes
        byte_rate = self._audio_format.sample_rate_hz * bytes_per_frame
        header = struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF",
            36 + self._data_bytes,
            b"WAVE",
            b"fmt ",
            16,
            3,
            self._audio_format.channels,
            self._audio_format.sample_rate_hz,
            byte_rate,
            bytes_per_frame,
            32,
            b"data",
            self._data_bytes,
        )
        self._file.seek(0)
        self._file.write(header)

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("WAV sink is closed")

    def _validate_bar(self, bar: PreparedRapBar) -> None:
        if bar.audio.format != self._audio_format:
            raise ValueError("prepared bar audio format does not match recorder format")


class CompositeAudioSink:
    """Primary sink plus a recorder committed by primary completion notices."""

    def __init__(self, primary: SoundDeviceAudioSink | "TimedAudioSink" | "NullAudioSink", recorder: Float32WavAudioSink) -> None:
        self._primary = primary
        self._recorder = recorder
        self._bars_by_number: dict[int, PreparedRapBar] = {}
        self._retained_notices: deque[AudioPlaybackNotice] = deque()

    def start(self) -> None:
        self._primary.start()
        self._recorder.start()

    def enqueue(self, bar: PreparedRapBar) -> None:
        self._bars_by_number[bar.bar] = bar
        self._primary.enqueue(bar)
        self._recorder.enqueue(bar)

    def request_stop_after_bar(self) -> None:
        self._primary.request_stop_after_bar()
        self._recorder.request_stop_after_bar()

    def reset(self) -> None:
        self._retain_primary_notices()
        self._primary.reset()
        self._recorder.reset()
        self._bars_by_number.clear()
        self._retained_notices.clear()

    def snapshot(self) -> AudioPlaybackSnapshot:
        return self._primary.snapshot()

    def drain_notices(self) -> tuple[AudioPlaybackNotice, ...]:
        self._retain_primary_notices()
        notices = tuple(self._retained_notices)
        self._retained_notices.clear()
        return notices

    def close(self) -> None:
        self._retain_primary_notices()
        self._primary.close()
        self._retain_primary_notices()
        self._recorder.close()

    def _retain_primary_notices(self) -> None:
        notices = self._primary.drain_notices()
        self._commit_completed_bars(notices)
        self._retained_notices.extend(notices)

    def _commit_completed_bars(self, notices: tuple[AudioPlaybackNotice, ...]) -> None:
        for notice in notices:
            if notice.kind != AudioPlaybackNoticeKind.BAR_COMPLETED or notice.bar is None:
                continue
            bar = self._bars_by_number.pop(notice.bar, None)
            if bar is not None:
                self._recorder.mark_completed(bar)


class NullAudioSink:
    """Deterministic manual sink for tests that do not need a clock or device."""

    def __init__(self, *, audio_format: AudioFormat) -> None:
        _require_float32(audio_format)
        self._audio_format = audio_format
        self._queued: deque[PreparedRapBar] = deque()
        self._recorded: list[PreparedRapBar] = []
        self._notices: SimpleQueue[AudioPlaybackNotice] = SimpleQueue()
        self._state = PlaybackState.STOPPED
        self._absolute_frame = 0
        self._stop_requested = False

    @property
    def recorded_bars(self) -> tuple[PreparedRapBar, ...]:
        return tuple(self._recorded)

    def start(self) -> None:
        if self._state == PlaybackState.CLOSED:
            raise RuntimeError("audio sink is closed")
        self._state = PlaybackState.RUNNING
        self._stop_requested = False

    def enqueue(self, bar: PreparedRapBar) -> None:
        self._validate_bar(bar)
        if self._state == PlaybackState.CLOSED:
            raise RuntimeError("audio sink is closed")
        self._queued.append(bar)

    def request_stop_after_bar(self) -> None:
        if self._state == PlaybackState.RUNNING:
            self._state = PlaybackState.STOPPED
            self._stop_requested = False
            self._publish(AudioPlaybackNoticeKind.STOPPED, None, "playback stopped")

    def complete_next(self) -> PreparedRapBar | None:
        if self._state not in (PlaybackState.RUNNING, PlaybackState.STOP_REQUESTED) or not self._queued:
            return None
        bar = self._queued.popleft()
        self._publish(AudioPlaybackNoticeKind.BAR_STARTED, bar.bar, "bar playback started")
        self._absolute_frame += bar.audio.frame_count
        self._recorded.append(bar)
        self._publish(AudioPlaybackNoticeKind.BAR_COMPLETED, bar.bar, "bar playback completed")
        if self._stop_requested:
            self._state = PlaybackState.STOPPED
            self._stop_requested = False
            self._publish(AudioPlaybackNoticeKind.STOPPED, None, "playback stopped")
        return bar

    def reset(self) -> None:
        self._queued.clear()
        self._recorded.clear()
        self._absolute_frame = 0
        self._stop_requested = False
        if self._state != PlaybackState.CLOSED:
            self._state = PlaybackState.STOPPED
        self._notices = SimpleQueue()

    def snapshot(self) -> AudioPlaybackSnapshot:
        return AudioPlaybackSnapshot(
            state=self._state,
            current_bar=None,
            frame_in_bar=0,
            absolute_frame=self._absolute_frame,
            queue_depth=len(self._queued),
            underrun_count=0,
        )

    def drain_notices(self) -> tuple[AudioPlaybackNotice, ...]:
        notices: list[AudioPlaybackNotice] = []
        while True:
            try:
                notices.append(self._notices.get_nowait())
            except Empty:
                return tuple(notices)

    def close(self) -> None:
        self._state = PlaybackState.CLOSED
        self._queued.clear()

    def _publish(self, kind: AudioPlaybackNoticeKind, bar: int | None, message: str) -> None:
        self._notices.put(
            AudioPlaybackNotice(
                kind=kind,
                bar=bar,
                absolute_frame=self._absolute_frame,
                queue_depth=len(self._queued),
                message=message,
            )
        )

    def _validate_bar(self, bar: PreparedRapBar) -> None:
        if bar.audio.format != self._audio_format:
            raise ValueError("prepared bar audio format does not match sink format")


class TimedAudioSink:
    """Device-free realtime primary sink for standalone WAV output."""

    def __init__(
        self,
        *,
        audio_format: AudioFormat,
        poll_interval_seconds: float = 0.01,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        _require_float32(audio_format)
        if poll_interval_seconds <= 0:
            raise ValueError("poll interval must be positive")
        self._audio_format = audio_format
        self._poll_interval_seconds = poll_interval_seconds
        self._clock = clock
        self._lock = Lock()
        self._wake = Event()
        self._notices: SimpleQueue[AudioPlaybackNotice] = SimpleQueue()
        self._queued: deque[PreparedRapBar] = deque()
        self._state = PlaybackState.STOPPED
        self._active: PreparedRapBar | None = None
        self._frame_in_bar = 0
        self._absolute_frame = 0
        self._fractional_frames = 0.0
        self._last_clock = 0.0
        self._stop_requested = False
        self._thread: Thread | None = None

    def start(self) -> None:
        with self._lock:
            if self._state == PlaybackState.CLOSED:
                raise RuntimeError("audio sink is closed")
            if self._state in (PlaybackState.RUNNING, PlaybackState.STOP_REQUESTED):
                return
            self._state = PlaybackState.RUNNING
            self._stop_requested = False
            self._last_clock = self._clock()
            if self._thread is None:
                self._thread = Thread(target=self._run, name="rap-wav-clock", daemon=True)
                self._thread.start()
        self._wake.set()

    def enqueue(self, bar: PreparedRapBar) -> None:
        self._validate_bar(bar)
        with self._lock:
            if self._state == PlaybackState.CLOSED:
                raise RuntimeError("audio sink is closed")
            self._queued.append(bar)
        self._wake.set()

    def request_stop_after_bar(self) -> None:
        with self._lock:
            if self._state not in (PlaybackState.RUNNING, PlaybackState.STOP_REQUESTED):
                return
            if self._active is None:
                self._finish_stop_locked()
            else:
                self._stop_requested = True
                self._state = PlaybackState.STOP_REQUESTED
        self._wake.set()

    def reset(self) -> None:
        with self._lock:
            self._queued.clear()
            self._active = None
            self._frame_in_bar = 0
            self._absolute_frame = 0
            self._fractional_frames = 0.0
            self._stop_requested = False
            if self._state != PlaybackState.CLOSED:
                self._state = PlaybackState.STOPPED
            self._notices = SimpleQueue()
        self._wake.set()

    def snapshot(self) -> AudioPlaybackSnapshot:
        with self._lock:
            return AudioPlaybackSnapshot(
                state=self._state,
                current_bar=self._active.bar if self._active is not None else None,
                frame_in_bar=self._frame_in_bar,
                absolute_frame=self._absolute_frame,
                queue_depth=len(self._queued),
                underrun_count=0,
            )

    def drain_notices(self) -> tuple[AudioPlaybackNotice, ...]:
        notices: list[AudioPlaybackNotice] = []
        while True:
            try:
                notices.append(self._notices.get_nowait())
            except Empty:
                return tuple(notices)

    def close(self) -> None:
        with self._lock:
            if self._state == PlaybackState.CLOSED:
                return
            self._state = PlaybackState.CLOSED
            thread = self._thread
        self._wake.set()
        if thread is not None and thread is not current_thread():
            thread.join(timeout=1.0)

    def _run(self) -> None:
        while True:
            with self._lock:
                if self._state == PlaybackState.CLOSED:
                    return
                if self._state in (PlaybackState.RUNNING, PlaybackState.STOP_REQUESTED):
                    now = self._clock()
                    elapsed = max(0.0, now - self._last_clock)
                    self._last_clock = now
                    total_frames = self._fractional_frames + elapsed * self._audio_format.sample_rate_hz
                    whole_frames = int(total_frames)
                    self._fractional_frames = total_frames - whole_frames
                    if whole_frames:
                        self._advance_locked(whole_frames)
            self._wake.wait(self._poll_interval_seconds)
            self._wake.clear()

    def _advance_locked(self, frames: int) -> None:
        remaining = frames
        while remaining:
            if self._active is None:
                if self._stop_requested:
                    self._finish_stop_locked()
                    return
                if not self._queued:
                    self._absolute_frame += remaining
                    return
                self._active = self._queued.popleft()
                self._frame_in_bar = 0
                self._publish_locked(AudioPlaybackNoticeKind.BAR_STARTED, self._active.bar, "bar playback started")

            active = self._active
            assert active is not None
            consumed = min(remaining, active.audio.frame_count - self._frame_in_bar)
            self._frame_in_bar += consumed
            self._absolute_frame += consumed
            remaining -= consumed
            if self._frame_in_bar != active.audio.frame_count:
                continue

            self._publish_locked(AudioPlaybackNoticeKind.BAR_COMPLETED, active.bar, "bar playback completed")
            self._active = None
            self._frame_in_bar = 0
            if self._stop_requested:
                self._finish_stop_locked()
                return

    def _finish_stop_locked(self) -> None:
        if self._state == PlaybackState.STOPPED:
            return
        self._state = PlaybackState.STOPPED
        self._stop_requested = False
        self._publish_locked(AudioPlaybackNoticeKind.STOPPED, None, "playback stopped")

    def _publish_locked(self, kind: AudioPlaybackNoticeKind, bar: int | None, message: str) -> None:
        self._notices.put(
            AudioPlaybackNotice(
                kind=kind,
                bar=bar,
                absolute_frame=self._absolute_frame,
                queue_depth=len(self._queued),
                message=message,
            )
        )

    def _validate_bar(self, bar: PreparedRapBar) -> None:
        if bar.audio.format != self._audio_format:
            raise ValueError("prepared bar audio format does not match sink format")
