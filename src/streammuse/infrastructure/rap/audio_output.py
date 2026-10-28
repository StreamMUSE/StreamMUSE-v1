"""Device, recording, and test adapters for realtime rap audio."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from queue import Empty, SimpleQueue
import struct
from threading import Event, Lock, RLock, Thread, current_thread
import time
from typing import Callable, Iterable, Protocol

import numpy as np

from streammuse.domain.rap import (
    AudioFormat,
    AudioPlaybackNotice,
    AudioPlaybackNoticeKind,
    AudioPlaybackSnapshot,
    PcmAudio,
    PlaybackState,
    PreparedRapBar,
)
from streammuse.domain.rap.audio import RapMixConfig
from streammuse.infrastructure.rap.live_mix import (
    CallbackAudio, GAIN_NAMES, GainRamp, MAX_MIX_FRAMES, SMOOTHING_SECONDS,
)


class OutputStream(Protocol):
    def start(self) -> None: ...

    def stop(self) -> None: ...

    def close(self) -> None: ...


OutputStreamFactory = Callable[..., OutputStream]
CallbackStopFactory = Callable[[], BaseException]


@dataclass(frozen=True)
class OutputDeviceDescriptor:
    """Host/name is the selection identity; index is only a current lookup hint."""

    index: int
    name: str
    host_api: str
    max_output_channels: int
    default_sample_rate_hz: float
    is_default: bool
    available: bool
    error: str | None = None

    def to_payload(self) -> dict[str, object]:
        return asdict(self)


OutputDeviceSelection = OutputDeviceDescriptor | Mapping[str, object] | str | int | None


def list_output_devices(audio_format: AudioFormat = AudioFormat()) -> tuple[OutputDeviceDescriptor, ...]:
    """Enumerate output-capable devices, including explicit format failures."""
    _require_float32(audio_format)
    try:
        import sounddevice

        devices = sounddevice.query_devices()
        host_apis = sounddevice.query_hostapis()
        default_output = sounddevice.default.device[1]
    except Exception as error:
        raise RuntimeError(f"audio device enumeration failed: {error}") from error
    result = []
    for index, device in enumerate(devices):
        channels = int(device["max_output_channels"])
        if channels <= 0:
            continue
        failure = None
        try:
            sounddevice.check_output_settings(
                device=index, samplerate=audio_format.sample_rate_hz,
                channels=audio_format.channels, dtype="float32",
            )
        except Exception as error:
            failure = str(error)
        result.append(OutputDeviceDescriptor(
            index=index, name=str(device["name"]),
            host_api=str(host_apis[int(device["hostapi"])]["name"]),
            max_output_channels=channels,
            default_sample_rate_hz=float(device["default_samplerate"]),
            is_default=index == default_output, available=failure is None, error=failure,
        ))
    return tuple(result)


def validate_output_device(
    device: OutputDeviceSelection, audio_format: AudioFormat = AudioFormat(),
) -> OutputDeviceDescriptor:
    """Resolve a fresh host/name descriptor and check the exact output format."""
    devices = list_output_devices(audio_format)
    if isinstance(device, OutputDeviceDescriptor):
        matches = [item for item in devices if (item.host_api, item.name) == (device.host_api, device.name)]
    elif isinstance(device, Mapping):
        name, host_api = device.get("name"), device.get("host_api")
        if not isinstance(name, str) or not name or not isinstance(host_api, str) or not host_api:
            raise ValueError("output device descriptor requires host_api and name")
        matches = [item for item in devices if (item.host_api, item.name) == (host_api, name)]
    elif device is None:
        matches = [item for item in devices if item.is_default]
    elif type(device) is int:
        matches = [item for item in devices if item.index == device]
    elif isinstance(device, str):
        matches = [item for item in devices if item.name == device]
    else:
        raise ValueError("output device must be a host/name descriptor, name, index, or default")
    if not matches:
        raise ValueError("selected output device is missing or unavailable")
    if len(matches) != 1:
        raise ValueError("output device descriptor is ambiguous; choose a unique host/name")
    selected = matches[0]
    if not selected.available:
        raise ValueError(f"output device {selected.name} is unavailable: {selected.error}")
    return selected


def _require_float32(audio_format: AudioFormat) -> None:
    if audio_format.sample_width_bytes != 4:
        raise ValueError("audio output requires 32-bit float PCM")


def _buffered_seconds(
    active: PreparedRapBar | None,
    frame_in_bar: int,
    queued: Iterable[PreparedRapBar],
) -> float:
    seconds = 0.0
    if active is not None:
        seconds += max(0, active.audio.frame_count - frame_in_bar) / active.audio.format.sample_rate_hz
    seconds += sum(bar.audio.duration_seconds for bar in queued)
    return seconds


def _default_output_stream_factory(
    *, audio_format: AudioFormat, callback: Callable[..., None], device: str | int | None = None
) -> OutputStream:
    # Keep PortAudio and its device discovery out of text-only imports.
    import sounddevice

    return sounddevice.OutputStream(
        samplerate=audio_format.sample_rate_hz,
        channels=audio_format.channels,
        dtype="float32",
        callback=callback,
        device=device,
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
        device: str | None = None,
        stream_factory: OutputStreamFactory | None = None,
        callback_stop_factory: CallbackStopFactory | None = None,
    ) -> None:
        _require_float32(audio_format)
        self._audio_format = audio_format
        self._device = device
        self._device_descriptor: OutputDeviceDescriptor | None = None
        self._device_error: str | None = None
        self._stream_factory = stream_factory
        self._callback_stop_factory = callback_stop_factory or _default_callback_stop_factory
        self._queued: deque[PreparedRapBar] = deque()
        self._queue_lock = Lock()
        self._state_lock = RLock()
        self._notices: SimpleQueue[AudioPlaybackNotice] = SimpleQueue()
        self._stream: OutputStream | None = None
        self._stream_started = False
        self._state = PlaybackState.STOPPED
        self._active: PreparedRapBar | None = None
        self._last_completed_bar: int | None = None
        self._frame_in_bar = 0
        self._absolute_frame = 0
        self._underrun_count = 0
        self._queue_underrun_active = False
        self._stop_requested = False
        self._epoch = 0
        self._queued_audio: deque[CallbackAudio] = deque()
        self._active_audio: CallbackAudio | None = None
        self._completed_audio: dict[int, np.ndarray] = {}
        self._live_requested: tuple[float, ...] | None = None
        self._live_revision = 0
        self._clear_live_evidence_locked()

    def set_live_gains(self, *, drum_gain: float, vocal_gain: float, chord_gain: float) -> dict:
        """Accept finite absolute gains in [0, 2], without seeking or changing chords.

        The returned snapshot acknowledges the request, not audible application.
        Poll output_device_status()['live_mix'] for per-stem callback evidence;
        unsupported stems have no applied value, even when a request is retained.
        """
        config = RapMixConfig(drum_gain=drum_gain, vocal_gain=vocal_gain, chord_gain=chord_gain)
        requested = tuple(float(getattr(config, name)) for name in GAIN_NAMES)
        with self._state_lock:
            if self._state == PlaybackState.CLOSED:
                raise RuntimeError("audio sink is closed")
            if requested != self._live_requested:
                self._live_requested = requested
                self._live_revision += 1
            return self._live_status_locked()

    def _clear_live_evidence_locked(self) -> None:
        self._live_ramp: GainRamp | None = None
        self._live_applied: dict[str, float | None] | None = None
        self._live_applied_revision: int | None = None
        self._live_supported_gains: tuple[str, ...] = ()
        self._live_bar: int | None = None
        self._live_frame: int | None = None
        self._live_limitation: str | None = "no callback PCM observed in this playback epoch"

    def _live_status_locked(self) -> dict[str, object]:
        return {
            "requested": dict(zip(GAIN_NAMES, self._live_requested)) if self._live_requested is not None else None,
            "applied": dict(self._live_applied) if self._live_applied is not None else None,
            "revision": self._live_revision,
            "applied_revision": self._live_applied_revision,
            "supported": bool(self._live_supported_gains),
            "supported_gains": list(self._live_supported_gains),
            "smoothing_ms": SMOOTHING_SECONDS * 1000,
            "bar": self._live_bar,
            "absolute_frame": self._live_frame,
            "limitation": self._live_limitation,
            "observation": "callback_pcm_not_acoustic_verification",
        }

    def _observe_live_mix_locked(self, bar: PreparedRapBar, audio: CallbackAudio, ramp: GainRamp | None) -> None:
        self._live_ramp = ramp
        self._live_bar = bar.bar
        self._live_frame = self._absolute_frame
        self._live_limitation = audio.limitation
        self._live_supported_gains = tuple(
            name for name, stem in zip(GAIN_NAMES, audio.stems)
            if stem is not None and not (name == "vocal_gain" and audio.vocal_is_auxiliary)
        )
        self._live_applied = (
            {name: gain if name in self._live_supported_gains else None for name, gain in zip(GAIN_NAMES, ramp.gains)}
            if ramp is not None else None
        )
        self._live_applied_revision = ramp.revision if ramp is not None and not ramp.remaining else None

    @property
    def output_device(self) -> OutputDeviceDescriptor | None:
        with self._state_lock:
            return self._device_descriptor

    def list_output_devices(self) -> tuple[OutputDeviceDescriptor, ...]:
        return list_output_devices(self._audio_format)

    def validate_output_device(self, device: OutputDeviceSelection) -> OutputDeviceDescriptor:
        return validate_output_device(device, self._audio_format)

    def set_output_device(self, device: OutputDeviceSelection) -> OutputDeviceDescriptor:
        """Open a replacement while stopped; preserve queue and playback cursor."""
        with self._state_lock:
            if self._state != PlaybackState.STOPPED:
                raise RuntimeError("output device can only change while stopped")
            try:
                selected = self.validate_output_device(device)
                replacement = self._open_stream(selected.index)
            except Exception as error:
                self._device_error = str(error)
                self._publish(AudioPlaybackNoticeKind.DEVICE_FAILED, None, str(error))
                raise
            old_stream = self._stream
            self._stream = replacement
            self._stream_started = False
            self._device = selected.index
            self._device_descriptor = selected
            self._device_error = None
            self._epoch += 1
        if old_stream is not None:
            self._stop_and_close_stream(old_stream)
        return selected

    def output_device_status(self) -> dict[str, object]:
        with self._state_lock:
            self._observe_stream_failure_locked()
            return {
                "backend": "sounddevice",
                "audible": self._stream_started and self._state in (PlaybackState.RUNNING, PlaybackState.STOP_REQUESTED),
                "supports_device_selection": True,
                "supports_live_gains": True,
                "device": self._device_descriptor.to_payload() if self._device_descriptor else None,
                "error": self._device_error,
                "observation": "device_stream_state_not_acoustic_verification",
                "live_mix": self._live_status_locked(),
            }

    def _observe_stream_failure_locked(self) -> None:
        if not self._stream_started or self._state not in (PlaybackState.RUNNING, PlaybackState.STOP_REQUESTED):
            return
        try:
            inactive = getattr(self._stream, "active", None) is False
            message = "audio output stream stopped unexpectedly; device may be unavailable"
        except Exception as error:
            inactive = True
            message = f"audio output stream status failed: {error}"
        if inactive:
            self._state = PlaybackState.STOPPED
            self._stream_started = False
            self._stop_requested = False
            self._device_error = message
            self._publish(AudioPlaybackNoticeKind.DEVICE_FAILED, self._active.bar if self._active else None, message)

    def _open_stream(self, device: str | int | None) -> OutputStream:
        if self._stream_factory is None:
            return _default_output_stream_factory(audio_format=self._audio_format, callback=self._callback, device=device)
        options = {"audio_format": self._audio_format, "callback": self._callback}
        if device is not None:
            options["device"] = device
        return self._stream_factory(**options)

    def start(self) -> bool:
        with self._state_lock:
            if self._state == PlaybackState.CLOSED:
                raise RuntimeError("audio sink is closed")
            if self._state in (PlaybackState.RUNNING, PlaybackState.STOP_REQUESTED):
                return True
            stream = self._stream
            start_epoch = self._epoch
        try:
            # CallbackStop makes PortAudio inactive, not stopped. Finish the
            # previous host lifecycle outside the callback/state lock first.
            if stream is not None:
                stream.stop()
            with self._state_lock:
                if self._state == PlaybackState.CLOSED:
                    raise RuntimeError("audio sink is closed")
                if start_epoch != self._epoch:
                    return False
                if self._state in (PlaybackState.RUNNING, PlaybackState.STOP_REQUESTED):
                    return True
                start_epoch = self._epoch
                stream = self._stream
                if self._device_descriptor is not None:
                    self._device_descriptor = self.validate_output_device(self._device_descriptor)
                    self._device = self._device_descriptor.index
                if self._stream is None:
                    self._stream = self._open_stream(self._device)
                self._state = PlaybackState.RUNNING
                self._stream_started = False
                self._stop_requested = False
                self._device_error = None
                stream = self._stream
            stream.start()
        except Exception as error:
            if stream is not None:
                self._stop_and_close_stream(stream)
            with self._state_lock:
                if self._state == PlaybackState.CLOSED:
                    raise RuntimeError("audio sink is closed") from error
                if start_epoch != self._epoch:
                    if self._stream is stream:
                        self._stream = None
                    return False
                self._stream = None
                self._stream_started = False
                self._state = PlaybackState.STOPPED
                self._stop_requested = False
                self._device_error = str(error)
            self._publish(AudioPlaybackNoticeKind.DEVICE_FAILED, None, str(error))
            return False

        with self._state_lock:
            cancelled = start_epoch != self._epoch or self._state not in (
                PlaybackState.RUNNING,
                PlaybackState.STOP_REQUESTED,
            )
            if not cancelled:
                self._stream_started = True
        if cancelled:
            self._stop_and_close_stream(stream)
            with self._state_lock:
                if self._stream is stream:
                    self._stream = None
        return not cancelled

    def enqueue(self, bar: PreparedRapBar) -> None:
        self._enqueue(bar, capture=False)

    def _enqueue(self, bar: PreparedRapBar, *, capture: bool) -> None:
        self._validate_bar(bar)
        audio = CallbackAudio.prepare(bar, capture=capture)
        with self._state_lock:
            if self._state == PlaybackState.CLOSED:
                raise RuntimeError("audio sink is closed")
            with self._queue_lock:
                self._queued_audio.append(audio)
                self._queued.append(bar)

    def _take_completed_audio(self, bar: PreparedRapBar) -> PcmAudio:
        with self._state_lock:
            samples = self._completed_audio.pop(id(bar))
        return PcmAudio(bar.audio.format, bar.audio.frame_count, samples.tobytes())

    def request_stop_after_bar(self) -> AudioPlaybackSnapshot:
        with self._state_lock:
            if self._state not in (PlaybackState.RUNNING, PlaybackState.STOP_REQUESTED):
                return self._snapshot_locked()
            if self._active is None:
                self._finish_stop_locked()
                stream = self._stream
            else:
                self._stop_requested = True
                self._state = PlaybackState.STOP_REQUESTED
                stream = None
            snapshot = self._snapshot_locked()
        if stream is not None:
            stream.stop()
        return snapshot

    def reset(self) -> None:
        with self._state_lock:
            stream = self._stream
            self._stream = None
            self._stream_started = False
            self._epoch += 1
            self._active = None
            self._active_audio = None
            with self._queue_lock:
                self._queued.clear()
                self._queued_audio.clear()
            self._completed_audio.clear()
            self._clear_live_evidence_locked()
            self._last_completed_bar = None
            self._frame_in_bar = 0
            self._absolute_frame = 0
            self._underrun_count = 0
            self._queue_underrun_active = False
            self._stop_requested = False
            if self._state != PlaybackState.CLOSED:
                self._state = PlaybackState.STOPPED
            self._notices = SimpleQueue()
        if stream is not None:
            self._stop_and_close_stream(stream)

    def discard_pending(self) -> None:
        """Clear one stopped playback epoch without destroying the device stream."""

        with self._state_lock:
            if self._state == PlaybackState.CLOSED:
                raise RuntimeError("audio sink is closed")
            stream = self._stream
            self._stream_started = False
            self._epoch += 1
            self._active = None
            self._active_audio = None
            with self._queue_lock:
                self._queued.clear()
                self._queued_audio.clear()
            self._completed_audio.clear()
            self._clear_live_evidence_locked()
            self._last_completed_bar = None
            self._frame_in_bar = 0
            self._absolute_frame = 0
            self._underrun_count = 0
            self._queue_underrun_active = False
            self._stop_requested = False
            self._state = PlaybackState.STOPPED
            self._notices = SimpleQueue()
        if stream is not None:
            stream.stop()

    def snapshot(self) -> AudioPlaybackSnapshot:
        with self._state_lock:
            self._observe_stream_failure_locked()
            return self._snapshot_locked()

    def _snapshot_locked(self) -> AudioPlaybackSnapshot:
        with self._queue_lock:
            queued = tuple(self._queued)
        return AudioPlaybackSnapshot(
            state=self._state,
            current_bar=self._active.bar if self._active is not None else None,
            last_completed_bar=self._last_completed_bar,
            frame_in_bar=self._frame_in_bar,
            absolute_frame=self._absolute_frame,
            queue_depth=len(queued),
            underrun_count=self._underrun_count,
            buffered_seconds=_buffered_seconds(self._active, self._frame_in_bar, queued),
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
            self._active_audio = None
            with self._queue_lock:
                self._queued.clear()
                self._queued_audio.clear()
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

            with self._state_lock:
                if callback_epoch != self._epoch:
                    return
                audio = self._active_audio
                start = self._frame_in_bar
                requested, revision, ramp = self._live_requested, self._live_revision, self._live_ramp
            available = active.audio.frame_count - start
            to_copy = min(frames - copied, available, MAX_MIX_FRAMES)
            samples, ramp = audio.mix(active, start, to_copy, requested, revision, ramp)
            outdata[copied : copied + to_copy, :] = samples
            copied += to_copy
            with self._state_lock:
                if callback_epoch != self._epoch:
                    return
                if audio.capture is not None:
                    audio.capture[start:start + to_copy] = samples
                self._frame_in_bar += to_copy
                self._absolute_frame += to_copy
                if to_copy:
                    self._observe_live_mix_locked(active, audio, ramp)
                completed = self._frame_in_bar == active.audio.frame_count
                if completed:
                    self._last_completed_bar = active.bar
                    if audio.capture is not None:
                        self._completed_audio[id(active)] = audio.capture

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
                self._active_audio = None
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
                self._active_audio = self._queued_audio.popleft()
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

    @staticmethod
    def _stop_and_close_stream(stream: OutputStream) -> None:
        try:
            stream.stop()
        except Exception:
            pass
        try:
            stream.close()
        except Exception:
            pass

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
                queued = tuple(self._queued)
            self._notices.put(
                AudioPlaybackNotice(
                    kind=kind,
                    bar=bar,
                    absolute_frame=absolute_frame,
                    queue_depth=len(queued),
                    message=message,
                    buffered_seconds=_buffered_seconds(self._active, self._frame_in_bar, queued),
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
        self._start_frames: dict[int, int] = {}
        self._data_bytes = 0
        self._closed = False
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self._path.open("w+b")
        self._write_header()
        self._file.flush()

    @property
    def frame_count(self) -> int:
        bytes_per_frame = self._audio_format.channels * self._audio_format.sample_width_bytes
        return self._data_bytes // bytes_per_frame

    def start(self) -> None:
        self._ensure_open()

    def enqueue(self, bar: PreparedRapBar) -> None:
        self._ensure_open()
        self._validate_bar(bar)
        self._queued.append(bar)

    def request_stop_after_bar(self) -> None:
        self._ensure_open()
        self._sync_header()

    def mark_started(self, bar: PreparedRapBar, absolute_frame: int) -> None:
        self._ensure_open()
        self._validate_bar(bar)
        if absolute_frame < 0:
            raise ValueError("absolute_frame must not be negative")
        self._start_frames[bar.bar] = absolute_frame

    def mark_completed(self, bar: PreparedRapBar, *, played_audio: PcmAudio | None = None) -> None:
        self._ensure_open()
        audio = bar.audio if played_audio is None else played_audio
        if audio.format != bar.audio.format or audio.frame_count != bar.audio.frame_count:
            raise ValueError("played audio must match the completed bar format and frame count")
        if not self._queued:
            raise ValueError("cannot complete a bar that was not enqueued")
        expected = self._queued.popleft()
        if expected != bar:
            raise ValueError("completed bar does not match the queued recorder bar")
        start_frame = self._start_frames.pop(bar.bar, self.frame_count)
        gap_frames = max(0, start_frame - self.frame_count)
        self._file.seek(44 + self._data_bytes)
        self._write_silence(gap_frames)
        self._file.write(audio.data)
        bytes_per_frame = self._audio_format.channels * self._audio_format.sample_width_bytes
        self._data_bytes += gap_frames * bytes_per_frame + len(audio.data)
        self._sync_header()

    def discard_pending(self) -> None:
        self._ensure_open()
        self._queued.clear()
        self._start_frames.clear()
        self._sync_header()

    def reset(self) -> None:
        self._ensure_open()
        self._queued.clear()
        self._start_frames.clear()
        self._data_bytes = 0
        self._file.seek(0)
        self._file.truncate(0)
        self._write_header()
        self._file.flush()

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

    def _sync_header(self) -> None:
        self._write_header()
        self._file.flush()

    def _write_silence(self, frames: int) -> None:
        remaining = frames * self._audio_format.channels * self._audio_format.sample_width_bytes
        chunk = bytes(1024 * 1024)
        while remaining:
            count = min(remaining, len(chunk))
            self._file.write(chunk[:count])
            remaining -= count

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("WAV sink is closed")

    def _validate_bar(self, bar: PreparedRapBar) -> None:
        if bar.audio.format != self._audio_format:
            raise ValueError("prepared bar audio format does not match recorder format")


class CompositeAudioSink:
    """Primary sink plus a recorder committed by primary completion notices."""

    def __init__(
        self,
        primary: SoundDeviceAudioSink | "TimedAudioSink" | "NullAudioSink",
        recorder: Float32WavAudioSink,
        *,
        fallback: SoundDeviceAudioSink | "TimedAudioSink" | "NullAudioSink" | None = None,
    ) -> None:
        self._primary = primary
        self._device_primary = primary if isinstance(primary, SoundDeviceAudioSink) else None
        self._fallback = fallback
        self._fallback_error: str | None = None
        self._recorder = recorder
        self._bars_by_number: dict[int, PreparedRapBar] = {}
        self._bar_start_frames: dict[int, int] = {}
        self._retained_notices: deque[AudioPlaybackNotice] = deque()
        self._recording_origin_frame = 0
        self._live_lock = RLock()
        self._nondevice_live_requested: dict[str, float] | None = None
        self._nondevice_live_revision = 0

    def output_device_status(self) -> dict[str, object]:
        status = self._primary.output_device_status()
        callback_recording = isinstance(self._primary, SoundDeviceAudioSink)
        status["supports_live_gains"] = callback_recording
        status["recording"] = {
            "source": "completed_callback_pcm" if callback_recording else "prepared_bar_pcm",
            "exact_completed_bars": callback_recording,
            "limitation": (
                "incomplete bars and trailing silence are omitted; callback PCM is not acoustic verification"
                if callback_recording else "no device callback; recording contains prepared bars, not played live gains"
            ),
        }
        if not callback_recording:
            with self._live_lock:
                status["live_mix"] = {
                    "supported": False, "supported_gains": [],
                    "requested": dict(self._nondevice_live_requested) if self._nondevice_live_requested is not None else None,
                    "applied": None, "revision": self._nondevice_live_revision, "applied_revision": None,
                    "smoothing_ms": None, "bar": None, "absolute_frame": None,
                    "limitation": "current backend does not support live gains; only future controller-rendered bars can change",
                    "observation": "unavailable",
                }
        if self._device_primary is not None and self._primary is not self._device_primary:
            device_status = self._device_primary.output_device_status()
            if not callback_recording:
                for key in ("requested", "revision"):
                    status["live_mix"][key] = device_status["live_mix"][key]
            status.update({
                "supports_device_selection": True,
                "device": device_status.get("device"),
                "error": self._fallback_error,
                "fallback_from": "sounddevice",
                "recovery_requires_reset": True,
            })
        return status

    def set_live_gains(self, *, drum_gain: float, vocal_gain: float, chord_gain: float) -> dict:
        """Forward live gains or report explicit non-device/fallback limitations."""
        gains = dict(drum_gain=drum_gain, vocal_gain=vocal_gain, chord_gain=chord_gain)
        if isinstance(self._primary, SoundDeviceAudioSink):
            return self._primary.set_live_gains(**gains)
        if self._device_primary is not None:
            self._device_primary.set_live_gains(**gains)
        else:
            config = RapMixConfig(**gains)
            gains = {name: float(getattr(config, name)) for name in GAIN_NAMES}
            with self._live_lock:
                if self._primary.snapshot().state == PlaybackState.CLOSED:
                    raise RuntimeError("audio sink is closed")
                if gains != self._nondevice_live_requested:
                    self._nondevice_live_requested = gains
                    self._nondevice_live_revision += 1
                return self.output_device_status()["live_mix"]
        return self.output_device_status()["live_mix"]

    def list_output_devices(self) -> tuple[OutputDeviceDescriptor, ...]:
        if self._device_primary is None:
            raise RuntimeError("current audio backend has no output device selection")
        return self._device_primary.list_output_devices()

    def validate_output_device(self, device: OutputDeviceSelection) -> OutputDeviceDescriptor:
        if self._device_primary is None:
            raise RuntimeError("current audio backend has no output device selection")
        return self._device_primary.validate_output_device(device)

    def set_output_device(self, device: OutputDeviceSelection) -> OutputDeviceDescriptor:
        if self._device_primary is None:
            raise RuntimeError("current audio backend has no output device selection")
        if self._primary.snapshot().state != PlaybackState.STOPPED:
            raise RuntimeError("output device can only change while stopped")
        return self._device_primary.set_output_device(device)

    def start(self) -> bool:
        self._primary.start()
        snapshot = self._primary.snapshot()
        notices = self._retain_primary_notices()
        failure = next((n.message for n in notices if n.kind == AudioPlaybackNoticeKind.DEVICE_FAILED), None)
        if (
            snapshot.state not in (PlaybackState.RUNNING, PlaybackState.STOP_REQUESTED)
            and self._fallback is not None and failure is not None
        ):
            failed_primary = self._primary
            requested = failed_primary.output_device_status().get("live_mix", {}).get("requested")
            self._primary = self._fallback
            self._fallback = None
            self._fallback_error = failure
            if failed_primary is self._device_primary:
                failed_primary.reset()
            else:
                failed_primary.close()
            if isinstance(self._primary, SoundDeviceAudioSink) and requested is not None:
                self._primary.set_live_gains(**requested)
            for bar in self._bars_by_number.values():
                if isinstance(self._primary, SoundDeviceAudioSink):
                    self._primary._enqueue(bar, capture=True)
                else:
                    self._primary.enqueue(bar)
            self._primary.start()
        self._recorder.start()
        return self._primary.snapshot().state in (PlaybackState.RUNNING, PlaybackState.STOP_REQUESTED)

    def enqueue(self, bar: PreparedRapBar) -> None:
        self._bars_by_number[bar.bar] = bar
        if isinstance(self._primary, SoundDeviceAudioSink):
            self._primary._enqueue(bar, capture=True)
        else:
            self._primary.enqueue(bar)
        self._recorder.enqueue(bar)

    def request_stop_after_bar(self) -> AudioPlaybackSnapshot:
        snapshot = self._primary.request_stop_after_bar()
        self._recorder.request_stop_after_bar()
        return snapshot

    def reset(self) -> None:
        self._retain_primary_notices()
        self._primary.reset()
        if self._device_primary is not None and self._primary is not self._device_primary:
            self._fallback = self._primary
            self._primary = self._device_primary
            self._primary.reset()
        self._fallback_error = None
        self._recorder.reset()
        self._bars_by_number.clear()
        self._bar_start_frames.clear()
        self._recording_origin_frame = 0
        self._retained_notices.clear()

    def discard_pending(self) -> None:
        self._retain_primary_notices()
        self._primary.discard_pending()
        self._recorder.discard_pending()
        self._bars_by_number.clear()
        self._bar_start_frames.clear()
        self._retained_notices.clear()
        self._recording_origin_frame = self._recorder.frame_count

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
        if self._device_primary is not None and self._primary is not self._device_primary:
            self._device_primary.close()
        if self._fallback is not None:
            self._fallback.close()
        self._retain_primary_notices()
        self._recorder.close()

    def _retain_primary_notices(self) -> tuple[AudioPlaybackNotice, ...]:
        notices = self._primary.drain_notices()
        self._commit_completed_bars(notices)
        self._retained_notices.extend(notices)
        return notices

    def _commit_completed_bars(self, notices: tuple[AudioPlaybackNotice, ...]) -> None:
        for notice in notices:
            if notice.bar is None:
                continue
            if notice.kind == AudioPlaybackNoticeKind.BAR_STARTED:
                bar = self._bars_by_number.get(notice.bar)
                if bar is not None:
                    start_frame = self._recording_origin_frame + notice.absolute_frame
                    self._bar_start_frames[notice.bar] = start_frame
                    self._recorder.mark_started(bar, start_frame)
                continue
            if notice.kind != AudioPlaybackNoticeKind.BAR_COMPLETED:
                continue
            bar = self._bars_by_number.pop(notice.bar, None)
            if bar is not None:
                played_audio = self._primary._take_completed_audio(bar) if isinstance(self._primary, SoundDeviceAudioSink) else None
                self._recorder.mark_completed(bar, played_audio=played_audio)
            self._bar_start_frames.pop(notice.bar, None)


class NullAudioSink:
    """Deterministic manual sink for tests that do not need a clock or device."""

    def output_device_status(self) -> dict[str, object]:
        return {"backend": "null", "audible": False, "supports_device_selection": False}

    def __init__(self, *, audio_format: AudioFormat) -> None:
        _require_float32(audio_format)
        self._audio_format = audio_format
        self._queued: deque[PreparedRapBar] = deque()
        self._recorded: list[PreparedRapBar] = []
        self._notices: SimpleQueue[AudioPlaybackNotice] = SimpleQueue()
        self._state = PlaybackState.STOPPED
        self._absolute_frame = 0
        self._last_completed_bar: int | None = None
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

    def request_stop_after_bar(self) -> AudioPlaybackSnapshot:
        if self._state == PlaybackState.RUNNING:
            self._state = PlaybackState.STOPPED
            self._stop_requested = False
            self._publish(AudioPlaybackNoticeKind.STOPPED, None, "playback stopped")
        return self.snapshot()

    def complete_next(self) -> PreparedRapBar | None:
        if self._state not in (PlaybackState.RUNNING, PlaybackState.STOP_REQUESTED) or not self._queued:
            return None
        bar = self._queued.popleft()
        self._publish(AudioPlaybackNoticeKind.BAR_STARTED, bar.bar, "bar playback started")
        self._absolute_frame += bar.audio.frame_count
        self._recorded.append(bar)
        self._last_completed_bar = bar.bar
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
        self._last_completed_bar = None
        self._stop_requested = False
        if self._state != PlaybackState.CLOSED:
            self._state = PlaybackState.STOPPED
        self._notices = SimpleQueue()

    def discard_pending(self) -> None:
        if self._state == PlaybackState.CLOSED:
            raise RuntimeError("audio sink is closed")
        self._queued.clear()
        self._absolute_frame = 0
        self._last_completed_bar = None
        self._stop_requested = False
        self._state = PlaybackState.STOPPED
        self._notices = SimpleQueue()

    def snapshot(self) -> AudioPlaybackSnapshot:
        return AudioPlaybackSnapshot(
            state=self._state,
            current_bar=None,
            last_completed_bar=self._last_completed_bar,
            frame_in_bar=0,
            absolute_frame=self._absolute_frame,
            queue_depth=len(self._queued),
            underrun_count=0,
            buffered_seconds=_buffered_seconds(None, 0, self._queued),
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
                buffered_seconds=_buffered_seconds(None, 0, self._queued),
            )
        )

    def _validate_bar(self, bar: PreparedRapBar) -> None:
        if bar.audio.format != self._audio_format:
            raise ValueError("prepared bar audio format does not match sink format")


class TimedAudioSink:
    """Device-free realtime primary sink for standalone WAV output."""

    def output_device_status(self) -> dict[str, object]:
        return {"backend": "timed", "audible": False, "supports_device_selection": False}

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
        self._last_completed_bar: int | None = None
        self._frame_in_bar = 0
        self._absolute_frame = 0
        self._fractional_frames = 0.0
        self._last_clock = 0.0
        self._underrun_count = 0
        self._queue_underrun_active = False
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

    def request_stop_after_bar(self) -> AudioPlaybackSnapshot:
        with self._lock:
            if self._state not in (PlaybackState.RUNNING, PlaybackState.STOP_REQUESTED):
                return self._snapshot_locked()
            if self._active is None:
                self._finish_stop_locked()
            else:
                self._stop_requested = True
                self._state = PlaybackState.STOP_REQUESTED
            snapshot = self._snapshot_locked()
        self._wake.set()
        return snapshot

    def reset(self) -> None:
        with self._lock:
            self._queued.clear()
            self._active = None
            self._last_completed_bar = None
            self._frame_in_bar = 0
            self._absolute_frame = 0
            self._fractional_frames = 0.0
            self._underrun_count = 0
            self._queue_underrun_active = False
            self._stop_requested = False
            if self._state != PlaybackState.CLOSED:
                self._state = PlaybackState.STOPPED
            self._notices = SimpleQueue()
        self._wake.set()

    def discard_pending(self) -> None:
        with self._lock:
            if self._state == PlaybackState.CLOSED:
                raise RuntimeError("audio sink is closed")
            self._queued.clear()
            self._active = None
            self._last_completed_bar = None
            self._frame_in_bar = 0
            self._absolute_frame = 0
            self._fractional_frames = 0.0
            self._underrun_count = 0
            self._queue_underrun_active = False
            self._stop_requested = False
            self._state = PlaybackState.STOPPED
            self._notices = SimpleQueue()
        self._wake.set()

    def snapshot(self) -> AudioPlaybackSnapshot:
        with self._lock:
            return self._snapshot_locked()

    def _snapshot_locked(self) -> AudioPlaybackSnapshot:
        return AudioPlaybackSnapshot(
            state=self._state,
            current_bar=self._active.bar if self._active is not None else None,
            last_completed_bar=self._last_completed_bar,
            frame_in_bar=self._frame_in_bar,
            absolute_frame=self._absolute_frame,
            queue_depth=len(self._queued),
            underrun_count=self._underrun_count,
            buffered_seconds=_buffered_seconds(self._active, self._frame_in_bar, self._queued),
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
                    if not self._queue_underrun_active:
                        self._queue_underrun_active = True
                        self._underrun_count += 1
                        self._publish_locked(
                            AudioPlaybackNoticeKind.UNDERRUN,
                            None,
                            "prepared bar queue empty",
                        )
                    self._absolute_frame += remaining
                    return
                self._active = self._queued.popleft()
                self._frame_in_bar = 0
                self._queue_underrun_active = False
                self._publish_locked(AudioPlaybackNoticeKind.BAR_STARTED, self._active.bar, "bar playback started")

            active = self._active
            assert active is not None
            consumed = min(remaining, active.audio.frame_count - self._frame_in_bar)
            self._frame_in_bar += consumed
            self._absolute_frame += consumed
            remaining -= consumed
            if self._frame_in_bar != active.audio.frame_count:
                continue

            self._last_completed_bar = active.bar
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
                buffered_seconds=_buffered_seconds(self._active, self._frame_in_bar, self._queued),
            )
        )

    def _validate_bar(self, bar: PreparedRapBar) -> None:
        if bar.audio.format != self._audio_format:
            raise ValueError("prepared bar audio format does not match sink format")
