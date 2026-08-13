"""Callback-based PortAudio speaker playback."""

from __future__ import annotations

import math
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from . import (
    SpeakerDeviceError,
    SpeakerPlaybackError,
    VoiceDependencyError,
    _json_safe,
)
from .audio import PortAudioClockMapper, resample_float32
from .synthesizer import SynthesizedAudio


@dataclass(frozen=True)
class SpeakerDevice:
    index: int
    name: str
    max_output_channels: int
    default_sample_rate_hz: float
    hostapi: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return _json_safe(
            {
                "index": self.index,
                "name": self.name,
                "max_output_channels": self.max_output_channels,
                "default_sample_rate_hz": self.default_sample_rate_hz,
                "hostapi": self.hostapi,
            }
        )


@dataclass(frozen=True)
class SpeakerPlayback:
    completed_normally: bool
    playback_start_offset_ms: float
    first_dac_sample_offset_ms: float | None
    playback_drained_offset_ms: float | None
    stream_inactive_offset_ms: float | None
    sample_rate_hz: int
    device: str
    error: SpeakerPlaybackError | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def _import_sounddevice() -> Any:
    try:
        import sounddevice  # type: ignore[import-not-found]
    except (ImportError, OSError) as exc:
        raise VoiceDependencyError(
            "Speech playback requires sounddevice. Install it with "
            "`uv sync --extra speech`."
        ) from exc
    return sounddevice


def _device_from_mapping(index: int, value: Any) -> SpeakerDevice:
    getter = (
        value.get
        if hasattr(value, "get")
        else lambda key, default=None: getattr(value, key, default)
    )
    return SpeakerDevice(
        index=int(index),
        name=str(getter("name", f"Device {index}")),
        max_output_channels=int(getter("max_output_channels", 0) or 0),
        default_sample_rate_hz=float(getter("default_samplerate", 0.0) or 0.0),
        hostapi=int(getter("hostapi")) if getter("hostapi") is not None else None,
    )


def enumerate_output_devices(
    *,
    sounddevice_module: Any | None = None,
) -> tuple[SpeakerDevice, ...]:
    sd = sounddevice_module or _import_sounddevice()
    try:
        devices = sd.query_devices()
    except Exception as exc:
        raise SpeakerDeviceError(f"Could not enumerate speaker devices: {exc}") from exc
    return tuple(
        device
        for index, raw in enumerate(devices)
        if (device := _device_from_mapping(index, raw)).max_output_channels > 0
    )


class SpeakerPlayer:
    def __init__(
        self,
        *,
        device: str | int | None = None,
        sounddevice_module: Any | None = None,
        now: Callable[[], float] | None = None,
        wait_margin_s: float = 1.0,
    ) -> None:
        self.requested_device = device
        self._sounddevice = sounddevice_module
        self._now = now or time.perf_counter
        self._wait_margin_s = float(wait_margin_s)
        self._device: SpeakerDevice | None = None
        self._started = False
        self._closed = False
        self._active_stream: Any | None = None
        self._active_inactive_event: threading.Event | None = None
        self._active_abort_requested: threading.Event | None = None

    def start(self) -> None:
        if self._started and not self._closed:
            return
        sd = self._sounddevice or _import_sounddevice()
        self._sounddevice = sd
        try:
            resolved_device = self.requested_device
            if resolved_device is None:
                default_device = getattr(sd, "default", None)
                default_pair = getattr(default_device, "device", None)
                if isinstance(default_pair, (tuple, list)):
                    resolved_device = default_pair[-1]
            raw = sd.query_devices(resolved_device, "output")
            raw_index = raw.get("index") if hasattr(raw, "get") else getattr(raw, "index", None)
            index = int(
                raw_index
                if raw_index is not None
                else (resolved_device if resolved_device is not None else 0)
            )
            device = _device_from_mapping(index, raw)
        except Exception as exc:
            raise SpeakerDeviceError(
                f"Could not resolve speaker device {self.requested_device!r}: {exc}"
            ) from exc
        if device.max_output_channels <= 0:
            raise SpeakerDeviceError(f"Device {device.name!r} has no output channels")
        self._device = device
        self._started = True
        self._closed = False

    @property
    def provenance(self) -> dict[str, Any]:
        return {
            "device": None if self._device is None else self._device.as_dict(),
        }

    def play(self, audio: SynthesizedAudio) -> SpeakerPlayback:
        if not self._started or self._closed or self._device is None:
            raise SpeakerPlaybackError("Speaker player has not been started")
        sd = self._sounddevice
        if sd is None:
            raise SpeakerPlaybackError("sounddevice is unavailable")
        play_started_s = self._now()
        samples, sample_rate = self._prepare_audio(audio)
        audio_prepared_s = self._now()
        inactive_event = threading.Event()
        abort_requested = threading.Event()
        callback_errors: queue.Queue[BaseException] = queue.Queue(maxsize=1)
        termination_kind = ["running"]
        frame_cursor = [0]
        first_callback_s: list[float | None] = [None]
        first_dac_s: list[float | None] = [None]
        last_frame_submitted_s: list[float | None] = [None]
        stream_inactive_s: list[float | None] = [None]
        clock = PortAudioClockMapper("outputBufferDacTime")

        def finished_callback() -> None:
            stream_inactive_s[0] = self._now()
            inactive_event.set()

        def callback(
            outdata: Any,
            frames: int,
            time_info: Any,
            status: Any,
        ) -> None:
            finished = False
            try:
                callback_local_s = self._now()
                if first_callback_s[0] is None:
                    first_callback_s[0] = callback_local_s
                if status:
                    raise SpeakerPlaybackError(f"Speaker callback status: {status}")
                start = frame_cursor[0]
                end = min(start + int(frames), samples.size)
                outdata[...] = 0
                if end > start:
                    outdata[: end - start, 0] = samples[start:end]
                    if first_dac_s[0] is None:
                        first_dac_s[0] = clock.buffer_start(
                            time_info,
                            callback_local_s=callback_local_s,
                        )
                frame_cursor[0] = end
                finished = end >= samples.size
                if finished:
                    last_frame_submitted_s[0] = callback_local_s
                    termination_kind[0] = "normal"
            except Exception as exc:
                termination_kind[0] = "callback_error"
                self._set_callback_error(callback_errors, exc)
                raise sd.CallbackAbort
            if finished:
                raise sd.CallbackStop

        stream: Any | None = None
        playback_start_s = self._now()
        stream_open_started_s = playback_start_s
        stream_opened_s: float | None = None
        stream_started_s: float | None = None
        try:
            try:
                stream_open_started_s = self._now()
                stream = sd.OutputStream(
                    samplerate=sample_rate,
                    blocksize=0,
                    device=self._device.index,
                    channels=1,
                    dtype="float32",
                    callback=callback,
                    finished_callback=finished_callback,
                )
                stream_opened_s = self._now()
            except Exception as exc:
                raise SpeakerPlaybackError(
                    f"Could not open speaker stream: {exc}"
                ) from exc
            self._active_stream = stream
            self._active_inactive_event = inactive_event
            self._active_abort_requested = abort_requested
            playback_start_s = self._now()
            try:
                stream.start()
                stream_started_s = self._now()
            except Exception as exc:
                raise SpeakerPlaybackError(
                    f"Could not start speaker stream: {exc}"
                ) from exc
            latency = self._stream_latency(stream)
            timeout_s = audio.duration_ms / 1000.0 + latency + self._wait_margin_s
            playback_error: SpeakerPlaybackError | None = None
            if not inactive_event.wait(max(0.001, timeout_s)):
                abort_requested.set()
                try:
                    stream.abort()
                except Exception as exc:
                    playback_error = SpeakerPlaybackError(
                        f"Could not abort timed-out speaker stream: {exc}"
                    )
                inactive_event.wait(max(0.05, latency + self._wait_margin_s))
                if playback_error is None:
                    playback_error = SpeakerPlaybackError(
                        f"Speaker playback did not finish within {timeout_s:.3f}s"
                    )
            callback_error = self._get_callback_error(callback_errors)
            if callback_error is not None:
                if isinstance(callback_error, SpeakerPlaybackError):
                    playback_error = callback_error
                else:
                    raise callback_error
            completed_normally = (
                termination_kind[0] == "normal"
                and not abort_requested.is_set()
                and playback_error is None
            )
            inactive_s = stream_inactive_s[0]
            drained_s = inactive_s if completed_normally else None
            timing = {
                "schema_version": 1,
                "clock": "monotonic",
                "origin": "speaker_play_start",
                "anchors_ms": {
                    "audio_prepared": _offset_ms(
                        audio_prepared_s,
                        play_started_s,
                    ),
                    "stream_open_started": _offset_ms(
                        stream_open_started_s,
                        play_started_s,
                    ),
                    "stream_opened": _offset_ms(
                        stream_opened_s,
                        play_started_s,
                    ),
                    "stream_start_requested": _offset_ms(
                        playback_start_s,
                        play_started_s,
                    ),
                    "stream_started": _offset_ms(
                        stream_started_s,
                        play_started_s,
                    ),
                    "first_callback": _offset_ms(
                        first_callback_s[0],
                        play_started_s,
                    ),
                    "first_dac_sample": _offset_ms(
                        first_dac_s[0],
                        play_started_s,
                    ),
                    "last_frame_submitted": _offset_ms(
                        last_frame_submitted_s[0],
                        play_started_s,
                    ),
                    "stream_inactive": _offset_ms(
                        inactive_s,
                        play_started_s,
                    ),
                },
                "durations_ms": {
                    "audio_prepare": _elapsed_ms(
                        play_started_s,
                        audio_prepared_s,
                    ),
                    "stream_open": _optional_elapsed_ms(
                        stream_open_started_s,
                        stream_opened_s,
                    ),
                    "stream_start": _optional_elapsed_ms(
                        playback_start_s,
                        stream_started_s,
                    ),
                    "start_to_first_callback": _optional_elapsed_ms(
                        playback_start_s,
                        first_callback_s[0],
                    ),
                    "start_to_first_dac": _optional_elapsed_ms(
                        playback_start_s,
                        first_dac_s[0],
                    ),
                    "first_dac_to_inactive": _optional_elapsed_ms(
                        first_dac_s[0],
                        inactive_s,
                    ),
                    "pipeline_total": _optional_elapsed_ms(
                        play_started_s,
                        inactive_s,
                    ),
                },
            }
            return SpeakerPlayback(
                completed_normally=completed_normally,
                playback_start_offset_ms=max(
                    0.0, (playback_start_s - play_started_s) * 1000.0
                ),
                first_dac_sample_offset_ms=_offset_ms(
                    first_dac_s[0], play_started_s
                ),
                playback_drained_offset_ms=_offset_ms(
                    drained_s, play_started_s
                ),
                stream_inactive_offset_ms=_offset_ms(
                    inactive_s, play_started_s
                ),
                sample_rate_hz=sample_rate,
                device=self._device.name,
                error=playback_error,
                metadata={"timing_breakdown": timing},
            )
        except BaseException:
            abort_requested.set()
            if stream is not None:
                try:
                    stream.abort()
                except BaseException:
                    pass
                inactive_event.wait(0.25)
            raise
        finally:
            self._active_stream = None
            self._active_inactive_event = None
            self._active_abort_requested = None
            if stream is not None:
                try:
                    stream.close()
                except BaseException:
                    pass

    def _prepare_audio(
        self,
        audio: SynthesizedAudio,
    ) -> tuple[np.ndarray, int]:
        if self._sounddevice is None or self._device is None:
            raise SpeakerPlaybackError("Speaker player has not been started")
        requested_rate = int(audio.sample_rate_hz)
        try:
            self._sounddevice.check_output_settings(
                device=self._device.index,
                channels=1,
                dtype="float32",
                samplerate=requested_rate,
            )
            return audio.samples, requested_rate
        except Exception:
            target_rate = int(round(self._device.default_sample_rate_hz))
            if target_rate <= 0:
                raise SpeakerDeviceError(
                    f"Speaker {self._device.name!r} has no usable sample rate"
                )
            try:
                self._sounddevice.check_output_settings(
                    device=self._device.index,
                    channels=1,
                    dtype="float32",
                    samplerate=target_rate,
                )
            except Exception as exc:
                raise SpeakerDeviceError(
                    f"Speaker {self._device.name!r} cannot play mono float32 audio "
                    f"at {requested_rate} or {target_rate} Hz: {exc}"
                ) from exc
            return (
                resample_float32(audio.samples, requested_rate, target_rate),
                target_rate,
            )

    @staticmethod
    def _stream_latency(stream: Any) -> float:
        try:
            value = stream.latency
            if isinstance(value, (tuple, list)):
                value = value[-1]
            latency = float(value)
            return latency if math.isfinite(latency) and latency >= 0 else 0.0
        except (AttributeError, TypeError, ValueError):
            return 0.0

    @staticmethod
    def _set_callback_error(
        callback_errors: queue.Queue[BaseException],
        error: BaseException,
    ) -> None:
        try:
            callback_errors.put_nowait(error)
        except queue.Full:
            pass

    @staticmethod
    def _get_callback_error(
        callback_errors: queue.Queue[BaseException],
    ) -> BaseException | None:
        try:
            return callback_errors.get_nowait()
        except queue.Empty:
            return None

    def drain(self) -> None:
        event = self._active_inactive_event
        if event is not None:
            event.wait()

    def abort_active(self) -> None:
        stream = self._active_stream
        if stream is not None:
            abort_requested = self._active_abort_requested
            if abort_requested is not None:
                abort_requested.set()
            try:
                stream.abort()
            finally:
                event = self._active_inactive_event
                if event is not None:
                    event.wait(0.25)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.abort_active()


def _offset_ms(value_s: float | None, origin_s: float) -> float | None:
    if value_s is None:
        return None
    return max(0.0, (value_s - origin_s) * 1000.0)


def _elapsed_ms(start_s: float, end_s: float) -> float:
    return max(0.0, (float(end_s) - float(start_s)) * 1000.0)


def _optional_elapsed_ms(
    start_s: float | None,
    end_s: float | None,
) -> float | None:
    if start_s is None or end_s is None:
        return None
    return _elapsed_ms(start_s, end_s)
