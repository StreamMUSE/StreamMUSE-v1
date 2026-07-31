"""Per-turn microphone capture with streaming WebRTC VAD endpointing."""

from __future__ import annotations

import math
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Protocol

import numpy as np

from . import (
    AudioQueueOverflowError,
    MicrophoneCaptureError,
    MicrophoneDeviceError,
    VoiceDependencyError,
    _json_safe,
)
from .audio import PortAudioClockMapper, resample_float32

if TYPE_CHECKING:
    from streammuse.application.tasks.human_input import VoiceInputConfig


SUPPORTED_VAD_SAMPLE_RATES: tuple[int, ...] = (16_000, 48_000, 32_000, 8_000)
_VAD_FRAME_MS = 20
_QUEUE_POLL_S = 0.02


class _Vad(Protocol):
    def is_speech(self, frame: bytes, sample_rate: int) -> bool: ...


@dataclass(frozen=True)
class MicrophoneDevice:
    """A JSON-safe description of an input-capable PortAudio device."""

    index: int
    name: str
    max_input_channels: int
    default_sample_rate_hz: float
    hostapi: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return _json_safe(
            {
                "index": self.index,
                "name": self.name,
                "max_input_channels": self.max_input_channels,
                "default_sample_rate_hz": self.default_sample_rate_hz,
                "hostapi": self.hostapi,
            }
        )


@dataclass(frozen=True)
class CapturedUtterance:
    """One bounded utterance normalized for faster-whisper."""

    audio: np.ndarray
    sample_rate_hz: int
    capture_sample_rate_hz: int
    endpoint_reason: str
    deadline_expired: bool
    wait_for_speech_ms: float
    utterance_ms: float
    endpoint_silence_ms: float
    last_voiced_offset_ms: float | None
    endpoint_detected_offset_ms: float
    audio_overflow: bool = False

    @property
    def has_speech(self) -> bool:
        return self.last_voiced_offset_ms is not None and self.audio.size > 0


@dataclass(frozen=True)
class _AudioChunk:
    data: bytes
    frame_count: int
    captured_start_s: float
    captured_end_s: float


class _PortAudioClockMapper(PortAudioClockMapper):
    """Backward-compatible input-clock specialization."""

    def __init__(self) -> None:
        super().__init__("inputBufferAdcTime")


def _import_sounddevice() -> Any:
    try:
        import sounddevice  # type: ignore[import-not-found]
    except (ImportError, OSError) as exc:
        raise VoiceDependencyError(
            "Voice input requires sounddevice. Install it with "
            "`uv sync --extra voice` (or install the project's voice extra)."
        ) from exc
    return sounddevice


def _import_vad_factory() -> Callable[[int], _Vad]:
    try:
        import webrtcvad  # type: ignore[import-not-found]
    except (ImportError, OSError) as exc:
        raise VoiceDependencyError(
            "Voice input requires webrtcvad-wheels. Install it with "
            "`uv sync --extra voice` (or install the project's voice extra)."
        ) from exc
    return webrtcvad.Vad


def _device_from_mapping(index: int, value: Any) -> MicrophoneDevice:
    getter = value.get if hasattr(value, "get") else lambda key, default=None: getattr(value, key, default)
    return MicrophoneDevice(
        index=int(index),
        name=str(getter("name", f"Device {index}")),
        max_input_channels=int(getter("max_input_channels", 0) or 0),
        default_sample_rate_hz=float(getter("default_samplerate", 0.0) or 0.0),
        hostapi=int(getter("hostapi")) if getter("hostapi") is not None else None,
    )


def enumerate_input_devices(*, sounddevice_module: Any | None = None) -> tuple[MicrophoneDevice, ...]:
    """Return input-capable devices without importing sounddevice eagerly."""

    sd = sounddevice_module or _import_sounddevice()
    try:
        devices = sd.query_devices()
    except Exception as exc:
        raise MicrophoneDeviceError(f"Could not enumerate microphone devices: {exc}") from exc

    result = []
    for index, raw_device in enumerate(devices):
        device = _device_from_mapping(index, raw_device)
        if device.max_input_channels > 0:
            result.append(device)
    return tuple(result)


class MicrophoneCapture:
    """Open the selected microphone only while capturing a human turn."""

    def __init__(
        self,
        config: VoiceInputConfig,
        *,
        sounddevice_module: Any | None = None,
        vad_factory: Callable[[int], _Vad] | None = None,
        now: Callable[[], float] | None = None,
    ) -> None:
        self.config = config
        self._sounddevice = sounddevice_module
        self._vad_factory = vad_factory
        self._now = now or time.perf_counter
        self._device: MicrophoneDevice | None = None
        self._sample_rate_hz: int | None = None
        self._vad: _Vad | None = None
        self._started = False
        self._closed = False
        self._close_event = threading.Event()
        self._active_stream: Any | None = None
        self._active_queue: queue.Queue[_AudioChunk] | None = None
        self._active_stream_closed = False
        self._state_lock = threading.Lock()

    @property
    def provenance(self) -> dict[str, Any]:
        selected = self._device.as_dict() if self._device is not None else None
        return _json_safe(
            {
                "microphone_device_requested": self.config.microphone_device,
                "microphone_device": selected,
                "capture_sample_rate_hz": self._sample_rate_hz,
                "vad_frame_ms": _VAD_FRAME_MS,
                "vad_aggressiveness": self.config.vad_aggressiveness,
                "start_timeout_ms": self.config.start_timeout_ms,
                "end_silence_ms": self.config.end_silence_ms,
                "max_utterance_ms": self.config.max_utterance_ms,
                "pre_roll_ms": self.config.pre_roll_ms,
                "queue_max_chunks": self.config.queue_max_chunks,
            }
        )

    def start(self) -> None:
        """Resolve and validate a microphone configuration without leaving it open."""

        if self._closed:
            raise MicrophoneCaptureError("The microphone capture has already been closed.")
        if self._started:
            return

        sd = self._sounddevice or _import_sounddevice()
        vad_factory = self._vad_factory or _import_vad_factory()
        device = self._resolve_device(sd)
        sample_rate = self._select_sample_rate(sd, device)
        try:
            vad = vad_factory(int(self.config.vad_aggressiveness))
        except Exception as exc:
            raise MicrophoneDeviceError(f"Could not initialize WebRTC VAD: {exc}") from exc

        self._preflight_stream(sd, device, sample_rate)

        self._sounddevice = sd
        self._device = device
        self._sample_rate_hz = sample_rate
        self._vad = vad
        self._started = True

    def _preflight_stream(self, sd: Any, device: MicrophoneDevice, sample_rate: int) -> None:
        """Exercise PortAudio permissions and ownership without keeping the mic open."""

        stream_device: int | str | None = (
            device.index if device.index >= 0 else self.config.microphone_device
        )

        def discard_callback(indata: Any, frames: int, time_info: Any, status: Any) -> None:
            del indata, frames, time_info, status

        stream: Any | None = None
        stopped = False
        closed = False
        try:
            stream = sd.RawInputStream(
                samplerate=sample_rate,
                blocksize=0,
                device=stream_device,
                channels=1,
                dtype="int16",
                callback=discard_callback,
            )
            stream.start()
            stream.stop()
            stopped = True
            stream.close()
            closed = True
        except BaseException as exc:
            # Cleanup is best effort so a secondary PortAudio failure cannot replace
            # the open/start/stop/close error that made startup unusable.
            if stream is not None and not stopped:
                try:
                    stream.stop()
                except BaseException:
                    pass
            if stream is not None and not closed:
                try:
                    stream.close()
                except BaseException:
                    pass
            if not isinstance(exc, Exception):
                raise
            raise MicrophoneDeviceError(
                f"Could not preflight microphone {device.name!r}: {exc}"
            ) from exc

    def _resolve_device(self, sd: Any) -> MicrophoneDevice:
        requested = self.config.microphone_device
        try:
            raw = sd.query_devices(requested, "input")
        except Exception as exc:
            label = "default input device" if requested is None else repr(requested)
            raise MicrophoneDeviceError(f"Could not select microphone {label}: {exc}") from exc

        index: int
        if requested is None:
            default_device = getattr(getattr(sd, "default", None), "device", (None, None))
            try:
                default_input = default_device[0]
            except (IndexError, KeyError, TypeError):
                default_input = default_device
            index = int(default_input) if default_input is not None else -1
        elif isinstance(requested, int):
            index = requested
        else:
            try:
                all_devices = sd.query_devices()
                index = next(
                    idx
                    for idx, candidate in enumerate(all_devices)
                    if candidate is raw or (
                        str(candidate.get("name", "")) == str(raw.get("name", ""))
                        and int(candidate.get("max_input_channels", 0)) > 0
                    )
                )
            except Exception:
                index = -1

        device = _device_from_mapping(index, raw)
        if device.max_input_channels < 1:
            raise MicrophoneDeviceError(f"Selected device {device.name!r} has no input channels.")
        return device

    def _select_sample_rate(self, sd: Any, device: MicrophoneDevice) -> int:
        failures: list[str] = []
        device_selector: int | str | None = self.config.microphone_device
        if device_selector is None and device.index >= 0:
            device_selector = device.index
        for sample_rate in SUPPORTED_VAD_SAMPLE_RATES:
            try:
                sd.check_input_settings(
                    device=device_selector,
                    channels=1,
                    dtype="int16",
                    samplerate=sample_rate,
                )
            except Exception as exc:
                failures.append(f"{sample_rate} Hz: {exc}")
                continue
            return sample_rate
        detail = "; ".join(failures)
        raise MicrophoneDeviceError(
            f"Microphone {device.name!r} cannot open at a WebRTC VAD-supported sample "
            f"rate ({', '.join(str(rate) for rate in SUPPORTED_VAD_SAMPLE_RATES)} Hz). "
            f"44.1 kHz-only devices are not supported. Details: {detail}"
        )

    def _calibrate_stream_clock(
        self,
        stream: Any,
        mapper: _PortAudioClockMapper,
    ) -> None:
        # Avoid perturbing lightweight fakes or backends that do not expose the
        # PortAudio stream clock. Real sounddevice streams define ``time`` on the
        # class and use the same clock as callback time_info.
        if not hasattr(type(stream), "time") and "time" not in getattr(stream, "__dict__", {}):
            return
        local_before_s = self._now()
        try:
            portaudio_s = float(stream.time)
        except Exception:
            return
        local_after_s = self._now()
        mapper.calibrate(
            portaudio_s=portaudio_s,
            local_s=(local_before_s + local_after_s) / 2.0,
        )

    def capture(self, *, timeout_s: float | None) -> CapturedUtterance:
        """Capture one bounded utterance and return 16-kHz float32 audio."""

        if not self._started:
            raise MicrophoneCaptureError("MicrophoneCapture.start() must succeed before capture().")
        with self._state_lock:
            if self._closed:
                raise MicrophoneCaptureError("The microphone capture has already been closed.")
        assert self._sounddevice is not None
        assert self._sample_rate_hz is not None
        assert self._vad is not None

        sample_rate = self._sample_rate_hz
        frame_bytes = sample_rate * _VAD_FRAME_MS // 1000 * 2
        chunk_queue: queue.Queue[_AudioChunk] = queue.Queue(maxsize=int(self.config.queue_max_chunks))
        callback_errors: queue.Queue[BaseException] = queue.Queue(maxsize=1)
        audio_clock = _PortAudioClockMapper()
        capture_start_s = self._now()
        game_deadline_s = None if timeout_s is None else capture_start_s + max(0.0, float(timeout_s))
        start_timeout_s = capture_start_s + float(self.config.start_timeout_ms) / 1000.0

        def callback(indata: Any, frames: int, time_info: Any, status: Any) -> None:
            callback_local_s = self._now()
            if status:
                self._set_callback_error(
                    callback_errors,
                    MicrophoneCaptureError(f"PortAudio callback status: {status}"),
                )
                return
            try:
                captured_start_s, captured_end_s = audio_clock.capture_interval(
                    time_info,
                    callback_local_s=callback_local_s,
                    frame_count=int(frames),
                    sample_rate=sample_rate,
                )
                chunk_queue.put_nowait(
                    _AudioChunk(
                        data=bytes(indata),
                        frame_count=int(frames),
                        captured_start_s=captured_start_s,
                        captured_end_s=captured_end_s,
                    )
                )
            except queue.Full:
                self._set_callback_error(
                    callback_errors,
                    AudioQueueOverflowError(
                        "Microphone callback queue overflowed; audio was not silently discarded. "
                        "Close competing CPU-intensive processes or increase queue_max_chunks."
                    )
                )
            except Exception as exc:
                self._set_callback_error(
                    callback_errors,
                    MicrophoneCaptureError(f"Could not copy PortAudio input buffer: {exc}"),
                )

        stream: Any | None = None
        try:
            assert self._device is not None
            stream_device: int | str | None = (
                self._device.index if self._device.index >= 0 else self.config.microphone_device
            )
            with self._state_lock:
                if self._closed:
                    raise MicrophoneCaptureError(
                        "The microphone capture was closed before the input stream opened."
                    )
                stream = self._sounddevice.RawInputStream(
                    samplerate=sample_rate,
                    blocksize=0,
                    device=stream_device,
                    channels=1,
                    dtype="int16",
                    callback=callback,
                )
                self._active_stream = stream
                self._active_queue = chunk_queue
                self._active_stream_closed = False
                self._calibrate_stream_clock(stream, audio_clock)
                stream.start()
                self._calibrate_stream_clock(stream, audio_clock)
        except BaseException as exc:
            if stream is not None:
                try:
                    self._close_stream(stream)
                except BaseException:
                    pass
            with self._state_lock:
                self._active_stream = None
                self._active_queue = None
                self._active_stream_closed = False
            if not isinstance(exc, Exception):
                raise
            raise MicrophoneCaptureError(f"Could not open microphone input stream: {exc}") from exc

        try:
            result = self._consume_audio(
                chunk_queue,
                frame_bytes=frame_bytes,
                sample_rate=sample_rate,
                capture_start_s=capture_start_s,
                game_deadline_s=game_deadline_s,
                start_timeout_s=start_timeout_s,
                callback_errors=callback_errors,
            )
        except BaseException:
            try:
                self._close_stream(stream)
            except BaseException:
                pass
            raise
        else:
            self._close_stream(stream)
        finally:
            with self._state_lock:
                if self._active_stream is stream:
                    self._active_stream = None
                    self._active_queue = None
                    self._active_stream_closed = False
        return result

    def _consume_audio(
        self,
        chunk_queue: queue.Queue[_AudioChunk],
        *,
        frame_bytes: int,
        sample_rate: int,
        capture_start_s: float,
        game_deadline_s: float | None,
        start_timeout_s: float,
        callback_errors: queue.Queue[BaseException],
    ) -> CapturedUtterance:
        accumulator = bytearray()
        pre_roll_frames = max(0, math.ceil(float(self.config.pre_roll_ms) / _VAD_FRAME_MS))
        pre_roll: deque[bytes] = deque(maxlen=pre_roll_frames or 1)
        captured: list[bytes] = []
        speech_started = False
        speech_started_s: float | None = None
        first_voiced_offset_ms: float | None = None
        last_voiced_offset_ms: float | None = None
        audio_cursor_ms = 0.0
        trailing_silence_ms = 0.0
        endpoint_reason: str | None = None
        deadline_expired = False

        while endpoint_reason is None:
            self._raise_callback_error(callback_errors)
            if self._close_event.is_set():
                raise MicrophoneCaptureError("Microphone capture was closed while waiting for audio.")

            now_s = self._now()
            deadline_reached = game_deadline_s is not None and now_s >= game_deadline_s
            if deadline_reached and chunk_queue.empty():
                endpoint_reason = "deadline"
                deadline_expired = True
                break
            if not speech_started and now_s >= start_timeout_s and chunk_queue.empty():
                endpoint_reason = "start_timeout"
                break
            if (
                speech_started_s is not None
                and now_s >= speech_started_s + float(self.config.max_utterance_ms) / 1000.0
                and chunk_queue.empty()
            ):
                endpoint_reason = "max_utterance"
                break

            wait_s = _QUEUE_POLL_S
            next_limit = start_timeout_s if not speech_started else None
            if game_deadline_s is not None:
                next_limit = game_deadline_s if next_limit is None else min(next_limit, game_deadline_s)
            if next_limit is not None:
                wait_s = max(0.001, min(wait_s, next_limit - now_s))
            try:
                if deadline_reached:
                    chunk = chunk_queue.get_nowait()
                else:
                    chunk = chunk_queue.get(timeout=wait_s)
            except queue.Empty:
                if deadline_reached:
                    endpoint_reason = "deadline"
                    deadline_expired = True
                    break
                continue
            expected_bytes = chunk.frame_count * 2
            if chunk.frame_count < 0 or len(chunk.data) != expected_bytes:
                raise MicrophoneCaptureError(
                    "PortAudio returned an inconsistent mono int16 buffer: "
                    f"frames={chunk.frame_count}, bytes={len(chunk.data)}, expected={expected_bytes}."
                )
            chunk_start_s = chunk.captured_start_s
            chunk_start_offset_ms = max(0.0, (chunk_start_s - capture_start_s) * 1000.0)
            pending_audio_ms = len(accumulator) / 2 / sample_rate * 1000.0
            expected_chunk_start_ms = audio_cursor_ms + pending_audio_ms
            if (
                accumulator
                and chunk_start_offset_ms
                > expected_chunk_start_ms + _VAD_FRAME_MS
            ):
                # A partial VAD frame cannot bridge a real callback gap. Drop the
                # sub-frame residue and resume from the new chunk's capture time.
                accumulator.clear()
                audio_cursor_ms = max(audio_cursor_ms, chunk_start_offset_ms)
            elif not accumulator:
                audio_cursor_ms = max(audio_cursor_ms, chunk_start_offset_ms)
            chunk_data = chunk.data
            ends_at_deadline = False
            if game_deadline_s is not None and chunk.captured_end_s > game_deadline_s:
                allowed_frames = math.floor(
                    max(0.0, game_deadline_s - chunk_start_s) * sample_rate
                )
                allowed_frames = min(chunk.frame_count, allowed_frames)
                chunk_data = chunk.data[: allowed_frames * 2]
                ends_at_deadline = True
            accumulator.extend(chunk_data)

            while len(accumulator) >= frame_bytes and endpoint_reason is None:
                frame_start_ms = audio_cursor_ms
                frame_end_ms = frame_start_ms + _VAD_FRAME_MS
                if (
                    not speech_started
                    and capture_start_s + frame_end_ms / 1000.0 > start_timeout_s
                ):
                    endpoint_reason = "start_timeout"
                    break
                if (
                    speech_started_s is not None
                    and capture_start_s + frame_end_ms / 1000.0
                    > speech_started_s + float(self.config.max_utterance_ms) / 1000.0
                ):
                    endpoint_reason = "max_utterance"
                    break
                frame = bytes(accumulator[:frame_bytes])
                del accumulator[:frame_bytes]
                audio_cursor_ms = frame_end_ms
                try:
                    voiced = bool(self._vad.is_speech(frame, sample_rate))  # type: ignore[union-attr]
                except Exception as exc:
                    raise MicrophoneCaptureError(f"WebRTC VAD failed: {exc}") from exc

                if not speech_started:
                    if voiced:
                        speech_started = True
                        speech_started_s = capture_start_s + frame_start_ms / 1000.0
                        first_voiced_offset_ms = frame_start_ms
                        if pre_roll_frames:
                            captured.extend(pre_roll)
                        captured.append(frame)
                        last_voiced_offset_ms = frame_end_ms
                    elif pre_roll_frames:
                        pre_roll.append(frame)
                    continue

                captured.append(frame)
                if voiced:
                    last_voiced_offset_ms = frame_end_ms
                    trailing_silence_ms = 0.0
                else:
                    trailing_silence_ms += _VAD_FRAME_MS

                utterance_elapsed_ms = frame_end_ms - float(first_voiced_offset_ms or 0.0)
                if utterance_elapsed_ms >= float(self.config.max_utterance_ms):
                    endpoint_reason = "max_utterance"
                elif trailing_silence_ms >= float(self.config.end_silence_ms):
                    endpoint_reason = "trailing_silence"

            if endpoint_reason is None and ends_at_deadline:
                endpoint_reason = "deadline"
                deadline_expired = True

            self._raise_callback_error(callback_errors)

        endpoint_offset_ms = max(audio_cursor_ms, (self._now() - capture_start_s) * 1000.0)
        if not speech_started:
            return CapturedUtterance(
                audio=np.zeros(0, dtype=np.float32),
                sample_rate_hz=16_000,
                capture_sample_rate_hz=sample_rate,
                endpoint_reason=endpoint_reason,
                deadline_expired=deadline_expired,
                wait_for_speech_ms=endpoint_offset_ms,
                utterance_ms=0.0,
                endpoint_silence_ms=0.0,
                last_voiced_offset_ms=None,
                endpoint_detected_offset_ms=endpoint_offset_ms,
            )

        pcm = b"".join(captured)
        audio = self._pcm_to_float32(pcm, sample_rate)
        first_voiced = float(first_voiced_offset_ms or 0.0)
        last_voiced = float(last_voiced_offset_ms or first_voiced)
        return CapturedUtterance(
            audio=audio,
            sample_rate_hz=16_000,
            capture_sample_rate_hz=sample_rate,
            endpoint_reason=endpoint_reason,
            deadline_expired=deadline_expired,
            wait_for_speech_ms=first_voiced,
            utterance_ms=max(0.0, last_voiced - first_voiced),
            endpoint_silence_ms=max(0.0, audio_cursor_ms - last_voiced),
            last_voiced_offset_ms=last_voiced,
            endpoint_detected_offset_ms=endpoint_offset_ms,
        )

    @staticmethod
    def _pcm_to_float32(pcm: bytes, sample_rate: int) -> np.ndarray:
        source = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        return resample_float32(source, sample_rate, 16_000)

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
    def _raise_callback_error(callback_errors: queue.Queue[BaseException]) -> None:
        try:
            error = callback_errors.get_nowait()
        except queue.Empty:
            return
        raise error

    def _close_stream(self, stream: Any) -> None:
        with self._state_lock:
            if self._active_stream is stream:
                if self._active_stream_closed:
                    return
                self._active_stream_closed = True
        first_error: BaseException | None = None
        try:
            stream.stop()
        except BaseException as exc:
            first_error = exc
        try:
            stream.close()
        except BaseException as exc:
            if first_error is None:
                first_error = exc
        if first_error is None:
            return
        if not isinstance(first_error, Exception):
            raise first_error
        raise MicrophoneCaptureError(
            f"Could not clean up microphone input stream: {first_error}"
        ) from first_error

    def close(self) -> None:
        """Stop active capture and release resources. Safe to call repeatedly."""

        with self._state_lock:
            if self._closed:
                return
            self._closed = True
            self._close_event.set()
            stream = self._active_stream
        if stream is not None:
            self._close_stream(stream)
