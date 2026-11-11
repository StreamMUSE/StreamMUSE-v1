"""Callback output for inference spans, exclusively leased by the rap runtime."""

from queue import Full, Queue
from threading import Event, RLock, Thread
import time
import wave

import numpy as np

from streammuse.infrastructure.rap.audio_output import (
    _NATIVE_CONTROL_LOCK, _default_output_stream_factory,
)


class InferenceAudioOutput:
    def __init__(self, *, audio_format, device=None, stream_factory=None, on_error=None,
                 on_warning=None, recording_path=None, active_probe=None):
        self.audio_format = audio_format
        self.device = device
        self._factory = stream_factory or _default_output_stream_factory
        self._on_error = on_error
        self._on_warning = on_warning
        self._active_probe = active_probe or (lambda: True)
        self._stream = None
        self._lock = RLock()
        self._opened_device = None
        self._last_callback = None
        self._callback_count = 0
        self._errors = 0
        self._recording_path = recording_path
        self._record_queue = Queue(maxsize=128)
        self._writer = None
        self._recording_error = None
        self._recording_enabled = False
        self._recording_dropped_frames = 0
        self._recording_skipped_idle_frames = 0
        self._warnings = 0
        self._last_warning = None

    def set_error_handler(self, handler, *, warning_handler=None, active_probe=None):
        self._on_error = handler
        self._on_warning = warning_handler
        if active_probe is not None:
            self._active_probe = active_probe

    def start(self, render):
        with self._lock, _NATIVE_CONTROL_LOCK:
            if self._stream is not None:
                if self._opened_device == self.device:
                    return
                self.suspend()

            def callback(outdata, frames, time_info, status):
                try:
                    self._callback_count += 1
                    self._last_callback = {
                        "clock_domain": "portaudio", "frames": frames,
                        "current_time": getattr(time_info, "currentTime", None),
                        "output_buffer_dac_time": getattr(time_info, "outputBufferDacTime", None),
                        "status": str(status) if status else None,
                    }
                    if status:
                        self._warn(f"PortAudio callback status: {status}")
                    active_before = self._active_probe()
                    outdata[:] = render(frames)
                    record_active = active_before or self._active_probe()
                    if self._recording_enabled and record_active:
                        try:
                            self._record_queue.put_nowait(outdata.copy())
                        except Full:
                            self._recording_dropped_frames += frames
                            if self._recording_dropped_frames == frames:
                                self._warn("inference recording queue full; playback continues")
                    elif self._recording_enabled:
                        self._recording_skipped_idle_frames += frames
                except Exception as error:
                    outdata.fill(0)
                    self._errors += 1
                    if self._on_error is not None:
                        self._on_error(str(error))

            if self._recording_path is not None and self._writer is None:
                self._recording_enabled = True
                self._writer = Thread(target=self._write_recording, name="inference-pcm-recording", daemon=True)
                self._writer.start()
            stream = self._factory(audio_format=self.audio_format, callback=callback, device=self.device)
            try:
                stream.start()
            except BaseException:
                stream.close()
                raise
            self._stream = stream
            self._opened_device = self.device

    def snapshot(self):
        return {"callback_count": self._callback_count, "callback_errors": self._errors,
                "last_callback": self._last_callback, "device": self.device,
                "recording_path": str(self._recording_path) if self._recording_path else None,
                "recording_error": self._recording_error,
                "recording_dropped_frames": self._recording_dropped_frames,
                "recording_skipped_idle_frames": self._recording_skipped_idle_frames,
                "recording_timeline": "active callbacks including silent WAIT; stopped periods omitted",
                "warnings": self._warnings, "last_warning": self._last_warning}

    def _warn(self, message):
        self._warnings += 1
        self._last_warning = message
        if self._on_warning is not None:
            self._on_warning(message)

    def _write_recording(self):
        try:
            with wave.open(str(self._recording_path), "wb") as recording:
                recording.setnchannels(self.audio_format.channels)
                recording.setsampwidth(2)
                recording.setframerate(self.audio_format.sample_rate_hz)
                while True:
                    block = self._record_queue.get()
                    if block is None:
                        break
                    recording.writeframes((np.clip(block, -1, 1) * 32767).astype("<i2").tobytes())
        except Exception as error:
            self._recording_error = str(error)
            self._recording_enabled = False
            self._warn(f"inference recording failed: {error}")

    def suspend(self):
        with self._lock, _NATIVE_CONTROL_LOCK:
            stream, self._stream = self._stream, None
            if stream is not None:
                try:
                    stream.stop()
                finally:
                    stream.close()

    def close(self):
        self.suspend()
        self._recording_enabled = False
        writer, self._writer = self._writer, None
        deadline = time.monotonic() + 2
        while writer is not None and writer.is_alive():
            try:
                self._record_queue.put(None, timeout=0.05)
                break
            except Full:
                if time.monotonic() >= deadline:
                    break
        if writer is not None:
            writer.join(timeout=max(0, deadline - time.monotonic()))
            if writer.is_alive():
                self._recording_error = "recording writer did not finish within two seconds"
                self._warn(self._recording_error)


class TimedInferenceStream:
    """Clocked callback driver for the existing WAV-only CLI output choice."""

    def __init__(self, *, audio_format, callback, device=None):
        self._format, self._callback = audio_format, callback
        self._stop = Event()
        self._thread = None

    def start(self):
        def run():
            frames = max(1, round(self._format.sample_rate_hz * 0.01))
            block = np.zeros((frames, self._format.channels), dtype=np.float32)
            origin, total = time.monotonic(), 0
            while not self._stop.is_set():
                self._callback(block, frames, None, None)
                total += frames
                self._stop.wait(max(0, origin + total / self._format.sample_rate_hz - time.monotonic()))
        self._thread = Thread(target=run, name="inference-wav-clock", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()

    def close(self):
        self.stop()
