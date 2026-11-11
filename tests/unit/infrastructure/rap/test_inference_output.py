import numpy as np
from types import SimpleNamespace
import wave
from threading import Event
from time import monotonic

from streammuse.domain.rap.audio import AudioFormat
from streammuse.infrastructure.rap.inference_output import InferenceAudioOutput


class Stream:
    def __init__(self, **kwargs):
        self.callback = kwargs["callback"]
        self.running = False
        self.closed = False

    def start(self):
        self.running = True

    def stop(self):
        self.running = False

    def close(self):
        self.closed = True


def test_native_callback_pulls_exact_frames_and_suspends_for_legacy():
    streams = []

    def factory(**kwargs):
        streams.append(Stream(**kwargs))
        return streams[-1]

    output = InferenceAudioOutput(audio_format=AudioFormat(), stream_factory=factory)
    output.start(lambda frames: np.full((frames, 2), 0.25, dtype=np.float32))
    out = np.empty((137, 2), dtype=np.float32)
    streams[-1].callback(out, 137, None, None)
    assert np.allclose(out, 0.25)
    output.suspend()
    assert streams[-1].closed
    assert not streams[-1].running
    output.start(lambda frames: np.zeros((frames, 2), dtype=np.float32))
    assert len(streams) == 2
    output.close()
    assert all(stream.closed for stream in streams)


def test_callback_failure_is_silent_and_reported():
    errors, streams = [], []

    def factory(**kwargs):
        streams.append(Stream(**kwargs))
        return streams[-1]

    def broken(frames):
        raise RuntimeError("callback failure")

    output = InferenceAudioOutput(audio_format=AudioFormat(), stream_factory=factory, on_error=errors.append)
    output.start(broken)
    out = np.ones((64, 2), dtype=np.float32)
    streams[-1].callback(out, 64, None, None)
    assert not np.any(out)
    assert errors == ["callback failure"]
    output.close()


def test_device_changes_reopen_and_callback_status_is_visible():
    streams, errors, warnings = [], [], []

    def factory(**kwargs):
        streams.append(Stream(**kwargs))
        return streams[-1]

    output = InferenceAudioOutput(audio_format=AudioFormat(), stream_factory=factory,
                                   on_error=errors.append, on_warning=warnings.append)
    render = lambda frames: np.full((frames, 2), 0.25, dtype=np.float32)
    output.start(render)
    output.device = "different device"
    output.start(render)
    assert len(streams) == 2 and streams[0].closed
    block = np.zeros((32, 2))
    streams[-1].callback(block, 32,
                         SimpleNamespace(currentTime=10, outputBufferDacTime=10.02), "output underflow")
    assert not errors
    assert warnings and "underflow" in warnings[-1]
    assert np.allclose(block, 0.25)
    assert output.snapshot()["last_callback"]["output_buffer_dac_time"] == 10.02
    output.close()


def test_callback_audio_recording_is_flushed_on_close(tmp_path):
    streams = []

    def factory(**kwargs):
        streams.append(Stream(**kwargs))
        return streams[-1]

    path = tmp_path / "inference.wav"
    output = InferenceAudioOutput(audio_format=AudioFormat(), stream_factory=factory, recording_path=path)
    output.start(lambda frames: np.full((frames, 2), 0.25, dtype=np.float32))
    streams[-1].callback(np.zeros((137, 2)), 137, None, None)
    output.close()
    with wave.open(str(path), "rb") as recording:
        assert recording.getnframes() == 137
        assert recording.getnchannels() == 2
        assert recording.getframerate() == 48000
        assert np.max(np.frombuffer(recording.readframes(137), dtype="<i2")) > 8000


def test_recording_overflow_never_silences_audio(tmp_path, monkeypatch):
    streams, errors, warnings = [], [], []
    entered, release = Event(), Event()

    class SlowWriter:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def setnchannels(self, *args):
            pass
        def setsampwidth(self, *args):
            pass
        def setframerate(self, *args):
            pass
        def writeframes(self, data):
            entered.set()
            release.wait(2)

    monkeypatch.setattr(wave, "open", lambda *args: SlowWriter())
    def factory(**kwargs):
        streams.append(Stream(**kwargs))
        return streams[-1]
    output = InferenceAudioOutput(audio_format=AudioFormat(), stream_factory=factory,
        recording_path=tmp_path / "slow.wav", on_error=errors.append, on_warning=warnings.append)
    output.start(lambda frames: np.full((frames, 2), 0.25, dtype=np.float32))
    block = np.zeros((32, 2))
    try:
        streams[-1].callback(block, 32, None, None)
        assert entered.wait(1)
        for _ in range(130):
            streams[-1].callback(block, 32, None, None)
            assert np.allclose(block, 0.25)
        assert not errors
        assert output.snapshot()["recording_dropped_frames"] > 0
        assert warnings
    finally:
        release.set()
        output.close()


def test_failed_recording_writer_does_not_block_close_or_audio(tmp_path, monkeypatch):
    streams, warned = [], Event()
    def broken(*args):
        raise OSError("disk unavailable")
    monkeypatch.setattr(wave, "open", broken)
    def factory(**kwargs):
        streams.append(Stream(**kwargs))
        return streams[-1]
    output = InferenceAudioOutput(audio_format=AudioFormat(), stream_factory=factory,
        recording_path=tmp_path / "failed.wav", on_warning=lambda message: warned.set())
    output.start(lambda frames: np.full((frames, 2), 0.25, dtype=np.float32))
    assert warned.wait(1)
    block = np.zeros((32, 2))
    for _ in range(150):
        streams[-1].callback(block, 32, None, None)
    assert np.allclose(block, 0.25)
    started = monotonic()
    output.close()
    assert monotonic() - started < 1
    assert "disk unavailable" in output.snapshot()["recording_error"]


def test_recording_omits_stopped_callbacks_but_keeps_last_active_block(tmp_path):
    streams = []
    active = [True]
    def factory(**kwargs):
        streams.append(Stream(**kwargs))
        return streams[-1]
    def render(frames):
        active[0] = False
        return np.full((frames, 2), 0.25, dtype=np.float32)
    path = tmp_path / "active-only.wav"
    output = InferenceAudioOutput(audio_format=AudioFormat(), stream_factory=factory, recording_path=path,
                                   active_probe=lambda: active[0])
    output.start(render)
    block = np.empty((32, 2), dtype=np.float32)
    for _ in range(10):
        streams[-1].callback(block, 32, None, None)
    output.close()
    with wave.open(str(path), "rb") as recording:
        assert recording.getnframes() == 32
    assert output.snapshot()["recording_skipped_idle_frames"] == 288
