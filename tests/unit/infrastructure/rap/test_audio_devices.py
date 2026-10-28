"""Sounddevice selection validates stable descriptors and preserves stopped state."""

from types import SimpleNamespace
import sys

import numpy as np
import pytest

from streammuse.domain.rap import AudioFormat, AudioPlaybackNoticeKind, PcmAudio, PlaybackState, PreparedRapBar
from streammuse.infrastructure.rap import audio_output


class Stream:
    def __init__(self, *, callback, device, **kwargs):
        self.callback = callback
        self.device = device
        self.closed = False
        self.started = False

    def start(self):
        self.started = True

    def stop(self):
        self.started = False

    def close(self):
        self.closed = True

    @property
    def active(self):
        return self.started


class Backend:
    def __init__(self):
        self.devices = [
            {"name": "Mic", "hostapi": 0, "max_output_channels": 0, "default_samplerate": 48_000.0},
            {"name": "Speakers", "hostapi": 0, "max_output_channels": 2, "default_samplerate": 48_000.0},
            {"name": "Headphones", "hostapi": 0, "max_output_channels": 2, "default_samplerate": 44_100.0},
            {"name": "Unsupported", "hostapi": 0, "max_output_channels": 1, "default_samplerate": 22_050.0},
        ]
        self.default = SimpleNamespace(device=(0, 1))
        self.streams = []
        self.fail_open = None

    def query_devices(self):
        return self.devices

    def query_hostapis(self):
        return [{"name": "Core Audio"}]

    def check_output_settings(self, *, device, samplerate, channels, dtype):
        assert (samplerate, channels, dtype) == (48_000, 2, "float32")
        if self.devices[device]["max_output_channels"] < channels:
            raise RuntimeError("unsupported channel count")

    def OutputStream(self, **kwargs):
        if self.fail_open is not None and kwargs["device"] == self.fail_open:
            raise RuntimeError("device busy")
        stream = Stream(**kwargs)
        self.streams.append(stream)
        return stream


@pytest.fixture
def backend(monkeypatch):
    backend = Backend()
    monkeypatch.setitem(sys.modules, "sounddevice", backend)
    return backend


def bar(number):
    audio = PcmAudio(AudioFormat(), 4, np.full((4, 2), 0.25, dtype=np.float32).tobytes())
    return PreparedRapBar(number, "one", "test", None, (), audio, (), (), 0.0)


def test_device_enumeration_filters_inputs_and_reports_format_failures(backend):
    devices = audio_output.list_output_devices()

    assert [item.name for item in devices] == ["Speakers", "Headphones", "Unsupported"]
    assert devices[0].host_api == "Core Audio" and devices[0].is_default
    assert devices[1].available and devices[1].default_sample_rate_hz == 44_100
    assert not devices[2].available and "channel" in devices[2].error
    assert devices[0].to_payload()["name"] == "Speakers"


def test_device_descriptor_is_resolved_by_host_and_name_after_index_changes(backend):
    selected = audio_output.list_output_devices()[1]
    backend.devices[1], backend.devices[2] = backend.devices[2], backend.devices[1]

    resolved = audio_output.validate_output_device(selected)

    assert resolved.index == 1 and resolved.name == "Headphones"
    assert audio_output.validate_output_device(selected.to_payload()) == resolved
    with pytest.raises(ValueError, match="unavailable|missing"):
        audio_output.validate_output_device({"host_api": "Core Audio", "name": "Unplugged"})


def test_device_switch_is_stopped_only_and_preserves_resume_position_and_queue(backend):
    sink = audio_output.SoundDeviceAudioSink(audio_format=AudioFormat(), callback_stop_factory=lambda: StopIteration())
    sink.enqueue(bar(0))
    sink.enqueue(bar(1))
    assert sink.start()
    stream = backend.streams[-1]
    stream.callback(np.zeros((2, 2), dtype=np.float32), 2, None, 0)
    with pytest.raises(RuntimeError, match="stopped"):
        sink.set_output_device({"host_api": "Core Audio", "name": "Headphones"})
    sink.request_stop_after_bar()
    with pytest.raises(StopIteration):
        stream.callback(np.zeros((2, 2), dtype=np.float32), 2, None, 0)
    stopped = sink.snapshot()
    assert stopped.state is PlaybackState.STOPPED and stopped.absolute_frame == 4

    selected = sink.set_output_device({"host_api": "Core Audio", "name": "Headphones"})
    assert selected.name == "Headphones"
    assert stream.closed
    assert sink.snapshot() == stopped
    assert sink.start()
    block = np.zeros((4, 2), dtype=np.float32)
    backend.streams[-1].callback(block, 4, None, 0)
    assert np.all(block == 0.25)
    assert sink.snapshot().last_completed_bar == 1
    assert sink.snapshot().absolute_frame == 8
    sink.close()


def test_failed_device_switch_preserves_selection_and_reports_notice(backend):
    sink = audio_output.SoundDeviceAudioSink(audio_format=AudioFormat())
    selected = sink.set_output_device({"host_api": "Core Audio", "name": "Speakers"})
    stream = backend.streams[-1]
    backend.fail_open = 2

    with pytest.raises(RuntimeError, match="busy"):
        sink.set_output_device({"host_api": "Core Audio", "name": "Headphones"})
    assert sink.output_device == selected
    assert not stream.closed
    assert sink.snapshot().state is PlaybackState.STOPPED
    assert sink.drain_notices()[-1].kind is AudioPlaybackNoticeKind.DEVICE_FAILED
    sink.close()


def test_selected_device_disappearing_before_start_is_not_audible_success(backend):
    sink = audio_output.SoundDeviceAudioSink(audio_format=AudioFormat())
    sink.set_output_device({"host_api": "Core Audio", "name": "Headphones"})
    backend.devices.pop(2)

    assert sink.start() is False
    assert sink.snapshot().state is PlaybackState.STOPPED
    assert sink.drain_notices()[-1].kind is AudioPlaybackNoticeKind.DEVICE_FAILED
    assert sink.output_device_status()["audible"] is False
    sink.close()


def test_nondevice_outputs_are_explicitly_not_audible():
    sink = audio_output.TimedAudioSink(audio_format=AudioFormat())
    assert sink.output_device_status() == {"backend": "timed", "audible": False, "supports_device_selection": False}
    sink.close()


def test_unexpected_device_stream_stop_is_visible_and_retains_unfinished_position(backend):
    sink = audio_output.SoundDeviceAudioSink(audio_format=AudioFormat())
    sink.enqueue(bar(0))
    assert sink.start()
    stream = backend.streams[-1]
    stream.callback(np.zeros((2, 2), dtype=np.float32), 2, None, 0)
    stream.started = False

    snapshot = sink.snapshot()

    assert snapshot.state is PlaybackState.STOPPED
    assert snapshot.current_bar == 0 and snapshot.frame_in_bar == 2
    notice = sink.drain_notices()[-1]
    assert notice.kind is AudioPlaybackNoticeKind.DEVICE_FAILED
    assert "unexpectedly" in notice.message
    assert sink.output_device_status()["audible"] is False
    assert sink.output_device_status()["error"]
    sink.close()


def test_failed_restart_does_not_discard_an_unfinished_bar_after_device_failure(backend):
    sink = audio_output.SoundDeviceAudioSink(audio_format=AudioFormat())
    sink.enqueue(bar(0))
    assert sink.start()
    stream = backend.streams[-1]
    stream.callback(np.zeros((2, 2), dtype=np.float32), 2, None, 0)
    stream.started = False
    before = sink.snapshot()

    def failed_start():
        raise RuntimeError("device disconnected")

    stream.start = failed_start
    assert sink.start() is False
    after = sink.snapshot()
    assert (after.current_bar, after.frame_in_bar, after.absolute_frame) == (before.current_bar, before.frame_in_bar, before.absolute_frame)
    assert after.current_bar == 0 and after.frame_in_bar == 2
    sink.close()


def test_composite_keeps_device_selection_available_during_recoverable_fallback(backend, tmp_path):
    backend.fail_open = 1
    live = audio_output.SoundDeviceAudioSink(audio_format=AudioFormat(), device=1)
    sink = audio_output.CompositeAudioSink(
        live, audio_output.Float32WavAudioSink(tmp_path / "fallback.wav", AudioFormat()),
        fallback=audio_output.NullAudioSink(audio_format=AudioFormat()),
    )
    try:
        assert sink.start()
        assert sink.list_output_devices()[0].name == "Speakers"
        selected = sink.validate_output_device({"host_api": "Core Audio", "name": "Headphones"})
        with pytest.raises(RuntimeError, match="stopped"):
            sink.set_output_device(selected)
        sink.request_stop_after_bar()
        assert sink.set_output_device(selected).name == "Headphones"
        assert sink.output_device_status()["recovery_requires_reset"] is True
        assert sink.output_device_status()["audible"] is False
        sink.reset()
        assert sink.start()
        assert sink.output_device_status()["backend"] == "sounddevice"
        assert sink.output_device_status()["audible"] is True
        assert backend.streams[-1].device == 2
    finally:
        sink.close()
