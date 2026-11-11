"""Sounddevice selection validates stable descriptors and preserves stopped state."""

from types import SimpleNamespace
from threading import Event, Thread
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
        self.discovery_calls = 0
        self.probe_calls = 0

    def query_devices(self):
        self.discovery_calls += 1
        return self.devices

    def query_hostapis(self):
        return [{"name": "Core Audio"}]

    def check_output_settings(self, *, device, samplerate, channels, dtype):
        self.probe_calls += 1
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


@pytest.mark.parametrize("stop_requested", [False, True])
@pytest.mark.parametrize("operation", ["list", "validate"])
def test_active_device_queries_use_cached_format_results(backend, stop_requested, operation):
    sink = audio_output.SoundDeviceAudioSink(audio_format=AudioFormat())
    cached = sink.list_output_devices()
    sink.enqueue(bar(0))
    assert sink.start()
    backend.streams[-1].callback(np.zeros((1, 2), np.float32), 1, None, 0)
    if stop_requested:
        assert sink.request_stop_after_bar().state is PlaybackState.STOP_REQUESTED
    calls = backend.discovery_calls, backend.probe_calls
    backend.devices.clear()
    try:
        if operation == "list":
            assert sink.list_output_devices() == cached
        else:
            assert sink.validate_output_device(cached[1].to_payload()) == cached[1]
            with pytest.raises(ValueError, match="unavailable"):
                sink.validate_output_device(cached[2])
        assert (backend.discovery_calls, backend.probe_calls) == calls
    finally:
        sink.close()


def test_running_queries_without_a_stopped_snapshot_do_not_probe(backend):
    sink = audio_output.SoundDeviceAudioSink(
        audio_format=AudioFormat(), stream_factory=lambda **kwargs: Stream(device=None, **kwargs),
    )
    assert sink.start()
    calls = backend.discovery_calls, backend.probe_calls
    try:
        sink.list_output_devices()
        with pytest.raises(ValueError, match="missing|unavailable"):
            sink.validate_output_device("Unplugged")
        assert (backend.discovery_calls, backend.probe_calls) == calls
    finally:
        sink.close()


@pytest.mark.parametrize("device, expected", [(None, "Speakers"), ("Headphones", "Headphones"), (2, "Headphones")])
def test_native_start_primes_device_cache_without_a_prior_settings_query(backend, device, expected):
    sink = audio_output.SoundDeviceAudioSink(audio_format=AudioFormat(), device=device)
    try:
        assert sink.start()
        calls = backend.discovery_calls, backend.probe_calls
        assert calls == (1, 3)
        assert sink.output_device.name == expected
        assert backend.streams[-1].device == sink.output_device.index
        assert [item.name for item in sink.list_output_devices()] == ["Speakers", "Headphones", "Unsupported"]
        assert sink.validate_output_device(expected).name == expected
        assert (backend.discovery_calls, backend.probe_calls) == calls
    finally:
        sink.close()


def test_injected_start_does_not_discover_native_devices(backend):
    sink = audio_output.SoundDeviceAudioSink(
        audio_format=AudioFormat(), stream_factory=lambda **kwargs: Stream(device=None, **kwargs),
    )
    try:
        assert sink.start()
        assert (backend.discovery_calls, backend.probe_calls) == (0, 0)
    finally:
        sink.close()


def test_stopped_device_listing_refreshes_after_playback(backend):
    sink = audio_output.SoundDeviceAudioSink(audio_format=AudioFormat())
    before = sink.list_output_devices()
    assert sink.start()
    sink.request_stop_after_bar()
    backend.devices[2]["name"] = "Replacement"
    calls = backend.discovery_calls, backend.probe_calls
    try:
        after = sink.list_output_devices()
        assert [item.name for item in before] == ["Speakers", "Headphones", "Unsupported"]
        assert [item.name for item in after] == ["Speakers", "Replacement", "Unsupported"]
        assert (backend.discovery_calls, backend.probe_calls) == (calls[0] + 1, calls[1] + 3)
    finally:
        sink.close()


@pytest.mark.parametrize("scope", ["global", "same_sink", "different_sinks"])
def test_concurrent_stopped_discovery_and_format_probes_are_serialized(backend, monkeypatch, scope):
    entered, release, second_attempted, overlapped = Event(), Event(), Event(), Event()
    errors = []
    original_probe = backend.check_output_settings

    def probe(**kwargs):
        if not entered.is_set():
            entered.set()
            assert release.wait(2)
        elif not release.is_set():
            overlapped.set()
        original_probe(**kwargs)

    monkeypatch.setattr(backend, "check_output_settings", probe)
    first_sink = audio_output.SoundDeviceAudioSink(audio_format=AudioFormat())
    second_sink = first_sink if scope == "same_sink" else audio_output.SoundDeviceAudioSink(audio_format=AudioFormat())
    first = audio_output.list_output_devices if scope == "global" else first_sink.list_output_devices
    second = audio_output.list_output_devices if scope == "global" else second_sink.list_output_devices

    def run(operation, attempted=None):
        if attempted is not None:
            attempted.set()
        try:
            operation()
        except Exception as error:
            errors.append(error)

    threads = [Thread(target=run, args=(first,)), Thread(target=run, args=(second, second_attempted))]
    threads[0].start()
    try:
        assert entered.wait(1)
        threads[1].start()
        assert second_attempted.wait(1)
        assert not overlapped.wait(0.05)
        assert backend.discovery_calls == 1
    finally:
        release.set()
        for thread in threads:
            if thread.ident is not None:
                thread.join(2)
        first_sink.close()
        second_sink.close()
    assert all(not thread.is_alive() for thread in threads)
    assert errors == []
    assert backend.discovery_calls == 2 and backend.probe_calls == 6


@pytest.mark.parametrize("operation", ["open", "start", "stop", "close", "reset", "discard", "restart", "switch"])
def test_native_lifecycle_excludes_device_discovery(backend, monkeypatch, operation):
    sink = audio_output.SoundDeviceAudioSink(audio_format=AudioFormat())
    if operation not in ("open", "start"):
        assert sink.start()
    if operation in ("restart", "switch"):
        sink.request_stop_after_bar()
    entered, release, attempted, discovered = Event(), Event(), Event(), Event()
    errors = []
    method = {"open": "OutputStream", "start": "start", "close": "close"}.get(operation, "stop")
    owner = backend if operation == "open" else Stream
    original = getattr(owner, method)

    def blocked(*args, **kwargs):
        entered.set()
        assert release.wait(2)
        return original(*args, **kwargs)

    monkeypatch.setattr(owner, method, blocked)
    actions = {
        "open": sink.start, "start": sink.start, "stop": sink.request_stop_after_bar,
        "close": sink.close, "reset": sink.reset, "discard": sink.discard_pending,
        "restart": sink.start,
        "switch": lambda: sink.set_output_device("Headphones"),
    }

    def run(action):
        try:
            action()
        except Exception as error:
            errors.append(error)

    lifecycle = Thread(target=run, args=(actions[operation],))
    lifecycle.start()
    discovery = None
    try:
        assert entered.wait(1)
        original_query = backend.query_devices

        def query():
            discovered.set()
            return original_query()

        monkeypatch.setattr(backend, "query_devices", query)

        def list_devices():
            attempted.set()
            audio_output.list_output_devices()

        discovery = Thread(target=run, args=(list_devices,))
        discovery.start()
        assert attempted.wait(1)
        assert not discovered.wait(0.05)
        if operation not in ("open", "switch"):
            # Native stop/close/start must leave callback state accessible.
            observed = Event()
            reader = Thread(target=lambda: (sink.snapshot(), observed.set()))
            reader.start()
            assert observed.wait(1)
            reader.join(1)
    finally:
        release.set()
        lifecycle.join(2)
        if discovery is not None:
            discovery.join(2)
        sink.close()
    assert not lifecycle.is_alive() and not discovery.is_alive()
    assert errors == []


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
