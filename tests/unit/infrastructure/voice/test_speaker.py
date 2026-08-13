from __future__ import annotations

import numpy as np
import pytest

from streammuse.infrastructure.voice.speaker import SpeakerPlayer
from streammuse.infrastructure.voice.synthesizer import SynthesizedAudio


class CallbackStop(Exception):
    pass


class CallbackAbort(Exception):
    pass


class _Default:
    device = (0, 2)


class FakeSoundDevice:
    CallbackStop = CallbackStop
    CallbackAbort = CallbackAbort
    default = _Default()

    def __init__(self, *, mode: str = "normal") -> None:
        self.mode = mode
        self.streams: list[FakeStream] = []

    def query_devices(self, device=None, kind=None):
        assert kind == "output"
        return {
            "index": 2,
            "name": "Test Speaker",
            "max_output_channels": 2,
            "default_samplerate": 48_000.0,
            "hostapi": 1,
        }

    def check_output_settings(self, **kwargs) -> None:
        return None

    def OutputStream(self, **kwargs):
        stream = FakeStream(self, **kwargs)
        self.streams.append(stream)
        return stream


class FakeStream:
    latency = 0.0

    def __init__(self, owner: FakeSoundDevice, **kwargs) -> None:
        self.owner = owner
        self.callback = kwargs["callback"]
        self.finished_callback = kwargs["finished_callback"]
        self.closed = False
        self.control_exception: BaseException | None = None

    def start(self) -> None:
        if self.owner.mode == "normal":
            self._invoke_callback()
        elif self.owner.mode == "type_error":
            self._invoke_callback(outdata=_BrokenOutput())

    def abort(self) -> None:
        if self.owner.mode == "abort_then_final":
            self._invoke_callback()
        elif self.owner.mode not in {"normal", "type_error"}:
            self.finished_callback()

    def _invoke_callback(self, outdata=None) -> None:
        target = (
            np.empty((16, 1), dtype=np.float32)
            if outdata is None
            else outdata
        )
        try:
            self.callback(
                target,
                16,
                {
                    "currentTime": 10.0,
                    "outputBufferDacTime": 10.01,
                },
                None,
            )
        except (CallbackStop, CallbackAbort) as exc:
            self.control_exception = exc
            self.finished_callback()

    def close(self) -> None:
        self.closed = True


class _BrokenOutput:
    def __setitem__(self, key, value) -> None:
        raise TypeError("bad callback buffer")


class ManualClock:
    def __init__(self) -> None:
        self.value = 20.0

    def __call__(self) -> float:
        self.value += 0.001
        return self.value


def _audio() -> SynthesizedAudio:
    return SynthesizedAudio(np.ones(8, dtype=np.float32), 48_000)


def test_callback_raises_stop_and_finished_callback_marks_drain() -> None:
    sd = FakeSoundDevice()
    player = SpeakerPlayer(sounddevice_module=sd, now=ManualClock())
    player.start()

    playback = player.play(_audio())

    assert isinstance(sd.streams[0].control_exception, CallbackStop)
    assert playback.completed_normally is True
    assert playback.first_dac_sample_offset_ms is not None
    assert (
        playback.playback_drained_offset_ms
        == playback.stream_inactive_offset_ms
    )
    timing = playback.metadata["timing_breakdown"]
    assert timing["schema_version"] == 1
    assert timing["anchors_ms"]["first_callback"] is not None
    assert timing["anchors_ms"]["first_dac_sample"] is not None
    assert timing["durations_ms"]["audio_prepare"] >= 0.0


def test_abort_latch_wins_if_final_callback_runs_after_timeout() -> None:
    sd = FakeSoundDevice(mode="abort_then_final")
    player = SpeakerPlayer(
        sounddevice_module=sd,
        now=ManualClock(),
        wait_margin_s=0.0,
    )
    player.start()

    playback = player.play(_audio())

    assert isinstance(sd.streams[0].control_exception, CallbackStop)
    assert playback.error is not None
    assert playback.completed_normally is False
    assert playback.stream_inactive_offset_ms is not None
    assert playback.playback_drained_offset_ms is None


def test_unknown_callback_exception_is_propagated_unchanged() -> None:
    sd = FakeSoundDevice(mode="type_error")
    player = SpeakerPlayer(sounddevice_module=sd, now=ManualClock())
    player.start()

    with pytest.raises(TypeError, match="bad callback buffer"):
        player.play(_audio())
