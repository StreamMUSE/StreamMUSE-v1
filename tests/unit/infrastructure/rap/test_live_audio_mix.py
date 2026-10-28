"""Sample-level live mix and completed-callback recording contracts."""

from dataclasses import replace
from threading import Event, Thread

import numpy as np
import pytest

from streammuse.domain.rap import AudioFormat, PcmAudio, PlaybackState, PreparedRapBar
from streammuse.domain.rap.audio import RapMixConfig
from streammuse.domain.rap.interaction import CaptureWindow, TapRecord, quantize_capture
from streammuse.domain.timing import Tempo
from streammuse.infrastructure.rap.audio_output import (
    CompositeAudioSink,
    Float32WavAudioSink,
    NullAudioSink,
    SoundDeviceAudioSink,
)
from streammuse.infrastructure.rap.live_mix import CallbackAudio
from streammuse.infrastructure.rap.drums import ProceduralBoomBapRenderer
from streammuse.infrastructure.rap.flow_echo import FlowEchoRenderer
from streammuse.infrastructure.rap.scenarios import default_scenario
from streammuse.infrastructure.rap.templates import BUILTIN_TEMPLATES


FORMAT = AudioFormat(sample_rate_hz=1_000, channels=2)
GAINS = {"drum_gain": 1.0, "vocal_gain": 1.0, "chord_gain": 1.0}


class CallbackStopped(Exception):
    pass


class Stream:
    def __init__(self, *, callback, **kwargs):
        self.callback = callback
        self.active = False

    def start(self):
        self.active = True

    def stop(self):
        self.active = False

    def close(self):
        self.active = False


def pcm(value, frames=100):
    return PcmAudio(FORMAT, frames, np.full((frames, 2), value, dtype=np.float32).tobytes())


def bar(number=0, frames=100, config=None):
    config = config or RapMixConfig(**GAINS, chord_loop="minor_cycle")
    # Independent, hand-computable mix: drums .2, vocals .3, chords .1.
    value = (.2 * config.drum_gain + .3 * config.vocal_gain +
             (.1 * config.chord_gain if config.chord_loop != "off" else 0)) * config.master_gain
    return PreparedRapBar(
        number, "test", "test", None, (), pcm(min(value, config.peak_limit), frames), (), (), 0.0,
        drum_stem=pcm(.2, frames), vocal_stem=pcm(.3, frames), chord_stem=pcm(.1, frames),
        mix_config=config,
    )


@pytest.fixture
def sink():
    result = SoundDeviceAudioSink(
        audio_format=FORMAT, stream_factory=Stream, callback_stop_factory=CallbackStopped,
    )
    yield result
    result.close()


def render(sink, frames):
    output = np.empty((frames, 2), dtype=np.float32)
    try:
        sink._callback(output, frames, None, 0)
    except CallbackStopped:
        pass
    return output


@pytest.fixture
def echo_sink():
    result = SoundDeviceAudioSink(audio_format=AudioFormat(), stream_factory=Stream)
    yield result
    result.close()


def flow_echo_bar(*, chord_loop="off", idle=False):
    template = BUILTIN_TEMPLATES.get(default_scenario().segments[0].template_id)
    renderer = FlowEchoRenderer(drums=ProceduralBoomBapRenderer(seed=2), idle_template=template)
    renderer.set_mix_config(RapMixConfig(drum_gain=0, vocal_gain=0, chord_gain=0, chord_loop=chord_loop))
    captured = quantize_capture(
        CaptureWindow(0, 0.0, 4.0), (TapRecord(.75, .8), TapRecord(2.75, .8)), template_id="live-echo",
    ).template
    return renderer.render(bar=0, tempo=Tempo(60, 4, 4), audio_format=AudioFormat(),
                           template=None if idle else captured)


def test_real_flow_echo_keeps_captured_cues_when_live_vocals_are_muted(echo_sink):
    sink = echo_sink
    echo = flow_echo_bar()
    sink.enqueue(echo)
    sink.start()
    before = render(sink, 12000)

    sink.set_live_gains(drum_gain=0, vocal_gain=0, chord_gain=0)
    played = np.concatenate([before, render(sink, echo.audio.frame_count - 12000)])

    assert played[36000, 0] == pytest.approx(.35)
    assert played[132000, 0] == pytest.approx(.35)
    assert played == pytest.approx(np.frombuffer(echo.audio.data, np.float32).reshape(-1, 2))


@pytest.mark.parametrize("idle", [False, True])
def test_real_echo_and_idle_evidence_does_not_claim_vocal_gain_controls_auxiliary(echo_sink, idle):
    sink = echo_sink
    sink.enqueue(flow_echo_bar(chord_loop="minor_cycle", idle=idle))
    sink.start()
    render(sink, 1)
    initial = sink.output_device_status()["live_mix"]
    assert set(initial["supported_gains"]) == {"drum_gain", "chord_gain"}
    assert initial["applied"]["vocal_gain"] is None

    accepted = sink.set_live_gains(drum_gain=0, vocal_gain=0, chord_gain=0)
    render(sink, 480)
    status = sink.output_device_status()["live_mix"]
    assert status["requested"]["vocal_gain"] == 0
    assert status["applied"]["vocal_gain"] is None
    assert set(status["supported_gains"]) == {"drum_gain", "chord_gain"}
    assert "percussion" in status["limitation"]
    assert status["applied_revision"] == accepted["revision"]


@pytest.mark.parametrize("gain,stem", [("drum_gain", "drum_stem"), ("chord_gain", "chord_stem")])
def test_real_flow_echo_keeps_immediate_drum_and_chord_gains(echo_sink, gain, stem):
    sink = echo_sink
    echo = flow_echo_bar(chord_loop="minor_cycle")
    cue = np.frombuffer(echo.vocal_stem.data, np.float32).reshape(-1, 2)
    controlled = np.frombuffer(getattr(echo, stem).data, np.float32).reshape(-1, 2)
    sink.enqueue(echo)
    sink.start()
    render(sink, 12000)
    sink.set_live_gains(**{name: float(name == gain) for name in GAINS})

    increased = render(sink, 36000)

    envelope = np.concatenate([np.arange(1, 481) / 480, np.ones(35520)])
    expected = cue[12000:48000] + controlled[12000:48000] * envelope[:, None]
    assert increased == pytest.approx(np.clip(expected, -.95, .95), abs=1e-7)
    assert np.max(np.abs(increased - cue[12000:48000])) > .01
    assert sink.snapshot().current_bar == 0 and sink.snapshot().frame_in_bar == 48000

    sink.set_live_gains(drum_gain=0, vocal_gain=0, chord_gain=0)
    decreased = render(sink, 86400)
    assert decreased[480:] == pytest.approx(cue[48480:134400], abs=1e-7)
    assert decreased[84000, 0] == pytest.approx(.35)


def test_auxiliary_echo_does_not_override_requested_vocal_gain_for_following_espeak(echo_sink):
    sink = echo_sink
    echo = flow_echo_bar()
    speech = bar(1, frames=960)
    speech = replace(speech, source="lyric_stream_espeak", **{
        name: replace(getattr(speech, name), format=AudioFormat())
        for name in ("audio", "drum_stem", "vocal_stem", "chord_stem")
    })
    sink.enqueue(echo)
    sink.enqueue(speech)
    sink.start()
    sink.set_live_gains(drum_gain=0, vocal_gain=0, chord_gain=0)
    played = render(sink, echo.audio.frame_count)
    assert played[36000, 0] == pytest.approx(.35)

    assert not render(sink, 480).any()
    evidence = sink.output_device_status()["live_mix"]
    assert "vocal_gain" in evidence["supported_gains"]
    assert evidence["applied"]["vocal_gain"] == 0
    sink.set_live_gains(drum_gain=0, vocal_gain=1, chord_gain=0)
    assert render(sink, 480)[:, 0] == pytest.approx(.3 * np.arange(1, 481) / 480)


@pytest.mark.parametrize("name,contribution", [("drum_gain", .2), ("vocal_gain", .3), ("chord_gain", .1)])
def test_same_bar_gain_changes_over_ten_ms_without_changing_position(sink, name, contribution):
    prepared = bar()
    sink.enqueue(prepared)
    sink.start()
    assert render(sink, 7) == pytest.approx(np.full((7, 2), .6))
    before = sink.snapshot()
    requested = {**GAINS, name: 0.0}

    accepted = sink.set_live_gains(**requested)

    assert sink.snapshot() == before
    assert accepted["requested"] == requested
    assert accepted["applied_revision"] != accepted["revision"]
    ramp = np.concatenate([render(sink, 3), render(sink, 7)])[:, 0]
    assert ramp == pytest.approx(.6 - contribution * np.arange(1, 11) / 10)
    assert render(sink, 2) == pytest.approx(np.full((2, 2), .6 - contribution))
    assert sink.snapshot().current_bar == 0
    assert sink.snapshot().frame_in_bar == sink.snapshot().absolute_frame == 19
    evidence = sink.output_device_status()["live_mix"]
    assert evidence["requested"] == evidence["applied"] == requested
    assert evidence["applied_revision"] == accepted["revision"]
    assert evidence["bar"] == 0 and evidence["absolute_frame"] == 19
    assert evidence["observation"] == "callback_pcm_not_acoustic_verification"
    assert prepared == bar()


@pytest.mark.parametrize("name,contribution", [("drum_gain", .2), ("vocal_gain", .3), ("chord_gain", .1)])
def test_zero_gain_is_reversible_from_retained_raw_stems(sink, name, contribution):
    muted = {key: 0.0 for key in GAINS}
    sink.enqueue(bar(config=RapMixConfig(**muted, chord_loop="minor_cycle")))
    sink.start()
    assert not render(sink, 1).any()

    sink.set_live_gains(**{**muted, name: 1.0})

    assert render(sink, 10)[:, 0] == pytest.approx(contribution * np.arange(1, 11) / 10)
    assert render(sink, 1)[0, 0] == pytest.approx(contribution)


def test_new_revision_retargets_from_last_sample_and_duplicate_does_not_restart_ramp(sink):
    sink.enqueue(bar())
    sink.start()
    render(sink, 1)
    muted = {key: 0.0 for key in GAINS}
    first = sink.set_live_gains(**muted)
    assert render(sink, 4)[-1, 0] == pytest.approx(.36)
    assert sink.set_live_gains(**muted)["revision"] == first["revision"]
    assert render(sink, 1)[0, 0] == pytest.approx(.3)

    latest = sink.set_live_gains(**GAINS)

    assert latest["revision"] > first["revision"]
    assert render(sink, 10)[:, 0] == pytest.approx(np.arange(.33, .601, .03))
    assert sink.output_device_status()["live_mix"]["applied_revision"] == latest["revision"]


def test_live_ramp_crosses_bar_boundaries_and_stop_keeps_queued_audio(sink):
    sink.enqueue(bar(0, 6))
    sink.enqueue(bar(1, 20))
    sink.enqueue(bar(2, 20))
    sink.start()
    render(sink, 1)
    sink.set_live_gains(drum_gain=0, vocal_gain=0, chord_gain=0)
    assert render(sink, 10)[:, 0] == pytest.approx(np.arange(.54, -.001, -.06), abs=1e-7)
    sink.request_stop_after_bar()
    assert not render(sink, 30).any()
    stopped = sink.snapshot()
    assert stopped.state == PlaybackState.STOPPED
    assert stopped.absolute_frame == 26 and stopped.queue_depth == 1
    sink.start()
    assert not render(sink, 10).any()
    assert sink.snapshot().current_bar == 2 and sink.snapshot().absolute_frame == 36


@pytest.mark.parametrize("reset_method", ["reset", "discard_pending"])
def test_live_gain_request_survives_new_playback_epoch_without_stale_evidence(sink, reset_method):
    sink.enqueue(bar())
    sink.start()
    sink.set_live_gains(drum_gain=0, vocal_gain=0, chord_gain=0)
    render(sink, 10)
    getattr(sink, reset_method)()
    status = sink.output_device_status()["live_mix"]
    assert status["requested"] == {key: 0.0 for key in GAINS}
    assert status["applied"] is None and status["applied_revision"] is None
    sink.enqueue(bar(1))
    sink.start()
    render(sink, 10)
    assert not render(sink, 1).any()


def test_master_and_peak_limit_remain_effective_during_live_mix(sink):
    sink.enqueue(bar(config=RapMixConfig(**GAINS, chord_loop="minor_cycle", master_gain=.5, peak_limit=.4)))
    sink.start()
    sink.set_live_gains(drum_gain=2, vocal_gain=2, chord_gain=2)
    output = render(sink, 30)
    assert np.isfinite(output).all()
    assert np.max(np.abs(output)) <= np.float32(.4)
    assert output[0, 0] == pytest.approx(.33)
    assert output[-1, 0] == pytest.approx(.4)


@pytest.mark.parametrize("value", [-1, 2.01, float("nan"), float("inf"), True, "1"])
def test_invalid_live_gain_is_rejected_atomically(sink, value):
    before = sink.set_live_gains(**GAINS)
    with pytest.raises(ValueError, match="gain"):
        sink.set_live_gains(drum_gain=.5, vocal_gain=value, chord_gain=.5)
    assert sink.output_device_status()["live_mix"] == before


def test_legacy_bar_remains_audible_without_claiming_live_mix_support(sink):
    legacy = replace(bar(), drum_stem=None, vocal_stem=None, chord_stem=None, mix_config=None)
    sink.enqueue(legacy)
    sink.start()
    sink.set_live_gains(drum_gain=0, vocal_gain=0, chord_gain=0)
    assert render(sink, 20) == pytest.approx(np.full((20, 2), .6))
    status = sink.output_device_status()["live_mix"]
    assert status["supported"] is False
    assert status["applied"] is None and status["applied_revision"] is None
    assert "stem" in status["limitation"]


def test_chords_off_are_not_fabricated_by_live_gain(sink):
    sink.enqueue(bar(config=RapMixConfig(**GAINS, chord_loop="off")))
    sink.start()
    sink.set_live_gains(drum_gain=0, vocal_gain=0, chord_gain=2)
    render(sink, 10)
    assert not render(sink, 1).any()
    status = sink.output_device_status()["live_mix"]
    assert status["applied"]["chord_gain"] is None
    assert set(status["supported_gains"]) == {"drum_gain", "vocal_gain"}
    assert "chord" in status["limitation"]


def test_composite_wav_is_exact_completed_callback_pcm_with_live_gain_changes(sink, tmp_path):
    path = tmp_path / "live.wav"
    composite = CompositeAudioSink(sink, Float32WavAudioSink(path, FORMAT))
    composite.enqueue(bar(0, 25))
    composite.enqueue(bar(1, 25))
    composite.start()
    played = [render(sink, 4)]
    composite.set_live_gains(drum_gain=0, vocal_gain=.5, chord_gain=0)
    played.append(render(sink, 6))
    composite.set_live_gains(**GAINS)
    composite.request_stop_after_bar()
    played.append(render(sink, 20)[:15])
    composite.close()

    assert path.read_bytes()[44:] == np.concatenate(played).tobytes()
    recording = composite.output_device_status()["recording"]
    assert recording["source"] == "completed_callback_pcm"
    assert recording["exact_completed_bars"] is True
    assert "incomplete" in recording["limitation"]


def test_composite_nondevice_backend_does_not_claim_live_gain_support(tmp_path):
    composite = CompositeAudioSink(NullAudioSink(audio_format=FORMAT), Float32WavAudioSink(tmp_path / "null.wav", FORMAT))
    try:
        result = composite.set_live_gains(**GAINS)
        status = composite.output_device_status()
        assert status["live_mix"] == result
        assert result["requested"] == GAINS and result["applied"] is None
        assert result["applied_revision"] is None
        assert not status["supports_live_gains"]
        assert status["live_mix"]["supported"] is False
        assert status["recording"]["source"] == "prepared_bar_pcm"
        assert status["recording"]["exact_completed_bars"] is False
    finally:
        composite.close()


def test_device_reports_live_gain_capability_before_first_callback(sink):
    status = sink.output_device_status()
    assert status["supports_live_gains"] is True
    assert status["live_mix"]["applied"] is None


def test_gain_requested_in_nonaudible_fallback_is_retained_for_device_recovery(tmp_path):
    attempts = 0

    def factory(**kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("device missing")
        return Stream(**kwargs)

    live = SoundDeviceAudioSink(audio_format=FORMAT, stream_factory=factory)
    composite = CompositeAudioSink(live, Float32WavAudioSink(tmp_path / "recovery.wav", FORMAT),
                                   fallback=NullAudioSink(audio_format=FORMAT))
    try:
        assert composite.start()
        accepted = composite.set_live_gains(drum_gain=0, vocal_gain=0, chord_gain=0)
        assert not accepted["supported"] and accepted["applied_revision"] is None
        assert composite.output_device_status()["live_mix"] == accepted
        composite.reset()
        composite.enqueue(bar())
        assert composite.start()
        render(live, 10)
        assert not render(live, 1).any()
        assert composite.output_device_status()["live_mix"]["applied_revision"] == accepted["revision"]
    finally:
        composite.close()


def test_first_live_change_smoothly_leaves_pre_rendered_peak_normalization(sink):
    prepared = bar()
    # A transient elsewhere in the bar can have attenuated the entire old mix.
    vocals = np.full((100, 2), .3, np.float32)
    vocals[99] = 1.6
    old_mix = (vocals + np.float32(.3)) * np.float32(.5)
    prepared = replace(prepared, vocal_stem=PcmAudio(FORMAT, 100, vocals.tobytes()),
                       audio=PcmAudio(FORMAT, 100, old_mix.tobytes()))
    sink.enqueue(prepared)
    sink.start()
    assert render(sink, 1)[0, 0] == pytest.approx(.3)

    sink.set_live_gains(drum_gain=0, vocal_gain=1, chord_gain=1)

    # Smoothly leave .3 for the new .4, without jumping to raw .58 first.
    assert render(sink, 10)[:, 0] == pytest.approx(np.arange(.31, .401, .01))


def test_same_prepared_bar_can_be_queued_more_than_once(sink):
    prepared = bar(frames=10)
    sink.enqueue(prepared)
    sink.enqueue(prepared)
    sink.start()
    sink.set_live_gains(drum_gain=0, vocal_gain=0, chord_gain=0)

    output = render(sink, 20)

    assert not output[10:].any()
    assert sink.snapshot().absolute_frame == 20


def test_zero_frame_bar_does_not_crash_or_advance_gain_evidence(sink):
    sink.enqueue(bar(frames=0))
    sink.enqueue(bar(1, frames=20))
    sink.start()
    sink.set_live_gains(drum_gain=0, vocal_gain=0, chord_gain=0)
    assert render(sink, 10)[:, 0] == pytest.approx(np.arange(.54, -.001, -.06), abs=1e-7)
    assert sink.output_device_status()["live_mix"]["bar"] == 1


def test_sounddevice_fallback_also_records_exact_live_callback_output(tmp_path):
    def unavailable(**kwargs):
        raise RuntimeError("device unavailable")

    first = SoundDeviceAudioSink(audio_format=FORMAT, stream_factory=unavailable)
    fallback = SoundDeviceAudioSink(audio_format=FORMAT, stream_factory=Stream)
    path = tmp_path / "fallback-device.wav"
    composite = CompositeAudioSink(first, Float32WavAudioSink(path, FORMAT), fallback=fallback)
    try:
        composite.enqueue(bar(frames=20))
        composite.set_live_gains(drum_gain=0, vocal_gain=0, chord_gain=0)
        assert composite.start()
        output = render(fallback, 20)
        assert not output[10:].any()
        composite.drain_notices()
        assert path.read_bytes()[44:] == output.tobytes()
    finally:
        composite.close()


def test_large_callback_uses_bounded_mixing_blocks_and_sample_rate_based_smoothing(monkeypatch):
    audio_format = AudioFormat(sample_rate_hz=48_000, channels=2)
    prepared = bar(frames=5000)
    prepared = replace(prepared, **{
        name: replace(getattr(prepared, name), format=audio_format)
        for name in ("audio", "drum_stem", "vocal_stem", "chord_stem")
    })
    counts = []
    original = CallbackAudio.mix

    def observed(self, bar, start, count, *args):
        counts.append(count)
        return original(self, bar, start, count, *args)

    monkeypatch.setattr(CallbackAudio, "mix", observed)
    sink = SoundDeviceAudioSink(audio_format=audio_format, stream_factory=Stream)
    try:
        sink.enqueue(prepared)
        sink.start()
        sink.set_live_gains(drum_gain=0, vocal_gain=0, chord_gain=0)
        output = render(sink, 5000)
        assert output[:480, 0] == pytest.approx(.6 * (1 - np.arange(1, 481) / 480), abs=1e-7)
        assert not output[480:].any()
        assert sum(counts) == 5000 and max(counts) <= 1024
    finally:
        sink.close()


@pytest.mark.parametrize("transition", ["update", "reset"])
def test_inflight_callback_cannot_claim_newer_gain_revision_or_reset_epoch(sink, transition):
    entered, release = Event(), Event()

    class PausedOutput:
        def fill(self, value):
            pass

        def __setitem__(self, key, value):
            entered.set()
            assert release.wait(2)

    sink.enqueue(bar())
    sink.start()
    first = sink.set_live_gains(drum_gain=0, vocal_gain=0, chord_gain=0)
    worker = Thread(target=sink._callback, args=(PausedOutput(), 10, None, 0))
    worker.start()
    try:
        assert entered.wait(2)
        if transition == "update":
            latest = sink.set_live_gains(**GAINS)
        else:
            sink.reset()
    finally:
        release.set()
        worker.join(2)
    assert not worker.is_alive()
    status = sink.output_device_status()["live_mix"]
    if transition == "update":
        assert status["revision"] == latest["revision"]
        assert status["applied_revision"] == first["revision"]
        assert render(sink, 10)[:, 0] == pytest.approx(np.arange(.06, .601, .06))
    else:
        assert status["applied"] is None and status["bar"] is None
        assert sink.snapshot().absolute_frame == 0


@pytest.mark.parametrize("missing", ["vocal_stem", "drum_stem", "chord_stem"])
def test_partial_stems_keep_original_pcm_with_explicit_limitation(sink, missing):
    sink.enqueue(replace(bar(), **{missing: None}))
    sink.start()
    sink.set_live_gains(drum_gain=0, vocal_gain=0, chord_gain=0)
    assert render(sink, 20) == pytest.approx(np.full((20, 2), .6))
    assert not sink.output_device_status()["live_mix"]["supported"]


@pytest.mark.parametrize("invalid", [pcm(float("nan")), pcm(0, frames=10)])
def test_invalid_stems_are_rejected_at_enqueue_before_entering_callback(sink, invalid):
    with pytest.raises(ValueError, match="stem"):
        sink.enqueue(replace(bar(), vocal_stem=invalid))
    assert sink.snapshot().queue_depth == 0


def test_stopped_device_switch_preserves_live_gains(sink, monkeypatch):
    from streammuse.infrastructure.rap.audio_output import OutputDeviceDescriptor

    device = OutputDeviceDescriptor(1, "Test", "Test API", 2, 1000, True, True)
    monkeypatch.setattr(sink, "validate_output_device", lambda _: device)
    accepted = sink.set_live_gains(drum_gain=0, vocal_gain=0, chord_gain=0)
    sink.set_output_device(device)
    sink.enqueue(bar())
    sink.start()
    render(sink, 10)
    assert not render(sink, 1).any()
    assert sink.output_device_status()["live_mix"]["applied_revision"] == accepted["revision"]


def test_live_gains_preserve_active_chord_selection_until_next_bar(sink):
    current = bar(frames=20)
    future = bar(1, frames=20, config=RapMixConfig(**GAINS, chord_loop="off"))
    sink.enqueue(current)
    sink.enqueue(future)
    sink.start()
    render(sink, 1)
    sink.set_live_gains(drum_gain=0, vocal_gain=0, chord_gain=2)
    output = render(sink, 39)
    assert output[9:19] == pytest.approx(np.full((10, 2), .2))
    assert not output[19:].any()
    assert current.mix_config.chord_loop == "minor_cycle"


def test_composite_continuation_records_new_mix_without_rewriting_old_pcm(sink, tmp_path):
    path = tmp_path / "continued-live.wav"
    composite = CompositeAudioSink(sink, Float32WavAudioSink(path, FORMAT))
    try:
        composite.enqueue(bar(0, frames=20))
        composite.start()
        composite.set_live_gains(drum_gain=0, vocal_gain=0, chord_gain=0)
        first = render(sink, 1)
        composite.request_stop_after_bar()
        rest = render(sink, 19)
        composite.discard_pending()
        composite.enqueue(bar(1, frames=20))
        composite.set_live_gains(drum_gain=0, vocal_gain=1, chord_gain=0)
        composite.start()
        second = render(sink, 20)
        composite.drain_notices()
        assert path.read_bytes()[44:] == np.concatenate([first, rest, second]).tobytes()
    finally:
        composite.close()
