import numpy as np

from streammuse.domain.rap.audio import AudioFormat, RapMixConfig
from streammuse.domain.rap.interaction import CaptureWindow, TapRecord, quantize_capture
from streammuse.domain.timing import Tempo
from streammuse.infrastructure.rap.drums import ProceduralBoomBapRenderer
from streammuse.infrastructure.rap.flow_echo import FlowEchoRenderer
from streammuse.infrastructure.rap.templates import BUILTIN_TEMPLATES
from streammuse.infrastructure.rap.scenarios import default_scenario


def test_echo_preserves_drums_and_adds_distinct_onset_hits_without_lyrics():
    idle = BUILTIN_TEMPLATES.get(default_scenario().segments[0].template_id)
    renderer = FlowEchoRenderer(drums=ProceduralBoomBapRenderer(seed=2), idle_template=idle)
    template = quantize_capture(CaptureWindow(0, 0.0, 4.0), (TapRecord(0.75, 0.8),), template_id="tap").template
    common = dict(bar=3, tempo=Tempo(60, 4, 4), audio_format=AudioFormat())
    drums = renderer.render(**common)
    echo = renderer.render(**common, template=template)
    baseline = np.frombuffer(drums.audio.data, dtype=np.float32).reshape(-1, 2)
    samples = np.frombuffer(echo.audio.data, dtype=np.float32).reshape(-1, 2)
    assert echo.audio.frame_count == 192000
    assert echo.text == "" and echo.scheduled == ()
    assert echo.source == "flow_echo"
    assert np.array_equal(samples[:36000], baseline[:36000])
    assert np.max(np.abs(samples[36000:38000] - baseline[36000:38000])) > 0.1
    assert np.max(np.abs(samples)) <= 0.95


def test_echo_mix_update_changes_future_drums_without_muting_captured_percussion():
    idle = BUILTIN_TEMPLATES.get(default_scenario().segments[0].template_id)
    renderer = FlowEchoRenderer(drums=ProceduralBoomBapRenderer(seed=2), idle_template=idle)
    template = quantize_capture(CaptureWindow(0, 0.0, 4.0), (TapRecord(0.75, 0.8),), template_id="tap").template
    renderer.set_mix_config(RapMixConfig(drum_gain=0.0, vocal_gain=0.0, revision=4))
    echo = renderer.render(bar=0, tempo=Tempo(60, 4, 4), audio_format=AudioFormat(), template=template)
    samples = np.frombuffer(echo.audio.data, dtype=np.float32).reshape(-1, 2)
    assert np.max(np.abs(samples[:36000])) == 0
    assert np.max(np.abs(samples[36000:38000])) > 0.1
    assert echo.drum_stem is not None and echo.vocal_stem is not None
    assert echo.mix_config.revision == 4


def test_chords_continue_through_idle_and_capture_echo_bars():
    idle = BUILTIN_TEMPLATES.get(default_scenario().segments[0].template_id)
    renderer = FlowEchoRenderer(drums=ProceduralBoomBapRenderer(seed=2), idle_template=idle)
    renderer.set_mix_config(RapMixConfig(chord_loop="minor_cycle", drum_gain=0))
    prepared = renderer.render(bar=2, tempo=Tempo(60, 4, 4), audio_format=AudioFormat())
    assert prepared.chord_stem is not None
    assert prepared.chord_stem.frame_count == 192000
    assert np.max(np.abs(np.frombuffer(prepared.audio.data, dtype=np.float32))) > 0
