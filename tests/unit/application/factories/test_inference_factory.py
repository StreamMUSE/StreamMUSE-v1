from __future__ import annotations

import pytest

from streammuse.application.config import ApplicationConfig, InferenceConfig, InputConfig, OutputConfig, TempoConfig
from streammuse.application.factories import InferenceEngineFactory, InputSourceFactory, OutputSinkFactory


def test_factories_create_basic_components():
    cfg = ApplicationConfig(
        tempo=TempoConfig(),
        input=InputConfig(type="list"),
        output=OutputConfig(type="console"),
        inference=InferenceConfig(type="http", server_generate_url="http://x/generate_accompaniment"),
    )
    inp = InputSourceFactory.create(cfg, list_events=[])
    out = OutputSinkFactory.create(cfg)
    eng = InferenceEngineFactory.create(cfg)
    assert hasattr(inp, "read_events")
    assert hasattr(out, "output_event")
    assert hasattr(eng, "generate_accompaniment")


@pytest.mark.parametrize(
    ("generation_interval_ticks", "generation_length_frames", "should_raise"),
    [
        (3, 16, False),
        (2, 20, False),
        (2, 17, True),
    ],
)
def test_http_lekai_validates_generation_length_only(
    generation_interval_ticks,
    generation_length_frames,
    should_raise,
):
    cfg = ApplicationConfig(
        inference=InferenceConfig(
            type="http",
            server_generate_url="http://x/generate_accompaniment",
            model_name="lekai",
            generation_interval_ticks=generation_interval_ticks,
            generation_length_frames=generation_length_frames,
        )
    )
    if should_raise:
        with pytest.raises(ValueError, match="generation-length-frames.*multiple of 4"):
            InferenceEngineFactory.create(cfg)
    else:
        eng = InferenceEngineFactory.create(cfg)
        assert hasattr(eng, "generate_accompaniment")


def test_http_non_lekai_skips_multiple_of_4_checks():
    cfg = ApplicationConfig(
        inference=InferenceConfig(
            type="http",
            server_generate_url="http://x/generate_accompaniment",
            model_name="stanley",
            generation_interval_ticks=2,
            generation_length_frames=10,
        )
    )
    eng = InferenceEngineFactory.create(cfg)
    assert hasattr(eng, "generate_accompaniment")


def test_http_factory_propagates_checkpoint_path_for_http_client():
    cfg = ApplicationConfig(
        inference=InferenceConfig(
            type="http",
            server_generate_url="http://x/generate_accompaniment",
            model_name="lekai",
            generation_interval_ticks=4,
            generation_length_frames=20,
            checkpoint_path="/tmp/lekai.ckpt",
        )
    )

    eng = InferenceEngineFactory.create(cfg)
    assert getattr(eng, "_config").checkpoint_path == "/tmp/lekai.ckpt"
