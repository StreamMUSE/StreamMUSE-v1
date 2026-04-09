import time

import pytest

from streammuse.application.config import ApplicationConfig, InferenceConfig, InputConfig, OutputConfig, TempoConfig
from streammuse.application.factories import InferenceEngineFactory, InputSourceFactory, OutputSinkFactory
from streammuse.application.services.real_time_music_service import RealTimeMusicService
from streammuse.domain.logging import SessionManager
from streammuse.domain.musical import EventType, MusicalEvent
from streammuse.domain.timing import PlaybackScheduler, Tempo


class _FakeInference:
    def generate_accompaniment(self, *args, **kwargs):
        raise NotImplementedError

    def inject_history(self, *args, **kwargs):
        raise NotImplementedError

    def set_injection_offset(self, *args, **kwargs):
        return None

    def clear_history(self):
        return {"success": True, "melody_history": [], "accompaniment_history": []}


class _CollectingOutput:
    def __init__(self):
        self.ticks = []
        self.events = []
        self.status = []

    def output_event(self, event, source):
        self.events.append((source, event))

    def output_tick(self, tick, bar, beat):
        self.ticks.append((tick, bar, beat))

    def output_stats(self, **kwargs):
        return None

    def output_status(self, state, message=""):
        self.status.append((state, message))

    def output_config(self, config):
        return None

    def close(self):
        return None


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


def test_real_time_music_service_emits_ticks_and_user_events():
    cfg = ApplicationConfig(input=InputConfig(type="list"))
    inp = InputSourceFactory.create(
        cfg,
        list_events=[MusicalEvent(tick=0, pitch=60, event_type=EventType.NOTE_ON, velocity=100)],
    )
    out = _CollectingOutput()
    tempo = Tempo(bpm=120.0, ticks_per_beat=4, beats_per_bar=4)
    svc = RealTimeMusicService(
        input_source=inp,
        inference_engine=_FakeInference(),
        output_sink=out,
        tempo=tempo,
        scheduler=PlaybackScheduler(),
        now=lambda: 0.0,
        sleep=lambda _: None,
    )
    svc.start(max_ticks=3)
    # Let threads run briefly (they don't sleep).
    time.sleep(0.01)
    svc.stop()

    assert any(s[0] == "running" for s in out.status)
    assert len(out.ticks) >= 1
    assert any(e[0] == "user" and e[1].pitch == 60 for e in out.events)


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


@pytest.mark.parametrize("output_type", ["json_log", "composite"])
def test_output_factory_propagates_inference_log_detail(tmp_path, output_type):
    session_manager = SessionManager(base_log_dir=str(tmp_path))
    session_manager.create_session_directory()

    cfg = ApplicationConfig(
        output=OutputConfig(type=output_type, inference_log_detail="full"),
        inference=InferenceConfig(type="http", server_generate_url="http://x/generate_accompaniment"),
    )

    sink = OutputSinkFactory.create(cfg, session_manager=session_manager)
    assert getattr(sink, "inference_log_detail", "summary") == "full"

