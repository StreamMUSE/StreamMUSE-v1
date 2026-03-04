from streammuse.domain.musical import EventType, MusicalEvent
from streammuse.infrastructure.inference.stanley_engine import StanleyInferenceConfig, StanleyInferenceEngine


class _FakeLegacy:
    def __init__(self):
        self.generation_length_frames = 20
        self.melody_history = []
        self.accompaniment_history = []
        self.offset = 0

    def generate_accompaniment(
        self,
        melody_notes,
        generation_start_tick,
        acc_notes=None,
        generation_length_frames=None,
        prompt_length_ticks=None,
    ):
        # Return one duration-note (as legacy dict) starting at generation_start_tick.
        return (
            [{"pitch": 64, "tick": generation_start_tick, "duration": 4, "program": 1}],
            1.0,
            2.0,
            3.0,
            4.0,
        )

    def clear_history(self):
        self.melody_history = []
        self.accompaniment_history = []

    def set_injection_offset(self, offset_ticks: int):
        self.offset = offset_ticks


def test_stanley_adapter_converts_notes_to_events():
    cfg = StanleyInferenceConfig(checkpoint_path="unused.ckpt")
    eng = StanleyInferenceEngine(config=cfg, legacy_engine=_FakeLegacy())

    events, timing = eng.generate_accompaniment(
        melody_events=[MusicalEvent(tick=0, pitch=60, event_type=EventType.NOTE_ON, velocity=100)],
        generation_start_tick=10,
        generation_length_frames=20,
        prompt_length_ticks=None,
    )
    # Should be note_on + note_off for the generated note
    assert len(events) == 2
    assert events[0].event_type == EventType.NOTE_ON and events[0].pitch == 64
    assert events[1].event_type == EventType.NOTE_OFF and events[1].pitch == 64
    assert timing.inference_end_time == 3.0


def test_stanley_adapter_inject_populates_legacy_histories():
    cfg = StanleyInferenceConfig(checkpoint_path="unused.ckpt")
    legacy = _FakeLegacy()
    eng = StanleyInferenceEngine(config=cfg, legacy_engine=legacy)

    eng.inject_history(
        melody_events=[MusicalEvent(tick=0, pitch=60, event_type=EventType.NOTE_ON, velocity=100)],
        accompaniment_events=[],
        injection_length_ticks=50,
    )
    assert legacy.offset == 50
    assert len(legacy.melody_history) >= 1

