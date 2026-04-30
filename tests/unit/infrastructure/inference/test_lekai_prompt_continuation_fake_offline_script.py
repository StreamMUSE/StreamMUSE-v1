from __future__ import annotations

import importlib.util
from pathlib import Path

from streammuse.domain.musical import EventType, MusicalEvent


_SCRIPT_PATH = Path(__file__).resolve().parents[4] / "scripts" / "run_lekai_prompt_continuation_fake_offline.py"
_SPEC = importlib.util.spec_from_file_location("run_lekai_prompt_continuation_fake_offline", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
fake_offline = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(fake_offline)


def _event(tick: int, pitch: int = 60) -> MusicalEvent:
    return MusicalEvent(tick=tick, pitch=pitch, event_type=EventType.NOTE_ON, velocity=64)


def test_normalise_base_url_accepts_base_and_known_endpoints():
    assert fake_offline.normalise_base_url("http://127.0.0.1:8000") == "http://127.0.0.1:8000"
    assert (
        fake_offline.normalise_base_url("http://127.0.0.1:8000/generate_accompaniment")
        == "http://127.0.0.1:8000"
    )
    assert (
        fake_offline.normalise_base_url("http://127.0.0.1:8000/prompt_continuation/status")
        == "http://127.0.0.1:8000"
    )


def test_split_prompt_and_append_chunks_keeps_empty_rest_chunks():
    events = [_event(0), _event(31), _event(44)]

    prompt_events, append_chunks = fake_offline.split_prompt_and_append_chunks(
        events,
        prompt_length_ticks=32,
        append_until_tick=48,
        append_interval_ticks=4,
    )

    assert [event.tick for event in prompt_events] == [0, 31]
    assert [chunk["observed_until_tick"] for chunk in append_chunks] == [36, 40, 44, 48]
    assert [len(chunk["events"]) for chunk in append_chunks] == [0, 0, 0, 1]
    assert append_chunks[-1]["events"][0].tick == 44


def test_event_payloads_preserve_type_pitch_tick_velocity():
    payload = fake_offline.event_payloads([_event(12, pitch=67)])

    assert payload == [{"type": "note_on", "pitch": 67, "tick": 12, "velocity": 64, "channel": 0, "program": 0}]
