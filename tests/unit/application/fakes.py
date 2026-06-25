"""Shared no-op fakes for application-layer tests."""

from __future__ import annotations

from streammuse.domain.musical import EventType, MusicalEvent


class NoopInput:
    def read_events(self):
        return iter([])

    def close(self):
        return None


class NoopOutput:
    def output_event(self, event, source):
        return None

    def output_tick(self, tick, bar, beat):
        return None

    def output_stats(self, **kwargs):
        return None

    def output_status(self, state, message=""):
        return None

    def output_config(self, config):
        return None

    def close(self):
        return None


class NoopInference:
    def generate_accompaniment(self, *args, **kwargs):
        raise NotImplementedError

    def inject_history(self, *args, **kwargs):
        return None

    def set_injection_offset(self, offset_ticks: int):
        return None

    def clear_history(self):
        return {"success": True, "melody_history": [], "accompaniment_history": []}


def note(pitch: int, tick: int) -> MusicalEvent:
    return MusicalEvent(tick=tick, pitch=pitch, event_type=EventType.NOTE_ON, velocity=100)
