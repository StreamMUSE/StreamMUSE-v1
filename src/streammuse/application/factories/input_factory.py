"""InputSource factory."""

from __future__ import annotations

from streammuse.application.config import ApplicationConfig
from streammuse.domain.interfaces import InputSource
from streammuse.domain.musical import MusicalEvent
from streammuse.infrastructure.input import KeyboardInput, ListInput, MidiDeviceInput, MidiFileInput, MidiFileInputConfig


class InputSourceFactory:
    @staticmethod
    def create(app_config: ApplicationConfig, *, list_events: list[MusicalEvent] | None = None) -> InputSource:
        cfg = app_config.input
        tempo = app_config.tempo

        if cfg.type == "midi_device":
            return MidiDeviceInput(device_name=cfg.midi_device_name)

        if cfg.type == "keyboard":
            return KeyboardInput()

        if cfg.type == "midi_file":
            if not cfg.midi_file_path:
                raise ValueError("midi_file_path is required for midi_file input")
            return MidiFileInput(
                cfg.midi_file_path,
                config=MidiFileInputConfig(
                    bpm=float(tempo.bpm),
                    ticks_per_beat=int(tempo.ticks_per_beat),
                    delay_ticks=int(cfg.midi_file_delay_ticks),
                ),
            )

        if cfg.type == "list":
            return ListInput(list_events or [])

        raise ValueError(f"Unknown input type: {cfg.type}")

