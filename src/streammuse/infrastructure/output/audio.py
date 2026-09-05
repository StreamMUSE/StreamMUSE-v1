"""Audio (MIDI out) OutputSink adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import mido

from streammuse.domain.musical import EventType, MusicalEvent


@dataclass(frozen=True)
class AudioOutputConfig:
    port_name: Optional[str] = None
    default_program: int = 0
    mute_melody_output: bool = False


class AudioOutputSink:
    """Outputs note_on/note_off events to a MIDI output port."""

    def __init__(self, config: AudioOutputConfig | None = None) -> None:
        self._config = config or AudioOutputConfig()
        self._port: mido.ports.BaseOutput | None = None

    def _ensure_port(self) -> None:
        if self._port is not None:
            return
        self._port = mido.open_output(self._config.port_name)
        # Best-effort program change on channel 0.
        try:
            self._port.send(
                mido.Message("program_change", channel=0, program=int(self._config.default_program))
            )
        except Exception:
            pass

    def output_event(self, event: MusicalEvent, source: str) -> None:
        if self._config.mute_melody_output and source == "user":
            return
        if event.is_placeholder or event.pitch == -1:
            return
        self._ensure_port()
        if self._port is None:
            return

        if event.event_type == EventType.NOTE_ON and event.velocity > 0:
            self._port.send(
                mido.Message(
                    "note_on",
                    note=int(event.pitch),
                    velocity=int(event.velocity),
                    channel=int(event.channel),
                )
            )
            return

        if event.event_type == EventType.NOTE_OFF:
            self._port.send(
                mido.Message(
                    "note_off",
                    note=int(event.pitch),
                    velocity=0,
                    channel=int(event.channel),
                )
            )
            return

    def output_tick(self, tick: int, bar: int, beat: int) -> None:
        return

    def output_stats(
        self,
        hit_rate=None,
        avg_backup_level=None,
        round_trip_ms=None,
        server_process_ms=None,
        network_latency_ms=None,
        total_hits=None,
        total_ticks=None,
    ) -> None:
        return

    def output_status(self, state: str, message: str = "") -> None:
        return

    def output_config(self, config) -> None:
        return

    def close(self) -> None:
        if self._port is not None:
            try:
                self._port.reset()
            except Exception:
                pass
            try:
                self._port.close()
            except Exception:
                pass
            self._port = None
