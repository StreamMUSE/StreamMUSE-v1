"""MIDI metronome output aligned with playback ticks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import mido

from streammuse.domain.musical import MusicalEvent


@dataclass(frozen=True)
class MetronomeOutputConfig:
    port_name: Optional[str] = None
    ticks_per_beat: int = 4
    beats_per_bar: int = 4
    channel: int = 9
    beat_note: int = 77
    downbeat_note: int = 76
    velocity: int = 80
    downbeat_velocity: int = 110


class MetronomeOutputSink:
    """Plays MIDI percussion clicks on beat boundaries."""

    def __init__(self, config: MetronomeOutputConfig | None = None) -> None:
        self._config = config or MetronomeOutputConfig()
        self._port: mido.ports.BaseOutput | None = None
        self._active_note: int | None = None
        self._disabled_reason: str | None = None

    def _ensure_port(self) -> bool:
        if self._disabled_reason is not None:
            return False
        if self._port is not None:
            return True
        try:
            self._port = mido.open_output(self._config.port_name)
        except Exception as exc:
            self._disabled_reason = str(exc)
            print(
                "[metronome] realtime MIDI output unavailable; "
                f"continuing without live click: {exc}"
            )
            return False
        return True

    def _send_note_off(self) -> None:
        if self._port is None or self._active_note is None:
            return
        self._port.send(
            mido.Message(
                "note_off",
                note=int(self._active_note),
                velocity=0,
                channel=int(self._config.channel),
            )
        )
        self._active_note = None

    def output_metronome_tick(self, tick: int, bar: int, beat: int) -> None:
        _ = bar, beat
        if not self._ensure_port():
            return
        self._send_note_off()

        if int(tick) % int(self._config.ticks_per_beat) != 0:
            return

        ticks_per_bar = int(self._config.ticks_per_beat) * int(self._config.beats_per_bar)
        is_downbeat = ticks_per_bar > 0 and int(tick) % ticks_per_bar == 0
        note = self._config.downbeat_note if is_downbeat else self._config.beat_note
        velocity = self._config.downbeat_velocity if is_downbeat else self._config.velocity

        self._port.send(
            mido.Message(
                "note_on",
                note=int(note),
                velocity=int(velocity),
                channel=int(self._config.channel),
            )
        )
        self._active_note = int(note)

    def output_event(self, event: MusicalEvent, source: str) -> None:
        _ = event, source

    def output_tick(self, tick: int, bar: int, beat: int) -> None:
        _ = tick, bar, beat

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
        _ = (
            hit_rate,
            avg_backup_level,
            round_trip_ms,
            server_process_ms,
            network_latency_ms,
            total_hits,
            total_ticks,
        )

    def output_status(self, state: str, message: str = "") -> None:
        _ = state, message

    def output_config(self, config: Dict[str, Any]) -> None:
        _ = config

    def close(self) -> None:
        self._send_note_off()
        if self._port is not None:
            try:
                self._port.close()
            except Exception:
                pass
            self._port = None
