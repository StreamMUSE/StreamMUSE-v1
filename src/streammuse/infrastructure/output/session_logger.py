"""Session logging output sink combining MIDI and JSON logging."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from streammuse.domain.musical import MusicalEvent
from streammuse.infrastructure.output.json_logger import JsonLoggerOutputSink
from streammuse.infrastructure.output.midi_file import MidiFileOutputConfig, MidiFileOutputSink


class SessionLoggerOutputSink:
    def __init__(
        self,
        session_dir: Path,
        include_midi: bool = True,
        include_json: bool = True,
        inference_log_detail: str = "summary",
        bpm: float = 120.0,
        ticks_per_beat: int = 4,
    ) -> None:
        self.session_dir = Path(session_dir)
        self.include_midi = include_midi
        self.include_json = include_json
        self.inference_log_detail = inference_log_detail

        self.midi_sink: Optional[MidiFileOutputSink] = None
        self.json_sink: Optional[JsonLoggerOutputSink] = None

        if self.include_midi:
            midi_config = MidiFileOutputConfig(
                bpm=float(bpm),
                ticks_per_beat=int(ticks_per_beat),
                output_path=str(self.session_dir / "combined.mid"),
            )
            self.midi_sink = MidiFileOutputSink(midi_config)

        if self.include_json:
            self.json_sink = JsonLoggerOutputSink(
                self.session_dir,
                inference_log_detail=self.inference_log_detail,
            )

    def output_event(self, event: MusicalEvent, source: str) -> None:
        if self.midi_sink:
            self.midi_sink.output_event(event, source)
        if self.json_sink:
            self.json_sink.output_event(event, source)

    def output_tick(self, tick: int, bar: int, beat: int) -> None:
        if self.midi_sink:
            self.midi_sink.output_tick(tick, bar, beat)
        if self.json_sink:
            self.json_sink.output_tick(tick, bar, beat)

    def output_stats(
        self,
        hit_rate: Optional[float] = None,
        avg_backup_level: Optional[float] = None,
        round_trip_ms: Optional[float] = None,
        server_process_ms: Optional[float] = None,
        network_latency_ms: Optional[float] = None,
        total_hits: Optional[int] = None,
        total_ticks: Optional[int] = None,
    ) -> None:
        if self.midi_sink:
            self.midi_sink.output_stats(
                hit_rate,
                avg_backup_level,
                round_trip_ms,
                server_process_ms,
                network_latency_ms,
                total_hits,
                total_ticks,
            )
        if self.json_sink:
            self.json_sink.output_stats(
                hit_rate,
                avg_backup_level,
                round_trip_ms,
                server_process_ms,
                network_latency_ms,
                total_hits,
                total_ticks,
            )

    def output_status(self, state: str, message: str = "") -> None:
        if self.midi_sink:
            self.midi_sink.output_status(state, message)
        if self.json_sink:
            self.json_sink.output_status(state, message)

    def output_config(self, config: Dict[str, Any]) -> None:
        if self.midi_sink:
            self.midi_sink.output_config(config)
        if self.json_sink:
            self.json_sink.output_config(config)

    def log_inference(
        self,
        request: Dict[str, Any],
        response: Dict[str, Any],
        latency_ms: float,
        server_process_ms: float,
    ) -> None:
        if self.json_sink:
            self.json_sink.log_inference(request, response, latency_ms, server_process_ms)

    def save_metrics(self, session_config: Dict[str, Any]) -> None:
        if self.json_sink:
            self.json_sink.save_metrics(session_config)

    def close(self) -> None:
        if self.midi_sink:
            self.midi_sink.close()
        if self.json_sink:
            self.json_sink.close()
