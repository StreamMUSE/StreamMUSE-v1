"""Session logging output sink combining MIDI and JSON logging."""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from streammuse.domain.musical import EventType, MusicalEvent
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
        beats_per_bar: int = 4,
        record_metronome: bool = False,
        artifact_tier: str = "debug",
    ) -> None:
        self.session_dir = Path(session_dir)
        self.include_midi = include_midi
        self.include_json = include_json
        self.inference_log_detail = inference_log_detail
        self.artifact_tier = str(artifact_tier)
        self._session_config: Optional[Dict[str, Any]] = None
        self._closed = False
        self._metrics_saved = False
        self._bpm = float(bpm)
        self._ticks_per_beat = int(ticks_per_beat)
        self._beats_per_bar = int(beats_per_bar)
        self._schedule_trace_path = self.session_dir / "model_schedule_trace.jsonl"
        self._theoretical_midi_path = self.session_dir / "theoretical_model.mid"
        self._theoretical_summary_path = self.session_dir / "theoretical_model_summary.json"
        self._lifecycle_path = self.session_dir / "request_lifecycle.jsonl"
        self._validity_path = self.session_dir / "validity.json"
        self._artifact_lock = threading.Lock()

        self.midi_sink: Optional[MidiFileOutputSink] = None
        self.json_sink: Optional[JsonLoggerOutputSink] = None

        if self.include_midi:
            midi_config = MidiFileOutputConfig(
                bpm=float(bpm),
                ticks_per_beat=int(ticks_per_beat),
                beats_per_bar=int(beats_per_bar),
                output_path=str(self.session_dir / "combined.mid"),
                record_metronome=bool(record_metronome),
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

    def output_metronome_tick(self, tick: int, bar: int, beat: int) -> None:
        if self.midi_sink:
            self.midi_sink.output_metronome_tick(tick, bar, beat)

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
        self._session_config = dict(config)
        self._metrics_saved = False
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

    def log_model_schedule(self, rows: list[Dict[str, Any]]) -> None:
        if not rows:
            return
        self.session_dir.mkdir(parents=True, exist_ok=True)
        with self._schedule_trace_path.open("a", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, sort_keys=True) + "\n")

    def log_request_lifecycle(self, row: Dict[str, Any]) -> None:
        if self.json_sink:
            self.json_sink.log_request_lifecycle(row)
            return
        self.session_dir.mkdir(parents=True, exist_ok=True)
        with self._artifact_lock:
            with self._lifecycle_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(dict(row), sort_keys=True) + "\n")

    def finalize_validity(self, summary: Dict[str, Any]) -> None:
        if self.json_sink:
            self.json_sink.finalize_validity(summary)
            return
        self.session_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = self._validity_path.with_suffix(".json.tmp")
        with self._artifact_lock:
            tmp_path.write_text(
                json.dumps(dict(summary), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            tmp_path.replace(self._validity_path)

    def _write_theoretical_model_midi(self) -> None:
        if not self.include_midi:
            return

        self.session_dir.mkdir(parents=True, exist_ok=True)
        # An empty schedule trace is a required, meaningful artifact: it proves
        # that the exporter observed no model events instead of silently losing
        # a missing trace file.
        self._schedule_trace_path.touch(exist_ok=True)
        events: list[MusicalEvent] = []
        trace_lines = self._schedule_trace_path.read_text(encoding="utf-8").splitlines()
        parsed_row_count = 0
        forced_note_off_count = 0
        for line_number, line in enumerate(trace_lines, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid model schedule trace JSON at line {line_number}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(
                    f"model schedule trace row {line_number} must be an object"
                )
            parsed_row_count += 1
            if row.get("policy") == "forced_note_off":
                forced_note_off_count += 1
                continue
            event_type = row.get("type")
            logical_tick = row.get("logical_tick")
            if event_type not in {"note_on", "note_off"} or logical_tick is None:
                raise ValueError(
                    f"model schedule trace row {line_number} lacks event semantics"
                )
            action = row.get("action")
            if action not in {"scheduled", "dropped"}:
                raise ValueError(
                    f"model schedule trace row {line_number} has invalid action"
                )
            pitch = int(row.get("pitch", -1))
            # Placeholder rows carry pitch=-1 and intentionally do not become
            # audible MIDI notes.
            if pitch == -1:
                continue
            if not 0 <= pitch <= 127:
                raise ValueError(
                    f"model schedule trace row {line_number} has invalid pitch"
                )
            events.append(
                MusicalEvent(
                    tick=int(logical_tick),
                    pitch=pitch,
                    event_type=(EventType.NOTE_ON if event_type == "note_on" else EventType.NOTE_OFF),
                    velocity=int(row.get("velocity", 0 if event_type == "note_off" else 80)),
                    channel=int(row.get("channel", 0)),
                    program=int(row.get("program", 0)),
                    source="model",
                )
            )

        events.sort(key=lambda event: (
            int(event.tick),
            0 if event.event_type == EventType.NOTE_OFF else 1,
            int(event.pitch),
            int(event.channel),
            int(event.program),
        ))
        sink = MidiFileOutputSink(
            MidiFileOutputConfig(
                bpm=self._bpm,
                ticks_per_beat=self._ticks_per_beat,
                beats_per_bar=self._beats_per_bar,
                output_path=str(self._theoretical_midi_path),
                user_track_name="Melody",
                model_track_name="Theoretical Accompaniment",
                record_metronome=False,
            )
        )
        for event in events:
            sink.output_event(event, "model")
        sink.close()
        # Parse the file we actually wrote rather than inferring success from
        # the in-memory event list.  This also makes a valid empty MIDI an
        # explicit success artifact.
        import pretty_midi

        exported = pretty_midi.PrettyMIDI(str(self._theoretical_midi_path))
        note_count = sum(len(instrument.notes) for instrument in exported.instruments)
        trace_sha256 = hashlib.sha256(self._schedule_trace_path.read_bytes()).hexdigest()
        summary = {
            "schema_version": 2,
            "export_status": "success",
            "trace_path": self._schedule_trace_path.name,
            "trace_sha256": trace_sha256,
            "trace_row_count": parsed_row_count,
            "forced_note_off_excluded_count": forced_note_off_count,
            "event_count": len(events),
            "note_count": note_count,
            "empty": note_count == 0,
            "midi_path": self._theoretical_midi_path.name,
        }
        tmp_path = self._theoretical_summary_path.with_suffix(".json.tmp")
        tmp_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        tmp_path.replace(self._theoretical_summary_path)

    def save_metrics(self, session_config: Dict[str, Any]) -> None:
        self._session_config = dict(session_config)
        if self.json_sink:
            self.json_sink.save_metrics(self._session_config)
            self._metrics_saved = True

    def close(self) -> None:
        if self._closed:
            return

        if self._session_config is not None and self.json_sink and not self._metrics_saved:
            self.json_sink.save_metrics(self._session_config)
            self._metrics_saved = True

        first_error: Exception | None = None
        try:
            if self.midi_sink:
                self.midi_sink.close()
            self._write_theoretical_model_midi()
        except Exception as exc:
            first_error = exc
        try:
            if self.json_sink:
                self.json_sink.close()
        except Exception as exc:
            if first_error is None:
                first_error = exc
        self._closed = True
        if first_error is not None:
            raise first_error
