"""Session logging output sink combining MIDI and JSON logging."""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from streammuse.domain.musical import EventType, MusicalEvent
from streammuse.infrastructure.inference.serialization import event_from_dict
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
        close_active_notes_on_finalize: bool = True,
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
        self._close_active_notes_on_finalize = bool(
            close_active_notes_on_finalize
        )
        self._schedule_trace_path = self.session_dir / "model_schedule_trace.jsonl"
        self._system_trace_path = self.session_dir / "system_trace.jsonl"
        self._input_quantization_trace_path = (
            self.session_dir / "input_quantization_trace.jsonl"
        )
        self._replay_request_path = (
            self.session_dir / "prompt_continuation_replay_requests.jsonl"
        )
        self._replay_model_trace_path = (
            self.session_dir / "prompt_continuation_model_trace.json"
        )
        self._replay_session_seed_path = (
            self.session_dir / "prompt_continuation_session_seed.json"
        )
        self._replay_manifest_path = self.session_dir / "replay_audit_manifest.json"
        self._replay_melody_json_path = (
            self.session_dir / "prompt_continuation_replay_melody.json"
        )
        self._replay_melody_midi_path = (
            self.session_dir / "prompt_continuation_replay_melody.mid"
        )
        self._theoretical_midi_path = self.session_dir / "theoretical_model.mid"
        self._theoretical_summary_path = self.session_dir / "theoretical_model_summary.json"
        self._lifecycle_path = self.session_dir / "request_lifecycle.jsonl"
        self._validity_path = self.session_dir / "validity.json"
        self._artifact_lock = threading.Lock()
        self._input_quantization_rows: list[Dict[str, Any]] = []
        self._prompt_continuation_replay_request_count = 0
        self._prompt_continuation_replay_successful_request_count = 0
        self._prompt_continuation_replay_failed_request_count = 0
        self._prompt_continuation_replay_melody_events: list[Dict[str, Any]] = []
        self._prompt_continuation_session_seed: Dict[str, Any] | None = None

        self.midi_sink: Optional[MidiFileOutputSink] = None
        self.json_sink: Optional[JsonLoggerOutputSink] = None

        if self.include_midi:
            midi_config = MidiFileOutputConfig(
                bpm=float(bpm),
                ticks_per_beat=int(ticks_per_beat),
                beats_per_bar=int(beats_per_bar),
                output_path=str(self.session_dir / "combined.mid"),
                close_active_notes_on_finalize=self._close_active_notes_on_finalize,
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

    def log_system_trace(self, row: Dict[str, Any]) -> None:
        self.session_dir.mkdir(parents=True, exist_ok=True)
        with self._artifact_lock:
            with self._system_trace_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(dict(row), sort_keys=True) + "\n")

    def log_input_quantization(self, row: Dict[str, Any]) -> None:
        with self._artifact_lock:
            self._input_quantization_rows.append(dict(row))

    @staticmethod
    def _deterministic_json(payload: Any, *, indent: int | None = None) -> str:
        return json.dumps(
            payload,
            ensure_ascii=True,
            indent=indent,
            separators=(",", ":") if indent is None else None,
            sort_keys=True,
        )

    @classmethod
    def _json_copy(cls, payload: Any) -> Any:
        return json.loads(cls._deterministic_json(payload))

    @classmethod
    def _write_json_atomic_unlocked(cls, path: Path, payload: Any) -> None:
        tmp_path = path.with_name(f"{path.name}.tmp")
        tmp_path.write_text(
            cls._deterministic_json(payload, indent=2) + "\n",
            encoding="utf-8",
        )
        tmp_path.replace(path)

    @staticmethod
    def _artifact_summary(path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {"path": path.name, "status": "missing"}
        return {
            "path": path.name,
            "status": "present",
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    @staticmethod
    def _seed_provenance_complete(model_trace: Any) -> bool:
        if not isinstance(model_trace, dict):
            return False
        runtime_info = model_trace.get("runtime_info")
        if isinstance(runtime_info, dict):
            return runtime_info.get("seed_provenance_complete") is True
        return model_trace.get("seed_provenance_complete") is True

    @staticmethod
    def _trace_capture_complete(model_trace: Any) -> bool:
        if not isinstance(model_trace, dict):
            return False
        runtime_info = model_trace.get("runtime_info")
        if isinstance(runtime_info, dict):
            return runtime_info.get("trace_capture_complete") is True
        return model_trace.get("trace_capture_complete") is True

    @staticmethod
    def _model_evidence_complete(model_trace: Any) -> bool:
        if not isinstance(model_trace, dict):
            return False
        prompt = model_trace.get("prompt_generation_log")
        continuation = model_trace.get("continuation_generations")
        if not isinstance(prompt, dict):
            return False
        prompt_tokens = prompt.get("prompt_tokens")
        generated_tokens = prompt.get("generated_tokens")
        return (
            isinstance(prompt_tokens, list)
            and bool(prompt_tokens)
            and isinstance(generated_tokens, list)
            and bool(generated_tokens)
            and isinstance(continuation, list)
            and bool(continuation)
        )

    def log_prompt_continuation_replay_request(
        self, row: Dict[str, Any]
    ) -> None:
        frozen_row = self._json_copy(dict(row))
        request = frozen_row.get("request", {})
        melody_events = (
            request.get("melody_events", []) if isinstance(request, dict) else []
        )
        if not isinstance(melody_events, list):
            raise ValueError("replay request melody_events must be a list")
        if not all(isinstance(event, dict) for event in melody_events):
            raise ValueError("replay request melody_events must contain objects")

        self.session_dir.mkdir(parents=True, exist_ok=True)
        with self._artifact_lock:
            mode = (
                "w"
                if self._prompt_continuation_replay_request_count == 0
                else "a"
            )
            with self._replay_request_path.open(mode, encoding="utf-8") as f:
                f.write(self._deterministic_json(frozen_row) + "\n")
            self._prompt_continuation_replay_request_count += 1
            request_failed = (
                "error" in frozen_row
                or not isinstance(frozen_row.get("acknowledgement"), dict)
            )
            if request_failed:
                self._prompt_continuation_replay_failed_request_count += 1
            else:
                self._prompt_continuation_replay_successful_request_count += 1
                self._prompt_continuation_replay_melody_events.extend(
                    self._json_copy(melody_events)
                )

    def record_prompt_continuation_session_seed(
        self, provenance: Dict[str, Any]
    ) -> None:
        frozen = self._json_copy(dict(provenance))
        required_integer_fields = (
            "prompt_requested_seed",
            "prompt_effective_seed",
            "continuation_requested_seed",
            "continuation_effective_seed",
            "session_epoch",
        )
        for field in required_integer_fields:
            value = frozen.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"session seed field {field} must be a non-negative integer")
        if int(frozen["session_epoch"]) <= 0:
            raise ValueError("session seed field session_epoch must be positive")
        if not isinstance(frozen.get("session_id"), str) or not frozen["session_id"]:
            raise ValueError("session seed field session_id must be a non-empty string")
        for field in ("prompt_seed_source", "continuation_seed_source"):
            if frozen.get(field) not in {"system", "requested"}:
                raise ValueError(
                    f"session seed field {field} must be system or requested"
                )
        if frozen.get("success") is not True:
            raise ValueError("session seed initialization must be successful")

        self.session_dir.mkdir(parents=True, exist_ok=True)
        with self._artifact_lock:
            if (
                self._prompt_continuation_session_seed is not None
                and self._prompt_continuation_session_seed != frozen
            ):
                raise RuntimeError("prompt-continuation session seed already recorded")
            self._write_json_atomic_unlocked(self._replay_session_seed_path, frozen)
            self._prompt_continuation_session_seed = frozen

    def _session_seed_matches_model_trace(self, model_trace: Any) -> bool:
        seed = self._prompt_continuation_session_seed
        if seed is None or not isinstance(model_trace, dict):
            return False
        runtime_info = model_trace.get("runtime_info")
        if not isinstance(runtime_info, dict):
            return False
        return (
            seed.get("prompt_effective_seed")
            == runtime_info.get("prompt_sample_seed")
            and seed.get("continuation_effective_seed")
            == runtime_info.get("continuation_sample_seed")
            and seed.get("session_id") == runtime_info.get("session_id")
            and seed.get("session_epoch") == runtime_info.get("session_epoch")
        )

    def _write_replay_melody_unlocked(self) -> None:
        melody_payload = {
            "schema_version": 1,
            "source": self._replay_request_path.name,
            "tempo": {
                "bpm": self._bpm,
                "ticks_per_beat": self._ticks_per_beat,
                "beats_per_bar": self._beats_per_bar,
            },
            "event_count": len(self._prompt_continuation_replay_melody_events),
            "events": self._prompt_continuation_replay_melody_events,
        }
        self._write_json_atomic_unlocked(
            self._replay_melody_json_path,
            melody_payload,
        )

        midi_sink = MidiFileOutputSink(
            MidiFileOutputConfig(
                bpm=self._bpm,
                ticks_per_beat=self._ticks_per_beat,
                beats_per_bar=self._beats_per_bar,
                output_path=str(self._replay_melody_midi_path),
                user_track_name="Replay Melody",
                model_track_name="Unused Accompaniment",
                close_active_notes_on_finalize=(
                    self._close_active_notes_on_finalize
                ),
                record_metronome=False,
            )
        )
        for event_payload in self._prompt_continuation_replay_melody_events:
            midi_sink.output_event(event_from_dict(event_payload), source="user")
        midi_sink.close()

    def finalize_prompt_continuation_replay_audit(
        self,
        *,
        model_trace: Any,
        capture_status: str,
        capture_error: str | None = None,
    ) -> Dict[str, Any]:
        """Persist the self-contained artifacts needed by a later replay."""
        if capture_status not in {"captured", "unsupported", "error"}:
            raise ValueError(f"unsupported replay audit capture status: {capture_status}")

        self._flush_input_quantization_trace()
        self.session_dir.mkdir(parents=True, exist_ok=True)
        with self._artifact_lock:
            if self._prompt_continuation_replay_request_count == 0:
                self._replay_request_path.write_text("", encoding="utf-8")
            self._write_replay_melody_unlocked()
            if capture_status == "captured":
                model_trace_document = self._json_copy(model_trace)
            else:
                model_trace_document = {
                    "schema_version": 1,
                    "capture_status": capture_status,
                    "error": capture_error,
                }
            self._write_json_atomic_unlocked(
                self._replay_model_trace_path,
                model_trace_document,
            )

            artifacts = {
                "request_trace": self._artifact_summary(self._replay_request_path),
                "model_trace": self._artifact_summary(self._replay_model_trace_path),
                "session_seed": self._artifact_summary(
                    self._replay_session_seed_path
                ),
                "replay_melody_json": self._artifact_summary(
                    self._replay_melody_json_path
                ),
                "replay_melody_midi": self._artifact_summary(
                    self._replay_melody_midi_path
                ),
            }
            if self._input_quantization_trace_path.exists():
                artifacts["input_quantization_trace"] = self._artifact_summary(
                    self._input_quantization_trace_path
                )

            session_seed_recorded = self._prompt_continuation_session_seed is not None
            session_seed_matches_model_trace = self._session_seed_matches_model_trace(
                model_trace
            )
            seed_provenance_complete = (
                self._seed_provenance_complete(model_trace)
                and session_seed_recorded
                and session_seed_matches_model_trace
            )
            trace_capture_complete = self._trace_capture_complete(model_trace)
            model_evidence_complete = self._model_evidence_complete(model_trace)
            comparable = (
                capture_status == "captured"
                and self._prompt_continuation_replay_request_count > 0
                and self._prompt_continuation_replay_failed_request_count == 0
                and trace_capture_complete
                and seed_provenance_complete
                and model_evidence_complete
            )
            missing_requirements: list[str] = []
            if capture_status != "captured":
                missing_requirements.append("captured_model_trace")
            if self._prompt_continuation_replay_request_count <= 0:
                missing_requirements.append("protocol_requests")
            if self._prompt_continuation_replay_failed_request_count > 0:
                missing_requirements.append("failed_protocol_requests")
            if not trace_capture_complete:
                missing_requirements.append("complete_model_trace")
            if not model_evidence_complete:
                missing_requirements.append("prompt_continuation_model_evidence")
            if not seed_provenance_complete:
                missing_requirements.append("complete_seed_provenance")
            manifest: Dict[str, Any] = {
                "schema_version": 1,
                "audit": "human_live_to_realtime_midi_file_exact_match",
                "status": "ready_for_replay" if comparable else "incomplete",
                "comparable": comparable,
                "comparison_status": "not_run",
                "model_trace_capture_status": capture_status,
                "trace_capture_complete": trace_capture_complete,
                "model_evidence_complete": model_evidence_complete,
                "seed_provenance_complete": seed_provenance_complete,
                "session_seed_recorded": session_seed_recorded,
                "session_seed_matches_model_trace": (
                    session_seed_matches_model_trace
                ),
                "session_seed": self._json_copy(
                    self._prompt_continuation_session_seed
                ),
                "missing_requirements": missing_requirements,
                "request_count": self._prompt_continuation_replay_request_count,
                "successful_protocol_requests": (
                    self._prompt_continuation_replay_successful_request_count
                ),
                "failed_protocol_requests": (
                    self._prompt_continuation_replay_failed_request_count
                ),
                "melody_event_count": len(
                    self._prompt_continuation_replay_melody_events
                ),
                "tempo": {
                    "bpm": self._bpm,
                    "ticks_per_beat": self._ticks_per_beat,
                    "beats_per_bar": self._beats_per_bar,
                },
                "artifacts": artifacts,
            }
            if capture_error is not None:
                manifest["capture_error"] = str(capture_error)
            self._write_json_atomic_unlocked(self._replay_manifest_path, manifest)
            return self._json_copy(manifest)

    def _flush_input_quantization_trace(self) -> None:
        with self._artifact_lock:
            if not self._input_quantization_rows:
                return
            self.session_dir.mkdir(parents=True, exist_ok=True)
            with self._input_quantization_trace_path.open("w", encoding="utf-8") as f:
                for row in self._input_quantization_rows:
                    f.write(json.dumps(row, sort_keys=True) + "\n")
            self._input_quantization_rows.clear()

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
                close_active_notes_on_finalize=self._close_active_notes_on_finalize,
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
            self._flush_input_quantization_trace()
        except Exception as exc:
            first_error = exc
        try:
            if self.midi_sink:
                self.midi_sink.close()
            self._write_theoretical_model_midi()
        except Exception as exc:
            if first_error is None:
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
