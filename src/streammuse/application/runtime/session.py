"""Runtime session lifecycle wrapper."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from streammuse.domain.logging import SessionManager
from streammuse.domain.musical import MusicalEvent
from streammuse.infrastructure.inference.serialization import event_to_dict


@dataclass
class RuntimeSession:
    config: Any
    session_manager: SessionManager | None
    output_sink: Any
    service: Any
    session_config: dict[str, Any]
    inference_engine: Any | None = None
    prompt_client: Any | None = None
    websocket_sink: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    emit_output_config: bool = True
    write_summary_on_cleanup: bool = True
    _cleaned_up: bool = field(default=False, init=False, repr=False)

    @property
    def session_dir(self) -> Path:
        if self.session_manager is None:
            raise RuntimeError("this runtime session has no session directory")
        return self.session_manager.get_session_dir()

    @property
    def running(self) -> bool:
        return bool(getattr(self.service, "running", False))

    def start(
        self,
        *,
        max_ticks: int | None = None,
        run_stop_tick: int | None = None,
        analysis_end_tick: int | None = None,
        last_input_note_off_tick: int | None = None,
        request_cutoff_tick: int | None = None,
        drain_timeout_seconds: float | None = None,
    ) -> None:
        if run_stop_tick is None:
            run_stop_tick = max_ticks
        elif max_ticks is not None and run_stop_tick != max_ticks:
            raise ValueError("max_ticks and run_stop_tick must match")
        if self.emit_output_config:
            self.output_sink.output_config(self.session_config)
        if getattr(self.config, "continuation_mode", "standard") == "prompt_continuation":
            unsupported = {
                "analysis_end_tick": analysis_end_tick,
                "last_input_note_off_tick": last_input_note_off_tick,
                "request_cutoff_tick": request_cutoff_tick,
                "drain_timeout_seconds": drain_timeout_seconds,
            }
            selected = [name for name, value in unsupported.items() if value is not None]
            if selected:
                raise ValueError(
                    "prompt_continuation does not support standard run controls: "
                    + ", ".join(selected)
                )
            self.service.start(max_ticks=run_stop_tick)
            return
        self.service.start(
            run_stop_tick=run_stop_tick,
            analysis_end_tick=analysis_end_tick,
            last_input_note_off_tick=last_input_note_off_tick,
            request_cutoff_tick=request_cutoff_tick,
            drain_timeout_seconds=drain_timeout_seconds,
        )

    def stop(self) -> None:
        self.service.stop()

    def cleanup(self) -> None:
        if self._cleaned_up:
            return
        self._cleaned_up = True
        history_payload: object = {}
        try:
            if self.prompt_client is not None:
                self._save_prompt_continuation_history_logs(self.prompt_client)
                history_payload = self.prompt_client.clear_history()
            elif self.inference_engine is not None:
                history_payload = self.inference_engine.clear_history()
        except Exception as exc:
            self._record_cleanup_warning("clear_history", exc)
        if self._debug_artifacts_enabled():
            try:
                self._save_standard_history_logs(history_payload)
            except Exception as exc:
                self._record_cleanup_warning("history_logs", exc)
        try:
            self.output_sink.close()
        except Exception as exc:
            self._record_cleanup_warning("output_close", exc)
        if self.session_manager is not None and self.write_summary_on_cleanup:
            try:
                self.session_manager.save_summary(
                    {
                        "status": "completed",
                        "session_id": self.session_manager.get_session_id(),
                    }
                )
            except Exception as exc:
                self._record_cleanup_warning("session_summary", exc)

    def _record_cleanup_warning(self, operation: str, exc: Exception) -> None:
        warnings = self.metadata.setdefault("cleanup_warnings", [])
        if isinstance(warnings, list):
            warnings.append({"operation": operation, "error": str(exc)})

    def _debug_artifacts_enabled(self) -> bool:
        return (
            self.session_manager is not None
            and getattr(self.config.output, "session_artifact_tier", "debug") == "debug"
        )

    def _save_standard_history_logs(self, history_payload: object) -> None:
        if self.session_manager is None:
            return
        if not isinstance(history_payload, dict):
            return
        melody_history = history_payload.get("melody_history", [])
        accompaniment_history = history_payload.get("accompaniment_history", [])
        (self.session_dir / "melody_history.json").write_text(
            json.dumps(melody_history if isinstance(melody_history, list) else [], indent=2),
            encoding="utf-8",
        )
        (self.session_dir / "accompaniment_history.json").write_text(
            json.dumps(
                accompaniment_history if isinstance(accompaniment_history, list) else [],
                indent=2,
            ),
            encoding="utf-8",
        )

    def _save_prompt_continuation_history_logs(self, client: Any) -> None:
        if (
            self.session_manager is None
            or not self._debug_artifacts_enabled()
            or not hasattr(client, "prompt_history")
            or not hasattr(client, "raw_history")
        ):
            return
        prompt_events, prompt_status = client.prompt_history()
        raw_events, raw_status = client.raw_history()
        self._write_events("prompt_continuation_prompt_history.json", prompt_events)
        self._write_events("prompt_continuation_raw_history.json", raw_events)
        (self.session_dir / "prompt_continuation_history_status.json").write_text(
            json.dumps(
                {"prompt_status": prompt_status, "raw_status": raw_status},
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    def _write_events(self, name: str, events: list[MusicalEvent]) -> None:
        (self.session_dir / name).write_text(
            json.dumps([event_to_dict(event) for event in events], indent=2),
            encoding="utf-8",
        )
