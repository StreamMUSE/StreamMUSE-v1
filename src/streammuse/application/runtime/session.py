"""Runtime session lifecycle wrapper."""

from __future__ import annotations

import json
import threading
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
    prompt_session_audit_enabled: bool = False
    prompt_session_seed_provenance: dict[str, Any] | None = None
    _cleaned_up: bool = field(default=False, init=False, repr=False)
    _prompt_session_initialized: bool = field(default=False, init=False, repr=False)
    _cleanup_lock: threading.Lock = field(
        default_factory=threading.Lock,
        init=False,
        repr=False,
    )

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
        self._initialize_prompt_session_for_replay()
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

    def _initialize_prompt_session_for_replay(self) -> None:
        if (
            self._prompt_session_initialized
            or not self.prompt_session_audit_enabled
            or getattr(self.config, "continuation_mode", "standard")
            != "prompt_continuation"
        ):
            return
        if self.prompt_client is None:
            raise RuntimeError("prompt-continuation audit requires a prompt client")

        initialize_session = getattr(self.prompt_client, "initialize_session", None)
        recorder = getattr(
            self.output_sink,
            "record_prompt_continuation_session_seed",
            None,
        )
        if not callable(recorder):
            raise RuntimeError(
                "audit-capable output does not record prompt session seeds"
            )

        if self.prompt_session_seed_provenance is None:
            if not callable(initialize_session):
                raise RuntimeError(
                    "prompt client does not provide initialize_session()"
                )
            response = initialize_session()
            if not isinstance(response, dict) or response.get("success") is not True:
                raise RuntimeError(
                    "prompt-continuation session initialization did not succeed"
                )
        else:
            response = dict(self.prompt_session_seed_provenance)
            replay_audit = getattr(self.prompt_client, "replay_audit", None)
            if not callable(replay_audit):
                raise RuntimeError(
                    "prompt client does not provide replay_audit() for session adoption"
                )
            audit = replay_audit()
            runtime_info = audit.get("runtime_info") if isinstance(audit, dict) else None
            if not isinstance(runtime_info, dict):
                raise RuntimeError("active prompt session has no runtime provenance")
            expected = {
                "prompt_sample_seed": response.get("prompt_effective_seed"),
                "continuation_sample_seed": response.get(
                    "continuation_effective_seed"
                ),
                "session_id": response.get("session_id"),
                "session_epoch": response.get("session_epoch"),
            }
            mismatched = [
                field
                for field, value in expected.items()
                if runtime_info.get(field) != value
            ]
            if mismatched:
                raise RuntimeError(
                    "active prompt session differs from runner reset: "
                    + ", ".join(mismatched)
                )
        provenance = {
            "schema_version": 1,
            "runtime_session_id": (
                self.session_manager.get_session_id()
                if self.session_manager is not None
                else None
            ),
            **response,
        }
        recorder(provenance)
        self.metadata["prompt_continuation_session_seed"] = provenance
        self._prompt_session_initialized = True

    def stop(self) -> None:
        self.service.stop()

    def cleanup(self) -> None:
        with self._cleanup_lock:
            if self._cleaned_up:
                return
            self._cleaned_up = True
            self._cleanup_once()

    def _cleanup_once(self) -> None:
        history_payload: object = {}
        if self.prompt_client is not None:
            self._capture_prompt_continuation_replay_audit(self.prompt_client)
            try:
                self._save_prompt_continuation_history_logs(self.prompt_client)
                history_payload = self.prompt_client.clear_history()
            except Exception as exc:
                self._record_cleanup_warning("clear_history", exc)
        elif self.inference_engine is not None:
            try:
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

    def _capture_prompt_continuation_replay_audit(self, client: Any) -> None:
        finalizer = getattr(
            self.output_sink,
            "finalize_prompt_continuation_replay_audit",
            None,
        )
        if not callable(finalizer):
            return

        replay_audit = getattr(client, "replay_audit", None)
        model_trace: Any = None
        capture_status = "unsupported"
        capture_error: str | None = None
        if callable(replay_audit):
            try:
                model_trace = replay_audit()
                capture_status = "captured"
            except Exception as exc:
                capture_status = "error"
                capture_error = str(exc)
                self._record_cleanup_warning(
                    "prompt_continuation_replay_audit",
                    exc,
                )
        else:
            exc = RuntimeError("prompt client does not provide replay_audit()")
            capture_error = str(exc)
            self._record_cleanup_warning(
                "prompt_continuation_replay_audit",
                exc,
            )

        try:
            finalizer(
                model_trace=model_trace,
                capture_status=capture_status,
                capture_error=capture_error,
            )
        except Exception as exc:
            self._record_cleanup_warning(
                "prompt_continuation_replay_artifacts",
                exc,
            )

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
