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
    session_manager: SessionManager
    output_sink: Any
    service: Any
    session_config: dict[str, Any]
    inference_engine: Any | None = None
    prompt_client: Any | None = None
    websocket_sink: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def session_dir(self) -> Path:
        return self.session_manager.get_session_dir()

    @property
    def running(self) -> bool:
        return bool(getattr(self.service, "running", False))

    def start(self, *, max_ticks: int | None = None) -> None:
        self.output_sink.output_config(self.session_config)
        self.service.start(max_ticks=max_ticks)

    def stop(self) -> None:
        if self.running:
            self.service.stop()

    def cleanup(self) -> None:
        history_payload: object = {}
        try:
            if self.prompt_client is not None:
                self._save_prompt_continuation_history_logs(self.prompt_client)
                history_payload = self.prompt_client.clear_history()
            elif self.inference_engine is not None:
                history_payload = self.inference_engine.clear_history()
        finally:
            self._save_standard_history_logs(history_payload)
            try:
                self.output_sink.close()
            finally:
                self.session_manager.save_summary(
                    {
                        "status": "completed",
                        "session_id": self.session_manager.get_session_id(),
                    }
                )

    def _save_standard_history_logs(self, history_payload: object) -> None:
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
        if not hasattr(client, "prompt_history") or not hasattr(client, "raw_history"):
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
