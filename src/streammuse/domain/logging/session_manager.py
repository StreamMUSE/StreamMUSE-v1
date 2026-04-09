"""Session management for logging."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


class SessionManager:
    def __init__(self, base_log_dir: str = "logs", session_id: Optional[str] = None) -> None:
        self.base_log_dir = Path(base_log_dir)
        self.session_timestamp = datetime.now()
        self.session_id = session_id or self._generate_session_id()
        self.date_str = self.session_timestamp.strftime("%Y-%m-%d")

        self._legacy_session_dir = self.base_log_dir / f"session_{self.session_id}"
        self.session_dir = self.base_log_dir / self.date_str / f"session_{self.session_id}"

    def _generate_session_id(self) -> str:
        return self.session_timestamp.strftime("%H%M%S")

    def create_session_directory(self) -> Path:
        # Backward compatibility: if legacy session path exists, keep using it.
        if self._legacy_session_dir.exists() and not self.session_dir.exists():
            self.session_dir = self._legacy_session_dir
            return self.session_dir

        self.session_dir.mkdir(parents=True, exist_ok=True)
        return self.session_dir

    def save_config(self, config: Dict[str, Any]) -> None:
        config_path = self.get_session_dir() / "session_config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

    def save_summary(self, summary: Dict[str, Any]) -> None:
        summary_path = self.get_session_dir() / "session_summary.txt"
        with open(summary_path, "w") as f:
            for key, value in summary.items():
                f.write(f"{key}: {value}\n")

    def get_session_dir(self) -> Path:
        if not self.session_dir.exists() and self._legacy_session_dir.exists():
            return self._legacy_session_dir
        return self.session_dir

    def get_session_id(self) -> str:
        return self.session_id
