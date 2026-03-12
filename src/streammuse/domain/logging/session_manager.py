"""Session management for logging."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


class SessionManager:
    def __init__(self, base_log_dir: str = "logs") -> None:
        self.base_log_dir = Path(base_log_dir)
        self.session_id = self._generate_session_id()
        self.session_dir = self.base_log_dir / f"session_{self.session_id}"

    def _generate_session_id(self) -> str:
        now = datetime.now()
        return now.strftime("%Y%m%d-%H%M%S")

    def create_session_directory(self) -> Path:
        self.session_dir.mkdir(parents=True, exist_ok=True)
        return self.session_dir

    def save_config(self, config: Dict[str, Any]) -> None:
        config_path = self.session_dir / "session_config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

    def save_summary(self, summary: Dict[str, Any]) -> None:
        summary_path = self.session_dir / "session_summary.txt"
        with open(summary_path, "w") as f:
            for key, value in summary.items():
                f.write(f"{key}: {value}\n")

    def get_session_dir(self) -> Path:
        return self.session_dir

    def get_session_id(self) -> str:
        return self.session_id
