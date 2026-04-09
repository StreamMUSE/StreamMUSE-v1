from __future__ import annotations

import re
from datetime import datetime

from streammuse.domain.logging import SessionManager


def test_session_manager_creates_date_partitioned_session_directory(tmp_path):
    manager = SessionManager(base_log_dir=str(tmp_path))
    session_dir = manager.create_session_directory()

    today = datetime.now().strftime("%Y-%m-%d")
    assert session_dir.parent.name == today
    assert re.fullmatch(r"session_\d{6}", session_dir.name)
    assert re.fullmatch(r"\d{6}", manager.get_session_id())


def test_session_manager_prefers_legacy_session_path_when_present(tmp_path):
    legacy_id = "20260409-143052"
    legacy_dir = tmp_path / f"session_{legacy_id}"
    legacy_dir.mkdir(parents=True, exist_ok=True)

    manager = SessionManager(base_log_dir=str(tmp_path), session_id=legacy_id)
    resolved = manager.create_session_directory()

    assert resolved == legacy_dir
    assert manager.get_session_dir() == legacy_dir
