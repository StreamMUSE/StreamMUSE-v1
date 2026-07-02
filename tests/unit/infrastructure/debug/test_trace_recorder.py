from __future__ import annotations

import json

from streammuse.domain.debug.trace import DebugTraceEvent
from streammuse.infrastructure.debug.trace_recorder import JsonlDebugTraceRecorder


def test_recorder_writes_manifest_trace_and_artifact(tmp_path) -> None:
    recorder = JsonlDebugTraceRecorder(
        root_dir=tmp_path,
        run_id="run-1",
        runner_kind="offline_direct",
        scenario="lekai-prompt-continuation",
    )

    artifact = recorder.artifact("events", [{"tick": 0, "pitch": 60}], name_hint="melody")
    recorder.record(
        DebugTraceEvent(
            run_id="run-1",
            runner_kind="offline_direct",
            scenario="lekai-prompt-continuation",
            stage="normalized_melody_events",
            output_refs=[artifact],
            output_hash="abc",
            summary={"event_count": 1},
        )
    )
    recorder.close()

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    rows = [json.loads(line) for line in (tmp_path / "trace.jsonl").read_text().splitlines()]

    assert manifest["run_id"] == "run-1"
    assert manifest["runner_kind"] == "offline_direct"
    assert rows[0]["stage"] == "normalized_melody_events"
    assert (tmp_path / rows[0]["output_refs"][0]["path"]).exists()
