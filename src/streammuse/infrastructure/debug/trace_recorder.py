"""JSONL debug trace recorder and artifact store."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Protocol

from streammuse.domain.debug.canonical import hash_jsonable
from streammuse.domain.debug.trace import ArtifactRef, DebugTraceEvent


class DebugTraceRecorder(Protocol):
    def record(self, event: DebugTraceEvent) -> None: ...

    def artifact(self, kind: str, payload: object, *, name_hint: str) -> ArtifactRef: ...


class JsonlDebugTraceRecorder:
    def __init__(
        self,
        *,
        root_dir: Path | str,
        run_id: str,
        runner_kind: str,
        scenario: str,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.run_id = run_id
        self.runner_kind = runner_kind
        self.scenario = scenario
        self.artifact_root = self.root_dir / "artifacts"
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        self.trace_path = self.root_dir / "trace.jsonl"
        self.manifest_path = self.root_dir / "manifest.json"
        self._trace_file = self.trace_path.open("a", encoding="utf-8")
        self._artifact_counter = 0
        self._write_manifest()

    def _write_manifest(self) -> None:
        self.manifest_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "run_id": self.run_id,
                    "runner_kind": self.runner_kind,
                    "scenario": self.scenario,
                    "trace_path": "trace.jsonl",
                    "artifact_root": "artifacts",
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    @staticmethod
    def _safe_name(value: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
        return cleaned or "artifact"

    def artifact(self, kind: str, payload: object, *, name_hint: str) -> ArtifactRef:
        self._artifact_counter += 1
        safe_kind = self._safe_name(kind)
        safe_name = self._safe_name(name_hint)
        rel_path = Path("artifacts") / safe_kind / f"{self._artifact_counter:04d}_{safe_name}.json"
        path = self.root_dir / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
        digest = hash_jsonable(payload)
        return ArtifactRef(kind=kind, path=str(rel_path), hash=digest)

    def record(self, event: DebugTraceEvent) -> None:
        self._trace_file.write(json.dumps(event.to_dict(), sort_keys=True, default=str) + "\n")
        self._trace_file.flush()

    def close(self) -> None:
        if not self._trace_file.closed:
            self._trace_file.close()

    def __enter__(self) -> "JsonlDebugTraceRecorder":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
