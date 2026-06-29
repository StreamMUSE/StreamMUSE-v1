"""Read-only web viewer for replay debugger traces."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles


STATIC_DIR = Path(__file__).parent / "static"


def create_app(*, trace_dir: str | Path | None = None) -> FastAPI:
    root = Path(trace_dir).expanduser().resolve() if trace_dir is not None else None
    app = FastAPI(title="StreamMUSE Replay Debugger")

    if STATIC_DIR.exists():
        css_dir = STATIC_DIR / "css"
        js_dir = STATIC_DIR / "js"
        if css_dir.exists():
            app.mount("/css", StaticFiles(directory=str(css_dir)), name="debug-css")
        if js_dir.exists():
            app.mount("/js", StaticFiles(directory=str(js_dir)), name="debug-js")

    @app.get("/")
    async def index() -> FileResponse:
        path = STATIC_DIR / "index.html"
        if not path.exists():
            raise HTTPException(status_code=500, detail="debug static/index.html missing")
        return FileResponse(str(path))

    @app.get("/favicon.ico")
    async def favicon() -> Response:
        return Response(status_code=204)

    @app.get("/api/trace")
    async def api_trace() -> dict[str, Any]:
        if root is None:
            raise HTTPException(status_code=400, detail="trace_dir was not configured")
        return _load_trace_directory(root)

    @app.get("/artifact/{artifact_path:path}")
    async def artifact(artifact_path: str) -> FileResponse:
        if root is None:
            raise HTTPException(status_code=400, detail="trace_dir was not configured")
        path = (root / artifact_path).resolve()
        if not str(path).startswith(str(root)) or not path.exists():
            raise HTTPException(status_code=404, detail="artifact not found")
        return FileResponse(str(path))

    return app


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[Any]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _load_trace_directory(root: Path) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise HTTPException(status_code=404, detail="trace directory is missing manifest.json")

    comparison_path = root / "comparison.json"
    trace_path = root / "trace.jsonl"
    if trace_path.exists():
        return {
            "manifest": _read_json(manifest_path),
            "trace": _read_jsonl(trace_path),
            "comparison": _read_json(comparison_path) if comparison_path.exists() else None,
        }

    child_trace_paths = [
        root / "offline_direct" / "trace.jsonl",
        root / "realtime_sim" / "trace.jsonl",
    ]
    traces: list[Any] = []
    for child_trace_path in child_trace_paths:
        if child_trace_path.exists():
            traces.extend(_read_jsonl(child_trace_path))
    if not traces:
        raise HTTPException(status_code=404, detail="trace directory is missing trace.jsonl files")
    comparison = _read_json(comparison_path) if comparison_path.exists() else None
    return {
        "manifest": _read_json(manifest_path),
        "trace": traces,
        "comparison": _prefix_replay_artifact_paths(comparison) if comparison else None,
    }


def _prefix_replay_artifact_paths(comparison: dict[str, Any]) -> dict[str, Any]:
    normalized = deepcopy(comparison)
    left_runner = normalized.get("left_runner", "offline_direct")
    right_runner = normalized.get("right_runner", "realtime_sim")
    for stage in normalized.get("stages", []):
        _prefix_refs(stage.get("left_refs", []), left_runner)
        _prefix_refs(stage.get("right_refs", []), right_runner)
    return normalized


def _prefix_refs(refs: list[dict[str, Any]], runner: str) -> None:
    for ref in refs:
        path = ref.get("path")
        if not isinstance(path, str) or not path:
            continue
        if path.startswith("offline_direct/") or path.startswith("realtime_sim/"):
            continue
        ref["path"] = f"{runner}/{path}"


def main() -> int:
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Serve a StreamMUSE debug trace viewer")
    parser.add_argument("--trace-dir", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8011)
    args = parser.parse_args()
    uvicorn.run(create_app(trace_dir=args.trace_dir), host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
