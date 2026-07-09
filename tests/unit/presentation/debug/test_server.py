from __future__ import annotations

import json

from fastapi.testclient import TestClient

from streammuse.presentation.debug.server import create_app


def test_debug_viewer_serves_static_app(tmp_path) -> None:
    app = create_app(trace_dir=tmp_path)

    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert "StreamMUSE Replay Debugger" in response.text


def test_debug_viewer_includes_human_orientation_sections(tmp_path) -> None:
    app = create_app(trace_dir=tmp_path)

    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert "What this screen is proving" in response.text
    assert "Same melody" in response.text
    assert "Offline direct" in response.text
    assert "Realtime simulation" in response.text
    assert "How to read this" in response.text
    assert "If a stage turns red" in response.text


def test_debug_viewer_loads_trace_directory(tmp_path) -> None:
    (tmp_path / "manifest.json").write_text(json.dumps({"scenario": "x"}), encoding="utf-8")
    (tmp_path / "trace.jsonl").write_text(
        json.dumps({"stage": "input_midi_loaded"}) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "comparison.json").write_text(
        json.dumps({"first_mismatch_stage": "prompt_tokenization"}),
        encoding="utf-8",
    )
    app = create_app(trace_dir=tmp_path)

    with TestClient(app) as client:
        response = client.get("/api/trace")

    assert response.status_code == 200
    payload = response.json()
    assert payload["manifest"]["scenario"] == "x"
    assert payload["trace"][0]["stage"] == "input_midi_loaded"
    assert payload["comparison"]["first_mismatch_stage"] == "prompt_tokenization"


def test_debug_viewer_loads_replay_comparison_directory(tmp_path) -> None:
    (tmp_path / "manifest.json").write_text(
        json.dumps({"scenario": "lekai-prompt-continuation", "runs": {}}),
        encoding="utf-8",
    )
    (tmp_path / "comparison.json").write_text(
        json.dumps({"first_mismatch_stage": None}),
        encoding="utf-8",
    )
    offline = tmp_path / "offline_direct"
    sim = tmp_path / "realtime_sim"
    offline.mkdir()
    sim.mkdir()
    (offline / "trace.jsonl").write_text(
        json.dumps({"stage": "input_midi_loaded", "runner_kind": "offline_direct"}) + "\n",
        encoding="utf-8",
    )
    (sim / "trace.jsonl").write_text(
        json.dumps({"stage": "input_midi_loaded", "runner_kind": "realtime_sim"}) + "\n",
        encoding="utf-8",
    )
    app = create_app(trace_dir=tmp_path)

    with TestClient(app) as client:
        response = client.get("/api/trace")

    assert response.status_code == 200
    payload = response.json()
    assert [row["runner_kind"] for row in payload["trace"]] == ["offline_direct", "realtime_sim"]
    assert payload["comparison"]["first_mismatch_stage"] is None


def test_debug_viewer_prefixes_replay_artifact_paths_by_runner(tmp_path) -> None:
    (tmp_path / "manifest.json").write_text(
        json.dumps({"scenario": "lekai-prompt-continuation", "runs": {}}),
        encoding="utf-8",
    )
    (tmp_path / "comparison.json").write_text(
        json.dumps(
            {
                "left_runner": "offline_direct",
                "right_runner": "realtime_sim",
                "stages": [
                    {
                        "stage": "normalized_melody_events",
                        "match": True,
                        "left_refs": [{"kind": "events", "path": "artifacts/events/events.json"}],
                        "right_refs": [{"kind": "events", "path": "artifacts/events/events.json"}],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    offline = tmp_path / "offline_direct"
    sim = tmp_path / "realtime_sim"
    (offline / "artifacts" / "events").mkdir(parents=True)
    (sim / "artifacts" / "events").mkdir(parents=True)
    (offline / "trace.jsonl").write_text(
        json.dumps({"stage": "normalized_melody_events", "runner_kind": "offline_direct"}) + "\n",
        encoding="utf-8",
    )
    (sim / "trace.jsonl").write_text(
        json.dumps({"stage": "normalized_melody_events", "runner_kind": "realtime_sim"}) + "\n",
        encoding="utf-8",
    )
    (offline / "artifacts" / "events" / "events.json").write_text("[1]", encoding="utf-8")
    (sim / "artifacts" / "events" / "events.json").write_text("[2]", encoding="utf-8")
    app = create_app(trace_dir=tmp_path)

    with TestClient(app) as client:
        trace_response = client.get("/api/trace")
        offline_artifact = client.get("/artifact/offline_direct/artifacts/events/events.json")
        sim_artifact = client.get("/artifact/realtime_sim/artifacts/events/events.json")

    assert trace_response.status_code == 200
    stage = trace_response.json()["comparison"]["stages"][0]
    assert stage["left_refs"][0]["path"] == "offline_direct/artifacts/events/events.json"
    assert stage["right_refs"][0]["path"] == "realtime_sim/artifacts/events/events.json"
    assert offline_artifact.json() == [1]
    assert sim_artifact.json() == [2]


def test_debug_viewer_reports_missing_trace(tmp_path) -> None:
    app = create_app(trace_dir=tmp_path)

    with TestClient(app) as client:
        response = client.get("/api/trace")

    assert response.status_code == 404
