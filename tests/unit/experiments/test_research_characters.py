from __future__ import annotations

import json

import httpx

from scripts.experiments.research_day import characters, concurrent_counting
from streammuse.application.rap.characters import CharacterExperiment
from streammuse.infrastructure.inference.local_chat_client import LocalChatModelClient


def install_model(monkeypatch, handler):
    monkeypatch.setattr(characters, "CharacterExperiment", lambda config: CharacterExperiment(
        config, client_factory=lambda cfg: LocalChatModelClient(cfg, transport=httpx.MockTransport(handler)),
    ))


def reply(text):
    return httpx.Response(200, json={"choices": [{"message": {"content": text}}]})


def test_dialogue_cli_writes_separate_proposal_accepted_and_manifest_artifacts(tmp_path, monkeypatch):
    install_model(monkeypatch, lambda request: reply("hello"))
    assert characters.main(["--turns", "8", "--output-dir", str(tmp_path)]) == 0
    run_dir = next(path for path in tmp_path.iterdir() if path.is_dir())
    summary = json.loads((run_dir / "summary.json").read_text())
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert summary["state"] == "completed"
    assert manifest["real_model_required"] is True
    assert len((run_dir / "accepted.jsonl").read_text().splitlines()) == 8
    assert len((run_dir / "proposals.jsonl").read_text().splitlines()) == 8
    assert (run_dir / "events.jsonl").exists()


def test_counting_cli_compares_all_policies_without_hiding_failed_naive(tmp_path, monkeypatch):
    def handler(request):
        payload = json.loads(request.content)
        obs = json.loads(payload["messages"][-1]["content"])
        if obs["accepted"] and obs["accepted"][-1]["actor_id"] == obs["actor_id"]:
            return reply("WAIT")
        return reply(str(obs["version"] + 1))

    install_model(monkeypatch, handler)
    assert concurrent_counting.main(["--policy", "all", "--trials", "1", "--output-dir", str(tmp_path)]) == 1
    comparison = json.loads((tmp_path / "comparison.json").read_text())
    assert len(comparison["runs"]) == 3
    results = {run["policy"]: run for run in comparison["runs"]}
    assert results["fixed_turn"]["state"] == "completed"
    assert results["coordinated"]["state"] == "completed"
    assert results["naive"]["state"] == "failed"
    assert all(run["seed"] == 0 for run in comparison["runs"])


def test_failed_cli_keeps_failure_artifacts_and_nonzero_exit(tmp_path, monkeypatch):
    def handler(request):
        raise httpx.ConnectError("unavailable", request=request)

    install_model(monkeypatch, handler)
    assert characters.main(["--output-dir", str(tmp_path)]) == 1
    run_dir = next(path for path in tmp_path.iterdir() if path.is_dir())
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["state"] == "failed"
    assert summary["error"]
    assert (run_dir / "accepted.jsonl").read_text() == ""
    assert len((run_dir / "proposals.jsonl").read_text().splitlines()) == 1
