"""Focused checks for the server-only benchmark harness."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location("moss_bench", ROOT / "scripts/benchmark_moss_server.py")
bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench)


def test_corpus_has_twenty_valid_two_bar_requests(tmp_path):
    reference = tmp_path / "reference.wav"
    reference.write_bytes(b"reference identity only")
    args = SimpleNamespace(root=tmp_path, reference=reference, model=tmp_path / ("a" * 40),
                           warmups=3, gpu=0)
    bench.prepare(args)
    requests = bench.load_requests(tmp_path / "corpus.json")
    assert len(requests) == 20
    assert len({r.text for r in requests}) == 20
    assert all(len(r.syllables) == 18 for r in requests)
    assert all(r.duration_seconds == pytest.approx(16 / 3) for r in requests)
    assert all(len({s.tick_in_chunk for s in r.syllables}) == 18 for r in requests)
    before = (tmp_path / "corpus.json").read_bytes()
    bench.prepare(args)
    assert (tmp_path / "corpus.json").read_bytes() == before


def test_different_evidence_cannot_overwrite(tmp_path):
    path = tmp_path / "evidence.json"
    bench.write_json(path, {"pin": "first"})
    with pytest.raises(FileExistsError):
        bench.write_json(path, {"pin": "changed"})
    assert json.loads(path.read_text()) == {"pin": "first"}


def test_summary_excludes_warmups_and_failed_totals(tmp_path):
    for label in ("A1", "B1", "B2", "A2"):
        folder = tmp_path / label
        folder.mkdir()
        backend = "inprocess" if label.startswith("A") else "sglang-omni"
        row = {"block": label, "backend": backend, "sample": 0, "warmup": False,
               "success": True, "moss_ms": 200 if label.startswith("A") else 100,
               "total_ms": 250 if label.startswith("A") else 150, "source_audio": None}
        warm = {**row, "warmup": True, "total_ms": 999999, "moss_ms": 999999}
        failed = {**row, "sample": 1, "success": False, "total_ms": 1}
        (folder / "records.jsonl").write_text("\n".join(json.dumps(x) for x in (row, warm, failed)))
    bench.summarize(SimpleNamespace(root=tmp_path))
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["backends"]["inprocess"]["n"] == 4
    assert summary["backends"]["inprocess"]["total_ms"]["median"] == 250
    assert summary["paired"]["moss_ms"][0]["speedup"] == 2
    assert len(summary["paired"]["moss_ms"]) == 1
