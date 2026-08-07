"""Tests for standalone rap-demo assembly and CLI lifecycle."""

import json
from pathlib import Path

from streammuse.presentation.rap_demo.cli import build_demo, build_parser, main


class FakeClock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


def test_parser_defaults_to_showcase_local_endpoint_and_model() -> None:
    args = build_parser().parse_args([])

    assert args.model_url == "http://127.0.0.1:8001/v1"
    assert args.model == "qwen-rap"
    assert args.generator == "phrase_bank"
    assert args.no_web is False


def test_build_demo_prevalidates_fallbacks_and_runs_finite_terminal_session(tmp_path: Path, capsys) -> None:
    args = build_parser().parse_args(
        [
            "--generator",
            "scripted_failure",
            "--max-bars",
            "1",
            "--terminal-detail",
            "full",
            "--log-dir",
            str(tmp_path),
            "--no-web",
        ]
    )
    clock = FakeClock()
    demo = build_demo(args, clock=clock, sleep=clock.sleep)

    demo.run(max_bars=args.max_bars)

    output = capsys.readouterr().out
    assert "SESSION" in output
    assert "FREEZE" in output
    assert "SYLLABLE" in output
    manifest = json.loads((demo.session_dir / "session.json").read_text(encoding="utf-8"))
    assert manifest["scenario_id"] == "default_research_demo"
    assert manifest["generator"] == "scripted_failure"


def test_main_runs_assembled_demo_and_treats_zero_bars_as_unbounded(monkeypatch) -> None:
    calls = []

    class FakeDemo:
        def run(self, *, max_bars: int) -> None:
            calls.append(max_bars)

    monkeypatch.setattr("streammuse.presentation.rap_demo.cli.build_demo", lambda args: FakeDemo())

    assert main(["--max-bars", "0", "--no-web"]) == 0
    assert calls == [0]
