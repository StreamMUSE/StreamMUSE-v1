"""Tests for standalone rap-demo assembly and CLI lifecycle."""

import json
from pathlib import Path

import pytest

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
    assert manifest["generator_config"]["name"] == "scripted_failure"
    assert manifest["model_config"]["name"] == "none"
    assert manifest["templates"][0]["definition"]["slots"]
    assert (demo.session_dir / "events.jsonl").is_file()
    assert (demo.session_dir / "summary.json").is_file()
    assert (demo.session_dir / "bars.csv").is_file()
    summary = json.loads((demo.session_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["bars"]["frozen"] == 1


def test_main_runs_assembled_demo_and_treats_zero_bars_as_unbounded(monkeypatch) -> None:
    calls = []

    class FakeDemo:
        def run(self, *, max_bars: int) -> None:
            calls.append(max_bars)

    monkeypatch.setattr("streammuse.presentation.rap_demo.cli.build_demo", lambda args: FakeDemo())

    assert main(["--max-bars", "0", "--no-web"]) == 0
    assert calls == [0]


@pytest.mark.parametrize("max_bars", (0, 2))
def test_non_looping_scenario_rejects_runs_past_its_schedule(tmp_path: Path, max_bars: int) -> None:
    scenario_path = tmp_path / "scenario.json"
    scenario_path.write_text(
        json.dumps(
            {
                "scenario_id": "finite",
                "tempo_bpm": 92.0,
                "loop": False,
                "segments": [
                    {
                        "start_bar": 0,
                        "bars": 1,
                        "topic": "space",
                        "template_id": "baseline_syncopated_9",
                        "fallback_lines": ["space dreams rise while bright stars cross dark night"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    args = build_parser().parse_args(
        ["--scenario", str(scenario_path), "--max-bars", str(max_bars), "--log-dir", str(tmp_path / "logs")]
    )

    with pytest.raises(ValueError, match="max-bars"):
        build_demo(args)
