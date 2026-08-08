"""Tests for standalone rap-demo assembly and CLI lifecycle."""

import json
from pathlib import Path

import pytest

from streammuse.presentation.rap_demo.cli import _repository_state, build_demo, build_parser, main


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
    assert args.terminal_layout == "auto"


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
    assert "[SESSION][START]" in output
    assert "[BAR 01][SELECT] frozen" in output
    assert "[BAR 01][PLAY] syllable" in output
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
    events = [json.loads(line) for line in (demo.session_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    started = next(event for event in events if event["event_type"] == "session_started")
    assert started["payload"]["generator_config"] == manifest["generator_config"]
    assert started["payload"]["model_config"] == manifest["model_config"]
    assert manifest["generator_config"]["output_length_policy"] is None


def test_build_demo_records_resolved_terminal_layout(tmp_path: Path) -> None:
    args = build_parser().parse_args(
        [
            "--terminal-layout",
            "stream",
            "--max-bars",
            "1",
            "--log-dir",
            str(tmp_path),
        ]
    )
    demo = build_demo(args)
    try:
        manifest = json.loads((demo.session_dir / "session.json").read_text(encoding="utf-8"))
        assert demo.session_metadata["terminal_layout"] == "stream"
        assert demo.session_metadata["tempo_bpm"] == 92.0
        assert manifest["terminal_layout"] == "stream"
    finally:
        demo.close()


def test_main_runs_assembled_demo_and_treats_zero_bars_as_unbounded(monkeypatch) -> None:
    calls = []

    class FakeDemo:
        def run(self, *, max_bars: int) -> None:
            calls.append(max_bars)

    monkeypatch.setattr("streammuse.presentation.rap_demo.cli.build_demo", lambda args: FakeDemo())

    assert main(["--max-bars", "0", "--no-web"]) == 0
    assert calls == [0]


def test_main_serves_web_monitor_with_the_assembled_runtime(monkeypatch) -> None:
    calls: list[object] = []

    class FakeDemo:
        projector = object()
        websocket_queue = object()

    demo = FakeDemo()
    app = object()
    monkeypatch.setattr("streammuse.presentation.rap_demo.cli.build_demo", lambda _args: demo)
    monkeypatch.setattr(
        "streammuse.presentation.rap_demo.cli.create_app",
        lambda **kwargs: calls.append(kwargs) or app,
    )
    monkeypatch.setattr(
        "streammuse.presentation.rap_demo.cli.uvicorn.run",
        lambda actual_app, **kwargs: calls.append((actual_app, kwargs)),
    )

    assert main(["--host", "0.0.0.0", "--port", "8123", "--max-bars", "1"]) == 0
    assert calls[0] == {"runtime": demo, "projector": demo.projector, "websocket_queue": demo.websocket_queue}
    assert calls[1] == (app, {"host": "0.0.0.0", "port": 8123, "log_level": "warning"})


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


def test_build_demo_rolls_back_dispatcher_recorder_and_generator_on_late_assembly_failure(
    tmp_path: Path, monkeypatch
) -> None:
    calls: list[str] = []

    class Recorder:
        def __init__(self, *_args, **_kwargs) -> None:
            calls.append("recorder_created")

        def __call__(self, _event) -> None:
            return None

        def close(self) -> None:
            calls.append("recorder_closed")

    class Dispatcher:
        def __init__(self, *_args, **_kwargs) -> None:
            calls.append("dispatcher_created")

        def start(self) -> None:
            calls.append("dispatcher_started")

        def flush_and_close(self) -> None:
            calls.append("dispatcher_closed")

    monkeypatch.setattr("streammuse.presentation.rap_demo.cli.RapSessionRecorder", Recorder)
    monkeypatch.setattr("streammuse.presentation.rap_demo.cli.RapEventDispatcher", Dispatcher)
    monkeypatch.setattr(
        "streammuse.presentation.rap_demo.cli._build_generator",
        lambda _args: (
            object(),
            lambda: calls.append("generator_stopped"),
            lambda: calls.append("generator_closed"),
        ),
    )
    monkeypatch.setattr(
        "streammuse.presentation.rap_demo.cli.RollingRapController",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("controller failed")),
    )
    args = build_parser().parse_args(["--max-bars", "1", "--log-dir", str(tmp_path)])

    with pytest.raises(RuntimeError, match="controller failed"):
        build_demo(args)

    assert calls[-4:] == ["dispatcher_closed", "recorder_closed", "generator_stopped", "generator_closed"]


def test_repository_discovery_failure_is_never_reported_as_clean(monkeypatch) -> None:
    monkeypatch.setattr(
        "streammuse.presentation.rap_demo.cli.subprocess.run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("git unavailable")),
    )

    assert _repository_state() == ("unknown", True, False)
