from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from streammuse.application.tasks import InteractiveTaskRunResult, TaskRunResult
from streammuse.presentation.task.cli import build_parser, main


def test_task_cli_runs_zip_zap_zop_benchmark(tmp_path, capsys) -> None:
    result = TaskRunResult(
        output_dir=str(tmp_path / "run"),
        runner_kind="offline_benchmark",
        task_name="zip_zap_zop",
        turn_count=2,
        valid_count=2,
        invalid_count=0,
        deadline_miss_count=0,
    )

    with patch("streammuse.presentation.task.cli.run_task", return_value=result) as run_task:
        exit_code = main(
            [
                "run",
                "--task",
                "zip_zap_zop",
                "--runner",
                "offline_benchmark",
                "--max-turns",
                "2",
                "--model-url",
                "http://localhost:8000/v1",
                "--model",
                "gemma",
                "--top-p",
                "0.8",
                "--output-dir",
                str(tmp_path),
            ]
        )

    assert exit_code == 0
    assert run_task.call_args.kwargs["task_name"] == "zip_zap_zop"
    assert run_task.call_args.kwargs["runner_kind"] == "offline_benchmark"
    assert run_task.call_args.kwargs["max_turns"] == 2
    assert run_task.call_args.kwargs["top_p"] == 0.8
    assert "offline_benchmark completed: 2 turns" in capsys.readouterr().out


def test_task_cli_plays_zip_zap_zop_interactive(tmp_path, capsys) -> None:
    result = InteractiveTaskRunResult(
        output_dir=str(tmp_path / "play"),
        task_name="zip_zap_zop",
        turn_count=2,
        human_turn_count=1,
        llm_turn_count=1,
        valid_count=2,
        invalid_count=0,
        deadline_miss_count=0,
    )

    with patch("streammuse.presentation.task.cli.play_task", return_value=result) as play_task:
        exit_code = main(
            [
                "play",
                "--task",
                "zip_zap_zop",
                "--llm-first",
                "--show-expected",
                "--history-limit",
                "3",
                "--deadline-mode",
                "challenge",
                "--challenge-stage-turns",
                "4",
                "--challenge-deadline-ms-list",
                "1000,500",
                "--max-turns",
                "2",
                "--model-url",
                "http://localhost:8000/v1",
                "--model",
                "gemma",
                "--timeout-s",
                "5",
                "--output-dir",
                str(tmp_path),
            ]
        )

    assert exit_code == 0
    assert play_task.call_args.kwargs["task_name"] == "zip_zap_zop"
    assert play_task.call_args.kwargs["human_first"] is False
    assert play_task.call_args.kwargs["show_expected"] is True
    assert play_task.call_args.kwargs["history_limit"] == 3
    assert play_task.call_args.kwargs["timeout_s"] == 5.0
    assert play_task.call_args.kwargs["deadline_mode"] == "challenge"
    assert play_task.call_args.kwargs["challenge_stage_turns"] == 4
    assert play_task.call_args.kwargs["challenge_deadline_ms_list"] == (1000.0, 500.0)
    assert "interactive completed: 2 turns" in capsys.readouterr().out


def test_task_cli_play_parser_has_common_args_and_actor_switch(tmp_path) -> None:
    args = build_parser().parse_args(
        [
            "play",
            "--task",
            "zip_zap_zop",
            "--llm-first",
            "--max-turns",
            "7",
            "--deadline-mode",
            "hard",
            "--challenge-stage-turns",
            "9",
            "--challenge-deadline-ms-list",
            "9000,3000",
            "--temperature",
            "0.1",
            "--deadline-ms",
            "3000",
            "--timeout-s",
            "9",
            "--start-number",
            "3",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert args.command == "play"
    assert args.task == "zip_zap_zop"
    assert args.human_first is False
    assert args.max_turns == 7
    assert args.max_tokens == 8
    assert args.deadline_mode == "hard"
    assert args.challenge_stage_turns == 9
    assert args.challenge_deadline_ms_list == (9000.0, 3000.0)
    assert args.temperature == 0.1
    assert args.deadline_ms == 3000.0
    assert args.timeout_s == 9.0
    assert args.start_number == 3


def test_task_cli_play_defaults_to_menu_deadline_mode() -> None:
    args = build_parser().parse_args(["play", "--task", "zip_zap_zop"])

    assert args.deadline_mode == "menu"
    assert args.challenge_stage_turns == 20
    assert args.challenge_deadline_ms_list == (10000.0, 5000.0, 3000.0, 2000.0, 1000.0)


def test_task_cli_run_keeps_batch_max_tokens_default() -> None:
    args = build_parser().parse_args(["run", "--task", "zip_zap_zop"])

    assert args.command == "run"
    assert args.max_tokens == 32
    assert args.top_p is None


def test_task_cli_play_rejects_conflicting_first_actor_flags() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["play", "--task", "zip_zap_zop", "--human-first", "--llm-first"])


def test_task_cli_rejects_unknown_task(tmp_path) -> None:
    with patch("streammuse.presentation.task.cli.run_task") as run_task:
        exit_code = main(
            [
                "run",
                "--task",
                "unknown",
                "--runner",
                "offline_benchmark",
                "--output-dir",
                str(Path(tmp_path)),
            ]
        )

    assert exit_code == 2
    run_task.assert_not_called()
