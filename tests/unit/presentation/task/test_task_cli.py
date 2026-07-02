from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from streammuse.application.tasks import TaskRunResult
from streammuse.presentation.task.cli import main


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
                "--output-dir",
                str(tmp_path),
            ]
        )

    assert exit_code == 0
    assert run_task.call_args.kwargs["task_name"] == "zip_zap_zop"
    assert run_task.call_args.kwargs["runner_kind"] == "offline_benchmark"
    assert run_task.call_args.kwargs["max_turns"] == 2
    assert "offline_benchmark completed: 2 turns" in capsys.readouterr().out


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
