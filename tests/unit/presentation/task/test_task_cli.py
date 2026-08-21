from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from streammuse.application.tasks import InteractiveTaskRunResult, TaskRunResult, TaskWebConfig
from streammuse.domain.tasks import AnimalNamingTask, ZipZapZopTask
from streammuse.infrastructure.voice import MicrophoneDevice, VoiceDependencyError
from streammuse.presentation.task.cli import build_parser, create_task, main, play_task


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
                "--live-output",
                "--output-dir",
                str(tmp_path),
            ]
        )

    assert exit_code == 0
    assert run_task.call_args.kwargs["task_name"] == "zip_zap_zop"
    assert run_task.call_args.kwargs["runner_kind"] == "offline_benchmark"
    assert run_task.call_args.kwargs["max_turns"] == 2
    assert run_task.call_args.kwargs["top_p"] == 0.8
    assert run_task.call_args.kwargs["live_output"] is True
    assert "offline_benchmark completed: 2 turns" in capsys.readouterr().out


def test_task_cli_runs_animal_naming_benchmark(tmp_path, capsys) -> None:
    result = TaskRunResult(
        output_dir=str(tmp_path / "run"),
        runner_kind="offline_benchmark",
        task_name="animal_naming",
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
                "animal_naming",
                "--runner",
                "offline_benchmark",
                "--max-turns",
                "2",
                "--output-dir",
                str(tmp_path),
            ]
        )

    assert exit_code == 0
    assert run_task.call_args.kwargs["task_name"] == "animal_naming"
    assert run_task.call_args.kwargs["runner_kind"] == "offline_benchmark"
    assert run_task.call_args.kwargs["live_output"] is False
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


def test_task_cli_parses_web_ui_options_and_rejects_orphans(capsys) -> None:
    args = build_parser().parse_args(
        ["play", "--task", "zip_zap_zop", "--web-ui", "--web-port", "9012"]
    )
    assert args.web_ui is True
    assert args.web_port == 9012

    with patch("streammuse.presentation.task.cli.play_task") as play:
        exit_code = main(
            ["play", "--task", "zip_zap_zop", "--web-port", "9012"]
        )
    assert exit_code == 2
    play.assert_not_called()
    assert "require --web-ui" in capsys.readouterr().out


def test_task_web_gate_runs_before_game_resource_construction(tmp_path) -> None:
    run_dir = tmp_path / "web-gate"
    order: list[str] = []

    class FakeWebServer:
        url = "http://127.0.0.1:8002/?token=test"

        def __init__(self, **kwargs: object) -> None:
            del kwargs

        def start(self) -> None:
            order.append("server_start")

        def wait_for_viewer(self) -> None:
            order.append("viewer_ready")

        def close(self) -> None:
            order.append("server_close")

    def create_source(*args: object, **kwargs: object) -> _TrackingSource:
        del args, kwargs
        order.append("source_create")
        return _TrackingSource()

    with (
        patch("streammuse.presentation.task.cli._new_run_dir", return_value=run_dir),
        patch("streammuse.presentation.task_web.TaskWebServer", FakeWebServer),
        patch(
            "streammuse.application.factories.human_input_factory.HumanInputFactory.create",
            side_effect=create_source,
        ),
        patch(
            "streammuse.application.factories.speech_output_factory.SpeechOutputFactory.create",
            side_effect=RuntimeError("stop after gate"),
        ),
        pytest.raises(RuntimeError, match="stop after gate"),
    ):
        play_task(
            task_name="zip_zap_zop",
            max_turns=1,
            model_url="http://localhost:8000/v1",
            model="fake",
            timeout_s=1.0,
            max_tokens=8,
            temperature=0.0,
            deadline_ms=1000.0,
            output_dir=str(tmp_path),
            deadline_mode="soft",
            task_web_config=TaskWebConfig(enabled=True),
        )

    assert order.index("viewer_ready") < order.index("source_create")


def test_task_web_wait_interrupt_writes_manifest_without_constructing_game_resources(
    tmp_path, capsys
) -> None:
    run_dir = tmp_path / "web-interrupt"

    class InterruptingWebServer:
        url = "http://127.0.0.1:8002/?token=test"

        def __init__(self, **kwargs: object) -> None:
            del kwargs

        def start(self) -> None:
            return None

        def wait_for_viewer(self) -> None:
            raise KeyboardInterrupt

        def close(self) -> None:
            return None

    with (
        patch("streammuse.presentation.task.cli._new_run_dir", return_value=run_dir),
        patch("streammuse.presentation.task_web.TaskWebServer", InterruptingWebServer),
        patch(
            "streammuse.application.factories.human_input_factory.HumanInputFactory.create"
        ) as create_source,
        patch("streammuse.presentation.task.cli._build_client") as build_client,
    ):
        exit_code = main(
            [
                "play",
                "--task",
                "zip_zap_zop",
                "--deadline-mode",
                "soft",
                "--web-ui",
                "--output-dir",
                str(tmp_path),
            ]
        )

    assert exit_code == 130
    create_source.assert_not_called()
    build_client.assert_not_called()
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "user_interrupt"
    assert manifest["task_web"]["enabled"] is True
    assert "interactive session interrupted" in capsys.readouterr().out


def test_task_cli_run_keeps_batch_max_tokens_default() -> None:
    args = build_parser().parse_args(["run", "--task", "zip_zap_zop"])

    assert args.command == "run"
    assert args.max_tokens == 32
    assert args.top_p is None


def test_task_cli_plays_animal_naming_interactive(tmp_path, capsys) -> None:
    result = InteractiveTaskRunResult(
        output_dir=str(tmp_path / "play"),
        task_name="animal_naming",
        turn_count=2,
        human_turn_count=1,
        llm_turn_count=1,
        valid_count=2,
        invalid_count=0,
        deadline_miss_count=0,
    )

    with patch(
        "streammuse.presentation.task.cli.play_task",
        return_value=result,
    ) as mocked_play_task:
        exit_code = main(
            [
                "play",
                "--task",
                "animal_naming",
                "--max-turns",
                "2",
                "--deadline-mode",
                "soft",
                "--output-dir",
                str(tmp_path),
            ]
        )

    assert exit_code == 0
    assert mocked_play_task.call_args.kwargs["task_name"] == "animal_naming"
    assert mocked_play_task.call_args.kwargs["max_turns"] == 2
    assert mocked_play_task.call_args.kwargs["human_input_config"].mode == "terminal"
    assert "interactive completed: 2 turns" in capsys.readouterr().out


def test_task_cli_passes_voice_configuration_to_animal_naming(
    tmp_path,
    capsys,
) -> None:
    result = InteractiveTaskRunResult(
        output_dir=str(tmp_path / "play"),
        task_name="animal_naming",
        turn_count=0,
        human_turn_count=0,
        llm_turn_count=0,
        valid_count=0,
        invalid_count=0,
        deadline_miss_count=0,
    )

    with patch(
        "streammuse.presentation.task.cli.play_task",
        return_value=result,
    ) as mocked_play_task:
        exit_code = main(
            [
                "play",
                "--task",
                "animal_naming",
                "--human-input",
                "voice",
                "--voice-max-utterance-ms",
                "1500",
                "--voice-local-files-only",
            ]
        )

    assert exit_code == 0
    config = mocked_play_task.call_args.kwargs["human_input_config"]
    assert config.mode == "voice"
    assert config.voice.model == "tiny.en"
    assert config.voice.compute_type == "int8"
    assert config.voice.max_utterance_ms == 1500.0
    assert config.voice.local_files_only is True
    assert "interactive completed" in capsys.readouterr().out


def test_task_cli_passes_bidirectional_voice_configuration_to_animal_naming(
    tmp_path,
) -> None:
    result = InteractiveTaskRunResult(
        output_dir=str(tmp_path / "play"),
        task_name="animal_naming",
        turn_count=0,
        human_turn_count=0,
        llm_turn_count=0,
        valid_count=0,
        invalid_count=0,
        deadline_miss_count=0,
    )

    with patch(
        "streammuse.presentation.task.cli.play_task",
        return_value=result,
    ) as mocked_play_task:
        exit_code = main(
            [
                "play",
                "--task",
                "animal_naming",
                "--human-input",
                "voice",
                "--voice-max-utterance-ms",
                "1000",
                "--speech-output",
                "audio",
                "--speech-backend",
                "null",
                "--llm-deadline-basis",
                "audio_end",
                "--model-url",
                "http://127.0.0.1:8101/v1",
                "--model",
                "Qwen/Qwen3.6-27B",
                "--deadline-mode",
                "soft",
                "--deadline-ms",
                "5000",
            ]
        )

    assert exit_code == 0
    kwargs = mocked_play_task.call_args.kwargs
    assert kwargs["task_name"] == "animal_naming"
    assert kwargs["human_input_config"].mode == "voice"
    assert kwargs["human_input_config"].voice.max_utterance_ms == 1000.0
    assert kwargs["speech_output_config"].mode == "audio"
    assert kwargs["speech_output_config"].backend == "null"
    assert kwargs["speech_output_config"].llm_deadline_basis == "audio_end"
    assert kwargs["model_url"] == "http://127.0.0.1:8101/v1"
    assert kwargs["model"] == "Qwen/Qwen3.6-27B"
    assert kwargs["deadline_ms"] == 5000.0


def test_animal_naming_task_ignores_zip_zap_zop_start_number() -> None:
    default_task = create_task("animal_naming", start_number=1, history_limit=6)
    offset_task = create_task("animal_naming", start_number=999, history_limit=6)

    assert isinstance(default_task, AnimalNamingTask)
    assert isinstance(offset_task, AnimalNamingTask)
    assert default_task.initial_state() == offset_task.initial_state()


@pytest.mark.parametrize("max_turns", ["0", "-1"])
def test_task_cli_rejects_nonpositive_interactive_turn_count(
    max_turns: str,
    capsys,
) -> None:
    with patch("streammuse.presentation.task.cli.play_task") as mocked_play_task:
        exit_code = main(
            [
                "play",
                "--task",
                "animal_naming",
                "--max-turns",
                max_turns,
            ]
        )

    assert exit_code == 2
    mocked_play_task.assert_not_called()
    assert "max_turns must be > 0" in capsys.readouterr().out


def test_task_cli_rejects_animal_naming_turns_above_whitelist_size(
    capsys,
) -> None:
    with patch("streammuse.presentation.task.cli.play_task") as mocked_play_task:
        exit_code = main(
            [
                "play",
                "--task",
                "animal_naming",
                "--max-turns",
                str(len(AnimalNamingTask().animals) + 1),
            ]
        )

    assert exit_code == 2
    mocked_play_task.assert_not_called()
    assert "cannot exceed the whitelist size (91)" in capsys.readouterr().out


@pytest.mark.parametrize("max_turns", [1, len(AnimalNamingTask().animals)])
def test_task_cli_accepts_animal_naming_turn_boundaries(
    max_turns: int,
    tmp_path,
) -> None:
    result = InteractiveTaskRunResult(
        output_dir=str(tmp_path / "play"),
        task_name="animal_naming",
        turn_count=0,
        human_turn_count=0,
        llm_turn_count=0,
        valid_count=0,
        invalid_count=0,
        deadline_miss_count=0,
    )
    with patch(
        "streammuse.presentation.task.cli.play_task",
        return_value=result,
    ) as mocked_play_task:
        exit_code = main(
            [
                "play",
                "--task",
                "animal_naming",
                "--max-turns",
                str(max_turns),
                "--deadline-mode",
                "soft",
            ]
        )

    assert exit_code == 0
    assert mocked_play_task.call_args.kwargs["max_turns"] == max_turns


def test_play_task_direct_call_validates_animal_naming_turn_limit(
    tmp_path,
) -> None:
    with pytest.raises(ValueError, match="whitelist size"):
        play_task(
            task_name="animal_naming",
            max_turns=len(AnimalNamingTask().animals) + 1,
            model_url="http://localhost:8000/v1",
            model="fake",
            timeout_s=1.0,
            max_tokens=8,
            temperature=0.0,
            deadline_ms=1000.0,
            output_dir=str(tmp_path),
            deadline_mode="soft",
        )


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


def test_task_cli_voice_mode_passes_typed_configuration(tmp_path, capsys) -> None:
    result = InteractiveTaskRunResult(
        output_dir=str(tmp_path / "play"),
        task_name="zip_zap_zop",
        turn_count=0,
        human_turn_count=0,
        llm_turn_count=0,
        valid_count=0,
        invalid_count=0,
        deadline_miss_count=0,
    )

    with patch("streammuse.presentation.task.cli.play_task", return_value=result) as play_task:
        exit_code = main(
            [
                "play",
                "--task",
                "zip_zap_zop",
                "--human-input",
                "voice",
                "--microphone-device",
                "3",
                "--voice-model-cache",
                str(tmp_path / "models"),
                "--voice-model-revision",
                "model-commit",
                "--voice-local-files-only",
                "--voice-save-audio",
                "--voice-max-utterance-ms",
                "1000",
            ]
        )

    assert exit_code == 0
    config = play_task.call_args.kwargs["human_input_config"]
    assert config.mode == "voice"
    assert config.voice.model == "tiny.en"
    assert config.voice.device == "cpu"
    assert config.voice.compute_type == "int8"
    assert config.voice.microphone_device == 3
    assert config.voice.model_revision == "model-commit"
    assert config.voice.local_files_only is True
    assert config.voice.save_audio is True
    assert config.voice.max_utterance_ms == 1000.0
    assert "interactive completed" in capsys.readouterr().out


def test_task_cli_rejects_voice_options_in_terminal_mode(capsys) -> None:
    with patch("streammuse.presentation.task.cli.play_task") as play_task:
        exit_code = main(
            ["play", "--task", "zip_zap_zop", "--voice-model", "tiny.en"]
        )

    assert exit_code == 2
    play_task.assert_not_called()
    assert "voice options require --human-input voice" in capsys.readouterr().out


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "-inf"])
def test_task_cli_rejects_invalid_voice_max_utterance(value: str) -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            [
                "play",
                "--task",
                "zip_zap_zop",
                "--human-input",
                "voice",
                "--voice-max-utterance-ms",
                value,
            ]
        )


def test_voice_devices_needs_no_task_or_model_client(capsys) -> None:
    devices = (
        MicrophoneDevice(
            index=2,
            name="Built-in Microphone",
            max_input_channels=1,
            default_sample_rate_hz=48_000.0,
            hostapi=0,
        ),
    )

    with (
        patch("streammuse.infrastructure.voice.enumerate_input_devices", return_value=devices) as enumerate_devices,
        patch("streammuse.presentation.task.cli._build_client") as build_client,
    ):
        exit_code = main(["voice-devices"])

    assert exit_code == 0
    enumerate_devices.assert_called_once_with()
    build_client.assert_not_called()
    output = capsys.readouterr().out
    assert "[2] Built-in Microphone" in output
    assert "48000 Hz" in output


def test_voice_devices_reports_missing_optional_dependency_once(capsys) -> None:
    with patch(
        "streammuse.infrastructure.voice.enumerate_input_devices",
        side_effect=VoiceDependencyError("install with uv sync --extra voice"),
    ):
        exit_code = main(["voice-devices"])

    assert exit_code == 1
    output = capsys.readouterr().out
    assert output.count("voice input error:") == 1
    assert "uv sync --extra voice" in output


def test_task_cli_maps_keyboard_interrupt_to_exit_code_130(capsys) -> None:
    with patch("streammuse.presentation.task.cli.play_task", side_effect=KeyboardInterrupt):
        exit_code = main(["play", "--task", "zip_zap_zop"])

    assert exit_code == 130
    assert "interactive session interrupted" in capsys.readouterr().out


def test_task_cli_does_not_treat_runtime_value_error_as_configuration_error(capsys) -> None:
    with (
        patch("streammuse.presentation.task.cli.play_task", side_effect=ValueError("runtime failed")),
        pytest.raises(ValueError, match="runtime failed"),
    ):
        main(["play", "--task", "zip_zap_zop"])

    assert "invalid human input configuration" not in capsys.readouterr().out


def test_task_cli_rejects_explicit_empty_voice_model(capsys) -> None:
    with patch("streammuse.presentation.task.cli.play_task") as play_task:
        exit_code = main(
            [
                "play",
                "--task",
                "zip_zap_zop",
                "--human-input",
                "voice",
                "--voice-model",
                "",
            ]
        )

    assert exit_code == 2
    play_task.assert_not_called()
    assert "model must not be empty" in capsys.readouterr().out


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "-inf"])
def test_task_cli_rejects_non_positive_or_non_finite_deadline(value: str) -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            ["play", "--task", "zip_zap_zop", "--deadline-ms", value]
        )


@pytest.mark.parametrize("value", ["1000,nan", "1000,inf", "1000,0", "1000,-1"])
def test_task_cli_rejects_invalid_challenge_deadline_list(value: str) -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            [
                "play",
                "--task",
                "zip_zap_zop",
                "--challenge-deadline-ms-list",
                value,
            ]
        )


def test_task_cli_rejects_voice_game_outside_spoken_integer_range(capsys) -> None:
    with patch("streammuse.presentation.task.cli.play_task") as play_task:
        exit_code = main(
            [
                "play",
                "--task",
                "zip_zap_zop",
                "--human-input",
                "voice",
                "--start-number",
                str(ZipZapZopTask.max_spoken_integer),
                "--max-turns",
                "2",
            ]
        )

    assert exit_code == 2
    play_task.assert_not_called()
    assert "spoken integer range" in capsys.readouterr().out


def test_play_task_source_construction_failure_writes_startup_manifest(tmp_path) -> None:
    run_dir = tmp_path / "failed-source"

    with (
        patch("streammuse.presentation.task.cli._new_run_dir", return_value=run_dir),
        patch(
            "streammuse.application.factories.human_input_factory.HumanInputFactory.create",
            side_effect=RuntimeError("source construction failed"),
        ),
        patch("streammuse.presentation.task.cli._build_client") as build_client,
        pytest.raises(RuntimeError, match="source construction failed"),
    ):
        main(["play", "--task", "zip_zap_zop"])

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    build_client.assert_not_called()
    assert manifest["schema_version"] == 2
    assert manifest["status"] == "startup_error"
    assert manifest["human_input"] == {"mode": "terminal"}
    assert manifest["startup_error"]["message"] == "source construction failed"
    assert not (run_dir / "response_trace.jsonl").exists()


def test_play_task_client_failure_preserves_primary_error_and_closes_source(tmp_path) -> None:
    run_dir = tmp_path / "failed-client"
    source = _FailingCloseSource()

    with (
        patch("streammuse.presentation.task.cli._new_run_dir", return_value=run_dir),
        patch("streammuse.application.factories.human_input_factory.HumanInputFactory.create", return_value=source),
        patch(
            "streammuse.presentation.task.cli._build_client",
            side_effect=ValueError("client construction failed"),
        ),
        pytest.raises(ValueError, match="client construction failed"),
    ):
        main(["play", "--task", "zip_zap_zop"])

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert source.close_count == 1
    assert manifest["status"] == "startup_error"
    assert manifest["startup_error"]["message"] == "client construction failed"
    assert manifest["human_input"] == {"mode": "terminal", "source": "fake"}


def test_play_task_runtime_entry_failure_uses_outer_resource_fallback(tmp_path) -> None:
    run_dir = tmp_path / "failed-runtime-entry"
    source = _TrackingSource()
    client = _TrackingClient()

    with (
        patch("streammuse.presentation.task.cli._new_run_dir", return_value=run_dir),
        patch(
            "streammuse.application.factories.human_input_factory.HumanInputFactory.create",
            return_value=source,
        ),
        patch("streammuse.presentation.task.cli._build_client", return_value=client),
        patch(
            "streammuse.presentation.task.cli.InteractiveTaskRuntime.play",
            side_effect=RuntimeError("runtime entry failed"),
        ),
        pytest.raises(RuntimeError, match="runtime entry failed"),
    ):
        main(["play", "--task", "zip_zap_zop"])

    assert source.close_count == 1
    assert client.close_count == 1


class _FailingCloseSource:
    mode = "terminal"

    def __init__(self) -> None:
        self.close_count = 0

    @property
    def provenance(self) -> dict[str, object]:
        return {"mode": "terminal", "source": "fake"}

    def close(self) -> None:
        self.close_count += 1
        raise RuntimeError("source close failed")


class _TrackingSource:
    mode = "terminal"

    def __init__(self) -> None:
        self.close_count = 0
        self.closed = False

    @property
    def provenance(self) -> dict[str, object]:
        return {"mode": "terminal", "source": "tracking"}

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.close_count += 1


class _TrackingClient:
    def __init__(self) -> None:
        self.close_count = 0

    def close(self) -> None:
        self.close_count += 1
