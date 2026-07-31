from __future__ import annotations

from unittest.mock import patch

from streammuse.application.tasks import InteractiveTaskRunResult
from streammuse.infrastructure.voice import SpeakerPlaybackError
from streammuse.infrastructure.voice.speaker import SpeakerDevice
from streammuse.presentation.task.cli import build_parser, main


def _result() -> InteractiveTaskRunResult:
    return InteractiveTaskRunResult(
        output_dir="/tmp/run",
        task_name="zip_zap_zop",
        turn_count=0,
        human_turn_count=0,
        llm_turn_count=0,
        valid_count=0,
        invalid_count=0,
        deadline_miss_count=0,
    )


def _base() -> list[str]:
    return [
        "play",
        "--task",
        "zip_zap_zop",
        "--deadline-mode",
        "soft",
    ]


def test_speech_argparse_defaults_preserve_explicitness() -> None:
    args = build_parser().parse_args(_base())

    assert args.speech_output is None
    assert args.speech_backend is None
    assert args.speech_prewarm is None
    assert args.speech_save_audio is None
    assert args.llm_deadline_basis is None


def test_valid_speech_arguments_build_strong_config(capsys) -> None:
    with patch(
        "streammuse.presentation.task.cli.play_task",
        return_value=_result(),
    ) as play_task:
        exit_code = main(
            _base()
            + [
                "--speech-output",
                "audio",
                "--speech-backend",
                "null",
                "--speech-rate",
                "1.25",
                "--speaker-device",
                "2",
                "--no-speech-prewarm",
                "--speech-cache-miss",
                "skip",
                "--speech-on-error",
                "warn",
                "--speech-save-audio",
                "--llm-deadline-basis",
                "audio_end",
            ]
        )

    assert exit_code == 0
    config = play_task.call_args.kwargs["speech_output_config"]
    assert config.mode == "audio"
    assert config.backend == "null"
    assert config.rate == 1.25
    assert config.speaker_device == 2
    assert config.prewarm is False
    assert config.cache_miss == "skip"
    assert config.on_error == "warn"
    assert config.save_audio is True
    assert config.llm_deadline_basis == "audio_end"
    assert "interactive completed" in capsys.readouterr().out


def test_speech_options_are_rejected_while_output_is_off(capsys) -> None:
    exit_code = main(_base() + ["--speech-backend", "null"])

    assert exit_code == 2
    assert "require --speech-output audio" in capsys.readouterr().out


def test_audio_end_is_rejected_while_output_is_off(capsys) -> None:
    exit_code = main(_base() + ["--llm-deadline-basis", "audio_end"])

    assert exit_code == 2
    assert "audio_end" in capsys.readouterr().out


def test_non_kokoro_backend_rejects_model_options(capsys) -> None:
    exit_code = main(
        _base()
        + [
            "--speech-output",
            "audio",
            "--speech-backend",
            "system",
            "--speech-model",
            "repo",
            "--speech-model-revision",
            "commit",
        ]
    )

    assert exit_code == 2
    assert "kokoro" in capsys.readouterr().out


def test_kokoro_requires_model_and_revision(capsys) -> None:
    exit_code = main(
        _base()
        + [
            "--speech-output",
            "audio",
            "--speech-backend",
            "kokoro",
        ]
    )

    assert exit_code == 2
    assert "model and model revision" in capsys.readouterr().out


def test_speaker_devices_does_not_require_task_or_model(capsys) -> None:
    devices = (
        SpeakerDevice(
            index=2,
            name="Built-in Output",
            max_output_channels=2,
            default_sample_rate_hz=48_000.0,
            hostapi=0,
        ),
    )
    with patch(
        "streammuse.infrastructure.voice.speaker.enumerate_output_devices",
        return_value=devices,
    ):
        exit_code = main(["speaker-devices"])

    assert exit_code == 0
    assert "[2] Built-in Output" in capsys.readouterr().out


def test_speech_infrastructure_errors_use_output_label(capsys) -> None:
    with patch(
        "streammuse.presentation.task.cli.play_task",
        side_effect=SpeakerPlaybackError("speaker failed"),
    ):
        exit_code = main(
            _base()
            + [
                "--speech-output",
                "audio",
                "--speech-backend",
                "null",
            ]
        )

    assert exit_code == 1
    assert "speech output error: speaker failed" in capsys.readouterr().out
