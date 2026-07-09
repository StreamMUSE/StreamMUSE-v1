from __future__ import annotations

from unittest.mock import patch

from streammuse.presentation.debug.cli import main


def test_debug_cli_replay_invokes_replay_service(tmp_path) -> None:
    argv = [
        "streammuse-debug",
        "replay",
        "--scenario",
        "lekai-prompt-continuation",
        "--midi-file",
        "song.mid",
        "--compare",
        "offline,sim",
        "--output-dir",
        str(tmp_path),
    ]

    with patch("sys.argv", argv), patch(
        "streammuse.presentation.debug.cli.run_replay"
    ) as run_replay:
        run_replay.return_value = {"output_dir": str(tmp_path), "comparison_path": None}

        assert main() == 0

    config = run_replay.call_args.args[0]
    assert config.scenario == "lekai-prompt-continuation"
    assert config.midi_file == "song.mid"
    assert config.compare == ("offline", "sim")
