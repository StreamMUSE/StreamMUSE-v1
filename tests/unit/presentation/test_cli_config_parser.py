"""Tests for CLI config parser."""

from __future__ import annotations

import argparse
import sys
from typing import Any

import pytest

from streammuse.presentation.cli.config_parser import args_to_config, parse_args


def _make_args(**overrides: Any) -> argparse.Namespace:
    base = {
        "tempo": 120.0,
        "ticks_per_beat": 4,
        "beats_per_bar": 4,
        "input_mode": "midi_device",
        "midi_device_name": None,
        "midi_file_path": None,
        "midi_file_delay_ticks": 0,
        "midi_file_trim_leading_rest": False,
        "injection_file": None,
        "injection_length": 0,
        "inject_acc_file": None,
        "output_type": "console",
        "midi_out_port": None,
        "midi_file_output_path": None,
        "close_active_notes_on_finalize": True,
        "enable_metronome": False,
        "metronome_port": None,
        "metronome_channel": 9,
        "inference_log_detail": "summary",
        "session_artifact_tier": "debug",
        "inference_type": "http",
        "server_url": "http://localhost:8000/generate_accompaniment",
        "model_name": "stanley",
        "inference_mode": "sliding_window",
        "timeout_s": 30.0,
        "checkpoint_path": None,
        "model_size": "0.12B",
        "model_max_seq_len_frames": 96,
        "generation_length_frames": 20,
        "generation_interval_ticks": 2,
        "prompt_length_ticks": 32,
        "continuation_mode": "standard",
        "model_condition_bpm": None,
        "rap_topic": None,
        "rap_pattern": "boom_bap",
        "rap_generator": "phrase_bank",
        "rap_lookahead_bars": 2,
        "rap_candidate_count": 12,
        "rap_model_url": "http://localhost:8000/v1",
        "rap_model": "local-model",
        "rap_timeout_s": 5.0,
        "count_in_beats": 0,
        "input_snap_forward_fraction": 0.4,
        "max_ticks": None,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def test_parse_args_defaults() -> None:
    """args_to_config returns default config values for default args."""
    args = _make_args()
    config = args_to_config(args)

    assert config.tempo.bpm == 120.0
    assert config.tempo.ticks_per_beat == 4
    assert config.tempo.beats_per_bar == 4
    assert config.input.type == "midi_device"
    assert config.input.midi_file_trim_leading_rest is False
    assert config.input.injection_file is None
    assert config.input.injection_length_ticks == 0
    assert config.input.injection_acc_file is None
    assert config.output.type == "console"
    assert config.output.close_active_notes_on_finalize is True
    assert config.output.inference_log_detail == "summary"
    assert config.output.session_artifact_tier == "debug"
    assert config.output.metronome_enabled is False
    assert config.output.metronome_port is None
    assert config.output.metronome_channel == 9
    assert config.inference.type == "http"
    assert config.inference.server_generate_url == "http://localhost:8000/generate_accompaniment"
    assert config.inference.generation_interval_ticks == 2
    assert config.inference.generation_length_frames == 20
    assert config.inference.prompt_length_ticks == 32
    assert config.inference.model_condition_bpm is None
    assert config.rap.topic is None
    assert config.rap.pattern == "boom_bap"
    assert config.rap.lookahead_bars == 2
    assert config.count_in_beats == 0
    assert config.continuation_mode == "standard"
    assert config.input_snap_forward_fraction == 0.4


def test_parse_args_exposes_rt_horizon_and_drain_contract(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "streammuse-cli",
            "--analysis-end-tick",
            "224",
            "--last-input-note-off-tick",
            "224",
            "--request-cutoff-tick",
            "220",
            "--run-stop-tick",
            "320",
            "--tail-beats",
            "24",
            "--drain-timeout-s",
            "15",
            "--model-condition-bpm",
            "120",
        ],
    )

    args = parse_args()

    assert args.analysis_end_tick == 224
    assert args.last_input_note_off_tick == 224
    assert args.request_cutoff_tick == 220
    assert args.run_stop_tick == 320
    assert args.tail_beats == 24
    assert args.drain_timeout_s == 15.0
    assert args.model_condition_bpm == 120


@pytest.mark.parametrize(
    ("flag", "expected"),
    [
        (None, True),
        ("--close-active-notes-on-finalize", True),
        ("--no-close-active-notes-on-finalize", False),
    ],
)
def test_parse_args_exposes_midi_finalize_policy(monkeypatch, flag, expected) -> None:
    argv = ["streammuse-cli"]
    if flag is not None:
        argv.append(flag)
    monkeypatch.setattr(sys, "argv", argv)

    args = parse_args()

    assert args.close_active_notes_on_finalize is expected


def test_args_to_config_separates_playback_and_model_condition_bpm() -> None:
    config = args_to_config(_make_args(tempo=60.0, model_condition_bpm=120))

    assert config.tempo.bpm == 60.0
    assert config.inference.model_condition_bpm == 120


def test_args_to_config_keyboard_input() -> None:
    """Test config with keyboard input."""
    args = _make_args(
        tempo=100.0,
        ticks_per_beat=8,
        beats_per_bar=3,
        input_mode="keyboard",
        output_type="audio",
        midi_out_port="Virtual MIDI Port",
        inference_log_detail="full",
        session_artifact_tier="normal",
        server_url="http://example.com/generate",
        timeout_s=60.0,
    )

    config = args_to_config(args)

    assert config.tempo.bpm == 100.0
    assert config.tempo.ticks_per_beat == 8
    assert config.tempo.beats_per_bar == 3
    assert config.input.type == "keyboard"
    assert config.output.type == "audio"
    assert config.output.midi_out_port == "Virtual MIDI Port"
    assert config.output.inference_log_detail == "full"
    assert config.inference.server_generate_url == "http://example.com/generate"
    assert config.inference.timeout_s == 60.0


def test_args_to_config_stanley_engine() -> None:
    """Test config with Stanley inference engine."""
    args = _make_args(
        inference_type="stanley",
        checkpoint_path="/path/to/checkpoint.ckpt",
        model_size="0.25B",
        model_max_seq_len_frames=128,
        generation_length_frames=30,
        generation_interval_ticks=4,
    )

    config = args_to_config(args)

    assert config.inference.type == "stanley"
    assert config.inference.checkpoint_path == "/path/to/checkpoint.ckpt"
    assert config.inference.model_size == "0.25B"
    assert config.inference.model_max_seq_len_frames == 128
    assert config.inference.generation_length_frames == 30
    assert config.inference.generation_interval_ticks == 4


def test_args_to_config_midi_file() -> None:
    """Test config with MIDI file input."""
    args = _make_args(
        input_mode="midi_file",
        midi_file_path="/path/to/song.mid",
        midi_file_delay_ticks=8,
        midi_file_trim_leading_rest=True,
        output_type="midi_file",
        midi_file_output_path="/path/to/output.mid",
        close_active_notes_on_finalize=False,
        inference_log_detail="full",
    )

    config = args_to_config(args)

    assert config.input.type == "midi_file"
    assert config.input.midi_file_path == "/path/to/song.mid"
    assert config.input.midi_file_delay_ticks == 8
    assert config.input.midi_file_trim_leading_rest is True
    assert config.output.type == "midi_file"
    assert config.output.midi_file_output_path == "/path/to/output.mid"
    assert config.output.close_active_notes_on_finalize is False
    assert config.output.inference_log_detail == "full"


def test_args_to_config_metronome_fields() -> None:
    args = _make_args(
        output_type="audio",
        midi_out_port="Music Port",
        enable_metronome=True,
        metronome_port="Click Port",
        metronome_channel=10,
        count_in_beats=4,
    )

    config = args_to_config(args)

    assert config.output.metronome_enabled is True
    assert config.output.metronome_port == "Click Port"
    assert config.output.metronome_channel == 10
    assert config.count_in_beats == 4


def test_args_to_config_clamps_negative_count_in() -> None:
    args = _make_args(count_in_beats=-2)
    config = args_to_config(args)
    assert config.count_in_beats == 0


def test_args_to_config_prompt_continuation_mode() -> None:
    config = args_to_config(
        _make_args(
            continuation_mode="prompt_continuation",
            prompt_length_ticks=64,
        )
    )

    assert config.continuation_mode == "prompt_continuation"
    assert config.inference.prompt_length_ticks == 64


def test_args_to_config_rejects_rap_with_prompt_continuation() -> None:
    with pytest.raises(ValueError, match="rap.*prompt_continuation"):
        args_to_config(
            _make_args(
                continuation_mode="prompt_continuation",
                rap_topic="space travel",
            )
        )


def test_args_to_config_clamps_input_snap_forward_fraction() -> None:
    assert args_to_config(_make_args(input_snap_forward_fraction=-0.5)).input_snap_forward_fraction == 0.0
    assert args_to_config(_make_args(input_snap_forward_fraction=1.5)).input_snap_forward_fraction == 1.0


def test_args_to_config_maps_rap_settings_and_clamps_positive_dimensions() -> None:
    config = args_to_config(
        _make_args(
            rap_topic="space travel",
            rap_pattern="trap_sparse",
            rap_generator="local_chat",
            rap_lookahead_bars=0,
            rap_candidate_count=-4,
            rap_model_url="http://chat.example/v1",
            rap_model="rap-model",
            rap_timeout_s=2.5,
        )
    )

    assert config.rap.topic == "space travel"
    assert config.rap.pattern == "trap_sparse"
    assert config.rap.generator == "local_chat"
    assert config.rap.lookahead_bars == 1
    assert config.rap.candidate_count == 1
    assert config.rap.model_url == "http://chat.example/v1"
    assert config.rap.model == "rap-model"
    assert config.rap.timeout_s == 2.5


def test_args_to_config_injection_fields() -> None:
    args = _make_args(
        input_mode="midi_file",
        midi_file_path="/path/to/input.mid",
        injection_file="/path/to/mel/1.mid",
        injection_length=16,
        inject_acc_file="/path/to/acc/1.mid",
    )

    config = args_to_config(args)

    assert config.input.injection_file == "/path/to/mel/1.mid"
    assert config.input.injection_length_ticks == 16
    assert config.input.injection_acc_file == "/path/to/acc/1.mid"


@pytest.mark.parametrize(
    ("input_mode", "output_type"),
    [
        ("keyboard", "audio"),
        ("midi_file", "midi_file"),
        ("midi_device", "console"),
    ],
)
def test_args_to_config_mode_matrix(input_mode: str, output_type: str) -> None:
    args = _make_args(input_mode=input_mode, output_type=output_type)
    config = args_to_config(args)
    assert config.input.type == input_mode
    assert config.output.type == output_type
