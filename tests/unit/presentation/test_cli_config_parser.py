"""Tests for CLI config parser."""

from __future__ import annotations

import argparse
from typing import Any

import pytest

from streammuse.presentation.cli.config_parser import args_to_config


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
        "enable_metronome": False,
        "metronome_port": None,
        "metronome_channel": 9,
        "inference_log_detail": "summary",
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
    assert config.output.inference_log_detail == "summary"
    assert config.output.metronome_enabled is False
    assert config.output.metronome_port is None
    assert config.output.metronome_channel == 9
    assert config.inference.type == "http"
    assert config.inference.server_generate_url == "http://localhost:8000/generate_accompaniment"
    assert config.inference.generation_interval_ticks == 2
    assert config.inference.generation_length_frames == 20
    assert config.inference.prompt_length_ticks == 32
    assert config.continuation_mode == "standard"
    assert config.count_in_beats == 0
    assert config.input_snap_forward_fraction == 0.4


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
        inference_log_detail="full",
    )

    config = args_to_config(args)

    assert config.input.type == "midi_file"
    assert config.input.midi_file_path == "/path/to/song.mid"
    assert config.input.midi_file_delay_ticks == 8
    assert config.input.midi_file_trim_leading_rest is True
    assert config.output.type == "midi_file"
    assert config.output.midi_file_output_path == "/path/to/output.mid"
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


def test_args_to_config_clamps_input_snap_forward_fraction() -> None:
    assert args_to_config(_make_args(input_snap_forward_fraction=-0.5)).input_snap_forward_fraction == 0.0
    assert args_to_config(_make_args(input_snap_forward_fraction=1.5)).input_snap_forward_fraction == 1.0


def test_args_to_config_prompt_continuation_mode() -> None:
    args = _make_args(continuation_mode="prompt_continuation", prompt_length_ticks=64)
    config = args_to_config(args)
    assert config.continuation_mode == "prompt_continuation"
    assert config.inference.prompt_length_ticks == 64


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
