"""Integration tests for CLI entry point."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import mido
import pytest

from streammuse.application.config import ApplicationConfig, InferenceConfig, InputConfig, OutputConfig, TempoConfig
from streammuse.domain.musical import MusicalEvent
from streammuse.presentation.cli.cli import main


def _write_prompt_midi(path, *, pitch: int) -> None:
    mid = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.Message("note_on", note=pitch, velocity=64, time=0))
    track.append(mido.Message("note_off", note=pitch, velocity=0, time=480))
    mid.save(str(path))


class MockInputSource:
    """Mock input source for testing."""

    def read_events(self):
        """Yield no events."""
        return
        yield  # Make it a generator

    def close(self) -> None:
        """No-op."""


class MockOutputSink:
    """Mock output sink for testing."""

    def output_event(self, event: MusicalEvent, source: str) -> None:
        """No-op."""

    def output_tick(self, tick: int, bar: int, beat: int) -> None:
        """No-op."""

    def output_stats(
        self,
        hit_rate=None,
        avg_backup_level=None,
        round_trip_ms=None,
        server_process_ms=None,
        network_latency_ms=None,
        total_hits=None,
        total_ticks=None,
    ) -> None:
        """No-op."""

    def output_status(self, state: str, message: str = "") -> None:
        """No-op."""

    def output_config(self, config: dict) -> None:
        """No-op."""

    def close(self) -> None:
        """No-op."""


class MockInferenceEngine:
    """Mock inference engine for testing."""

    def generate_accompaniment(
        self,
        melody_events: list[MusicalEvent],
        generation_start_tick: int,
        generation_length_frames: int,
        prompt_length_ticks: int | None = None,
    ):
        """Return empty accompaniment."""
        from streammuse.domain.interfaces.timing_info import TimingInfo

        return [], TimingInfo(
            request_arrival_time=0.0,
            response_output_time=0.0,
            preprocess_start_time=0.0,
            inference_start_time=0.0,
            inference_end_time=0.0,
            postprocess_start_time=0.0,
        )

    def inject_history(
        self,
        melody_events: list[MusicalEvent],
        accompaniment_events: list[MusicalEvent],
        injection_length_ticks: int,
    ) -> None:
        """No-op."""

    def set_injection_offset(self, offset_ticks: int) -> None:
        """No-op."""

    def clear_history(self):
        """No-op."""
        return {"success": True, "melody_history": [], "accompaniment_history": []}


@patch("streammuse.presentation.cli.cli.InputSourceFactory")
@patch("streammuse.presentation.cli.cli.OutputSinkFactory")
@patch("streammuse.presentation.cli.cli.InferenceEngineFactory")
@patch("streammuse.presentation.cli.cli.parse_args")
@patch("sys.argv", ["streammuse-cli", "--max-ticks", "1"])
def test_cli_main_early_exit(
    mock_parse_args,
    mock_inference_factory,
    mock_output_factory,
    mock_input_factory,
) -> None:
    """Test CLI main function with early exit."""
    # Setup mocks
    mock_args = MagicMock()
    mock_args.max_ticks = 1
    mock_args.log_dir = "logs"
    mock_parse_args.return_value = mock_args

    mock_input_source = MockInputSource()
    mock_output_sink = MockOutputSink()
    mock_inference_engine = MockInferenceEngine()

    mock_input_factory.create.return_value = mock_input_source
    mock_output_factory.create.return_value = mock_output_sink
    mock_inference_factory.create.return_value = mock_inference_engine

    # Mock args_to_config to return a valid config
    with patch("streammuse.presentation.cli.cli.args_to_config") as mock_args_to_config:
        mock_config = ApplicationConfig(
            tempo=TempoConfig(),
            input=InputConfig(),
            output=OutputConfig(),
            inference=InferenceConfig(),
        )
        mock_args_to_config.return_value = mock_config

        # Run main (should exit quickly due to max_ticks=1)
        # We'll catch SystemExit from signal handler
        try:
            result = main()
            assert result == 0
        except SystemExit:
            pass  # Expected from signal handler


@patch("streammuse.presentation.cli.cli.InputSourceFactory")
@patch("streammuse.presentation.cli.cli.OutputSinkFactory")
@patch("streammuse.presentation.cli.cli.InferenceEngineFactory")
@patch("streammuse.presentation.cli.cli.parse_args")
def test_cli_injection_rejects_non_midi_input_mode(
    mock_parse_args,
    mock_inference_factory,
    mock_output_factory,
    mock_input_factory,
) -> None:
    mock_args = MagicMock()
    mock_args.max_ticks = 1
    mock_args.log_dir = "logs"
    mock_parse_args.return_value = mock_args

    cfg = ApplicationConfig(
        tempo=TempoConfig(),
        input=InputConfig(
            type="keyboard",
            injection_file="/tmp/not-used.mid",
            injection_length_ticks=16,
        ),
        output=OutputConfig(),
        inference=InferenceConfig(),
    )

    with patch("streammuse.presentation.cli.cli.args_to_config", return_value=cfg):
        result = main()

    assert result == 1
    mock_output_factory.create.assert_not_called()
    mock_inference_factory.create.assert_not_called()
    mock_input_factory.create.assert_not_called()


@patch("streammuse.presentation.cli.cli.InputSourceFactory")
@patch("streammuse.presentation.cli.cli.OutputSinkFactory")
@patch("streammuse.presentation.cli.cli.InferenceEngineFactory")
@patch("streammuse.presentation.cli.cli.parse_args")
def test_cli_rejects_unwired_prompt_continuation_mode(
    mock_parse_args,
    mock_inference_factory,
    mock_output_factory,
    mock_input_factory,
) -> None:
    mock_args = MagicMock()
    mock_args.max_ticks = 1
    mock_args.log_dir = "logs"
    mock_parse_args.return_value = mock_args

    cfg = ApplicationConfig(
        tempo=TempoConfig(),
        input=InputConfig(),
        output=OutputConfig(),
        inference=InferenceConfig(),
        continuation_mode="prompt_continuation",
    )

    with patch("streammuse.presentation.cli.cli.args_to_config", return_value=cfg):
        result = main()

    assert result == 1
    mock_output_factory.create.assert_not_called()
    mock_inference_factory.create.assert_not_called()
    mock_input_factory.create.assert_not_called()


@patch("streammuse.presentation.cli.cli.RealTimeMusicService")
@patch("streammuse.presentation.cli.cli.InputSourceFactory")
@patch("streammuse.presentation.cli.cli.OutputSinkFactory")
@patch("streammuse.presentation.cli.cli.InferenceEngineFactory")
@patch("streammuse.presentation.cli.cli.parse_args")
def test_cli_injection_calls_clear_then_inject_and_then_creates_input(
    mock_parse_args,
    mock_inference_factory,
    mock_output_factory,
    mock_input_factory,
    mock_service_cls,
    tmp_path,
) -> None:
    mel_dir = tmp_path / "mel"
    acc_dir = tmp_path / "acc"
    mel_dir.mkdir()
    acc_dir.mkdir()

    mel_file = mel_dir / "1.mid"
    acc_file = acc_dir / "1.mid"
    _write_prompt_midi(mel_file, pitch=60)
    _write_prompt_midi(acc_file, pitch=48)

    mock_args = MagicMock()
    mock_args.max_ticks = 1
    mock_args.log_dir = str(tmp_path / "logs")
    mock_parse_args.return_value = mock_args

    cfg = ApplicationConfig(
        tempo=TempoConfig(),
        input=InputConfig(
            type="midi_file",
            midi_file_path=str(mel_file),
            injection_file=str(mel_file),
            injection_length_ticks=16,
            injection_acc_file=None,
        ),
        output=OutputConfig(),
        inference=InferenceConfig(),
    )

    output_sink = MockOutputSink()
    inference_engine = MockInferenceEngine()
    service = MagicMock()
    service.running = False
    mock_service_cls.return_value = service
    mock_output_factory.create.return_value = output_sink
    mock_inference_factory.create.return_value = inference_engine

    injected_state = {"done": False}
    call_order: list[str] = []

    def _clear_history_side_effect():
        call_order.append("clear_history")
        return {"success": True, "melody_history": [], "accompaniment_history": []}

    def _inject_history_side_effect(*args, **kwargs):
        _ = args, kwargs
        call_order.append("inject_history")
        injected_state["done"] = True

    inference_engine.inject_history = MagicMock(side_effect=_inject_history_side_effect)
    inference_engine.clear_history = MagicMock(side_effect=_clear_history_side_effect)

    def _input_create_side_effect(*args, **kwargs):
        _ = args, kwargs
        assert injected_state["done"] is True
        return MockInputSource()

    mock_input_factory.create.side_effect = _input_create_side_effect

    with patch("streammuse.presentation.cli.cli.args_to_config", return_value=cfg):
        result = main()

    assert result == 0
    assert inference_engine.clear_history.call_count == 1
    assert inference_engine.inject_history.call_count == 1
    assert call_order == ["clear_history", "inject_history"]
