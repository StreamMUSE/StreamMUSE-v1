"""Integration tests for CLI entry point."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from streammuse.application.config import ApplicationConfig, InferenceConfig, InputConfig, OutputConfig, TempoConfig
from streammuse.application.services.real_time_music_service import RealTimeMusicService
from streammuse.domain.interfaces import InferenceEngine, InputSource, OutputSink
from streammuse.domain.musical import MusicalEvent
from streammuse.domain.musical.events import EventType
from streammuse.domain.timing import PlaybackScheduler, Tempo
from streammuse.presentation.cli.cli import main


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

    def clear_history(self) -> None:
        """No-op."""


@pytest.fixture
def mock_config() -> ApplicationConfig:
    """Create a test configuration."""
    return ApplicationConfig(
        tempo=TempoConfig(bpm=120.0, ticks_per_beat=4, beats_per_bar=4),
        input=InputConfig(type="list"),
        output=OutputConfig(type="console"),
        inference=InferenceConfig(
            type="http",
            generation_interval_ticks=2,
            generation_length_frames=20,
        ),
    )


def test_service_creation(mock_config: ApplicationConfig) -> None:
    """Test that service can be created with mock dependencies."""
    input_source = MockInputSource()
    output_sink = MockOutputSink()
    inference_engine = MockInferenceEngine()
    tempo = Tempo(bpm=120.0, ticks_per_beat=4, beats_per_bar=4)
    scheduler = PlaybackScheduler()

    service = RealTimeMusicService(
        input_source=input_source,
        inference_engine=inference_engine,
        output_sink=output_sink,
        tempo=tempo,
        scheduler=scheduler,
        generation_interval_ticks=mock_config.inference.generation_interval_ticks,
        generation_length_frames=mock_config.inference.generation_length_frames,
    )

    assert not service.running
    assert service._generation_interval_ticks == 2
    assert service._generation_length_frames == 20


def test_service_start_stop(mock_config: ApplicationConfig) -> None:
    """Test that service can start and stop."""
    input_source = MockInputSource()
    output_sink = MockOutputSink()
    inference_engine = MockInferenceEngine()
    tempo = Tempo(bpm=120.0, ticks_per_beat=4, beats_per_bar=4)
    scheduler = PlaybackScheduler()

    service = RealTimeMusicService(
        input_source=input_source,
        inference_engine=inference_engine,
        output_sink=output_sink,
        tempo=tempo,
        scheduler=scheduler,
        generation_interval_ticks=mock_config.inference.generation_interval_ticks,
        generation_length_frames=mock_config.inference.generation_length_frames,
    )

    # Start with max_ticks=5 for quick test
    service.start(max_ticks=5)

    # Wait a bit for threads to run
    import time

    time.sleep(0.5)

    # Stop service
    service.stop()

    # Service should be stopped
    assert not service.running


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
