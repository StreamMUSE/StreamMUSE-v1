from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from streammuse.application.config import (
    ApplicationConfig,
    InferenceConfig,
    InputConfig,
    OutputConfig,
    RapConfig,
    TempoConfig,
)
from streammuse.application.runtime import RuntimeSession, RuntimeSessionBuilder
from streammuse.domain.logging import SessionManager
from streammuse.infrastructure.output.composite import CompositeOutputSink
from streammuse.infrastructure.output.metronome import MetronomeOutputSink
from streammuse.infrastructure.output.midi_file import MidiFileOutputSink
from streammuse.infrastructure.output.session_logger import SessionLoggerOutputSink
from streammuse.infrastructure.output.websocket import WebSocketOutputSink


@patch("streammuse.application.runtime.builder.RealTimeMusicService")
@patch("streammuse.application.runtime.builder.InputSourceFactory")
@patch("streammuse.application.runtime.builder.InferenceEngineFactory")
@patch("streammuse.application.runtime.builder.OutputSinkFactory")
def test_builder_creates_standard_runtime_session(
    output_factory,
    inference_factory,
    input_factory,
    service_cls,
    tmp_path,
) -> None:
    config = ApplicationConfig(
        output=OutputConfig(type="console"),
        inference=InferenceConfig(generation_interval_ticks=8),
    )
    output = MagicMock()
    input_source = MagicMock()
    engine = MagicMock()
    service = MagicMock(running=False)
    output_factory.create.return_value = output
    input_factory.create.return_value = input_source
    inference_factory.create.return_value = engine
    service_cls.return_value = service

    session = RuntimeSessionBuilder(config=config, log_dir=str(tmp_path)).build_cli()

    assert session.service is service
    assert session.output_sink is output
    service_cls.assert_called_once()
    assert service_cls.call_args.kwargs["generation_interval_ticks"] == 8


@patch("streammuse.application.runtime.builder.RealTimeMusicService")
@patch("streammuse.application.runtime.builder.InputSourceFactory")
@patch("streammuse.application.runtime.builder.InferenceEngineFactory")
def test_builder_creates_web_standard_runtime_session(
    inference_factory,
    input_factory,
    service_cls,
    tmp_path,
) -> None:
    config = ApplicationConfig(
        output=OutputConfig(close_active_notes_on_finalize=False),
        inference=InferenceConfig(generation_interval_ticks=4),
    )
    input_factory.create.return_value = MagicMock()
    inference_factory.create.return_value = MagicMock()
    service = MagicMock(running=False)
    service_cls.return_value = service

    session = RuntimeSessionBuilder(config=config, log_dir=str(tmp_path)).build_web()

    assert session.service is service
    assert isinstance(session.websocket_sink, WebSocketOutputSink)
    midi_sink = next(
        sink for sink in session.output_sink.sinks if isinstance(sink, MidiFileOutputSink)
    )
    assert midi_sink._config.close_active_notes_on_finalize is False
    assert session.session_config["close_active_notes_on_finalize"] is False
    assert service_cls.call_args.kwargs["generation_interval_ticks"] == 4


@patch("streammuse.application.runtime.builder.RealTimeMusicService")
@patch("streammuse.application.runtime.builder.InputSourceFactory")
@patch("streammuse.application.runtime.builder.InferenceEngineFactory")
def test_builder_web_composite_honors_metronome_and_midi_recording_config(
    inference_factory,
    input_factory,
    service_cls,
    tmp_path,
) -> None:
    config = ApplicationConfig(
        tempo=TempoConfig(bpm=90.0, ticks_per_beat=6, beats_per_bar=3),
        output=OutputConfig(
            metronome_enabled=True,
            metronome_port="click-port",
            metronome_channel=8,
        ),
    )
    input_factory.create.return_value = MagicMock()
    inference_factory.create.return_value = MagicMock()
    service_cls.return_value = MagicMock(running=False)

    session = RuntimeSessionBuilder(config=config, log_dir=str(tmp_path)).build_web()

    midi_sink = next(
        sink for sink in session.output_sink.sinks if isinstance(sink, MidiFileOutputSink)
    )
    metronome_sink = next(
        sink for sink in session.output_sink.sinks if isinstance(sink, MetronomeOutputSink)
    )
    assert midi_sink._config.beats_per_bar == 3
    assert midi_sink._config.record_metronome is True
    assert metronome_sink._config.port_name == "click-port"
    assert metronome_sink._config.ticks_per_beat == 6
    assert metronome_sink._config.beats_per_bar == 3
    assert metronome_sink._config.channel == 8


@patch("streammuse.application.runtime.builder.RealTimeMusicService")
@patch("streammuse.application.runtime.builder.InputSourceFactory")
@patch("streammuse.application.runtime.builder.InferenceEngineFactory")
def test_builder_creates_unique_web_session_directory_on_fast_restart(
    inference_factory,
    input_factory,
    service_cls,
    tmp_path,
) -> None:
    input_factory.create.return_value = MagicMock()
    inference_factory.create.return_value = MagicMock()
    service_cls.return_value = MagicMock(running=False)
    builder = RuntimeSessionBuilder(config=ApplicationConfig(), log_dir=str(tmp_path))

    first = builder.build_web()
    second = builder.build_web()

    assert first.session_dir != second.session_dir
    assert first.session_dir.exists()
    assert second.session_dir.exists()


@patch("streammuse.application.runtime.builder.RealTimeMusicService")
@patch("streammuse.application.runtime.builder.InputSourceFactory")
@patch("streammuse.application.runtime.builder.InferenceEngineFactory")
def test_builder_adds_trace_only_session_logger_to_web_when_enabled(
    inference_factory,
    input_factory,
    service_cls,
    tmp_path,
) -> None:
    config = ApplicationConfig(input_quantization_trace_enabled=True)
    input_factory.create.return_value = MagicMock()
    inference_factory.create.return_value = MagicMock()
    service_cls.return_value = MagicMock(running=False)

    session = RuntimeSessionBuilder(config=config, log_dir=str(tmp_path)).build_web()

    assert isinstance(session.output_sink, CompositeOutputSink)
    session_sinks = [
        sink
        for sink in session.output_sink.sinks
        if isinstance(sink, SessionLoggerOutputSink)
    ]
    midi_sinks = [
        sink
        for sink in session.output_sink.sinks
        if isinstance(sink, MidiFileOutputSink)
    ]
    assert len(session_sinks) == 1
    assert len(midi_sinks) == 1
    assert session_sinks[0].midi_sink is None
    assert session_sinks[0].json_sink is None
    assert service_cls.call_args.kwargs["input_quantization_trace_enabled"] is True


@patch("streammuse.application.runtime.builder.RealTimeMusicService")
@patch("streammuse.application.runtime.builder.InputSourceFactory")
@patch("streammuse.application.runtime.builder.InferenceEngineFactory")
def test_builder_reuses_cli_session_logger_for_quantization_trace(
    inference_factory,
    input_factory,
    service_cls,
    tmp_path,
) -> None:
    config = ApplicationConfig(
        output=OutputConfig(type="session"),
        input_quantization_trace_enabled=True,
    )
    input_factory.create.return_value = MagicMock()
    inference_factory.create.return_value = MagicMock()
    service_cls.return_value = MagicMock(running=False)

    session = RuntimeSessionBuilder(config=config, log_dir=str(tmp_path)).build_cli()

    assert isinstance(session.output_sink, SessionLoggerOutputSink)
    assert service_cls.call_args.kwargs["input_quantization_trace_enabled"] is True


@patch("streammuse.application.runtime.builder.RealTimeMusicService")
@patch("streammuse.application.runtime.builder.InputSourceFactory")
@patch("streammuse.application.runtime.builder.InferenceEngineFactory")
@patch("streammuse.application.runtime.builder.OutputSinkFactory")
def test_builder_preserves_standard_rap_and_session_contract(
    output_factory,
    inference_factory,
    input_factory,
    service_cls,
    tmp_path,
) -> None:
    controller = MagicMock()
    config = ApplicationConfig(
        output=OutputConfig(type="console", session_artifact_tier="debug"),
        inference=InferenceConfig(
            generation_interval_ticks=8,
            generation_length_frames=24,
        ),
        rap=RapConfig(topic="space travel"),
        count_in_beats=4,
        input_snap_forward_fraction=0.25,
    )
    output_factory.create.return_value = MagicMock()
    input_factory.create.return_value = MagicMock()
    inference_factory.create.return_value = MagicMock()
    service_cls.return_value = MagicMock(running=False)

    session = RuntimeSessionBuilder(
        config=config,
        log_dir=str(tmp_path),
        tick_observer_factory=lambda _tempo: controller,
    ).build_cli()

    kwargs = service_cls.call_args.kwargs
    assert kwargs["generation_interval_ticks"] == 8
    assert kwargs["generation_length_frames"] == 24
    assert kwargs["count_in_beats"] == 4
    assert kwargs["input_snap_forward_fraction"] == 0.25
    assert kwargs["tick_observer"] is controller
    assert session.session_config["session_artifact_tier"] == "debug"
    assert session.session_config["rap_enabled"] is True
    assert session.session_config["continuation_mode"] == "standard"


def test_runtime_session_forwards_standard_run_horizons() -> None:
    service = MagicMock(running=False)
    output = MagicMock()
    session = RuntimeSession(
        config=ApplicationConfig(),
        session_manager=None,
        output_sink=output,
        service=service,
        session_config={"mode": "standard"},
        emit_output_config=False,
    )

    session.start(
        run_stop_tick=64,
        analysis_end_tick=48,
        last_input_note_off_tick=44,
        request_cutoff_tick=40,
        drain_timeout_seconds=3.5,
    )

    output.output_config.assert_not_called()
    service.start.assert_called_once_with(
        run_stop_tick=64,
        analysis_end_tick=48,
        last_input_note_off_tick=44,
        request_cutoff_tick=40,
        drain_timeout_seconds=3.5,
    )


def test_runtime_session_maps_prompt_run_stop_to_max_ticks() -> None:
    service = MagicMock(running=False)
    session = RuntimeSession(
        config=ApplicationConfig(continuation_mode="prompt_continuation"),
        session_manager=None,
        output_sink=MagicMock(),
        service=service,
        session_config={},
        emit_output_config=False,
    )

    session.start(run_stop_tick=64)

    service.start.assert_called_once_with(max_ticks=64)


def test_runtime_session_records_server_generated_seeds_before_service_start(
    tmp_path,
) -> None:
    calls = []
    response = {
        "success": True,
        "prompt_requested_seed": 17,
        "prompt_effective_seed": 17,
        "continuation_requested_seed": 17,
        "continuation_effective_seed": 17,
        "prompt_seed_source": "system",
        "continuation_seed_source": "system",
        "session_id": "server-session",
        "session_epoch": 1,
        "pending_boundary_generations": 0,
        "scheduler_phase": "idle",
        "scheduler_is_running": False,
    }

    class _PromptClient:
        def initialize_session(self):
            calls.append("initialize_session")
            return dict(response)

    class _AuditOutput:
        def record_prompt_continuation_session_seed(self, provenance):
            calls.append("record_seed")
            assert provenance["runtime_session_id"] == "web-session"
            assert provenance["prompt_effective_seed"] == 17

        def output_config(self, _config):
            calls.append("output_config")

    class _Service:
        running = False

        def start(self, *, max_ticks=None):
            calls.append(("service_start", max_ticks))

    manager = SessionManager(str(tmp_path), session_id="web-session")
    manager.create_session_directory()
    session = RuntimeSession(
        config=ApplicationConfig(continuation_mode="prompt_continuation"),
        session_manager=manager,
        output_sink=_AuditOutput(),
        service=_Service(),
        session_config={"mode": "prompt_continuation"},
        prompt_client=_PromptClient(),
        prompt_session_audit_enabled=True,
    )

    session.start(run_stop_tick=64)

    assert calls == [
        "initialize_session",
        "record_seed",
        "output_config",
        ("service_start", 64),
    ]
    assert session.metadata["prompt_continuation_session_seed"][
        "continuation_effective_seed"
    ] == 17


def test_runtime_session_without_audit_logger_does_not_initialize_prompt_session() -> None:
    client = MagicMock()
    service = MagicMock(running=False)
    session = RuntimeSession(
        config=ApplicationConfig(continuation_mode="prompt_continuation"),
        session_manager=None,
        output_sink=MagicMock(),
        service=service,
        session_config={},
        prompt_client=client,
        prompt_session_audit_enabled=False,
        emit_output_config=False,
    )

    session.start(run_stop_tick=16)

    client.initialize_session.assert_not_called()
    service.start.assert_called_once_with(max_ticks=16)


def test_runtime_session_adopts_runner_seed_without_reinitializing(tmp_path) -> None:
    runtime_info = {
        "prompt_sample_seed": 7,
        "continuation_sample_seed": 7,
        "session_id": "runner-session",
        "session_epoch": 4,
    }

    class _PromptClient:
        def initialize_session(self):
            raise AssertionError("runner-owned session must not be reinitialized")

        def replay_audit(self):
            return {"runtime_info": dict(runtime_info)}

    manager = SessionManager(str(tmp_path), session_id="cli-session")
    session_dir = manager.create_session_directory()
    output = SessionLoggerOutputSink(
        session_dir=session_dir,
        include_midi=False,
        include_json=False,
    )
    service = MagicMock(running=False)
    session = RuntimeSession(
        config=ApplicationConfig(continuation_mode="prompt_continuation"),
        session_manager=manager,
        output_sink=output,
        service=service,
        session_config={},
        prompt_client=_PromptClient(),
        prompt_session_audit_enabled=True,
        prompt_session_seed_provenance={
            "success": True,
            "prompt_requested_seed": 7,
            "prompt_effective_seed": 7,
            "continuation_requested_seed": 7,
            "continuation_effective_seed": 7,
            "prompt_seed_source": "requested",
            "continuation_seed_source": "requested",
            "session_id": "runner-session",
            "session_epoch": 4,
            "provenance_mode": "adopted_active_session",
        },
        emit_output_config=False,
    )

    session.start(run_stop_tick=64)

    service.start.assert_called_once_with(max_ticks=64)
    artifact = json.loads(
        (session_dir / "prompt_continuation_session_seed.json").read_text(
            encoding="utf-8"
        )
    )
    assert artifact["prompt_effective_seed"] == runtime_info["prompt_sample_seed"]
    assert (
        artifact["continuation_effective_seed"]
        == runtime_info["continuation_sample_seed"]
    )
    assert artifact["session_id"] == runtime_info["session_id"]
    assert artifact["session_epoch"] == runtime_info["session_epoch"]


def test_runtime_session_stop_calls_idempotent_service_stop_after_natural_end() -> None:
    service = MagicMock(running=False)
    session = RuntimeSession(
        config=ApplicationConfig(),
        session_manager=None,
        output_sink=MagicMock(),
        service=service,
        session_config={},
        emit_output_config=False,
    )

    session.stop()

    service.stop.assert_called_once_with()


def test_builder_passes_session_generation_config_to_prompt_http_client() -> None:
    inference = InferenceConfig(
        server_generate_url="http://prompt.example",
        timeout_s=12.5,
        inference_mode="stateful",
        checkpoint_path="model.safetensors",
        prompt_selection_mode="rule_s_if_else",
        prompt_batch_candidates=10,
        temperature=1.1,
        top_p=0.95,
        top_k=50,
        repetition_penalty=1.0,
    )
    builder = RuntimeSessionBuilder(config=ApplicationConfig(inference=inference))

    with (
        patch(
            "streammuse.infrastructure.inference.prompt_continuation_http_client."
            "PromptContinuationHttpClientConfig"
        ) as client_config_cls,
        patch(
            "streammuse.infrastructure.inference.prompt_continuation_http_client."
            "PromptContinuationHttpClient"
        ) as client_cls,
    ):
        client_config = client_config_cls.return_value
        client = builder._create_prompt_client()

    client_config_cls.assert_called_once_with(
        base_url="http://prompt.example",
        timeout_s=12.5,
        model_name="lekai_prompt_continuation",
        inference_mode="stateful",
        checkpoint_path="model.safetensors",
        prompt_selection_mode="rule_s_if_else",
        prompt_batch_candidates=10,
        temperature=1.1,
        top_p=0.95,
        top_k=50,
        repetition_penalty=1.0,
    )
    client_cls.assert_called_once_with(client_config)
    assert client is client_cls.return_value
    assert {
        name: builder._session_config()[name]
        for name in (
            "prompt_selection_mode",
            "prompt_batch_candidates",
            "temperature",
            "top_p",
            "top_k",
            "repetition_penalty",
        )
    } == {
        "prompt_selection_mode": "rule_s_if_else",
        "prompt_batch_candidates": 10,
        "temperature": 1.1,
        "top_p": 0.95,
        "top_k": 50,
        "repetition_penalty": 1.0,
    }


@patch("streammuse.application.runtime.builder.InputSourceFactory")
def test_builder_creates_prompt_continuation_runtime_with_override(
    input_factory,
    tmp_path,
) -> None:
    config = ApplicationConfig(
        tempo=TempoConfig(bpm=90.6),
        continuation_mode="prompt_continuation",
        count_in_beats=4,
        input_snap_forward_fraction=0.25,
        inference=InferenceConfig(
            prompt_length_ticks=64,
            generation_interval_ticks=4,
        ),
    )
    prompt_client = MagicMock()
    output = MagicMock()
    service_cls = MagicMock()
    service = MagicMock(running=False)
    service_cls.return_value = service
    input_factory.create.return_value = MagicMock()
    builder = RuntimeSessionBuilder(
        config=config,
        log_dir=str(tmp_path),
        prompt_client_override=prompt_client,
        output_sink_override=output,
    )

    with patch.object(
        builder,
        "_prompt_continuation_service_cls",
        return_value=service_cls,
    ):
        session = builder.build_cli()

    assert session.service is service
    assert session.prompt_client is prompt_client
    assert session.inference_engine is None
    assert service_cls.call_args.kwargs["prompt_length_ticks"] == 64
    assert service_cls.call_args.kwargs["generation_interval_ticks"] == 4
    assert service_cls.call_args.kwargs["count_in_beats"] == 4
    assert service_cls.call_args.kwargs["input_snap_forward_fraction"] == 0.25
    assert service_cls.call_args.kwargs["model_condition_bpm"] == 91
    assert session.session_config["effective_model_bpm"] == 91


@patch("streammuse.application.runtime.builder.InputSourceFactory")
def test_builder_wires_midi_file_source_tick_opt_in(
    input_factory,
    tmp_path,
) -> None:
    config = ApplicationConfig(
        input=InputConfig(
            type="midi_file",
            midi_file_path="melody.mid",
            midi_file_source_tick_mode=True,
        ),
        continuation_mode="prompt_continuation",
    )
    service_cls = MagicMock(return_value=MagicMock(running=False))
    input_factory.create.return_value = MagicMock()
    builder = RuntimeSessionBuilder(
        config=config,
        log_dir=str(tmp_path),
        prompt_client_override=MagicMock(),
        output_sink_override=MagicMock(),
    )

    with patch.object(
        builder,
        "_prompt_continuation_service_cls",
        return_value=service_cls,
    ):
        session = builder.build_cli()

    assert service_cls.call_args.kwargs["source_tick_input"] is True
    assert service_cls.call_args.kwargs["input_snap_forward_fraction"] == 0.0
    assert session.session_config["midi_file_source_tick_mode"] is True


@patch("streammuse.application.runtime.builder.InputSourceFactory")
def test_builder_web_prompt_runtime_always_has_replay_audit_logger(
    input_factory,
    tmp_path,
) -> None:
    config = ApplicationConfig(continuation_mode="prompt_continuation")
    prompt_client = MagicMock()
    service_cls = MagicMock(return_value=MagicMock(running=False))
    input_factory.create.return_value = MagicMock()
    builder = RuntimeSessionBuilder(
        config=config,
        log_dir=str(tmp_path),
        prompt_client_override=prompt_client,
    )

    with patch.object(
        builder,
        "_prompt_continuation_service_cls",
        return_value=service_cls,
    ):
        session = builder.build_web()

    audit_sinks = [
        sink
        for sink in session.output_sink.sinks
        if isinstance(sink, SessionLoggerOutputSink)
    ]
    assert len(audit_sinks) == 1
    assert audit_sinks[0].midi_sink is None
    assert audit_sinks[0].json_sink is None
    assert session.prompt_session_audit_enabled is True


@patch("streammuse.application.runtime.builder.InputSourceFactory")
def test_builder_cli_adopts_external_prompt_session_provenance(
    input_factory,
    tmp_path,
    monkeypatch,
) -> None:
    values = {
        "LEKAI_PROMPT_REQUESTED_SEED": "7",
        "LEKAI_PROMPT_EFFECTIVE_SEED": "7",
        "LEKAI_CONTINUATION_REQUESTED_SEED": "7",
        "LEKAI_CONTINUATION_EFFECTIVE_SEED": "7",
        "LEKAI_PROMPT_SESSION_ID": "runner-session",
        "LEKAI_PROMPT_SESSION_EPOCH": "4",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    config = ApplicationConfig(
        continuation_mode="prompt_continuation",
        output=OutputConfig(type="session"),
    )
    input_factory.create.return_value = MagicMock()
    builder = RuntimeSessionBuilder(
        config=config,
        log_dir=str(tmp_path),
        prompt_client_override=MagicMock(),
    )

    with patch.object(
        builder,
        "_prompt_continuation_service_cls",
        return_value=MagicMock(return_value=MagicMock(running=False)),
    ):
        session = builder.build_cli()

    assert session.prompt_session_audit_enabled is True
    assert session.prompt_session_seed_provenance == {
        "success": True,
        "prompt_requested_seed": 7,
        "prompt_effective_seed": 7,
        "continuation_requested_seed": 7,
        "continuation_effective_seed": 7,
        "prompt_seed_source": "requested",
        "continuation_seed_source": "requested",
        "session_id": "runner-session",
        "session_epoch": 4,
        "provenance_mode": "adopted_active_session",
    }


@patch("streammuse.application.runtime.builder.InputSourceFactory")
def test_prompt_runtime_uses_explicit_model_condition_bpm_override(
    input_factory,
    tmp_path,
) -> None:
    config = ApplicationConfig(
        tempo=TempoConfig(bpm=80.0),
        continuation_mode="prompt_continuation",
        inference=InferenceConfig(model_condition_bpm=137),
    )
    service_cls = MagicMock(return_value=MagicMock(running=False))
    input_factory.create.return_value = MagicMock()
    builder = RuntimeSessionBuilder(
        config=config,
        log_dir=str(tmp_path),
        prompt_client_override=MagicMock(),
        output_sink_override=MagicMock(),
    )

    with patch.object(
        builder,
        "_prompt_continuation_service_cls",
        return_value=service_cls,
    ):
        session = builder.build_cli()

    assert service_cls.call_args.kwargs["model_condition_bpm"] == 137
    assert session.session_config["tempo_bpm"] == 80.0
    assert session.session_config["model_condition_bpm"] == 137
    assert session.session_config["effective_model_bpm"] == 137


@patch("streammuse.application.runtime.builder.RealTimeMusicService")
@patch("streammuse.application.runtime.builder.InputSourceFactory")
@patch("streammuse.application.runtime.builder.InferenceEngineFactory")
@patch("streammuse.application.runtime.builder.OutputSinkFactory")
def test_builder_allows_eight_steps_per_beat_for_standard_runtime(
    output_factory,
    inference_factory,
    input_factory,
    service_cls,
    tmp_path,
) -> None:
    config = ApplicationConfig(
        tempo=TempoConfig(bpm=120.0, ticks_per_beat=8, beats_per_bar=4),
        output=OutputConfig(type="console"),
    )
    output_factory.create.return_value = MagicMock()
    inference_factory.create.return_value = MagicMock()
    input_factory.create.return_value = MagicMock()
    service_cls.return_value = MagicMock(running=False)

    session = RuntimeSessionBuilder(config=config, log_dir=str(tmp_path)).build_cli()

    assert session.service is service_cls.return_value
    assert service_cls.call_args.kwargs["tempo"].ticks_per_beat == 8


def test_runtime_session_cleanup_is_idempotent_and_closes_output_after_clear_error() -> None:
    engine = MagicMock()
    engine.clear_history.side_effect = RuntimeError("server unavailable")
    output = MagicMock()
    session = RuntimeSession(
        config=ApplicationConfig(),
        session_manager=None,
        output_sink=output,
        service=MagicMock(running=False),
        session_config={},
        inference_engine=engine,
        emit_output_config=False,
    )

    session.cleanup()
    session.cleanup()

    engine.clear_history.assert_called_once_with()
    output.close.assert_called_once_with()


def test_runtime_session_captures_replay_audit_before_history_clear(tmp_path) -> None:
    calls = []
    model_snapshot = {
        "runtime_info": {
            "seed_provenance_complete": True,
            "trace_capture_complete": True,
            "prompt_sample_seed": 17,
            "continuation_sample_seed": 23,
            "session_id": "server-session",
            "session_epoch": 1,
        },
        "prompt_generation_log": {
            "prompt_tokens": [1],
            "generated_tokens": [1, 2],
        },
        "continuation_generations": [{"request_id": "request-1"}],
    }

    class _PromptClient:
        def replay_audit(self):
            calls.append("replay_audit")
            return model_snapshot

        def clear_history(self):
            calls.append("clear_history")
            return {"success": True}

    class _OrderedSessionLogger(SessionLoggerOutputSink):
        def finalize_prompt_continuation_replay_audit(self, **kwargs):
            calls.append("write_audit")
            return super().finalize_prompt_continuation_replay_audit(**kwargs)

    manager = SessionManager(str(tmp_path), session_id="replay")
    session_dir = manager.create_session_directory()
    output = _OrderedSessionLogger(
        session_dir=session_dir,
        include_midi=False,
        include_json=False,
    )
    output.record_prompt_continuation_session_seed(
        {
            "schema_version": 1,
            "runtime_session_id": "replay",
            "success": True,
            "prompt_requested_seed": 17,
            "prompt_effective_seed": 17,
            "continuation_requested_seed": 23,
            "continuation_effective_seed": 23,
            "prompt_seed_source": "system",
            "continuation_seed_source": "system",
            "session_id": "server-session",
            "session_epoch": 1,
        }
    )
    output.log_prompt_continuation_replay_request(
        {
            "schema_version": 1,
            "sequence": 1,
            "operation": "start",
            "request": {
                "melody_events": [
                    {
                        "type": "note_on",
                        "pitch": 60,
                        "tick": 0,
                        "velocity": 100,
                        "channel": 0,
                        "program": 0,
                    }
                ],
                "observed_until_tick": 32,
            },
            "acknowledgement": {"accepted": True},
        }
    )
    session = RuntimeSession(
        config=ApplicationConfig(continuation_mode="prompt_continuation"),
        session_manager=manager,
        output_sink=output,
        service=MagicMock(running=False),
        session_config={},
        prompt_client=_PromptClient(),
        emit_output_config=False,
    )

    session.cleanup()
    session.cleanup()

    assert calls == ["replay_audit", "write_audit", "clear_history"]
    model_trace = json.loads(
        (session_dir / "prompt_continuation_model_trace.json").read_text(
            encoding="utf-8"
        )
    )
    assert model_trace == model_snapshot
    manifest = json.loads(
        (session_dir / "replay_audit_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["comparable"] is True
    assert (session_dir / "prompt_continuation_replay_melody.json").exists()
    assert (session_dir / "prompt_continuation_replay_melody.mid").exists()


def test_runtime_session_skips_replay_endpoint_without_audit_capable_sink() -> None:
    class _PromptClient:
        def __init__(self) -> None:
            self.replay_audit_calls = 0
            self.clear_history_calls = 0

        def replay_audit(self):
            self.replay_audit_calls += 1
            return {"runtime_info": {"seed_provenance_complete": True}}

        def clear_history(self):
            self.clear_history_calls += 1
            return {"success": True}

    class _PlainOutput:
        def __init__(self) -> None:
            self.close_calls = 0

        def close(self):
            self.close_calls += 1

    client = _PromptClient()
    output = _PlainOutput()
    session = RuntimeSession(
        config=ApplicationConfig(continuation_mode="prompt_continuation"),
        session_manager=None,
        output_sink=output,
        service=MagicMock(running=False),
        session_config={},
        prompt_client=client,
        emit_output_config=False,
    )

    session.cleanup()

    assert client.replay_audit_calls == 0
    assert client.clear_history_calls == 1
    assert output.close_calls == 1
    assert session.metadata == {}


def test_runtime_session_missing_replay_endpoint_warns_without_blocking_cleanup(
    tmp_path,
) -> None:
    class _LegacyPromptClient:
        def __init__(self) -> None:
            self.clear_calls = 0

        def clear_history(self):
            self.clear_calls += 1
            return {"success": True}

    manager = SessionManager(str(tmp_path), session_id="legacy")
    session_dir = manager.create_session_directory()
    output = SessionLoggerOutputSink(
        session_dir=session_dir,
        include_midi=False,
        include_json=False,
    )
    client = _LegacyPromptClient()
    session = RuntimeSession(
        config=ApplicationConfig(continuation_mode="prompt_continuation"),
        session_manager=manager,
        output_sink=output,
        service=MagicMock(running=False),
        session_config={},
        prompt_client=client,
        emit_output_config=False,
    )

    session.cleanup()

    assert client.clear_calls == 1
    assert session.metadata["cleanup_warnings"] == [
        {
            "operation": "prompt_continuation_replay_audit",
            "error": "prompt client does not provide replay_audit()",
        }
    ]
    manifest = json.loads(
        (session_dir / "replay_audit_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "incomplete"
    assert manifest["comparable"] is False
    assert manifest["model_trace_capture_status"] == "unsupported"
    assert manifest["missing_requirements"] == [
        "captured_model_trace",
        "protocol_requests",
        "complete_model_trace",
        "prompt_continuation_model_evidence",
        "complete_seed_provenance",
    ]
