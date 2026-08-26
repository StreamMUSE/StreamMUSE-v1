"""Shared builder for StreamMUSE runtime sessions."""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import Any

from streammuse.application.config import ApplicationConfig
from streammuse.application.factories import (
    InferenceEngineFactory,
    InputSourceFactory,
    OutputSinkFactory,
)
from streammuse.application.runtime.session import RuntimeSession
from streammuse.application.services.input_timing import effective_input_snap_forward_fraction
from streammuse.application.services.real_time_music_service import RealTimeMusicService
from streammuse.domain.logging import SessionManager
from streammuse.domain.timing import PlaybackScheduler, Tempo
from streammuse.infrastructure.output import (
    AudioOutputConfig,
    AudioOutputSink,
    CompositeOutputSink,
    ConsoleOutputConfig,
    ConsoleOutputSink,
    MetronomeOutputConfig,
    MetronomeOutputSink,
    MidiFileOutputConfig,
    MidiFileOutputSink,
    WebSocketOutputSink,
)


class RuntimeSessionBuilder:
    def __init__(
        self,
        *,
        config: ApplicationConfig,
        log_dir: str = "logs",
        prompt_client_override: Any | None = None,
        output_sink_override: Any | None = None,
        before_input_create: Callable[[Any | None, Any | None, Any], None] | None = None,
        tick_observer_factory: Callable[[Tempo], Any | None] | None = None,
    ) -> None:
        self.config = config
        self.log_dir = log_dir
        self.prompt_client_override = prompt_client_override
        self.output_sink_override = output_sink_override
        self.before_input_create = before_input_create
        self.tick_observer_factory = tick_observer_factory

    def build_cli(self) -> RuntimeSession:
        session_manager = None
        if self.config.output.type != "midi_file":
            session_manager = self._create_session_manager(
                save_config=self.config.output.session_artifact_tier == "debug"
            )
        output_sink = self.output_sink_override or OutputSinkFactory.create(self.config, session_manager)
        return self._build(
            session_manager=session_manager,
            output_sink=output_sink,
            emit_output_config=session_manager is not None,
            write_summary_on_cleanup=(
                session_manager is not None
                and self.config.output.session_artifact_tier == "debug"
            ),
        )

    def build_web(self) -> RuntimeSession:
        session_manager = self._create_session_manager(save_config=True, ensure_unique=True)
        output_sink, websocket_sink = self._build_web_composite(session_manager=session_manager)
        return self._build(
            session_manager=session_manager,
            output_sink=output_sink,
            websocket_sink=websocket_sink,
            emit_output_config=True,
            write_summary_on_cleanup=False,
        )

    def _create_session_manager(
        self,
        *,
        save_config: bool,
        ensure_unique: bool = False,
    ) -> SessionManager:
        session_manager = SessionManager(self.log_dir)
        if ensure_unique:
            base_session_id = session_manager.get_session_id()
            suffix = 1
            while session_manager.get_session_dir().exists():
                session_manager = SessionManager(
                    self.log_dir,
                    session_id=f"{base_session_id}_{suffix:02d}",
                )
                suffix += 1
        session_manager.create_session_directory()
        if save_config:
            session_manager.save_config(self._session_config())
        return session_manager

    def _session_config(self) -> dict[str, Any]:
        return {
            "tempo_bpm": self.config.tempo.bpm,
            "ticks_per_beat": self.config.tempo.ticks_per_beat,
            "beats_per_bar": self.config.tempo.beats_per_bar,
            "input_type": self.config.input.type,
            "output_type": self.config.output.type,
            "close_active_notes_on_finalize": (
                self.config.output.close_active_notes_on_finalize
            ),
            "metronome_enabled": self.config.output.metronome_enabled,
            "metronome_port": self.config.output.metronome_port,
            "metronome_channel": self.config.output.metronome_channel,
            "count_in_beats": self.config.count_in_beats,
            "input_snap_forward_fraction": self._input_snap_forward_fraction(),
            "continuation_mode": self._continuation_mode(),
            "inference_type": self.config.inference.type,
            "prompt_length_ticks": self._prompt_length_ticks(default=None),
            "generation_interval_ticks": self.config.inference.generation_interval_ticks,
            "generation_length_frames": self.config.inference.generation_length_frames,
            "session_artifact_tier": self.config.output.session_artifact_tier,
            "midi_file_trim_leading_rest": self.config.input.midi_file_trim_leading_rest,
            "rap_enabled": self.config.rap.topic is not None,
            "rap_topic": self.config.rap.topic,
            "rap_pattern": self.config.rap.pattern,
            "rap_generator": self.config.rap.generator,
            "rap_lookahead_bars": self.config.rap.lookahead_bars,
        }

    def _build(
        self,
        *,
        session_manager: SessionManager | None,
        output_sink: Any,
        websocket_sink: WebSocketOutputSink | None = None,
        emit_output_config: bool = True,
        write_summary_on_cleanup: bool = True,
    ) -> RuntimeSession:
        tempo = Tempo(
            bpm=self.config.tempo.bpm,
            ticks_per_beat=self.config.tempo.ticks_per_beat,
            beats_per_bar=self.config.tempo.beats_per_bar,
        )
        scheduler = PlaybackScheduler()

        inference_engine = None
        prompt_client = None
        if self._continuation_mode() == "prompt_continuation":
            if self.config.rap.topic:
                raise ValueError("rap cannot be combined with prompt_continuation mode")
            service_cls = self._prompt_continuation_service_cls()
            prompt_client = self.prompt_client_override or self._create_prompt_client()
            if self.before_input_create is not None:
                self.before_input_create(None, prompt_client, output_sink)
            input_source = InputSourceFactory.create(self.config)
            service = service_cls(
                input_source=input_source,
                prompt_client=prompt_client,
                output_sink=output_sink,
                tempo=tempo,
                scheduler=scheduler,
                prompt_length_ticks=self._prompt_length_ticks(default=32),
                generation_interval_ticks=self.config.inference.generation_interval_ticks,
                count_in_beats=self.config.count_in_beats,
            )
        else:
            inference_engine = InferenceEngineFactory.create(self.config)
            if self.before_input_create is not None:
                self.before_input_create(inference_engine, None, output_sink)
            input_source = InputSourceFactory.create(self.config)
            tick_observer = (
                self.tick_observer_factory(tempo)
                if self.tick_observer_factory is not None
                else None
            )
            service = RealTimeMusicService(
                input_source=input_source,
                inference_engine=inference_engine,
                output_sink=output_sink,
                tempo=tempo,
                scheduler=scheduler,
                generation_interval_ticks=self.config.inference.generation_interval_ticks,
                generation_length_frames=self.config.inference.generation_length_frames,
                count_in_beats=self.config.count_in_beats,
                input_snap_forward_fraction=self._input_snap_forward_fraction(),
                tick_observer=tick_observer,
            )

        return RuntimeSession(
            config=self.config,
            session_manager=session_manager,
            output_sink=output_sink,
            service=service,
            session_config=self._session_config(),
            inference_engine=inference_engine,
            prompt_client=prompt_client,
            websocket_sink=websocket_sink,
            emit_output_config=emit_output_config,
            write_summary_on_cleanup=write_summary_on_cleanup,
        )

    def _continuation_mode(self) -> str:
        return str(getattr(self.config, "continuation_mode", "standard"))

    def _prompt_length_ticks(self, *, default: int | None) -> int | None:
        return getattr(self.config.inference, "prompt_length_ticks", default)

    def _input_snap_forward_fraction(self) -> float:
        return effective_input_snap_forward_fraction(
            self.config.input.type,
            float(getattr(self.config, "input_snap_forward_fraction", 0.0)),
        )

    def _prompt_continuation_service_cls(self) -> Any:
        try:
            from streammuse.application.services.prompt_continuation_realtime_service import (
                PromptContinuationRealtimeService,
            )
        except ImportError as exc:
            raise RuntimeError(
                "prompt-continuation runtime requires the prompt-continuation "
                "service package to be present on this branch"
            ) from exc
        return PromptContinuationRealtimeService

    def _create_prompt_client(self) -> Any:
        try:
            from streammuse.infrastructure.inference.prompt_continuation_http_client import (
                PromptContinuationHttpClient,
                PromptContinuationHttpClientConfig,
            )
        except ImportError as exc:
            raise RuntimeError(
                "prompt-continuation runtime requires the prompt-continuation "
                "HTTP client package to be present on this branch"
            ) from exc
        return PromptContinuationHttpClient(
            PromptContinuationHttpClientConfig(
                base_url=self.config.inference.server_generate_url,
                timeout_s=float(self.config.inference.timeout_s),
                model_name="lekai_prompt_continuation",
                inference_mode=self.config.inference.inference_mode,
                checkpoint_path=self.config.inference.checkpoint_path,
            )
        )

    def _build_web_composite(
        self,
        *,
        session_manager: SessionManager,
    ) -> tuple[CompositeOutputSink, WebSocketOutputSink]:
        ws_sink = WebSocketOutputSink()
        auto_midi = MidiFileOutputSink(
            MidiFileOutputConfig(
                bpm=float(self.config.tempo.bpm),
                ticks_per_beat=int(self.config.tempo.ticks_per_beat),
                beats_per_bar=int(self.config.tempo.beats_per_bar),
                output_path=str(session_manager.get_session_dir() / "combined.mid"),
                record_metronome=bool(self.config.output.metronome_enabled),
                close_active_notes_on_finalize=bool(
                    self.config.output.close_active_notes_on_finalize
                ),
            )
        )
        sinks: list[Any] = [ws_sink, auto_midi, ConsoleOutputSink(ConsoleOutputConfig())]
        if self.config.output.midi_out_port:
            try:
                audio_sink = AudioOutputSink(AudioOutputConfig(port_name=self.config.output.midi_out_port))
                audio_sink._ensure_port()  # noqa: SLF001
                sinks.append(audio_sink)
            except Exception as exc:
                print(
                    "  Audio output: DISABLED - "
                    f"could not open '{self.config.output.midi_out_port}': {exc}",
                    file=sys.stderr,
                    flush=True,
                )
        if self.config.output.metronome_enabled:
            sinks.append(
                MetronomeOutputSink(
                    MetronomeOutputConfig(
                        port_name=(
                            self.config.output.metronome_port
                            or self.config.output.midi_out_port
                        ),
                        ticks_per_beat=int(self.config.tempo.ticks_per_beat),
                        beats_per_bar=int(self.config.tempo.beats_per_bar),
                        channel=int(self.config.output.metronome_channel),
                    )
                )
            )
        return CompositeOutputSink(sinks), ws_sink
