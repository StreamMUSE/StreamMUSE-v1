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
from streammuse.application.services.prompt_continuation_realtime_service import (
    PromptContinuationRealtimeService,
)
from streammuse.application.services.real_time_music_service import RealTimeMusicService
from streammuse.domain.logging import SessionManager
from streammuse.domain.timing import PlaybackScheduler, Tempo
from streammuse.infrastructure.inference.prompt_continuation_http_client import (
    PromptContinuationHttpClient,
    PromptContinuationHttpClientConfig,
)
from streammuse.infrastructure.output import (
    AudioOutputConfig,
    AudioOutputSink,
    CompositeOutputSink,
    ConsoleOutputConfig,
    ConsoleOutputSink,
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
    ) -> None:
        self.config = config
        self.log_dir = log_dir
        self.prompt_client_override = prompt_client_override
        self.output_sink_override = output_sink_override
        self.before_input_create = before_input_create

    def build_cli(self) -> RuntimeSession:
        session_manager = self._create_session_manager()
        output_sink = self.output_sink_override or OutputSinkFactory.create(self.config, session_manager)
        return self._build(session_manager=session_manager, output_sink=output_sink)

    def build_web(self) -> RuntimeSession:
        session_manager = self._create_session_manager()
        output_sink, websocket_sink = self._build_web_composite(session_manager=session_manager)
        return self._build(
            session_manager=session_manager,
            output_sink=output_sink,
            websocket_sink=websocket_sink,
        )

    def _create_session_manager(self) -> SessionManager:
        session_manager = SessionManager(self.log_dir)
        session_manager.create_session_directory()
        session_manager.save_config(self._session_config())
        return session_manager

    def _session_config(self) -> dict[str, Any]:
        return {
            "tempo_bpm": self.config.tempo.bpm,
            "ticks_per_beat": self.config.tempo.ticks_per_beat,
            "beats_per_bar": self.config.tempo.beats_per_bar,
            "input_type": self.config.input.type,
            "output_type": self.config.output.type,
            "metronome_enabled": self.config.output.metronome_enabled,
            "metronome_port": self.config.output.metronome_port,
            "metronome_channel": self.config.output.metronome_channel,
            "count_in_beats": self.config.count_in_beats,
            "continuation_mode": self.config.continuation_mode,
            "inference_type": self.config.inference.type,
            "prompt_length_ticks": self.config.inference.prompt_length_ticks,
            "generation_interval_ticks": self.config.inference.generation_interval_ticks,
            "generation_length_frames": self.config.inference.generation_length_frames,
        }

    def _build(
        self,
        *,
        session_manager: SessionManager,
        output_sink: Any,
        websocket_sink: WebSocketOutputSink | None = None,
    ) -> RuntimeSession:
        tempo = Tempo(
            bpm=self.config.tempo.bpm,
            ticks_per_beat=self.config.tempo.ticks_per_beat,
            beats_per_bar=self.config.tempo.beats_per_bar,
        )
        scheduler = PlaybackScheduler()

        inference_engine = None
        prompt_client = None
        if self.config.continuation_mode == "prompt_continuation":
            prompt_client = self.prompt_client_override or self._create_prompt_client()
            if self.before_input_create is not None:
                self.before_input_create(None, prompt_client, output_sink)
            input_source = InputSourceFactory.create(self.config)
            service = PromptContinuationRealtimeService(
                input_source=input_source,
                prompt_client=prompt_client,
                output_sink=output_sink,
                tempo=tempo,
                scheduler=scheduler,
                prompt_length_ticks=self.config.inference.prompt_length_ticks,
                generation_interval_ticks=self.config.inference.generation_interval_ticks,
            )
        else:
            inference_engine = InferenceEngineFactory.create(self.config)
            if self.before_input_create is not None:
                self.before_input_create(inference_engine, None, output_sink)
            input_source = InputSourceFactory.create(self.config)
            service = RealTimeMusicService(
                input_source=input_source,
                inference_engine=inference_engine,
                output_sink=output_sink,
                tempo=tempo,
                scheduler=scheduler,
                generation_interval_ticks=self.config.inference.generation_interval_ticks,
                generation_length_frames=self.config.inference.generation_length_frames,
                count_in_beats=self.config.count_in_beats,
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
        )

    def _create_prompt_client(self) -> PromptContinuationHttpClient:
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
                output_path=str(session_manager.get_session_dir() / "combined.mid"),
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
        return CompositeOutputSink(sinks), ws_sink
