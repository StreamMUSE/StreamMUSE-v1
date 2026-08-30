"""CLI assembly for the terminal-only live rap showcase."""

from __future__ import annotations

import argparse
import importlib
from math import isfinite
import platform
import shutil
import subprocess
import time
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from queue import Queue
from typing import Any, Callable
from uuid import uuid4

import uvicorn

from streammuse.application.rap.monitoring import RapEventDispatcher, RapEventPublisher, RapStateProjector
from streammuse.application.rap.monitoring_payloads import flow_template_payload
from streammuse.application.rap.realtime import RollingRapController
from streammuse.application.rap.runtime import RapAudioDemoDependencies, RapDemoDependencies, RapTickLoop
from streammuse.domain.rap import ScoreWeights
from streammuse.domain.timing import Tempo
from streammuse.infrastructure.inference.local_chat_client import LocalChatModelClient, LocalChatModelClientConfig
from streammuse.infrastructure.rap.fallback import PrevalidatedFallbackCatalog
from streammuse.infrastructure.rap.generators import (
    LocalChatCandidateGenerator,
    PhraseBankGenerator,
    ScriptedFailureGenerator,
)
from streammuse.infrastructure.rap.prosody import CmuProsodyAnalyzer
from streammuse.infrastructure.rap.recorder import RapSessionRecorder, build_session_manifest, event_to_dict
from streammuse.infrastructure.rap.scenarios import default_scenario, load_scenario
from streammuse.infrastructure.rap.templates import BUILTIN_TEMPLATES
from streammuse.presentation.rap_demo.terminal import TerminalRapSink
from streammuse.presentation.rap_demo.server import create_app


class _WebSocketQueueSink:
    def __init__(self, queue: Queue[dict[str, object]]) -> None:
        self._queue = queue

    def __call__(self, event) -> None:
        self._queue.put(event_to_dict(event))


class RapAudioFactories:
    """Lazy construction seam for audio-only adapters and real-device tests."""

    def create_synthesizer(self):
        from streammuse.infrastructure.rap.speech import EspeakPhonemeSynthesizer

        return EspeakPhonemeSynthesizer()

    def create_drums(self, *, seed: int):
        from streammuse.infrastructure.rap.drums import ProceduralBoomBapRenderer

        return ProceduralBoomBapRenderer(seed=seed)

    def create_time_stretcher(self):
        from streammuse.infrastructure.rap.time_stretch import RubberBandTimeStretcher

        return RubberBandTimeStretcher()

    def create_phrase_synthesizer(self, *, analyzer, sample_rate_hz: int):
        from streammuse.infrastructure.rap.speech import EspeakEventPhraseSynthesizer

        return EspeakEventPhraseSynthesizer(
            analyzer,
            output_sample_rate_hz=sample_rate_hz,
        )

    def create_time_map_stretcher(self):
        from streammuse.infrastructure.rap.time_stretch import RubberBandTimeMapStretcher

        return RubberBandTimeMapStretcher()

    def create_remote_client(self, *, base_url: str, clock: Callable[[], float], audio_transport: str):
        from streammuse.infrastructure.rap.remote_chunk_client import RemoteChunkClient

        return RemoteChunkClient(base_url, clock=clock, audio_transport=audio_transport)

    def create_opus_codec(self):
        from streammuse.infrastructure.rap.opus_codec import FFmpegOpusCodec

        return FFmpegOpusCodec()

    def create_sink(
        self,
        *,
        output: str,
        audio_format,
        audio_file: Path,
        audio_device: str | None,
    ):
        from streammuse.infrastructure.rap.audio_output import (
            CompositeAudioSink,
            Float32WavAudioSink,
            SoundDeviceAudioSink,
            TimedAudioSink,
        )

        if output == "live":
            return SoundDeviceAudioSink(audio_format=audio_format, device=audio_device)
        recorder = Float32WavAudioSink(audio_file, audio_format)
        if output == "wav":
            return CompositeAudioSink(TimedAudioSink(audio_format=audio_format), recorder)
        primary = SoundDeviceAudioSink(audio_format=audio_format, device=audio_device)
        fallback = TimedAudioSink(audio_format=audio_format)
        return CompositeAudioSink(primary, recorder, fallback=fallback)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="streammuse-rap-demo",
        description="Run the terminal-only realtime rap showcase.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--scenario", type=Path)
    parser.add_argument("--generator", choices=("phrase_bank", "local_chat", "scripted_failure"), default="phrase_bank")
    parser.add_argument("--model-url", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--model", default="qwen-rap")
    parser.add_argument("--timeout-s", type=float, default=5.0)
    parser.add_argument("--candidate-count", type=int, default=8)
    parser.add_argument("--lookahead-bars", type=int, default=2)
    parser.add_argument("--audio-output", choices=("none", "live", "wav", "composite"), default="none")
    parser.add_argument(
        "--rap-audio-renderer",
        choices=("espeak", "espeak_adaptive", "moss_aligned_remote"),
        default="espeak",
    )
    parser.add_argument("--rap-audio-transport", choices=("pcm", "opus"), default="pcm")
    parser.add_argument("--rap-render-url", default="http://127.0.0.1:8020")
    parser.add_argument("--rap-render-profile", choices=("realtime",), default="realtime")
    parser.add_argument("--rap-render-startup-timeout", type=float, default=120.0)
    parser.add_argument("--rap-render-rolling-timeout", type=float, default=5.0)
    parser.add_argument("--tempo", type=float, default=None, help="Override the scenario playback tempo")
    parser.add_argument("--audio-device", default=None)
    parser.add_argument("--sample-rate", type=int, default=48_000)
    parser.add_argument("--voice", default="en-us")
    parser.add_argument("--voice-speed", type=int, default=175)
    parser.add_argument("--voice-pitch", type=int, default=50)
    parser.add_argument("--max-compression", type=float, default=2.0)
    parser.add_argument("--audio-file", type=Path, default=None)
    parser.add_argument("--minimum-score", type=float, default=0.55)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--max-bars", type=int, default=12, help="Zero runs until interrupted")
    parser.add_argument("--log-dir", type=Path, default=Path("logs/rap"))
    parser.add_argument("--terminal-detail", choices=("summary", "candidates", "full"), default="full")
    parser.add_argument("--terminal-layout", choices=("auto", "split", "stream"), default="auto")
    parser.add_argument("--host", default="127.0.0.1", help="Web monitor bind address")
    parser.add_argument("--port", type=int, default=8012, help="Web monitor port")
    parser.add_argument("--no-web", action="store_true", help="Run only the terminal monitor")
    return parser


def build_demo(
    args: argparse.Namespace,
    *,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    audio_factories: RapAudioFactories | Any | None = None,
) -> RapDemoDependencies | RapAudioDemoDependencies:
    if args.max_bars < 0:
        raise ValueError("max-bars must not be negative")
    if args.candidate_count <= 0 or args.lookahead_bars <= 0:
        raise ValueError("candidate-count and lookahead-bars must be positive")
    if not isfinite(args.timeout_s) or args.timeout_s <= 0:
        raise ValueError("timeout-s must be positive")
    if not isfinite(args.minimum_score) or not 0.0 <= args.minimum_score <= 1.0:
        raise ValueError("minimum-score must be between zero and one")
    _validate_audio_options(args, require_external=audio_factories is None)

    scenario = load_scenario(args.scenario) if args.scenario else default_scenario()
    if not scenario.loop and args.max_bars == 0:
        raise ValueError("max-bars 0 requires a looping scenario")
    if not scenario.loop and args.max_bars > scenario.total_bars:
        raise ValueError("max-bars exceeds the non-looping scenario length")
    tempo = Tempo(args.tempo if args.tempo is not None else scenario.tempo_bpm, 4, 4)
    analyzer = CmuProsodyAnalyzer()
    fallbacks = PrevalidatedFallbackCatalog.build(scenario, BUILTIN_TEMPLATES, analyzer)
    session_id = f"rap-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}"
    session_dir = args.log_dir / session_id
    manifest = _build_manifest(args, scenario, tempo)
    manifest.update(
        {
            "session_id": session_id,
            "max_bars": args.max_bars,
            "terminal_layout": args.terminal_layout,
            "terminal_detail": args.terminal_detail,
        }
    )
    if args.rap_audio_transport != "pcm" and args.rap_audio_renderer != "moss_aligned_remote":
        raise ValueError("rap-audio-transport opus requires moss_aligned_remote")
    if args.rap_audio_renderer == "moss_aligned_remote":
        manifest["generator_config"] = {
            "name": "remote_service",
            "candidate_count": None,
            "prompt_schema_version": "remote_chunk_v1",
            "candidate_parser_version": "remote_chunk_v1",
            "temperature": None,
            "output_length_policy": None,
        }
        manifest["model_config"] = {
            "name": "remote_service",
            "base_url": args.rap_render_url,
            "timeout_seconds": args.rap_render_rolling_timeout,
            "max_retries": None,
            "retry_delay_seconds": None,
            "top_p": None,
            "extra_payload": {"profile": args.rap_render_profile},
        }
    if args.audio_output != "none":
        return _build_audio_demo(
            args,
            tempo=tempo,
            scenario=scenario,
            analyzer=analyzer,
            fallbacks=fallbacks,
            session_dir=session_dir,
            manifest=manifest,
            clock=clock,
            audio_factories=audio_factories or RapAudioFactories(),
        )
    generator, stop_primary, close_primary = _build_generator(args)
    recorder = None
    dispatcher = None
    try:
        recorder = RapSessionRecorder(session_dir, manifest)
        publisher = RapEventPublisher(session_id)
        projector = RapStateProjector()
        websocket_queue: Queue[dict[str, object]] = Queue()
        dispatcher = RapEventDispatcher(
            publisher.queue,
            sinks=(
                recorder,
                projector,
                _WebSocketQueueSink(websocket_queue),
                TerminalRapSink(args.terminal_detail, layout=args.terminal_layout),
            ),
        )
        dispatcher.start()
        controller = RollingRapController(
            tempo=tempo,
            scenario=scenario,
            templates=BUILTIN_TEMPLATES,
            fallback_catalog=fallbacks,
            analyzer=analyzer,
            weights=ScoreWeights(),
            publisher=publisher,
            primary_generator=generator,
            candidate_count=args.candidate_count,
            lookahead_bars=args.lookahead_bars,
            minimum_score=args.minimum_score,
            seed=args.seed,
            planning_bar_limit=args.max_bars or (None if scenario.loop else scenario.total_bars),
            stop_primary=stop_primary,
            close_primary=close_primary,
            monotonic=clock,
        )
        tick_loop = RapTickLoop(tempo, on_tick=controller.on_tick, clock=clock, sleep=sleep)
    except BaseException:
        try:
            if dispatcher is not None:
                dispatcher.flush_and_close()
        finally:
            try:
                if recorder is not None:
                    recorder.close()
            finally:
                try:
                    if stop_primary is not None:
                        stop_primary()
                finally:
                    if close_primary is not None:
                        close_primary()
        raise

    generator_config = manifest["generator_config"]
    model_config = manifest["model_config"]
    assert isinstance(generator_config, dict) and isinstance(model_config, dict)
    session_metadata = {
        "scenario_id": scenario.scenario_id,
        "tempo_bpm": tempo.bpm,
        "ticks_per_beat": tempo.ticks_per_beat,
        "beats_per_bar": tempo.beats_per_bar,
        "max_bars": args.max_bars,
        "generator": generator_config["name"],
        "model_url": model_config["base_url"],
        "model": model_config["name"],
        "generator_config": generator_config,
        "model_config": model_config,
        "candidate_count": args.candidate_count,
        "lookahead_bars": args.lookahead_bars,
        "minimum_score": args.minimum_score,
        "seed": args.seed,
        "score_weights": manifest["score_weights"],
        "terminal_layout": args.terminal_layout,
        "terminal_detail": args.terminal_detail,
    }
    return RapDemoDependencies(
        tempo,
        controller,
        publisher,
        dispatcher,
        tick_loop,
        session_dir,
        session_metadata=session_metadata,
        recorder=recorder,
        projector=projector,
        websocket_queue=websocket_queue,
        configured_max_bars=args.max_bars,
    )


def _build_audio_demo(
    args: argparse.Namespace,
    *,
    tempo: Tempo,
    scenario,
    analyzer,
    fallbacks,
    session_dir: Path,
    manifest: dict[str, object],
    clock: Callable[[], float],
    audio_factories: RapAudioFactories | Any,
) -> RapAudioDemoDependencies:
    """Assemble the audio graph only after explicit audio mode selection."""

    from streammuse.application.rap.audio_coordination import BarAudioCoordinator
    from streammuse.application.rap.bar_renderer import DeterministicRapBarRenderer
    from streammuse.application.rap.chunk_audio import RemoteMossChunkPreparationStrategy
    from streammuse.application.rap.chunk_realtime import RollingRapChunkController
    from streammuse.application.rap.audio_service import RapAudioController
    from streammuse.application.rap.playback import RapPlaybackService
    from streammuse.domain.rap import AudioFormat, RemoteCandidatePolicy

    audio_format = AudioFormat(sample_rate_hz=args.sample_rate, channels=2, sample_width_bytes=4)
    audio_file = args.audio_file or session_dir / "mixed.wav"
    manifest["audio"] = {
        "output": args.audio_output,
        "sample_rate_hz": audio_format.sample_rate_hz,
        "channels": audio_format.channels,
        "sample_width_bytes": audio_format.sample_width_bytes,
        "voice": args.voice,
        "voice_speed": args.voice_speed,
        "voice_pitch": args.voice_pitch,
        "max_compression": args.max_compression,
        "time_stretcher": (
            "rubberband_r3_sparse_timemap"
            if args.rap_audio_renderer == "espeak_adaptive"
            else "rubberband_r3"
        ),
        "adaptive_anchor_policy": (
            "gate_d_adaptive_error"
            if args.rap_audio_renderer == "espeak_adaptive"
            else None
        ),
        "audio_device": args.audio_device,
        "renderer": args.rap_audio_renderer,
        "transport": args.rap_audio_transport if args.rap_audio_renderer == "moss_aligned_remote" else "pcm",
        "render_url": args.rap_render_url if args.rap_audio_renderer == "moss_aligned_remote" else None,
        "render_profile": args.rap_render_profile if args.rap_audio_renderer == "moss_aligned_remote" else None,
        "render_startup_timeout_seconds": args.rap_render_startup_timeout,
        "render_rolling_timeout_seconds": args.rap_render_rolling_timeout,
        "artifact_paths": {"wav": str(audio_file)} if args.audio_output in ("wav", "composite") else {},
    }
    generator = stop_primary = close_primary = None
    if args.rap_audio_renderer in {"espeak", "espeak_adaptive"}:
        generator, stop_primary, close_primary = _build_generator(args)
    recorder = None
    dispatcher = None
    coordinator = None
    playback = None
    controller: RapAudioController | None = None
    remote_client = None
    try:
        recorder = RapSessionRecorder(session_dir, manifest)
        publisher = RapEventPublisher(str(manifest["session_id"]))
        projector = RapStateProjector()
        websocket_queue: Queue[dict[str, object]] = Queue()
        dispatcher = RapEventDispatcher(
            publisher.queue,
            sinks=(
                recorder,
                projector,
                _WebSocketQueueSink(websocket_queue),
                TerminalRapSink(args.terminal_detail, layout=args.terminal_layout),
            ),
        )
        dispatcher.start()
        synthesizer = audio_factories.create_synthesizer()
        drums = audio_factories.create_drums(seed=args.seed)
        time_stretcher = audio_factories.create_time_stretcher()
        fallback_renderer = DeterministicRapBarRenderer(
            tempo=tempo,
            audio_format=audio_format,
            synthesizer=synthesizer,
            drums=drums,
            time_stretcher=time_stretcher,
            voice=args.voice,
            speed_wpm=args.voice_speed,
            pitch=args.voice_pitch,
            max_compression=args.max_compression,
        )
        renderer = fallback_renderer
        if args.rap_audio_renderer == "espeak_adaptive":
            from streammuse.application.rap.adaptive_bar_renderer import (
                AdaptiveContinuousRapBarRenderer,
            )

            renderer = AdaptiveContinuousRapBarRenderer(
                tempo=tempo,
                audio_format=audio_format,
                phrase_synthesizer=audio_factories.create_phrase_synthesizer(
                    analyzer=analyzer,
                    sample_rate_hz=audio_format.sample_rate_hz,
                ),
                drums=drums,
                time_map_stretcher=audio_factories.create_time_map_stretcher(),
                fallback_renderer=fallback_renderer,
                voice=args.voice,
                speed_wpm=args.voice_speed,
                pitch=args.voice_pitch,
            )
        sink = audio_factories.create_sink(
            output=args.audio_output,
            audio_format=audio_format,
            audio_file=audio_file,
            audio_device=args.audio_device,
        )
        playback = RapPlaybackService(
            tempo=tempo,
            sink=sink,
            publisher=publisher,
            on_tick=lambda tick: controller.on_tick(tick),
            monotonic=clock,
        )
        planning_bar_limit = args.max_bars or (None if scenario.loop else scenario.total_bars)
        if args.rap_audio_renderer in {"espeak", "espeak_adaptive"}:
            coordinator = BarAudioCoordinator(renderer, publisher=publisher)
            controller = RollingRapController(
                tempo=tempo,
                scenario=scenario,
                templates=BUILTIN_TEMPLATES,
                fallback_catalog=fallbacks,
                analyzer=analyzer,
                weights=ScoreWeights(),
                publisher=publisher,
                primary_generator=generator,
                candidate_count=args.candidate_count,
                lookahead_bars=args.lookahead_bars,
                minimum_score=args.minimum_score,
                seed=args.seed,
                planning_bar_limit=planning_bar_limit,
                stop_primary=stop_primary,
                close_primary=close_primary,
                audio_coordinator=coordinator,
                on_audio_committed=playback.enqueue,
                monotonic=clock,
            )
        else:
            if args.rap_audio_transport == "opus":
                audio_factories.create_opus_codec().probe()
            remote_client = audio_factories.create_remote_client(
                base_url=args.rap_render_url,
                clock=clock,
                audio_transport=args.rap_audio_transport,
            )
            try:
                health = remote_client.health(timeout_seconds=min(5.0, args.rap_render_startup_timeout))
                if not health.ready:
                    raise OSError("remote rap renderer is not ready")
                strategy = RemoteMossChunkPreparationStrategy(
                    client=remote_client,
                    audio_format=audio_format,
                    drums=drums,
                    prosody=analyzer,
                    tempo=tempo,
                    clock=clock,
                )
                controller = RollingRapChunkController(
                    tempo=tempo,
                    scenario=scenario,
                    templates=BUILTIN_TEMPLATES,
                    fallback_catalog=fallbacks,
                    analyzer=analyzer,
                    fallback_renderer=renderer,
                    preparation_strategy=strategy,
                    publisher=publisher,
                    enqueue=playback.enqueue,
                    session_id=str(manifest["session_id"]),
                    policy=RemoteCandidatePolicy.realtime_default(),
                    seed=args.seed,
                    planning_bar_limit=planning_bar_limit,
                    startup_timeout_seconds=args.rap_render_startup_timeout,
                    rolling_timeout_seconds=args.rap_render_rolling_timeout,
                    monotonic=clock,
                )
            except BaseException:
                remote_client.close()
                remote_client = None
                raise
    except BaseException:
        if playback is not None:
            playback.close()
        if controller is not None:
            controller.close()
        elif coordinator is not None:
            coordinator.close()
        elif remote_client is not None:
            remote_client.close()
        if dispatcher is not None:
            dispatcher.flush_and_close()
        if recorder is not None:
            recorder.close()
        if stop_primary is not None:
            stop_primary()
        if close_primary is not None:
            close_primary()
        raise

    generator_config = manifest["generator_config"]
    model_config = manifest["model_config"]
    assert isinstance(generator_config, dict) and isinstance(model_config, dict)
    session_metadata = {
        "scenario_id": scenario.scenario_id,
        "tempo_bpm": tempo.bpm,
        "ticks_per_beat": tempo.ticks_per_beat,
        "beats_per_bar": tempo.beats_per_bar,
        "max_bars": args.max_bars,
        "generator": generator_config["name"],
        "model_url": model_config["base_url"],
        "model": model_config["name"],
        "generator_config": generator_config,
        "model_config": model_config,
        "candidate_count": args.candidate_count,
        "lookahead_bars": args.lookahead_bars,
        "minimum_score": args.minimum_score,
        "seed": args.seed,
        "score_weights": manifest["score_weights"],
        "terminal_layout": args.terminal_layout,
        "terminal_detail": args.terminal_detail,
        "audio": manifest["audio"],
    }
    return RapAudioDemoDependencies(
        tempo=tempo,
        controller=controller,
        coordinator=coordinator,
        playback=playback,
        publisher=publisher,
        dispatcher=dispatcher,
        session_dir=session_dir,
        session_metadata=session_metadata,
        recorder=recorder,
        projector=projector,
        websocket_queue=websocket_queue,
        configured_max_bars=args.max_bars,
    )


def _validate_audio_options(args: argparse.Namespace, *, require_external: bool) -> None:
    if args.tempo is not None and (not isfinite(args.tempo) or args.tempo <= 0):
        raise ValueError("tempo must be positive")
    if not isfinite(args.sample_rate) or args.sample_rate <= 0:
        raise ValueError("sample-rate must be positive")
    if args.audio_output != "none" and args.sample_rate != 48_000:
        raise ValueError("realtime rap audio requires a 48,000 Hz sample rate")
    if not 80 <= args.voice_speed <= 450:
        raise ValueError("voice-speed must be between 80 and 450")
    if not 0 <= args.voice_pitch <= 99:
        raise ValueError("voice-pitch must be between 0 and 99")
    if not isfinite(args.max_compression) or not 1.0 <= args.max_compression <= 4.0:
        raise ValueError("max-compression must be between 1.0 and 4.0")
    if args.rap_audio_renderer == "moss_aligned_remote":
        if args.audio_output == "none":
            raise ValueError("moss_aligned_remote requires audio output")
        if args.max_bars and args.max_bars % 2:
            raise ValueError("remote max-bars must be zero or even")
        if args.lookahead_bars != 2:
            raise ValueError("moss_aligned_remote requires lookahead-bars 2")
        if not isinstance(args.rap_render_url, str) or not args.rap_render_url:
            raise ValueError("rap-render-url must not be empty")
        if not isfinite(args.rap_render_startup_timeout) or args.rap_render_startup_timeout <= 0:
            raise ValueError("rap-render-startup-timeout must be positive")
        if not isfinite(args.rap_render_rolling_timeout) or args.rap_render_rolling_timeout <= 0:
            raise ValueError("rap-render-rolling-timeout must be positive")
    if args.audio_output == "none" or not require_external:
        return
    if not isinstance(args.voice, str) or not args.voice:
        raise ValueError("voice must not be empty")
    if shutil.which("espeak-ng") is None:
        raise OSError("audio output requires the espeak-ng executable")
    if shutil.which("rubberband") is None:
        raise OSError("audio output requires the rubberband executable")
    if args.audio_output in ("live", "composite"):
        try:
            importlib.import_module("sounddevice")
        except ImportError as error:
            raise OSError("live audio output requires sounddevice") from error


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        demo = build_demo(args)
        if args.no_web:
            if getattr(demo, "autostart", True):
                demo.run(max_bars=args.max_bars)
            else:
                try:
                    demo.start()
                finally:
                    demo.close()
        else:
            app = create_app(runtime=demo, projector=demo.projector, websocket_queue=demo.websocket_queue)
            print(f"Rap monitor: http://{args.host}:{args.port}")
            uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}")
        return 2
    return 0


def _build_generator(args: argparse.Namespace):
    if args.generator == "phrase_bank":
        return PhraseBankGenerator(), None, None
    if args.generator == "scripted_failure":
        return ScriptedFailureGenerator(), None, None
    client = LocalChatModelClient(
        LocalChatModelClientConfig(
            base_url=args.model_url,
            model=args.model,
            timeout_s=args.timeout_s,
        )
    )
    return LocalChatCandidateGenerator(client), client.abort, client.close


def _build_manifest(args: argparse.Namespace, scenario, tempo: Tempo) -> dict[str, object]:
    revision, dirty, git_state_known = _repository_state()
    templates = []
    seen: set[str] = set()
    for segment in scenario.segments:
        if segment.template_id in seen:
            continue
        seen.add(segment.template_id)
        payload = flow_template_payload(BUILTIN_TEMPLATES.get(segment.template_id))
        templates.append(
            {
                "template_id": payload["template_id"],
                "name": payload["name"],
                "definition": {
                    "ticks_per_beat": payload["ticks_per_beat"],
                    "beats_per_bar": payload["beats_per_bar"],
                    "slots": [
                        {
                            "tick_in_bar": slot["tick_in_bar"],
                            "duration_ticks": slot["duration_ticks"],
                            "target_stress": slot["target_stress"],
                            "boundary_strength": slot["boundary_strength"],
                            "rhyme_group": slot["rhyme_group"],
                        }
                        for slot in payload["slots"]
                    ],
                },
                "provenance": dict(payload["provenance"]),
            }
        )
    scenario_payload = {
        "scenario_id": scenario.scenario_id,
        "tempo_bpm": tempo.bpm,
        "loop": scenario.loop,
        "segments": [
            {
                "start_bar": segment.start_bar,
                "bars": segment.bars,
                "topic": segment.topic,
                "template_id": segment.template_id,
                "fallback_lines": list(segment.fallback_lines),
            }
            for segment in scenario.segments
        ],
    }
    try:
        package_version = version("streammuse")
    except PackageNotFoundError:
        package_version = "0.1.0"
    manifest = build_session_manifest(
        scenario_id=scenario.scenario_id,
        scenario=scenario_payload,
        seed=args.seed,
        tempo={
            "bpm": tempo.bpm,
            "ticks_per_beat": tempo.ticks_per_beat,
            "beats_per_bar": tempo.beats_per_bar,
        },
        templates=templates,
        generator_config={
            "name": args.generator,
            "candidate_count": args.candidate_count,
            "prompt_schema_version": "beat_aligned_flow_v1",
            "candidate_parser_version": "plain_lines_v1",
            "temperature": 0.8 if args.generator == "local_chat" else None,
            "output_length_policy": "max(64,candidate_count*24)" if args.generator == "local_chat" else None,
        },
        model_config={
            "name": args.model if args.generator == "local_chat" else "none",
            "base_url": args.model_url if args.generator == "local_chat" else None,
            "timeout_seconds": args.timeout_s if args.generator == "local_chat" else None,
            "max_retries": 0 if args.generator == "local_chat" else None,
            "retry_delay_seconds": 0.25 if args.generator == "local_chat" else None,
            "top_p": None,
            "extra_payload": None,
        },
        score_weights=asdict(ScoreWeights()),
        minimum_score=args.minimum_score,
        timeout_seconds=args.timeout_s,
        lookahead_bars=args.lookahead_bars,
        python_version=platform.python_version(),
        platform=platform.platform(),
        package_version=package_version,
        git_revision=revision,
        git_dirty=dirty,
    )
    manifest["git_state_known"] = git_state_known
    return manifest


def _repository_state() -> tuple[str, bool, bool]:
    root = Path(__file__).resolve().parents[4]
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True, timeout=2.0
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"], cwd=root, check=True, capture_output=True, text=True, timeout=2.0
            ).stdout.strip()
        )
        return revision or "unknown", dirty, bool(revision)
    except (OSError, subprocess.SubprocessError):
        return "unknown", True, False


if __name__ == "__main__":
    raise SystemExit(main())
