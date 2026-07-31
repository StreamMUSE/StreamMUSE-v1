"""CLI entry point for StreamMUSE realtime task runs."""

from __future__ import annotations

import argparse
import json
import math
import uuid
from contextlib import ExitStack, contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Iterator

from streammuse.application.tasks import (
    InteractiveTaskRunResult,
    InteractiveTaskRuntime,
    InteractiveTaskRuntimeConfig,
    SpeechOutputConfig,
    StdTerminalIO,
    TaskRunResult,
    TaskRuntime,
    TaskRuntimeConfig,
    TerminalIO,
)
from streammuse.application.tasks.human_input import HumanInputConfig, VoiceInputConfig
from streammuse.domain.tasks import AnimalNamingTask, DeadlineMode, InteractiveTask, RealtimeTask, ZipZapZopTask
from streammuse.infrastructure.inference.local_chat_client import (
    LocalChatModelClient,
    LocalChatModelClientConfig,
)


TASKS = {"animal_naming": AnimalNamingTask, "zip_zap_zop": ZipZapZopTask}
INTERACTIVE_TASKS = {"zip_zap_zop"}
RUNNERS = {"offline_benchmark", "realtime_loop"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run StreamMUSE generic realtime tasks")
    subcommands = parser.add_subparsers(dest="command")

    run = subcommands.add_parser("run", help="Run a task against a local model server")
    _add_common_task_args(run, max_tokens_default=32)
    run.add_argument("--runner", choices=sorted(RUNNERS), default="offline_benchmark")
    run.add_argument("--tick-rate-hz", type=float, default=1.0)
    run.add_argument(
        "--history-limit",
        type=int,
        default=0,
        help="Recent turns of history shown to the model each turn (0 = memoryless, current default)",
    )
    run.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Optional nucleus sampling value sent to the OpenAI-compatible server",
    )
    run.add_argument("--live-output", action="store_true", help="Print each completed task turn to stdout")

    play = subcommands.add_parser("play", help="Play an interactive task with a local model server")
    _add_common_task_args(play, max_tokens_default=8)
    first_actor = play.add_mutually_exclusive_group()
    first_actor.add_argument("--human-first", dest="human_first", action="store_true", default=True)
    first_actor.add_argument("--llm-first", dest="human_first", action="store_false")
    play.add_argument("--show-expected", action="store_true")
    play.add_argument("--history-limit", type=int, default=8)
    play.add_argument("--deadline-mode", choices=["menu", "soft", "hard", "challenge"], default="menu")
    play.add_argument("--challenge-stage-turns", type=int, default=20)
    play.add_argument(
        "--challenge-deadline-ms-list",
        type=_parse_deadline_ms_list,
        default=(10000.0, 5000.0, 3000.0, 2000.0, 1000.0),
    )
    play.add_argument("--human-input", choices=["terminal", "voice"], default="terminal")
    play.add_argument("--voice-model", default=None, help="faster-whisper model (voice default: tiny.en)")
    play.add_argument("--voice-device", default=None, help="faster-whisper device (voice default: cpu)")
    play.add_argument("--voice-compute-type", default=None, help="CTranslate2 compute type (voice default: int8)")
    play.add_argument("--microphone-device", type=_parse_microphone_device, default=None)
    play.add_argument("--voice-model-cache", default=None)
    play.add_argument("--voice-model-revision", default=None)
    play.add_argument("--voice-local-files-only", action="store_true")
    play.add_argument("--voice-save-audio", action="store_true")
    play.add_argument("--speech-output", choices=["off", "audio"], default=None)
    play.add_argument(
        "--speech-backend",
        choices=["system", "espeak_ng", "kokoro", "null"],
        default=None,
    )
    play.add_argument("--speech-voice", default=None)
    play.add_argument("--speech-rate", type=_positive_finite_float, default=None)
    play.add_argument(
        "--speaker-device",
        type=_parse_audio_device,
        default=None,
    )
    play.add_argument("--speech-model", default=None)
    play.add_argument("--speech-model-cache", default=None)
    play.add_argument("--speech-model-revision", default=None)
    play.add_argument(
        "--speech-local-files-only",
        action="store_true",
        default=None,
    )
    play.add_argument(
        "--speech-synthesis-timeout-s",
        type=_positive_finite_float,
        default=None,
    )
    prewarm = play.add_mutually_exclusive_group()
    prewarm.add_argument(
        "--speech-prewarm",
        dest="speech_prewarm",
        action="store_true",
        default=None,
    )
    prewarm.add_argument(
        "--no-speech-prewarm",
        dest="speech_prewarm",
        action="store_false",
    )
    play.add_argument(
        "--speech-cache-miss",
        choices=["synthesize", "skip"],
        default=None,
    )
    play.add_argument("--speech-cache-max-entries", type=int, default=None)
    play.add_argument("--speech-cache-max-bytes", type=int, default=None)
    play.add_argument("--speech-guard-ms", type=float, default=None)
    play.add_argument(
        "--speech-on-error",
        choices=["fail", "warn"],
        default=None,
    )
    play.add_argument("--speech-save-audio", action="store_true", default=None)
    play.add_argument(
        "--llm-deadline-basis",
        choices=["text", "audio_end"],
        default=None,
    )

    subcommands.add_parser("voice-devices", help="List microphone input devices without loading a model")
    subcommands.add_parser(
        "speaker-devices",
        help="List speaker output devices without loading a TTS model",
    )
    return parser


def _add_common_task_args(parser: argparse.ArgumentParser, *, max_tokens_default: int) -> None:
    parser.add_argument("--task", required=True)
    parser.add_argument("--max-turns", type=int, default=20)
    parser.add_argument("--model-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="local-model")
    parser.add_argument("--timeout-s", type=float, default=30.0)
    parser.add_argument("--max-tokens", type=int, default=max_tokens_default)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--deadline-ms", type=_positive_finite_float, default=1000.0)
    parser.add_argument("--output-dir", default="task_runs")
    parser.add_argument("--start-number", type=int, default=1)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "voice-devices":
        return _list_voice_devices()
    if args.command == "speaker-devices":
        return _list_speaker_devices()
    if args.command not in {"run", "play"}:
        parser.print_help()
        return 2
    if args.task not in TASKS:
        print(f"unsupported task: {args.task}")
        return 2
    if args.command == "play" and args.task not in INTERACTIVE_TASKS:
        print(f"unsupported interactive task: {args.task}")
        return 2

    if args.command == "run":
        result = run_task(
            task_name=args.task,
            runner_kind=args.runner,
            max_turns=int(args.max_turns),
            model_url=args.model_url,
            model=args.model,
            timeout_s=float(args.timeout_s),
            max_tokens=int(args.max_tokens),
            temperature=float(args.temperature),
            tick_rate_hz=float(args.tick_rate_hz),
            deadline_ms=float(args.deadline_ms),
            output_dir=args.output_dir,
            start_number=int(args.start_number),
            history_limit=int(args.history_limit),
            top_p=args.top_p,
            live_output=bool(args.live_output),
        )
        print(
            f"{result.runner_kind} completed: {result.turn_count} turns, "
            f"{result.valid_count} valid, {result.invalid_count} invalid, "
            f"{result.deadline_miss_count} deadline misses"
        )
        print(f"trace: {result.output_dir}")
        return 0

    try:
        human_input_config = _human_input_config_from_args(args)
        _validate_voice_game_range(
            task_name=args.task,
            start_number=int(args.start_number),
            max_turns=int(args.max_turns),
            config=human_input_config,
        )
    except (TypeError, ValueError) as exc:
        print(f"invalid human input configuration: {exc}")
        return 2
    try:
        speech_output_config = _speech_output_config_from_args(args)
    except (TypeError, ValueError) as exc:
        print(f"invalid speech output configuration: {exc}")
        return 2

    try:
        result = play_task(
            task_name=args.task,
            max_turns=int(args.max_turns),
            model_url=args.model_url,
            model=args.model,
            timeout_s=float(args.timeout_s),
            max_tokens=int(args.max_tokens),
            temperature=float(args.temperature),
            deadline_ms=float(args.deadline_ms),
            output_dir=args.output_dir,
            start_number=int(args.start_number),
            human_first=bool(args.human_first),
            show_expected=bool(args.show_expected),
            history_limit=int(args.history_limit),
            deadline_mode=args.deadline_mode,
            challenge_stage_turns=int(args.challenge_stage_turns),
            challenge_deadline_ms_list=args.challenge_deadline_ms_list,
            human_input_config=human_input_config,
            speech_output_config=speech_output_config,
        )
    except KeyboardInterrupt:
        print("interactive session interrupted")
        return 130
    except Exception as exc:
        if _is_speech_output_error(exc):
            print(f"speech output error: {exc}")
            return 1
        if _is_voice_infrastructure_error(exc):
            print(f"voice input error: {exc}")
            return 1
        raise
    print(f"interactive completed: {result.turn_count} turns")
    print(f"trace: {result.output_dir}")
    return 0


def run_task(
    *,
    task_name: str,
    runner_kind: str,
    max_turns: int,
    model_url: str,
    model: str,
    timeout_s: float,
    max_tokens: int,
    temperature: float,
    tick_rate_hz: float,
    deadline_ms: float,
    output_dir: str,
    start_number: int = 1,
    history_limit: int = 0,
    top_p: float | None = None,
    extra_payload: dict[str, object] | None = None,
    oracle_history: bool = False,
    live_output: bool = False,
) -> TaskRunResult:
    task = create_task(
        task_name,
        start_number=start_number,
        history_limit=history_limit,
        oracle_history=oracle_history,
    )
    run_dir = _new_run_dir(output_dir, task_name=task.name, runner_kind=runner_kind)
    client = _build_client(
        model_url=model_url,
        model=model,
        timeout_s=timeout_s,
        top_p=top_p,
        extra_payload=extra_payload,
    )
    runtime = TaskRuntime(
        config=TaskRuntimeConfig(
            runner_kind=runner_kind,  # type: ignore[arg-type]
            output_dir=str(run_dir),
            tick_rate_hz=tick_rate_hz,
            deadline_ms=deadline_ms,
            max_tokens=max_tokens,
            temperature=temperature,
            live_output=live_output,
        ),
        model_client=client,
    )
    try:
        return runtime.run(task, max_turns=max_turns)
    finally:
        _close_client(client)


def play_task(
    *,
    task_name: str,
    max_turns: int,
    model_url: str,
    model: str,
    timeout_s: float,
    max_tokens: int,
    temperature: float,
    deadline_ms: float,
    output_dir: str,
    start_number: int = 1,
    human_first: bool = True,
    show_expected: bool = False,
    history_limit: int = 8,
    deadline_mode: DeadlineMode = "menu",
    challenge_stage_turns: int = 20,
    challenge_deadline_ms_list: tuple[float, ...] = (10000.0, 5000.0, 3000.0, 2000.0, 1000.0),
    terminal: TerminalIO | None = None,
    human_input_config: HumanInputConfig | None = None,
    speech_output_config: SpeechOutputConfig | None = None,
) -> InteractiveTaskRunResult:
    task = create_task(task_name, start_number=start_number, history_limit=history_limit)
    run_dir = _new_run_dir(output_dir, task_name=task.name, runner_kind="interactive")
    active_terminal = terminal or StdTerminalIO()
    input_config = human_input_config or HumanInputConfig()
    output_config = speech_output_config or SpeechOutputConfig()
    _validate_voice_game_range(
        task_name=task_name,
        start_number=start_number,
        max_turns=max_turns,
        config=input_config,
    )
    runtime_config = _interactive_runtime_config(
        run_dir=run_dir,
        deadline_ms=deadline_ms,
        max_tokens=max_tokens,
        temperature=temperature,
        human_first=human_first,
        show_expected=show_expected,
        deadline_mode=deadline_mode,
        challenge_stage_turns=challenge_stage_turns,
        challenge_deadline_ms_list=challenge_deadline_ms_list,
        speech_guard_ms=output_config.guard_ms,
        speech_on_error=output_config.on_error,
        llm_deadline_basis=output_config.llm_deadline_basis,
    )

    with ExitStack() as resources:
        try:
            from streammuse.application.factories.human_input_factory import HumanInputFactory

            human_response_source = HumanInputFactory.create(
                input_config,
                terminal=active_terminal,
                artifact_root=run_dir / "artifacts" / "turn",
            )
        except BaseException as exc:
            _best_effort_startup_manifest(
                run_dir,
                task_name=task.name,
                max_turns=max_turns,
                config=input_config,
                speech_config=output_config,
                runtime_config=runtime_config,
                error=exc,
            )
            raise
        resources.enter_context(_preserving_close(human_response_source.close))
        try:
            from streammuse.application.factories.speech_output_factory import (
                SpeechOutputFactory,
            )

            speech_output_sink = SpeechOutputFactory.create(
                output_config,
                artifact_root=run_dir / "artifacts" / "turn",
            )
            resources.enter_context(
                _preserving_close(speech_output_sink.close)
            )
            client = _build_client(model_url=model_url, model=model, timeout_s=timeout_s)
            resources.enter_context(_preserving_close(lambda: _close_client(client)))
            runtime = InteractiveTaskRuntime(
                config=runtime_config,
                model_client=client,
                terminal=active_terminal,
                human_response_source=human_response_source,
                speech_output_sink=speech_output_sink,
            )
        except BaseException as exc:
            _best_effort_startup_manifest(
                run_dir,
                task_name=task.name,
                max_turns=max_turns,
                config=input_config,
                speech_config=output_config,
                runtime_config=runtime_config,
                error=exc,
                human_response_source=human_response_source,
                speech_output_sink=speech_output_sink
                if "speech_output_sink" in locals()
                else None,
            )
            raise
        return runtime.play(task, max_turns=max_turns)


def create_task(
    task_name: str,
    *,
    start_number: int,
    history_limit: int = 8,
    oracle_history: bool = False,
) -> RealtimeTask | InteractiveTask:
    if task_name == "animal_naming":
        return AnimalNamingTask()
    if task_name == "zip_zap_zop":
        return ZipZapZopTask(
            start_number=start_number,
            history_limit=history_limit,
            oracle_history=oracle_history,
        )
    raise ValueError(f"unsupported task: {task_name}")


def _parse_deadline_ms_list(raw: str) -> tuple[float, ...]:
    values: list[float] = []
    for part in str(raw or "").split(","):
        stripped = part.strip()
        if not stripped:
            continue
        value = float(stripped)
        if not math.isfinite(value) or value <= 0:
            raise argparse.ArgumentTypeError("challenge deadlines must be finite and positive")
        values.append(value)
    if not values:
        raise argparse.ArgumentTypeError("challenge deadline list cannot be empty")
    return tuple(values)


def _parse_audio_device(raw: str) -> str | int:
    value = str(raw).strip()
    if not value:
        raise argparse.ArgumentTypeError("microphone device must not be empty")
    if value.lstrip("+-").isdigit():
        return int(value)
    return value


_parse_microphone_device = _parse_audio_device


def _positive_finite_float(raw: str) -> float:
    value = float(raw)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("value must be finite and > 0")
    return value


def _human_input_config_from_args(args: argparse.Namespace) -> HumanInputConfig:
    voice_values = (
        args.voice_model,
        args.voice_device,
        args.voice_compute_type,
        args.microphone_device,
        args.voice_model_cache,
        args.voice_model_revision,
    )
    voice_flags = (bool(args.voice_local_files_only), bool(args.voice_save_audio))
    if args.human_input == "terminal":
        if any(value is not None for value in voice_values) or any(voice_flags):
            raise ValueError("voice options require --human-input voice")
        return HumanInputConfig()

    return HumanInputConfig(
        mode="voice",
        voice=VoiceInputConfig(
            model="tiny.en" if args.voice_model is None else args.voice_model,
            device="cpu" if args.voice_device is None else args.voice_device,
            compute_type="int8" if args.voice_compute_type is None else args.voice_compute_type,
            microphone_device=args.microphone_device,
            model_cache=args.voice_model_cache,
            model_revision=args.voice_model_revision,
            local_files_only=bool(args.voice_local_files_only),
            save_audio=bool(args.voice_save_audio),
        ),
    )


def _speech_output_config_from_args(
    args: argparse.Namespace,
) -> SpeechOutputConfig:
    speech_option_names = (
        "speech_backend",
        "speech_voice",
        "speech_rate",
        "speaker_device",
        "speech_model",
        "speech_model_cache",
        "speech_model_revision",
        "speech_local_files_only",
        "speech_synthesis_timeout_s",
        "speech_prewarm",
        "speech_cache_miss",
        "speech_cache_max_entries",
        "speech_cache_max_bytes",
        "speech_guard_ms",
        "speech_on_error",
        "speech_save_audio",
    )
    mode = "off" if args.speech_output is None else args.speech_output
    if mode == "off":
        explicit = [
            f"--{name.replace('_', '-')}"
            for name in speech_option_names
            if getattr(args, name) is not None
        ]
        if explicit:
            raise ValueError(
                "speech options require --speech-output audio: "
                + ", ".join(explicit)
            )
        return SpeechOutputConfig(
            llm_deadline_basis=(
                "text"
                if args.llm_deadline_basis is None
                else args.llm_deadline_basis
            )
        )

    return SpeechOutputConfig(
        mode="audio",
        backend=(
            "system" if args.speech_backend is None else args.speech_backend
        ),
        voice=args.speech_voice,
        rate=1.0 if args.speech_rate is None else args.speech_rate,
        speaker_device=args.speaker_device,
        model=args.speech_model,
        model_cache=args.speech_model_cache,
        model_revision=args.speech_model_revision,
        local_files_only=bool(args.speech_local_files_only),
        synthesis_timeout_s=(
            10.0
            if args.speech_synthesis_timeout_s is None
            else args.speech_synthesis_timeout_s
        ),
        prewarm=True if args.speech_prewarm is None else args.speech_prewarm,
        cache_miss=(
            "synthesize"
            if args.speech_cache_miss is None
            else args.speech_cache_miss
        ),
        cache_max_entries=(
            512
            if args.speech_cache_max_entries is None
            else args.speech_cache_max_entries
        ),
        cache_max_bytes=(
            67_108_864
            if args.speech_cache_max_bytes is None
            else args.speech_cache_max_bytes
        ),
        guard_ms=(
            200.0 if args.speech_guard_ms is None else args.speech_guard_ms
        ),
        on_error=(
            "fail" if args.speech_on_error is None else args.speech_on_error
        ),
        save_audio=bool(args.speech_save_audio),
        llm_deadline_basis=(
            "text"
            if args.llm_deadline_basis is None
            else args.llm_deadline_basis
        ),
    )


def _validate_voice_game_range(
    *,
    task_name: str,
    start_number: int,
    max_turns: int,
    config: HumanInputConfig,
) -> None:
    if config.mode != "voice" or task_name != "zip_zap_zop":
        return
    turn_count = max(1, int(max_turns))
    end_number = int(start_number) + turn_count - 1
    minimum = min(int(start_number), end_number)
    maximum = max(int(start_number), end_number)
    if (
        minimum < ZipZapZopTask.min_spoken_integer
        or maximum > ZipZapZopTask.max_spoken_integer
    ):
        raise ValueError(
            "voice Zip-Zap-Zop turns must stay within the spoken integer range "
            f"[{ZipZapZopTask.min_spoken_integer}, {ZipZapZopTask.max_spoken_integer}]"
        )


def _interactive_runtime_config(
    *,
    run_dir: Path,
    deadline_ms: float,
    max_tokens: int,
    temperature: float,
    human_first: bool,
    show_expected: bool,
    deadline_mode: DeadlineMode,
    challenge_stage_turns: int,
    challenge_deadline_ms_list: tuple[float, ...],
    speech_guard_ms: float = 200.0,
    speech_on_error: str = "fail",
    llm_deadline_basis: str = "text",
) -> InteractiveTaskRuntimeConfig:
    return InteractiveTaskRuntimeConfig(
        output_dir=str(run_dir),
        deadline_ms=deadline_ms,
        max_tokens=max_tokens,
        temperature=temperature,
        human_first=human_first,
        show_expected=show_expected,
        deadline_mode=deadline_mode,
        challenge_stage_turns=challenge_stage_turns,
        challenge_deadline_ms_list=challenge_deadline_ms_list,
        speech_guard_ms=speech_guard_ms,
        speech_on_error=speech_on_error,  # type: ignore[arg-type]
        llm_deadline_basis=llm_deadline_basis,  # type: ignore[arg-type]
    )


@contextmanager
def _preserving_close(
    close: Callable[[], None],
) -> Iterator[None]:
    """Close a resource without replacing an exception already in flight."""

    try:
        yield
    except BaseException:
        try:
            close()
        except BaseException:
            pass
        raise
    else:
        close()


def _best_effort_startup_manifest(
    run_dir: Path,
    *,
    task_name: str,
    max_turns: int,
    config: HumanInputConfig,
    speech_config: SpeechOutputConfig,
    runtime_config: InteractiveTaskRuntimeConfig,
    error: BaseException,
    human_response_source: object | None = None,
    speech_output_sink: object | None = None,
) -> None:
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
        human_input = _startup_human_input(config, human_response_source)
        schedule = [
            float(value)
            for value in runtime_config.challenge_deadline_ms_list
            if float(value) > 0
        ]
        current_deadline_ms = (
            schedule[0]
            if runtime_config.deadline_mode == "challenge" and schedule
            else float(runtime_config.deadline_ms)
        )
        interrupted = isinstance(error, KeyboardInterrupt)
        status = "user_interrupt" if interrupted else "startup_error"
        manifest: dict[str, object] = {
            "schema_version": 2,
            "run_id": f"interactive-{uuid.uuid4().hex[:8]}",
            "mode": "interactive",
            "task_name": task_name,
            "status": status,
            "max_turns": int(max_turns),
            "deadline_ms": float(runtime_config.deadline_ms),
            "deadline_mode": runtime_config.deadline_mode,
            "configured_deadline_mode": runtime_config.deadline_mode,
            "current_deadline_ms": current_deadline_ms,
            "challenge_stage_turns": int(runtime_config.challenge_stage_turns),
            "challenge_deadline_ms_list": schedule,
            "max_tokens": int(runtime_config.max_tokens),
            "temperature": float(runtime_config.temperature),
            "human_first": bool(runtime_config.human_first),
            "show_expected": bool(runtime_config.show_expected),
            "response_trace_path": "response_trace.jsonl",
            "artifact_root": "artifacts",
            "human_input": human_input,
        }
        speech_output = _startup_speech_output(
            speech_config,
            speech_output_sink,
        )
        if speech_output is not None:
            manifest["speech_output"] = speech_output
        if interrupted:
            manifest["stop_reason"] = "user_interrupt"
        else:
            manifest["startup_error"] = {
                "type": type(error).__name__,
                "message": str(error),
            }
        (run_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
    except BaseException:
        return


def _startup_human_input(config: HumanInputConfig, source: object | None) -> dict[str, object]:
    if source is not None:
        try:
            provenance = getattr(source, "provenance")
            if isinstance(provenance, dict):
                return {**provenance, "mode": config.mode}
        except BaseException:
            pass
    result: dict[str, object] = {"mode": config.mode}
    if config.voice is not None:
        result["configuration"] = asdict(config.voice)
    return result


def _startup_speech_output(
    config: SpeechOutputConfig,
    sink: object | None,
) -> dict[str, object] | None:
    if config.mode == "off":
        return None
    if sink is not None:
        try:
            provenance = getattr(sink, "provenance")
            if isinstance(provenance, dict):
                return {**provenance, "mode": "audio"}
        except BaseException:
            pass
    return {
        "mode": "audio",
        "configuration": asdict(config),
    }


def _list_voice_devices() -> int:
    try:
        from streammuse.infrastructure.voice import enumerate_input_devices

        devices = enumerate_input_devices()
    except Exception as exc:
        if _is_voice_infrastructure_error(exc):
            print(f"voice input error: {exc}")
            return 1
        raise

    if not devices:
        print("No microphone input devices found.")
        return 0
    print("Microphone input devices:")
    for device in devices:
        hostapi = "unknown" if device.hostapi is None else str(device.hostapi)
        print(
            f"[{device.index}] {device.name} "
            f"(channels={device.max_input_channels}, "
            f"default_rate={device.default_sample_rate_hz:g} Hz, hostapi={hostapi})"
        )
    return 0


def _list_speaker_devices() -> int:
    try:
        from streammuse.infrastructure.voice.speaker import (
            enumerate_output_devices,
        )

        devices = enumerate_output_devices()
    except Exception as exc:
        if _is_speech_output_error(exc):
            print(f"speech output error: {exc}")
            return 1
        raise
    if not devices:
        print("No speaker output devices found.")
        return 0
    print("Speaker output devices:")
    for device in devices:
        hostapi = "unknown" if device.hostapi is None else str(device.hostapi)
        print(
            f"[{device.index}] {device.name} "
            f"(channels={device.max_output_channels}, "
            f"default_rate={device.default_sample_rate_hz:g} Hz, "
            f"hostapi={hostapi})"
        )
    return 0


def _is_voice_infrastructure_error(exc: BaseException) -> bool:
    from streammuse.infrastructure.voice import VoiceInfrastructureError

    return isinstance(exc, VoiceInfrastructureError)


def _is_speech_output_error(exc: BaseException) -> bool:
    from streammuse.infrastructure.voice import SpeechOutputError

    return isinstance(exc, SpeechOutputError)


def _build_client(
    *,
    model_url: str,
    model: str,
    timeout_s: float,
    top_p: float | None = None,
    extra_payload: dict[str, object] | None = None,
) -> LocalChatModelClient:
    return LocalChatModelClient(
        LocalChatModelClientConfig(
            base_url=model_url,
            model=model,
            timeout_s=timeout_s,
            max_retries=0,
            top_p=top_p,
            extra_payload=extra_payload,
        )
    )


def _close_client(client: object) -> None:
    close = getattr(client, "close", None)
    if callable(close):
        close()


def _new_run_dir(output_dir: str, *, task_name: str, runner_kind: str) -> Path:
    import time
    import uuid

    root = Path(output_dir).expanduser().resolve()
    return root / f"{task_name}_{runner_kind}_{time.strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:8]}"


if __name__ == "__main__":
    raise SystemExit(main())
