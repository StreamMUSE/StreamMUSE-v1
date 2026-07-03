"""CLI entry point for StreamMUSE realtime task runs."""

from __future__ import annotations

import argparse
from pathlib import Path

from streammuse.application.tasks import (
    InteractiveTaskRunResult,
    InteractiveTaskRuntime,
    InteractiveTaskRuntimeConfig,
    StdTerminalIO,
    TaskRunResult,
    TaskRuntime,
    TaskRuntimeConfig,
    TerminalIO,
)
from streammuse.domain.tasks import InteractiveTask, RealtimeTask, ZipZapZopTask
from streammuse.infrastructure.inference.local_chat_client import (
    LocalChatModelClient,
    LocalChatModelClientConfig,
)


TASKS = {"zip_zap_zop": ZipZapZopTask}
RUNNERS = {"offline_benchmark", "realtime_loop"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run StreamMUSE generic realtime tasks")
    subcommands = parser.add_subparsers(dest="command")

    run = subcommands.add_parser("run", help="Run a task against a local model server")
    _add_common_task_args(run, max_tokens_default=32)
    run.add_argument("--runner", choices=sorted(RUNNERS), default="offline_benchmark")
    run.add_argument("--tick-rate-hz", type=float, default=1.0)

    play = subcommands.add_parser("play", help="Play an interactive task with a local model server")
    _add_common_task_args(play, max_tokens_default=8)
    first_actor = play.add_mutually_exclusive_group()
    first_actor.add_argument("--human-first", dest="human_first", action="store_true", default=True)
    first_actor.add_argument("--llm-first", dest="human_first", action="store_false")
    play.add_argument("--show-expected", action="store_true")
    play.add_argument("--history-limit", type=int, default=8)
    return parser


def _add_common_task_args(parser: argparse.ArgumentParser, *, max_tokens_default: int) -> None:
    parser.add_argument("--task", required=True)
    parser.add_argument("--max-turns", type=int, default=20)
    parser.add_argument("--model-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="local-model")
    parser.add_argument("--timeout-s", type=float, default=30.0)
    parser.add_argument("--max-tokens", type=int, default=max_tokens_default)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--deadline-ms", type=float, default=1000.0)
    parser.add_argument("--output-dir", default="task_runs")
    parser.add_argument("--start-number", type=int, default=1)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command not in {"run", "play"}:
        parser.print_help()
        return 2
    if args.task not in TASKS:
        print(f"unsupported task: {args.task}")
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
        )
        print(
            f"{result.runner_kind} completed: {result.turn_count} turns, "
            f"{result.valid_count} valid, {result.invalid_count} invalid, "
            f"{result.deadline_miss_count} deadline misses"
        )
        print(f"trace: {result.output_dir}")
        return 0

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
    )
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
) -> TaskRunResult:
    task = create_task(task_name, start_number=start_number)
    run_dir = _new_run_dir(output_dir, task_name=task.name, runner_kind=runner_kind)
    client = _build_client(model_url=model_url, model=model, timeout_s=timeout_s)
    runtime = TaskRuntime(
        config=TaskRuntimeConfig(
            runner_kind=runner_kind,  # type: ignore[arg-type]
            output_dir=str(run_dir),
            tick_rate_hz=tick_rate_hz,
            deadline_ms=deadline_ms,
            max_tokens=max_tokens,
            temperature=temperature,
        ),
        model_client=client,
    )
    return runtime.run(task, max_turns=max_turns)


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
    terminal: TerminalIO | None = None,
) -> InteractiveTaskRunResult:
    task = create_task(task_name, start_number=start_number, history_limit=history_limit)
    run_dir = _new_run_dir(output_dir, task_name=task.name, runner_kind="interactive")
    client = _build_client(model_url=model_url, model=model, timeout_s=timeout_s)
    runtime = InteractiveTaskRuntime(
        config=InteractiveTaskRuntimeConfig(
            output_dir=str(run_dir),
            deadline_ms=deadline_ms,
            max_tokens=max_tokens,
            temperature=temperature,
            human_first=human_first,
            show_expected=show_expected,
        ),
        model_client=client,
        terminal=terminal or StdTerminalIO(),
    )
    return runtime.play(task, max_turns=max_turns)


def create_task(task_name: str, *, start_number: int, history_limit: int = 8) -> RealtimeTask | InteractiveTask:
    if task_name == "zip_zap_zop":
        return ZipZapZopTask(start_number=start_number, history_limit=history_limit)
    raise ValueError(f"unsupported task: {task_name}")


def _build_client(*, model_url: str, model: str, timeout_s: float) -> LocalChatModelClient:
    return LocalChatModelClient(
        LocalChatModelClientConfig(
            base_url=model_url,
            model=model,
            timeout_s=timeout_s,
            max_retries=0,
        )
    )


def _new_run_dir(output_dir: str, *, task_name: str, runner_kind: str) -> Path:
    import time
    import uuid

    root = Path(output_dir).expanduser().resolve()
    return root / f"{task_name}_{runner_kind}_{time.strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:8]}"


if __name__ == "__main__":
    raise SystemExit(main())
