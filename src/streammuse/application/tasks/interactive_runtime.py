"""Interactive terminal runtime for human/LLM tasks."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

from streammuse.domain.tasks import (
    ChatModelResponse,
    InteractiveActor,
    InteractiveTask,
    InteractiveTurnRecord,
    LocalChatModel,
    TaskRefereeResult,
    TaskState,
)


class TerminalIO(Protocol):
    def write(self, text: str) -> None: ...

    def prompt(self, text: str) -> str: ...


class StdTerminalIO:
    def write(self, text: str) -> None:
        print(text, flush=True)

    def prompt(self, text: str) -> str:
        return input(text)


@dataclass(frozen=True)
class InteractiveTaskRuntimeConfig:
    output_dir: str
    deadline_ms: float = 3000.0
    max_tokens: int = 8
    temperature: float = 0.0
    human_first: bool = True
    show_expected: bool = False


@dataclass(frozen=True)
class InteractiveTaskRunResult:
    output_dir: str
    task_name: str
    turn_count: int
    human_turn_count: int
    llm_turn_count: int
    valid_count: int
    invalid_count: int
    deadline_miss_count: int


@dataclass
class _InteractiveStats:
    valid_count: int = 0
    invalid_count: int = 0
    deadline_miss_count: int = 0
    llm_latency_ms_sum: float = 0.0
    llm_turn_count: int = 0
    human_turn_count: int = 0
    quit_requested: bool = False

    @property
    def turn_count(self) -> int:
        return self.human_turn_count + self.llm_turn_count

    @property
    def avg_llm_latency_ms(self) -> float | None:
        if self.llm_turn_count <= 0:
            return None
        return self.llm_latency_ms_sum / float(self.llm_turn_count)


class InteractiveTaskRuntime:
    def __init__(
        self,
        *,
        config: InteractiveTaskRuntimeConfig,
        model_client: LocalChatModel,
        terminal: TerminalIO | None = None,
        now: Callable[[], float] | None = None,
    ) -> None:
        self.config = config
        self.model_client = model_client
        self.terminal = terminal or StdTerminalIO()
        self._now = now or time.perf_counter

    def play(self, task: InteractiveTask, *, max_turns: int) -> InteractiveTaskRunResult:
        run_dir = Path(self.config.output_dir).expanduser().resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "artifacts" / "turn").mkdir(parents=True, exist_ok=True)
        run_id = f"interactive-{uuid.uuid4().hex[:8]}"
        self._write_initial_manifest(run_dir, run_id=run_id, task=task, max_turns=max_turns)

        state = task.initial_state()
        transcript: list[InteractiveTurnRecord] = []
        stats = _InteractiveStats()

        self._write_banner(task.name)
        try:
            while stats.turn_count < int(max_turns) and not stats.quit_requested:
                is_human_turn = (stats.turn_count % 2 == 0) if self.config.human_first else (stats.turn_count % 2 == 1)
                actor: InteractiveActor = "human" if is_human_turn else "llm"
                if actor == "human":
                    next_state = self._run_human_turn(
                        task=task,
                        state=state,
                        transcript=transcript,
                        stats=stats,
                        run_dir=run_dir,
                    )
                    if next_state is not None:
                        state = next_state
                else:
                    state = self._run_llm_turn(
                        task=task,
                        state=state,
                        transcript=transcript,
                        stats=stats,
                        run_dir=run_dir,
                    )

            result = self._result(run_dir, task.name, stats)
            self._write_run_summary(run_dir, task.name, result, status="stopped" if stats.quit_requested else "completed")
            self.terminal.write(self._summary_text(result, stats))
            return result
        except Exception:
            result = self._result(run_dir, task.name, stats)
            self._write_run_summary(run_dir, task.name, result, status="error")
            raise

    def _run_human_turn(
        self,
        *,
        task: InteractiveTask,
        state: TaskState,
        transcript: list[InteractiveTurnRecord],
        stats: _InteractiveStats,
        run_dir: Path,
    ) -> TaskState | None:
        prompt_text = task.build_human_prompt(state, transcript)
        number = self._number_from_state(state)

        while True:
            start_s = self._now()
            response_text = self.terminal.prompt(f"[{number if number is not None else state.turn_index + 1}] You > {prompt_text} ")
            elapsed_ms = max(0.0, (self._now() - start_s) * 1000.0)
            stripped = str(response_text or "").strip()
            if stripped.startswith(":"):
                self._handle_command(stripped, task=task, state=state, transcript=transcript, stats=stats)
                if stats.quit_requested:
                    return None
                continue
            break

        return self._finish_turn(
            task=task,
            state=state,
            transcript=transcript,
            stats=stats,
            run_dir=run_dir,
            actor="human",
            prompt_payload=prompt_text,
            response_text=response_text,
            elapsed_ms=elapsed_ms,
            prompt_tokens=None,
            completion_tokens=None,
        )

    def _run_llm_turn(
        self,
        *,
        task: InteractiveTask,
        state: TaskState,
        transcript: list[InteractiveTurnRecord],
        stats: _InteractiveStats,
        run_dir: Path,
    ) -> TaskState:
        number = self._number_from_state(state)
        self.terminal.write(f"[{number if number is not None else state.turn_index + 1}] LLM thinking...")
        messages = task.build_llm_messages(state, transcript)
        start_s = self._now()
        model_response = self.model_client.generate(
            messages,
            max_tokens=int(self.config.max_tokens),
            temperature=float(self.config.temperature),
        )
        elapsed_ms = max(float(model_response.latency_ms), (self._now() - start_s) * 1000.0)
        response_text = model_response.text
        self.terminal.write(f"    LLM > {response_text}")
        return self._finish_turn(
            task=task,
            state=state,
            transcript=transcript,
            stats=stats,
            run_dir=run_dir,
            actor="llm",
            prompt_payload=messages,
            response_text=response_text,
            elapsed_ms=elapsed_ms,
            prompt_tokens=model_response.prompt_tokens,
            completion_tokens=model_response.completion_tokens,
            raw_model_response=model_response,
        )

    def _finish_turn(
        self,
        *,
        task: InteractiveTask,
        state: TaskState,
        transcript: list[InteractiveTurnRecord],
        stats: _InteractiveStats,
        run_dir: Path,
        actor: InteractiveActor,
        prompt_payload: str | list[dict[str, str]],
        response_text: str,
        elapsed_ms: float,
        prompt_tokens: int | None,
        completion_tokens: int | None,
        raw_model_response: ChatModelResponse | None = None,
    ) -> TaskState:
        referee = task.validate_response(
            state,
            response_text,
            actor=actor,
            transcript=transcript,
        )
        deadline_missed = elapsed_ms > float(self.config.deadline_ms)
        number = self._number_from_state(state)
        record = InteractiveTurnRecord(
            turn_id=len(transcript),
            actor=actor,
            number=number,
            prompt=prompt_payload,
            response=str(response_text or "").strip(),
            expected=referee.expected_output,
            is_valid=bool(referee.is_valid),
            latency_ms=float(elapsed_ms),
            deadline_missed=bool(deadline_missed),
            metadata={
                "failure_reason": referee.failure_reason,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            },
        )
        next_state = task.advance_state(
            state,
            referee,
            response_text,
            actor=actor,
            transcript=transcript,
        )
        transcript.append(record)
        self._update_stats(stats, record)
        self._append_response_trace(run_dir, record)
        self._write_turn_artifact(
            run_dir=run_dir,
            record=record,
            state_before=state,
            state_after=next_state,
            referee=referee,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            raw_model_response=raw_model_response,
        )
        self.terminal.write(f"    {self._result_text(record)}")
        return next_state

    def _handle_command(
        self,
        command: str,
        *,
        task: InteractiveTask,
        state: TaskState,
        transcript: list[InteractiveTurnRecord],
        stats: _InteractiveStats,
    ) -> None:
        normalized = command.strip().lower()
        if normalized in {":quit", ":q", ":exit"}:
            stats.quit_requested = True
            self.terminal.write("Stopping interactive session.")
            return
        if normalized == ":help":
            self.terminal.write(
                "Commands: :help, :hint, :expected, :summary, :quit\n"
                "Answer the shown prompt with only the next value."
            )
            return
        if normalized == ":hint":
            hint = task.build_hint(state, transcript)
            self.terminal.write(hint or "No hint available.")
            return
        if normalized == ":expected":
            if not self.config.show_expected:
                self.terminal.write("Expected answers are hidden; rerun with --show-expected.")
                return
            expected = task.expected_for_state(state, transcript)
            self.terminal.write(expected or "N/A")
            return
        if normalized == ":summary":
            self.terminal.write(self._summary_text(self._result(Path(self.config.output_dir), task.name, stats), stats))
            return
        self.terminal.write(f"Unknown command: {command}. Type :help for commands.")

    def _write_banner(self, task_name: str) -> None:
        self.terminal.write(f"StreamMUSE {task_name}")
        self.terminal.write("Commands: :help, :hint, :expected, :summary, :quit")

    def _update_stats(self, stats: _InteractiveStats, record: InteractiveTurnRecord) -> None:
        if record.is_valid:
            stats.valid_count += 1
        else:
            stats.invalid_count += 1
        if record.deadline_missed:
            stats.deadline_miss_count += 1
        if record.actor == "human":
            stats.human_turn_count += 1
        else:
            stats.llm_turn_count += 1
            stats.llm_latency_ms_sum += float(record.latency_ms or 0.0)

    def _result_text(self, record: InteractiveTurnRecord) -> str:
        latency = f"  {float(record.latency_ms or 0.0):.1f} ms"
        if record.is_valid:
            if self.config.show_expected and record.expected is not None:
                return f"OK expected={record.expected}{latency}"
            return f"OK{latency}"
        return f"MISS expected={record.expected} actual={record.response}{latency}"

    def _summary_text(self, result: InteractiveTaskRunResult, stats: _InteractiveStats) -> str:
        avg = stats.avg_llm_latency_ms
        avg_text = "n/a" if avg is None else f"{avg:.1f} ms"
        return (
            f"Summary: {result.turn_count} turns, {result.valid_count} valid, "
            f"{result.invalid_count} invalid, {result.deadline_miss_count} deadline misses, "
            f"avg LLM latency {avg_text}"
        )

    def _number_from_state(self, state: TaskState) -> int | None:
        if "current_number" not in state.data:
            return None
        return int(state.data.get("current_number", 0)) + 1

    def _append_response_trace(self, run_dir: Path, record: InteractiveTurnRecord) -> None:
        row = asdict(record)
        with (run_dir / "response_trace.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    def _write_turn_artifact(
        self,
        *,
        run_dir: Path,
        record: InteractiveTurnRecord,
        state_before: TaskState,
        state_after: TaskState,
        referee: TaskRefereeResult,
        prompt_tokens: int | None,
        completion_tokens: int | None,
        raw_model_response: ChatModelResponse | None,
    ) -> None:
        payload: dict[str, Any] = {
            "turn": asdict(record),
            "state_before": asdict(state_before),
            "state_after": asdict(state_after),
            "referee": asdict(referee),
            "tokens": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            },
        }
        if raw_model_response is not None:
            payload["model_response_raw"] = raw_model_response.raw
        path = run_dir / "artifacts" / "turn" / f"{record.turn_id + 1:04d}_turn_{record.turn_id:04d}.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")

    def _write_initial_manifest(self, run_dir: Path, *, run_id: str, task: InteractiveTask, max_turns: int) -> None:
        manifest = {
            "schema_version": 1,
            "run_id": run_id,
            "mode": "interactive",
            "task_name": task.name,
            "status": "running",
            "max_turns": int(max_turns),
            "deadline_ms": float(self.config.deadline_ms),
            "max_tokens": int(self.config.max_tokens),
            "temperature": float(self.config.temperature),
            "human_first": bool(self.config.human_first),
            "show_expected": bool(self.config.show_expected),
            "response_trace_path": "response_trace.jsonl",
            "artifact_root": "artifacts",
        }
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    def _write_run_summary(
        self,
        run_dir: Path,
        task_name: str,
        result: InteractiveTaskRunResult,
        *,
        status: str,
    ) -> None:
        summary = asdict(result)
        manifest_path = run_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.update(
            {
                "status": status,
                "task_name": task_name,
                "turn_count": result.turn_count,
                "human_turn_count": result.human_turn_count,
                "llm_turn_count": result.llm_turn_count,
                "valid_count": result.valid_count,
                "invalid_count": result.invalid_count,
                "deadline_miss_count": result.deadline_miss_count,
            }
        )
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    def _result(self, run_dir: Path, task_name: str, stats: _InteractiveStats) -> InteractiveTaskRunResult:
        return InteractiveTaskRunResult(
            output_dir=str(run_dir),
            task_name=task_name,
            turn_count=stats.turn_count,
            human_turn_count=stats.human_turn_count,
            llm_turn_count=stats.llm_turn_count,
            valid_count=stats.valid_count,
            invalid_count=stats.invalid_count,
            deadline_miss_count=stats.deadline_miss_count,
        )
