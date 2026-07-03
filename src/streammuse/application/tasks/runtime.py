"""Runtime runners for generic realtime tasks."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Literal

from streammuse.domain.debug.trace import DebugTraceEvent
from streammuse.domain.tasks import LocalChatModel, RealtimeTask, TaskRefereeResult, TaskState, TaskTurn
from streammuse.infrastructure.debug.trace_recorder import JsonlDebugTraceRecorder


TaskRunnerKind = Literal["offline_benchmark", "realtime_loop"]


@dataclass(frozen=True)
class TaskRuntimeConfig:
    runner_kind: TaskRunnerKind
    output_dir: str
    tick_rate_hz: float = 1.0
    deadline_ms: float = 1000.0
    max_tokens: int = 32
    temperature: float = 0.0


@dataclass(frozen=True)
class TaskRunResult:
    output_dir: str
    runner_kind: str
    task_name: str
    turn_count: int
    valid_count: int
    invalid_count: int
    deadline_miss_count: int


class TaskRuntime:
    def __init__(
        self,
        *,
        config: TaskRuntimeConfig,
        model_client: LocalChatModel,
        now: Callable[[], float] | None = None,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        self.config = config
        self.model_client = model_client
        self._now = now or time.perf_counter
        self._sleep = sleep or time.sleep

    def run(self, task: RealtimeTask, *, max_turns: int) -> TaskRunResult:
        run_dir = Path(self.config.output_dir).expanduser().resolve()
        recorder = JsonlDebugTraceRecorder(
            root_dir=run_dir,
            run_id=f"{self.config.runner_kind}-{uuid.uuid4().hex[:8]}",
            runner_kind=self.config.runner_kind,
            scenario=f"task:{task.name}",
        )
        valid_count = 0
        invalid_count = 0
        deadline_miss_count = 0
        state = task.initial_state()
        next_deadline_s = self._now() + self._interval_s()

        try:
            for _ in range(int(max_turns)):
                if self.config.runner_kind == "realtime_loop":
                    now_s = self._now()
                    if now_s < next_deadline_s - self._interval_s():
                        self._sleep((next_deadline_s - self._interval_s()) - now_s)

                turn = task.build_turn(state)
                turn_start_s = self._now()
                model_response = self.model_client.generate(
                    turn.messages,
                    max_tokens=int(self.config.max_tokens),
                    temperature=float(self.config.temperature),
                )
                elapsed_ms = max(float(model_response.latency_ms), (self._now() - turn_start_s) * 1000.0)
                referee = task.validate_response(state, model_response.text)
                next_state = task.advance_state(state, referee, model_response.text)
                deadline_missed = self.config.runner_kind == "realtime_loop" and elapsed_ms > self.config.deadline_ms

                if referee.is_valid:
                    valid_count += 1
                else:
                    invalid_count += 1
                if deadline_missed:
                    deadline_miss_count += 1

                self._record_turn(
                    recorder=recorder,
                    turn=turn,
                    previous_state=state,
                    next_state=next_state,
                    response_text=model_response.text,
                    referee=referee,
                    elapsed_ms=elapsed_ms,
                    deadline_missed=deadline_missed,
                    prompt_tokens=model_response.prompt_tokens,
                    completion_tokens=model_response.completion_tokens,
                )
                self._append_response_trace(
                    run_dir=run_dir,
                    turn=turn,
                    response_text=model_response.text,
                )
                state = next_state
                next_deadline_s += self._interval_s()

            result = TaskRunResult(
                output_dir=str(run_dir),
                runner_kind=self.config.runner_kind,
                task_name=task.name,
                turn_count=int(max_turns),
                valid_count=valid_count,
                invalid_count=invalid_count,
                deadline_miss_count=deadline_miss_count,
            )
            self._write_run_summary(run_dir, task.name, result)
            return result
        finally:
            recorder.close()

    def _interval_s(self) -> float:
        return 1.0 / max(float(self.config.tick_rate_hz), 0.0001)

    def _append_response_trace(self, *, run_dir: Path, turn: TaskTurn, response_text: str) -> None:
        row = {
            "turn_id": int(turn.turn_id),
            "prompt": turn.messages,
            "response": response_text,
        }
        with (run_dir / "response_trace.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _record_turn(
        self,
        *,
        recorder: JsonlDebugTraceRecorder,
        turn: TaskTurn,
        previous_state: TaskState,
        next_state: TaskState,
        response_text: str,
        referee: TaskRefereeResult,
        elapsed_ms: float,
        deadline_missed: bool,
        prompt_tokens: int | None,
        completion_tokens: int | None,
    ) -> None:
        payload = {
            "turn": {
                "task_name": turn.task_name,
                "turn_id": turn.turn_id,
                "messages": turn.messages,
                "expected_output": turn.expected_output,
                "metadata": turn.metadata,
            },
            "state_before": asdict(previous_state),
            "model_response": response_text,
            "referee": asdict(referee),
            "state_after": asdict(next_state),
            "timing": {
                "latency_ms": elapsed_ms,
                "deadline_ms": float(self.config.deadline_ms),
                "deadline_missed": bool(deadline_missed),
            },
            "tokens": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            },
        }
        ref = recorder.artifact("task_turn", payload, name_hint=f"turn_{turn.turn_id:04d}")
        recorder.record(
            DebugTraceEvent(
                run_id=recorder.run_id,
                runner_kind=recorder.runner_kind,
                scenario=recorder.scenario,
                stage="task_turn",
                logical_tick=turn.turn_id,
                output_refs=[ref],
                output_hash=ref.hash,
                summary={
                    "runner_kind": self.config.runner_kind,
                    "task_name": turn.task_name,
                    "turn_id": turn.turn_id,
                    "is_valid": referee.is_valid,
                    "expected_output": referee.expected_output,
                    "failure_reason": referee.failure_reason,
                    "latency_ms": elapsed_ms,
                    "deadline_missed": deadline_missed,
                },
                metrics={
                    "latency_ms": elapsed_ms,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                },
                params={
                    "tick_rate_hz": self.config.tick_rate_hz,
                    "deadline_ms": self.config.deadline_ms,
                    "max_tokens": self.config.max_tokens,
                    "temperature": self.config.temperature,
                },
            )
        )

    def _write_run_summary(self, run_dir: Path, task_name: str, result: TaskRunResult) -> None:
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        manifest.update(
            {
                "task_name": task_name,
                "runner_kind": self.config.runner_kind,
                "turn_count": result.turn_count,
                "valid_count": result.valid_count,
                "invalid_count": result.invalid_count,
                "deadline_miss_count": result.deadline_miss_count,
                "response_trace_path": "response_trace.jsonl",
            }
        )
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        (run_dir / "run_summary.json").write_text(
            json.dumps(asdict(result), indent=2, sort_keys=True),
            encoding="utf-8",
        )
