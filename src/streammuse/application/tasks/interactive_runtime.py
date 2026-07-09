"""Interactive terminal runtime for human/LLM tasks."""

from __future__ import annotations

import json
import select
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Protocol

import requests

from streammuse.domain.tasks import (
    ChatModelResponse,
    DeadlineMode,
    InteractiveActor,
    InteractiveTask,
    InteractiveTurnRecord,
    LocalChatModel,
    TaskRefereeResult,
    TaskState,
)


ResolvedDeadlineMode = Literal["soft", "hard", "challenge"]


@dataclass(frozen=True)
class TimedPromptResult:
    text: str
    timed_out: bool = False


class TerminalIO(Protocol):
    def write(self, text: str) -> None: ...

    def prompt(self, text: str) -> str: ...

    def prompt_with_timeout(self, text: str, timeout_s: float) -> TimedPromptResult: ...


class StdTerminalIO:
    def write(self, text: str) -> None:
        print(text, flush=True)

    def prompt(self, text: str) -> str:
        return input(text)

    def prompt_with_timeout(self, text: str, timeout_s: float) -> TimedPromptResult:
        sys.stdout.write(text)
        sys.stdout.flush()
        ready, _, _ = select.select([sys.stdin], [], [], max(0.0, float(timeout_s)))
        if not ready:
            sys.stdout.write("\n")
            sys.stdout.flush()
            return TimedPromptResult(text="", timed_out=True)
        return TimedPromptResult(text=sys.stdin.readline().rstrip("\n"), timed_out=False)


@dataclass(frozen=True)
class InteractiveTaskRuntimeConfig:
    output_dir: str
    deadline_ms: float = 3000.0
    max_tokens: int = 8
    temperature: float = 0.0
    human_first: bool = True
    show_expected: bool = False
    deadline_mode: DeadlineMode = "menu"
    challenge_stage_turns: int = 20
    challenge_deadline_ms_list: tuple[float, ...] = (10000.0, 5000.0, 3000.0, 2000.0, 1000.0)


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
    deadline_mode: str = "soft"
    final_deadline_ms: float | None = None
    winner: str | None = None
    loser: str | None = None
    stop_reason: str = "completed"
    deadline_misses: tuple[dict[str, Any], ...] = ()
    invalid_responses: tuple[dict[str, Any], ...] = ()


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


@dataclass
class _DeadlineSessionState:
    mode: ResolvedDeadlineMode
    current_deadline_ms: float
    stage_index: int = 0
    stage_turn_count: int = 0
    deadline_misses: list[dict[str, Any]] = field(default_factory=list)
    invalid_responses: list[dict[str, Any]] = field(default_factory=list)
    winner: InteractiveActor | None = None
    loser: InteractiveActor | None = None
    stop_reason: str | None = None


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
        deadline_state = self._build_deadline_state()
        self._write_initial_manifest(
            run_dir,
            run_id=run_id,
            task=task,
            max_turns=max_turns,
            deadline_state=deadline_state,
        )

        state = task.initial_state()
        transcript: list[InteractiveTurnRecord] = []
        stats = _InteractiveStats()

        self._write_banner(task.name, deadline_state)
        try:
            while (
                stats.turn_count < int(max_turns)
                and not stats.quit_requested
                and deadline_state.stop_reason is None
            ):
                is_human_turn = (stats.turn_count % 2 == 0) if self.config.human_first else (stats.turn_count % 2 == 1)
                actor: InteractiveActor = "human" if is_human_turn else "llm"
                if actor == "human":
                    next_state = self._run_human_turn(
                        task=task,
                        state=state,
                        transcript=transcript,
                        stats=stats,
                        run_dir=run_dir,
                        deadline_state=deadline_state,
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
                        deadline_state=deadline_state,
                    )

            status = deadline_state.stop_reason or ("user_quit" if stats.quit_requested else "completed")
            result = self._result(run_dir, task.name, stats, deadline_state, stop_reason=status)
            self._write_run_summary(run_dir, task.name, result, status=status)
            self.terminal.write(self._summary_text(result, stats))
            return result
        except Exception:
            result = self._result(run_dir, task.name, stats, deadline_state, stop_reason="error")
            self._write_run_summary(run_dir, task.name, result, status="error")
            raise

    def _build_deadline_state(self) -> _DeadlineSessionState:
        mode = self._resolve_deadline_mode()
        deadline_ms = float(self.config.deadline_ms)
        if mode == "challenge":
            schedule = self._challenge_schedule()
            deadline_ms = schedule[0]
        return _DeadlineSessionState(mode=mode, current_deadline_ms=deadline_ms)

    def _resolve_deadline_mode(self) -> ResolvedDeadlineMode:
        mode = self.config.deadline_mode
        if mode in {"soft", "hard", "challenge"}:
            return mode  # type: ignore[return-value]
        while True:
            choice = self.terminal.prompt(
                "Select deadline mode:\n"
                "1) Soft deadline - keep playing after misses; report them at the end.\n"
                "2) Hard deadline - first timeout or wrong answer loses immediately.\n"
                "3) Challenge mode - pass clean stages to reduce the deadline until someone loses.\n"
                "Choice [1-3]: "
            ).strip().lower()
            if choice in {"1", "soft", "s"}:
                return "soft"
            if choice in {"2", "hard", "h"}:
                return "hard"
            if choice in {"3", "challenge", "c"}:
                return "challenge"
            self.terminal.write("Please choose 1, 2, or 3.")

    def _challenge_schedule(self) -> tuple[float, ...]:
        schedule = tuple(float(value) for value in self.config.challenge_deadline_ms_list if float(value) > 0)
        if schedule:
            return schedule
        return (max(1.0, float(self.config.deadline_ms)),)

    def _run_human_turn(
        self,
        *,
        task: InteractiveTask,
        state: TaskState,
        transcript: list[InteractiveTurnRecord],
        stats: _InteractiveStats,
        run_dir: Path,
        deadline_state: _DeadlineSessionState,
    ) -> TaskState | None:
        prompt_text = task.build_human_prompt(state, transcript)
        number = self._number_from_state(state)

        while True:
            prompt = (
                f"[{number if number is not None else state.turn_index + 1}] "
                f"You > {prompt_text} {self._deadline_suffix(deadline_state)} "
            )
            start_s = self._now()
            if deadline_state.mode == "soft":
                response_text = self.terminal.prompt(prompt)
                timed_out = False
            else:
                prompt_result = self.terminal.prompt_with_timeout(prompt, deadline_state.current_deadline_ms / 1000.0)
                response_text = prompt_result.text
                timed_out = bool(prompt_result.timed_out)
            elapsed_ms = max(0.0, (self._now() - start_s) * 1000.0)
            if timed_out:
                elapsed_ms = max(elapsed_ms, float(deadline_state.current_deadline_ms))
                self.terminal.write("    Deadline expired before input.")
                break
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
            deadline_state=deadline_state,
            forced_deadline_missed=timed_out,
        )

    def _run_llm_turn(
        self,
        *,
        task: InteractiveTask,
        state: TaskState,
        transcript: list[InteractiveTurnRecord],
        stats: _InteractiveStats,
        run_dir: Path,
        deadline_state: _DeadlineSessionState,
    ) -> TaskState:
        number = self._number_from_state(state)
        self.terminal.write(
            f"[{number if number is not None else state.turn_index + 1}] "
            f"LLM thinking... {self._deadline_suffix(deadline_state)}"
        )
        messages = task.build_llm_messages(state, transcript)
        start_s = self._now()
        try:
            model_response = self.model_client.generate(
                messages,
                max_tokens=int(self.config.max_tokens),
                temperature=float(self.config.temperature),
                timeout_s=None if deadline_state.mode == "soft" else deadline_state.current_deadline_ms / 1000.0,
            )
        except requests.Timeout as exc:
            elapsed_ms = max(0.0, (self._now() - start_s) * 1000.0)
            elapsed_ms = max(elapsed_ms, float(deadline_state.current_deadline_ms))
            self.terminal.write("    LLM > [timeout]")
            return self._finish_turn(
                task=task,
                state=state,
                transcript=transcript,
                stats=stats,
                run_dir=run_dir,
                actor="llm",
                prompt_payload=messages,
                response_text="",
                elapsed_ms=elapsed_ms,
                prompt_tokens=None,
                completion_tokens=None,
                raw_model_response=None,
                deadline_state=deadline_state,
                forced_deadline_missed=True,
                model_error=str(exc),
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
            deadline_state=deadline_state,
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
        deadline_state: _DeadlineSessionState,
        raw_model_response: ChatModelResponse | None = None,
        forced_deadline_missed: bool = False,
        model_error: str | None = None,
    ) -> TaskState:
        referee = task.validate_response(
            state,
            response_text,
            actor=actor,
            transcript=transcript,
        )
        deadline_missed = forced_deadline_missed or elapsed_ms > float(deadline_state.current_deadline_ms)
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
                "deadline_mode": deadline_state.mode,
                "deadline_ms": float(deadline_state.current_deadline_ms),
                "challenge_stage_index": deadline_state.stage_index,
                "challenge_stage_turn_count": deadline_state.stage_turn_count,
                "model_error": model_error,
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
        self._handle_turn_outcome(record, deadline_state)
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
        loss_text = self._loss_text(record, deadline_state)
        if loss_text:
            self.terminal.write(loss_text)
        return next_state

    def _handle_turn_outcome(self, record: InteractiveTurnRecord, deadline_state: _DeadlineSessionState) -> None:
        if record.deadline_missed:
            deadline_state.deadline_misses.append(self._deadline_miss_detail(record, deadline_state))
        if not record.is_valid:
            deadline_state.invalid_responses.append(self._invalid_response_detail(record, deadline_state))
        if deadline_state.mode == "soft":
            return
        if record.deadline_missed:
            deadline_state.loser = record.actor
            deadline_state.winner = self._opponent(record.actor)
            deadline_state.stop_reason = "deadline_loss"
            return
        if not record.is_valid:
            deadline_state.loser = record.actor
            deadline_state.winner = self._opponent(record.actor)
            deadline_state.stop_reason = "invalid_response_loss"
            return
        if deadline_state.mode == "challenge":
            deadline_state.stage_turn_count += 1
            stage_turns = max(1, int(self.config.challenge_stage_turns))
            schedule = self._challenge_schedule()
            if deadline_state.stage_turn_count >= stage_turns:
                if deadline_state.stage_index < len(schedule) - 1:
                    old_deadline = deadline_state.current_deadline_ms
                    deadline_state.stage_index += 1
                    deadline_state.stage_turn_count = 0
                    deadline_state.current_deadline_ms = schedule[deadline_state.stage_index]
                    self.terminal.write(
                        f"Stage passed: {stage_turns} turns under {self._format_ms(old_deadline)}."
                    )
                    self.terminal.write(f"New deadline: {self._format_ms(deadline_state.current_deadline_ms)}.")
                else:
                    deadline_state.stage_turn_count = 0

    def _deadline_miss_detail(
        self,
        record: InteractiveTurnRecord,
        deadline_state: _DeadlineSessionState,
    ) -> dict[str, Any]:
        return {
            "turn_id": record.turn_id,
            "actor": record.actor,
            "number": record.number,
            "latency_ms": record.latency_ms,
            "deadline_ms": float(record.metadata.get("deadline_ms", deadline_state.current_deadline_ms)),
            "expected": record.expected,
            "response": record.response,
        }

    def _invalid_response_detail(
        self,
        record: InteractiveTurnRecord,
        deadline_state: _DeadlineSessionState,
    ) -> dict[str, Any]:
        _ = deadline_state
        return {
            "turn_id": record.turn_id,
            "actor": record.actor,
            "number": record.number,
            "expected": record.expected,
            "response": record.response,
            "failure_reason": record.metadata.get("failure_reason"),
        }

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
            soft_state = _DeadlineSessionState(mode="soft", current_deadline_ms=float(self.config.deadline_ms))
            self.terminal.write(
                self._summary_text(
                    self._result(Path(self.config.output_dir), task.name, stats, soft_state, stop_reason="running"),
                    stats,
                )
            )
            return
        self.terminal.write(f"Unknown command: {command}. Type :help for commands.")

    def _write_banner(self, task_name: str, deadline_state: _DeadlineSessionState) -> None:
        self.terminal.write(f"StreamMUSE {task_name}")
        self.terminal.write(
            f"Deadline mode: {deadline_state.mode}, current deadline {self._format_ms(deadline_state.current_deadline_ms)}"
        )
        if deadline_state.mode == "challenge":
            self.terminal.write(
                f"Challenge stage 1: deadline {self._format_ms(deadline_state.current_deadline_ms)}, "
                f"target {max(1, int(self.config.challenge_stage_turns))} turns"
            )
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
        deadline = " deadline_missed" if record.deadline_missed else ""
        if record.is_valid:
            if self.config.show_expected and record.expected is not None:
                return f"OK expected={record.expected}{deadline}{latency}"
            return f"OK{deadline}{latency}"
        return f"MISS expected={record.expected} actual={record.response}{deadline}{latency}"

    def _loss_text(self, record: InteractiveTurnRecord, deadline_state: _DeadlineSessionState) -> str | None:
        if deadline_state.loser != record.actor:
            return None
        if deadline_state.stop_reason == "deadline_loss":
            return (
                f"Deadline missed: {record.actor} used {float(record.latency_ms or 0.0):.1f}ms "
                f"> {self._format_ms(float(record.metadata.get('deadline_ms', deadline_state.current_deadline_ms)))}\n"
                f"Winner: {deadline_state.winner}."
            )
        if deadline_state.stop_reason == "invalid_response_loss":
            return (
                f"Wrong answer: {record.actor} answered {record.response}, expected {record.expected}\n"
                f"Winner: {deadline_state.winner}."
            )
        return None

    def _summary_text(self, result: InteractiveTaskRunResult, stats: _InteractiveStats) -> str:
        avg = stats.avg_llm_latency_ms
        avg_text = "n/a" if avg is None else f"{avg:.1f} ms"
        lines = [
            f"Summary: {result.turn_count} turns, {result.valid_count} valid, "
            f"{result.invalid_count} invalid, {result.deadline_miss_count} deadline misses, "
            f"avg LLM latency {avg_text}"
        ]
        lines.append(f"Stop reason: {result.stop_reason}")
        if result.winner or result.loser:
            lines.append(f"Winner: {result.winner}; loser: {result.loser}")
        if result.deadline_misses:
            lines.append("Deadline misses:")
            for item in result.deadline_misses:
                lines.append(
                    f"  turn={item.get('turn_id')} actor={item.get('actor')} number={item.get('number')} "
                    f"latency={float(item.get('latency_ms') or 0.0):.1f}ms "
                    f"deadline={self._format_ms(float(item.get('deadline_ms') or 0.0))}"
                )
        if result.invalid_responses:
            lines.append("Invalid responses:")
            for item in result.invalid_responses:
                lines.append(
                    f"  turn={item.get('turn_id')} actor={item.get('actor')} number={item.get('number')} "
                    f"expected={item.get('expected')} response={item.get('response')}"
                )
        return "\n".join(lines)

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

    def _write_initial_manifest(
        self,
        run_dir: Path,
        *,
        run_id: str,
        task: InteractiveTask,
        max_turns: int,
        deadline_state: _DeadlineSessionState,
    ) -> None:
        manifest = {
            "schema_version": 1,
            "run_id": run_id,
            "mode": "interactive",
            "task_name": task.name,
            "status": "running",
            "max_turns": int(max_turns),
            "deadline_ms": float(self.config.deadline_ms),
            "deadline_mode": deadline_state.mode,
            "configured_deadline_mode": self.config.deadline_mode,
            "current_deadline_ms": float(deadline_state.current_deadline_ms),
            "challenge_stage_turns": int(self.config.challenge_stage_turns),
            "challenge_deadline_ms_list": list(self._challenge_schedule()),
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
                "deadline_mode": result.deadline_mode,
                "final_deadline_ms": result.final_deadline_ms,
                "winner": result.winner,
                "loser": result.loser,
                "stop_reason": result.stop_reason,
                "deadline_misses": list(result.deadline_misses),
                "invalid_responses": list(result.invalid_responses),
            }
        )
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    def _result(
        self,
        run_dir: Path,
        task_name: str,
        stats: _InteractiveStats,
        deadline_state: _DeadlineSessionState,
        *,
        stop_reason: str,
    ) -> InteractiveTaskRunResult:
        return InteractiveTaskRunResult(
            output_dir=str(run_dir),
            task_name=task_name,
            turn_count=stats.turn_count,
            human_turn_count=stats.human_turn_count,
            llm_turn_count=stats.llm_turn_count,
            valid_count=stats.valid_count,
            invalid_count=stats.invalid_count,
            deadline_miss_count=stats.deadline_miss_count,
            deadline_mode=deadline_state.mode,
            final_deadline_ms=float(deadline_state.current_deadline_ms),
            winner=deadline_state.winner,
            loser=deadline_state.loser,
            stop_reason=stop_reason,
            deadline_misses=tuple(deadline_state.deadline_misses),
            invalid_responses=tuple(deadline_state.invalid_responses),
        )

    @staticmethod
    def _opponent(actor: InteractiveActor) -> InteractiveActor:
        return "llm" if actor == "human" else "human"

    @staticmethod
    def _format_ms(value: float) -> str:
        if float(value).is_integer():
            return f"{int(value)}ms"
        return f"{value:.1f}ms"

    def _deadline_suffix(self, deadline_state: _DeadlineSessionState) -> str:
        return f"(deadline {self._format_ms(deadline_state.current_deadline_ms)})"
