"""Interactive terminal runtime for human/LLM tasks."""

from __future__ import annotations

import json
import math
import time
import uuid
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

import requests

from streammuse.application.tasks.human_input import (
    StdTerminalIO,
    TerminalHumanResponseSource,
    TerminalIO,
    TimedPromptResult as TimedPromptResult,
)
from streammuse.domain.tasks import (
    ChatModelResponse,
    DeadlineMode,
    HumanResponse,
    HumanResponseRequest,
    HumanResponseSource,
    InteractiveActor,
    InteractiveTask,
    InteractiveTurnRecord,
    LocalChatModel,
    SpeechAwareInteractiveTask,
    TaskRefereeResult,
    TaskState,
)


ResolvedDeadlineMode = Literal["soft", "hard", "challenge"]


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

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.deadline_ms)) or float(self.deadline_ms) <= 0:
            raise ValueError("deadline_ms must be finite and > 0")
        if any(
            not math.isfinite(float(value)) or float(value) <= 0
            for value in self.challenge_deadline_ms_list
        ):
            raise ValueError("challenge deadlines must be finite and > 0")


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
        human_response_source: HumanResponseSource | None = None,
        now: Callable[[], float] | None = None,
    ) -> None:
        self.config = config
        self.model_client = model_client
        self.terminal = terminal if terminal is not None else StdTerminalIO()
        self.human_response_source = (
            human_response_source
            if human_response_source is not None
            else TerminalHumanResponseSource(self.terminal)
        )
        self._now = now or time.perf_counter

    def play(self, task: InteractiveTask, *, max_turns: int) -> InteractiveTaskRunResult:
        run_dir = Path(self.config.output_dir).expanduser()
        run_id = f"interactive-{uuid.uuid4().hex[:8]}"
        transcript: list[InteractiveTurnRecord] = []
        stats = _InteractiveStats()
        deadline_state: _DeadlineSessionState | None = None
        result: InteractiveTaskRunResult | None = None
        manifest_written = False
        startup_complete = False
        run_dir_ready = False
        primary_error: BaseException | None = None
        source_close_attempted = False

        try:
            run_dir = run_dir.resolve()
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "artifacts" / "turn").mkdir(parents=True, exist_ok=True)
            run_dir_ready = True
            self._validate_human_input_task(task)
            deadline_state = self._build_deadline_state()
            self.human_response_source.start()
            human_input = self._human_input_provenance()
            self._write_initial_manifest(
                run_dir,
                run_id=run_id,
                task=task,
                max_turns=max_turns,
                deadline_state=deadline_state,
                human_input=human_input,
            )
            manifest_written = True
            startup_complete = True

            state = task.initial_state()
            self._write_banner(task.name, deadline_state)
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
            source_close_attempted = True
            self.human_response_source.close()
            result = self._result(run_dir, task.name, stats, deadline_state, stop_reason=status)
            self._write_run_summary(run_dir, task.name, result, status=status)
            self.terminal.write(self._summary_text(result, stats))
        except KeyboardInterrupt as exc:
            primary_error = exc
            active_deadline_state = deadline_state or self._fallback_deadline_state()
            try:
                if run_dir_ready and not manifest_written:
                    self._write_initial_manifest(
                        run_dir,
                        run_id=run_id,
                        task=task,
                        max_turns=max_turns,
                        deadline_state=active_deadline_state,
                        human_input=self._safe_human_input_provenance(),
                        status="user_interrupt",
                    )
                    manifest_written = True
                if run_dir_ready:
                    result = self._result(
                        run_dir,
                        task.name,
                        stats,
                        active_deadline_state,
                        stop_reason="user_interrupt",
                    )
                    self._write_run_summary(run_dir, task.name, result, status="user_interrupt")
            except BaseException:
                pass
            raise
        except BaseException as exc:
            primary_error = exc
            active_deadline_state = deadline_state or self._fallback_deadline_state()
            try:
                if run_dir_ready and not startup_complete:
                    self._write_initial_manifest(
                        run_dir,
                        run_id=run_id,
                        task=task,
                        max_turns=max_turns,
                        deadline_state=active_deadline_state,
                        human_input=self._safe_human_input_provenance(),
                        status="startup_error",
                        startup_error=exc,
                    )
                    manifest_written = True
                elif run_dir_ready:
                    result = self._result(run_dir, task.name, stats, active_deadline_state, stop_reason="error")
                    self._write_run_summary(run_dir, task.name, result, status="error")
            except BaseException:
                pass
            raise
        finally:
            if not source_close_attempted:
                try:
                    self.human_response_source.close()
                except BaseException:
                    if primary_error is None:
                        raise

        if result is None:
            raise RuntimeError("Interactive task ended without a result")
        return result

    def _build_deadline_state(self) -> _DeadlineSessionState:
        mode = self._resolve_deadline_mode()
        deadline_ms = float(self.config.deadline_ms)
        if mode == "challenge":
            schedule = self._challenge_schedule()
            deadline_ms = schedule[0]
        return _DeadlineSessionState(mode=mode, current_deadline_ms=deadline_ms)

    def _fallback_deadline_state(self) -> _DeadlineSessionState:
        configured_mode = self.config.deadline_mode
        mode: ResolvedDeadlineMode = configured_mode if configured_mode != "menu" else "soft"
        deadline_ms = float(self.config.deadline_ms)
        if mode == "challenge":
            deadline_ms = self._challenge_schedule()[0]
        return _DeadlineSessionState(mode=mode, current_deadline_ms=deadline_ms)

    def _validate_human_input_task(self, task: InteractiveTask) -> None:
        if self.human_response_source.mode == "voice" and not isinstance(task, SpeechAwareInteractiveTask):
            raise TypeError(
                f"Task {task.name!r} does not support voice input; "
                "use terminal input or a SpeechAwareInteractiveTask"
            )

    @staticmethod
    def _speech_task(task: InteractiveTask) -> SpeechAwareInteractiveTask:
        if not isinstance(task, SpeechAwareInteractiveTask):
            raise TypeError(f"Task {task.name!r} does not support voice input")
        return task

    def _human_input_provenance(self) -> dict[str, Any]:
        provenance = deepcopy(self.human_response_source.provenance)
        provenance["mode"] = self.human_response_source.mode
        return provenance

    def _safe_human_input_provenance(self) -> dict[str, Any]:
        try:
            return self._human_input_provenance()
        except BaseException as exc:
            return {
                "mode": self.human_response_source.mode,
                "provenance_error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
            }

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
            speech_context = None
            if self.human_response_source.mode == "voice":
                speech_task = self._speech_task(task)
                speech_context = speech_task.build_speech_context(state, transcript)
            turn_started_s = self._now()
            prompt_elapsed_ms = 0.0
            if self.human_response_source.mode == "voice":
                self.terminal.write(prompt.rstrip())
                prompt_elapsed_ms = max(0.0, (self._now() - turn_started_s) * 1000.0)
            timeout_s = self._remaining_human_timeout_s(deadline_state, prompt_elapsed_ms)
            request = HumanResponseRequest(
                turn_id=len(transcript),
                prompt=prompt,
                timeout_s=timeout_s,
                speech_context=speech_context,
            )
            read_started_s = self._now()
            if self.human_response_source.mode == "voice" and timeout_s == 0.0:
                response = HumanResponse(
                    text="",
                    status="no_speech",
                    deadline_expired=True,
                    latency_ms=0.0,
                )
            else:
                response = self.human_response_source.read_response(request)
            response_ready_s = self._now()
            wall_origin_s = (
                read_started_s
                if self.human_response_source.mode == "voice"
                else turn_started_s
            )
            wall_read_elapsed_ms = max(
                0.0,
                (response_ready_s - wall_origin_s) * 1000.0,
            )
            response_elapsed_ms = self._response_latency_ms(
                response,
                wall_elapsed_ms=wall_read_elapsed_ms,
                allow_artifact_exclusion=self.human_response_source.mode == "voice",
            )
            elapsed_ms = prompt_elapsed_ms + response_elapsed_ms
            response_text = str(response.text or "")
            stripped = response_text.strip()
            if (
                self.human_response_source.mode == "terminal"
                and stripped.startswith(":")
                and not response.deadline_expired
            ):
                self._handle_command(
                    stripped,
                    task=task,
                    state=state,
                    transcript=transcript,
                    stats=stats,
                )
                if stats.quit_requested:
                    return None
                continue
            timed_out = bool(response.deadline_expired) or (
                deadline_state.mode != "soft"
                and elapsed_ms >= float(deadline_state.current_deadline_ms)
            )
            if timed_out:
                elapsed_ms = max(elapsed_ms, float(deadline_state.current_deadline_ms))
                self.terminal.write("    Deadline expired before input.")
                break
            break

        canonical_response = str(response_text or "").strip()
        human_input_metadata = deepcopy(response.metadata)
        if self.human_response_source.mode == "voice" and prompt_elapsed_ms:
            self._shift_voice_stage_offsets(human_input_metadata, prompt_elapsed_ms)
        human_input_metadata.update(
            {
                "mode": self.human_response_source.mode,
                "status": response.status,
                "total_latency_ms": float(elapsed_ms),
            }
        )
        if self.human_response_source.mode == "voice":
            human_input_metadata["raw_transcript"] = str(response_text or "")
            if response.status == "ok":
                spoken_parse = self._speech_task(task).parse_spoken_response(
                    state,
                    transcript,
                    str(response_text or ""),
                )
                canonical_response = str(spoken_parse.canonical_text or "").strip()
                human_input_metadata["parse_status"] = spoken_parse.status
                human_input_metadata["parse_reason"] = spoken_parse.reason
            else:
                canonical_response = ""
                human_input_metadata["parse_status"] = response.status
                rejection_reasons = human_input_metadata.get(
                    "transcript_rejection_reasons"
                )
                if isinstance(rejection_reasons, list) and rejection_reasons:
                    human_input_metadata["parse_reason"] = ",".join(
                        str(reason) for reason in rejection_reasons
                    )
                else:
                    human_input_metadata["parse_reason"] = response.status
            human_input_metadata["canonical_response"] = canonical_response
            self.terminal.write(
                self._voice_transcript_text(
                    raw_transcript=str(response_text or ""),
                    canonical_response=canonical_response,
                    parse_status=str(human_input_metadata["parse_status"]),
                )
            )

        return self._finish_turn(
            task=task,
            state=state,
            transcript=transcript,
            stats=stats,
            run_dir=run_dir,
            actor="human",
            prompt_payload=prompt_text,
            response_text=canonical_response,
            elapsed_ms=elapsed_ms,
            prompt_tokens=None,
            completion_tokens=None,
            deadline_state=deadline_state,
            forced_deadline_missed=timed_out,
            human_input_metadata=human_input_metadata,
        )

    @staticmethod
    def _response_latency_ms(
        response: HumanResponse,
        *,
        wall_elapsed_ms: float,
        allow_artifact_exclusion: bool,
    ) -> float:
        reported_latency_ms = response.latency_ms
        if reported_latency_ms is not None:
            reported_latency_ms = float(reported_latency_ms)
        if reported_latency_ms is not None and (
            not math.isfinite(reported_latency_ms) or reported_latency_ms < 0
        ):
            raise ValueError("Human response latency_ms must be finite and non-negative")

        artifact_persistence_ms = 0.0
        if allow_artifact_exclusion and response.metadata.get("audio_artifact"):
            raw_persistence_ms = response.metadata.get("artifact_persistence_ms", 0.0)
            if isinstance(raw_persistence_ms, bool) or not isinstance(
                raw_persistence_ms,
                (int, float),
            ):
                raise ValueError("artifact_persistence_ms must be numeric")
            artifact_persistence_ms = float(raw_persistence_ms)
            if not math.isfinite(artifact_persistence_ms) or artifact_persistence_ms < 0:
                raise ValueError("artifact_persistence_ms must be finite and non-negative")
            if artifact_persistence_ms > wall_elapsed_ms + 1.0:
                raise ValueError("artifact_persistence_ms exceeds measured response wall time")
            if reported_latency_ms is None:
                raise ValueError("audio persistence exclusion requires a reported response latency")

        measured_pipeline_ms = max(0.0, wall_elapsed_ms - artifact_persistence_ms)
        if reported_latency_ms is None:
            return measured_pipeline_ms
        return max(reported_latency_ms, measured_pipeline_ms)

    @staticmethod
    def _remaining_human_timeout_s(
        deadline_state: _DeadlineSessionState,
        elapsed_ms: float,
    ) -> float | None:
        if deadline_state.mode == "soft":
            return None
        remaining_ms = max(0.0, float(deadline_state.current_deadline_ms) - elapsed_ms)
        return remaining_ms / 1000.0

    @staticmethod
    def _shift_voice_stage_offsets(metadata: dict[str, Any], offset_ms: float) -> None:
        for key in (
            "last_voiced_offset_ms",
            "endpoint_detected_offset_ms",
            "asr_start_offset_ms",
            "asr_end_offset_ms",
        ):
            value = metadata.get(key)
            if value is not None and isinstance(value, (int, float)) and math.isfinite(float(value)):
                metadata[key] = float(value) + offset_ms

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
        human_input_metadata: dict[str, Any] | None = None,
    ) -> TaskState:
        referee = task.validate_response(
            state,
            response_text,
            actor=actor,
            transcript=transcript,
        )
        deadline_missed = forced_deadline_missed or elapsed_ms > float(deadline_state.current_deadline_ms)
        number = self._number_from_state(state)
        metadata: dict[str, Any] = {
            "failure_reason": referee.failure_reason,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "deadline_mode": deadline_state.mode,
            "deadline_ms": float(deadline_state.current_deadline_ms),
            "challenge_stage_index": deadline_state.stage_index,
            "challenge_stage_turn_count": deadline_state.stage_turn_count,
            "model_error": model_error,
        }
        if human_input_metadata is not None:
            metadata["human_input"] = deepcopy(human_input_metadata)
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
            metadata=metadata,
        )
        next_state = task.advance_state(
            state,
            referee,
            response_text,
            actor=actor,
            transcript=transcript,
        )
        artifact_path = self._turn_artifact_path(run_dir, record.turn_id)
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
        try:
            self._append_response_trace(run_dir, record)
        except BaseException:
            try:
                artifact_path.unlink(missing_ok=True)
            except BaseException:
                pass
            raise
        transcript.append(record)
        self._update_stats(stats, record)
        self._handle_turn_outcome(record, deadline_state)
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
        actual = record.response or self._empty_voice_response_label(record)
        return f"MISS expected={record.expected} actual={actual}{deadline}{latency}"

    @staticmethod
    def _voice_transcript_text(
        *,
        raw_transcript: str,
        canonical_response: str,
        parse_status: str,
    ) -> str:
        raw_display = json.dumps(raw_transcript, ensure_ascii=False)
        if canonical_response:
            parsed_display = json.dumps(canonical_response, ensure_ascii=False)
        else:
            parsed_display = f"[{parse_status}]"
        return f"    ASR > raw={raw_display} parsed={parsed_display}"

    @staticmethod
    def _empty_voice_response_label(record: InteractiveTurnRecord) -> str:
        human_input = record.metadata.get("human_input")
        if not isinstance(human_input, dict) or human_input.get("mode") != "voice":
            return ""
        parse_status = str(human_input.get("parse_status") or "empty")
        return f"[{parse_status}]"

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
        path = run_dir / "response_trace.jsonl"
        row = (json.dumps(asdict(record), ensure_ascii=False, default=str) + "\n").encode(
            "utf-8"
        )
        existed = path.exists()
        original_size = path.stat().st_size if existed else 0
        try:
            self._write_response_trace_row(path, row)
        except BaseException:
            self._rollback_response_trace(
                path,
                original_size=original_size,
                existed=existed,
            )
            raise

    @staticmethod
    def _write_response_trace_row(path: Path, row: bytes) -> None:
        with path.open("ab") as handle:
            handle.write(row)
            handle.flush()

    @staticmethod
    def _rollback_response_trace(
        path: Path,
        *,
        original_size: int,
        existed: bool,
    ) -> None:
        try:
            if path.exists():
                with path.open("r+b") as handle:
                    handle.truncate(original_size)
                if not existed:
                    path.unlink(missing_ok=True)
        except BaseException:
            # Preserve the append failure. The run remains status=error even if
            # the underlying filesystem also refuses the best-effort rollback.
            pass

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
        path = self._turn_artifact_path(run_dir, record.turn_id)
        temporary = path.with_suffix(".json.tmp")
        try:
            temporary.write_text(
                json.dumps(payload, indent=2, sort_keys=True, default=str),
                encoding="utf-8",
            )
            temporary.replace(path)
        except BaseException:
            try:
                temporary.unlink(missing_ok=True)
            except BaseException:
                pass
            raise

    @staticmethod
    def _turn_artifact_path(run_dir: Path, turn_id: int) -> Path:
        return run_dir / "artifacts" / "turn" / f"{turn_id + 1:04d}_turn_{turn_id:04d}.json"

    def _write_initial_manifest(
        self,
        run_dir: Path,
        *,
        run_id: str,
        task: InteractiveTask,
        max_turns: int,
        deadline_state: _DeadlineSessionState,
        human_input: dict[str, Any],
        status: str = "running",
        startup_error: BaseException | None = None,
    ) -> None:
        manifest: dict[str, Any] = {
            "schema_version": 2,
            "run_id": run_id,
            "mode": "interactive",
            "task_name": task.name,
            "status": status,
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
            "human_input": deepcopy(human_input),
        }
        if startup_error is not None:
            manifest["startup_error"] = {
                "type": type(startup_error).__name__,
                "message": str(startup_error),
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
