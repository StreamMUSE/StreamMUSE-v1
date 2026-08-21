from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import requests

from streammuse.application.tasks import InteractiveTaskRuntime, InteractiveTaskRuntimeConfig, TimedPromptResult
from streammuse.domain.tasks import (
    AnimalNamingTask,
    ChatModelResponse,
    HumanInputMode,
    HumanResponse,
    HumanResponseRequest,
    HumanResponseStatus,
    InteractiveTurnRecord,
    SpeechContext,
    SpokenResponseParseResult,
    TaskState,
    TaskViewEvent,
    ZipZapZopTask,
)


class StaticModelClient:
    def __init__(
        self,
        responses: list[str],
        *,
        latency_ms: float = 25.0,
        errors: list[BaseException] | None = None,
    ) -> None:
        self.responses = list(responses)
        self.errors = list(errors or [])
        self.latency_ms = latency_ms
        self.requests: list[list[dict[str, str]]] = []
        self.kwargs: list[dict[str, object]] = []

    def generate(self, messages: list[dict[str, str]], **kwargs: object) -> ChatModelResponse:
        self.requests.append(messages)
        self.kwargs.append(kwargs)
        if self.errors:
            raise self.errors.pop(0)
        return ChatModelResponse(
            text=self.responses.pop(0),
            latency_ms=self.latency_ms,
            prompt_tokens=12,
            completion_tokens=1,
            raw={"fake": True},
        )


class ManualClock:
    def __init__(self) -> None:
        self.now_s = 0.0

    def now(self) -> float:
        return self.now_s

    def advance(self, seconds: float) -> None:
        self.now_s += seconds


class FakeTerminal:
    def __init__(
        self,
        inputs: list[object],
        *,
        clock: ManualClock | None = None,
        prompt_elapsed_s: float = 0.0,
        prompt_elapsed_sequence: list[float] | None = None,
        write_elapsed_s: float = 0.0,
    ) -> None:
        self.inputs = list(inputs)
        self.outputs: list[str] = []
        self.clock = clock
        self.prompt_elapsed_s = prompt_elapsed_s
        self.prompt_elapsed_sequence = list(prompt_elapsed_sequence or [])
        self.write_elapsed_s = write_elapsed_s
        self.timeout_prompts: list[float] = []

    def write(self, text: str) -> None:
        self.outputs.append(text)
        if self.clock is not None:
            self.clock.advance(self.write_elapsed_s)

    def prompt(self, text: str) -> str:
        self.outputs.append(f"PROMPT {text}")
        self._advance_prompt_clock()
        return str(self._pop_input())

    def prompt_with_timeout(self, text: str, timeout_s: float) -> TimedPromptResult:
        self.outputs.append(f"TIMED_PROMPT {text}")
        self.timeout_prompts.append(timeout_s)
        self._advance_prompt_clock()
        value = self._pop_input()
        if isinstance(value, TimedPromptResult):
            return value
        return TimedPromptResult(text=str(value), timed_out=False)

    def _advance_prompt_clock(self) -> None:
        if self.clock is None:
            return
        elapsed_s = (
            self.prompt_elapsed_sequence.pop(0)
            if self.prompt_elapsed_sequence
            else self.prompt_elapsed_s
        )
        self.clock.advance(elapsed_s)

    def _pop_input(self) -> object:
        assert self.inputs, "test terminal input exhausted"
        return self.inputs.pop(0)


class FakeHumanResponseSource:
    def __init__(
        self,
        responses: list[HumanResponse | BaseException],
        *,
        mode: HumanInputMode = "voice",
        provenance: dict[str, Any] | None = None,
        start_error: BaseException | None = None,
        close_error: BaseException | None = None,
        clock: ManualClock | None = None,
        response_elapsed_s: float = 0.0,
    ) -> None:
        self.responses = list(responses)
        self.mode = mode
        self._provenance = provenance or {"mode": mode}
        self.start_error = start_error
        self.close_error = close_error
        self.clock = clock
        self.response_elapsed_s = response_elapsed_s
        self.requests: list[HumanResponseRequest] = []
        self.start_count = 0
        self.close_count = 0

    def start(self) -> None:
        self.start_count += 1
        if self.start_error is not None:
            raise self.start_error

    @property
    def provenance(self) -> dict[str, Any]:
        return self._provenance

    def read_response(self, request: HumanResponseRequest) -> HumanResponse:
        self.requests.append(request)
        if self.clock is not None:
            self.clock.advance(self.response_elapsed_s)
        assert self.responses, "test human response source exhausted"
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response

    def close(self) -> None:
        self.close_count += 1
        if self.close_error is not None:
            raise self.close_error


class SpeechAwareZipZapZopTask(ZipZapZopTask):
    def build_speech_context(
        self,
        state: TaskState,
        transcript: list[InteractiveTurnRecord],
    ) -> SpeechContext:
        _ = state, transcript
        return SpeechContext(initial_prompt="Say the game answer.", hotwords=("Zip", "Zap", "Zop"))

    def parse_spoken_response(
        self,
        state: TaskState,
        transcript: list[InteractiveTurnRecord],
        raw_text: str,
    ) -> SpokenResponseParseResult:
        _ = state, transcript
        canonical = {"one": "1", "zip zap.": "ZipZap"}.get(raw_text.strip().lower())
        if canonical is None:
            return SpokenResponseParseResult(canonical_text=None, status="unrecognized", reason="unsupported_grammar")
        return SpokenResponseParseResult(canonical_text=canonical, status="ok")


class NonSpeechInteractiveTask:
    name = "plain_interactive"


class RecordingTaskEventSink:
    def __init__(self, error: BaseException | None = None) -> None:
        self.events: list[TaskViewEvent] = []
        self.error = error

    def emit(self, event: TaskViewEvent) -> None:
        if self.error is not None:
            raise self.error
        self.events.append(event)

    def close(self) -> None:
        return None


def _read_response_trace(trace_dir: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in (trace_dir / "response_trace.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_task_view_events_follow_attempts_and_do_not_leak_expected_by_default(tmp_path) -> None:
    terminal = FakeTerminal([":hint", "1"])
    sink = RecordingTaskEventSink()
    runtime = InteractiveTaskRuntime(
        config=InteractiveTaskRuntimeConfig(
            output_dir=str(tmp_path), deadline_mode="soft", deadline_ms=1000.0
        ),
        model_client=StaticModelClient([]),
        terminal=terminal,
        task_event_sink=sink,
        task_event_session_id="session",
    )

    result = runtime.play(ZipZapZopTask(start_number=1), max_turns=1)

    attempts = [event for event in sink.events if event.type == "turn_attempt_started"]
    assert [event.payload["attempt_index"] for event in attempts] == [0, 1]
    assert all(event.payload["deadline_ms"] == 1000.0 for event in attempts)
    finished = next(event for event in sink.events if event.type == "turn_finished")
    assert "expected" not in finished.payload
    assert finished.payload["stats"]["turn_count"] == 1
    assert sink.events[-1].type == "session_finished"
    assert result.turn_count == 1


def test_task_event_sink_exception_isolated_but_keyboard_interrupt_propagates(tmp_path) -> None:
    runtime = InteractiveTaskRuntime(
        config=InteractiveTaskRuntimeConfig(
            output_dir=str(tmp_path / "ordinary"), deadline_mode="soft"
        ),
        model_client=StaticModelClient([]),
        terminal=FakeTerminal(["1"]),
        task_event_sink=RecordingTaskEventSink(RuntimeError("viewer failed")),
    )
    result = runtime.play(ZipZapZopTask(), max_turns=1)
    assert result.turn_count == 1
    summary = json.loads((tmp_path / "ordinary" / "run_summary.json").read_text())
    assert summary["task_event_error_count"] > 0

    interrupting = InteractiveTaskRuntime(
        config=InteractiveTaskRuntimeConfig(
            output_dir=str(tmp_path / "interrupt"), deadline_mode="soft"
        ),
        model_client=StaticModelClient([]),
        terminal=FakeTerminal(["1"]),
        task_event_sink=RecordingTaskEventSink(KeyboardInterrupt()),
    )
    with pytest.raises(KeyboardInterrupt):
        interrupting.play(ZipZapZopTask(), max_turns=1)


def test_challenge_stage_event_follows_turn_and_expected_is_opt_in(tmp_path) -> None:
    sink = RecordingTaskEventSink()
    runtime = InteractiveTaskRuntime(
        config=InteractiveTaskRuntimeConfig(
            output_dir=str(tmp_path),
            deadline_mode="challenge",
            challenge_stage_turns=1,
            challenge_deadline_ms_list=(1000.0, 500.0),
            show_expected=True,
        ),
        model_client=StaticModelClient([]),
        terminal=FakeTerminal(["1"]),
        task_event_sink=sink,
    )

    runtime.play(ZipZapZopTask(), max_turns=1)

    types = [event.type for event in sink.events]
    assert types.index("turn_finished") < types.index("stage_changed")
    finished = next(event for event in sink.events if event.type == "turn_finished")
    assert finished.payload["expected"] == "1"
    stage = next(event for event in sink.events if event.type == "stage_changed")
    assert stage.payload["old_deadline_ms"] == 1000.0
    assert stage.payload["new_deadline_ms"] == 500.0


def test_animal_naming_view_event_is_task_neutral_and_preserves_failure_reason(tmp_path) -> None:
    sink = RecordingTaskEventSink()
    runtime = InteractiveTaskRuntime(
        config=InteractiveTaskRuntimeConfig(
            output_dir=str(tmp_path), deadline_mode="soft"
        ),
        model_client=StaticModelClient([]),
        terminal=FakeTerminal(["dragon"]),
        task_event_sink=sink,
    )

    runtime.play(AnimalNamingTask(), max_turns=1)

    attempt = next(event for event in sink.events if event.type == "turn_attempt_started")
    assert attempt.payload["display_value"] is None
    assert attempt.payload["prompt"] == "Name one unused animal:"
    finished = next(event for event in sink.events if event.type == "turn_finished")
    assert finished.payload["failure_reason"] == "UNKNOWN_ANIMAL"


def test_interactive_runtime_alternates_human_and_llm_turns(tmp_path) -> None:
    clock = ManualClock()
    terminal = FakeTerminal(["1", "Zip"], clock=clock)
    model = StaticModelClient(["2", "Zap"])
    runtime = InteractiveTaskRuntime(
        config=InteractiveTaskRuntimeConfig(
            output_dir=str(tmp_path),
            deadline_mode="soft",
            deadline_ms=1000.0,
            max_tokens=8,
            temperature=0.2,
            human_first=True,
        ),
        model_client=model,
        terminal=terminal,
        now=clock.now,
    )

    result = runtime.play(ZipZapZopTask(start_number=1), max_turns=4)

    assert result.turn_count == 4
    assert result.human_turn_count == 2
    assert result.llm_turn_count == 2
    assert result.valid_count == 4
    assert result.invalid_count == 0
    assert result.deadline_mode == "soft"
    rows = _read_response_trace(Path(result.output_dir))
    assert [row["actor"] for row in rows] == ["human", "llm", "human", "llm"]
    assert [row["number"] for row in rows] == [1, 2, 3, 4]
    assert [row["response"] for row in rows] == ["1", "2", "Zip", "Zap"]
    assert set(rows[0]) == {
        "turn_id",
        "actor",
        "number",
        "prompt",
        "response",
        "expected",
        "is_valid",
        "latency_ms",
        "deadline_missed",
        "metadata",
    }
    assert rows[0]["metadata"]["deadline_mode"] == "soft"  # type: ignore[index]
    assert "Human at 1: 1" in model.requests[0][-1]["content"]
    assert (Path(result.output_dir) / "artifacts" / "turn" / "0004_turn_0003.json").exists()
    manifest = json.loads((Path(result.output_dir) / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 2
    assert manifest["status"] == "completed"
    assert manifest["human_first"] is True
    assert manifest["human_input"] == {"mode": "terminal"}
    assert rows[0]["metadata"]["human_input"]["mode"] == "terminal"  # type: ignore[index]


def test_animal_naming_runtime_alternates_with_shared_open_ended_state(
    tmp_path,
) -> None:
    clock = ManualClock()
    terminal = FakeTerminal(["lion", "tiger"], clock=clock)
    model = StaticModelClient(["elephant", "bear"])
    runtime = InteractiveTaskRuntime(
        config=InteractiveTaskRuntimeConfig(
            output_dir=str(tmp_path),
            deadline_mode="soft",
            human_first=True,
        ),
        model_client=model,
        terminal=terminal,
        now=clock.now,
    )

    result = runtime.play(AnimalNamingTask(), max_turns=4)

    rows = _read_response_trace(Path(result.output_dir))
    assert result.valid_count == 4
    assert [row["actor"] for row in rows] == ["human", "llm", "human", "llm"]
    assert [row["response"] for row in rows] == [
        "lion",
        "elephant",
        "tiger",
        "bear",
    ]
    assert all(row["number"] is None for row in rows)
    assert all(row["expected"] is None for row in rows)
    assert rows[0]["metadata"]["referee_metadata"] == {  # type: ignore[index]
        "normalized_animal": "lion"
    }
    assert "Forbidden animal names: lion" in model.requests[0][-1]["content"]
    assert "lion, elephant, tiger" in model.requests[1][-1]["content"]
    assert not any("expected=None" in output for output in terminal.outputs)
    assert not any("number=None" in output for output in terminal.outputs)


def test_animal_naming_runtime_rejects_repeats_from_either_actor(tmp_path) -> None:
    clock = ManualClock()
    terminal = FakeTerminal(["lion", "elephant"], clock=clock)
    runtime = InteractiveTaskRuntime(
        config=InteractiveTaskRuntimeConfig(
            output_dir=str(tmp_path),
            deadline_mode="soft",
            human_first=True,
        ),
        model_client=StaticModelClient(["elephant", "lion"]),
        terminal=terminal,
        now=clock.now,
    )

    result = runtime.play(AnimalNamingTask(), max_turns=4)

    rows = _read_response_trace(Path(result.output_dir))
    assert result.valid_count == 2
    assert result.invalid_count == 2
    assert [row["metadata"]["failure_reason"] for row in rows] == [  # type: ignore[index]
        "NONE",
        "NONE",
        "REPEATED_ANIMAL",
        "REPEATED_ANIMAL",
    ]
    assert all("expected" not in detail for detail in result.invalid_responses)
    assert all("number" not in detail for detail in result.invalid_responses)
    assert any(
        "MISS reason=REPEATED_ANIMAL actual=elephant" in output
        for output in terminal.outputs
    )


def test_animal_naming_hard_loss_uses_reason_instead_of_expected_none(
    tmp_path,
) -> None:
    clock = ManualClock()
    terminal = FakeTerminal(["dragon"], clock=clock)
    runtime = InteractiveTaskRuntime(
        config=InteractiveTaskRuntimeConfig(
            output_dir=str(tmp_path),
            deadline_mode="hard",
        ),
        model_client=StaticModelClient([]),
        terminal=terminal,
        now=clock.now,
    )

    result = runtime.play(AnimalNamingTask(), max_turns=5)

    assert result.stop_reason == "invalid_response_loss"
    assert any(
        "Invalid response: human answered dragon (UNKNOWN_ANIMAL)" in output
        for output in terminal.outputs
    )
    assert not any("expected None" in output for output in terminal.outputs)


def test_animal_naming_runtime_records_unknown_llm_response(tmp_path) -> None:
    terminal = FakeTerminal([])
    runtime = InteractiveTaskRuntime(
        config=InteractiveTaskRuntimeConfig(
            output_dir=str(tmp_path),
            deadline_mode="soft",
            human_first=False,
        ),
        model_client=StaticModelClient(["dragon"]),
        terminal=terminal,
    )

    result = runtime.play(AnimalNamingTask(), max_turns=1)

    row = _read_response_trace(Path(result.output_dir))[0]
    assert result.invalid_count == 1
    assert row["actor"] == "llm"
    assert row["response"] == "dragon"
    assert row["expected"] is None
    assert row["metadata"]["failure_reason"] == "UNKNOWN_ANIMAL"  # type: ignore[index]
    assert any(
        "MISS reason=UNKNOWN_ANIMAL actual=dragon" in output
        for output in terminal.outputs
    )


def test_animal_naming_expected_command_explains_open_ended_rule(tmp_path) -> None:
    terminal = FakeTerminal([":expected", ":hint", ":quit"])
    runtime = InteractiveTaskRuntime(
        config=InteractiveTaskRuntimeConfig(
            output_dir=str(tmp_path),
            deadline_mode="soft",
        ),
        model_client=StaticModelClient([]),
        terminal=terminal,
    )

    result = runtime.play(AnimalNamingTask(), max_turns=1)

    assert result.turn_count == 0
    assert any("no single expected answer" in output for output in terminal.outputs)
    assert any("91 unused animal names remain" in output for output in terminal.outputs)


def test_animal_naming_voice_trace_keeps_raw_parse_and_referee_metadata(
    tmp_path,
) -> None:
    clock = ManualClock()
    source = FakeHumanResponseSource(
        [HumanResponse(text="A lion.", latency_ms=20.0)],
        clock=clock,
        response_elapsed_s=0.02,
    )
    terminal = FakeTerminal([])
    runtime = InteractiveTaskRuntime(
        config=InteractiveTaskRuntimeConfig(
            output_dir=str(tmp_path),
            deadline_mode="soft",
        ),
        model_client=StaticModelClient([]),
        terminal=terminal,
        human_response_source=source,
        now=clock.now,
    )

    result = runtime.play(AnimalNamingTask(), max_turns=1)

    row = _read_response_trace(Path(result.output_dir))[0]
    human_input = row["metadata"]["human_input"]  # type: ignore[index]
    assert row["response"] == "lion"
    assert row["is_valid"] is True
    assert human_input["raw_transcript"] == "A lion."
    assert human_input["canonical_response"] == "lion"
    assert row["metadata"]["referee_metadata"] == {  # type: ignore[index]
        "normalized_animal": "lion"
    }
    assert "lion" in source.requests[0].speech_context.hotwords  # type: ignore[union-attr]


def test_interactive_runtime_supports_llm_first(tmp_path) -> None:
    clock = ManualClock()
    terminal = FakeTerminal(["2"], clock=clock)
    runtime = InteractiveTaskRuntime(
        config=InteractiveTaskRuntimeConfig(output_dir=str(tmp_path), deadline_mode="soft", human_first=False),
        model_client=StaticModelClient(["1"]),
        terminal=terminal,
        now=clock.now,
    )

    result = runtime.play(ZipZapZopTask(start_number=1), max_turns=2)

    rows = _read_response_trace(Path(result.output_dir))
    assert result.turn_count == 2
    assert [row["actor"] for row in rows] == ["llm", "human"]
    assert [row["response"] for row in rows] == ["1", "2"]


def test_interactive_commands_do_not_consume_turns(tmp_path) -> None:
    clock = ManualClock()
    terminal = FakeTerminal([":help", ":hint", ":expected", ":summary", "1"], clock=clock)
    runtime = InteractiveTaskRuntime(
        config=InteractiveTaskRuntimeConfig(output_dir=str(tmp_path), deadline_mode="soft", show_expected=True),
        model_client=StaticModelClient([]),
        terminal=terminal,
        now=clock.now,
    )

    result = runtime.play(ZipZapZopTask(start_number=1), max_turns=1)

    rows = _read_response_trace(Path(result.output_dir))
    assert result.turn_count == 1
    assert len(rows) == 1
    assert rows[0]["response"] == "1"
    assert any("Commands:" in output for output in terminal.outputs)
    assert any("not divisible by 3, 4, or 5" in output for output in terminal.outputs)
    assert any(output == "1" for output in terminal.outputs)
    assert any("Summary: 0 turns" in output for output in terminal.outputs)


def test_interactive_runtime_marks_human_deadline_miss_in_soft_mode(tmp_path) -> None:
    clock = ManualClock()
    terminal = FakeTerminal(["1"], clock=clock, prompt_elapsed_s=0.02)
    runtime = InteractiveTaskRuntime(
        config=InteractiveTaskRuntimeConfig(output_dir=str(tmp_path), deadline_mode="soft", deadline_ms=10.0),
        model_client=StaticModelClient([]),
        terminal=terminal,
        now=clock.now,
    )

    result = runtime.play(ZipZapZopTask(start_number=1), max_turns=1)

    rows = _read_response_trace(Path(result.output_dir))
    assert result.stop_reason == "completed"
    assert result.deadline_miss_count == 1
    assert len(result.deadline_misses) == 1
    assert rows[0]["deadline_missed"] is True


def test_interactive_runtime_records_invalid_human_and_llm_responses_in_soft_mode(tmp_path) -> None:
    clock = ManualClock()
    terminal = FakeTerminal(["wrong"], clock=clock)
    runtime = InteractiveTaskRuntime(
        config=InteractiveTaskRuntimeConfig(output_dir=str(tmp_path), deadline_mode="soft", human_first=True),
        model_client=StaticModelClient(["also wrong"]),
        terminal=terminal,
        now=clock.now,
    )

    result = runtime.play(ZipZapZopTask(start_number=1), max_turns=2)

    rows = _read_response_trace(Path(result.output_dir))
    assert result.stop_reason == "completed"
    assert result.valid_count == 0
    assert result.invalid_count == 2
    assert len(result.invalid_responses) == 2
    assert rows[0]["actor"] == "human"
    assert rows[0]["expected"] == "1"
    assert rows[0]["is_valid"] is False
    assert rows[1]["actor"] == "llm"
    assert rows[1]["expected"] == "2"
    assert rows[1]["is_valid"] is False


def test_interactive_runtime_help_and_quit_stop_without_consuming_turn(tmp_path) -> None:
    clock = ManualClock()
    terminal = FakeTerminal([":help", ":quit"], clock=clock)
    runtime = InteractiveTaskRuntime(
        config=InteractiveTaskRuntimeConfig(output_dir=str(tmp_path), deadline_mode="soft"),
        model_client=StaticModelClient([]),
        terminal=terminal,
        now=clock.now,
    )

    result = runtime.play(ZipZapZopTask(start_number=1), max_turns=5)

    run_dir = Path(result.output_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert result.turn_count == 0
    assert result.stop_reason == "user_quit"
    assert manifest["status"] == "user_quit"
    assert not (run_dir / "response_trace.jsonl").exists()
    assert any("Answer the shown prompt" in output for output in terminal.outputs)
    assert any("Stopping interactive session" in output for output in terminal.outputs)


def test_menu_mode_prompts_for_deadline_mode_and_hard_invalid_loses(tmp_path) -> None:
    clock = ManualClock()
    terminal = FakeTerminal(["2", "wrong"], clock=clock)
    runtime = InteractiveTaskRuntime(
        config=InteractiveTaskRuntimeConfig(output_dir=str(tmp_path), deadline_mode="menu"),
        model_client=StaticModelClient([]),
        terminal=terminal,
        now=clock.now,
    )

    result = runtime.play(ZipZapZopTask(start_number=1), max_turns=5)

    assert result.deadline_mode == "hard"
    assert result.stop_reason == "invalid_response_loss"
    assert result.loser == "human"
    assert result.winner == "llm"
    assert result.turn_count == 1
    assert any("Select deadline mode" in output for output in terminal.outputs)


def test_hard_mode_human_timeout_loses_immediately(tmp_path) -> None:
    clock = ManualClock()
    terminal = FakeTerminal([TimedPromptResult(text="", timed_out=True)], clock=clock)
    runtime = InteractiveTaskRuntime(
        config=InteractiveTaskRuntimeConfig(output_dir=str(tmp_path), deadline_mode="hard", deadline_ms=10.0),
        model_client=StaticModelClient([]),
        terminal=terminal,
        now=clock.now,
    )

    result = runtime.play(ZipZapZopTask(start_number=1), max_turns=5)

    rows = _read_response_trace(Path(result.output_dir))
    assert result.stop_reason == "deadline_loss"
    assert result.loser == "human"
    assert result.winner == "llm"
    assert result.turn_count == 1
    assert result.deadline_miss_count == 1
    assert rows[0]["deadline_missed"] is True
    assert terminal.timeout_prompts == [0.01]


def test_hard_mode_terminal_command_dispatches_before_wall_clock_scoring_and_reprompts(
    tmp_path,
) -> None:
    clock = ManualClock()
    terminal = FakeTerminal(
        [":help", "1"],
        clock=clock,
        prompt_elapsed_sequence=[0.02, 0.001],
    )
    runtime = InteractiveTaskRuntime(
        config=InteractiveTaskRuntimeConfig(
            output_dir=str(tmp_path),
            deadline_mode="hard",
            deadline_ms=10.0,
        ),
        model_client=StaticModelClient([]),
        terminal=terminal,
        now=clock.now,
    )

    result = runtime.play(ZipZapZopTask(start_number=1), max_turns=1)

    row = _read_response_trace(Path(result.output_dir))[0]
    assert result.turn_count == 1
    assert result.stop_reason == "completed"
    assert row["latency_ms"] == pytest.approx(1.0)
    assert row["deadline_missed"] is False
    assert terminal.timeout_prompts == [0.01, 0.01]
    assert any("Answer the shown prompt" in output for output in terminal.outputs)


def test_hard_mode_llm_invalid_response_loses_immediately(tmp_path) -> None:
    clock = ManualClock()
    terminal = FakeTerminal([], clock=clock)
    runtime = InteractiveTaskRuntime(
        config=InteractiveTaskRuntimeConfig(output_dir=str(tmp_path), deadline_mode="hard", human_first=False),
        model_client=StaticModelClient(["wrong"]),
        terminal=terminal,
        now=clock.now,
    )

    result = runtime.play(ZipZapZopTask(start_number=1), max_turns=5)

    assert result.stop_reason == "invalid_response_loss"
    assert result.loser == "llm"
    assert result.winner == "human"
    assert result.turn_count == 1
    assert result.invalid_count == 1


def test_hard_mode_llm_timeout_loses_immediately(tmp_path) -> None:
    clock = ManualClock()
    terminal = FakeTerminal([], clock=clock)
    model = StaticModelClient([], errors=[requests.Timeout("slow")])
    runtime = InteractiveTaskRuntime(
        config=InteractiveTaskRuntimeConfig(output_dir=str(tmp_path), deadline_mode="hard", deadline_ms=10.0, human_first=False),
        model_client=model,
        terminal=terminal,
        now=clock.now,
    )

    result = runtime.play(ZipZapZopTask(start_number=1), max_turns=5)

    rows = _read_response_trace(Path(result.output_dir))
    assert result.stop_reason == "deadline_loss"
    assert result.loser == "llm"
    assert result.winner == "human"
    assert result.deadline_miss_count == 1
    assert rows[0]["metadata"]["model_error"] == "slow"  # type: ignore[index]
    assert model.kwargs[0]["timeout_s"] == 0.01


def test_challenge_mode_advances_stage_without_resetting_state(tmp_path) -> None:
    clock = ManualClock()
    terminal = FakeTerminal(["1", "Zip"], clock=clock)
    model = StaticModelClient(["2", "Zap"])
    runtime = InteractiveTaskRuntime(
        config=InteractiveTaskRuntimeConfig(
            output_dir=str(tmp_path),
            deadline_mode="challenge",
            challenge_stage_turns=2,
            challenge_deadline_ms_list=(1000.0, 500.0),
        ),
        model_client=model,
        terminal=terminal,
        now=clock.now,
    )

    result = runtime.play(ZipZapZopTask(start_number=1), max_turns=4)

    rows = _read_response_trace(Path(result.output_dir))
    assert result.stop_reason == "completed"
    assert result.final_deadline_ms == 500.0
    assert [row["number"] for row in rows] == [1, 2, 3, 4]
    assert rows[0]["metadata"]["deadline_ms"] == 1000.0  # type: ignore[index]
    assert rows[2]["metadata"]["deadline_ms"] == 500.0  # type: ignore[index]
    assert any("Stage passed" in output for output in terminal.outputs)


def test_challenge_mode_invalid_response_loses(tmp_path) -> None:
    clock = ManualClock()
    terminal = FakeTerminal(["wrong"], clock=clock)
    runtime = InteractiveTaskRuntime(
        config=InteractiveTaskRuntimeConfig(output_dir=str(tmp_path), deadline_mode="challenge"),
        model_client=StaticModelClient([]),
        terminal=terminal,
        now=clock.now,
    )

    result = runtime.play(ZipZapZopTask(start_number=1), max_turns=5)

    assert result.stop_reason == "invalid_response_loss"
    assert result.loser == "human"
    assert result.winner == "llm"
    assert result.turn_count == 1


def test_voice_source_is_started_once_parses_task_response_and_records_provenance(tmp_path) -> None:
    clock = ManualClock()
    source_metadata: dict[str, object] = {
        "asr_latency_ms": 4.0,
        "total_latency_ms": 999.0,
        "deadline_mode": "malicious",
        "failure_reason": "malicious",
    }
    source = FakeHumanResponseSource(
        [HumanResponse(text="one", latency_ms=20.0, metadata=source_metadata)],
        provenance={"mode": "voice", "model": "tiny.en", "compute_type": "int8"},
        clock=clock,
        response_elapsed_s=0.02,
    )
    terminal = FakeTerminal([])
    runtime = InteractiveTaskRuntime(
        config=InteractiveTaskRuntimeConfig(output_dir=str(tmp_path), deadline_mode="soft", deadline_ms=1000.0),
        model_client=StaticModelClient([]),
        terminal=terminal,
        human_response_source=source,
        now=clock.now,
    )

    result = runtime.play(SpeechAwareZipZapZopTask(start_number=1), max_turns=1)

    rows = _read_response_trace(Path(result.output_dir))
    human_input = rows[0]["metadata"]["human_input"]  # type: ignore[index]
    manifest = json.loads((Path(result.output_dir) / "manifest.json").read_text(encoding="utf-8"))
    assert source.start_count == 1
    assert source.close_count == 1
    assert len(source.requests) == 1
    assert source.requests[0].turn_id == 0
    assert source.requests[0].timeout_s is None
    assert source.requests[0].speech_context == SpeechContext(
        initial_prompt="Say the game answer.", hotwords=("Zip", "Zap", "Zop")
    )
    assert rows[0]["response"] == "1"
    assert rows[0]["metadata"]["deadline_mode"] == "soft"  # type: ignore[index]
    assert rows[0]["metadata"]["failure_reason"] == "NONE"  # type: ignore[index]
    assert human_input == {
        "asr_latency_ms": 4.0,
        "canonical_response": "1",
        "deadline_mode": "malicious",
        "failure_reason": "malicious",
        "mode": "voice",
        "parse_reason": None,
        "parse_status": "ok",
        "raw_transcript": "one",
        "status": "ok",
        "total_latency_ms": 20.0,
    }
    assert source_metadata == {
        "asr_latency_ms": 4.0,
        "total_latency_ms": 999.0,
        "deadline_mode": "malicious",
        "failure_reason": "malicious",
    }
    assert manifest["schema_version"] == 2
    assert manifest["human_input"] == {"mode": "voice", "model": "tiny.en", "compute_type": "int8"}
    assert any("You >" in output for output in terminal.outputs)
    assert '    ASR > raw="one" parsed="1"' in terminal.outputs


def test_voice_unrecognized_transcript_is_visible_in_terminal_result(tmp_path) -> None:
    source = FakeHumanResponseSource([HumanResponse(text="two to three")])
    terminal = FakeTerminal([])
    runtime = InteractiveTaskRuntime(
        config=InteractiveTaskRuntimeConfig(output_dir=str(tmp_path), deadline_mode="soft"),
        model_client=StaticModelClient([]),
        terminal=terminal,
        human_response_source=source,
    )

    result = runtime.play(SpeechAwareZipZapZopTask(start_number=3), max_turns=1)

    assert result.invalid_count == 1
    assert '    ASR > raw="two to three" parsed=[unrecognized]' in terminal.outputs
    assert any(
        "MISS expected=Zip actual=[unrecognized]" in output
        for output in terminal.outputs
    )


def test_voice_reported_latency_cannot_hide_unaccounted_response_source_time(tmp_path) -> None:
    clock = ManualClock()
    source = FakeHumanResponseSource(
        [HumanResponse(text="one", latency_ms=20.0, metadata={"total_latency_ms": 20.0})],
        clock=clock,
        response_elapsed_s=0.5,
    )
    runtime = InteractiveTaskRuntime(
        config=InteractiveTaskRuntimeConfig(output_dir=str(tmp_path), deadline_mode="hard", deadline_ms=100.0),
        model_client=StaticModelClient([]),
        terminal=FakeTerminal([]),
        human_response_source=source,
        now=clock.now,
    )

    result = runtime.play(SpeechAwareZipZapZopTask(start_number=1), max_turns=1)

    row = _read_response_trace(Path(result.output_dir))[0]
    assert result.stop_reason == "deadline_loss"
    assert row["latency_ms"] == 500.0
    assert row["deadline_missed"] is True
    assert row["metadata"]["human_input"]["total_latency_ms"] == 500.0  # type: ignore[index]


def test_voice_explicit_audio_persistence_time_is_excluded_from_scoring(tmp_path) -> None:
    clock = ManualClock()
    source = FakeHumanResponseSource(
        [
            HumanResponse(
                text="one",
                latency_ms=20.0,
                metadata={
                    "total_latency_ms": 20.0,
                    "audio_artifact": "artifacts/turn/0001_turn_0000_human.wav",
                    "artifact_persistence_ms": 480.0,
                },
            )
        ],
        clock=clock,
        response_elapsed_s=0.5,
    )
    runtime = InteractiveTaskRuntime(
        config=InteractiveTaskRuntimeConfig(
            output_dir=str(tmp_path),
            deadline_mode="hard",
            deadline_ms=100.0,
        ),
        model_client=StaticModelClient([]),
        terminal=FakeTerminal([]),
        human_response_source=source,
        now=clock.now,
    )

    result = runtime.play(SpeechAwareZipZapZopTask(start_number=1), max_turns=1)

    row = _read_response_trace(Path(result.output_dir))[0]
    assert result.stop_reason == "completed"
    assert row["latency_ms"] == pytest.approx(20.0)
    assert row["deadline_missed"] is False
    assert row["metadata"]["human_input"]["total_latency_ms"] == pytest.approx(20.0)  # type: ignore[index]


def test_voice_prompt_time_uses_turn_origin_and_only_remaining_deadline(tmp_path) -> None:
    clock = ManualClock()
    source = FakeHumanResponseSource(
        [
            HumanResponse(
                text="one",
                latency_ms=60.0,
                metadata={
                    "last_voiced_offset_ms": 20.0,
                    "endpoint_detected_offset_ms": 30.0,
                    "asr_start_offset_ms": 30.0,
                    "asr_end_offset_ms": 60.0,
                    "total_latency_ms": 60.0,
                },
            )
        ]
    )
    terminal = FakeTerminal([], clock=clock, write_elapsed_s=0.04)
    runtime = InteractiveTaskRuntime(
        config=InteractiveTaskRuntimeConfig(output_dir=str(tmp_path), deadline_mode="hard", deadline_ms=100.0),
        model_client=StaticModelClient([]),
        terminal=terminal,
        human_response_source=source,
        now=clock.now,
    )

    result = runtime.play(SpeechAwareZipZapZopTask(start_number=1), max_turns=1)

    row = _read_response_trace(Path(result.output_dir))[0]
    human_input = row["metadata"]["human_input"]  # type: ignore[index]
    assert source.requests[0].timeout_s == pytest.approx(0.06)
    assert row["latency_ms"] == pytest.approx(100.0)
    assert row["deadline_missed"] is True
    assert result.stop_reason == "deadline_loss"
    assert human_input["last_voiced_offset_ms"] == pytest.approx(60.0)
    assert human_input["asr_end_offset_ms"] == pytest.approx(100.0)
    assert human_input["total_latency_ms"] == pytest.approx(100.0)


def test_voice_prompt_that_consumes_deadline_skips_capture(tmp_path) -> None:
    clock = ManualClock()
    source = FakeHumanResponseSource([])
    terminal = FakeTerminal([], clock=clock, write_elapsed_s=0.02)
    runtime = InteractiveTaskRuntime(
        config=InteractiveTaskRuntimeConfig(output_dir=str(tmp_path), deadline_mode="hard", deadline_ms=10.0),
        model_client=StaticModelClient([]),
        terminal=terminal,
        human_response_source=source,
        now=clock.now,
    )

    result = runtime.play(SpeechAwareZipZapZopTask(start_number=1), max_turns=1)

    assert source.requests == []
    assert result.stop_reason == "deadline_loss"
    assert result.deadline_miss_count == 1


@pytest.mark.parametrize("status", ["no_speech", "empty_transcript"])
def test_voice_empty_response_statuses_are_invalid_without_forcing_soft_deadline(
    tmp_path,
    status: HumanResponseStatus,
) -> None:
    source = FakeHumanResponseSource([HumanResponse(text="", status=status)])
    terminal = FakeTerminal([])
    runtime = InteractiveTaskRuntime(
        config=InteractiveTaskRuntimeConfig(output_dir=str(tmp_path), deadline_mode="soft", deadline_ms=1000.0),
        model_client=StaticModelClient([]),
        terminal=terminal,
        human_response_source=source,
    )

    result = runtime.play(SpeechAwareZipZapZopTask(start_number=1), max_turns=1)

    row = _read_response_trace(Path(result.output_dir))[0]
    assert result.invalid_count == 1
    assert result.deadline_miss_count == 0
    assert row["response"] == ""
    assert row["deadline_missed"] is False
    assert row["metadata"]["human_input"]["status"] == status  # type: ignore[index]
    assert row["metadata"]["human_input"]["parse_status"] == status  # type: ignore[index]
    assert f'    ASR > raw="" parsed=[{status}]' in terminal.outputs
    assert any(f"actual=[{status}]" in output for output in terminal.outputs)


def test_voice_quality_rejected_transcript_is_audited_but_not_parsed(tmp_path) -> None:
    raw_text = ", ".join(["Zap"] * 112)
    source = FakeHumanResponseSource(
        [
            HumanResponse(
                text=raw_text,
                status="rejected_transcript",
                metadata={
                    "transcript_rejection_reasons": [
                        "excessive_token_repetition",
                        "excessive_compression_ratio",
                    ],
                    "asr": {
                        "quality_gate": {
                            "accepted": False,
                            "max_consecutive_token_repetitions": 112,
                            "max_segment_compression_ratio": 29.421,
                        }
                    },
                },
            )
        ]
    )
    terminal = FakeTerminal([])
    runtime = InteractiveTaskRuntime(
        config=InteractiveTaskRuntimeConfig(
            output_dir=str(tmp_path),
            deadline_mode="soft",
            deadline_ms=1000.0,
        ),
        model_client=StaticModelClient([]),
        terminal=terminal,
        human_response_source=source,
    )

    result = runtime.play(SpeechAwareZipZapZopTask(start_number=1), max_turns=1)

    row = _read_response_trace(Path(result.output_dir))[0]
    metadata = row["metadata"]["human_input"]  # type: ignore[index]
    assert result.invalid_count == 1
    assert row["response"] == ""
    assert metadata["raw_transcript"] == raw_text  # type: ignore[index]
    assert metadata["parse_status"] == "rejected_transcript"  # type: ignore[index]
    assert metadata["parse_reason"] == (  # type: ignore[index]
        "excessive_token_repetition,excessive_compression_ratio"
    )
    assert metadata["asr"]["quality_gate"] == {  # type: ignore[index]
        "accepted": False,
        "max_consecutive_token_repetitions": 112,
        "max_segment_compression_ratio": 29.421,
    }


def test_voice_deadline_expiration_forces_hard_deadline_loss(tmp_path) -> None:
    source = FakeHumanResponseSource(
        [HumanResponse(text="", status="no_speech", deadline_expired=True)]
    )
    runtime = InteractiveTaskRuntime(
        config=InteractiveTaskRuntimeConfig(output_dir=str(tmp_path), deadline_mode="hard", deadline_ms=10.0),
        model_client=StaticModelClient([]),
        terminal=FakeTerminal([]),
        human_response_source=source,
    )

    result = runtime.play(SpeechAwareZipZapZopTask(start_number=1), max_turns=2)

    row = _read_response_trace(Path(result.output_dir))[0]
    assert source.requests[0].timeout_s == pytest.approx(0.01, abs=0.001)
    assert result.stop_reason == "deadline_loss"
    assert result.loser == "human"
    assert result.deadline_miss_count == 1
    assert row["latency_ms"] == 10.0
    assert row["deadline_missed"] is True


def test_voice_source_runtime_failure_does_not_create_a_player_turn(tmp_path) -> None:
    source = FakeHumanResponseSource([RuntimeError("capture failed")])
    runtime = InteractiveTaskRuntime(
        config=InteractiveTaskRuntimeConfig(output_dir=str(tmp_path), deadline_mode="soft"),
        model_client=StaticModelClient([]),
        terminal=FakeTerminal([]),
        human_response_source=source,
    )

    with pytest.raises(RuntimeError, match="capture failed"):
        runtime.play(SpeechAwareZipZapZopTask(), max_turns=1)

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((tmp_path / "run_summary.json").read_text(encoding="utf-8"))
    assert source.start_count == 1
    assert source.close_count == 1
    assert manifest["status"] == "error"
    assert manifest["turn_count"] == 0
    assert summary["stop_reason"] == "error"
    assert not (tmp_path / "response_trace.jsonl").exists()
    assert not any((tmp_path / "artifacts" / "turn").iterdir())


def test_failure_recording_error_does_not_replace_voice_source_error(tmp_path) -> None:
    source = FakeHumanResponseSource([RuntimeError("capture failed")])
    runtime = InteractiveTaskRuntime(
        config=InteractiveTaskRuntimeConfig(output_dir=str(tmp_path), deadline_mode="soft"),
        model_client=StaticModelClient([]),
        terminal=FakeTerminal([]),
        human_response_source=source,
    )

    with (
        patch.object(runtime, "_write_run_summary", side_effect=OSError("summary disk failure")),
        pytest.raises(RuntimeError, match="capture failed"),
    ):
        runtime.play(SpeechAwareZipZapZopTask(), max_turns=1)

    assert source.close_count == 1


def test_turn_persistence_failure_does_not_commit_stats_or_leave_artifact(tmp_path) -> None:
    source = FakeHumanResponseSource([HumanResponse(text="one")])
    runtime = InteractiveTaskRuntime(
        config=InteractiveTaskRuntimeConfig(output_dir=str(tmp_path), deadline_mode="soft"),
        model_client=StaticModelClient([]),
        terminal=FakeTerminal([]),
        human_response_source=source,
    )

    with (
        patch.object(runtime, "_append_response_trace", side_effect=OSError("trace disk full")),
        pytest.raises(OSError, match="trace disk full"),
    ):
        runtime.play(SpeechAwareZipZapZopTask(), max_turns=1)

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((tmp_path / "run_summary.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "error"
    assert manifest["turn_count"] == 0
    assert summary["turn_count"] == 0
    assert not (tmp_path / "response_trace.jsonl").exists()
    assert not any((tmp_path / "artifacts" / "turn").iterdir())


def test_trace_append_rolls_back_a_row_written_before_the_failure(tmp_path) -> None:
    source = FakeHumanResponseSource([HumanResponse(text="one")])
    runtime = InteractiveTaskRuntime(
        config=InteractiveTaskRuntimeConfig(output_dir=str(tmp_path), deadline_mode="soft"),
        model_client=StaticModelClient([]),
        terminal=FakeTerminal([]),
        human_response_source=source,
    )

    def write_then_fail(path: Path, row: bytes) -> None:
        with path.open("ab") as handle:
            handle.write(row)
            handle.flush()
        raise OSError("trace close failed")

    with (
        patch.object(runtime, "_write_response_trace_row", side_effect=write_then_fail),
        pytest.raises(OSError, match="trace close failed"),
    ):
        runtime.play(SpeechAwareZipZapZopTask(), max_turns=1)

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((tmp_path / "run_summary.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "error"
    assert manifest["turn_count"] == 0
    assert summary["turn_count"] == 0
    assert not (tmp_path / "response_trace.jsonl").exists()
    assert not any((tmp_path / "artifacts" / "turn").iterdir())


def test_trace_rollback_preserves_rows_committed_before_a_later_failure(tmp_path) -> None:
    source = FakeHumanResponseSource([HumanResponse(text="one")])
    runtime = InteractiveTaskRuntime(
        config=InteractiveTaskRuntimeConfig(output_dir=str(tmp_path), deadline_mode="soft"),
        model_client=StaticModelClient(["2"]),
        terminal=FakeTerminal([]),
        human_response_source=source,
    )
    original_write = runtime._write_response_trace_row
    write_count = 0

    def fail_after_second_write(path: Path, row: bytes) -> None:
        nonlocal write_count
        write_count += 1
        if write_count == 1:
            original_write(path, row)
            return
        with path.open("ab") as handle:
            handle.write(row)
            handle.flush()
        raise OSError("second trace close failed")

    with (
        patch.object(
            runtime,
            "_write_response_trace_row",
            side_effect=fail_after_second_write,
        ),
        pytest.raises(OSError, match="second trace close failed"),
    ):
        runtime.play(SpeechAwareZipZapZopTask(), max_turns=2)

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((tmp_path / "run_summary.json").read_text(encoding="utf-8"))
    rows = _read_response_trace(tmp_path)
    artifacts = sorted((tmp_path / "artifacts" / "turn").iterdir())
    assert manifest["status"] == "error"
    assert manifest["turn_count"] == 1
    assert summary["turn_count"] == 1
    assert [row["turn_id"] for row in rows] == [0]
    assert [path.name for path in artifacts] == ["0001_turn_0000.json"]


def test_voice_source_startup_failure_writes_v2_manifest_and_closes(tmp_path) -> None:
    source = FakeHumanResponseSource(
        [],
        provenance={"mode": "voice", "model": "tiny.en", "device": "cpu"},
        start_error=RuntimeError("warmup failed"),
        close_error=RuntimeError("close failed"),
    )
    runtime = InteractiveTaskRuntime(
        config=InteractiveTaskRuntimeConfig(output_dir=str(tmp_path), deadline_mode="soft"),
        model_client=StaticModelClient([]),
        terminal=FakeTerminal([]),
        human_response_source=source,
    )

    with pytest.raises(RuntimeError, match="warmup failed"):
        runtime.play(SpeechAwareZipZapZopTask(), max_turns=1)

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert source.start_count == 1
    assert source.close_count == 1
    assert manifest["schema_version"] == 2
    assert manifest["status"] == "startup_error"
    assert manifest["startup_error"] == {"type": "RuntimeError", "message": "warmup failed"}
    assert manifest["human_input"] == {"mode": "voice", "model": "tiny.en", "device": "cpu"}
    assert not (tmp_path / "response_trace.jsonl").exists()
    assert not (tmp_path / "run_summary.json").exists()


def test_source_close_failure_changes_completed_run_to_error(tmp_path) -> None:
    source = FakeHumanResponseSource(
        [HumanResponse(text="one")],
        close_error=RuntimeError("microphone close failed"),
    )
    runtime = InteractiveTaskRuntime(
        config=InteractiveTaskRuntimeConfig(output_dir=str(tmp_path), deadline_mode="soft"),
        model_client=StaticModelClient([]),
        terminal=FakeTerminal([]),
        human_response_source=source,
    )

    with pytest.raises(RuntimeError, match="microphone close failed"):
        runtime.play(SpeechAwareZipZapZopTask(), max_turns=1)

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((tmp_path / "run_summary.json").read_text(encoding="utf-8"))
    assert source.close_count == 1
    assert manifest["status"] == "error"
    assert manifest["stop_reason"] == "error"
    assert summary["stop_reason"] == "error"


def test_voice_mode_rejects_non_speech_task_before_source_start(tmp_path) -> None:
    source = FakeHumanResponseSource([])
    runtime = InteractiveTaskRuntime(
        config=InteractiveTaskRuntimeConfig(output_dir=str(tmp_path), deadline_mode="soft"),
        model_client=StaticModelClient([]),
        terminal=FakeTerminal([]),
        human_response_source=source,
    )

    with pytest.raises(TypeError, match="does not support voice input"):
        runtime.play(NonSpeechInteractiveTask(), max_turns=1)  # type: ignore[arg-type]

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert source.start_count == 0
    assert source.close_count == 1
    assert manifest["status"] == "startup_error"
    assert manifest["startup_error"]["type"] == "TypeError"
    assert not (tmp_path / "response_trace.jsonl").exists()


def test_keyboard_interrupt_is_recorded_then_reraised_and_source_is_closed(tmp_path) -> None:
    source = FakeHumanResponseSource([KeyboardInterrupt()])
    runtime = InteractiveTaskRuntime(
        config=InteractiveTaskRuntimeConfig(output_dir=str(tmp_path), deadline_mode="soft"),
        model_client=StaticModelClient([]),
        terminal=FakeTerminal([]),
        human_response_source=source,
    )

    with pytest.raises(KeyboardInterrupt):
        runtime.play(SpeechAwareZipZapZopTask(), max_turns=1)

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((tmp_path / "run_summary.json").read_text(encoding="utf-8"))
    assert source.start_count == 1
    assert source.close_count == 1
    assert manifest["status"] == "user_interrupt"
    assert manifest["stop_reason"] == "user_interrupt"
    assert manifest["turn_count"] == 0
    assert summary["stop_reason"] == "user_interrupt"
    assert not (tmp_path / "response_trace.jsonl").exists()


def test_run_directory_failure_still_closes_human_response_source(tmp_path) -> None:
    source = FakeHumanResponseSource([])
    runtime = InteractiveTaskRuntime(
        config=InteractiveTaskRuntimeConfig(
            output_dir=str(tmp_path / "unwritable"),
            deadline_mode="soft",
        ),
        model_client=StaticModelClient([]),
        terminal=FakeTerminal([]),
        human_response_source=source,
    )

    with (
        patch("pathlib.Path.mkdir", side_effect=PermissionError("cannot create run directory")),
        pytest.raises(PermissionError, match="cannot create run directory"),
    ):
        runtime.play(SpeechAwareZipZapZopTask(), max_turns=1)

    assert source.start_count == 0
    assert source.close_count == 1
