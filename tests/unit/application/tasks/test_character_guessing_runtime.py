from __future__ import annotations

import json
from pathlib import Path

from streammuse.application.tasks import (
    InteractiveTaskRuntime,
    InteractiveTaskRuntimeConfig,
    TimedPromptResult,
)
from streammuse.domain.tasks import (
    CharacterGuessingTask,
    ChatModelResponse,
    TaskViewEvent,
    WORLD_HISTORICAL_FIGURES_EN,
)


class ScriptedModel:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.requests: list[list[dict[str, str]]] = []

    def generate(
        self,
        messages: list[dict[str, str]],
        **kwargs: object,
    ) -> ChatModelResponse:
        self.requests.append(messages)
        return ChatModelResponse(
            text=self.responses.pop(0),
            latency_ms=5.0,
            prompt_tokens=20,
            completion_tokens=8,
        )


class ScriptedTerminal:
    def __init__(self, inputs: list[object]) -> None:
        self.inputs = list(inputs)
        self.outputs: list[str] = []

    def write(self, text: str) -> None:
        self.outputs.append(text)

    def prompt(self, text: str) -> str:
        self.outputs.append(f"PROMPT {text}")
        value = self.inputs.pop(0)
        if isinstance(value, TimedPromptResult):
            return value.text
        return str(value)

    def prompt_with_timeout(
        self,
        text: str,
        timeout_s: float,
    ) -> TimedPromptResult:
        self.outputs.append(f"TIMED_PROMPT {text}")
        value = self.inputs.pop(0)
        if isinstance(value, TimedPromptResult):
            return value
        return TimedPromptResult(str(value), timed_out=False)


class RecordingSink:
    def __init__(self) -> None:
        self.events: list[TaskViewEvent] = []

    def emit(self, event: TaskViewEvent) -> None:
        self.events.append(event)

    def close(self) -> None:
        return None


def game(*, max_rounds: int = 20) -> CharacterGuessingTask:
    return CharacterGuessingTask(
        scope=WORLD_HISTORICAL_FIGURES_EN,
        max_rounds=max_rounds,
    )


def rows(path: str) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in (
            Path(path) / "response_trace.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    ]


def test_complete_character_game_stops_early_and_records_round_metrics(
    tmp_path,
) -> None:
    terminal = ScriptedTerminal(["", "yes", "no", "yes"])
    model = ScriptedModel(
        [
            "ASK: Was this person born before 1900?",
            "GUESS: Newton",
            "GUESS: Einstein",
        ]
    )
    sink = RecordingSink()
    runtime = InteractiveTaskRuntime(
        config=InteractiveTaskRuntimeConfig(
            output_dir=str(tmp_path),
            deadline_mode="soft",
            max_tokens=64,
            human_first=True,
        ),
        model_client=model,
        terminal=terminal,
        task_event_sink=sink,
        task_event_session_id="character-session",
    )

    result = runtime.play(game(), max_turns=120)
    trace = rows(result.output_dir)

    assert result.stop_reason == "character_guessed"
    assert result.winner == "llm"
    assert result.loser == "human"
    assert result.turn_count == 6
    assert result.logical_round_count == 3
    assert result.question_count == 1
    assert result.guess_count == 2
    assert result.rejected_guess_count == 1
    assert result.successful_guess is True
    assert result.successful_character_id == "albert_einstein"
    assert result.successful_guess_round == 3
    assert [item["actor"] for item in trace] == [
        "llm",
        "human",
        "llm",
        "human",
        "llm",
        "human",
    ]
    assert trace[0]["metadata"]["phase_before"] == "awaiting_llm"
    assert trace[0]["metadata"]["phase_after"] == "awaiting_human"
    assert trace[1]["metadata"]["round_completed"] is True
    assert not any("GUESS:" in line for line in terminal.outputs)
    assert any(
        "LLM > Is your person Albert Einstein?" in line
        for line in terminal.outputs
    )

    event_types = [event.type for event in sink.events]
    assert event_types[0] == "session_setup"
    assert "session_config" in event_types
    assert event_types.count("task_progress") == 7
    assert event_types[-1] == "session_finished"
    setup = sink.events[0].payload
    assert setup["public_state"]["scope_size"] == 44
    assert "target" not in json.dumps(setup).lower()

    summary = json.loads(
        (Path(result.output_dir) / "run_summary.json").read_text()
    )
    manifest = json.loads(
        (Path(result.output_dir) / "manifest.json").read_text()
    )
    assert summary["logical_round_count"] == 3
    assert manifest["successful_character_id"] == "albert_einstein"
    assert manifest["task_public_state"]["max_rounds"] == 20
    assert "target" not in json.dumps(manifest).lower()


def test_invalid_attempts_retry_the_same_actor_without_advancing_round(
    tmp_path,
) -> None:
    terminal = ScriptedTerminal(["", "maybe", "yes", "yes"])
    runtime = InteractiveTaskRuntime(
        config=InteractiveTaskRuntimeConfig(
            output_dir=str(tmp_path),
            deadline_mode="soft",
        ),
        model_client=ScriptedModel(
            [
                "Here is my reasoning",
                "ASK: Is this person a scientist?",
                "GUESS: Marie Curie",
            ]
        ),
        terminal=terminal,
    )

    result = runtime.play(game(), max_turns=30)
    trace = rows(result.output_dir)

    assert [item["actor"] for item in trace] == [
        "llm",
        "llm",
        "human",
        "human",
        "llm",
        "human",
    ]
    assert [item["metadata"]["logical_round"] for item in trace] == [
        1,
        1,
        1,
        1,
        2,
        2,
    ]
    assert result.invalid_count == 2
    assert result.logical_round_count == 2
    assert result.stop_reason == "character_guessed"


def test_three_soft_invalid_attempts_trigger_bounded_retry_stop(
    tmp_path,
) -> None:
    runtime = InteractiveTaskRuntime(
        config=InteractiveTaskRuntimeConfig(
            output_dir=str(tmp_path),
            deadline_mode="soft",
        ),
        model_client=ScriptedModel(["bad", "still bad", "also bad"]),
        terminal=ScriptedTerminal([""]),
    )

    result = runtime.play(game(), max_turns=120)

    assert result.turn_count == 3
    assert result.stop_reason == "retry_limit_reached"
    assert result.winner == "human"
    assert result.loser == "llm"
    assert result.logical_round_count == 0


def test_hard_deadline_loss_wins_over_confirmed_guess(tmp_path) -> None:
    sink = RecordingSink()
    runtime = InteractiveTaskRuntime(
        config=InteractiveTaskRuntimeConfig(
            output_dir=str(tmp_path),
            deadline_mode="hard",
            deadline_ms=1000.0,
        ),
        model_client=ScriptedModel(["GUESS: Marie Curie"]),
        terminal=ScriptedTerminal(
            ["", TimedPromptResult("yes", timed_out=True)]
        ),
        task_event_sink=sink,
    )

    result = runtime.play(game(), max_turns=120)

    assert result.stop_reason == "deadline_loss"
    assert result.winner == "llm"
    assert result.loser == "human"
    assert result.successful_guess is False
    assert result.logical_round_count == 1
    progress = [
        event.payload
        for event in sink.events
        if event.type == "task_progress"
    ]
    assert progress[-1]["stop_reason"] == "deadline_loss"
    assert progress[-1]["winner"] == "llm"
    assert progress[-1]["public_state"]["stop_reason"] == "deadline_loss"
    assert progress[-1]["public_state"]["successful_character_id"] is None


def test_round_limit_finishes_only_after_the_human_answer(tmp_path) -> None:
    sink = RecordingSink()
    runtime = InteractiveTaskRuntime(
        config=InteractiveTaskRuntimeConfig(
            output_dir=str(tmp_path),
            deadline_mode="soft",
        ),
        model_client=ScriptedModel(
            ["ASK: Is this person primarily known for science?"]
        ),
        terminal=ScriptedTerminal(["", "no"]),
        task_event_sink=sink,
    )

    result = runtime.play(game(max_rounds=1), max_turns=6)

    assert result.stop_reason == "round_limit_reached"
    assert result.winner == "human"
    assert result.turn_count == 2
    progress = [
        event.payload
        for event in sink.events
        if event.type == "task_progress"
    ]
    assert progress[1]["is_complete"] is False
    assert progress[-1]["is_complete"] is True


def test_challenge_stage_advances_on_round_not_llm_action(tmp_path) -> None:
    sink = RecordingSink()
    runtime = InteractiveTaskRuntime(
        config=InteractiveTaskRuntimeConfig(
            output_dir=str(tmp_path),
            deadline_mode="challenge",
            challenge_stage_turns=1,
            challenge_deadline_ms_list=(1000.0, 500.0),
        ),
        model_client=ScriptedModel(
            ["ASK: Is this person primarily known for art?"]
        ),
        terminal=ScriptedTerminal(["", "no"]),
        task_event_sink=sink,
    )

    result = runtime.play(game(max_rounds=1), max_turns=6)

    stage_events = [
        event for event in sink.events if event.type == "stage_changed"
    ]
    assert result.final_deadline_ms == 500.0
    assert len(stage_events) == 1
    assert stage_events[0].payload["new_deadline_ms"] == 500.0
