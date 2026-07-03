from __future__ import annotations

import json
from pathlib import Path

from streammuse.application.tasks import InteractiveTaskRuntime, InteractiveTaskRuntimeConfig
from streammuse.domain.tasks import ChatModelResponse, ZipZapZopTask


class StaticModelClient:
    def __init__(self, responses: list[str], *, latency_ms: float = 25.0) -> None:
        self.responses = list(responses)
        self.latency_ms = latency_ms
        self.requests: list[list[dict[str, str]]] = []

    def generate(self, messages: list[dict[str, str]], **_: object) -> ChatModelResponse:
        self.requests.append(messages)
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
    def __init__(self, inputs: list[str], *, clock: ManualClock | None = None, prompt_elapsed_s: float = 0.0) -> None:
        self.inputs = list(inputs)
        self.outputs: list[str] = []
        self.clock = clock
        self.prompt_elapsed_s = prompt_elapsed_s

    def write(self, text: str) -> None:
        self.outputs.append(text)

    def prompt(self, text: str) -> str:
        self.outputs.append(f"PROMPT {text}")
        if self.clock is not None:
            self.clock.advance(self.prompt_elapsed_s)
        assert self.inputs, "test terminal input exhausted"
        return self.inputs.pop(0)


def _read_response_trace(trace_dir: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in (trace_dir / "response_trace.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_interactive_runtime_alternates_human_and_llm_turns(tmp_path) -> None:
    clock = ManualClock()
    terminal = FakeTerminal(["1", "Zip"], clock=clock)
    model = StaticModelClient(["2", "Zap"])
    runtime = InteractiveTaskRuntime(
        config=InteractiveTaskRuntimeConfig(
            output_dir=str(tmp_path),
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
    assert "Human at 1: 1" in model.requests[0][-1]["content"]
    assert (Path(result.output_dir) / "artifacts" / "turn" / "0004_turn_0003.json").exists()
    manifest = json.loads((Path(result.output_dir) / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert manifest["human_first"] is True


def test_interactive_runtime_supports_llm_first(tmp_path) -> None:
    clock = ManualClock()
    terminal = FakeTerminal(["2"], clock=clock)
    runtime = InteractiveTaskRuntime(
        config=InteractiveTaskRuntimeConfig(output_dir=str(tmp_path), human_first=False),
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
        config=InteractiveTaskRuntimeConfig(output_dir=str(tmp_path), show_expected=True),
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


def test_interactive_runtime_marks_human_deadline_miss(tmp_path) -> None:
    clock = ManualClock()
    terminal = FakeTerminal(["1"], clock=clock, prompt_elapsed_s=0.02)
    runtime = InteractiveTaskRuntime(
        config=InteractiveTaskRuntimeConfig(output_dir=str(tmp_path), deadline_ms=10.0),
        model_client=StaticModelClient([]),
        terminal=terminal,
        now=clock.now,
    )

    result = runtime.play(ZipZapZopTask(start_number=1), max_turns=1)

    rows = _read_response_trace(Path(result.output_dir))
    assert result.deadline_miss_count == 1
    assert rows[0]["deadline_missed"] is True


def test_interactive_runtime_records_invalid_human_and_llm_responses(tmp_path) -> None:
    clock = ManualClock()
    terminal = FakeTerminal(["wrong"], clock=clock)
    runtime = InteractiveTaskRuntime(
        config=InteractiveTaskRuntimeConfig(output_dir=str(tmp_path), human_first=True),
        model_client=StaticModelClient(["also wrong"]),
        terminal=terminal,
        now=clock.now,
    )

    result = runtime.play(ZipZapZopTask(start_number=1), max_turns=2)

    rows = _read_response_trace(Path(result.output_dir))
    assert result.valid_count == 0
    assert result.invalid_count == 2
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
        config=InteractiveTaskRuntimeConfig(output_dir=str(tmp_path)),
        model_client=StaticModelClient([]),
        terminal=terminal,
        now=clock.now,
    )

    result = runtime.play(ZipZapZopTask(start_number=1), max_turns=5)

    run_dir = Path(result.output_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert result.turn_count == 0
    assert manifest["status"] == "stopped"
    assert not (run_dir / "response_trace.jsonl").exists()
    assert any("Answer the shown prompt" in output for output in terminal.outputs)
    assert any("Stopping interactive session" in output for output in terminal.outputs)
