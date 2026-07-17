from __future__ import annotations

import json
from pathlib import Path

from streammuse.application.tasks import TaskRuntime, TaskRuntimeConfig
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


class FakeClock:
    def __init__(self) -> None:
        self.now_s = 0.0
        self.sleeps: list[float] = []

    def now(self) -> float:
        return self.now_s

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now_s += seconds


def _read_trace(trace_dir: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in (trace_dir / "trace.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _read_response_trace(trace_dir: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in (trace_dir / "response_trace.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_offline_benchmark_records_one_trace_event_per_turn(tmp_path) -> None:
    task = ZipZapZopTask(start_number=1, history_limit=0)  # memoryless, matches run CLI default
    runtime = TaskRuntime(
        config=TaskRuntimeConfig(
            runner_kind="offline_benchmark",
            output_dir=str(tmp_path),
        ),
        model_client=StaticModelClient(["1", "2", "Zip"]),
    )

    result = runtime.run(task, max_turns=3)

    assert result.runner_kind == "offline_benchmark"
    assert result.turn_count == 3
    assert result.valid_count == 3
    trace = _read_trace(Path(result.output_dir))
    assert [row["stage"] for row in trace] == ["task_turn", "task_turn", "task_turn"]
    assert trace[0]["summary"]["turn_id"] == 0
    assert trace[0]["summary"]["deadline_missed"] is False

    response_trace = _read_response_trace(Path(result.output_dir))
    assert len(response_trace) == 3
    assert all(set(row) == {"turn_id", "prompt", "response"} for row in response_trace)
    assert response_trace[0]["turn_id"] == 0
    assert response_trace[0]["prompt"][-1]["content"] == "Current number: 1\nAnswer:"
    assert response_trace[0]["response"] == "1"
    assert response_trace[2]["turn_id"] == 2
    assert response_trace[2]["prompt"][-1]["content"] == "Current number: 3\nAnswer:"
    assert response_trace[2]["response"] == "Zip"

    manifest_path = Path(result.output_dir) / "manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["response_trace_path"] == "response_trace.jsonl"


def test_offline_benchmark_includes_recent_history_when_enabled(tmp_path) -> None:
    task = ZipZapZopTask(start_number=1, history_limit=2)
    runtime = TaskRuntime(
        config=TaskRuntimeConfig(
            runner_kind="offline_benchmark",
            output_dir=str(tmp_path),
        ),
        model_client=StaticModelClient(["1", "2", "Zip", "Zap"]),
    )

    result = runtime.run(task, max_turns=4)

    response_trace = _read_response_trace(Path(result.output_dir))
    # Turn 0 has no history yet.
    assert response_trace[0]["prompt"][-1]["content"] == "Current number: 1\nAnswer:"
    # Turn 3 (number 4) sees the last 2 turns (numbers 2 and 3), not turn 1.
    prompt = response_trace[3]["prompt"][-1]["content"]
    assert prompt.startswith("Recent turns:")
    assert "Assistant at 2: 2" in prompt
    assert "Assistant at 3: Zip" in prompt
    assert "Assistant at 1: 1" not in prompt  # outside the 2-turn window
    assert prompt.endswith("Current number: 4\nAnswer:")


def test_realtime_runner_marks_deadline_miss_with_fake_clock(tmp_path) -> None:
    clock = FakeClock()
    task = ZipZapZopTask(start_number=1)
    runtime = TaskRuntime(
        config=TaskRuntimeConfig(
            runner_kind="realtime_loop",
            output_dir=str(tmp_path),
            tick_rate_hz=1.0,
            deadline_ms=10.0,
        ),
        model_client=StaticModelClient(["1"], latency_ms=25.0),
        now=clock.now,
        sleep=clock.sleep,
    )

    result = runtime.run(task, max_turns=1)

    assert result.deadline_miss_count == 1
    trace = _read_trace(Path(result.output_dir))
    assert trace[0]["summary"]["deadline_missed"] is True
    assert trace[0]["summary"]["runner_kind"] == "realtime_loop"


def test_runtime_can_print_live_turn_output(tmp_path, capsys) -> None:
    task = ZipZapZopTask(start_number=1)
    runtime = TaskRuntime(
        config=TaskRuntimeConfig(
            runner_kind="offline_benchmark",
            output_dir=str(tmp_path),
            live_output=True,
        ),
        model_client=StaticModelClient(["1", "bad"]),
    )

    runtime.run(task, max_turns=2)

    output = capsys.readouterr().out
    assert "[turn 000] OK response='1' expected='1'" in output
    assert "[turn 001] BAD response='bad' expected='2' reason=EXPECTED_MISMATCH" in output
    assert "latency_ms=25.0" in output
