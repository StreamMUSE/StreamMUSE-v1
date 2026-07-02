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


def test_offline_benchmark_records_one_trace_event_per_turn(tmp_path) -> None:
    task = ZipZapZopTask(start_number=1)
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
    assert (Path(result.output_dir) / "manifest.json").exists()


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
