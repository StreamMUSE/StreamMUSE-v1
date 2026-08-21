from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import requests

from streammuse.application.tasks import (
    InteractiveTaskRuntime,
    InteractiveTaskRuntimeConfig,
)
from streammuse.domain.tasks import (
    AnimalNamingTask,
    ChatModelResponse,
    HumanResponse,
    SpeechPlayback,
    SpeechRequest,
    TaskViewEvent,
    ZipZapZopTask,
)
from streammuse.infrastructure.voice import SpeechOutputError


class ManualClock:
    def __init__(self) -> None:
        self.value = 0.0

    def now(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class FakeModel:
    def __init__(
        self,
        clock: ManualClock,
        *,
        text: str = "1",
        latency_ms: float = 100.0,
        error: BaseException | None = None,
    ) -> None:
        self.clock = clock
        self.text = text
        self.latency_ms = latency_ms
        self.error = error

    def generate(self, messages, **kwargs) -> ChatModelResponse:
        _ = messages, kwargs
        self.clock.advance(0.1)
        if self.error is not None:
            raise self.error
        return ChatModelResponse(
            text=self.text,
            latency_ms=self.latency_ms,
            prompt_tokens=3,
            completion_tokens=1,
        )


class FakeTerminal:
    def __init__(self) -> None:
        self.outputs: list[str] = []

    def write(self, text: str) -> None:
        self.outputs.append(text)

    def prompt(self, text: str) -> str:
        raise AssertionError(text)

    def prompt_with_timeout(self, text: str, timeout_s: float):
        raise AssertionError((text, timeout_s))


class FakeSpeechSink:
    mode = "audio"

    def __init__(
        self,
        clock: ManualClock,
        *,
        playback: SpeechPlayback | None = None,
        error: BaseException | None = None,
    ) -> None:
        self.clock = clock
        self.playback = playback or SpeechPlayback(
            status="ok",
            spoken_text="1",
            cached=True,
            completed_normally=True,
            playback_start_offset_ms=5.0,
            first_dac_sample_offset_ms=20.0,
            playback_drained_offset_ms=400.0,
            stream_inactive_offset_ms=400.0,
            metadata={"sample_rate_hz": 22_050, "device": "fake"},
        )
        self.error = error
        self.requests: list[SpeechRequest] = []
        self.prepared: tuple[str, ...] = ()
        self.drains = 0
        self.closed = 0

    def start(self) -> None:
        return None

    @property
    def provenance(self) -> dict[str, Any]:
        return {
            "mode": "audio",
            "backend": "fake",
            "prewarm_ms": 1.0,
            "prewarm_entry_count": len(self.prepared),
            "prewarm_truncated": False,
        }

    def prepare(self, phrases: tuple[str, ...]) -> None:
        self.prepared = phrases

    def speak(self, request: SpeechRequest) -> SpeechPlayback:
        self.requests.append(request)
        self.clock.advance(0.4)
        if self.error is not None:
            raise self.error
        return self.playback

    def drain(self) -> None:
        self.drains += 1

    def close(self) -> None:
        self.closed += 1


class FakeVoiceSource:
    mode = "voice"

    def __init__(self) -> None:
        self.requests = []
        self.closed = 0

    def start(self) -> None:
        return None

    @property
    def provenance(self):
        return {"mode": "voice", "device": "fake mic"}

    def read_response(self, request):
        self.requests.append(request)
        return HumanResponse(text="two")

    def close(self) -> None:
        self.closed += 1


class RecordingTaskEventSink:
    def __init__(self) -> None:
        self.events: list[TaskViewEvent] = []

    def emit(self, event: TaskViewEvent) -> None:
        self.events.append(event)

    def close(self) -> None:
        return None


def _trace(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (path / "response_trace.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]


def test_tts_emits_started_and_final_playback_states(tmp_path: Path) -> None:
    clock = ManualClock()
    event_sink = RecordingTaskEventSink()
    runtime = InteractiveTaskRuntime(
        config=InteractiveTaskRuntimeConfig(
            output_dir=str(tmp_path),
            deadline_mode="soft",
            human_first=False,
        ),
        model_client=FakeModel(clock),
        terminal=FakeTerminal(),
        speech_output_sink=FakeSpeechSink(clock),
        task_event_sink=event_sink,
        now=clock.now,
    )

    runtime.play(ZipZapZopTask(), max_turns=1)

    speech_events = [
        event.payload for event in event_sink.events if event.type == "speech_output"
    ]
    assert [event["status"] for event in speech_events] == ["started", "ok"]
    assert speech_events[-1]["completed_normally"] is True


def test_tts_exception_emits_one_internal_error_state(tmp_path: Path) -> None:
    clock = ManualClock()
    event_sink = RecordingTaskEventSink()
    runtime = InteractiveTaskRuntime(
        config=InteractiveTaskRuntimeConfig(
            output_dir=str(tmp_path), deadline_mode="soft", human_first=False
        ),
        model_client=FakeModel(clock),
        terminal=FakeTerminal(),
        speech_output_sink=FakeSpeechSink(clock, error=RuntimeError("speaker failed")),
        task_event_sink=event_sink,
        now=clock.now,
    )

    with pytest.raises(RuntimeError, match="speaker failed"):
        runtime.play(ZipZapZopTask(), max_turns=1)

    speech_events = [
        event.payload for event in event_sink.events if event.type == "speech_output"
    ]
    assert [event["status"] for event in speech_events] == ["started", "internal_error"]


def test_audio_end_uses_unrounded_text_plus_drained_measurement(
    tmp_path: Path,
) -> None:
    clock = ManualClock()
    sink = FakeSpeechSink(clock)
    runtime = InteractiveTaskRuntime(
        config=InteractiveTaskRuntimeConfig(
            output_dir=str(tmp_path),
            deadline_mode="soft",
            human_first=False,
            llm_deadline_basis="audio_end",
        ),
        model_client=FakeModel(clock),
        terminal=FakeTerminal(),
        speech_output_sink=sink,
        now=clock.now,
    )

    runtime.play(ZipZapZopTask(), max_turns=1)
    record = _trace(tmp_path)[0]
    speech = record["metadata"]["speech_output"]

    assert record["latency_ms"] == pytest.approx(500.0)
    assert speech["source_text"] == "1"
    assert speech["spoken_text"] == "1"
    assert speech["effective_deadline_basis"] == "audio_end"
    assert speech["playback_drained_offset_ms"] == pytest.approx(500.0)
    assert sink.prepared
    assert json.loads(
        (tmp_path / "manifest.json").read_text(encoding="utf-8")
    )["speech_output"]["backend"] == "fake"


def test_animal_naming_uses_on_demand_speech_without_vocabulary_prewarm(
    tmp_path: Path,
) -> None:
    clock = ManualClock()
    sink = FakeSpeechSink(
        clock,
        playback=SpeechPlayback(
            status="cache_miss_synthesized",
            spoken_text="lion",
            cached=False,
            synthesis_ms=50.0,
            completed_normally=True,
            first_dac_sample_offset_ms=70.0,
            playback_drained_offset_ms=300.0,
            stream_inactive_offset_ms=300.0,
        ),
    )
    runtime = InteractiveTaskRuntime(
        config=InteractiveTaskRuntimeConfig(
            output_dir=str(tmp_path),
            deadline_mode="soft",
            human_first=False,
        ),
        model_client=FakeModel(clock, text="Lion."),
        terminal=FakeTerminal(),
        speech_output_sink=sink,
        now=clock.now,
    )

    runtime.play(AnimalNamingTask(), max_turns=1)

    record = _trace(tmp_path)[0]
    assert sink.prepared == ()
    assert len(sink.requests) == 1
    assert sink.requests[0].text == "lion"
    assert sink.requests[0].source_text == "Lion."
    assert record["is_valid"] is True
    assert record["metadata"]["speech_output"]["status"] == (
        "cache_miss_synthesized"
    )
    assert record["metadata"]["referee_metadata"] == {
        "normalized_animal": "lion"
    }


def test_llm_turn_emits_session_aligned_timing_breakdown(
    tmp_path: Path,
) -> None:
    clock = ManualClock()
    sink = FakeSpeechSink(clock)
    runtime = InteractiveTaskRuntime(
        config=InteractiveTaskRuntimeConfig(
            output_dir=str(tmp_path),
            deadline_mode="soft",
            human_first=False,
        ),
        model_client=FakeModel(clock),
        terminal=FakeTerminal(),
        speech_output_sink=sink,
        now=clock.now,
        timing_now=clock.now,
    )

    runtime.play(ZipZapZopTask(), max_turns=1)

    breakdown = _trace(tmp_path)[0]["metadata"]["timing_breakdown"]
    anchors = breakdown["anchors_ms"]
    durations = breakdown["durations_ms"]
    assert breakdown["schema_version"] == 1
    assert breakdown["origin_session_offset_ms"] == pytest.approx(0.0)
    assert durations["llm_request"] == pytest.approx(100.0)
    assert anchors["speech.first_dac_sample"] == pytest.approx(120.0)
    assert anchors["speech.playback_drained"] == pytest.approx(500.0)
    assert durations["game_validation"] >= 0.0


def test_incomplete_playback_falls_back_to_text_and_is_counted(
    tmp_path: Path,
) -> None:
    clock = ManualClock()
    sink = FakeSpeechSink(
        clock,
        playback=SpeechPlayback(
            status="playback_failed",
            spoken_text="1",
            completed_normally=False,
            first_dac_sample_offset_ms=20.0,
            stream_inactive_offset_ms=200.0,
            error={"type": "SpeakerPlaybackError", "message": "failed"},
        ),
    )
    runtime = InteractiveTaskRuntime(
        config=InteractiveTaskRuntimeConfig(
            output_dir=str(tmp_path),
            deadline_mode="soft",
            human_first=False,
            llm_deadline_basis="audio_end",
            speech_on_error="warn",
        ),
        model_client=FakeModel(clock),
        terminal=FakeTerminal(),
        speech_output_sink=sink,
        now=clock.now,
    )

    runtime.play(ZipZapZopTask(), max_turns=1)
    record = _trace(tmp_path)[0]
    manifest = json.loads(
        (tmp_path / "manifest.json").read_text(encoding="utf-8")
    )

    assert record["latency_ms"] == pytest.approx(100.0)
    assert (
        record["metadata"]["speech_output"][
            "deadline_basis_fallback_reason"
        ]
        == "playback_incomplete"
    )
    assert manifest["audio_end_fallback_turn_count"] == 1
    assert manifest["audio_end_fallback_reasons"] == {
        "playback_incomplete": 1
    }


def test_fail_policy_records_turn_before_raising(tmp_path: Path) -> None:
    clock = ManualClock()
    sink = FakeSpeechSink(
        clock,
        playback=SpeechPlayback(
            status="synthesis_failed",
            spoken_text="1",
            error={"type": "SpeechSynthesisError", "message": "failed"},
        ),
    )
    runtime = InteractiveTaskRuntime(
        config=InteractiveTaskRuntimeConfig(
            output_dir=str(tmp_path),
            deadline_mode="soft",
            human_first=False,
            speech_on_error="fail",
        ),
        model_client=FakeModel(clock),
        terminal=FakeTerminal(),
        speech_output_sink=sink,
        now=clock.now,
    )

    with pytest.raises(SpeechOutputError, match="failed"):
        runtime.play(ZipZapZopTask(), max_turns=1)

    assert _trace(tmp_path)[0]["metadata"]["speech_output"]["status"] == (
        "synthesis_failed"
    )
    assert (tmp_path / "artifacts" / "turn" / "0001_turn_0000.json").exists()


@pytest.mark.parametrize(
    ("error", "status"),
    [
        (KeyboardInterrupt(), "interrupted"),
        (SystemExit(7), "interrupted"),
        (TypeError("program defect"), "internal_error"),
    ],
)
def test_post_generation_exceptions_are_recorded_then_rethrown_unchanged(
    tmp_path: Path,
    error: BaseException,
    status: str,
) -> None:
    clock = ManualClock()
    runtime = InteractiveTaskRuntime(
        config=InteractiveTaskRuntimeConfig(
                output_dir=str(tmp_path),
                deadline_mode="soft",
                human_first=False,
                speech_on_error="warn",
        ),
        model_client=FakeModel(clock),
        terminal=FakeTerminal(),
        speech_output_sink=FakeSpeechSink(clock, error=error),
        now=clock.now,
    )

    with pytest.raises(type(error)) as caught:
        runtime.play(ZipZapZopTask(), max_turns=1)

    assert caught.value is error
    speech = _trace(tmp_path)[0]["metadata"]["speech_output"]
    assert speech["status"] == status
    assert speech["error"]["type"] == type(error).__name__
    timing = _trace(tmp_path)[0]["metadata"]["timing_breakdown"]
    assert timing["durations_ms"]["speech_output"] is not None


def test_guard_runs_once_after_actual_machine_audio_before_voice_input(
    tmp_path: Path,
) -> None:
    clock = ManualClock()
    sink = FakeSpeechSink(clock)
    source = FakeVoiceSource()
    sleeps: list[float] = []
    runtime = InteractiveTaskRuntime(
        config=InteractiveTaskRuntimeConfig(
            output_dir=str(tmp_path),
            deadline_mode="soft",
            human_first=False,
            speech_guard_ms=200.0,
        ),
        model_client=FakeModel(clock),
        terminal=FakeTerminal(),
        human_response_source=source,
        speech_output_sink=sink,
        now=clock.now,
        sleep=sleeps.append,
    )

    runtime.play(ZipZapZopTask(), max_turns=2)
    records = _trace(tmp_path)

    assert sleeps == [0.2]
    assert sink.drains == 1
    assert records[1]["metadata"]["human_input"]["guard_ms"] == 200.0


def test_llm_timeout_marks_speech_not_attempted(tmp_path: Path) -> None:
    clock = ManualClock()
    runtime = InteractiveTaskRuntime(
        config=InteractiveTaskRuntimeConfig(
            output_dir=str(tmp_path),
            deadline_mode="hard",
            human_first=False,
            llm_deadline_basis="audio_end",
        ),
        model_client=FakeModel(
            clock,
            error=requests.Timeout("server timeout"),
        ),
        terminal=FakeTerminal(),
        speech_output_sink=FakeSpeechSink(clock),
        now=clock.now,
    )

    runtime.play(ZipZapZopTask(), max_turns=1)
    speech = _trace(tmp_path)[0]["metadata"]["speech_output"]

    assert speech["status"] == "not_attempted"
    assert speech["effective_deadline_basis"] == "text"
    assert speech["deadline_basis_fallback_reason"] == "llm_timeout"
