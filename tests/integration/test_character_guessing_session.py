from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from streammuse.application.tasks import (
    InteractiveTaskRuntime,
    InteractiveTaskRuntimeConfig,
)
from streammuse.domain.tasks import (
    CharacterGuessingTask,
    ChatModelResponse,
    HumanResponse,
    HumanResponseRequest,
    SpeechPlayback,
    SpeechRequest,
    WORLD_HISTORICAL_FIGURES_EN,
)
from streammuse.infrastructure.task_view import QueueTaskEventSink


class ScriptedModel:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.requests: list[list[dict[str, str]]] = []

    def generate(
        self,
        messages: list[dict[str, str]],
        **kwargs: object,
    ) -> ChatModelResponse:
        _ = kwargs
        self.requests.append(messages)
        return ChatModelResponse(
            text=self.responses.pop(0),
            latency_ms=5.0,
            prompt_tokens=20,
            completion_tokens=8,
        )


class SetupTerminal:
    def __init__(self) -> None:
        self.outputs: list[str] = []

    def write(self, text: str) -> None:
        self.outputs.append(text)

    def prompt(self, text: str) -> str:
        self.outputs.append(text)
        return ""

    def prompt_with_timeout(self, text: str, timeout_s: float) -> object:
        raise AssertionError((text, timeout_s))


class ScriptedVoiceSource:
    mode = "voice"

    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.requests: list[HumanResponseRequest] = []
        self.started = 0
        self.closed = 0

    def start(self) -> None:
        self.started += 1

    @property
    def provenance(self) -> dict[str, object]:
        return {"mode": "voice", "device": "fake microphone"}

    def read_response(
        self,
        request: HumanResponseRequest,
    ) -> HumanResponse:
        self.requests.append(request)
        raw = self.responses.pop(0)
        return HumanResponse(
            text=raw,
            latency_ms=10.0,
            metadata={"capture_reason": "fake"},
        )

    def close(self) -> None:
        self.closed += 1


class RecordingSpeechSink:
    mode = "audio"

    def __init__(self) -> None:
        self.requests: list[SpeechRequest] = []
        self.prepared: tuple[str, ...] = ()
        self.started = 0
        self.drained = 0
        self.closed = 0

    def start(self) -> None:
        self.started += 1

    @property
    def provenance(self) -> dict[str, object]:
        return {"mode": "audio", "backend": "fake"}

    def prepare(self, phrases: tuple[str, ...]) -> None:
        self.prepared = phrases

    def speak(self, request: SpeechRequest) -> SpeechPlayback:
        self.requests.append(request)
        return SpeechPlayback(
            status="cache_miss_synthesized",
            spoken_text=request.text,
            cached=False,
            completed_normally=True,
            playback_start_offset_ms=0.0,
            first_dac_sample_offset_ms=1.0,
            playback_drained_offset_ms=2.0,
            stream_inactive_offset_ms=2.0,
        )

    def drain(self) -> None:
        self.drained += 1

    def close(self) -> None:
        self.closed += 1


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_trace(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


def test_full_character_session_with_voice_speech_web_and_retries(
    tmp_path: Path,
) -> None:
    model = ScriptedModel(
        [
            "I need to think first.",
            "ASK: Is this person male?",
            "ASK: Is this person a scientist?",
            "GUESS: Newton",
            "GUESS: Einstein",
        ]
    )
    voice = ScriptedVoiceSource(
        ["maybe", "yes", "yes", "no", "yes"]
    )
    speech = RecordingSpeechSink()
    terminal = SetupTerminal()
    event_sink = QueueTaskEventSink(
        session_id="character-session",
        task="character_guessing",
    )
    runtime = InteractiveTaskRuntime(
        config=InteractiveTaskRuntimeConfig(
            output_dir=str(tmp_path),
            deadline_mode="soft",
            max_tokens=64,
            speech_guard_ms=0.0,
        ),
        model_client=model,
        terminal=terminal,
        human_response_source=voice,
        speech_output_sink=speech,
        task_event_sink=event_sink,
        task_event_session_id="character-session",
    )
    task = CharacterGuessingTask(
        scope=WORLD_HISTORICAL_FIGURES_EN,
        max_rounds=20,
    )

    result = runtime.play(task, max_turns=120)
    run_dir = Path(result.output_dir)
    trace = _read_trace(run_dir / "response_trace.jsonl")
    summary = _read_json(run_dir / "run_summary.json")
    manifest = _read_json(run_dir / "manifest.json")
    snapshot = event_sink.snapshot

    assert result.stop_reason == "character_guessed"
    assert result.logical_round_count == 4
    assert result.turn_count == 10
    assert result.invalid_count == 2
    assert result.question_count == 2
    assert result.guess_count == 2
    assert result.rejected_guess_count == 1
    assert result.successful_character_id == "albert_einstein"
    assert [item["actor"] for item in trace[:4]] == [
        "llm",
        "llm",
        "human",
        "human",
    ]
    assert [item["metadata"]["logical_round"] for item in trace[:4]] == [
        1,
        1,
        1,
        1,
    ]
    assert voice.requests[0].speech_context is not None
    assert voice.requests[0].speech_context.hotwords[:2] == ("yes", "no")
    assert speech.prepared == ()
    assert [request.text for request in speech.requests] == [
        "Is this person male?",
        "Is this person a scientist?",
        "Is your person Isaac Newton?",
        "Is your person Albert Einstein?",
    ]
    assert speech.drained == 4
    assert voice.started == voice.closed == 1
    assert speech.started == speech.closed == 1
    assert summary["logical_round_count"] == 4
    assert manifest["successful_guess"] is True
    assert snapshot.status == "finished"
    assert snapshot.setup is not None
    assert snapshot.setup["public_state"]["scope_size"] == 44
    assert snapshot.session_result is not None
    assert snapshot.session_result["stop_reason"] == "character_guessed"
    serialized_public_outputs = json.dumps(
        {
            "manifest": manifest,
            "snapshot": snapshot.envelope(),
        }
    ).lower()
    assert "secret_target" not in serialized_public_outputs
