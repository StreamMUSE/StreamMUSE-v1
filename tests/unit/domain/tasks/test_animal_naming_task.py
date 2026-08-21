from __future__ import annotations

import pytest

from streammuse.domain.tasks import (
    AnimalNamingTask,
    SpeechAwareInteractiveTask,
    SpeechContext,
    SpeechRenderableTask,
)


def test_animal_naming_task_accepts_whitelisted_animal_and_tracks_it() -> None:
    task = AnimalNamingTask()
    state = task.initial_state()
    turn = task.build_turn(state)

    result = task.validate_response(state, "Cat")
    next_state = task.advance_state(state, result, "Cat")

    assert turn.turn_id == 0
    assert turn.expected_output is None
    assert turn.messages[0]["content"] == (
        "You are playing Animal Naming. Respond with exactly one common real animal name. "
        "Never output an animal listed as forbidden. Do not explain your answer."
    )
    assert turn.messages[-1]["content"] == "Name an animal:"
    assert result.is_valid is True
    assert result.failure_reason == "NONE"
    assert result.metadata["normalized_animal"] == "cat"
    assert next_state.turn_index == 1
    assert next_state.data["used_animals"] == ["cat"]
    assert next_state.data["attempted_animals"] == ["cat"]


def test_animal_naming_task_rejects_repeated_animals() -> None:
    task = AnimalNamingTask()
    state = task.initial_state()
    first = task.validate_response(state, "cat")
    state = task.advance_state(state, first, "cat")

    repeated = task.validate_response(state, "CAT")
    next_state = task.advance_state(state, repeated, "CAT")

    assert repeated.is_valid is False
    assert repeated.failure_reason == "REPEATED_ANIMAL"
    assert repeated.metadata["normalized_animal"] == "cat"
    assert next_state.turn_index == 2
    assert next_state.data["used_animals"] == ["cat"]
    assert next_state.data["attempted_animals"] == ["cat"]


def test_animal_naming_task_rejects_unknown_animals() -> None:
    task = AnimalNamingTask()
    state = task.initial_state()

    result = task.validate_response(state, "dragon")

    assert result.is_valid is False
    assert result.failure_reason == "UNKNOWN_ANIMAL"
    assert result.metadata["normalized_animal"] == "dragon"


def test_animal_naming_task_tracks_invalid_attempts_as_forbidden() -> None:
    task = AnimalNamingTask()
    state = task.initial_state()
    result = task.validate_response(state, "dragon")
    state = task.advance_state(state, result, "dragon")

    turn = task.build_turn(state)

    assert state.data["used_animals"] == []
    assert state.data["attempted_animals"] == ["dragon"]
    assert turn.messages[-1]["content"] == (
        "Forbidden animal names: dragon\n"
        "Choose one animal name that is not in the forbidden list. Output only the new animal name:"
    )


def test_animal_naming_task_accepts_common_short_names() -> None:
    task = AnimalNamingTask()
    state = task.initial_state()

    result = task.validate_response(state, "rhino")
    bird_result = task.validate_response(state, "bird")

    assert result.is_valid is True
    assert result.metadata["normalized_animal"] == "rhino"
    assert bird_result.is_valid is True
    assert bird_result.metadata["normalized_animal"] == "bird"


def test_animal_naming_task_prompt_includes_used_animals() -> None:
    task = AnimalNamingTask()
    state = task.initial_state()
    result = task.validate_response(state, "cat")
    state = task.advance_state(state, result, "cat")

    turn = task.build_turn(state)

    assert turn.turn_id == 1
    assert turn.messages[-1]["content"] == (
        "Forbidden animal names: cat\n"
        "Choose one animal name that is not in the forbidden list. Output only the new animal name:"
    )
    assert turn.metadata["used_animals"] == ["cat"]


def test_animal_naming_task_implements_interactive_speech_contracts() -> None:
    task = AnimalNamingTask()

    assert isinstance(task, SpeechAwareInteractiveTask)
    assert isinstance(task, SpeechRenderableTask)
    for method_name in (
        "build_human_prompt",
        "build_llm_messages",
        "validate_response",
        "advance_state",
        "expected_for_state",
        "build_hint",
    ):
        assert callable(getattr(task, method_name))


def test_animal_naming_interactive_prompts_are_open_ended_and_state_driven() -> None:
    task = AnimalNamingTask()
    state = task.initial_state()
    human_result = task.validate_response(state, "Lion", actor="human", transcript=[])
    state = task.advance_state(
        state,
        human_result,
        "Lion",
        actor="human",
        transcript=[],
    )
    unknown_result = task.validate_response(
        state,
        "dragon",
        actor="llm",
        transcript=[],
    )
    state = task.advance_state(
        state,
        unknown_result,
        "dragon",
        actor="llm",
        transcript=[],
    )

    messages = task.build_llm_messages(state, [])

    assert task.build_human_prompt(state, []) == "Name one unused animal:"
    assert "with a human" in messages[0]["content"]
    assert "Allowed animal names:" in messages[1]["content"]
    allowed_line = messages[1]["content"].splitlines()[0]
    assert "lion" not in allowed_line.split(": ", 1)[1].split(", ")
    assert "elephant" in allowed_line.split(": ", 1)[1].split(", ")
    assert "Forbidden animal names: lion, dragon" in messages[1]["content"]
    assert task.expected_for_state(state, []) is None
    assert task.build_hint(state, []) == (
        "90 unused animal names remain. "
        "Name one animal that has not appeared before."
    )


def test_animal_naming_tracks_human_and_llm_in_shared_state() -> None:
    task = AnimalNamingTask()
    state = task.initial_state()

    human_result = task.validate_response(state, "lion", actor="human", transcript=[])
    state = task.advance_state(
        state,
        human_result,
        "lion",
        actor="human",
        transcript=[],
    )
    llm_result = task.validate_response(state, "elephant", actor="llm", transcript=[])
    state = task.advance_state(
        state,
        llm_result,
        "elephant",
        actor="llm",
        transcript=[],
    )

    assert state.data["used_animals"] == ["lion", "elephant"]
    assert state.data["attempted_animals"] == ["lion", "elephant"]
    assert [record["role"] for record in state.history] == ["human", "assistant"]
    assert task.validate_response(state, "LION", actor="llm").failure_reason == (
        "REPEATED_ANIMAL"
    )


@pytest.mark.parametrize(
    ("response", "normalized"),
    [
        ("Lion", "lion"),
        (" A LION. ", "lion"),
        ("the rhinoceros!", "rhinoceros"),
        ("Cat\nexplanation", "cat"),
    ],
)
def test_animal_naming_normalizes_text_consistently(
    response: str,
    normalized: str,
) -> None:
    result = AnimalNamingTask().validate_response(
        AnimalNamingTask().initial_state(),
        response,
    )

    assert result.metadata["normalized_animal"] == normalized


def test_animal_naming_remaining_animals_are_sorted_and_exclude_used() -> None:
    task = AnimalNamingTask(animals={"tiger", "cat", "lion"})
    state = task.initial_state()
    result = task.validate_response(state, "lion")
    state = task.advance_state(state, result, "lion")

    assert task.remaining_animals(state) == ("cat", "tiger")


def test_animal_naming_speech_context_contains_only_remaining_animals() -> None:
    task = AnimalNamingTask(animals={"tiger", "cat", "lion"})
    state = task.initial_state()
    result = task.validate_response(state, "lion")
    state = task.advance_state(state, result, "lion", actor="human")

    assert task.build_speech_context(state, []) == SpeechContext(
        initial_prompt=(
            "The speaker will say exactly one common English animal name."
        ),
        hotwords=("cat", "tiger"),
    )


@pytest.mark.parametrize(
    ("raw_text", "status", "canonical", "reason"),
    [
        ("Lion.", "ok", "lion", None),
        ("A rhinoceros.", "ok", "rhinoceros", None),
        ("Polar bear.", "ok", "polar bear", None),
        ("Dragon.", "ok", "dragon", None),
        ("", "empty", None, "empty_transcript"),
        ("maybe lion", "unrecognized", None, "explanatory_speech"),
        ("lion and tiger", "unrecognized", None, "explanatory_speech"),
        ("lion lion lion", "unrecognized", None, "repeated_speech_tokens"),
        ("I would like to say lion.", "unrecognized", None, "invalid_animal_phrase"),
    ],
)
def test_animal_naming_parses_bounded_spoken_animal_phrases(
    raw_text: str,
    status: str,
    canonical: str | None,
    reason: str | None,
) -> None:
    task = AnimalNamingTask()

    result = task.parse_spoken_response(task.initial_state(), [], raw_text)

    assert result.status == status
    assert result.canonical_text == canonical
    assert result.reason == reason


def test_spoken_unknown_and_repeated_animals_reach_the_referee() -> None:
    task = AnimalNamingTask()
    state = task.initial_state()
    parsed = task.parse_spoken_response(state, [], "Dragon.")

    assert parsed.canonical_text == "dragon"
    assert task.validate_response(
        state,
        parsed.canonical_text or "",
        actor="human",
    ).failure_reason == "UNKNOWN_ANIMAL"

    first = task.validate_response(state, "lion")
    state = task.advance_state(state, first, "lion", actor="llm")
    parsed = task.parse_spoken_response(state, [], "Lion.")

    assert parsed.canonical_text == "lion"
    assert task.validate_response(
        state,
        parsed.canonical_text or "",
        actor="human",
    ).failure_reason == "REPEATED_ANIMAL"
