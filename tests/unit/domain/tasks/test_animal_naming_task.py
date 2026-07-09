from __future__ import annotations

from streammuse.domain.tasks import AnimalNamingTask


def test_animal_naming_task_accepts_whitelisted_animal_and_tracks_it() -> None:
    task = AnimalNamingTask()
    state = task.initial_state()
    turn = task.build_turn(state)

    result = task.validate_response(state, "Cat")
    next_state = task.advance_state(state, result, "Cat")

    assert turn.turn_id == 0
    assert turn.expected_output is None
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
