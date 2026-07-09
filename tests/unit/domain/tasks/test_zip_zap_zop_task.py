from __future__ import annotations

from streammuse.domain.tasks import InteractiveTurnRecord, ZipZapZopTask


def test_zip_zap_zop_task_progresses_state_on_valid_response() -> None:
    task = ZipZapZopTask(start_number=1)
    turn = task.build_turn(task.initial_state())

    result = task.validate_response(turn.state, "1")
    next_state = task.advance_state(turn.state, result, "1")

    assert turn.turn_id == 0
    assert turn.expected_output == "1"
    assert turn.messages[-1] == {"role": "user", "content": "1:"}
    assert result.is_valid is True
    assert result.expected_output == "1"
    assert result.failure_reason == "NONE"
    assert next_state.turn_index == 1
    assert next_state.data["current_number"] == 1


def test_zip_zap_zop_task_referee_reports_expected_output() -> None:
    task = ZipZapZopTask(start_number=3)
    state = task.initial_state()

    result = task.validate_response(state, "3")
    next_state = task.advance_state(state, result, "3")

    assert result.is_valid is False
    assert result.expected_output == "Zip"
    assert result.failure_reason == "EXPECTED_MISMATCH"
    assert next_state.turn_index == 1
    assert next_state.data["current_number"] == 3


def test_zip_zap_zop_interactive_messages_can_include_bounded_transcript() -> None:
    task = ZipZapZopTask(start_number=1, history_limit=1)
    state = task.initial_state()
    first_result = task.validate_response(state, "1", actor="human", transcript=[])
    next_state = task.advance_state(state, first_result, "1", actor="human", transcript=[])
    transcript = [
        InteractiveTurnRecord(
            turn_id=0,
            actor="human",
            number=1,
            prompt="1:",
            response="1",
            expected="1",
            is_valid=True,
        )
    ]

    messages = task.build_llm_messages(next_state, transcript)

    assert task.build_human_prompt(state, []) == "1:"
    assert "with a human" in messages[0]["content"]
    assert "Human at 1: 1" in messages[-1]["content"]
    assert "Your turn. 2:" in messages[-1]["content"]
    assert next_state.history[-1]["role"] == "human"
    assert next_state.history[-1]["number"] == "1"


def test_zip_zap_zop_interactive_hint_and_expected_are_task_specific() -> None:
    task = ZipZapZopTask(start_number=15)
    state = task.initial_state()

    assert task.expected_for_state(state, []) == "ZipZop"
    assert task.build_hint(state, []) == "Check divisibility by: 3, 5."


def test_zip_zap_zop_oracle_history_records_expected_instead_of_model_output() -> None:
    task = ZipZapZopTask(start_number=3, history_limit=1, oracle_history=True)
    state = task.initial_state()

    result = task.validate_response(state, "wrong")
    next_state = task.advance_state(state, result, "wrong")

    assert result.expected_output == "Zip"
    assert next_state.history[-1]["content"] == "Zip"
    assert next_state.history[-1]["expected"] == "Zip"


def test_zip_zap_zop_default_history_records_model_output() -> None:
    task = ZipZapZopTask(start_number=3, history_limit=1)
    state = task.initial_state()

    result = task.validate_response(state, "wrong")
    next_state = task.advance_state(state, result, "wrong")

    assert next_state.history[-1]["content"] == "wrong"
