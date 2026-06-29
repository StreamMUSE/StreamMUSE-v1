from __future__ import annotations

from streammuse.domain.tasks import ZipZapZopTask


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
