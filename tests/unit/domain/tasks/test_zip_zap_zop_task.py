from __future__ import annotations

import pytest

from streammuse.domain.tasks import (
    InteractiveTurnRecord,
    SpeechAwareInteractiveTask,
    ZipZapZopTask,
)


def test_zip_zap_zop_task_progresses_state_on_valid_response() -> None:
    task = ZipZapZopTask(start_number=1)
    turn = task.build_turn(task.initial_state())

    result = task.validate_response(turn.state, "1")
    next_state = task.advance_state(turn.state, result, "1")

    assert turn.turn_id == 0
    assert turn.expected_output == "1"
    assert turn.messages[-1] == {"role": "user", "content": "Current number: 1\nAnswer:"}
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
    assert "Current number: 2\nAnswer:" in messages[-1]["content"]
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


def test_zip_zap_zop_exposes_static_speech_context_without_turn_answer() -> None:
    task = ZipZapZopTask(start_number=1)
    first_state = task.initial_state()
    later_state = task.advance_state(
        first_state,
        task.validate_response(first_state, "1"),
        "1",
    )

    first_context = task.build_speech_context(first_state, [])
    later_context = task.build_speech_context(later_state, [])

    assert isinstance(task, SpeechAwareInteractiveTask)
    assert first_context == later_context
    assert first_context.hotwords == ("Zip", "Zap", "Zop")
    assert first_context.initial_prompt is not None
    assert "1" not in first_context.initial_prompt
    assert "2" not in first_context.initial_prompt


@pytest.mark.parametrize(
    ("spoken", "canonical"),
    [
        (" zip! ", "Zip"),
        ("ZAP...", "Zap"),
        ("zOp?", "Zop"),
        ("zip zap", "ZipZap"),
        ("ZIP   ZOP!", "ZipZop"),
        ("zap zop", "ZapZop"),
        ("zip zap zop.", "ZipZapZop"),
        ("ZipZap", "ZipZap"),
        ("zipzapzop", "ZipZapZop"),
    ],
)
def test_zip_zap_zop_parses_only_valid_game_word_sequences(spoken: str, canonical: str) -> None:
    task = ZipZapZopTask()

    result = task.parse_spoken_response(task.initial_state(), [], spoken)

    assert result.canonical_text == canonical
    assert result.status == "ok"
    assert result.reason is None


@pytest.mark.parametrize(
    ("spoken", "canonical"),
    [
        ("7", "7"),
        ("007!", "7"),
        ("zero", "0"),
        ("nineteen.", "19"),
        ("twenty one", "21"),
        ("twenty-one", "21"),
        ("one hundred", "100"),
        ("one hundred and five", "105"),
        ("nine hundred ninety nine", "999"),
        ("one thousand and one", "1001"),
        ("twenty one thousand three hundred forty five", "21345"),
        ("one million two hundred thousand and six", "1200006"),
        ("-12", "-12"),
        ("minus twenty one", "-21"),
        ("-twenty-one", "-21"),
        ("negative one hundred and five", "-105"),
    ],
)
def test_zip_zap_zop_parses_digits_and_strict_english_number_phrases(
    spoken: str,
    canonical: str,
) -> None:
    task = ZipZapZopTask()

    result = task.parse_spoken_response(task.initial_state(), [], spoken)

    assert result.canonical_text == canonical
    assert result.status == "ok"


@pytest.mark.parametrize(
    ("spoken", "reason"),
    [
        ("", "empty_transcript"),
        ("...", "empty_transcript"),
        ("sap", "unrecognized_response"),
        ("zip zip", "invalid_game_word_sequence"),
        ("zop zap", "invalid_game_word_sequence"),
        ("zip sap", "invalid_game_word_sequence"),
        ("one twenty", "invalid_number_phrase"),
        ("hundred", "invalid_number_phrase"),
        ("one and two", "invalid_number_phrase"),
        ("one thousand and", "invalid_number_phrase"),
        ("minus", "unrecognized_response"),
        ("1000000000", "number_out_of_range"),
        ("9" * 5000, "number_out_of_range"),
    ],
)
def test_zip_zap_zop_rejects_empty_fuzzy_and_out_of_grammar_responses(
    spoken: str,
    reason: str,
) -> None:
    task = ZipZapZopTask(start_number=3)

    result = task.parse_spoken_response(task.initial_state(), [], spoken)

    assert result.canonical_text is None
    assert result.status == ("empty" if reason == "empty_transcript" else "unrecognized")
    assert result.reason == reason


def test_spoken_parser_does_not_use_the_expected_answer_to_repair_transcript() -> None:
    task = ZipZapZopTask(start_number=4)
    state = task.initial_state()

    parsed = task.parse_spoken_response(state, [], "sap")
    referee = task.validate_response(state, parsed.canonical_text or "")

    assert task.expected_for_state(state, []) == "Zap"
    assert parsed.canonical_text is None
    assert referee.is_valid is False
