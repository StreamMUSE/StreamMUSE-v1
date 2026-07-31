from __future__ import annotations

import pytest

from streammuse.domain.tasks import SpeechRenderableTask, ZipZapZopTask


@pytest.mark.parametrize(
    ("response", "spoken"),
    [
        ("Zip", "Zip"),
        ("ZipZap", "Zip Zap"),
        ("ZipZapZop", "Zip Zap Zop"),
        ("17", "17"),
        ("-17", "-17"),
        ("", None),
        ("not an answer", None),
    ],
)
def test_zip_zap_zop_builds_task_owned_spoken_text(
    response: str,
    spoken: str | None,
) -> None:
    task = ZipZapZopTask()

    assert isinstance(task, SpeechRenderableTask)
    assert (
        task.build_spoken_text(
            task.initial_state(),
            [],
            response,
            actor="llm",
        )
        == spoken
    )


def test_spoken_text_preserves_parseable_wrong_answer() -> None:
    task = ZipZapZopTask(start_number=3)

    assert (
        task.build_spoken_text(
            task.initial_state(),
            [],
            "16",
            actor="llm",
        )
        == "16"
    )


def test_speech_vocabulary_is_bounded_and_excludes_game_word_numbers() -> None:
    task = ZipZapZopTask(start_number=1)

    vocabulary = task.speech_vocabulary(
        task.initial_state(),
        max_turns=6,
    )

    assert vocabulary[:7] == (
        "Zip",
        "Zap",
        "Zop",
        "Zip Zap",
        "Zip Zop",
        "Zap Zop",
        "Zip Zap Zop",
    )
    assert vocabulary[7:] == ("1", "2")
