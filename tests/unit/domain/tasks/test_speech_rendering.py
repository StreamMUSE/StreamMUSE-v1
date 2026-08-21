from __future__ import annotations

import pytest

from streammuse.domain.tasks import AnimalNamingTask, SpeechRenderableTask, ZipZapZopTask


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


@pytest.mark.parametrize(
    ("response", "spoken"),
    [
        ("Lion.", "lion"),
        ("a rhinoceros", "rhinoceros"),
        ("dragon", "dragon"),
        ("lion and tiger", None),
        ("I choose lion", None),
        ("", None),
    ],
)
def test_animal_naming_builds_task_owned_spoken_text(
    response: str,
    spoken: str | None,
) -> None:
    task = AnimalNamingTask()

    assert (
        task.build_spoken_text(
            task.initial_state(),
            [],
            response,
            actor="llm",
        )
        == spoken
    )


def test_animal_naming_does_not_prewarm_the_open_vocabulary() -> None:
    task = AnimalNamingTask()

    assert task.speech_vocabulary(task.initial_state(), max_turns=20) == ()
