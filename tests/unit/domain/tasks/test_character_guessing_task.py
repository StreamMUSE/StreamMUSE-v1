from __future__ import annotations

import pytest

from streammuse.domain.tasks import (
    CharacterCandidate,
    CharacterGuessingTask,
    CharacterScope,
    FlowControlledInteractiveTask,
    PublicTaskStateProvider,
    SetupAwareInteractiveTask,
    SpeechAwareInteractiveTask,
    SpeechRenderableTask,
    WORLD_HISTORICAL_FIGURES_EN,
    canonicalize_character_name,
    normalize_yes_no,
)


def task(*, max_rounds: int = 3) -> CharacterGuessingTask:
    return CharacterGuessingTask(
        scope=WORLD_HISTORICAL_FIGURES_EN,
        max_rounds=max_rounds,
    )


def advance(
    game: CharacterGuessingTask,
    state,
    response: str,
    actor: str,
):
    referee = game.validate_response(
        state,
        response,
        actor=actor,  # type: ignore[arg-type]
        transcript=[],
    )
    return (
        referee,
        game.advance_state(
            state,
            referee,
            response,
            actor=actor,  # type: ignore[arg-type]
            transcript=[],
        ),
    )


def test_catalog_is_stable_resolves_aliases_and_rejects_collisions() -> None:
    scope = WORLD_HISTORICAL_FIGURES_EN

    assert len(scope.candidates) == 44
    assert [item.display_name for item in scope.candidates] == sorted(
        item.display_name for item in scope.candidates
    )
    assert scope.resolve("Einstein.").character_id == "albert_einstein"
    assert canonicalize_character_name("Martin Luther King Jr.") == (
        "martin luther king jr"
    )

    with pytest.raises(ValueError, match="collision"):
        CharacterScope(
            "collision",
            "Collision",
            "en",
            (
                CharacterCandidate("one", "Person One", ("Shared",)),
                CharacterCandidate("two", "Person Two", ("shared",)),
            ),
        )


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Yes.", "yes"),
        ("YEAH", "yes"),
        ("that's right", "yes"),
        ("Nope!", "no"),
        ("incorrect", "no"),
        ("maybe", None),
        ("yes because", None),
        ("yes\nno", None),
    ],
)
def test_yes_no_normalization_is_strict(
    raw: str,
    expected: str | None,
) -> None:
    assert normalize_yes_no(raw) == expected


def test_initial_state_setup_and_public_projection_never_contain_target() -> None:
    game = task()
    state = game.initial_state()
    setup = game.build_setup(state)
    public = game.public_state(state)

    assert isinstance(game, FlowControlledInteractiveTask)
    assert isinstance(game, SetupAwareInteractiveTask)
    assert isinstance(game, PublicTaskStateProvider)
    assert isinstance(game, SpeechAwareInteractiveTask)
    assert isinstance(game, SpeechRenderableTask)
    assert game.flow_decision(state, []).next_actor == "llm"
    assert setup.choices[0] == "Abraham Lincoln"
    assert public["current_round"] == 1
    assert "target" not in repr(state).lower()
    assert "target" not in repr(public).lower()


@pytest.mark.parametrize(
    ("raw", "reason"),
    [
        ("", "EMPTY_RESPONSE"),
        ("Was this person male?", "INVALID_ACTION_FORMAT"),
        ("ASK: Tell me their job.", "NOT_YES_NO_QUESTION"),
        (
            "ASK: Was this person Abraham Lincoln?",
            "CANDIDATE_NAME_REQUIRES_GUESS",
        ),
        (
            "ASK: Is your person Einstein?",
            "CANDIDATE_NAME_REQUIRES_GUESS",
        ),
        ("ASK: Is this person male?\nThanks", "MULTILINE_ACTION"),
        ("GUESS: Sherlock Holmes", "CHARACTER_OUT_OF_SCOPE"),
        (
            "ASK: Is this person male? GUESS: Albert Einstein",
            "MULTIPLE_ACTIONS",
        ),
    ],
)
def test_llm_protocol_rejects_invalid_actions(
    raw: str,
    reason: str,
) -> None:
    game = task()
    result = game.validate_response(
        game.initial_state(),
        raw,
        actor="llm",
        transcript=[],
    )

    assert result.is_valid is False
    assert result.failure_reason == reason


def test_invalid_attempts_keep_same_actor_and_preserve_pending_action() -> None:
    game = task()
    state = game.initial_state()

    _, state = advance(game, state, "thinking...", "llm")
    assert state.data["phase"] == "awaiting_llm"
    assert game.flow_decision(state, []).next_actor == "llm"

    valid, state = advance(
        game,
        state,
        "ASK: Was this person born before 1900?",
        "llm",
    )
    pending = state.data["pending_action"]
    assert valid.is_valid is True
    assert state.data["phase"] == "awaiting_human"

    invalid, state = advance(game, state, "maybe", "human")
    assert invalid.failure_reason == "INVALID_YES_NO"
    assert state.data["phase"] == "awaiting_human"
    assert state.data["pending_action"] == pending
    assert game.flow_decision(state, []).next_actor == "human"


def test_identity_question_must_retry_as_formal_guess() -> None:
    game = task()
    state = game.initial_state()

    invalid, state = advance(
        game,
        state,
        "ASK: Was this person Abraham Lincoln?",
        "llm",
    )
    assert invalid.failure_reason == "CANDIDATE_NAME_REQUIRES_GUESS"
    assert state.data["logical_round"] == 0
    assert game.flow_decision(state, []).next_actor == "llm"

    guess, state = advance(
        game,
        state,
        "GUESS: Abraham Lincoln",
        "llm",
    )
    assert guess.is_valid is True
    assert guess.metadata["action_kind"] == "guess"
    assert state.data["phase"] == "awaiting_human"


def test_question_round_updates_history_and_blocks_exact_repeat() -> None:
    game = task()
    state = game.initial_state()
    question = "ASK: Was this person born before 1900?"

    _, state = advance(game, state, question, "llm")
    human, state = advance(game, state, "Yeah.", "human")

    assert human.metadata["canonical_answer"] == "yes"
    assert state.data["logical_round"] == 1
    assert state.data["rounds"][0]["answer"] == "yes"
    assert game.flow_decision(state, []).next_actor == "llm"
    repeated = game.validate_response(
        state,
        question,
        actor="llm",
        transcript=[],
    )
    assert repeated.failure_reason == "REPEATED_QUESTION"
    prompt = game.build_llm_messages(state, [])[1]["content"]
    assert "WAS THIS PERSON BORN BEFORE 1900" not in prompt
    assert "Was this person born before 1900?" in prompt
    assert "Rounds remaining: 2" in prompt


def test_rejected_guess_returns_to_llm_and_cannot_be_repeated() -> None:
    game = task()
    state = game.initial_state()

    guess, state = advance(game, state, "GUESS: Einstein", "llm")
    assert guess.metadata["canonical_character_id"] == "albert_einstein"
    _, state = advance(game, state, "no", "human")

    assert state.data["completed"] is False
    assert game.flow_decision(state, []).next_actor == "llm"
    repeated = game.validate_response(
        state,
        "GUESS: Albert Einstein",
        actor="llm",
        transcript=[],
    )
    assert repeated.failure_reason == "REPEATED_GUESS"
    assert "Albert Einstein" in game.build_llm_messages(state, [])[1]["content"]


def test_confirmed_guess_completes_immediately_with_human_confirmation() -> None:
    game = task()
    state = game.initial_state()

    _, state = advance(game, state, "GUESS: Marie Curie", "llm")
    answer, state = advance(game, state, "correct", "human")
    decision = game.flow_decision(state, [])

    assert answer.metadata["guess_confirmed"] is True
    assert decision.is_complete is True
    assert decision.stop_reason == "character_guessed"
    assert decision.winner == "llm"
    assert state.data["successful_character_id"] == "marie_curie"
    assert game.public_state(state)["successful_character_name"] == "Marie Curie"


def test_round_limit_is_checked_only_after_human_answer() -> None:
    game = task(max_rounds=1)
    state = game.initial_state()

    _, state = advance(game, state, "ASK: Is this person a scientist?", "llm")
    assert game.flow_decision(state, []).is_complete is False
    _, state = advance(game, state, "no", "human")
    decision = game.flow_decision(state, [])

    assert decision.is_complete is True
    assert decision.stop_reason == "round_limit_reached"
    assert decision.winner == "human"
    assert state.data["logical_round"] == 1


def test_speech_and_display_render_only_valid_safe_actions() -> None:
    game = task()
    state = game.initial_state()

    assert (
        game.build_spoken_text(
            state,
            [],
            "ASK: Is this person a scientist?",
            actor="llm",
        )
        == "Is this person a scientist?"
    )
    assert (
        game.build_spoken_text(
            state,
            [],
            "GUESS: Newton",
            actor="llm",
        )
        == "Is your person Isaac Newton?"
    )
    assert (
        game.build_spoken_text(
            state,
            [],
            "Here is my reasoning",
            actor="llm",
        )
        is None
    )
    assert game.speech_vocabulary(state, max_turns=20) == ()
    speech = game.parse_spoken_response(state, [], "Yep.")
    assert speech.canonical_text == "yes"
    assert game.build_display_text(
        state,
        [],
        "GUESS: Newton",
        actor="llm",
    ) == "Is your person Isaac Newton?"
