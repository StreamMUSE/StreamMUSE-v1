"""Human-oracle, round-based Character Guessing game."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from streammuse.domain.tasks.character_scopes import (
    CharacterScope,
    canonicalize_character_name,
)
from streammuse.domain.tasks.models import (
    ChatMessage,
    InteractiveActor,
    InteractiveFlowDecision,
    InteractiveTaskSetup,
    InteractiveTurnRecord,
    SpeechContext,
    SpokenResponseParseResult,
    TaskRefereeResult,
    TaskState,
)


CharacterActionKind = Literal["ask", "guess"]
_QUESTION_START = re.compile(
    r"(?:is|was|were|did|does|do|has|have|had|can|could|would|will)\b",
    re.IGNORECASE,
)
_YES_ANSWERS = frozenset(
    {"yes", "yeah", "yep", "correct", "right", "that's right"}
)
_NO_ANSWERS = frozenset(
    {"no", "nope", "incorrect", "wrong", "that's wrong"}
)


@dataclass(frozen=True)
class CharacterAction:
    kind: CharacterActionKind
    text: str
    canonical_character_id: str | None = None
    display_name: str | None = None


def normalize_yes_no(value: object) -> str | None:
    text = str(value or "").strip().lower()
    if "\n" in text or "\r" in text:
        return None
    text = text.strip(" \t.!?,;:")
    text = re.sub(r"\s+", " ", text)
    if text in _YES_ANSWERS:
        return "yes"
    if text in _NO_ANSWERS:
        return "no"
    return None


def _question_key(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower().rstrip("?")).strip()


def _question_mentions_candidate(
    scope: CharacterScope,
    question: str,
) -> bool:
    normalized_question = f" {canonicalize_character_name(question)} "
    return any(
        f" {canonicalize_character_name(name)} " in normalized_question
        for candidate in scope.candidates
        for name in (candidate.display_name, *candidate.aliases)
    )


class CharacterGuessingTask:
    name = "character_guessing"

    def __init__(self, *, scope: CharacterScope, max_rounds: int = 20) -> None:
        if int(max_rounds) <= 0:
            raise ValueError("max_rounds must be > 0")
        self.scope = scope
        self.max_rounds = int(max_rounds)

    def initial_state(self) -> TaskState:
        return TaskState(
            task_name=self.name,
            data={
                "phase": "awaiting_llm",
                "scope_id": self.scope.scope_id,
                "scope_title": self.scope.title,
                "candidate_ids": list(self.scope.candidate_ids),
                "max_rounds": self.max_rounds,
                "logical_round": 0,
                "pending_action": None,
                "rounds": [],
                "asked_question_keys": [],
                "guessed_character_ids": [],
                "invalid_attempts_in_phase": 0,
                "last_invalid_reason": None,
                "completed": False,
                "stop_reason": None,
                "winner": None,
                "successful_character_id": None,
            },
        )

    def build_setup(self, state: TaskState) -> InteractiveTaskSetup:
        _ = state
        return InteractiveTaskSetup(
            title=self.scope.title,
            prompt=(
                "Choose one person privately. Do not type the name. "
                "Press Enter when ready."
            ),
            choices=tuple(
                candidate.display_name for candidate in self.scope.candidates
            ),
        )

    def public_state(self, state: TaskState) -> dict[str, object]:
        pending = state.data.get("pending_action")
        rounds = list(state.data.get("rounds", []))
        guesses = [
            item for item in rounds if item.get("action_kind") == "guess"
        ]
        successful = state.data.get("successful_character_id")
        match = next(
            (
                item
                for item in self.scope.candidates
                if item.character_id == successful
            ),
            None,
        )
        successful_name = match.display_name if match is not None else None
        logical_round = int(state.data.get("logical_round", 0))
        return {
            "scope_id": self.scope.scope_id,
            "scope_title": self.scope.title,
            "scope_size": len(self.scope.candidates),
            "candidates": [
                candidate.display_name for candidate in self.scope.candidates
            ],
            "phase": state.data.get("phase"),
            "logical_round": logical_round,
            "current_round": min(logical_round + 1, self.max_rounds),
            "max_rounds": self.max_rounds,
            "pending_action": (
                dict(pending) if isinstance(pending, dict) else None
            ),
            "question_count": sum(
                item.get("action_kind") == "ask" for item in rounds
            ),
            "guess_count": len(guesses),
            "rejected_guess_count": sum(
                item.get("answer") == "no" for item in guesses
            ),
            "completed": bool(state.data.get("completed")),
            "stop_reason": state.data.get("stop_reason"),
            "winner": state.data.get("winner"),
            "successful_character_id": successful,
            "successful_character_name": successful_name,
        }

    def flow_decision(
        self,
        state: TaskState,
        transcript: list[InteractiveTurnRecord],
    ) -> InteractiveFlowDecision:
        _ = transcript
        complete = bool(state.data.get("completed"))
        next_actor: InteractiveActor | None = None
        if not complete:
            next_actor = (
                "llm"
                if state.data.get("phase") == "awaiting_llm"
                else "human"
            )
        winner = state.data.get("winner") if complete else None
        loser: InteractiveActor | None = None
        if winner == "llm":
            loser = "human"
        elif winner == "human":
            loser = "llm"
        return InteractiveFlowDecision(
            next_actor=next_actor,
            logical_round=int(state.data.get("logical_round", 0)),
            is_complete=complete,
            stop_reason=(
                str(state.data.get("stop_reason")) if complete else None
            ),
            winner=winner,
            loser=loser,
            metadata=self.public_state(state),
        )

    def build_human_prompt(
        self,
        state: TaskState,
        transcript: list[InteractiveTurnRecord],
    ) -> str:
        _ = transcript
        pending = state.data.get("pending_action")
        if not isinstance(pending, dict):
            return "Yes or no:"
        return f"{pending.get('display_text')} Yes or no:"

    def build_llm_messages(
        self,
        state: TaskState,
        transcript: list[InteractiveTurnRecord],
    ) -> list[ChatMessage]:
        _ = transcript
        system = (
            "You are playing Character Guessing. The human privately selected "
            "exactly one person from the supplied candidate list. Output exactly "
            "one line in one of these forms: ASK: <one yes-or-no question ending "
            "in ?> or GUESS: <one candidate name>. Do not output reasoning, "
            "markdown, numbering, or multiple actions. Ask one simple question at "
            "a time, avoid repeats, and use GUESS when confident. ASK must be "
            "about an attribute and must not contain any candidate name or alias; "
            "use GUESS whenever naming a candidate. Human answers are the only "
            "facts for this game."
        )
        rounds = list(state.data.get("rounds", []))
        history = "None"
        if rounds:
            history = "\n".join(
                f"{index}. {str(item['action_kind']).upper()} "
                f"{item['action_text']!r} -> {str(item['answer']).upper()}"
                for index, item in enumerate(rounds, start=1)
            )
        rejected = [
            str(item.get("display_name"))
            for item in rounds
            if item.get("action_kind") == "guess"
            and item.get("answer") == "no"
        ]
        remaining = self.max_rounds - int(
            state.data.get("logical_round", 0)
        )
        repair = ""
        if state.data.get("last_invalid_reason"):
            repair = (
                "\nYour previous attempt was rejected: "
                f"{state.data['last_invalid_reason']}. "
                "Follow the one-line protocol exactly."
            )
        user = (
            f"Scope: {self.scope.title}\n"
            f"Candidates: "
            f"{', '.join(item.display_name for item in self.scope.candidates)}\n"
            f"Completed rounds:\n{history}\n"
            f"Rejected guesses: "
            f"{', '.join(rejected) if rejected else 'None'}\n"
            f"Rounds remaining: {remaining}{repair}\n"
            "Next action:"
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def parse_action(
        self,
        state: TaskState,
        response_text: str,
    ) -> CharacterAction:
        raw = str(response_text or "").strip()
        if not raw:
            raise ValueError("EMPTY_RESPONSE")
        if "\n" in raw or "\r" in raw:
            raise ValueError("MULTILINE_ACTION")
        match = re.fullmatch(
            r"(ASK|GUESS)\s*:\s*(.+)",
            raw,
            re.IGNORECASE,
        )
        if match is None:
            raise ValueError("INVALID_ACTION_FORMAT")
        tag, body = match.group(1).lower(), match.group(2).strip()
        if re.search(r"\b(?:ASK|GUESS)\s*:", body, re.IGNORECASE):
            raise ValueError("MULTIPLE_ACTIONS")
        if tag == "ask":
            if len(body) > 180:
                raise ValueError("QUESTION_TOO_LONG")
            if body.count("?") != 1 or not body.endswith("?"):
                raise ValueError("NOT_YES_NO_QUESTION")
            if _QUESTION_START.match(body) is None:
                raise ValueError("NOT_YES_NO_QUESTION")
            if _question_mentions_candidate(self.scope, body):
                raise ValueError("CANDIDATE_NAME_REQUIRES_GUESS")
            if _question_key(body) in set(
                state.data.get("asked_question_keys", [])
            ):
                raise ValueError("REPEATED_QUESTION")
            return CharacterAction("ask", body)

        candidate = self.scope.resolve(body)
        if candidate is None:
            raise ValueError("CHARACTER_OUT_OF_SCOPE")
        if candidate.character_id in set(
            state.data.get("guessed_character_ids", [])
        ):
            raise ValueError("REPEATED_GUESS")
        return CharacterAction(
            "guess",
            candidate.display_name,
            canonical_character_id=candidate.character_id,
            display_name=candidate.display_name,
        )

    def validate_response(
        self,
        state: TaskState,
        response_text: str,
        *,
        actor: InteractiveActor | None = None,
        transcript: list[InteractiveTurnRecord] | None = None,
    ) -> TaskRefereeResult:
        _ = transcript
        phase = str(state.data.get("phase"))
        expected_actor = (
            "llm" if phase == "awaiting_llm" else "human"
        )
        if actor != expected_actor:
            return TaskRefereeResult(
                False,
                failure_reason="WRONG_ACTOR",
                metadata={"phase": phase},
            )
        if actor == "llm":
            try:
                action = self.parse_action(state, response_text)
            except ValueError as exc:
                return TaskRefereeResult(
                    False,
                    failure_reason=str(exc),
                    metadata={"phase": phase},
                )
            display_text = (
                action.text
                if action.kind == "ask"
                else f"Is your person {action.display_name}?"
            )
            return TaskRefereeResult(
                True,
                failure_reason="NONE",
                metadata={
                    "phase": phase,
                    "action_kind": action.kind,
                    "action_text": action.text,
                    "canonical_character_id": (
                        action.canonical_character_id
                    ),
                    "display_name": action.display_name,
                    "display_text": display_text,
                    "round_completed": False,
                },
            )

        canonical = normalize_yes_no(response_text)
        if canonical is None:
            return TaskRefereeResult(
                False,
                expected_output="yes or no",
                failure_reason="INVALID_YES_NO",
                metadata={"phase": phase},
            )
        pending = state.data.get("pending_action")
        if not isinstance(pending, dict):
            return TaskRefereeResult(
                False,
                failure_reason="MISSING_PENDING_ACTION",
            )
        return TaskRefereeResult(
            True,
            expected_output="yes or no",
            failure_reason="NONE",
            metadata={
                "phase": phase,
                "canonical_answer": canonical,
                "action_kind": pending.get("kind"),
                "canonical_character_id": (
                    pending.get("canonical_character_id")
                ),
                "display_text": canonical,
                "round_completed": True,
                "guess_confirmed": (
                    pending.get("kind") == "guess" and canonical == "yes"
                ),
            },
        )

    def advance_state(
        self,
        state: TaskState,
        referee_result: TaskRefereeResult,
        response_text: str,
        *,
        actor: InteractiveActor | None = None,
        transcript: list[InteractiveTurnRecord] | None = None,
    ) -> TaskState:
        _ = response_text, transcript
        data = dict(state.data)
        if not referee_result.is_valid:
            data["invalid_attempts_in_phase"] = (
                int(data.get("invalid_attempts_in_phase", 0)) + 1
            )
            data["last_invalid_reason"] = referee_result.failure_reason
            return TaskState(
                self.name,
                state.turn_index + 1,
                data,
                state.history,
            )

        data["invalid_attempts_in_phase"] = 0
        data["last_invalid_reason"] = None
        metadata = referee_result.metadata
        if actor == "llm":
            pending = {
                "kind": metadata.get("action_kind"),
                "text": metadata.get("action_text"),
                "canonical_character_id": (
                    metadata.get("canonical_character_id")
                ),
                "display_name": metadata.get("display_name"),
                "display_text": metadata.get("display_text"),
            }
            data["pending_action"] = pending
            data["phase"] = "awaiting_human"
            if pending["kind"] == "ask":
                asked = list(data.get("asked_question_keys", []))
                asked.append(_question_key(str(pending["text"])))
                data["asked_question_keys"] = asked
            else:
                guessed = list(data.get("guessed_character_ids", []))
                guessed.append(str(pending["canonical_character_id"]))
                data["guessed_character_ids"] = guessed
        else:
            pending = dict(data.get("pending_action") or {})
            answer = str(metadata.get("canonical_answer"))
            round_number = int(data.get("logical_round", 0)) + 1
            rounds = list(data.get("rounds", []))
            rounds.append(
                {
                    "round": round_number,
                    "action_kind": pending.get("kind"),
                    "action_text": pending.get("text"),
                    "canonical_character_id": (
                        pending.get("canonical_character_id")
                    ),
                    "display_name": pending.get("display_name"),
                    "answer": answer,
                }
            )
            data["rounds"] = rounds
            data["logical_round"] = round_number
            data["pending_action"] = None
            guessed = (
                pending.get("kind") == "guess" and answer == "yes"
            )
            if guessed:
                data.update(
                    {
                        "phase": "finished",
                        "completed": True,
                        "stop_reason": "character_guessed",
                        "winner": "llm",
                        "successful_character_id": (
                            pending.get("canonical_character_id")
                        ),
                    }
                )
            elif round_number >= self.max_rounds:
                data.update(
                    {
                        "phase": "finished",
                        "completed": True,
                        "stop_reason": "round_limit_reached",
                        "winner": "human",
                    }
                )
            else:
                data["phase"] = "awaiting_llm"

        return TaskState(
            self.name,
            state.turn_index + 1,
            data,
            state.history,
        )

    def expected_for_state(
        self,
        state: TaskState,
        transcript: list[InteractiveTurnRecord],
    ) -> str | None:
        _ = transcript
        if state.data.get("phase") == "awaiting_human":
            return "yes or no"
        return "ASK: <question?> or GUESS: <candidate>"

    def build_hint(
        self,
        state: TaskState,
        transcript: list[InteractiveTurnRecord],
    ) -> str | None:
        _ = transcript
        if state.data.get("phase") == "awaiting_human":
            return "Answer only yes or no based on the person you chose."
        return (
            "Ask one discriminating yes-or-no question, or guess one listed "
            "candidate."
        )

    def build_speech_context(
        self,
        state: TaskState,
        transcript: list[InteractiveTurnRecord],
    ) -> SpeechContext:
        _ = state, transcript
        return SpeechContext(
            initial_prompt=(
                "The speaker will answer a yes-or-no question using a short "
                "affirmative or negative phrase."
            ),
            hotwords=(
                "yes",
                "no",
                "yeah",
                "yep",
                "nope",
                "correct",
                "incorrect",
                "right",
                "wrong",
            ),
        )

    def parse_spoken_response(
        self,
        state: TaskState,
        transcript: list[InteractiveTurnRecord],
        raw_text: str,
    ) -> SpokenResponseParseResult:
        _ = state, transcript
        canonical = normalize_yes_no(raw_text)
        if canonical is not None:
            return SpokenResponseParseResult(canonical, "ok")
        if not str(raw_text or "").strip():
            return SpokenResponseParseResult(
                None,
                "empty",
                "empty_transcript",
            )
        return SpokenResponseParseResult(
            None,
            "unrecognized",
            "invalid_yes_no",
        )

    def build_spoken_text(
        self,
        state: TaskState,
        transcript: list[InteractiveTurnRecord],
        response_text: str,
        *,
        actor: InteractiveActor,
    ) -> str | None:
        if actor != "llm":
            return None
        try:
            action = self.parse_action(state, response_text)
        except ValueError:
            return None
        if action.kind == "ask":
            return action.text
        return f"Is your person {action.display_name}?"

    def build_display_text(
        self,
        state: TaskState,
        transcript: list[InteractiveTurnRecord],
        response_text: str,
        *,
        actor: InteractiveActor,
    ) -> str:
        if actor == "human":
            return normalize_yes_no(response_text) or "[unrecognized]"
        spoken = self.build_spoken_text(
            state,
            transcript,
            response_text,
            actor=actor,
        )
        return spoken or "[invalid model response]"

    def speech_vocabulary(
        self,
        state: TaskState,
        *,
        max_turns: int,
    ) -> tuple[str, ...]:
        _ = state, max_turns
        return ()
