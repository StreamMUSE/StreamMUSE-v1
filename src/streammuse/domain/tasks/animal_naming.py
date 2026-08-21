"""Animal Naming task for generic realtime LLM benchmarking."""

from __future__ import annotations

import re
import string

from streammuse.domain.tasks.models import (
    ChatMessage,
    InteractiveActor,
    InteractiveTurnRecord,
    SpeechContext,
    SpokenResponseParseResult,
    TaskRefereeResult,
    TaskState,
    TaskTurn,
)


DEFAULT_ANIMALS = frozenset(
    {
        "aardvark",
        "alligator",
        "ant",
        "anteater",
        "antelope",
        "ape",
        "armadillo",
        "baboon",
        "badger",
        "bat",
        "bear",
        "beaver",
        "bee",
        "bird",
        "bison",
        "boar",
        "buffalo",
        "butterfly",
        "camel",
        "cat",
        "cheetah",
        "chicken",
        "chimpanzee",
        "cobra",
        "cougar",
        "cow",
        "coyote",
        "crab",
        "crocodile",
        "deer",
        "dog",
        "dolphin",
        "duck",
        "eagle",
        "elephant",
        "falcon",
        "fish",
        "flamingo",
        "fox",
        "frog",
        "giraffe",
        "goat",
        "goose",
        "gorilla",
        "hamster",
        "hawk",
        "hippopotamus",
        "horse",
        "hyena",
        "jaguar",
        "kangaroo",
        "koala",
        "lemur",
        "leopard",
        "lion",
        "lizard",
        "llama",
        "lobster",
        "monkey",
        "moose",
        "mouse",
        "octopus",
        "otter",
        "owl",
        "panda",
        "panther",
        "parrot",
        "penguin",
        "pig",
        "rabbit",
        "raccoon",
        "rat",
        "raven",
        "rhino",
        "rhinoceros",
        "salmon",
        "seal",
        "shark",
        "sheep",
        "skunk",
        "sloth",
        "snake",
        "spider",
        "squid",
        "squirrel",
        "tiger",
        "turtle",
        "walrus",
        "whale",
        "wolf",
        "zebra",
    }
)

_SPOKEN_FILLER_WORDS = frozenset(
    {
        "answer",
        "and",
        "animal",
        "choose",
        "guess",
        "i",
        "is",
        "like",
        "maybe",
        "my",
        "say",
        "think",
        "would",
    }
)


class AnimalNamingTask:
    name = "animal_naming"

    def __init__(self, *, animals: set[str] | frozenset[str] | None = None, max_history: int = 50) -> None:
        self.animals = frozenset(_normalize_animal(animal) for animal in (animals or DEFAULT_ANIMALS))
        self.max_history = int(max_history)

    def initial_state(self) -> TaskState:
        return TaskState(
            task_name=self.name,
            turn_index=0,
            data={"used_animals": [], "attempted_animals": []},
        )

    def build_turn(self, state: TaskState) -> TaskTurn:
        used_animals = list(state.data.get("used_animals", []))
        attempted_animals = list(state.data.get("attempted_animals", used_animals))
        return TaskTurn(
            task_name=self.name,
            turn_id=state.turn_index,
            state=state,
            expected_output=None,
            messages=self._messages(
                attempted_animals,
                with_human_context=False,
            ),
            metadata={"used_animals": used_animals, "attempted_animals": attempted_animals},
        )

    def build_human_prompt(
        self,
        state: TaskState,
        transcript: list[InteractiveTurnRecord],
    ) -> str:
        _ = state, transcript
        return "Name one unused animal:"

    def build_llm_messages(
        self,
        state: TaskState,
        transcript: list[InteractiveTurnRecord],
    ) -> list[ChatMessage]:
        _ = transcript
        used_animals = list(state.data.get("used_animals", []))
        attempted_animals = list(state.data.get("attempted_animals", used_animals))
        return self._messages(
            attempted_animals,
            with_human_context=True,
            allowed_animals=self.remaining_animals(state),
        )

    def build_speech_context(
        self,
        state: TaskState,
        transcript: list[InteractiveTurnRecord],
    ) -> SpeechContext:
        _ = transcript
        return SpeechContext(
            initial_prompt="The speaker will say exactly one common English animal name.",
            hotwords=self.remaining_animals(state),
        )

    def parse_spoken_response(
        self,
        state: TaskState,
        transcript: list[InteractiveTurnRecord],
        raw_text: str,
    ) -> SpokenResponseParseResult:
        _ = state, transcript
        normalized = _normalize_animal(raw_text)
        if not normalized:
            return SpokenResponseParseResult(
                canonical_text=None,
                status="empty",
                reason="empty_transcript",
            )
        if not re.fullmatch(r"[a-z]+(?: [a-z]+){0,2}", normalized):
            return SpokenResponseParseResult(
                canonical_text=None,
                status="unrecognized",
                reason="invalid_animal_phrase",
            )
        tokens = normalized.split()
        if len(set(tokens)) != len(tokens):
            return SpokenResponseParseResult(
                canonical_text=None,
                status="unrecognized",
                reason="repeated_speech_tokens",
            )
        if any(token in _SPOKEN_FILLER_WORDS for token in tokens):
            return SpokenResponseParseResult(
                canonical_text=None,
                status="unrecognized",
                reason="explanatory_speech",
            )
        return SpokenResponseParseResult(canonical_text=normalized, status="ok")

    def build_spoken_text(
        self,
        state: TaskState,
        transcript: list[InteractiveTurnRecord],
        response_text: str,
        *,
        actor: InteractiveActor,
    ) -> str | None:
        _ = actor
        parsed = self.parse_spoken_response(state, transcript, response_text)
        if parsed.status != "ok" or parsed.canonical_text is None:
            return None
        return parsed.canonical_text

    def speech_vocabulary(
        self,
        state: TaskState,
        *,
        max_turns: int,
    ) -> tuple[str, ...]:
        _ = state, max_turns
        # Each valid animal is used at most once per game, so eager synthesis has
        # high startup cost and little within-session reuse.
        return ()

    def expected_for_state(
        self,
        state: TaskState,
        transcript: list[InteractiveTurnRecord],
    ) -> str | None:
        _ = state, transcript
        return None

    def build_hint(
        self,
        state: TaskState,
        transcript: list[InteractiveTurnRecord],
    ) -> str | None:
        _ = transcript
        remaining_count = len(self.remaining_animals(state))
        return (
            f"{remaining_count} unused animal names remain. "
            "Name one animal that has not appeared before."
        )

    def validate_response(
        self,
        state: TaskState,
        response_text: str,
        *,
        actor: InteractiveActor | None = None,
        transcript: list[InteractiveTurnRecord] | None = None,
    ) -> TaskRefereeResult:
        _ = actor, transcript
        normalized = _normalize_animal(response_text)
        used_animals = set(state.data.get("used_animals", []))
        if not normalized:
            return TaskRefereeResult(
                is_valid=False,
                failure_reason="EMPTY_RESPONSE",
                metadata={"normalized_animal": normalized},
            )
        if normalized not in self.animals:
            return TaskRefereeResult(
                is_valid=False,
                failure_reason="UNKNOWN_ANIMAL",
                metadata={"normalized_animal": normalized},
            )
        if normalized in used_animals:
            return TaskRefereeResult(
                is_valid=False,
                failure_reason="REPEATED_ANIMAL",
                metadata={"normalized_animal": normalized},
            )
        return TaskRefereeResult(
            is_valid=True,
            failure_reason="NONE",
            metadata={"normalized_animal": normalized},
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
        _ = transcript
        normalized = str(referee_result.metadata.get("normalized_animal") or _normalize_animal(response_text))
        used_animals = list(state.data.get("used_animals", []))
        attempted_animals = list(state.data.get("attempted_animals", used_animals))
        if referee_result.is_valid and normalized not in used_animals:
            used_animals.append(normalized)
        if normalized and normalized not in attempted_animals:
            attempted_animals.append(normalized)

        history = list(state.history)
        history.append(
            {
                "role": "human" if actor == "human" else "assistant",
                "content": str(response_text or "").strip(),
                "valid": str(bool(referee_result.is_valid)),
                "normalized_animal": normalized,
            }
        )
        if len(history) > self.max_history:
            history = history[-self.max_history :]
        return TaskState(
            task_name=self.name,
            turn_index=state.turn_index + 1,
            data={**state.data, "used_animals": used_animals, "attempted_animals": attempted_animals},
            history=tuple(history),
        )

    def remaining_animals(self, state: TaskState) -> tuple[str, ...]:
        used_animals = {
            _normalize_animal(animal)
            for animal in state.data.get("used_animals", [])
        }
        return tuple(sorted(self.animals - used_animals))

    def _messages(
        self,
        attempted_animals: list[str],
        *,
        with_human_context: bool,
        allowed_animals: tuple[str, ...] = (),
    ) -> list[ChatMessage]:
        if with_human_context:
            system_prompt = (
                "You are playing Animal Naming with a human. "
                "Respond with exactly one common real English animal name. "
                "Never output an animal listed as forbidden. "
                "Do not explain, number, or add commentary to your answer."
            )
        else:
            system_prompt = (
                "You are playing Animal Naming. Respond with exactly one common real animal name. "
                "Never output an animal listed as forbidden. Do not explain your answer."
            )
        if with_human_context:
            allowed_prompt = f"Allowed animal names: {', '.join(allowed_animals)}\n"
            forbidden_prompt = (
                f"Forbidden animal names: {', '.join(attempted_animals)}\n"
                if attempted_animals
                else ""
            )
            user_prompt = (
                f"{allowed_prompt}{forbidden_prompt}"
                "Choose one allowed animal that is not forbidden. Output only the animal name:"
            )
        elif attempted_animals:
            user_prompt = (
                f"Forbidden animal names: {', '.join(attempted_animals)}\n"
                "Choose one animal name that is not in the forbidden list. Output only the new animal name:"
            )
        else:
            user_prompt = "Name an animal:"
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]


def _normalize_animal(value: object) -> str:
    text = str(value or "").strip().lower()
    text = text.splitlines()[0] if text else ""
    text = text.strip(string.whitespace + string.punctuation)
    text = re.sub(r"^(an?|the)\s+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text
