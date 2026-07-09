"""Animal Naming task for generic realtime LLM benchmarking."""

from __future__ import annotations

import re
import string

from streammuse.domain.tasks.models import TaskRefereeResult, TaskState, TaskTurn


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


class AnimalNamingTask:
    name = "animal_naming"

    def __init__(self, *, animals: set[str] | frozenset[str] | None = None, max_history: int = 50) -> None:
        self.animals = frozenset(_normalize_animal(animal) for animal in (animals or DEFAULT_ANIMALS))
        self.max_history = int(max_history)

    def initial_state(self) -> TaskState:
        return TaskState(task_name=self.name, turn_index=0, data={"used_animals": [], "attempted_animals": []})

    def build_turn(self, state: TaskState) -> TaskTurn:
        used_animals = list(state.data.get("used_animals", []))
        attempted_animals = list(state.data.get("attempted_animals", used_animals))
        system_prompt = (
            "You are playing Animal Naming. Respond with exactly one common real animal name. "
            "Never output an animal listed as forbidden. Do not explain your answer."
        )
        if attempted_animals:
            user_prompt = (
                f"Forbidden animal names: {', '.join(attempted_animals)}\n"
                "Choose one animal name that is not in the forbidden list. Output only the new animal name:"
            )
        else:
            user_prompt = "Name an animal:"
        return TaskTurn(
            task_name=self.name,
            turn_id=state.turn_index,
            state=state,
            expected_output=None,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            metadata={"used_animals": used_animals, "attempted_animals": attempted_animals},
        )

    def validate_response(self, state: TaskState, response_text: str) -> TaskRefereeResult:
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
    ) -> TaskState:
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
                "role": "assistant",
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


def _normalize_animal(value: object) -> str:
    text = str(value or "").strip().lower()
    text = text.splitlines()[0] if text else ""
    text = text.strip(string.whitespace + string.punctuation)
    text = re.sub(r"^(an?|the)\s+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text
