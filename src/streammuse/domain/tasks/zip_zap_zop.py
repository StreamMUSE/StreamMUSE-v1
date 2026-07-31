"""ProjectIsochron-style Zip-Zap-Zop task implemented as a StreamMUSE task."""

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


_GAME_WORDS = ("zip", "zap", "zop")
_VALID_GAME_SEQUENCES = {
    ("zip",),
    ("zap",),
    ("zop",),
    ("zip", "zap"),
    ("zip", "zop"),
    ("zap", "zop"),
    ("zip", "zap", "zop"),
}
_SMALL_NUMBERS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
}
_TENS = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}
_NUMBER_WORDS = frozenset(_SMALL_NUMBERS) | frozenset(_TENS) | {
    "and",
    "hundred",
    "thousand",
    "million",
}
_MAX_SPOKEN_NUMBER = 999_999_999


class ZipZapZopTask:
    name = "zip_zap_zop"
    max_spoken_integer = _MAX_SPOKEN_NUMBER
    min_spoken_integer = -_MAX_SPOKEN_NUMBER

    def __init__(
        self,
        *,
        start_number: int = 1,
        max_history: int = 50,
        history_limit: int = 8,
        oracle_history: bool = False,
    ) -> None:
        self.start_number = int(start_number)
        self.history_limit = max(0, int(history_limit))
        self.oracle_history = bool(oracle_history)
        # state.history is what build_turn (run mode) reads back, so it must retain at least
        # history_limit entries — otherwise a large window would be silently truncated.
        self.max_history = max(int(max_history), self.history_limit)

    def initial_state(self) -> TaskState:
        return TaskState(
            task_name=self.name,
            turn_index=0,
            data={"current_number": self.start_number - 1, "start_number": self.start_number},
        )

    def build_turn(self, state: TaskState) -> TaskTurn:
        next_number = self.current_number(state)
        expected = self.expected_response(next_number)
        if self.history_limit <= 0 or not state.history:
            user_content = f"Current number: {next_number}\nAnswer:"
        else:
            recent = state.history[-self.history_limit :]
            history_lines = []
            for record in recent:
                label = "Human" if record.get("role") == "human" else "Assistant"
                turn_number = record.get("number", "?")
                history_lines.append(f"{label} at {turn_number}: {record.get('content', '')}")
            user_content = "Recent turns:\n" + "\n".join(history_lines) + f"\nCurrent number: {next_number}\nAnswer:"
        return TaskTurn(
            task_name=self.name,
            turn_id=state.turn_index,
            state=state,
            expected_output=expected,
            messages=[
                {"role": "system", "content": self.rules_prompt()},
                {"role": "user", "content": user_content},
            ],
            metadata={"number": next_number},
        )

    def build_human_prompt(self, state: TaskState, transcript: list[InteractiveTurnRecord]) -> str:
        _ = transcript
        return f"{self.current_number(state)}:"

    def build_speech_context(
        self,
        state: TaskState,
        transcript: list[InteractiveTurnRecord],
    ) -> SpeechContext:
        _ = state, transcript
        return SpeechContext(
            initial_prompt="Valid game words are Zip, Zap, and Zop. Numbers may also be spoken.",
            hotwords=("Zip", "Zap", "Zop"),
        )

    def parse_spoken_response(
        self,
        state: TaskState,
        transcript: list[InteractiveTurnRecord],
        raw_text: str,
    ) -> SpokenResponseParseResult:
        _ = state, transcript
        normalized = _normalize_transcript(raw_text)
        if not normalized:
            return SpokenResponseParseResult(
                canonical_text=None,
                status="empty",
                reason="empty_transcript",
            )

        tokens = tuple(normalized.split())
        if tokens in _VALID_GAME_SEQUENCES:
            return _successful_parse("".join(word.title() for word in tokens))
        if len(tokens) == 1:
            canonical_game_words = {
                "".join(sequence): "".join(word.title() for word in sequence)
                for sequence in _VALID_GAME_SEQUENCES
            }
            if normalized in canonical_game_words:
                return _successful_parse(canonical_game_words[normalized])
        if any(token in _GAME_WORDS for token in tokens):
            return _unrecognized_parse("invalid_game_word_sequence")

        if re.fullmatch(r"-?[0-9]+", normalized):
            negative = normalized.startswith("-")
            digits = normalized.removeprefix("-").lstrip("0") or "0"
            if len(digits) > len(str(_MAX_SPOKEN_NUMBER)):
                return _unrecognized_parse("number_out_of_range")
            number = int(digits)
            if number > _MAX_SPOKEN_NUMBER:
                return _unrecognized_parse("number_out_of_range")
            return _successful_parse(str(-number if negative and number else number))

        leading_hyphen = normalized.startswith("-")
        number_text = normalized[1:] if leading_hyphen else normalized
        number_tokens = tuple(number_text.replace("-", " ").split())
        word_sign = number_tokens[:1] in {("minus",), ("negative",)}
        negative = leading_hyphen or word_sign
        if word_sign:
            number_tokens = number_tokens[1:]
        number = _parse_english_number(number_tokens)
        if number is not None:
            return _successful_parse(str(-number if negative and number else number))
        if number_tokens and all(token in _NUMBER_WORDS for token in number_tokens):
            return _unrecognized_parse("invalid_number_phrase")
        return _unrecognized_parse("unrecognized_response")

    def build_spoken_text(
        self,
        state: TaskState,
        transcript: list[InteractiveTurnRecord],
        response_text: str,
        *,
        actor: InteractiveActor,
    ) -> str | None:
        parsed = self.parse_spoken_response(state, transcript, response_text)
        if parsed.status != "ok" or parsed.canonical_text is None:
            return None
        canonical = parsed.canonical_text
        if canonical.lstrip("-").isdigit():
            return canonical
        words = re.findall(r"Zip|Zap|Zop", canonical)
        if "".join(words).casefold() != canonical.casefold():
            return None
        return " ".join(words)

    def speech_vocabulary(
        self,
        state: TaskState,
        *,
        max_turns: int,
    ) -> tuple[str, ...]:
        phrases = [
            "Zip",
            "Zap",
            "Zop",
            "Zip Zap",
            "Zip Zop",
            "Zap Zop",
            "Zip Zap Zop",
        ]
        start_number = int(state.data.get("start_number", self.start_number))
        for number in range(start_number, start_number + max(0, int(max_turns))):
            if self.expected_response(number) == str(number):
                phrases.append(str(number))
        return tuple(dict.fromkeys(phrases))

    def build_llm_messages(
        self,
        state: TaskState,
        transcript: list[InteractiveTurnRecord],
    ) -> list[ChatMessage]:
        number = self.current_number(state)
        if self.history_limit <= 0 or not transcript:
            user_content = f"Current number: {number}\nAnswer:"
        else:
            recent = transcript[-self.history_limit :]
            history_lines = []
            for record in recent:
                label = "Human" if record.actor == "human" else "Assistant"
                turn_number = record.number if record.number is not None else "?"
                history_lines.append(f"{label} at {turn_number}: {record.response}")
            user_content = "Recent turns:\n" + "\n".join(history_lines) + f"\nCurrent number: {number}\nAnswer:"

        return [
            {"role": "system", "content": self.rules_prompt(with_human_context=True)},
            {"role": "user", "content": user_content},
        ]

    def build_hint(self, state: TaskState, transcript: list[InteractiveTurnRecord]) -> str | None:
        _ = transcript
        number = self.current_number(state)
        divisors = [str(divisor) for divisor in (3, 4, 5) if number % divisor == 0]
        if not divisors:
            return f"{number} is not divisible by 3, 4, or 5."
        return f"Check divisibility by: {', '.join(divisors)}."

    def expected_for_state(self, state: TaskState, transcript: list[InteractiveTurnRecord]) -> str | None:
        _ = transcript
        return self.expected_response(self.current_number(state))

    def validate_response(
        self,
        state: TaskState,
        response_text: str,
        *,
        actor: InteractiveActor | None = None,
        transcript: list[InteractiveTurnRecord] | None = None,
    ) -> TaskRefereeResult:
        _ = actor, transcript
        next_number = self.current_number(state)
        expected = self.expected_response(next_number)
        normalized = str(response_text or "").strip()
        is_valid = normalized.lower() == expected.lower()
        return TaskRefereeResult(
            is_valid=is_valid,
            expected_output=expected,
            failure_reason="NONE" if is_valid else "EXPECTED_MISMATCH",
            metadata={"number": next_number, "actual_output": normalized},
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
        next_number = self.current_number(state)
        role = "human" if actor == "human" else "assistant"
        history_content = str(response_text or "").strip()
        if self.oracle_history and referee_result.expected_output is not None:
            history_content = str(referee_result.expected_output).strip()
        history = list(state.history)
        history.append(
            {
                "role": role,
                "content": history_content,
                "valid": str(bool(referee_result.is_valid)),
                "expected": referee_result.expected_output or "",
                "number": str(next_number),
            }
        )
        if len(history) > self.max_history:
            history = history[-self.max_history :]
        return TaskState(
            task_name=self.name,
            turn_index=state.turn_index + 1,
            data={**state.data, "current_number": next_number},
            history=tuple(history),
        )

    def current_number(self, state: TaskState) -> int:
        return int(state.data.get("current_number", self.start_number - 1)) + 1

    @staticmethod
    def rules_prompt(*, with_human_context: bool = False) -> str:
        prefix = "You are playing Zip-Zap-Zop."
        if with_human_context:
            prefix = "You are playing Zip-Zap-Zop with a human."
        return (
            f"{prefix} Respond with only the value for the current number shown by the user. "
            "Multiples of 3 are Zip, multiples of 4 are Zap, multiples of 5 are Zop, "
            "and combined multiples concatenate those words. Otherwise respond with the number."
        )

    @staticmethod
    def expected_response(number: int) -> str:
        value = ""
        if number % 3 == 0:
            value += "Zip"
        if number % 4 == 0:
            value += "Zap"
        if number % 5 == 0:
            value += "Zop"
        return value or str(number)


def _normalize_transcript(raw_text: str) -> str:
    text = str(raw_text or "").strip()
    text = text.rstrip(string.punctuation + string.whitespace)
    return " ".join(text.casefold().split())


def _successful_parse(canonical_text: str) -> SpokenResponseParseResult:
    return SpokenResponseParseResult(canonical_text=canonical_text, status="ok")


def _unrecognized_parse(reason: str) -> SpokenResponseParseResult:
    return SpokenResponseParseResult(
        canonical_text=None,
        status="unrecognized",
        reason=reason,
    )


def _parse_english_number(tokens: tuple[str, ...]) -> int | None:
    if not tokens:
        return None
    if tokens == ("zero",):
        return 0
    if "zero" in tokens or any(token not in _NUMBER_WORDS for token in tokens):
        return None

    remaining = tokens
    total = 0
    used_scale = False
    for scale_word, scale_value in (("million", 1_000_000), ("thousand", 1_000)):
        occurrences = remaining.count(scale_word)
        if occurrences > 1:
            return None
        if occurrences == 0:
            continue
        scale_index = remaining.index(scale_word)
        group = _parse_under_one_thousand(remaining[:scale_index])
        if group is None or group == 0:
            return None
        total += group * scale_value
        used_scale = True
        remaining = remaining[scale_index + 1 :]
        if remaining[:1] == ("and",):
            remaining = remaining[1:]
            if not remaining:
                return None

    if not remaining:
        return total if used_scale else None
    remainder = _parse_under_one_thousand(remaining)
    if remainder is None or (used_scale and remainder == 0):
        return None
    return total + remainder


def _parse_under_one_thousand(tokens: tuple[str, ...]) -> int | None:
    if not tokens:
        return None
    if tokens == ("zero",):
        return 0
    if tokens.count("hundred") > 1:
        return None
    if "hundred" not in tokens:
        if "and" in tokens:
            return None
        return _parse_under_one_hundred(tokens)

    hundred_index = tokens.index("hundred")
    if hundred_index != 1 or tokens[0] not in _SMALL_NUMBERS or not 1 <= _SMALL_NUMBERS[tokens[0]] <= 9:
        return None
    value = _SMALL_NUMBERS[tokens[0]] * 100
    remainder = tokens[2:]
    if not remainder:
        return value
    if remainder[:1] == ("and",):
        remainder = remainder[1:]
        if not remainder:
            return None
    elif "and" in remainder:
        return None
    tail = _parse_under_one_hundred(remainder)
    if tail is None or tail == 0:
        return None
    return value + tail


def _parse_under_one_hundred(tokens: tuple[str, ...]) -> int | None:
    if len(tokens) == 1:
        return _SMALL_NUMBERS.get(tokens[0], _TENS.get(tokens[0]))
    if len(tokens) == 2 and tokens[0] in _TENS:
        unit = _SMALL_NUMBERS.get(tokens[1])
        if unit is not None and 1 <= unit <= 9:
            return _TENS[tokens[0]] + unit
    return None
