"""Tests for offline and local-chat lyric candidate adapters."""

import json

import pytest
import requests

from streammuse.domain.rap import CandidateBatch, CandidateRequest, FlowTemplate
from streammuse.domain.tasks import ChatModelResponse
from streammuse.infrastructure.inference.local_chat_client import (
    LocalChatChoice,
    LocalChatChoicesResponse,
)
from streammuse.infrastructure.rap import generators as generator_module
from streammuse.infrastructure.rap.generators import (
    IndependentChoiceCandidateGenerator,
    LocalChatCandidateGenerator,
    PhraseBankGenerator,
    _sanitize_error,
)
from streammuse.infrastructure.rap.prosody import CmuProsodyAnalyzer
from streammuse.infrastructure.rap.templates import BUILTIN_TEMPLATES


class FailingClient:
    def generate(self, *args, **kwargs):
        raise RuntimeError(
            "connection refused; Authorization: Bearer super-secret-token"
        )


class FakeClient:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def generate(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return ChatModelResponse(
            text=self.text,
            latency_ms=12.5,
            prompt_tokens=21,
            completion_tokens=13,
            raw={"id": "chatcmpl-123"},
        )


class ResponseClient:
    def __init__(self, response: object) -> None:
        self.response = response

    def generate(self, *args, **kwargs) -> object:
        return self.response


class RaisingLatencyResponse:
    text = "a valid line"
    prompt_tokens = 7
    completion_tokens = 3

    @property
    def latency_ms(self) -> float:
        raise RuntimeError("diagnostic clock unavailable")


class RaisingPromptTokensResponse:
    text = "a valid line"
    latency_ms = 8.0
    completion_tokens = 3

    @property
    def prompt_tokens(self) -> int:
        raise RuntimeError("prompt token counter unavailable")


class FakeChoicesClient:
    def __init__(
        self, choices: tuple[str, ...], *, warnings: tuple[str, ...] = ()
    ) -> None:
        self.choices = choices
        self.warnings = warnings
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def generate_choices(self, *args, **kwargs) -> LocalChatChoicesResponse:
        self.calls.append((args, kwargs))
        return LocalChatChoicesResponse(
            choices=tuple(
                LocalChatChoice(index=index, text=text)
                for index, text in enumerate(self.choices)
            ),
            latency_ms=17.5,
            prompt_tokens=31,
            completion_tokens=22,
            raw={"id": "choices-1"},
            warnings=self.warnings,
        )


def request_for_bar(
    bar: int = 2, *, flow_template: FlowTemplate | None = None
) -> CandidateRequest:
    return CandidateRequest(
        request_id=f"bar-{bar}-request-1",
        target_bar=bar,
        topic="space travel",
        flow_template=flow_template or BUILTIN_TEMPLATES.get("baseline_syncopated_9"),
        count=4,
        context_lines=("stars cross the night",),
        seed=20260807,
    )


def test_phrase_bank_normalizes_empty_topic_and_returns_requested_candidates() -> None:
    request = CandidateRequest(
        request_id="phrase-bank-1",
        target_bar=0,
        topic="!!!",
        flow_template=BUILTIN_TEMPLATES.get("baseline_syncopated_9"),
        count=3,
        context_lines=(),
        seed=20260807,
    )
    batch = PhraseBankGenerator().generate(request)

    assert batch.source == "phrase_bank"
    assert batch.request_id == "phrase-bank-1"
    assert batch.prompt == ()
    assert batch.raw_response == ""
    assert batch.latency_ms == 0.0
    assert len(batch.candidates) == 3
    assert all("the moment" in line for line in batch.candidates)


@pytest.mark.parametrize("topic", ("space", "deep sea", "code", "space travel"))
def test_phrase_bank_candidates_match_requested_flow_syllable_count(topic: str) -> None:
    request = CandidateRequest(
        request_id="phrase-bank-exact-flow",
        target_bar=1,
        topic=topic,
        flow_template=BUILTIN_TEMPLATES.get("baseline_syncopated_9"),
        count=8,
        context_lines=(),
        seed=20260807,
    )

    batch = PhraseBankGenerator().generate(request)
    analyzer = CmuProsodyAnalyzer()

    assert len(batch.candidates) == request.count
    assert len(set(batch.candidates)) == request.count
    assert all(topic in line for line in batch.candidates)
    assert {
        len(analyzer.analyze(candidate).syllables) for candidate in batch.candidates
    } == {request.required_syllables}


def test_local_chat_request_preserves_structure_history_and_raw_diagnostics() -> None:
    client = FakeClient(
        "1. Space travel keeps the whole night bright\n2. We move through stars with rhythm"
    )
    request = request_for_bar()
    batch = LocalChatCandidateGenerator(client).generate(request)

    assert batch.source == "local_chat"
    assert batch.request_id == request.request_id
    assert "exactly 9 spoken syllables" in batch.prompt[-1]["content"]
    assert "stars cross the night" in batch.prompt[-1]["content"]
    assert (
        batch.raw_response
        == "1. Space travel keeps the whole night bright\n2. We move through stars with rhythm"
    )
    assert batch.latency_ms == 12.5
    assert batch.prompt_tokens == 21
    assert batch.completion_tokens == 13
    assert batch.warning == "requested_4_received_2"
    assert batch.candidates == (
        "Space travel keeps the whole night bright",
        "We move through stars with rhythm",
    )
    assert client.calls[0][1]["max_tokens"] > 0


def test_local_chat_prompt_contains_actual_flow_not_only_template_id() -> None:
    request = request_for_bar(
        flow_template=BUILTIN_TEMPLATES.get("baseline_syncopated_9")
    )
    client = FakeClient("one line")

    LocalChatCandidateGenerator(client).generate(request)

    user = client.calls[0][0][0][1]["content"]
    assert "Syllable ticks: [0, 2, 3, 5, 7, 8, 10, 13, 15]" in user
    assert "Target stress: [1.0, 0.2, 0.7, 0.2, 0.6, 1.0, 0.2, 0.7, 0.9]" in user
    assert "Boundary strengths: [0, 0, 0, 0, 0, 0, 0, 0, 3]" in user
    assert 'Rhyme groups: [null, null, null, null, null, null, null, null, "A"]' in user
    assert "S . w M | . w . M | S . w . | . M . S" in user
    assert "plain lyric lines without syllable markup" in user


def test_local_chat_prompt_requires_internal_drafting_and_spoken_count_verification() -> (
    None
):
    client = FakeClient("one line")

    LocalChatCandidateGenerator(client).generate(request_for_bar())

    prompt = client.calls[0][0][0]
    assert "pronunciation-aware prosody checker" in prompt[0]["content"]
    user = prompt[1]["content"]
    draft = user.index("draft extra lines")
    spoken_count = user.index(
        "count every line using normal American spoken pronunciation"
    )
    reject = user.index("Silently discard or rewrite every line")
    contractions = user.index("Contractions count as spoken")
    spelling = user.index("do not rely on spelling")
    assert draft < spoken_count < reject < contractions < spelling


def test_local_chat_error_returns_explicit_empty_batch_without_phrase_bank_fallback() -> (
    None
):
    batch = LocalChatCandidateGenerator(FailingClient()).generate(request_for_bar())

    assert batch.source == "local_chat"
    assert batch.candidates == ()
    assert batch.error_type == "generation_error"
    assert "connection refused" in (batch.error_message or "")
    assert "super-secret-token" not in (batch.error_message or "")


def test_local_chat_empty_response_is_an_explicit_generation_error() -> None:
    batch = LocalChatCandidateGenerator(FakeClient("  \n ")).generate(request_for_bar())

    assert batch.candidates == ()
    assert batch.error_type == "generation_error"
    assert (
        batch.error_message
        == "local chat candidate generation returned no usable lines"
    )
    assert batch.raw_response == "  \n "
    assert batch.latency_ms == 12.5
    assert batch.prompt_tokens == 21
    assert batch.completion_tokens == 13


def test_local_chat_error_batch_retains_safe_transport_diagnostic_as_warning() -> None:
    class TimeoutClient:
        def generate(self, _messages, **_kwargs):
            raise requests.Timeout(
                "target_url=http://127.0.0.1:18001/v1/chat/completions "
                "exception_class=ReadTimeout exception_repr=ReadTimeout('')"
            )

    batch = LocalChatCandidateGenerator(TimeoutClient()).generate(request_for_bar())

    assert batch.error_type == "generation_error"
    assert batch.warning == batch.error_message
    assert "target_url=http://127.0.0.1:18001/v1/chat/completions" in (
        batch.error_message or ""
    )


def test_local_chat_parsing_deduplicates_without_filtering_by_requested_syllable_count() -> (
    None
):
    batch = LocalChatCandidateGenerator(
        FakeClient("1. one\n2. one\n3. a much longer candidate line stays raw")
    ).generate(request_for_bar())

    assert batch.candidates == ("one", "a much longer candidate line stays raw")


def test_local_chat_parsing_preserves_surrounding_quote_characters() -> None:
    batch = LocalChatCandidateGenerator(
        FakeClient('  1. "quoted candidate"  ')
    ).generate(request_for_bar())

    assert batch.candidates == ('"quoted candidate"',)


@pytest.mark.parametrize("text", (None, 17, object()))
def test_local_chat_malformed_text_returns_an_explicit_error_batch(
    text: object,
) -> None:
    response = type(
        "MalformedTextResponse",
        (),
        {"text": text, "latency_ms": 8.0, "prompt_tokens": 4, "completion_tokens": 2},
    )()

    batch = LocalChatCandidateGenerator(ResponseClient(response)).generate(
        request_for_bar()
    )

    assert batch.candidates == ()
    assert batch.error_type == "generation_error"
    assert "text must be a string" in (batch.error_message or "")
    assert batch.raw_response == ""
    assert batch.latency_ms == 8.0
    assert batch.prompt_tokens == 4
    assert batch.completion_tokens == 2


@pytest.mark.parametrize(
    "field,value",
    (("latency_ms", "slow"), ("prompt_tokens", -1), ("completion_tokens", "many")),
)
def test_local_chat_malformed_diagnostics_return_an_explicit_error_batch(
    field: str, value: object
) -> None:
    attributes = {
        "text": "a valid line",
        "latency_ms": 8.0,
        "prompt_tokens": 4,
        "completion_tokens": 2,
    }
    attributes[field] = value
    response = type("MalformedDiagnosticsResponse", (), attributes)()

    batch = LocalChatCandidateGenerator(ResponseClient(response)).generate(
        request_for_bar()
    )

    assert batch.candidates == ()
    assert batch.error_type == "generation_error"
    assert field in (batch.error_message or "")
    assert batch.raw_response == "a valid line"


def test_local_chat_raising_diagnostics_return_an_explicit_error_batch() -> None:
    batch = LocalChatCandidateGenerator(
        ResponseClient(RaisingLatencyResponse())
    ).generate(request_for_bar())

    assert batch.candidates == ()
    assert batch.error_type == "generation_error"
    assert "diagnostic clock unavailable" in (batch.error_message or "")
    assert batch.raw_response == "a valid line"
    assert batch.latency_ms == 0.0
    assert batch.prompt_tokens == 7
    assert batch.completion_tokens == 3


def test_local_chat_raising_token_diagnostics_return_an_explicit_error_batch() -> None:
    batch = LocalChatCandidateGenerator(
        ResponseClient(RaisingPromptTokensResponse())
    ).generate(request_for_bar())

    assert batch.candidates == ()
    assert batch.error_type == "generation_error"
    assert "prompt token counter unavailable" in (batch.error_message or "")
    assert batch.raw_response == "a valid line"
    assert batch.latency_ms == 8.0
    assert batch.prompt_tokens is None
    assert batch.completion_tokens == 3


def test_local_chat_contains_success_batch_validation_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_batch = generator_module.CandidateBatch

    def build_batch(**kwargs: object) -> CandidateBatch:
        if kwargs["candidates"]:
            raise ValueError("candidate batch validation failed")
        return original_batch(**kwargs)

    monkeypatch.setattr(generator_module, "CandidateBatch", build_batch)
    batch = LocalChatCandidateGenerator(FakeClient("a valid line")).generate(
        request_for_bar()
    )

    assert batch.candidates == ()
    assert batch.error_type == "generation_error"
    assert "candidate batch validation failed" in (batch.error_message or "")


def test_local_chat_redacts_environment_style_secrets_without_hiding_error_context() -> (
    None
):
    client = ResponseClient(
        type(
            "FailingResponse",
            (),
            {
                "text": property(
                    lambda self: (_ for _ in ()).throw(
                        RuntimeError(
                            "connection refused; OPENAI_API_KEY=sk-live; "
                            "SERVICE_TOKEN=token-value; DB_PASSWORD: db-secret; "
                            "Authorization: Bearer auth-secret"
                        )
                    )
                )
            },
        )()
    )

    batch = LocalChatCandidateGenerator(client).generate(request_for_bar())

    assert batch.error_type == "generation_error"
    assert "connection refused" in (batch.error_message or "")
    for secret in ("sk-live", "token-value", "db-secret", "auth-secret"):
        assert secret not in (batch.error_message or "")
    assert "OPENAI_API_KEY=[REDACTED]" in (batch.error_message or "")
    assert "SERVICE_TOKEN=[REDACTED]" in (batch.error_message or "")
    assert "DB_PASSWORD: [REDACTED]" in (batch.error_message or "")
    assert "Authorization: [REDACTED]" in (batch.error_message or "")


def test_local_chat_redacts_generic_and_compound_secret_assignments_only() -> None:
    client = ResponseClient(
        type(
            "CompoundSecretResponse",
            (),
            {
                "text": property(
                    lambda self: (_ for _ in ()).throw(
                        RuntimeError(
                            "request failed; GROQ_KEY=groq-secret; PRIVATE_KEY: private-secret; "
                            "SECRET_KEY=secret-key; AWS_ACCESS_KEY_ID=aws-key; API_KEY=api-secret; "
                            "TOKEN: token-secret; PASSWORD=password-secret; SECRET: secret-value; "
                            "AUTHORIZATION: Bearer header-secret; ordinary secret reference remains"
                        )
                    )
                )
            },
        )()
    )

    batch = LocalChatCandidateGenerator(client).generate(request_for_bar())

    assert "ordinary secret reference remains" in (batch.error_message or "")
    for secret in (
        "groq-secret",
        "private-secret",
        "secret-key",
        "aws-key",
        "api-secret",
        "token-secret",
        "password-secret",
        "secret-value",
        "header-secret",
    ):
        assert secret not in (batch.error_message or "")
    for name in (
        "GROQ_KEY=[REDACTED]",
        "PRIVATE_KEY: [REDACTED]",
        "SECRET_KEY=[REDACTED]",
        "AWS_ACCESS_KEY_ID=[REDACTED]",
        "API_KEY=[REDACTED]",
        "TOKEN: [REDACTED]",
        "PASSWORD=[REDACTED]",
        "SECRET: [REDACTED]",
        "AUTHORIZATION: [REDACTED]",
    ):
        assert name in (batch.error_message or "")


def test_local_chat_redacts_a_standalone_bearer_token_without_hiding_other_prose() -> (
    None
):
    client = ResponseClient(
        type(
            "BareBearerResponse",
            (),
            {
                "text": property(
                    lambda self: (_ for _ in ()).throw(
                        RuntimeError(
                            "request failed; Bearer bare-secret; ordinary non-assignment prose remains"
                        )
                    )
                )
            },
        )()
    )

    batch = LocalChatCandidateGenerator(client).generate(request_for_bar())

    assert "bare-secret" not in (batch.error_message or "")
    assert "Bearer [REDACTED]" in (batch.error_message or "")
    assert "ordinary non-assignment prose remains" in (batch.error_message or "")


def test_local_chat_error_batch_sanitizes_secrets_exactly_once() -> None:
    class FailingClient:
        def generate(self, *args, **kwargs) -> object:
            raise RuntimeError("OPENAI_API_KEY=api-secret; Bearer bearer-secret")

    batch = LocalChatCandidateGenerator(FailingClient()).generate(request_for_bar())

    assert batch.error_message == "OPENAI_API_KEY=[REDACTED]; Bearer [REDACTED]"


def test_sanitize_error_is_idempotent_for_redacted_assignments_and_bearer_tokens() -> (
    None
):
    raw = "OPENAI_API_KEY=api-secret; Bearer bearer-secret"
    expected = "OPENAI_API_KEY=[REDACTED]; Bearer [REDACTED]"

    assert _sanitize_error(raw) == expected
    assert _sanitize_error(expected) == expected


@pytest.mark.parametrize(
    ("message", "expected"),
    (
        ("OPENAI_API_KEY=api-secret", "OPENAI_API_KEY=[REDACTED]"),
        ("OPENAI_API_KEY=[REDACTED]", "OPENAI_API_KEY=[REDACTED]"),
        ("Authorization: Bearer auth-secret", "Authorization: [REDACTED]"),
        ("Authorization: Bearer [REDACTED]", "Authorization: [REDACTED]"),
        ("Bearer bearer-secret", "Bearer [REDACTED]"),
        ("Bearer [REDACTED]", "Bearer [REDACTED]"),
    ),
)
def test_sanitize_error_is_idempotent_for_raw_and_redacted_secret_forms(
    message: str,
    expected: str,
) -> None:
    sanitized = message

    for _ in range(2):
        sanitized = _sanitize_error(sanitized)
        assert sanitized == expected


def test_candidate_batch_prompt_is_defensively_and_deeply_immutable() -> None:
    message = {"role": "user", "content": "original"}
    batch = LocalChatCandidateGenerator(FakeClient("a line")).generate(
        request_for_bar()
    )

    direct_batch = CandidateBatch(
        request_id="immutable-prompt",
        candidates=("a line",),
        source="test",
        prompt=(message,),
        raw_response="a line",
        latency_ms=1.0,
    )
    message["content"] = "mutated outside"

    assert direct_batch.prompt[0]["content"] == "original"
    with pytest.raises(TypeError):
        direct_batch.prompt[0]["content"] = "mutated inside"
    with pytest.raises(TypeError):
        dict.__setitem__(direct_batch.prompt[0], "content", "dict bypass")
    assert (
        json.loads(json.dumps({"prompt": direct_batch.prompt_json}))["prompt"][0][
            "content"
        ]
        == "original"
    )
    prompt_copy = direct_batch.prompt_json
    prompt_copy[0]["content"] = "mutable JSON copy"
    assert direct_batch.prompt[0]["content"] == "original"
    with pytest.raises(TypeError):
        batch.prompt[-1]["content"] = "mutated generated prompt"


def test_independent_choice_generator_requests_one_line_per_api_choice() -> None:
    client = FakeChoicesClient(
        (
            "1. Moon light rides",
            "moon   light rides!",
            "Blue sky moves\nthis second line must be ignored",
            "* Beat stays clean",
        )
    )
    request = request_for_bar()

    batch = IndependentChoiceCandidateGenerator(client).generate(request)

    assert batch.source == "local_chat_independent"
    assert batch.candidates == (
        "Moon light rides",
        "Blue sky moves",
        "Beat stays clean",
    )
    assert batch.latency_ms == 17.5
    assert batch.prompt_tokens == 31
    assert batch.completion_tokens == 22
    assert batch.warning == "requested_4_received_3"
    assert batch.provider_choice_indices == (0, 2, 3)
    assert client.calls[0][1] == {"n": 4, "max_tokens": 32, "temperature": 1.0}


def test_independent_choice_prompt_contains_complete_flow_context_and_visible_seed() -> (
    None
):
    client = FakeChoicesClient(("one line",))
    request = request_for_bar(
        flow_template=BUILTIN_TEMPLATES.get("baseline_syncopated_9")
    )

    IndependentChoiceCandidateGenerator(client).generate(request)

    messages = client.calls[0][0][0]
    assert "Return exactly one plain lyric line" in messages[0]["content"]
    user = messages[1]["content"]
    assert "exactly 9 spoken syllables" in user
    assert "normal American spoken pronunciation" in user
    assert "Spell out every number with words; never emit digits" in user
    assert "Syllable ticks: [0, 2, 3, 5, 7, 8, 10, 13, 15]" in user
    assert "Target stress: [1.0, 0.2, 0.7, 0.2, 0.6, 1.0, 0.2, 0.7, 0.9]" in user
    assert "stars cross the night" in user
    assert f"Variation seed: {request.seed}" in user
    assert "seed" not in client.calls[0][1]


def test_independent_choice_generator_preserves_bounded_client_warnings() -> None:
    client = FakeChoicesClient(("one line",), warnings=("choice[2] malformed",))

    batch = IndependentChoiceCandidateGenerator(client).generate(request_for_bar())

    assert batch.warning == "choice[2] malformed; requested_4_received_1"


def test_independent_choice_generator_preserves_over_returned_choice_indices() -> None:
    client = FakeChoicesClient(("one", "two", "three", "four", "five"))

    batch = IndependentChoiceCandidateGenerator(client).generate(request_for_bar())

    assert batch.candidates == ("one", "two", "three", "four", "five")
    assert batch.provider_choice_indices == (0, 1, 2, 3, 4)


def test_candidate_batch_provider_choice_indices_are_optional_and_aligned() -> None:
    batch = CandidateBatch(
        request_id="provider-default",
        candidates=("one",),
        source="test",
        prompt=(),
        raw_response="one",
        latency_ms=0.0,
    )

    assert batch.provider_choice_indices == ()
    with pytest.raises(ValueError, match="provider_choice_indices"):
        CandidateBatch(
            request_id="provider-mismatch",
            candidates=("one", "two"),
            source="test",
            prompt=(),
            raw_response="one\ntwo",
            latency_ms=0.0,
            provider_choice_indices=(4,),
        )


def test_independent_choice_generator_returns_explicit_error_batch() -> None:
    class FailingChoicesClient:
        def generate_choices(self, *_args, **_kwargs):
            raise RuntimeError("request failed; API_KEY=private-value")

    batch = IndependentChoiceCandidateGenerator(FailingChoicesClient()).generate(
        request_for_bar()
    )

    assert batch.source == "local_chat_independent"
    assert batch.candidates == ()
    assert batch.error_type == "generation_error"
    assert "private-value" not in (batch.error_message or "")
    assert "API_KEY=[REDACTED]" in (batch.error_message or "")
