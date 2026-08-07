"""Tests for offline and local-chat lyric candidate adapters."""

import json

import pytest

from streammuse.domain.rap import CandidateBatch, CandidateRequest
from streammuse.domain.tasks import ChatModelResponse
from streammuse.infrastructure.rap import generators as generator_module
from streammuse.infrastructure.rap.generators import LocalChatCandidateGenerator, PhraseBankGenerator


class FailingClient:
    def generate(self, *args, **kwargs):
        raise RuntimeError("connection refused; Authorization: Bearer super-secret-token")


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


def request_for_bar(bar: int = 2) -> CandidateRequest:
    return CandidateRequest(
        request_id=f"bar-{bar}-request-1",
        target_bar=bar,
        topic="space travel",
        template_id="baseline_syncopated_9",
        required_syllables=9,
        count=4,
        context_lines=("stars cross the night",),
        seed=20260807,
    )


def test_phrase_bank_normalizes_empty_topic_and_returns_requested_candidates() -> None:
    request = CandidateRequest(
        request_id="phrase-bank-1",
        target_bar=0,
        topic="!!!",
        template_id="baseline_syncopated_9",
        required_syllables=9,
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


def test_local_chat_request_preserves_structure_history_and_raw_diagnostics() -> None:
    client = FakeClient("1. Space travel keeps the whole night bright\n2. We move through stars with rhythm")
    request = request_for_bar()
    batch = LocalChatCandidateGenerator(client).generate(request)

    assert batch.source == "local_chat"
    assert batch.request_id == request.request_id
    assert "exactly 9 spoken syllables" in batch.prompt[-1]["content"]
    assert "stars cross the night" in batch.prompt[-1]["content"]
    assert batch.raw_response == "1. Space travel keeps the whole night bright\n2. We move through stars with rhythm"
    assert batch.latency_ms == 12.5
    assert batch.prompt_tokens == 21
    assert batch.completion_tokens == 13
    assert batch.candidates == (
        "Space travel keeps the whole night bright",
        "We move through stars with rhythm",
    )
    assert client.calls[0][1]["max_tokens"] > 0


def test_local_chat_error_returns_explicit_empty_batch_without_phrase_bank_fallback() -> None:
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
    assert batch.error_message == "local chat candidate generation returned no usable lines"
    assert batch.raw_response == "  \n "
    assert batch.latency_ms == 12.5
    assert batch.prompt_tokens == 21
    assert batch.completion_tokens == 13


def test_local_chat_parsing_deduplicates_without_filtering_by_requested_syllable_count() -> None:
    batch = LocalChatCandidateGenerator(FakeClient("1. one\n2. one\n3. a much longer candidate line stays raw")).generate(
        request_for_bar()
    )

    assert batch.candidates == ("one", "a much longer candidate line stays raw")


def test_local_chat_parsing_preserves_surrounding_quote_characters() -> None:
    batch = LocalChatCandidateGenerator(FakeClient('  1. "quoted candidate"  ')).generate(request_for_bar())

    assert batch.candidates == ('"quoted candidate"',)


@pytest.mark.parametrize("text", (None, 17, object()))
def test_local_chat_malformed_text_returns_an_explicit_error_batch(text: object) -> None:
    response = type(
        "MalformedTextResponse",
        (),
        {"text": text, "latency_ms": 8.0, "prompt_tokens": 4, "completion_tokens": 2},
    )()

    batch = LocalChatCandidateGenerator(ResponseClient(response)).generate(request_for_bar())

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
def test_local_chat_malformed_diagnostics_return_an_explicit_error_batch(field: str, value: object) -> None:
    attributes = {"text": "a valid line", "latency_ms": 8.0, "prompt_tokens": 4, "completion_tokens": 2}
    attributes[field] = value
    response = type("MalformedDiagnosticsResponse", (), attributes)()

    batch = LocalChatCandidateGenerator(ResponseClient(response)).generate(request_for_bar())

    assert batch.candidates == ()
    assert batch.error_type == "generation_error"
    assert field in (batch.error_message or "")
    assert batch.raw_response == "a valid line"


def test_local_chat_raising_diagnostics_return_an_explicit_error_batch() -> None:
    batch = LocalChatCandidateGenerator(ResponseClient(RaisingLatencyResponse())).generate(request_for_bar())

    assert batch.candidates == ()
    assert batch.error_type == "generation_error"
    assert "diagnostic clock unavailable" in (batch.error_message or "")
    assert batch.raw_response == "a valid line"
    assert batch.latency_ms == 0.0
    assert batch.prompt_tokens == 7
    assert batch.completion_tokens == 3


def test_local_chat_raising_token_diagnostics_return_an_explicit_error_batch() -> None:
    batch = LocalChatCandidateGenerator(ResponseClient(RaisingPromptTokensResponse())).generate(request_for_bar())

    assert batch.candidates == ()
    assert batch.error_type == "generation_error"
    assert "prompt token counter unavailable" in (batch.error_message or "")
    assert batch.raw_response == "a valid line"
    assert batch.latency_ms == 8.0
    assert batch.prompt_tokens is None
    assert batch.completion_tokens == 3


def test_local_chat_contains_success_batch_validation_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    original_batch = generator_module.CandidateBatch

    def build_batch(**kwargs: object) -> CandidateBatch:
        if kwargs["candidates"]:
            raise ValueError("candidate batch validation failed")
        return original_batch(**kwargs)

    monkeypatch.setattr(generator_module, "CandidateBatch", build_batch)
    batch = LocalChatCandidateGenerator(FakeClient("a valid line")).generate(request_for_bar())

    assert batch.candidates == ()
    assert batch.error_type == "generation_error"
    assert "candidate batch validation failed" in (batch.error_message or "")


def test_local_chat_redacts_environment_style_secrets_without_hiding_error_context() -> None:
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


def test_candidate_batch_prompt_is_defensively_and_deeply_immutable() -> None:
    message = {"role": "user", "content": "original"}
    batch = LocalChatCandidateGenerator(FakeClient("a line")).generate(request_for_bar())

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
    assert json.loads(json.dumps({"prompt": direct_batch.prompt}))["prompt"][0]["content"] == "original"
    with pytest.raises(TypeError):
        batch.prompt[-1]["content"] = "mutated generated prompt"
