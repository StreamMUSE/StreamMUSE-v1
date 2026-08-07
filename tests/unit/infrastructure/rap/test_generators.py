"""Tests for offline and local-chat lyric candidate adapters."""

from streammuse.domain.rap import CandidateRequest
from streammuse.domain.tasks import ChatModelResponse
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


def test_local_chat_parsing_does_not_filter_by_requested_syllable_count() -> None:
    batch = LocalChatCandidateGenerator(FakeClient("1. one\n2. one\n3. a much longer candidate line stays raw")).generate(
        request_for_bar()
    )

    assert batch.candidates == ("one", "a much longer candidate line stays raw")
