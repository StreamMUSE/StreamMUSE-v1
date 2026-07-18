"""Tests for offline and local-chat lyric candidate adapters."""

from types import SimpleNamespace

from streammuse.infrastructure.rap.generators import LocalChatCandidateGenerator, PhraseBankGenerator


class FailingClient:
    def generate(self, *args, **kwargs):
        raise RuntimeError("server unavailable")


class FakeClient:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def generate(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return SimpleNamespace(text=self.text)


def test_phrase_bank_normalizes_empty_topic_and_returns_requested_candidates() -> None:
    batch = PhraseBankGenerator().generate("!!!", 3)

    assert batch.source == "phrase_bank"
    assert len(batch.candidates) == 3
    assert all("the moment" in line for line in batch.candidates)


def test_local_chat_parses_numbered_lines_and_limits_candidates() -> None:
    client = FakeClient("1. Space travel keeps the whole night bright\n2. We move through stars with rhythm")
    batch = LocalChatCandidateGenerator(client, PhraseBankGenerator()).generate("space travel", 1)

    assert batch.source == "local_chat"
    assert batch.candidates == ("Space travel keeps the whole night bright",)
    assert client.calls[0][1]["max_tokens"] > 0


def test_local_chat_falls_back_to_phrase_bank_after_client_error() -> None:
    batch = LocalChatCandidateGenerator(FailingClient(), PhraseBankGenerator()).generate("space travel", 4)

    assert batch.source == "phrase_bank"
    assert batch.warning == "local chat candidate generation failed: server unavailable"
    assert len(batch.candidates) == 4


def test_local_chat_falls_back_after_empty_response() -> None:
    batch = LocalChatCandidateGenerator(FakeClient("  \n "), PhraseBankGenerator()).generate("space travel", 4)

    assert batch.source == "phrase_bank"
    assert batch.warning == "local chat candidate generation returned no usable lines"
