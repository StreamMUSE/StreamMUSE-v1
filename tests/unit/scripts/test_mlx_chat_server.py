from __future__ import annotations

import pytest

from scripts.mlx_chat_server import ChatRequestError, completion_payload, parse_chat_request
from streammuse.infrastructure.inference.local_chat_client import LocalChatModelClient

MODEL = "qwen-rap"


def _payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You write lyrics."},
            {"role": "user", "content": "Give lines about space."},
        ],
        "temperature": 1.0,
        "max_tokens": 32,
        "n": 16,
    }
    payload.update(overrides)
    return payload


def test_parses_the_independent_choice_request_shape() -> None:
    request = parse_chat_request(_payload(top_p=0.9), model_id=MODEL)

    assert request.n == 16
    assert request.max_tokens == 32
    assert request.temperature == 1.0
    assert request.top_p == 0.9
    assert [message["role"] for message in request.messages] == ["system", "user"]


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"model": "other"}, "not served"),
        ({"stream": True}, "streaming"),
        ({"n": 0}, "n must"),
        ({"n": True}, "n must"),
        ({"max_tokens": 10_000}, "max_tokens"),
        ({"temperature": -1}, "temperature"),
        ({"messages": []}, "messages"),
        ({"messages": [{"role": "tool", "content": "x"}]}, "role"),
        ({"logprobs": True}, "unsupported request fields"),
    ),
)
def test_rejects_out_of_contract_requests(overrides: dict[str, object], message: str) -> None:
    with pytest.raises(ChatRequestError, match=message):
        parse_chat_request(_payload(**overrides), model_id=MODEL)


def test_completion_payload_is_parsed_by_the_render_server_client() -> None:
    payload = completion_payload(
        model_id=MODEL,
        texts=["first line", "second line"],
        finish_reasons=["stop", "length"],
        prompt_tokens=100,
        completion_tokens=12,
    )

    parsed = LocalChatModelClient._parse_choices_response(payload, latency_ms=5.0)

    assert [(choice.index, choice.text) for choice in parsed.choices] == [
        (0, "first line"),
        (1, "second line"),
    ]
    assert payload["usage"] == {"prompt_tokens": 100, "completion_tokens": 12, "total_tokens": 112}
