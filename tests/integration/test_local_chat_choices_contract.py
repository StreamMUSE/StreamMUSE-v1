"""Live contract check for the candidate generator's chat server.

The render server asks for 16 candidates per bar with one ``n=16`` request.
Some OpenAI-compatible servers silently ignore ``n`` (mlx_lm.server) or return
fewer choices, which leaves every chunk without enough valid candidates. Point
this at a running server to verify it before a session:

    STREAMMUSE_CHAT_URL=http://127.0.0.1:8001/v1 STREAMMUSE_CHAT_MODEL=qwen-rap \
        uv run pytest tests/integration/test_local_chat_choices_contract.py -q

Skipped unless STREAMMUSE_CHAT_URL is set, so CI never needs a model server.
"""

from __future__ import annotations

import os

import pytest

from streammuse.infrastructure.inference.local_chat_client import (
    LocalChatModelClient,
    LocalChatModelClientConfig,
)

CHAT_URL = os.environ.get("STREAMMUSE_CHAT_URL")

pytestmark = pytest.mark.skipif(
    not CHAT_URL, reason="set STREAMMUSE_CHAT_URL to check a live chat server"
)


def test_n16_request_returns_sixteen_non_empty_indexed_choices() -> None:
    client = LocalChatModelClient(
        LocalChatModelClientConfig(
            base_url=str(CHAT_URL),
            model=os.environ.get("STREAMMUSE_CHAT_MODEL", "qwen-rap"),
            timeout_s=120.0,
        )
    )
    try:
        response = client.generate_choices(
            [
                {"role": "system", "content": "You write one-line rap lyrics."},
                {"role": "user", "content": "Give one nine-syllable line about the ocean."},
            ],
            n=16,
            max_tokens=32,
            temperature=1.0,
        )
    finally:
        client.close()

    assert len(response.choices) == 16
    assert sorted(choice.index for choice in response.choices) == list(range(16))
    assert all(choice.text.strip() for choice in response.choices)
