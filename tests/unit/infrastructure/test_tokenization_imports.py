"""Unit tests for infrastructure inference tokenization package (moved from tokenizers/)."""

import pytest


def test_tokenization_package_imports():
    """Tokenization package can be imported and exposes Xinyue types."""
    from streammuse.infrastructure.inference.tokenization import (
        XinyueTokenizer,
        XinyueTokenizerConfig,
        XINYUE_SPECIAL_TOKENS,
    )

    assert XinyueTokenizer is not None
    assert XinyueTokenizerConfig is not None
    assert isinstance(XINYUE_SPECIAL_TOKENS, (list, dict))
