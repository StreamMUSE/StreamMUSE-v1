"""Unit tests for infrastructure inference config package (moved from schema/)."""

import pytest


def test_config_package_imports():
    """Config package can be imported and exposes main schema types."""
    from streammuse.infrastructure.inference.config import (
        ModelSchema,
        DataModuleSchema,
        ProjectSchema,
    )

    assert ModelSchema is not None
    assert DataModuleSchema is not None
    assert ProjectSchema is not None


def test_model_schema_uses_transformers_roformer():
    """Model schema uses RoFormerConfig from transformers fork."""
    from streammuse.infrastructure.inference.config.model_schema import M2AModelSchema

    schema = M2AModelSchema()
    assert schema.network_schema is not None
    assert hasattr(schema.network_schema, "vocab_size")
