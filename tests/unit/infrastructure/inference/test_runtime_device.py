import pytest
import torch

from streammuse.infrastructure.inference import runtime_device


def test_resolve_device_auto_prefers_mps_when_cuda_unavailable(monkeypatch):
    monkeypatch.setattr(runtime_device.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(runtime_device, "is_mps_available", lambda: True)
    assert runtime_device.resolve_device("auto") == "mps"


def test_resolve_device_auto_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(runtime_device.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(runtime_device, "is_mps_available", lambda: False)
    assert runtime_device.resolve_device("auto") == "cpu"


def test_resolve_device_explicit_mps_unavailable_raises(monkeypatch):
    monkeypatch.setattr(runtime_device, "is_mps_available", lambda: False)
    with pytest.raises(RuntimeError, match="MPS is not available"):
        runtime_device.resolve_device("mps")


def test_resolve_dtype_auto_policy():
    assert runtime_device.resolve_dtype("cpu", "auto") == torch.float32
    assert runtime_device.resolve_dtype("cuda", "auto") == torch.float16
    assert runtime_device.resolve_dtype("mps", "auto") == torch.float16


def test_resolve_dtype_cpu_float16_raises():
    with pytest.raises(RuntimeError, match="float16"):
        runtime_device.resolve_dtype("cpu", "float16")


def test_parse_env_bool():
    assert runtime_device.parse_env_bool("true", default=False) is True
    assert runtime_device.parse_env_bool("0", default=True) is False
    assert runtime_device.parse_env_bool(None, default=True) is True
    assert runtime_device.parse_env_bool("unexpected", default=False) is False
