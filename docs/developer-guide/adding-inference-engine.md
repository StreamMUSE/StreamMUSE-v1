---
title: 新增 InferenceEngine
description: 如何为 StreamMUSE 添加新的推理引擎适配器
---

# 新增 InferenceEngine

本指南说明如何为 StreamMUSE 添加一个新的推理引擎（如 Lekai 模型、ONNX 引擎等）。

---

## 步骤概览

1. 在 `infrastructure/inference/` 中创建新文件
2. 实现 `InferenceEngine` 协议（4 个方法）
3. 在 `InferenceEngineFactory` 中注册新类型
4. 在 `InferenceConfig.InferenceType` 中添加新 Literal 值
5. 在 `InferenceConfig` 中添加所需配置字段
6. 编写测试

---

## 第一步：实现 InferenceEngine 协议

```python
# src/streammuse/infrastructure/inference/lekai_engine.py
from __future__ import annotations
from typing import List, Optional
from streammuse.domain.interfaces import InferenceEngine, TimingInfo
from streammuse.domain.musical import MusicalEvent

class LekaiInferenceEngine(InferenceEngine):
    """适配 Lekai 模型的推理引擎。"""

    def __init__(self, *, model_path: str, device: str = "cpu") -> None:
        # 加载模型
        self._model = load_lekai_model(model_path, device=device)

    def generate_accompaniment(
        self,
        melody_events: List[MusicalEvent],
        generation_start_tick: int,
        generation_length_frames: int,
        prompt_length_ticks: Optional[int] = None,
    ) -> tuple[List[MusicalEvent], TimingInfo]:
        import time
        t0 = time.time()
        
        # 将 MusicalEvent 转为模型需要的输入格式
        model_input = self._events_to_model_input(melody_events, generation_start_tick)
        
        t1 = time.time()
        output = self._model.predict(model_input)   # 模型推理
        t2 = time.time()
        
        # 将模型输出转为 MusicalEvent
        acc_events = self._model_output_to_events(output, generation_start_tick)
        t3 = time.time()
        
        timing = TimingInfo(
            request_arrival_time=t0,
            inference_start_time=t1,
            inference_end_time=t2,
            response_output_time=t3,
            preprocess_start_time=t0,
            postprocess_start_time=t2,
        )
        return acc_events, timing

    def inject_history(
        self,
        melody_events: List[MusicalEvent],
        accompaniment_events: List[MusicalEvent],
        injection_length_ticks: int,
    ) -> None:
        # 若模型支持历史注入则实现；否则 pass
        pass

    def set_injection_offset(self, offset_ticks: int) -> None:
        self._injection_offset = offset_ticks

    def clear_history(self) -> Dict[str, Any]:
        self._model.reset()
        return {
            "success": True,
            "message": "History cleared",
            "melody_history": [],
            "accompaniment_history": [],
        }
```

**关键约定**：
- `generate_accompaniment()` 返回 `(List[MusicalEvent], TimingInfo)`
- 所有 4 个方法都必须实现（即使是空实现）
- `TimingInfo` 的 6 个必填字段必须使用实际时间戳
- 生成的事件 `source` 字段设为 `"model"`

---

## 第二步：注册到 Factory

```python
# src/streammuse/application/factories/inference_factory.py
from streammuse.infrastructure.inference.lekai_engine import LekaiInferenceEngine

class InferenceEngineFactory:
    @staticmethod
    def create(app_config):
        cfg = app_config.inference
        ...
        elif cfg.type == "lekai":
            if cfg.checkpoint_path is None:
                raise ValueError("lekai mode requires checkpoint_path")
            return LekaiInferenceEngine(
                model_path=cfg.checkpoint_path,
                device=cfg.lekai_device or "cpu",
            )
        ...
```

---

## 第三步：扩展配置

```python
# src/streammuse/application/config/models.py
InferenceType = Literal["http", "stanley", "lekai"]

@dataclass(frozen=True)
class InferenceConfig:
    ...
    lekai_device: str = "cpu"
```

---

## 第四步：测试

```python
# tests/unit/infrastructure/test_lekai_engine.py
from unittest.mock import MagicMock
from streammuse.infrastructure.inference.lekai_engine import LekaiInferenceEngine

def test_generate_returns_events_and_timing():
    engine = LekaiInferenceEngine(model_path="path/to/model")
    events, timing = engine.generate_accompaniment([], 0, 20)
    assert isinstance(events, list)
    assert timing.inference_start_time <= timing.inference_end_time
```
