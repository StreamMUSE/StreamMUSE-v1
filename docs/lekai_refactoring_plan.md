# Lekai Model 重构建议：为 Engine 提取核心功能

为了让 `InferenceEngineLekai` (实时引擎) 能够达到与 `lekai_model` (离线模型) 相同的生成质量，我们需要将 `model.py` 中耦合的逻辑拆解为可复用的函数。

## 1. 必须提取的功能 (Critical)

### A. 采样逻辑 (Sampling Logic)
目前 Engine 使用的是 `argmax` (贪婪搜索)，这会导致生成的音乐非常死板、重复。而 Model 中实现了完整的采样策略。

*   **现状**:
    *   **Model**: `Temperature` + `Repetition Penalty` + `Top-K` + `Top-P` + `Multinomial Sampling`。
    *   **Engine**: `torch.argmax` (仅取概率最大的，无随机性)。
*   **建议**:
    *   在 `lekai_model/utils.py` 或 `model.py` 中创建一个函数 `sample_token(...)`。
    *   **Engine 调用**: Engine 在生成每个 Token 时调用此函数，而不是自己写 `argmax`。

```python
# 建议的函数签名
def sample_token(logits, temperature=1.0, top_k=50, top_p=0.95, repetition_penalty=1.0, generated_tokens=None):
    # ... 实现 model.py 中的采样逻辑 ...
    return next_token
```

### B. 单步生成 (Step-wise Generation)
Model 的生成逻辑被包裹在 `max_iterations` 循环中，Engine 难以直接复用。

*   **现状**:
    *   **Model**: 深度耦合在 `for iteration in range(max_iterations)` 循环中。
    *   **Engine**: 自己重新实现了一个简单的 `for` 循环。
*   **建议**:
    *   将模型的前向传播和 KV Cache 更新封装到一个 `generate_step` 方法中。

---

## 2. 建议提取的功能 (Recommended)

### C. Prompt 头部构建 (Header Construction)
确保 Engine 和 Model 使用完全相同的特殊 Token 顺序，避免训练和推理不一致。

*   **现状**:
    *   **Model**: `[BOS, TimeSig, BPM]`
    *   **Engine**: 手动构建列表 `[BOS, TimeSig, BPM]`
*   **建议**:
    *   提取为 `build_start_tokens(config, bpm, time_sig)`。

### D. 拍子 Token 化 (Beat Tokenization)
虽然 Engine 输入是 Note，Model 输入是 Tensor，但最终都转为 Token。

*   **现状**:
    *   **Model**: `process_measure_with_beat_interleaving` (处理 Tensor)
    *   **Engine**: `get_tokens_for_beat` (处理 Note -> Pianoroll)
*   **建议**:
    *   保持现状即可，因为输入数据源不同（一个是文件，一个是实时数据），强行统一成本较高。但需确保两者使用的 `PianoRollTokenizer` 参数一致。

---

## 3. 重构路线图

1.  **创建 `lekai_model/generation_utils.py`**:
    *   将 `model.py` 第 200-240 行左右的采样代码（Temperature, Top-k, Top-p）移动到这里。
2.  **修改 `lekai_model/model.py`**:
    *   让 `generate_accompaniment` 调用新的 `generation_utils.sample_token`。
3.  **修改 `app/inference_engines/transformer_engine_lekai.py`**:
    *   导入 `generation_utils.sample_token`。
    *   在 `generate_accompaniment` 的循环中，用它替换 `torch.argmax`。

这样可以确保 Engine 拥有与离线模型完全一致的“创造力”。
