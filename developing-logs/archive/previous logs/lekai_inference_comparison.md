# Lekai 模型推理方案对比：Offline vs Engine

本文档对比了 Lekai 模型在 **离线推理 (Offline Inference)** 和 **实时引擎推理 (Engine Inference)** 两种场景下的实现方案区别。

## 1. 方案概述

### A. 离线推理 (Offline Inference)
*   **代码位置**: `lekai_model/model.py` (中的 `generate_accompaniment` 方法) 或 `lekai_model/inference.py`。
*   **场景**: 批量生成、测试集评估、非实时生成整首曲子。
*   **核心逻辑**: 一次性加载整首曲子的旋律（Part 0），然后在一个长循环中从头到尾生成伴奏（Part 1）。

### B. 引擎推理 (Engine Inference)
*   **代码位置**: `app/inference_engines/transformer_engine_lekai.py`。
*   **场景**: 实时交互、流式生成、DAW 插件后端。
*   **核心逻辑**: 每次 API 调用只生成一小段（如 1 拍），使用“滑动窗口”机制构建上下文。

---

## 2. 核心区别详解

| 特性 | 离线推理 (Offline) | 引擎推理 (Engine) |
| :--- | :--- | :--- |
| **上下文 (Context)** | **全局累积**：上下文从 0 开始一直累积，直到达到模型最大长度 (2048/3500 tokens)。 | **滑动窗口**：仅保留最近 N 拍（如 `context_beats=32`）作为上下文。旧的历史会被丢弃。 |
| **KV Cache (显存状态)** | **持久化**：`past_key_values` 在整个生成过程中一直保留并更新。 | **无状态/重计算**：每次 API 调用都是独立的。必须重新计算当前窗口内所有历史 Token 的 KV Cache (Prefill 阶段)。 |
| **生成粒度** | **整曲**：一次调用生成整首曲子的所有拍子。 | **单拍/短片段**：一次调用通常只生成 1 拍 (或指定帧数)。 |
| **输入数据** | **完整文件**：读取 `.npz` 文件，包含整首曲子的完整结构。 | **流式数据**：接收增量的 Note List，需要手动维护 `history` 列表。 |
| **计算效率** | **高**：每个 Token 只计算一次。 | **低**：每次生成新的一拍时，都要重复计算过去 32 拍的 Attention（因为没有跨请求保留 KV Cache）。 |
| **延迟 (Latency)** | 不关注单拍延迟，关注总吞吐量。 | 关注单次调用的响应速度。滑动窗口限制了计算量，保证延迟可控。 |

---

## 3. 详细机制对比

### 离线推理 (Offline) 流程
1.  **初始化**: 编码 BOS, TimeSig, BPM 等全局 Token。
2.  **预加载**: 读取整首曲子的 Melody (Part 0) 到内存。
3.  **循环生成**:
    *   `position=0`: 注入当前拍的 Melody Token。
    *   `position=1`: 模型自回归生成当前拍的 Acc Token。
    *   **关键**: `past_key_values` 持续传递给下一次迭代，模型“记得”从开头到现在的所有内容。
4.  **结束**: 当所有 Melody 处理完或达到最大 Token 限制。

### 引擎推理 (Engine) 流程
1.  **接收请求**: 收到新的 Melody Notes 和生成指令。
2.  **更新历史**: 将新 Notes 追加到 `self.melody_history`。
3.  **构建窗口**:
    *   计算当前生成位置 `current_beat`。
    *   回溯 `context_beats` (如 32 拍)，截取这段时间内的 Melody 和 Acc 历史。
    *   **重新 Token 化**: 将这段历史重新转换为 Token 序列 `[BOS, ..., Bar, Mel, Bar, Acc, ...]`.
4.  **生成**:
    *   将构建好的 Prompt 输入模型。
    *   模型重新计算整个 Prompt 的 Attention (Prefill)。
    *   生成当前拍的 Acc Token。
5.  **返回**: 解码 Token 为 Notes，返回结果，**丢弃** 显存中的 KV Cache。

## 4. 总结与建议

*   **Offline 方案** 适合生成质量要求最高、且不需要实时的场景。因为它能看到最完整的历史信息。
*   **Engine 方案** 是为了实时性做的妥协。
    *   **优点**: 显存占用稳定（不会随曲子变长而爆炸），延迟稳定。
    *   **缺点**: 丢失了很久以前的上下文（超过 32 拍之前的）；计算有冗余（重复计算历史）。

**改进建议**:
如果 Engine 推理发现长结构（如乐句重复）生成效果不佳，可以尝试增大 `context_beats` 参数（例如从 32 增加到 64 或 128），前提是显存和计算延迟允许。
