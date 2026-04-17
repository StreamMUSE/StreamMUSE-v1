# Lekai Offline vs Realtime Root-Cause Report

日期: 2026-04-17

## 0. 结论摘要

这次问题不是单点 bug，而是多因素叠加，按影响排序如下：

1. **实验对比不等价（最主要）**：realtime 批跑统一设置了 `max_ticks=256`，而 offline 是整首生成。大多数样本被硬截断，直接导致 melody 少音、accompaniment 听起来“不完整”。
2. **realtime 播放链路有大量“有效事件损耗”**：server 返回的 accompaniment 事件里，实际被播放的比例只有约 22% 到 63%。
3. **offline 与 realtime 并非同一解码路径/同一超参**：offline 和 realtime 使用了不同的生成实现与采样参数，理论上不应期待波形级一致。
4. **oldinput 中 3 首歌出现“全空响应”**：所有请求都返回 0 个 accompaniment 事件，属于 realtime 解码退化现象（与晚起旋律/稀疏上下文强相关）。
5. **melody 的少量额外缺失**：除截断外，还存在 1-3 note 级别边界损失，来自 realtime wall-clock 打点与停止边界。

---

## 1. 分析范围与证据

### 1.1 对比目录

- offline:
  - `output/lekai_offline-prompt0`
  - `output/lekai_offline-prompt0-oldinput`
- realtime:
  - `output/lekai_batch_fixed_inj`
  - `output/lekai_batch_fixed_inj-oldinput`

### 1.2 使用的工件

- `batch_summary.json`（运行参数与每首命令）
- `session_*/inferences.json`（每次请求/响应的事件计数）
- `session_*/melody_history.json`、`accompaniment_history.json`
- `session_*/events.jsonl`
- `session_*/performance.json`
- `session_*/combined.mid`

### 1.3 核心代码路径

- offline 主入口: `scripts/run_lekai_offline.py`
- offline 解码: `src/streammuse/infrastructure/inference/lekai_model/model.py`
- realtime 服务编排: `src/streammuse/application/services/real_time_music_service.py`
- realtime Lekai 后端: `src/streammuse/infrastructure/inference/lekai_http_backend.py`

---

## 2. 关键事实（定量）

## 2.1 realtime 实验配置核对

从两个 `batch_summary.json` 读取到：

- `generation_interval_ticks = 4`
- `generation_length_frames = 4`
- `max_ticks = 256`
- `injection_mode = none`

注意：目录名是 `fixed_inj`，但实际 **没有 injection**。

## 2.2 截断影响非常大

`max_ticks=256` 在 120 BPM、4 ticks/beat 下对应约 32 秒：

$$
T_{sec} = 256 \times \frac{60}{120\times 4} = 32s
$$

- new_input: 5 首中 4 首 `src_max_tick > 256`
- old_input: 10 首中 10 首 `src_max_tick > 256`

这意味着 realtime 实际只跑到前 32 秒，offline 却是整首（最长可到 100+ 秒级）。

## 2.3 melody 保留率（realtime vs 原始）

new_input（节选）：

- `1.mid`: 60/121 (49.6%)
- `3.mid`: 98/219 (44.7%)
- `4.mid`: 52/52 (100.0%, 唯一没被截断)

old_input（节选）：

- `001.mid`: 62/264 (23.5%)
- `003.mid`: 66/422 (15.6%)
- `005.mid`: 67/367 (18.3%)

这与“melody 少音”的主观感受完全一致，且主因是强截断。

## 2.4 oldinput 中 3 首歌出现全空 accompaniment

以下三首在 realtime 中每个请求都返回 0 accompaniment 事件：

- `002.mid`: 43/43 请求为 0
- `003.mid`: 27/27 请求为 0
- `005.mid`: 37/37 请求为 0

对应 `accompaniment_history.json` 与 `combined.mid` 的 accompaniment 轨也为 0。

## 2.5 server 返回事件与实际播放事件差距很大

对比 `inferences.json`（response 事件总数）与 `performance.json`（model_output_events）：

new_input:

- `1.mid`: 182 -> 40 (22%)
- `2.mid`: 182 -> 44 (24%)
- `3.mid`: 358 -> 171 (48%)
- `4.mid`: 181 -> 53 (29%)
- `5.mid`: 466 -> 156 (33%)

old_input（非全空样本）也普遍只有约 31% 到 63% 被真正播放。

这会显著造成“伴奏稀疏/不连贯”的听感。

---

## 3. 离线与实时：逐层对照

## 3.1 输入与停止条件

offline：

- 按数据集 item 一次性生成整首（无 realtime tick 停止约束）。

realtime：

- 服务循环在 `tick >= max_ticks` 时直接停止（`real_time_music_service.py`）。
- 本次批跑统一 `max_ticks=256`，导致大部分歌曲未播完。

## 3.2 模型调用方式

offline：

- 直接调用 `PianoLLaMA.generate_accompaniment(...)`。
- 默认 `delay_beats=-1`，按训练对齐方式交错生成。

realtime：

- 走 `LekaiHttpBackend._generate_with_interleaved_prompt(...)` 自定义路径。
- `generation_length_frames=4` 时每次仅生成 1 beat。
- prompt 上下文限制为最近 `context_beats=32`（默认）。

结论：两者不是同一实现分支。

## 3.3 采样参数不一致

offline 默认（`run_lekai_offline.py`）：

- `temperature=0.8`
- `top_k=50`
- `top_p=0.95`

realtime（`lekai_http_backend.py` 固定写死）：

- `temperature=1.2`
- `top_k=10`
- `top_p=0.9`

这会显著改变输出分布。

## 3.4 realtime 调度导致事件损耗

`real_time_music_service.py` 中：

1. 收到新响应先 `clear_future_events(from_tick=generation_start_tick)`
2. 仅当 `ev.tick >= current_tick` 才会被 schedule

当响应延迟接近窗口大小时，很多“理论已生成”的事件不会进入实际播放轨。

---

## 4. “mel 少音”和“acc 不对”的分解解释

## 4.1 melody 少音

主因：`max_ticks=256` 截断。

次因：realtime 输入线程按 wall-clock 打点（非直接使用 MIDI 原 tick），在边界处会出现 1-3 note 级别偏差。

## 4.2 accompaniment 不对

由三层叠加造成：

1. **只听到前 32 秒**，很多歌仍在 intro/过渡段。
2. **server 返回事件大量未落地播放**（22%-63% 留存）。
3. **realtime 解码路径和采样参数与 offline 不一致**，并在部分样本上退化为全空输出。

---

## 5. 为什么“老系统 interval=4, gen=4 基本赶得上”

当前证据显示，问题不在单一“算力跟不上”，而在**系统行为差异**：

- 本次实验有固定截断（256 ticks）。
- 当前 realtime 的事件调度与回写策略会丢弃一部分生成事件。
- 当前 realtime 解码分支和离线分支并不一致，且超参不同。

因此“同样 interval=4, gen=4”不等于“同样输出质量”。

---

## 6. 根因排序（最终）

1. **配置层根因**：`max_ticks=256` 把比较变成“前 32 秒 realtime” vs “整首 offline”。
2. **系统层根因**：realtime 中响应事件到播放事件有明显损耗。
3. **算法层根因**：realtime 解码路径 + 采样参数与 offline 不一致，导致部分样本空输出。
4. **边界层根因**：wall-clock 打点与停止边界引入小幅 melody 偏差。

---

## 7. 建议的验证顺序（下一轮实验）

1. 先把对比做等价：每首 `max_ticks = src_max_tick + 16`（或更高），禁止 256 固定值。
2. 保持 `interval=4/gen=4` 不变，复跑 old_input 全集，先看“全空输出”是否消失。
3. 将 realtime 采样参数对齐到 offline（0.8/50/0.95）做 A/B。
4. 统计“response->played”留存率，目标至少提升到 >80%。
5. 若留存率仍低，调整调度策略（减少过晚响应丢弃、优化窗口重排）。

---

## 8. 一句话结论

这次“offline 很好、realtime 很差”主要不是模型本体失效，而是**实验不等价 + realtime 播放链路损耗 + 解码路径差异**共同造成的结果。
