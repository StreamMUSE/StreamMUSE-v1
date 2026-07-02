---
title: 一致性测试（Realtime vs Offline）
description: 验证实时模拟与离线生成在确定性参数下逐拍一致的金标准端到端测试
---

# 一致性测试（Realtime vs Offline）

`tests/consistency/` 是系统的**金标准端到端回归测试**：在确定性（贪心）采样下，真实的实时链路（`streammuse-cli` → 3 线程 `RealTimeMusicService` → HTTP → Lekai server）与离线一次性生成（`scripts/run_lekai_offline.py`），应产出**完全一致的伴奏**。

它默认跳过，需要真实 checkpoint + GPU，单次运行数分钟。

## 怎么跑

```bash
# 默认：song 4 × tempo {15, 120}，约 5 分钟
LEKAI_CHECKPOINT_PATH=models/ModelLekai/epoch_4_1104_1204/model.safetensors \
  uv run pytest tests/consistency/ -v

# 指定 GPU（不污染调用方环境）
STREAMMUSE_CONSISTENCY_GPU=1 LEKAI_CHECKPOINT_PATH=... uv run pytest tests/consistency/ -v

# 发版前全量：3 首非空歌 × 4 个 tempo 档（约 40 分钟）
STREAMMUSE_CONSISTENCY_SONGS=4,5,2 STREAMMUSE_CONSISTENCY_TEMPOS=120,90,60,15 \
  LEKAI_CHECKPOINT_PATH=... uv run pytest tests/consistency/ -v
```

不设 `LEKAI_CHECKPOINT_PATH` 时整目录 skip，日常 `uv run pytest tests/` 不受影响。

## 它断言什么

对每个 `(歌, tempo)`：

1. **零丢弃前置**：实时推理没有因为跟不上调度而丢请求（从 `inferences.json` 的 `generation_start_tick` 是否连续步进检测）。违反即报 *run invalid: inference too slow*，是性能问题不是一致性回归。
2. **零迟到调度前置**：读取 `model_schedule_trace.jsonl`，确认没有 `clamped_partial_note`、`dropped_past_note`、`forced_note_off` 等 late scheduling policy。准点的跨 batch `current_isolated_note_off` 允许出现。
3. **断言一**：实时输出 == 离线输出，在 pianoroll `(beat, pitch)` 层、截断到旋律窗口内，完全相等。
4. **断言二**：同一首歌不同 tempo 的实时输出两两相同——证明时钟速度不会泄漏进生成内容。

## 三个必须遵守的设计约束（踩坑换来的）

这些不是随意选择，违反任何一条都会让测试要么假绿要么假红。详见 `developing-logs/plans/2026-06-12-consistency-final-test-plan.md` 的 Phase 0 记录。

### 1. 只用非空歌

贪心解码（top_k=1, temp=0）下，模型对**很多歌会生成全空的伴奏**（这是已知模型行为，非 bug）。空对空的比较毫无意义。实测各歌非空程度：

| song | condition_idx | 非空拍数 | 适合 |
|---|---|---|---|
| 4 | 4 | 56/76 | ✅ 默认 |
| 5 | 1 | 19/116 | ✅ |
| 2 | 0 | 9/96 | ✅ |
| 3 | 2 | 1/151 | ⚠️ 几乎空 |
| 1 | 3 | 0/141 | ❌ 全空 |

> 注意 `condition_idx` 与歌曲编号**不相等**：`PianoDataset` 按 `[2,5,3,1,4].npz` 排序，映射为 `{1:3, 2:0, 3:2, 4:4, 5:1}`。

### 2. 在 pianoroll 层比，不在 MIDI note 层比

实时链路把持续音在**每个拍边界重新触发**（note_off + note_on），离线保持为**一个长音符**。两者是同一个 pianoroll，但 raw `(tick, pitch, type)` 事件对比会有大量假性 mismatch（实测仅 ~78%）。`midi_pianoroll.py` 把每个音符展开成它覆盖的 `(beat, pitch)` 格子集合再比，归一化掉这个表示差异。

### 3. 截断到旋律窗口

旋律 MIDI 通常比 `--max-ticks` 短。实时跑过旋律末尾后，在没有旋律可条件的区间继续挂音，而离线在数据末尾就停。必须把对比截断到旋律最后一拍（`SongSpec.melody_last_beat`），否则尾部会假性 mismatch。

> 滑窗（`LEKAI_PROMPT_CONTEXT_BEATS`）**不是**分叉源——全 context 与默认 32 拍结果完全相同。

## BPM 一致性

BPM 是模型的条件 token（prompt 开头）。两侧必须落在同一个 `encode_bpm` 桶（`<90` 慢 / `90–200` 中 / `>200` 快），否则第一个 token 就不同、后面全部分叉。测试两侧都钉 `LEKAI_DEFAULT_BPM=120`（server）+ `--bpm 120`（offline，新增的直传口子），与各歌原生 BPM 无关。

## tempo 阶梯的诊断语义

`--tempo` 只控制 wall-clock 节奏，不影响生成内容，所以可以放慢时钟消除"推理赶不上"的干扰：

- **tempo 15 红** → 真一致性回归，或当前机器连慢速金标准都触发了 late scheduling；
- **tempo 15 绿、tempo 120 红** → 推理速度跟不上实时（环境/性能），非系统 bug；
- 全绿 → 一致 且 当前机器能跑满 120 BPM 实时。

## 红了怎么查

1. 先看是不是 *run invalid: inference too slow* 或 late scheduling policy → 换空闲 GPU 或只跑 tempo 15。
2. 若是 late scheduling，打开 session 目录下的 `model_schedule_trace.jsonl`，看 `logical_tick`、`scheduled_tick` 和 `policy`。
3. 看断言一的 `cmp.summary()`：`only_in_realtime` / `only_in_offline` 哪边多了/少了，落在哪些 `(beat, pitch)`。
4. 若差异在尾部 → 检查旋律窗口截断是否正确。
5. 若差异在开头 → 大概率 BPM token 不一致，核对两侧 `[PROMPT_DEBUG]` 的第 3 个 initial token。
6. 工件都在 `output/consistency/<timestamp>/`（server.log、各 session 目录、offline 输出），可直接复盘。
