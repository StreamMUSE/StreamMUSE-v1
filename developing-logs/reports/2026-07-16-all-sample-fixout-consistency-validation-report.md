# Lekai Fixout 全样本 Offline / Realtime 一致性验证报告（2026-07-16）

## 1. 验证目的

此前的修复报告只把 `inputs_lekai/4`、`inputs_lekai/5` 作为正式 GPU consistency
样本，并额外手工检查了 `nyan_cat` 和 `river_flows`。这些结果不足以证明整个样本集
都没有遗漏。

本轮不以 pytest 是否通过作为结论，而是重新运行调查报告定义的全部 15 个真实样本：

- `prompts/inputs_lekai`：5 首；
- `prompts/old_input`：10 首。

每首都实际执行以下两个生产入口：

1. `scripts/run_lekai_offline.py` 生成新的 offline reference；
2. MIDI-file input -> `RealTimeMusicService` -> HTTP Lekai server，生成完整 realtime session。

然后直接读取产出的 MIDI、`inferences.json` 和 `model_schedule_trace.jsonl` 做 artifact
级比较。pytest 未参与本轮 pass/fail 判定。

## 2. 最终结论

结论必须分成“模型生成逻辑”和“实际 wall-clock 播放”两层。

### 2.1 模型生成逻辑

在默认 `128-beat` realtime context 下，120 BPM 全量运行的
`theoretical_model.mid` 与 offline reference 为：

```text
15 / 15 完全一致
```

这表示 server 实际生成并交给 scheduler 的 accompaniment 时间线，在全部 15 个样本上
都和 offline 一致。此前修复的 BAR raw-token history、小节边界 slot 和 offline
`delay_beats=-1` 解码逻辑没有再发现样本级分叉。

### 2.2 实际 wall-clock 播放

120 BPM 下：

```text
theoretical vs offline: 15 / 15
combined vs offline:    13 / 15
严格有效运行:           10 / 15
```

15 BPM 金标准时钟下：

```text
theoretical vs offline: 15 / 15
combined vs offline:    15 / 15
dropped request:         0
late schedule event:     0
严格有效运行:           15 / 15
```

因此，本轮可以证明：

- offline / realtime 的模型推理逻辑在全样本上已经一致；
- scheduler 能准时拿到 response 时，实际播放 MIDI 也在全样本上完全一致；
- 不能声称 120 BPM 的实际播放已经全样本一致。`nyan_cat` 和 `river_flows` 仍会因
  response 超过一个 tick 的 deadline 而触发 late recovery，改变 `combined.mid`；
- 120 BPM 的剩余问题是 realtime latency / lookahead 问题，不是 server 自回归上下文
  再次分叉。

## 3. 固定实验配置

### 3.1 Checkpoint

```text
/data/home/bowenzheng/mbzuai-projects/models/ModelLekai/epoch_4_1104_1204/model.safetensors
SHA-256: 905819e7ac7ac4864e5cc0308b87b04eec10a9b552aa506cec534fdd7567558b
```

所有 server 的 `/runtime_info` 均确认：

```text
mode=real_model
resolved_device=cuda
resolved_dtype=float16
prompt_context_beats=128
history_retention_ticks=512
temperature=0.0
top_k=1
top_p=0.0
repetition_penalty=1.2
```

被验证的 `lekai_http_backend.py` runtime source SHA-256 为：

```text
ec7b67ad6788407c10f2fa9bced88769d1c9d245a6b093fe3d4cf00ba6b90662
```

### 3.2 共同推理参数

```text
model conditioning BPM = 120
generation interval ticks = 4
generation length frames = 4
ticks per beat = 4
offline gt prefix beats = 0
offline/realtime sampling = greedy fixout
tail beats = 24
```

Realtime 播放 tempo 分三层：

- 120 BPM：15 首全部运行，检查正常 realtime 速度；
- 60 BPM：只对 120 BPM 的 5 个失败/无效项做诊断复验；
- 15 BPM：15 首全部重新运行，作为不受常规 deadline 干扰的金标准矩阵。

### 3.3 GPU 隔离

运行前 8 张 H200 均空闲。Realtime 阶段使用 GPU 0-7，每张 GPU 一个独立 Lekai
server，sample 以长度从长到短进入动态队列。没有让多个 server 共享同一张 GPU。

## 4. 比较方法

### 4.1 三类 artifact

每首比较：

1. `offline/*_generated.mid`：offline one-shot reference；
2. `session/theoretical_model.mid`：server 返回给 scheduler 的逻辑 tick 时间线；
3. `session/combined.mid`：经过 wall-clock scheduler、late recovery 后实际录制的时间线。

MIDI 统一展开成旋律窗口内的 `(model_timestep, pitch)` active-cell 集合，每拍保留
4 个 model timestep。这样既会归一化 offline 长持续音和 realtime 每拍重触发之间的
表示差异，也不会像旧 pytest 的 `(beat, pitch)` 比较那样掩盖同一拍内部的 tick 偏移。

### 4.2 有效性检查

除了 MIDI cell，还检查：

- `inferences.json` 中 `generation_start_tick` 是否每次严格增加 4；
- 是否因 latest-only queue 出现 dropped request；
- `model_schedule_trace.jsonl` 是否出现 late/clamped/dropped/forced policy；
- theoretical 和 combined 的差异是否与这些 scheduling policy 对齐。

严格 pass 条件为：

```text
theoretical == offline
AND combined == offline
AND dropped == 0
AND late == 0
```

## 5. 120 BPM 全样本结果

| Sample | Offline model-step cells | Theoretical | Combined | Dropped | Late | 严格结果 |
|---|---:|---:|---:|---:|---:|---|
| inputs_lekai/1 | 0 | 0/0 | 0/0 | 0 | 0 | PASS |
| inputs_lekai/2 | 0 | 0/0 | 0/0 | 0 | 0 | PASS |
| inputs_lekai/3 | 0 | 0/0 | 0/0 | 0 | 0 | PASS |
| inputs_lekai/4 | 212 | 212/212 | 212/212 | 0 | 2 | INVALID |
| inputs_lekai/5 | 80 | 80/80 | 80/80 | 0 | 0 | PASS |
| old_input/001 | 0 | 0/0 | 0/0 | 0 | 0 | PASS |
| old_input/002 | 0 | 0/0 | 0/0 | 6 | 0 | INVALID |
| old_input/003 | 0 | 0/0 | 0/0 | 0 | 0 | PASS |
| old_input/004 | 0 | 0/0 | 0/0 | 0 | 0 | PASS |
| old_input/005 | 0 | 0/0 | 0/0 | 1 | 0 | INVALID |
| old_input/nyan_cat | 393 | 393/393 | 359/427 | 0 | 158 | FAIL |
| old_input/princess_mononoke | 0 | 0/0 | 0/0 | 0 | 0 | PASS |
| old_input/river_flows | 350 | 350/350 | 349/351 | 0 | 3 | FAIL |
| old_input/rush_e | 0 | 0/0 | 0/0 | 0 | 0 | PASS |
| old_input/spirited_away | 0 | 0/0 | 0/0 | 0 | 0 | PASS |

这里的 `INVALID` 表示 MIDI 内容碰巧相同，但该 run 发生 dropped/late，不能作为干净的
实时一致性证据。

需要特别区分：

- 11 首 offline 输出为空，空对空属于有效但较弱的证据；
- 4 首非空样本为 `inputs_lekai/4`、`inputs_lekai/5`、`nyan_cat`、
  `river_flows`；
- 这 4 首的 theoretical 全部 100%；
- actual combined 在 `nyan_cat` 和 `river_flows` 上被 late recovery 改变。

## 6. 60 BPM 诊断复验

针对 120 BPM 的 5 个失败/无效项，各自独占 GPU 复跑：

| Sample | Theoretical | Combined | Dropped | Late | 结果 |
|---|---:|---:|---:|---:|---|
| inputs_lekai/4 | 212/212 | 212/212 | 0 | 0 | PASS |
| old_input/002 | 0/0 | 0/0 | 1 | 0 | INVALID |
| old_input/005 | 0/0 | 0/0 | 0 | 0 | PASS |
| old_input/nyan_cat | 393/393 | 392/394 | 0 | 3 | FAIL |
| old_input/river_flows | 350/350 | 349/351 | 0 | 3 | FAIL |

失败请求的实测延迟：

| Sample | generation start tick | round-trip | 60 BPM tick budget | 后果 |
|---|---:|---:|---:|---|
| nyan_cat | 272 | 344.92 ms | 250 ms | response 在 tick 273 处理；timestep 272 的 pitch 95 被 pitch 90 替代 |
| river_flows | 308 | 324.44 ms | 250 ms | response 在 tick 309 处理；timestep 308 的 pitch 52 被 pitch 47 替代 |
| old_input/002 | 916 | 2034.21 ms | 250 ms | 超过 1 秒 beat interval，合并掉一个请求 |

这三组数据证明 60 BPM 仍不一定能为所有长 context 请求提供足够预算。

## 7. 为什么一个约 300 ms 的请求会在 60 BPM 迟到

当前主循环在 beat 尾部发送下一 beat 的请求：

```python
if tick > 0 and tick % ticks_per_beat == ticks_per_beat - 1:
    next_generation_tick = tick + 1
    self._enqueue_inference_request(next_generation_tick, notes_for_next_request)
```

因此虽然每隔一个 beat 发一次请求，真正的 lookahead 只有一个 tick：

| Realtime tempo | 每 beat 时长 | 每 tick 时长 | response 的目标预算 |
|---|---:|---:|---:|
| 120 BPM | 500 ms | 125 ms | 约 1 tick |
| 60 BPM | 1000 ms | 250 ms | 约 1 tick |
| 15 BPM | 4000 ms | 1000 ms | 约 1 tick |

超过该窗口后，scheduler 会 clamp、drop 或恢复 partial/open note。`theoretical_model.mid`
保留模型的逻辑 tick，`combined.mid` 则记录恢复后的实际 tick，所以两者会产生差异。

## 8. 15 BPM 全样本金标准结果

| Sample | Offline model-step cells | Theoretical | Combined | Dropped | Late | 结果 |
|---|---:|---:|---:|---:|---:|---|
| inputs_lekai/1 | 0 | 0/0 | 0/0 | 0 | 0 | PASS |
| inputs_lekai/2 | 0 | 0/0 | 0/0 | 0 | 0 | PASS |
| inputs_lekai/3 | 0 | 0/0 | 0/0 | 0 | 0 | PASS |
| inputs_lekai/4 | 212 | 212/212 | 212/212 | 0 | 0 | PASS |
| inputs_lekai/5 | 80 | 80/80 | 80/80 | 0 | 0 | PASS |
| old_input/001 | 0 | 0/0 | 0/0 | 0 | 0 | PASS |
| old_input/002 | 0 | 0/0 | 0/0 | 0 | 0 | PASS |
| old_input/003 | 0 | 0/0 | 0/0 | 0 | 0 | PASS |
| old_input/004 | 0 | 0/0 | 0/0 | 0 | 0 | PASS |
| old_input/005 | 0 | 0/0 | 0/0 | 0 | 0 | PASS |
| old_input/nyan_cat | 393 | 393/393 | 393/393 | 0 | 0 | PASS |
| old_input/princess_mononoke | 0 | 0/0 | 0/0 | 0 | 0 | PASS |
| old_input/river_flows | 350 | 350/350 | 350/350 | 0 | 0 | PASS |
| old_input/rush_e | 0 | 0/0 | 0/0 | 0 | 0 | PASS |
| old_input/spirited_away | 0 | 0/0 | 0/0 | 0 | 0 | PASS |

本轮最慢的单次 inference 为 `old_input/002` 的 `1063.80 ms`。该样本输出为空，
所以没有需要 late recovery 的 note event；同时 15 BPM 的 beat interval 为 4 秒，worker
没有积压到下一 beat，因此没有 dropped request。

## 9. 可复现 runner

新增独立脚本：

```text
scripts/run_all_sample_consistency.py
```

120 BPM 全量命令：

```bash
.venv/bin/python scripts/run_all_sample_consistency.py \
  --output-dir output/consistency/all_samples_20260716-full120 \
  --gpus 0,1,2,3,4,5,6,7 \
  --tempo 120
```

15 BPM 全量命令：

```bash
.venv/bin/python scripts/run_all_sample_consistency.py \
  --output-dir output/consistency/all_samples_20260716-full15 \
  --gpus 0,1,2,3,4,5,6,7 \
  --tempo 15
```

脚本会：

- 校验 NPZ/MIDI stem 一一对应且总数严格为 5 + 10；
- 重新生成当前代码版本的 offline reference；
- 每张 GPU 启动独立 server；
- 保存每个 CLI log、server log、runtime info 和完整 session；
- 每首立即写一个 result JSON，避免中断后丢失已完成结果；
- 输出 `summary.json` 和 `summary.md`。

## 10. Artifact 位置

120 BPM 全量：

```text
output/consistency/all_samples_20260716-full120/
```

60 BPM 失败项诊断：

```text
output/consistency/all_samples_20260716-failures60/
```

15 BPM 全量金标准：

```text
output/consistency/all_samples_20260716-full15/
```

每个目录的关键文件：

```text
manifest.json
summary.json
summary.md
offline/<dataset>/*_generated.mid
results/<dataset>__<sample>.json
realtime/<dataset>__<sample>/tempo_<bpm>/<date>/session_<time>/
servers/gpu_<n>/runtime_info.json
servers/gpu_<n>/server.log
```

## 11. 对上一份修复报告的修正

上一份报告中“GPU consistency 四组全部通过”只描述 song 4/5，不应被理解为整个样本集
已经验证。本报告补齐了缺失覆盖，并给出更严格的最终表述：

1. 全部 15 首的 server theoretical inference 与 offline 一致；
2. 全部 15 首在 scheduler 赶得上的 15 BPM 条件下，combined 与 offline 一致；
3. 120 BPM 下 actual combined 仍不是全量一致，剩余问题属于 deadline/lookahead；
4. 后续若要把“120 BPM combined 15/15”作为目标，需要单独优化请求提前量、模型延迟或
   KV/cache 路径，不能再通过少量 sample 的 pytest 通过来替代全量 artifact 验证。
