# Offline vs Realtime Fixout 一致性修复报告（2026-07-16）

## 1. 结论摘要

本次工作针对调查报告
`developing-logs/reports/2026-07-10-fixout-consistency-investigation-report.md`
中记录的 single-stage Lekai offline / MIDI-file realtime simulation 不一致问题进行了代码级复核、修复和完整回归。

最终结论如下：

1. 调查报告定位的核心 bug 是正确的：realtime 曾把模型生成的 `BAR(255)` 删除并替换为 `EMPTY(169)`，导致下一轮自回归上下文与 offline 分叉。
2. 报告建议的“一行删除替换逻辑”不足以完整修复。server 之前只保存 MIDI event；BAR 是静音结构 token，event round-trip 无法恢复它，因此必须额外保存精确 raw token history。
3. `delay_beats=-1` 在小节边界存在两个相邻 part1 slot。需要区分“当前可播放 beat”和“part0 BAR 后的结构 slot”，否则 offline MIDI 与 realtime 都会发生拍位错位。
4. 原报告关于 context window 的排除结论只对当时验证的 song 4/5 成立，不能外推。`river_flows` 在默认 32-beat window 下从 beat 36 开始分叉；context 200 恢复 100%。最终将默认 context 调整为 128 beats，与默认 512-tick history retention 对齐。
5. MIDI-file input 还有一个独立时钟 race：旧代码在解析 MIDI 之后才建立 input 起点，解析耗时会把输入整体向后平移。现在改为解析前锚定起点。

最终 GPU 金标准结果：

| Case | Realtime cells | Offline cells | Matched | Dropped | Late | 结果 |
|---|---:|---:|---:|---:|---:|---|
| song 4, 15 BPM | 53 | 53 | 53 / 53 | 0 | 0 | 100% |
| song 4, 120 BPM | 53 | 53 | 53 / 53 | 0 | 0 | 100% |
| song 5, 15 BPM | 35 | 35 | 35 / 35 | 0 | 0 | 100% |
| song 5, 120 BPM | 35 | 35 | 35 / 35 | 0 | 0 | 100% |

CPU / unit 全套结果为 **257 passed, 2 skipped**；GPU consistency 四组全部通过。

## 2. 调查报告复核

### 2.1 报告判断正确的部分

报告给出的首个 token 分叉是决定性证据：

```text
offline : ..., 170, 255, ...
realtime: ..., 170, 169, ...
                    ^ first divergence
```

旧 realtime 后处理为：

```python
valid_tokens = [token for token in raw_tokens if token != pad_token_id]
if valid_tokens and valid_tokens[-1] == bar_token:
    valid_tokens.pop()
if not valid_tokens:
    return [169]
```

这确实改变了模型实际生成的 token，且会从下一轮请求开始级联改变所有 logits。song 5 是这条路径的天然判别样本。

### 2.2 报告建议不完整的部分

只把上述逻辑改成 `return raw_tokens` 仍不够。旧 server 在请求之间只保留：

```text
raw tokens -> pianoroll -> MIDI events -> server accompaniment_history
```

而 `[255]` 解码为一个静音 beat，不产生 note event。下一次请求若从 event history 重新编码，无法知道上一轮静音来自：

- 模型原始生成的 `BAR(255)`；
- `EMPTY(169)`；
- 普通的无 note beat。

因此下一轮仍会丢掉 255。修复必须把 raw tokens 作为一等状态保存，而不能只修当前 response 的 decode。

### 2.3 报告中需要修正的结论

原报告用 song 5 将 `LEKAI_PROMPT_CONTEXT_BEATS` 从 32 提高到 200，结果仍然不一致，于是排除了 context window。这个实验对 song 5 本身成立，但不足以证明所有歌曲都对窗口不敏感。

本次对 `river_flows` 的控制实验如下：

| 配置 | Realtime theoretical | Offline | Matched | 说明 |
|---|---:|---:|---:|---|
| context 32 | 145 | 145 | 58 / 145（25.0%） | 从 beat 36 分叉 |
| context 200 | 145 | 145 | 145 / 145（100%） | 完全恢复 |
| 最终默认 context 128 | 145 | 145 | 145 / 145（100%） | 无需额外 env |

因此 river_flows 的旧不一致是两个问题叠加：BAR/token-history bug 加上过短 context。

## 3. `delay_beats=-1` 的真实序列语义

offline one-shot 并不是简单的：

```text
ACC_0, MEL_0, ACC_1, MEL_1, ...
```

它的开头和小节边界包含额外结构位置。简化后的顺序为：

```text
BOS, TS, BPM,
part0_PAD, generated_acc_-1,
part0_PAD, generated_acc_0,
mel_0, generated_acc_1,
mel_1, ...,
mel_3, generated_pre_boundary_slot,
part0_BAR, generated_post_bar_slot,
mel_4, generated_acc_5, ...
```

在小节边界，`generated_pre_boundary_slot` 有两种合法情况：

| pre-boundary slot | 可播放内容 | post-BAR slot |
|---|---|---|
| 音乐 token | pre-boundary slot | 只用于结构上下文 |
| `[255]` | post-BAR slot | 当前 beat 的可播放内容 |

这也是为什么不能简单地在每个四拍边界强制插入固定 `[BAR, BAR]`，也不能把两个 slot 都当成两个播放 beat。

## 4. 实现改动

### 4.1 保存精确 accompaniment token history

文件：`src/streammuse/infrastructure/inference/lekai_http_backend.py`

新增两份 server-side token 状态：

```python
self._accompaniment_token_history: Dict[int, List[int]] = {}
self._accompaniment_bar_token_history: Dict[int, List[int]] = {}
```

- `_accompaniment_token_history[beat]` 保存该播放 beat 的原始模型输出；
- `_accompaniment_bar_token_history[beat]` 保存小节边界注入 part0 BAR 后生成的额外 part1 slot；
- 下一次 HTTP 请求重建 prompt 时优先使用 raw token，不再由 MIDI event 反推；
- `inject_history()`、`clear_history()` 和 `_trim_histories()` 同步维护这些状态。

### 4.2 分离“模型上下文 token”和“可播放 token”

模型生成函数现在原样返回 raw token：

```python
return raw_tokens or [part1_empty_marker]
```

只有进入 pianoroll decoder 时才过滤 transformer PAD：

```python
@staticmethod
def _playable_part1_tokens(raw_tokens: List[int]) -> List[int]:
    playable = [token for token in raw_tokens if token != 258]
    return playable or [169]
```

关键点是：`BAR(255)` 仍留在自回归上下文中，同时在播放层被解释为一个完整的静音 beat，而不是被压缩掉。

### 4.3 小节边界的条件式第二次生成

边界处先生成 pre-boundary slot：

```python
generated_beat_tokens = self._generate_part1_tokens_from_prompt(prompt_tokens, ...)
generated_is_bar = (
    self._playable_part1_tokens(generated_beat_tokens) == [bar_token]
)
```

- 若 pre-boundary 是 BAR，post-BAR slot 是当前 beat 的声音，必须同步生成；
- 若 pre-boundary 已经是音乐，当前 response 可立即返回，该结构 slot 由单 worker executor 在两次 beat request 之间生成；
- 下一次请求开始时 `_resolve_pending_boundary_generations()` 收取结果，再重建精确 prompt；
- model forward 用 lock 串行化，避免主请求与 background slot 同时访问同一个模型实例。

该设计避免了每个小节边界都在 response critical path 中强制做两次完整推理。

### 4.4 修复 offline `delay_beats=-1` MIDI 解码

文件：`src/streammuse/infrastructure/inference/lekai_model/Token2Midi.py`

旧 decoder 先 flatten 全部 slot，再删除 `>=173` 的 token。这样会删除 BAR 并压缩时间，也无法区分结构 slot 和播放 slot。

新增 `_delay_minus_one_playable_part1_beats()`，按 slot 逐个建立因果播放时间线：

```python
if has_boundary_slot:
    if tokens == [255]:
        tokens = boundary_tokens or [169]
    playable.append(tokens)
    index += 2
    continue
```

`model.py` 在 result metadata 中增加 `delay_beats`，仅当 `delay_beats == -1` 时走新 decoder；其他模式保留旧行为。

因此修复后 song 5 / nyan_cat / river_flows 的 offline cell 数与旧 artifact 可能略有变化。这不是采样 token 改变，而是旧 MIDI renderer 的非因果压缩被修正：

- song 5：旧报告 37 cells，修复后 35 cells；
- nyan_cat：旧报告 295 cells，修复后 293 cells；
- river_flows：旧报告 146 cells，修复后 145 cells。

### 4.5 默认 context 从 32 beats 调整为 128 beats

最终默认值：

```python
TIMESTEPS_PER_BEAT = 4
DEFAULT_PROMPT_CONTEXT_BEATS = 128
DEFAULT_HISTORY_MAX_TICKS = DEFAULT_PROMPT_CONTEXT_BEATS * TIMESTEPS_PER_BEAT
```

这样 prompt context 与 server 默认保留的 512 ticks 对齐。仍可用以下环境变量显式覆盖：

```bash
LEKAI_PROMPT_CONTEXT_BEATS=64
LEKAI_HISTORY_MAX_TICKS=256
```

但若 context 小于 one-shot offline 的有效历史，模型输出可以合法地不同。developer guide 已补充该约束。

### 4.6 修复 MIDI-file simulation 的输入起点 race

旧顺序：

```text
parse MIDI -> start_time = now() -> emit events
```

新顺序：

```python
start_time = self._now()
notes, ... = self._midi_to_notes(...)
```

解析耗时现在计入模拟时间，不会把 MIDI 输入相对于 `RealTimeMusicService.timeline_start_time` 整体推迟。回归测试注入 200ms 解析耗时，验证第一条 event 的 sleep 会相应减少 200ms。

### 4.7 扩大 consistency 判别样本

`tests/consistency/test_realtime_offline_consistency.py` 的默认歌曲由 song 4 改为 song 4 + song 5：

```python
DEFAULT_SONGS = "4,5"
DEFAULT_TEMPOS = "15,120"
```

song 4 覆盖原金标准，song 5 专门覆盖模型在非小节位置生成 BAR 的路径。

## 5. 测试覆盖

### 5.1 新增 / 加强的 unit tests

`test_lekai_http_backend.py` 覆盖：

- raw `BAR(255)` 不被改写；
- 下一次请求复用 exact token history；
- 小节边界的 offline slot 顺序；
- pre-boundary 为 BAR 时使用 post-BAR slot 播放；
- pre-boundary 为音乐时 background 生成结构 slot；
- pending future 在 clear 时排空；
- token history 随 history window 裁剪；
- 默认 context 128 与 env override。

`test_token_to_midi.py` 覆盖：

- 非边界 BAR 保留一个静音 beat；
- 边界 BAR / post-BAR slot 正确配对；
- 输出时间轴每个 playable beat 固定占 4 timesteps。

`test_midi_file_input.py` 覆盖：

- MIDI 解析耗时不会整体移动 simulation schedule。

### 5.2 CPU / repository full test

```bash
uv run pytest tests/ -q
```

结果：

```text
257 passed, 2 skipped, 1 warning in 6.77s
```

两个 skip 是未提供 GPU checkpoint 环境变量时的 opt-in consistency cases。随后已单独运行 GPU suite。

### 5.3 最终 GPU consistency matrix

使用的 checkpoint：

```text
/data/home/bowenzheng/mbzuai-projects/models/ModelLekai/epoch_4_1104_1204/model.safetensors
SHA-256: 905819e7ac7ac4864e5cc0308b87b04eec10a9b552aa506cec534fdd7567558b
```

命令模板：

```bash
LEKAI_CHECKPOINT_PATH=/data/home/bowenzheng/mbzuai-projects/models/ModelLekai/epoch_4_1104_1204/model.safetensors \
STREAMMUSE_CONSISTENCY_SONGS=4 \
STREAMMUSE_CONSISTENCY_TEMPOS=15,120 \
STREAMMUSE_CONSISTENCY_GPU=0 \
uv run pytest tests/consistency/test_realtime_offline_consistency.py -v
```

最终结果：

| Case | Match | dropped | late | p95 latency | max latency |
|---|---:|---:|---:|---:|---:|
| song 4 @ 15 | 53 / 53 | 0 | 0 | 87.66 ms | 123.54 ms |
| song 4 @ 120 | 53 / 53 | 0 | 0 | 108.08 ms | 111.66 ms |
| song 5 @ 15 | 35 / 35 | 0 | 0 | 96.76 ms | 320.68 ms |
| song 5 @ 120 | 35 / 35 | 0 | 0 | 67.95 ms | 91.10 ms |

pytest 结果：

```text
song 4: 1 passed in 391.43s
song 5: 1 passed in 542.55s
```

### 5.4 原调查中三个失败样本的复验

| Song / 配置 | Theoretical vs offline | Combined vs offline | dropped | late |
|---|---:|---:|---:|---:|
| song 5，最终 gold @ 120 | 35 / 35 | 35 / 35 | 0 | 0 |
| nyan_cat，120 BPM 并发压力运行 | 293 / 293 | 292 / 293 | 0 | 10 |
| river_flows，旧 context 32 | 58 / 145 | 58 / 145 | 0 | 0 |
| river_flows，context 200 | 145 / 145 | 145 / 145 | 0 | 3 |
| river_flows，最终默认 128 @ 90 | 145 / 145 | 145 / 145 | 0 | 3 |

解释：

- `theoretical_model.mid` 表示 server 实际生成并交给 scheduler 的模型时间线；它用于隔离 token/context 与 wall-clock scheduling。
- nyan_cat 的 theoretical 已 100%，120 BPM 并发运行中一个 cell 因 late recovery 改变，不是模型上下文分叉。
- river_flows 32-beat 的 theoretical 自身就只有 25%，证明它是 context bug；128/200 后 theoretical 恢复 100%。
- river_flows 90 BPM 的三条 late trace 是一对已过期的短 note 加一个 isolated note_off recovery，最终没有改变 pianoroll cells。
- 正式 gold song 4/5 比这些手工压力运行更严格：四组均要求 `late == 0`，并且全部通过。

## 6. Artifact 位置

最终 gold song 4：

```text
output/consistency/20260716-184212/
  offline_song4/004_4_generated.mid
  song4_tempo15/2026-07-16/session_184239/
  song4_tempo120/2026-07-16/session_184802/
```

最终 gold song 5：

```text
output/consistency/20260716-184224/
  offline_song5/001_5_generated.mid
  song5_tempo15/2026-07-16/session_184249/
  song5_tempo120/2026-07-16/session_185029/
```

old_input 深入复验：

```text
output/consistency/20260716-old-input-regression/
```

其中包含 nyan_cat / river_flows 的 offline MIDI、combined MIDI、theoretical MIDI、inferences 和 schedule trace。

## 7. Checkpoint 误用诊断记录

测试过程中曾误用：

```text
/data/home/bowenzheng/mbzuai-projects/models/lekai_continuation_model/model.safetensors
SHA-256: d93139044a8614aeb66c58b5696371a575389199fbe64653b8994d3f6b056271
```

它与 single-stage checkpoint 文件大小相同（681,256,800 bytes），但内容不同，是 two-stage continuation 权重。在 single-stage offline 初始化 prompt 下，其首 token 明确为 PAD：

```text
top-1 PAD(258): 15.875
top-2 token 256: 4.559
```

因此旧 one-shot generator 会连续输出 PAD，最后得到空 accompaniment。该失败不属于本次代码回归，已从正式测试结果中排除。以后复现实验应记录 checkpoint path 和 hash，不能只看文件大小。

## 8. 兼容性与剩余边界

### 8.1 未改变的行为

- 没有修改 temperature、top-k、top-p 或 repetition penalty 语义；
- 没有固定普通运行的随机 seed；
- rule-based fallback 路径未改；
- `delay_beats != -1` 的 legacy MIDI decoder 未改；
- client scheduling / partial-note recovery 策略未改。

### 8.2 有意改变的行为

- realtime prompt 现在保留模型生成的 raw BAR/PAD token；
- `delay_beats=-1` offline MIDI 使用因果 beat mapping，不再删除 BAR 后压缩时间；
- 默认 prompt context 从 32 增加到 128 beats；
- MIDI-file input 的播放时钟从 MIDI 解析前开始。

### 8.3 仍需注意的边界

1. **超过 128 beats 的严格 one-shot 一致性**：默认 server 会在 128 beats 后裁剪 history，而 offline one-shot 仍可保留更长历史。若歌曲对更早上下文敏感，两者之后仍可能不同。可同时提高：

   ```bash
   LEKAI_PROMPT_CONTEXT_BEATS=200
   LEKAI_HISTORY_MAX_TICKS=800
   ```

   但更长 prefill 会增加 latency，并受模型 `max_position_embeddings=3500` 限制。

2. **边界同步第二次生成**：若 pre-boundary slot 本身是 BAR，post-BAR slot 就是当前可播放内容，必须同步完成。密集歌曲上仍可能出现少量 latency outlier。进一步优化需要提前一拍 lookahead / persistent KV cache，不应通过伪造结构 token 解决。

3. **随机采样模式**：本报告的逐结果一致性使用 greedy fixout。普通 temperature sampling 即使分布和逻辑一致，也不保证两次独立进程逐 token 相同。

## 9. 最终判断

调查报告指出的 BAR normalization bug 已确认并修复；同时补齐了 raw token state、小节边界 slot 语义、offline 因果 MIDI decode、MIDI-file 时钟 race 和 context-window 反例。

在最终代码与正确 single-stage checkpoint 下：

- repository full test 全绿；
- song 4 / song 5 的 15、120 BPM realtime combined MIDI 均与 offline 100% 一致；
- 原失败样本 nyan_cat / river_flows 的 server theoretical context 均与 offline 100% 一致；
- `river_flows` 已证明默认 128-beat context 修复了旧 32-beat 的内容分叉。

因此本次 fixout consistency 的模型逻辑回归已完成闭环。剩余差异只出现在个别高负载 wall-clock late scheduling 场景，属于可独立优化的实时性能问题，而不是 offline/realtime 自回归上下文不一致。
