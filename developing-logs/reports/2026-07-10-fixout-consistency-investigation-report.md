# Offline vs Realtime Fixout 一致性调查报告（2026-07-10）

## 背景与问题

用户手头有两批贪心（fix-output：temperature=0, top_k=1, top_p=0）产出：

- `output/offline_fixout/`——`run_lekai_offline.py` 一次性生成（inputs_lekai 5 首 + old_input 10 首，`--bpm 120 --gt-prefix-beats 0`）；
- `output/realtime_sim_full/fixout/`——真实 `RealTimeMusicService`（`--input-mode midi_file`，tempo 120，`LEKAI_DEFAULT_BPM=120`，同贪心 env）。

理论上两者应完全一致（金标准测试历史上验证过），问题：**现在是不是完全不相同了？**

## 结论（TL;DR）

1. **不是"完全不相同"**：15 首中 **12 首在 pianoroll 层完全一致**（含金标准 song 4，53/53 cells 100% match）；零丢请求、零迟到调度。
2. **3 首真实分叉**：song 5、nyan_cat、river_flows——全部是"贪心下处于空塌缩边缘"的歌。
3. **根因已定位到具体代码行**：realtime 后端（[lekai_http_backend.py:423-427](../../src/streammuse/infrastructure/inference/lekai_http_backend.py)）会把模型生成的 **bar token (255) 剥掉并替换成空拍标记 `[169]`** 再放回自回归上下文；offline（`model.py` 生成循环）则把生成的 `[255]` **原样保留**在序列里。两边从该 token 起上下文分叉，贪心级联放大。
4. **这不是新引入的回归**：该替换逻辑写于 **2026-04-09**（commit `ca1b1b8c` "debug, reference to the old system, deeply debug"），早于 6/25 的金标准测试。历史上"验证过完全相同"的只有 song 4——它恰好从不在非小节位置自发吐 bar token，踩不到这条路径。**这三首歌从来就不一致，只是从未被测过。**

## 逐首对比结果（pianoroll 层，旋律窗口内）

方法：`tests/consistency/midi_pianoroll.py` 的 `(beat, pitch)` 归一化对比，窗口截断到各歌旋律最后一拍；同时从 `inferences.json` 检查丢请求（generation_start_tick 连续性）、从 `model_schedule_trace.jsonl` 检查迟到调度策略。

| 歌 | 窗口(拍) | rt cells | off cells | match% | 一致 | drops | late |
|---|---|---|---|---|---|---|---|
| lekai/2 | 74 | 0 | 0 | 100 | ✅ | 0 | – |
| **lekai/5** | 91 | **0** | **37** | **0** | ❌ | 0 | – |
| lekai/3 | 120 | 0 | 0 | 100 | ✅ | 0 | – |
| lekai/1 | 110 | 0 | 0 | 100 | ✅ | 0 | – |
| **lekai/4（金标准）** | 57 | 53 | 53 | **100** | ✅ | 0 | 0 |
| old/rush_e | 162 | 0 | 0 | 100 | ✅ | 0 | – |
| old/002 | 221 | 0 | 0 | 100 | ✅ | 0 | – |
| **old/nyan_cat** | 103 | **0** | **295** | **0** | ❌ | 0 | – |
| old/001 | 271 | 0 | 0 | 100 | ✅ | 0 | – |
| **old/river_flows** | 96 | 48 | 146 | **8.99** | ❌ | 0 | 0 |
| old/003 | 286 | 0 | 0 | 100 | ✅ | 0 | – |
| old/spirited_away | 107 | 0 | 0 | 100 | ✅ | 0 | – |
| old/005 | 290 | 0 | 0 | 100 | ✅ | 0 | – |
| old/004 | 202 | 0 | 0 | 100 | ✅ | 0 | – |
| old/princess_mononoke | 108 | 0 | 0 | 100 | ✅ | 0 | – |

**一致 12/15，不一致 3/15。** 注：12 首一致里有 10 首是"空对空"的平凡一致（贪心空塌缩，两边都空——本身就是一种一致）；真正非空且一致的是 song 4。

## 排查链（假设逐个排除的完整过程）

以 song 5 为样本（offline 37 cells 从 beat 1 开始，realtime 全空）：

### 假设 1：滑窗上下文（❌ 排除）

realtime 默认 `LEKAI_PROMPT_CONTEXT_BEATS=32`，offline 全上下文。将 realtime 以 `LEKAI_PROMPT_CONTEXT_BEATS=200`（覆盖全曲）重跑 song 5 → **依然 0 cells，0% match**。滑窗不是原因。（song 4 当年也验证过 32 vs 200 无差别。）

### 假设 2：两边喂的旋律不一致（❌ 排除，两个粒度）

realtime 吃 `mel/5.mid`，offline 吃 `5.npz` 的 part0，且分叉三首的旋律长度都非整拍（91.8 / 103.5 拍），可疑。对比"模型实际看到的旋律"：

- **拍级** `(beat, pitch)`：song5 / song4 / nyan_cat 全部 **100% 一致**（191=191、84=84、288=288 cells）；
- **sub-beat 级** `(model_tick, pitch)`（每拍 4 个 timestep，token 的真实分辨率）：song5 / song4 / nyan_cat / river_flows 全部 **100% 一致**。

旋律输入在任何分辨率上都相同。

### 假设 3：小节结构（pickup 弱起小节）（❌ 排除）

检查全部 npz 的 measure 结构：**所有歌所有 measure 都是整 16 timesteps（4 拍）**，无弱起小节。

### 决定性手段：token 级 diff（✅ 找到分叉点）

两边重跑 song 5 并把 generation log 重定向到独立目录（`LEKAI_OFFLINE_LOG_DIR` / `LEKAI_RT_LOG_DIR`），逐 token 对比：

```
offline  前13: [257, 263, 265, 173, 255, 173, 169, 143, 6, 83, 2, 170, 255, ...]
realtime 前13: [257, 263, 265, 173, 255, 173, 169, 143, 6, 83, 2, 170, 169, ...]
                └────────── 完全一致的 12-token 前缀 ──────────┘   ↑ 第一个分叉
```

第 13 个 token（acc_1 槽位）：offline=**255**（bar），realtime=**169**（空拍）。255 出现在 beat 1——非小节边界，即模型"自发"吐了个 bar token。

### 验证模型意志：直接前向（✅ 排除数值平局）

加载 checkpoint，对该 12-token 前缀分别用两种计算路径求下一 token 的 logits：

| 路径 | top-2 (token, logit) | 255 与 169 差 |
|---|---|---|
| 整段前向（realtime 方式） | (255, 10.50), (169, 9.58) | **+0.922** |
| 增量 KV cache（offline 方式） | (255, 10.50), (169, 9.58) | **+0.922** |

两条路径逐位一致、argmax 都是 **255**、差距 0.92 远非平局——**模型在两边都输出了 255**。fp16 数值噪声假设排除。

### 真凶：生成后处理不对称（✅ 根因）

realtime 后端 `_generate_part1_tokens_from_prompt`（[lekai_http_backend.py:420-428](../../src/streammuse/infrastructure/inference/lekai_http_backend.py)）：

```python
if token_val in {part1_end_marker, part1_empty_marker, bar_token}:
    break
valid_tokens = [token for token in raw_tokens if token != pad_token_id]
if valid_tokens and valid_tokens[-1] == bar_token:   # 生成了 255
    valid_tokens.pop()                                # → 剥掉
if not valid_tokens:
    return [169]                                      # → 替换成空拍标记
```

而 offline（`model.py` 交错生成循环）：生成的 255 终止该拍，且 **`[255]` 原样留在自回归序列里**。

于是：模型两边都吐 255 → offline 上下文含 `..., 255, ...`，realtime 上下文含 `..., 169, ...` → 下一拍起 prompt 不同 → 贪心逐拍级联 → song 5 整体塌空。river_flows 的 9% 部分匹配 = 分叉点出现之前的部分。song 4 全曲从未自发吐过非小节位 bar token，所以 100% 一致。

## 时间线澄清（"以前测过完全相同"为什么和现在不矛盾）

| 时间 | 事件 |
|---|---|
| 2026-04-09 | `ca1b1b8c` 引入 bar-token 剥除替换逻辑（realtime 侧） |
| 2026-06-25 | 金标准一致性测试建成，验证 **song 4** 100%（默认歌单只有 4） |
| 2026-07-03 | 用户跑金标准测试 PASSED（还是 song 4） |
| 2026-07-10 | 15 首全量对比首次覆盖 song5/nyan_cat/river_flows → 暴露分叉 |

**验证过的不变量（song 4）从未回归**；分叉三首是覆盖盲区，其不一致自 4/9 起即存在。

## 修复建议（待拍板）

两边对"模型在非小节位自发吐 bar token"的处理必须统一：

- **方案 1（推荐）：realtime 对齐 offline**——`seq` 保留生成的原始 token（含 255），仅在 pianoroll 解码出事件时按空拍处理。改动局限在 realtime 后端一处；offline 参考实现不动，历史产出可比性保持。
- 方案 2：offline 也做 169 替换——但 offline 是参考实现，动它会让所有历史 offline 产出失去可比性，不推荐。

修复后动作：重跑三首分叉歌验证收敛到 100%；把 **song 5 加进金标准测试歌单**（它是这条路径的天然判别样本，防止将来再回归）。

## 本次调查的工件

- 全量对比脚本输出：本报告表格（内联运行，未落盘）
- song 5 全上下文重跑 session：`/tmp/rt_s5_ctx200/`
- token diff 用的 gen logs：`/tmp/diag_off/song_001/`（offline）、`/tmp/diag_rt/fake_rt_gen_*.json`（realtime）、产物 `/tmp/diag_out/`
- 被对比的原始两批产出：`output/offline_fixout/`、`output/realtime_sim_full/fixout/`

> `/tmp` 下的诊断工件重启会丢；如需存档可拷进 `output/`。
