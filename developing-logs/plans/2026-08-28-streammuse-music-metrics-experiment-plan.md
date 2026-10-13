# StreamMUSE Short Paper 音乐指标与实验预案

日期：2026-08-28  
状态：指标与实验范围草案  
用途：为 short paper 提供简洁的音乐质量描述，不承担完整的方法有效性论证。

## 1. 范围

本阶段先确定“将来可能报告哪些 music metrics，以及最少需要做哪些实验”。目前不预设一定存在以下方法或 conditions：

- `single_n1`
- `random_n5`
- `rule_s_n5`
- offline/realtime pair

最终比较对象应在论文方法和主张确定后再冻结。若论文只介绍一个系统，就报告该系统的 descriptive metrics 和代表性样例；若论文确实提出一个改动，再增加对应 baseline/proposed comparison。没有实际研究问题时，不人为制造 baseline 或 ablation。

## 2. 可能采用的指标

### 2.1 主表候选

| Metric | 中文含义 | 方向 | 在 short paper 中回答的问题 |
|---|---|---:|---|
| Valid Output Rate | 能成功得到有效 accompaniment 的输出比例 | 越高越好 | 系统是否经常空生成、损坏或提前无输出 |
| FMD | 生成结果与真实音乐在 embedding 分布上的距离 | 越低越好 | 整体音乐分布是否接近真实数据 |
| JSD(Pitch) | 生成与真实 accompaniment 的音高分布差异 | 越低越好 | 使用的音高是否偏离真实数据 |
| JSD(Onset) | 生成与真实 accompaniment 的 onset 分布差异 | 越低越好 | 节奏和发音位置分布是否偏离真实数据 |
| CR | Melody 时长中得到协和 accompaniment 支持的比例 | 越高越好 | accompaniment 是否在和声上支持 Melody |
| UR | Melody 时长中没有 accompaniment overlap 的比例 | 越低越好 | accompaniment 是否存在大量缺失或空段 |
| NLL_wavg | continuation model 对生成序列的加权 NLL | 越低通常越好 | 模型内部认为生成序列是否流畅；只作补充 |

这里的 `Valid Output Rate` 是实验质量门槛，不是 StreamMUSE 论文中的乐理指标，但必须与音乐指标一起报告，否则空输出可能从 FMD/JSD 等计算中被静默排除。

StreamMUSE 的 CR/UR 定义为 duration-weighted：

- Melody 有 accompaniment overlap，且至少存在一个 interval class 属于 `{0,3,4,7,8,9}`：`consonant`；
- Melody 没有任何 accompaniment overlap：`unsupported`；
- 有 overlap 但没有上述协和 interval：`dissonant`。

因此不需要再单独增加一个主指标：

\[
\mathrm{Dissonant\ Rate}=1-\mathrm{CR}-\mathrm{UR}.
\]

### 2.2 可选结构指标

如果 short paper 会提到“重复、单调或长期结构”，只额外选择一个指标：

| 候选 | 用途 | 建议 |
|---|---|---|
| Unique Beat Ratio | 计算前 `t` beats 中首次出现的 beat pattern 数量除以 `t` | 更接近 BEAT 的论文口径；应比较 GT 曲线，不能简单追求更高 |
| Repetition Rate | 统计相邻或跨小节 pattern 重复比例 | 更容易解释和实现，但必须冻结 pattern 与窗口定义 |

short paper 不需要两者都放。若正文不讨论重复问题，两者都可以不做。

### 2.3 暂不作为必做项

- `JS_GC`、`JS_SC`：可作为补充，但主表已有 FMD/JSD，短论文中容易显得指标过多。
- Prompt PPL：适合 Prompt selection 诊断；如果论文不讨论 selection，就不需要。
- 正式大规模人工听评：只有在论文声称“人更喜欢”或“整体听感更好”时才需要。
- Rule-S 的 N 消融、`random_n5` control：只有 Rule-S 成为论文方法和主要主张时才需要。
- ISR、Staleness、latency：由系统评价部分负责，不属于这里的 music metrics。

## 3. 最少需要的实验

### E0. Metric sanity check

这是内部验收，不一定写入正文，也不需要 GPU。

使用少量人工构造 MIDI 检查：

1. generated=reference 时 FMD/JSD 接近 0；
2. 全程协和时 `CR=1, UR=0`；
3. 全程无 accompaniment 时 `CR=0, UR=1`；
4. 空 MIDI、缺失 track、提前停止不会被静默忽略；
5. Prompt、continuation 与 reference 使用相同的 beat window；
6. leading-rest trim 后 Melody 与 GT accompaniment 仍然对齐。

`JSD(Pitch)` 和 `JSD(Onset)` 的 histogram bins、平滑项和 log base 必须先从 StreamMUSE evaluator 或作者代码核对。若使用本地定义，需要在论文中写清楚，不能直接与原论文数值横向比较。

### E1. Descriptive music evaluation

这是 short paper 最可能真正需要的主实验。

1. 冻结一个 test cohort、GT reference、生成长度和 checkpoint。
2. 对最终确定的每个 condition 生成同样数量的 samples。
3. 计算 Valid Output Rate、FMD、JSD(Pitch)、JSD(Onset)、CR、UR；NLL_wavg 放补充列。
4. 如果正文讨论重复，再补一个 Unique Beat Ratio 或 Repetition Rate。
5. 选择 3–5 个代表性音频/MIDI 样例，包括正常结果和至少一个失败例子。

Conditions 暂时写成占位符：

```text
Condition A: TBD
Condition B: TBD, only if the paper contains an actual comparison
Condition C: TBD, only if required by the final method
```

如果最后只有一个系统版本，主表可以只有一行 system output，加一行 GT/reference 作为解释基准；不需要为了填表强行增加三种生成方案。

### E2. Lightweight comparison（按需）

只有论文最终包含明确对比时才做，例如：

- before vs after 某个真实方法改动；
- base model vs finetuned model；
- offline vs realtime 最终导出音乐；
- Prompt condition A vs Prompt condition B。

要求仅是保持 Melody、checkpoint、解码参数、生成窗口和样本数量一致。作为 short paper 的 descriptive comparison，可以报告均值、标准差或 bootstrap interval；不必预设复杂的显著性检验和大规模消融。

### E3. Qualitative examples

short paper 介绍系统时，少量可复现案例通常比堆叠更多相近指标更有效：

- 2–3 个代表性成功案例；
- 1–2 个失败或边界案例；
- 对应 Melody、GT（若存在）和系统输出；
- 简单指出密度、覆盖、重复或 Melody fit 上的现象；
- 不从个别案例推广到整个数据集。

人工试听可以用于内部挑选和 sanity check，但只要论文不作 perceptual superiority claim，就不必把它扩展成正式用户研究。

## 4. 数据规模预期

当前已有的 25-piece × 2 conditions × 2 seeds = 100 outputs 可以先用于 evaluator pipeline smoke test，但未来论文 conditions 尚未确定，不能直接把这批结果视为正式实验。

对于正式 descriptive evaluation：

- 优先增加独立 pieces，而不是只对少数曲目增加大量 seeds；
- FMD 是 corpus-level metric，样本过少时 covariance estimate 不稳定；
- 若资源允许，建议每个最终 condition 至少约 100 个 outputs，并保证相同的独立 piece set；
- 该数量是 short-paper 的实用目标，不是统计充分性的理论保证；
- GT reference 不能因每首生成多个 seeds 就重复复制并计算为更多独立样本。

若最后样本数量明显低于这个规模，仍可报告结果，但应称为 preliminary/descriptive evaluation，并避免比较很小的 FMD 差值。

## 5. 评价窗口和失败政策

- 若任务使用 8-beat accompaniment Prompt，主要音乐指标建议计算 continuation-only 区间，例如 `[8, 32)` beats。
- 可以补充 full-output `[0, 32)`，但不能只评价被选择的 Prompt 本身。
- 所有 conditions 使用相同固定窗口；生成提前停止时不能缩短分母。
- 空输出保留在 Valid Output Rate 的完整分母中。
- FMD/JSD 无法处理的空输出写为 `NaN + failure reason`。
- 比较 FMD/JSD 时使用各 conditions 共同有效的 matched cases，并同时报告被排除数量和完整分母失败率。
- GT 必须来自同一个 dataset record，且应用与 Melody 相同的 trim offset；不使用网上“同名但不同演奏”的版本替代。

## 6. 预计论文表格

正文可以只保留一张紧凑表：

| Setting | Valid (%) | FMD ↓ | JSD(Pitch) ↓ | JSD(Onset) ↓ | CR ↑ | UR ↓ | NLL_wavg ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Condition A | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Condition B（按需） | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

如果确实讨论重复，再单独放一个小图或一列：

```text
Unique Beat Ratio curve vs GT
```

不要把所有指标加权成一个“overall music score”。

## 7. 可以与不可以支持的表述

使用上述 descriptive metrics 可以写：

- “We characterize the generated accompaniment using distributional, rhythmic, and harmonic metrics.”
- “The system obtains the following FMD/JSD/CR/UR values on the frozen test set.”
- 若有真实对比：“Condition B shows a descriptive improvement in CR/UR over Condition A.”

仅靠这些结果不应写：

- “The method is perceptually preferred by listeners.”
- “The method improves overall musicality.”
- “The method generalizes to all musical styles.”
- “The improvement is caused by Rule-S”，除非 Rule-S 和必要 controls 最终确实进入实验。

## 8. 推荐顺序

```text
冻结最终论文主张与 conditions
-> 冻结 metric definitions 和 evaluation window
-> E0 evaluator sanity check
-> 用现有 100 outputs 做 pipeline smoke test
-> 按最终 conditions 生成正式 outputs
-> E1 主表 + E3 少量案例
-> 仅在正文需要时增加 repetition 或 lightweight comparison
```

## 9. 当前建议

short paper 的默认方案可以收敛为：

```text
主要：Valid Output Rate + FMD + JSD(Pitch) + JSD(Onset) + CR + UR
补充：NLL_wavg
可选：Unique Beat Ratio 或 Repetition Rate（二选一）
```

这组指标足以完成“介绍并描述系统生成结果”的目的。只有当论文后来增加“某个方法明显更好”的主张时，再扩展 comparison controls、seeds、人工听评或统计检验。

## 10. 依据

- [BEAT: Tokenizing and Generating Symbolic Music by Uniform Temporal Steps, arXiv:2604.19532v3](https://arxiv.org/abs/2604.19532)
- [Real-Time Language Model Jamming: A Case Study for Live Music Accompaniment Generation, arXiv:2606.11886v1](https://arxiv.org/abs/2606.11886v1)
- [Frechet Music Distance: A Metric for Generative Symbolic Music Evaluation, arXiv:2412.07948](https://arxiv.org/abs/2412.07948)

