# 本周工作汇报：Lekai Continuation 三项实验

> PPT 内容草案，预计 8 页。整体保持简短，每页只表达一个主要结论。

## 01 标题页

**Lekai Continuation：扰动鲁棒性、生成参数与跨风格表现**

- 本周工作汇报
- 2026.08

## 02 本周完成的三个问题

1. **Melody timing robustness**  
   Melody onset 发生模型可见的时间扰动后，continuation 是否明显退化？
2. **Greedy vs. nucleus sampling**  
   调整 `temperature/top-p` 能否缓解 accompaniment 单调和重复？
3. **Cross-style evaluation**  
   相同模型生成的 accompaniment 在不同 metadata-derived 曲目子集上的表现差异有多大？

共同设置：base epoch15 continuation checkpoint、offline inference、8-beat GT accompaniment Prompt。

## 03 实验一：Melody 时间扰动鲁棒性

**目的**

检验模型是否对 Melody onset 的轻微变化过度敏感。

**Setting**

- 4 首曲子；clean、10% notes perturbed、50% notes perturbed。
- 被选中的 Melody onset 移动 `±1 model tick`，即 `±0.25 beat`。
- Greedy 1 份；nucleus sampling 3 seeds；共 48 个输出。
- 扰动在 4 steps/beat 量化后仍然可见，其他 note 信息保持不变。

**结果**

| Decoder | 10% perturb roll Jaccard | 50% perturb roll Jaccard |
|---|---:|---:|
| Greedy | 0.660 | 0.649 |
| Nucleus sampling | 0.235 | 0.167 |

- 轻微扰动通常没有造成严重的听感退化，但 accompaniment 对扰动的适应不明显。
- 输出变化高度依赖曲子；sampling 更容易因早期 token 改变而进入不同生成轨迹。
- Jaccard 衡量输出变化，不直接代表音乐质量。

## 04 实验二：生成参数与单调性

**目的**

检验 nucleus sampling 能否缓解 greedy accompaniment 的重复、pitch collapse 和异常长音。

**Setting**

| Condition | Parameters | Outputs |
|---|---|---:|
| Greedy | `temperature=1.0, top_k=1, top_p=1.0` | 3 |
| Nucleus sampling | `temperature=1.1, top_k=0, top_p=0.95` | 15 |

- 3 首相同曲子、相同 8-beat GT accompaniment Prompt。
- Sampling 使用 5 个 seeds；`top_k=0`，不与 top-p 同时筛选。

## 05 实验二：结果与结论

| 指标 | Greedy | Sampling |
|---|---:|---:|
| 可评价输出 | 1/3 | 13/15 |
| Pitch-class entropy | 0.793 | 2.993 |
| Adjacent exact repeat rate | 23.2% | 2.9% |
| 人工评分：variety（可配对曲目） | 2.0 | 4.0 |
| 人工评分：musical quality | 3.2 | 3.2 |
| 人工评分：Melody–acc fit | 3.0 | 2.8 |

**结论**

- Sampling 明显缓解完全重复、pitch collapse 和 hanging notes。
- 伴奏变化性提高，但整体听感没有同步提高，Melody–accompaniment fit 仍是主要问题。
- Sampling 还会引入 seed-dependent 的稀疏或静音片段。

## 06 实验三：跨风格测试

**目的**

比较相同模型生成的 accompaniment 在不同风格曲目中的质量、重复、稀疏程度与 Melody fit，同时观察各子集的适用性和失败模式。

**Setting**

- 5 个 metadata-derived 子集：Classical、Pop、Jazz/Blues/Ragtime、Film/TV、Game/Anime。
- 每类 4 首，共 20 首；类别仅根据曲名、作者和作品 metadata 划分。
- 每首：1 个 greedy control + 3 个 nucleus sampling outputs。
- 共 80 个生成结果；77 个包含 continuation accompaniment。

## 07 实验三：跨子集观察

| 子集 | 本轮主要观察 |
|---|---|
| Classical | Sampling 通常增加变化，但仍可能断奏、重复或复制 Prompt pattern。 |
| Pop | 最容易评价、相对稳定；sampling 稍微缓解重复，但整体质量未明显提高。 |
| Jazz/Blues/Ragtime | 4 首均难以形成有效评价；当前曲目、任务或听评方式可能不匹配。 |
| Film/TV | 方差较大；部分 sampling 明显优于 greedy，也出现空生成和稀疏结果。 |
| Game/Anime | 前 8 beats 经常不能体现整首风格，Prompt 代表性成为主要限制。 |

补充客观现象：sampling 下 Film/TV 与 Game/Anime 的 empty-bar rate 约为 `0.20`，高于 Classical 的 `0.05` 和 Pop 的 `0.075`。

## 08 总结与下一步

**本周结论**

1. Melody onset 扰动会改变生成轨迹，但轻微扰动未表现出一致的严重听感退化。
2. Nucleus sampling 能改善 greedy 的重复与退化，但不保证更好的音乐质量或 Melody fit。
3. 模型 accompaniment 在不同子集中的表现差异明显；差异不仅来自 decoder，也与 Prompt 是否能代表整首风格、曲目是否适合任务有关。

**下一步**

- 为跨风格测试选择更有代表性的 Prompt 区间，而不只固定使用开头 8 beats。
- 将“输出是否成功”“变化性”和“音乐质量”分开评价。
- 扩充可评价曲目和听评者后，再进行正式的跨风格统计比较。

## 制作备注

- Task 1 使用流程图：`clean Melody → onset perturbation → continuation → paired comparison`。
- Task 2 使用 greedy/sampling 左右对比，突出“重复减少，但质量没有提高”。
- Task 3 使用五行风格矩阵，不绘制误导性的统一平均柱状图。
- 当前所有人工听评均为单人探索性评价，PPT 中不使用“显著提升”等表述。
