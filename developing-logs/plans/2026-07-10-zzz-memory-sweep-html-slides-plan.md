# ZipZapZop Memory Sweep HTML Slides 计划（2026-07-10）

## 目标

做一个可直接用于汇报的 HTML/PPT 风格 slide deck，讲清楚 ZipZapZop memory sweep 实验：

- 我们为什么做这个实验；
- 实验怎么设计；
- 7 个模型跑出来的主要结果是什么；
- memory、APC、oracle history、speculative decoding 分别告诉了我们什么；
- 哪些结论可以相信，哪些地方有实验限制；
- 下一步应该做什么。

目标听众默认是项目组内部技术讨论，所以 slides 不需要从 LLM 基础讲起，但需要把 ZipZapZop task、history window、APC、oracle history 这些实验变量讲清楚。

## 数据来源

- 实验报告：`developing-logs/reports/2026-07-10-zzz-memory-sweep-experiment-report.md`
- 正式结果汇总：`task_runs/zzz_memory_full_matrix_parallel_20260710-090000/matrix_summary_formal.csv`
- 正式结果 Markdown：`task_runs/zzz_memory_full_matrix_parallel_20260710-090000/matrix_summary_formal.md`
- 原始 per-turn 数据：各模型目录下的 `per_turn.csv`

注意：做 slides 时应使用 `matrix_summary_formal.*`，不要直接使用 `matrix_summary.*`，因为后者包含每个模型的 3-turn smoke/pilot rows。

## Slide Deck 大纲

### Slide 1：Title

标题：`ZipZapZop Memory Sweep: Accuracy, Latency, and Context Effects`

内容：

- 日期：2026-07-10
- 项目：StreamMUSE / ProjectIsochron-style LLM realtime task framework
- 一句话 takeaway：不同模型对 history 的收益差异很大，APC 和 self-history pollution 会显著改变 latency/accuracy 解读。

视觉建议：

- 简洁标题页；
- 背景可以用一个轻量 grid / timeline / game-loop 图案；
- 不要放太多文字。

### Slide 2：Why This Experiment

核心问题：

- LLM 在 realtime turn-by-turn task 里，到底需要看多少历史？
- 更长 history 是否一定更准？
- 更长 prompt 是否一定更慢？
- vLLM 的 prefix caching 会不会改变 latency 结论？
- 模型自己答错后，把错误写进历史，会不会污染后续表现？

建议呈现：

- 左侧列 4 个问题；
- 右侧用一个小图表示 human/model 一轮一轮玩 ZipZapZop。

### Slide 3：Task Definition

讲 ZipZapZop 是什么：

- 输入当前数字 `N`；
- 如果能被 3 整除输出 `Zip`；
- 如果能被 5 整除输出 `Zap`；
- 如果能同时被 3 和 5 整除输出 `ZipZap` / `ZipZapZop`（按当前 task 实现为准，写 slides 时核对代码里的 expected 规则）；
- 否则输出数字本身；
- 每轮要求只输出答案。

重点：

- 这个 task 本身很简单；
- 难点不是算术，而是 format following、历史污染、上下文增长、realtime latency。

视觉建议：

- 用 1 到 15 的小表格展示 expected outputs；
- 标出第一个容易出错的特殊点。

### Slide 4：Experiment Matrix

展示主实验矩阵：

- 模型：7 个；
- history window：`0`, `8`, `32`, `all`;
- temperature：`0`, `0.7`;
- normal 档 `temperature=0.7 + top_p=0.8`，重复 3 次；
- 每个 run 100 turns；
- 额外 oracle history；
- APC on/off；
- gpt-oss 额外 ngram speculative decoding。

建议图表：

- 一个矩阵图：models × history × temperature；
- 右侧列出 add-on experiments。

### Slide 5：Models Tested

表格列出 7 个模型：

| 模型 | 类型/规模 | 精度/量化 | GPU |
|---|---|---|---|
| Qwen3-8B | dense | BF16 | 1 |
| Qwen3.6-27B | dense | BF16 | 1 |
| Gemma-3-27B-it | dense | BF16 | 1 |
| Qwen3.6-35B-A3B | MoE | BF16 | 1 |
| gpt-oss-120b | MoE/reasoning | MXFP4 | 1 |
| Llama-3.3-70B | dense | BF16 | 2 |
| Qwen3-235B-A22B-FP8 | MoE | FP8 | 2 |

强调：

- 本实验没有对同一模型扫不同量化；
- 能 BF16 跑的都用 BF16；
- gpt-oss 和 Qwen235 是模型自身/显存现实决定的原生低精度版本。

### Slide 6：Engineering Setup

讲为了让实验可跑，做了什么：

- 新增 parallel matrix runner；
- 7 模型下载与 HF gated 权限；
- Qwen3/Qwen3.6 禁 thinking；
- gpt-oss 使用 `max_tokens=512` 并取 final content；
- prompt 改成 `Current number: N\nAnswer:`，避免模型把 `1:` 当成编号续写；
- Qwen235 workaround：
  - `VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER=0`;
  - `gpu_memory_utilization=0.90`;
  - `tensor_parallel_size=2`。

视觉建议：

- 用 pipeline 图：download -> serve -> smoke -> sweep -> merge -> formal summary。

### Slide 7：Main Accuracy Result

展示每个模型的最佳正式配置：

| 模型 | 最佳配置 | strict acc | median latency |
|---|---|---:|---:|
| Qwen3-235B-FP8 | recent-8, temp=0 | 1.000 | 146.5 ms |
| Qwen3-8B | recent-32 | 0.760 | 24.8-30.2 ms |
| Qwen3.6-27B | recent-32 | 0.960 | 88.4-89.7 ms |
| Qwen3.6-35B-A3B | memoryless/recent-32 | 0.890 | 79.2-82.5 ms |
| Gemma-3-27B-it | memoryless | 0.820 | 82.0 ms |
| Llama-3.3-70B | recent-8 | 0.730 | 99.9 ms |
| gpt-oss-120b | 多数档位 | 1.000 | 507.8-601.8 ms |

讲述重点：

- gpt-oss 几乎满分，但 latency 不是同一口径，因为它是 reasoning-mode total latency；
- Qwen235 是非 reasoning 模型里最强；
- Qwen3.6-27B 是性价比/稳定性最突出的模型。

### Slide 8：Accuracy by History Window

展示 history window 对 accuracy 的影响。

建议图表：

- grouped bar chart：每个模型一组，显示 temp=0 下 `memoryless/recent-8/recent-32/all` strict accuracy；
- 或 heatmap：模型 × history，颜色表示 accuracy。

讲述重点：

- history 更长并不保证更准；
- `recent-8` 对 Qwen235/Llama 有帮助；
- `recent-32` 对 Qwen3.6-27B 最好；
- `all` 有时受 self-history pollution 影响，反而不如短窗口。

### Slide 9：Latency by Model

展示不同模型的 median latency。

建议图表：

- bar chart：最佳配置下的 median latency；
- 单独把 gpt-oss 用不同颜色或 split axis 表示，因为它是 reasoning-mode total latency。

讲述重点：

- 小模型不一定准确；
- 大模型不一定最慢，例如 Qwen235 在短输出下可接受；
- gpt-oss 的高 latency 来自 reasoning/final 路径，不应直接和 8-token 短输出模型做纯速度对比。

### Slide 10：APC Changes the Latency Story

展示 APC speedup：

| 模型 | all speedup | recent-32 speedup |
|---|---:|---:|
| Qwen3-235B-FP8 | 2.32x | 1.01x |
| Qwen3-8B | 1.43x | 1.04x |
| Qwen3.6-27B | 1.01x | 0.99x |
| Qwen3.6-35B-A3B | 1.06x | 1.03x |
| Gemma-3-27B-it | 1.45x | 1.23x |
| Llama-3.3-70B | 2.06x | 1.22x |
| gpt-oss-120b | 1.14x | 1.08x |

讲述重点：

- `all` prompt 每轮只是追加，prefix cache 命中高；
- `recent-32` 是滑窗，前缀变化更大；
- 因此开启 APC 后，“更长 history 是否更慢”不再是简单线性关系。

### Slide 11：Self-History Pollution

展示 oracle history 对照：

| 模型 | normal all | oracle all | delta |
|---|---:|---:|---:|
| Qwen3-235B-FP8 | 0.94 | 0.99 | +0.05 |
| Qwen3-8B | 0.68 | 0.77 | +0.09 |
| Qwen3.6-27B | 0.95 | 0.94 | -0.01 |
| Qwen3.6-35B-A3B | 0.82 | 0.90 | +0.08 |
| Gemma-3-27B-it | 0.81 | 0.85 | +0.04 |
| Llama-3.3-70B | 0.58 | 0.74 | +0.16 |
| gpt-oss-120b | 1.00 | 1.00 | +0.00 |

讲述重点：

- 如果模型自己的错误被写入 history，后续表现可能明显变差；
- Llama 的差距最大；
- Qwen3.6-27B 和 gpt-oss 对这个问题最稳。

### Slide 12：Speculative Decoding Result

展示 gpt-oss ngram speculative decoding：

| config | baseline median | ngram median | ngram / baseline |
|---|---:|---:|---:|
| memoryless | 562.2 ms | 754.9 ms | 1.34x |
| recent-8 | 546.3 ms | 792.0 ms | 1.45x |
| recent-32 | 628.2 ms | 955.3 ms | 1.52x |
| all | 601.8 ms | 910.7 ms | 1.51x |

讲述重点：

- 本 task 上 ngram speculative decoding 没有加速，反而变慢；
- greedy accuracy 没变，说明正确性没坏；
- 可能原因：ZipZapZop 输出太短、reasoning/final 结构不适合 ngram speculation；
- 后续如果测 speculative decoding，应该换长文本 generation task 或扫 speculation 参数。

### Slide 13：Caveats

必须明确的限制：

- 正式分析必须使用 `matrix_summary_formal.*`；
- `matrix_summary.*` 包含 smoke/pilot rows；
- vLLM `/reset_prefix_cache` 返回 404，所以 APC-on 主矩阵没有做到每 run 清空 prefix cache；
- APC-on latency 应解释为同一 server 内连续 run 的 warm/cache 场景；
- gpt-oss latency 是 reasoning-mode total latency；
- 本次没有做同模型不同量化的 ablation。

视觉建议：

- 用 warning/callout 样式；
- 这页不要太花，强调“怎么正确解读结果”。

### Slide 14：Recommendations

建议结论：

- 如果需要稳健、非 reasoning、较好 accuracy/latency 平衡：优先看 `Qwen3.6-27B`；
- 如果追求最高准确率且能接受 2 GPU：`Qwen3-235B-FP8` 很强；
- 如果接受 reasoning latency、追求正确率：`gpt-oss-120b` 几乎满分；
- 对 realtime interactive task，不要默认 `all history` 最好；需要 per-task 选择 history window；
- 评估 latency 时必须显式报告 APC 状态。

### Slide 15：Next Steps

下一步实验建议：

- 做同模型不同量化 ablation：BF16 vs FP8 vs AWQ/GPTQ；
- 修复或替换 vLLM prefix cache reset 方案，重新跑一版真正 cold-start per-run APC-on；
- 画 per-turn latency 曲线，看 turn index/prompt tokens 和 latency 的关系；
- 增加更多 game/task，验证 ZipZapZop 结论是否泛化；
- 为 gpt-oss 单独做 reasoning output / final output latency 拆分。

## HTML Slide 实现建议

建议用一个单文件 HTML deck：

- 文件位置建议：`developing-logs/slides/2026-07-10-zzz-memory-sweep-slides.html`
- 技术选择：
  - 简单方案：纯 HTML/CSS/JS，自写 slide navigation；
  - 或使用 Reveal.js，如果项目里已经有类似 slides 依赖再考虑。
- 图表：
  - 第一版可以用内嵌 SVG/HTML table；
  - 后续如要更漂亮，可用 Chart.js，但需要确认是否允许引入外部依赖或本地 vendored bundle。

视觉风格：

- 深色或浅色都可以，但建议选择高对比、工程汇报风；
- 每页只保留 1 个核心观点；
- 表格行数多的地方，用 highlight 强调 2-3 个关键数字；
- gpt-oss 的 latency 图要单独标注 reasoning-mode，避免误读。

## Todo List

- [ ] 确认 ZipZapZop 规则展示是否完全匹配当前 `ZipZapZopTask` 实现。
- [ ] 从 `matrix_summary_formal.csv` 抽取 slides 需要的 4 张表。
- [ ] 生成 accuracy by history heatmap/bar chart 数据。
- [ ] 生成 APC speedup chart 数据。
- [ ] 生成 oracle delta chart 数据。
- [ ] 生成 gpt-oss ngram comparison chart 数据。
- [ ] 实现 HTML slide deck skeleton。
- [ ] 填入每页标题、核心结论、speaker notes。
- [ ] 加键盘左右键导航和 slide number。
- [ ] 在浏览器检查 16:9 桌面显示。
- [ ] 确认所有数字来自 `matrix_summary_formal.*`，没有混入 smoke/pilot rows。
