# StreamMUSE Eval Metrics Status

本文只记录实现审计与后续运行协议，不包含任何尚未实际计算的结果。审计日期：2026-08-30。

## 1. 仓库与版本基准

- 评测仓库：`F:\repos\eval`，远端为 `https://github.com/StreamMUSE/eval.git`。
- 音乐质量实现基准：`origin/main@6957a713034e704ccdd53e39ca472a1dad8528d8`（下文简称 `6957a71`）。所有“已支持”结论均以该 commit 为准。
- 当前本地 checkout：`nll-compute@45b4f45e08d90b5bba9806591e3cb5f2601a0355`，跟踪 `origin/nll-compute`；本次没有切换分支。
- 额外检查：`origin/ae-bench-integration@ce594a5678db5006f12bbed55c9442ecf56467f5`。该分支新增 `src/benchmarks/`、benchmark 分析入口和配置，主要处理 generation-length/latency/constraint 分析；这些新增功能**不属于** `origin/main@6957a71`，不能写成 main 已有。
- 同时审计的系统代码：当前 `StreamMUSE-v1` checkout `codex/failcase-analysis@ac3c603a`。本次未修改其源码或实验结果。

## 2. 指标状态

统一约定：生成目录和 GT 目录先按相同 MIDI basename 配对；除特别说明外，箭头表示主表期望方向。

| Metric | 中文含义 / 数学或代码定义 | 方向 | 所需输入 | `origin/main@6957a71` 入口 | 当前状态 | 关键限制 |
|---|---|---:|---|---|---|---|
| FMD | Fréchet Music Distance。对生成与 GT 的 accompaniment-only MIDI 提取嵌入，并计算两组高斯分布的 Fréchet 距离：$\lVert\mu_g-\mu_r\rVert_2^2+\mathrm{Tr}(\Sigma_g+\Sigma_r-2(\Sigma_g\Sigma_r)^{1/2})$。 | ↓ | 一整套同名配对的 generated/GT MIDI 数据集 | `evaluate_accompaniment_metrics.py::_FrechetDatasetBuilder`、`_compute_frechet_music_distance_dataset` | **主表已选；实现可用但较重**。默认 `clamp2` + `mle`。 | 数据集级指标，不是逐曲指标；依赖外部 `frechet-music-distance`、CLAMP2 模型和 cache；对样本数、协方差估计及数据集组成敏感。 |
| JSD(Pitch) | 生成与 GT accompaniment 的 128-bin MIDI pitch count 分布之 Jensen-Shannon divergence：$\mathrm{JSD}(P,Q)=\frac12\mathrm{KL}(P\Vert M)+\frac12\mathrm{KL}(Q\Vert M)$，$M=(P+Q)/2$。 | ↓ | 每首同名 generated/GT MIDI；generated 中需能正确排除 Melody | `evaluate_accompaniment_metrics.py::_compute_pitch_histogram`、`_js_divergence`、`evaluate_pair` | **主表已选；可直接运行**。 | 比较绝对音高分布，不直接评价和声、乐句或时序；main 只输出逐曲值及 five-number summary，不自动给 bootstrap CI。 |
| JSD(Onset) | 先令共同曲长 $L=\max$（该对 generated 与 GT 所有 accompaniment note end），再对每个 onset 使用 $\mathrm{clip}(t/L,0,1)$，放入默认 64 个均匀 bins 后计算 JSD。 | ↓ | 每首同名 generated/GT MIDI | `evaluate_accompaniment_metrics.py::_determine_distribution_bins`、`_compute_onset_histogram`、`_js_divergence` | **主表已选；可直接运行**。 | 当前实现是**整曲归一化 onset 位置直方图**，不是 beat-class/on-beat histogram；对曲长、裁剪窗口和尾部长度敏感。 |
| CR | Consonant Ratio（协和支撑率）。以 Melody note 的完整 duration 加权；若该 Melody note 的时间范围内存在任一 accompaniment note，且模 12 音程属于默认集合 `{0,3,4,7,8,9}`，则整颗 Melody note duration 计入 consonant。$\mathrm{CR}=T_{\mathrm{consonant}}/T_{\mathrm{melody}}$。 | ↑ | generated MIDI 内同时存在且能识别 Melody 与 Accompaniment | `evaluate_accompaniment_metrics.py::_compute_harmonic_consonance` | **主表已选；可直接运行**。 | 不是按真实重叠时长切片；只要有一个重叠伴奏音协和，即使同时还有不协和音，也把整颗 Melody note 判为 consonant。Melody/Acc 任一为空时返回 `None`。 |
| UR | Unsupported Ratio（无伴奏支撑率）。若某颗 Melody note 的整个时间范围内没有任何重叠 accompaniment note，则将该 Melody note 的完整 duration 计入 unsupported。$\mathrm{UR}=T_{\mathrm{unsupported}}/T_{\mathrm{melody}}$。 | ↓ | 同 CR | `evaluate_accompaniment_metrics.py::_compute_harmonic_consonance` | **主表已选；可直接运行**。 | 同样按 Melody note duration 加权，而不是按无覆盖子区间精确积分；不能单独区分合理留白与生成失败。 |
| NLLwavg | Token-weighted average NLL：$\mathrm{NLL}_{\mathrm{wavg}}=\sum_i\mathrm{total\_nll}_i/\sum_i\mathrm{total\_tokens}_i$。对应输出字段 `weighted_avg_nll` / `weighted_ave_nll`。 | ↓ | 模型侧逐文件 JSON：`total_nll`、`avg_nll`、`total_tokens` | `src/nll_compute/aggregate.py::aggregate_nll`；`add_nll_to_summary.py::compute_nll_metrics` | **主表已选；eval 侧聚合可用，原始打分链未就绪**。 | eval 主体主要负责聚合；原始 NLL 必须由与目标 checkpoint/tokenizer/mask 一致的模型侧 runner 产生。当前 `StreamMUSE-v1` checkout 未发现可直接使用的 NLL runner，不能只靠 eval repo 得到原始 NLL。 |
| Valid Output Rate | 有效输出率。eval main 尚无冻结定义。建议按 trial 级定义：$\mathrm{VOR}=N_{\mathrm{valid\ trials}}/N_{\mathrm{manifest\ trials}}$；`valid` 谓词必须在运行前明确（文件可解析、所需轨道存在、目标窗口内是否要求有效 accompaniment decision/note，REST 是否算有效）。 | ↑ | 冻结 manifest、每个 trial 的完成状态、输出 MIDI/decision log | **无直接入口** | **主表候选，但尚未定义和实现；优先归入系统表**。 | 不等于旧 `hit_rate`，也不能只以“成功生成 MIDI 文件”替代；分母必须包含 manifest 中失败/缺失 trial，REST 的语义需要先冻结。 |
| JSD(Duration) | 生成与 GT accompaniment note duration 分布的 JSD；默认 64 bins，上界由该对 pooled durations 的 0.995 quantile 决定，超出部分 clip 到末 bin。 | ↓ | 每首同名 generated/GT MIDI | `evaluate_accompaniment_metrics.py::_compute_duration_histogram`、`_determine_distribution_bins` | **仓库支持，但当前主表未选**。 | duration 以秒计，受 tempo 与 MIDI 规范化影响；每对曲目使用各自 bins，跨曲汇总的解释需谨慎。 |
| DR | Dissonant Ratio（不协和率），代码定义为 $\max(0,1-\mathrm{CR}-\mathrm{UR})$。 | ↓ | 同 CR | `evaluate_accompaniment_metrics.py::_compute_harmonic_consonance` 返回 `dissonant_ratio` | **仓库支持，但当前主表未选**。 | 是 CR/UR 的残差，不是独立逐时刻不协和分析；继承 CR 的“任一协和重叠即整音符协和”规则。 |
| Chord Accuracy | 在 chord annotation 的每个时间段内，按 accompaniment pitch-class 重叠 duration 构造权重，预测 chord label，再计算与参考 label 的 exact-match fraction。 | ↑ | generated accompaniment MIDI；`<annotation-root>/<piece>/chord_midi.txt` | `evaluate_accompaniment_metrics.py::_compute_chord_accuracy`、`_estimate_chord_sequence` | **仓库支持，但当前主表未选**。 | 必须有外部和弦标注；候选 chord 来自参考序列，且只统计 accompaniment 结束前的 segment，需在使用前审计覆盖偏差。 |
| PolyDis | 以 PolyDis VAE 分别编码 texture latent 与 chord latent；对每个默认 32-step（4 steps/beat，即 8 beats）窗口计算 latent L2 distance，并汇总 `txt_mean_distance`、`chd_mean_distance`。 | ↓ | generated、prompt/GT MIDI；外部 PolyDis repo 与 `polydis-v1.pt` | `evaluate_accompaniment_metrics.py::_compute_polydis_latent_similarity` | **仓库支持，但当前主表未选**。 | 当前入口围绕 prompt、generated continuation、GT continuation 的 latent distance，不是单一通用质量分；依赖外部旧模型、track/prompt 路径推断和窗口构造。 |

### 必须保留的实现语义

1. `NLLwavg` 明确使用 `sum(total_nll) / sum(total_tokens)`；逐文件 `avg_nll` 的算术平均是另一列，不能混用。
2. `Valid Output Rate` 与两种旧 `hit_rate` 都不同：
   - 当前 StreamMUSE `MetricsCalculator` 的 `hit_rate` 是 inference response/request 比例；
   - eval `compute_final_system_metric.py` 的 `global_hit_rate` 是旧 `tick_history.json` 中 `is_hit` 的比例。
3. Onset JSD 是整曲归一化位置分布，不是四分音符内或 beat-class 的 onset 分布。
4. CR/UR 按 Melody note duration 加权，采用上述“任一重叠协和音”与“完全无重叠”判定。
5. FMD 默认 `clamp2/mle`，必须作为固定 manifest 上的**数据集级**结果报告。

## 3. 当前 StreamMUSE 日志可用性

当前 checkout 中的关键入口如下：

- `src/streammuse/presentation/cli/cli.py::_save_prompt_continuation_history_logs`：CLI cleanup 在 clear 前保存：
  - `prompt_continuation_prompt_history.json`
  - `prompt_continuation_raw_history.json`
  - `prompt_continuation_history_status.json`
- `src/streammuse/infrastructure/output/session_logger.py::SessionLoggerOutputSink` 与 `json_logger.py::JsonLoggerOutputSink`：session 模式可写 `combined.mid`、`events.jsonl`、`inferences.json`、`performance.json`、`statistics.csv`。
- `src/streammuse/domain/logging/metrics_calculator.py::MetricsCalculator`：当前内置 latency/request-response 统计，不提供本文定义的 VOR、TTFP、`ISR_f` 或 deadline staleness。
- 当前检查到的 `src/streammuse/presentation/web/server.py` 直接组合 WebSocket、MIDI、Console sink；它会录 `combined.mid`，但没有复用 CLI 的 Prompt/raw-history 保存函数。正式使用 Web session 前，必须先验证实际部署分支是否额外保存 raw history，不能从 CLI 行为推断 Web 行为。

## 4. 系统指标与音乐质量指标分表

系统指标：TTFP、`ISR_f`、staleness、Valid Output Rate。它们依赖观察窗口、deadline、decision/REST 语义和实时日志。

音乐质量指标：FMD、JSD(Pitch)、JSD(Onset)、CR、UR、NLLwavg。它们依赖冻结后的 generated/GT MIDI 或模型侧 NLL 输出。

`compute_final_system_metric.py` 只实现旧 `is_hit`/`backup_level` 聚合，并计算旧加权项

$$
I\cdot\frac{32-\mathrm{backup\_level}}{32}.
$$

因此不得把其 `global_hit_rate` 直接重命名为 `ISR_f`，也不得未经时间单位和语义映射就把 `global_avg_backup` 直接称为新 staleness。即使 `summarize_metrics.py` 将旧字段写入 `ISR`/`Staleness` 列，也仍需按新定义重新验证。

建议音乐质量主表字段：

```text
System | Valid Output Rate | FMD | JSD(Pitch) | JSD(Onset) | CR | UR | NLLwavg
```

Valid Output Rate 最好放系统表；若版面或叙事要求，可以作为两张表的共享列，但定义、分母和 trial manifest 必须完全一致。

## 5. 后续正式运行流程

1. **冻结 manifest**：记录 `trial_id`、system、piece、seed、condition、generated path、GT path、预期状态及统一评测窗口。失败和缺失 trial 也必须保留，供 VOR 使用。
2. **构造同名 MIDI 配对**：为每个有效 trial 准备 `generated/<stem>.mid` 与 `gt/<stem>.mid`；检查数量、basename、tempo/裁剪窗口、Melody/Accompaniment track 名。generated 需保留可识别 Melody，CR/UR 才能计算。
3. **先跑轻量指标**：运行 `evaluate_accompaniment_metrics.py`，先读取 JSD(Pitch)、JSD(Onset)、CR、UR；同时保存逐曲 JSON，禁止只留终端摘要。
4. **再跑 FMD**：在同一 manifest 上增加 `--frechet-music-distance --frechet-model clamp2 --frechet-estimator mle`；固定依赖版本与 cache。
5. **模型侧计算 NLL**：使用目标 checkpoint/tokenizer 和冻结的 window/mask 规则生成逐文件 `total_nll/avg_nll/total_tokens` JSON；随后在 eval repo 用 `src/nll_compute/aggregate.py` 聚合 NLLwavg。
6. **汇总与审计**：按 manifest 左连接所有结果，先保留 missing/error，再生成主表。系统指标从 realtime decision/event logs 独立计算，不调用旧字段改名代替。

轻量指标与 FMD 的基准入口示例：

```bash
cd /path/to/eval

uv run python evaluate_accompaniment_metrics.py \
  --generated-dir /path/to/frozen/generated \
  --groundtruth-dir /path/to/frozen/gt \
  --melody-track-names Melody \
  --output-json /path/to/results/music_metrics.json

uv run python evaluate_accompaniment_metrics.py \
  --generated-dir /path/to/frozen/generated \
  --groundtruth-dir /path/to/frozen/gt \
  --melody-track-names Melody \
  --frechet-music-distance \
  --frechet-model clamp2 \
  --frechet-estimator mle \
  --output-json /path/to/results/music_metrics_with_fmd.json

uv run python -m src.nll_compute.runners.run_aggregate \
  --input /path/to/model_side/raw_nll.json \
  --output /path/to/results/nll_summary.json \
  --pretty
```

## 6. 尚未完成的验证

- 尚未冻结正式 manifest、trial 数、评测窗口和 Valid Output Rate 的 `valid` 谓词。
- 尚未验证所有系统输出的 basename、track name、Melody/Accompaniment 拆分与 GT 覆盖率。
- 尚未运行任何一项指标，因此本文没有 FMD/JSD/CR/UR/NLL/VOR 数值。
- 尚未验证 CLAMP2 权重、`frechet-music-distance` 版本、cache 和样本规模是否满足正式 FMD 运行。
- 当前 `StreamMUSE-v1` checkout 没有确认可用的模型侧 NLL runner；eval 中的 manifest runner还带有旧模型/旧 import 假设，不应直接用于当前 checkpoint。
- 尚未定义 TTFP、`ISR_f`、staleness 的日志字段映射及 REST/deadline 语义，也未验证 Web 实际会话是否保存完整 raw history。
- `origin/main@6957a71` 的音乐指标输出是逐曲值和 five-number summary；95% bootstrap confidence interval 尚未实现或运行。正式报告 CI 前，需要冻结按 piece 还是按 trial/seed 重采样。
- 尚未决定是否报告 JSD 的 mean、median 或 bootstrap estimate；主表统计量必须在运行前固定。
- `ae-bench-integration@ce594a5` 的 benchmark 分析尚未合并到 main，也未验证其输入与本次 realtime session 格式一致。
