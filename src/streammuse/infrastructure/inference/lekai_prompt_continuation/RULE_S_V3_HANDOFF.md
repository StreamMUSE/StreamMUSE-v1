# Rule-S v3 交接说明

## 版本与位置

- 分支：`codex/rule-s-v3`
- Rule-S v3 实现 commit：`e9ff5f79` (`Add Rule-S v3 prompt batch selection`)
- 当前分支 tip：`84edebd6` (`Add explicit beat cohort exclusions`)
- 本地：`F:\repos\StreamMUSE-rule-s-v3`
- H200：`/data/home/yuanxin/StreamMUSE-rule-s-v3`
- Spark：`/home/xiaosongma/StreamMUSE-rule-s-v3`

核心文件：

- `prompt_batch_selector.py`：特征、Rule-S v3 评分与候选选择。
- `prompt_engine.py`：生成候选 batch、提取 Melody/Accompaniment 特征并调用选择器。

## 启用方式

```bash
export LEKAI_PROMPT_SELECTION_MODE=rule_s_v3
export LEKAI_PROMPT_BATCH_CANDIDATES=5
```

Prompt Model 一次生成 5 个候选；系统只保留 Rule-S v3 得分最高的候选，作为后续 continuation 的 accompaniment Prompt。

## 精确公式

对候选 `i`：

```text
S_i = -0.45                * R_i(PPL)
      +0.47699162299076214 * R_i(PitchRange)
      +0.4666666666666666  * R_i(PitchClassEntropy)
      +0.3                 * R_i(DurationMatch)
      +0.5                 * R_i(TonalFit)
      -0.5                 * LowRegisterPenalty_i
```

选择 `S_i` 最大的候选。除低音区惩罚外，各项使用候选 batch 内的 normalized average rank；低音区惩罚直接使用 `[0, 1]` 数值。

### Normalized average rank

对指标 `m`，只在该指标可用的 eligible 候选集合 `A_m` 内排名。令 `n_m = |A_m|`：

```text
R_i(m) = [1 + count(x_j < x_i) + (count(x_j = x_i) - 1) / 2] / n_m
```

- 最小值的 rank 为 `1/n_m`，最大值为 `1`。
- 相同值使用 average rank。
- 某个可选指标缺失时，该候选在该项的 contribution 为 `0`，不会仅因此失去 eligibility。
- PPL 的权重为负，因此较低 PPL 更优；其余正权重 rank 项均为数值较高更优。

## 指标定义

### Prompt PPL

使用 Prompt Model 的 raw logits，仅对所需 Prompt beats 内生成的 accompaniment content target tokens 计算：

```text
PPL = exp(min(mean token NLL, 80))
```

没有可评分 accompaniment content token 时，PPL 不可用。

### Accompaniment pitch range

```text
PitchRange = max(active onset MIDI pitch) - min(active onset MIDI pitch)
```

单位为半音，只使用出现 onset 的 accompaniment 音高。

### Accompaniment pitch-class entropy

按音符持续 tick 数累计 12 个 pitch class 的权重并归一化为 `q_c`：

```text
PitchClassEntropy = -sum_c q_c * log2(q_c)
```

### Duration match

令 `d_acc`、`d_mel` 分别为 accompaniment 和 Melody 的音符持续 tick 数中位数：

```text
DurationMatch = exp(-abs(log((d_acc + epsilon) / (d_mel + epsilon))) / tau)
epsilon = 1e-6
tau = ln(2)
```

两者相等时为 `1`；比例为两倍或二分之一时为 `exp(-1) \approx 0.368`。任一侧无有效音长时该项不可用。

### Tonal fit

1. 按音符持续 tick 数构造 Melody 和 accompaniment 的 12 维 pitch-class evidence，并分别归一化。
2. 使用以下 major/minor soft templates 的 12 种旋转，共 24 个调性模板：

```text
major = [1.0, 0.1, 0.65, 0.1, 0.8, 0.7, 0.1, 0.9, 0.1, 0.65, 0.1, 0.6]
minor = [1.0, 0.1, 0.6, 0.8, 0.1, 0.7, 0.1, 0.9, 0.65, 0.1, 0.6, 0.1]
```

3. 选择与 Melody 点积最大的模板 `T*`。令 `f_max` 为最大点积，`u = mean(T*)`：

```text
KeyConfidence = clip((f_max - u) / (1 - u), 0, 1)
Compatibility = clip(dot(normalized_acc_evidence, T*), 0, 1)
TonalFit = KeyConfidence * Compatibility
```

Melody 或 accompaniment 没有有效 pitch-class evidence 时该项不可用。

### Low-register penalty

对 accompaniment 的每个音符 `j`，令 MIDI pitch 为 `p_j`、持续 tick 数为 `d_j`：

```text
LowRegisterPenalty = clip(
    sum_j d_j * (max(0, 36 - p_j) / 15)^2 / sum_j d_j,
    0,
    1
)
```

MIDI 36 及以上不惩罚；越低且持续越久，惩罚越大。没有 accompaniment 音符时返回 `0`。

## Eligibility、fallback 与 tie-break

候选必须同时满足：

1. `generated_beats >= required_beats`；
2. `acc_note_count > 0`；
3. Prompt PPL 可用且为有限数。

`DurationMatch`、`TonalFit` 或 `LowRegisterPenalty` 缺失不会取消 eligibility；对应 contribution 记为 `0`。

- 若没有 eligible 候选：选择第 1 个候选，并记录 `fallback_reason=no_eligible_candidate`。
- 若总分并列：先选择 PPL 更低者；若仍并列，选择 `candidate_number` 更小者。

## 权重来源与限制

- 从 Rule-S v1 pilot Spearman 相关性保留的三个权重：PPL `-0.45`、pitch range `+0.47699162299076214`、pitch-class entropy `+0.4666666666666666`。
- v1 的 average voice number (`+0.4958158260214505`) 已从 v3 公式中移除。
- Duration match `+0.3`、tonal fit `+0.5`、low-register penalty `-0.5` 是启发式、尚未验证的权重，不能表述为由相关性实验得到。

## 最小验证

在仓库根目录运行：

```bash
PYTHONPATH="$PWD/src:$PWD/transformers/src" \
python -m pytest \
  tests/unit/infrastructure/inference/test_prompt_batch_selector.py \
  tests/unit/infrastructure/inference/test_lekai_prompt_engine.py -q
```

主要测试位置：

- `tests/unit/infrastructure/inference/test_prompt_batch_selector.py`
- `tests/unit/infrastructure/inference/test_lekai_prompt_engine.py`
- `tests/unit/infrastructure/inference/test_run_matched_system_eval.py`

测试覆盖精确公式、移除 average voice number、音长匹配、调性匹配、低音区惩罚、可选指标缺失、fallback、tie-break，以及 Engine 的 `rule_s_v3` mode 接入。
