---
title: 旋律扰动鲁棒性实验工作流
description: 从 40-input staging 到 qualification、160-run pilot、分析和盲听归档的 fail-closed 执行顺序
---

# 旋律扰动鲁棒性实验工作流

实验规范以 [`developing-logs/plans/2026-07-10-melody-perturbation-robustness-plan.md`](../../developing-logs/plans/2026-07-10-melody-perturbation-robustness-plan.md) 为准，machine-readable 唯一契约是 C5 生成并冻结的 `campaign_config.json`。本文只说明工具顺序，不表示 qualification、160 个 formal run 或用户盲听已经执行。

## 关键约束

- 必须从提交后的 clean worktree 运行；formal `run` 禁止 `--allow-dirty`。
- playback `--tempo` 只控制 wall-clock；模型条件固定为 `--model-condition-bpm 120`。driver 会从 config 同时传 RT 的该参数和 offline 的 `--bpm 120`。
- 所有命令使用同一份 40-input manifest、config SHA-256 和 schedule SHA-256；不要在复制后重新 hash 来“接受”修改后的文件。
- qualification 先于最终 C5 freeze；listening selection 必须在 formal 和 objective analysis 前冻结并进入 config hash。
- `--limit`、`--run-id`、`--dry-run` 和 `--midi-only` 只适合开发验证，不构成 formal campaign 或最终听测包。

下面的 `<...>` 都应替换成绝对路径或明确的冻结值。

## 1. 生成并严格验证 40-input staging

```bash
uv run python scripts/perturb_melody.py generate \
  --mel-dir <CLEAN_MELODY_DIR> \
  --acc-dir <SOURCE_ACC_DIR> \
  --output-root <STAGING_DIR> \
  --manifest <STAGING_DIR>/input_manifest.json \
  --expected-song-count 5

uv run python scripts/midi_to_npz.py \
  --mel-dir <STAGING_DIR>/mel \
  --acc-dir <STAGING_DIR>/acc \
  --output-dir <STAGING_DIR>/npz \
  --ticks-per-beat 4 \
  --strict \
  --expected-manifest <STAGING_DIR>/input_manifest.json \
  --summary-json <STAGING_DIR>/conversion_summary.json
```

严格转换必须报告 40 converted、0 skipped、exact stem set，并逐输入通过 MIDI/NPZ roll gate。目录中出现 stale、extra、missing 或 hash mismatch 都应直接失败。

## 2. 在结果不可见时冻结听测 selection

每首 excerpt 起点应事先写入 `<EXCERPT_STARTS_JSON>`，schema 是 song stem 到整数 model beat 的 object（例如 `{"1": 0, "2": 0, "3": 0, "4": 0, "5": 0}`），且每个 25 秒窗口必须落在 manifest 的可靠 horizon 内。

```bash
uv run python scripts/prepare_robustness_listening.py freeze-selection \
  --input-manifest <STAGING_DIR>/input_manifest.json \
  --excerpt-starts-json <EXCERPT_STARTS_JSON> \
  --output <FREEZE_DIR>/listening_selection.json
```

产物必须恰好描述 24 clips，包含固定 pseed/sseed、20 个主 clips、2 个 anchors 和 2 个字面重复 trial，并使用冻结的 blind-order seed。

## 3. 创建并运行 qualification

先把 clean commit 的字面 SHA 作为 `<CLEAN_CODE_SHA>` 传入，不要在命令里动态取一个仍可能变化的工作树状态。

```bash
uv run python scripts/qualify_perturbation_campaign.py plan \
  --checkpoint <CHECKPOINT> \
  --input-manifest <STAGING_DIR>/input_manifest.json \
  --code-identity <CLEAN_CODE_SHA> \
  --dense-song 2 \
  --tail-song 2 \
  --tail-song 4 \
  --output-dir <QUAL_DIR>
```

`<QUAL_DIR>/qualification_plan.json` 会记录 candidate config、20-row qualification schedule、各自 hash 和准确的执行命令。按其中 hash 执行：

```bash
uv run python scripts/run_perturbation_matrix.py run \
  --qualification \
  --config <QUAL_DIR>/qualification_config.json \
  --config-sha256 <QUAL_CONFIG_SHA256> \
  --schedule <QUAL_DIR>/qualification_manifest.jsonl \
  --schedule-sha256 <QUAL_SCHEDULE_SHA256> \
  --output-root <QUAL_OUTPUT_DIR>

uv run python scripts/qualify_perturbation_campaign.py evaluate \
  --config <QUAL_DIR>/qualification_config.json \
  --schedule <QUAL_DIR>/qualification_manifest.jsonl \
  --output-root <QUAL_OUTPUT_DIR> \
  --static-summary <STAGING_DIR>/conversion_summary.json \
  --output <QUAL_DIR>/qualification_result.json
```

只有 offline/RT token determinism、40-input static gate、统一 tempo gate 和 tail 8/16/24 convergence 全部通过，`evaluate` 才返回成功。失败后应回 Phase A/B 修复、重新提交并重跑，不得继续 formal。

## 4. C5 freeze 与 160-row schedule

```bash
uv run python scripts/qualify_perturbation_campaign.py freeze \
  --candidate-config <QUAL_DIR>/qualification_config.json \
  --qualification-result <QUAL_DIR>/qualification_result.json \
  --listening-manifest <FREEZE_DIR>/listening_selection.json \
  --output <FREEZE_DIR>/campaign_config.json

uv run python scripts/run_perturbation_matrix.py schedule \
  --config <FREEZE_DIR>/campaign_config.json \
  --config-sha256 <CAMPAIGN_CONFIG_SHA256> \
  --output <FREEZE_DIR>/run_manifest.jsonl
```

保存工具输出的 config/schedule SHA-256。schedule 必须精确重建为 160 行；formal driver、audit、analyzer、listening 和 report 都使用这两个 hash。

## 5. Formal run 与完整性审计

```bash
uv run python scripts/run_perturbation_matrix.py run \
  --config <FREEZE_DIR>/campaign_config.json \
  --config-sha256 <CAMPAIGN_CONFIG_SHA256> \
  --schedule <FREEZE_DIR>/run_manifest.jsonl \
  --schedule-sha256 <RUN_MANIFEST_SHA256> \
  --output-root <CAMPAIGN_OUTPUT_DIR>

uv run python scripts/run_perturbation_matrix.py audit \
  --config <FREEZE_DIR>/campaign_config.json \
  --config-sha256 <CAMPAIGN_CONFIG_SHA256> \
  --schedule <FREEZE_DIR>/run_manifest.jsonl \
  --schedule-sha256 <RUN_MANIFEST_SHA256> \
  --output-root <CAMPAIGN_OUTPUT_DIR> \
  --output <CAMPAIGN_OUTPUT_DIR>/campaign_audit.json
```

driver 自启 dedicated loopback server，逐 run 校验 checkpoint/runtime/session/input/token/artifact 契约。audit 必须看到 160 expected verdict、0 missing、0 content-invalid、无 extra run id；operational-invalid 单列保留，不可用重试覆盖。

## 6. 分析与 targeted controls

```bash
uv run python scripts/analyze_perturbation_robustness.py campaign \
  --config <FREEZE_DIR>/campaign_config.json \
  --config-sha256 <CAMPAIGN_CONFIG_SHA256> \
  --schedule <FREEZE_DIR>/run_manifest.jsonl \
  --schedule-sha256 <RUN_MANIFEST_SHA256> \
  --output-root <CAMPAIGN_OUTPUT_DIR> \
  --report-dir <ANALYSIS_DIR>
```

正式分析不使用 `--allow-unpinned-schedule`。分析目录包含 run metrics、paired contrasts、factorial interaction、song-block bootstrap、targeted controls、QC、图表和 hash index；任一 provenance 或 endpoint control 不通过，都必须 fail-closed 或把相应 endpoint 标为 invalid。

## 7. 听测包、封存、解盲与最终报告

```bash
uv run python scripts/prepare_robustness_listening.py build \
  --config <FREEZE_DIR>/campaign_config.json \
  --config-sha256 <CAMPAIGN_CONFIG_SHA256> \
  --selection <FREEZE_DIR>/listening_selection.json \
  --schedule <FREEZE_DIR>/run_manifest.jsonl \
  --schedule-sha256 <RUN_MANIFEST_SHA256> \
  --output-root <CAMPAIGN_OUTPUT_DIR> \
  --controls-root <ANALYSIS_DIR>/controls \
  --package-dir <LISTENING_DIR> \
  --soundfont <SOUNDFONT>

uv run python scripts/prepare_robustness_listening.py audit \
  --package-dir <LISTENING_DIR>
```

听测工具使用冻结的 SciPy `resample_poly` 4× Kaiser polyphase reconstruction 测量 inter-sample true peak，只在超限时对整段作保护性衰减，并在 PCM16 写回后重新测量。audit 会从 WAV 独立重算 true peak，并核对实现名、4× factor、阈值和记录值；只有 integer sample peak 或 metadata 漂移都会 fail-closed。`--midi-only` 仍只能用于开发；不要在带真实 soundfont 的 WAV 包获得 `accepted_final=true` 前开始正式盲听。

用户填完 24 行 blind scores 后，必须先封存再解盲：

```bash
uv run python scripts/prepare_robustness_listening.py seal-scores \
  --package-dir <LISTENING_DIR>

uv run python scripts/prepare_robustness_listening.py unblind \
  --package-dir <LISTENING_DIR>

uv run python scripts/build_robustness_report.py \
  --config <FREEZE_DIR>/campaign_config.json \
  --config-sha256 <CAMPAIGN_CONFIG_SHA256> \
  --schedule <FREEZE_DIR>/run_manifest.jsonl \
  --schedule-sha256 <RUN_MANIFEST_SHA256> \
  --campaign-root <CAMPAIGN_OUTPUT_DIR> \
  --campaign-audit <CAMPAIGN_OUTPUT_DIR>/campaign_audit.json \
  --analysis-dir <ANALYSIS_DIR> \
  --listening-package <LISTENING_DIR> \
  --output-dir <REPORT_DIR>
```

`report.md` 与 `reproducibility_index.json` 只有在 staging、160 verdict、content validity、controls、analysis、最终盲听包、sealed scores 和 unblinded scores 全部通过交叉 hash 验证后才可标记 complete。
