---
title: 旋律扰动、生成伴奏与 flexible triangle 听测工作流
description: 从 fresh 40-input staging 到 fixed-20 qualification、160-run formal campaign、95 题 acc-only 盲听和 partial report
---

# 旋律扰动、生成伴奏与 flexible triangle 听测工作流

当前规范是 [`2026-07-17-melody-perturbation-generated-acc-listening-completion-plan.md`](../../developing-logs/plans/2026-07-17-melody-perturbation-generated-acc-listening-completion-plan.md)。它 supersede 旧 plan 的 24-clip quality-rating Phase F/T10。旧 v1 命令只用于历史包；本轮 DoD 必须使用 `listening_triangle_selection.v2`。

最终问题不是“扰动后的 melody 是否不同”，而是：在同一首歌、excerpt、checkpoint、sample seed 和 RT-theoretical pipeline 下，pitch/onset/both 扰动是否使模型生成的 accompaniment 可被人耳区分。听测只播放 generated accompaniment。

## 不可放宽的约束

- formal 运行必须来自 clean commit；qualification 禁止 retry，formal retry 只能按完整 `(pipeline, song, sample_seed)` 八行 block 执行。
- 40-input manifest、checkpoint、renderer identity、triangle selection、C5 config 和 160-row schedule 都要用路径与 SHA-256 绑定。
- qualification 固定 20 行：dense song `2`，tail songs `2,4`，只接受 `attempt-001`；tail 不收敛时停止。
- primary source 固定为 RT attempt 的 `theoretical_model.mid`，presentation 固定 `acc_solo`。不得用 staging `acc` 或扰动 melody 代替生成伴奏。
- selection 和 renderer 必须在 formal output 之前冻结；objective result 不得用于事后换 trial、seed 或 excerpt。
- 95 题是完整题库，不是一次答完的要求。每答一题即持久化；任意 `n>=1` 都可建立 immutable snapshot 和 partial report。

以下 `<...>` 均替换为绝对路径或已经冻结的 hash。

## 1. 本地 FluidSynth toolchain

系统没有 FluidSynth 时，可无 root 安装到被忽略的 durable toolchain 目录：

```bash
bash scripts/bootstrap_local_fluidsynth.sh
```

默认位置为 `task_runs/toolchains/fluidsynth-2.3.4/rootfs`。后续示例使用：

```bash
FS_ROOT="$PWD/task_runs/toolchains/fluidsynth-2.3.4/rootfs"
FS_BIN="$FS_ROOT/usr/bin/fluidsynth"
SOUNDFONT="$FS_ROOT/usr/share/sounds/sf2/FluidR3_GM.sf2"
```

formal freeze 直接传 rootfs 内的真实 executable，使 C5 绑定实际 FluidSynth binary 而不是 launcher 脚本；`scripts/local_fluidsynth.sh` 仅用于便捷探测。bootstrap 会保存下载包 SHA-256；C5 还会绑定 binary、版本、动态库、soundfont、program/bank、命令模板和 render policy。

## 2. Clean attestation 与 real-checkpoint gold gate

提交实现后，先确认 worktree clean。attestation 必须在写任何 formal artifact 前运行；它会生成并逐文件
hash-pin code、checkpoint、dependency/runtime/GPU 环境和 fixed-20 qualification spec。以下目录位于
被 gitignore 的 durable campaign root，因此写入后不会破坏后续 clean gate：

```bash
CAMPAIGN="$PWD/task_runs/melody_robustness_20260717_generated_acc_v3"
CHECKPOINT="$PWD/models/ModelLekai/epoch_4_1104_1204/model.safetensors"
CODE_SHA="$(git rev-parse HEAD)"
test -z "$(git status --porcelain)"
export CUDA_VISIBLE_DEVICES=0

uv run python scripts/qualify_perturbation_campaign.py attest \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$CAMPAIGN/00_freeze"
```

`attest` 会拒绝 dirty HEAD、缺失/空 checkpoint、无 CUDA GPU、已有同名 attestation 输出。生成的
`code_identity.json`、`checkpoint_identity.json`、`environment.json` 和
`qualification_spec.json` 均带 `.sha256` sidecar；environment 固定 `uv.lock`、
`pyproject.toml`、Python、Torch、CUDA、cuDNN、Torch-visible GPU 和 nvidia-smi physical GPU/driver identity。

随后运行 song 4/5 × tempo 15/120 的真实 checkpoint gold gate；不允许 skip：

```bash
STREAMMUSE_CONSISTENCY_GPU=0 \
STREAMMUSE_CONSISTENCY_SONGS=4,5 \
STREAMMUSE_CONSISTENCY_TEMPOS=15,120 \
LEKAI_CHECKPOINT_PATH="$CHECKPOINT" \
uv run pytest tests/consistency/ -v \
  --junitxml "$CAMPAIGN/00_freeze/gold_consistency.junit.xml"
```

## 3. Fresh 40-input staging

不要复用 `/tmp` smoke campaign：

```bash
uv run python scripts/perturb_melody.py generate \
  --mel-dir prompts/inputs_lekai/mel \
  --acc-dir prompts/inputs_lekai/acc \
  --output-root <CAMPAIGN>/10_staging \
  --manifest <CAMPAIGN>/10_staging/input_manifest.json \
  --expected-song-count 5

uv run python scripts/midi_to_npz.py \
  --mel-dir <CAMPAIGN>/10_staging/mel \
  --acc-dir <CAMPAIGN>/10_staging/acc \
  --output-dir <CAMPAIGN>/10_staging/npz \
  --ticks-per-beat 4 \
  --strict \
  --expected-manifest <CAMPAIGN>/10_staging/input_manifest.json \
  --summary-json <CAMPAIGN>/10_staging/conversion_summary.json
```

必须得到 40 converted、0 skipped、exact stem set，并逐输入通过 sidecar replay、MIDI/NPZ roll、source-acc hash 和 horizon gate。

## 4. 在看不到 formal result 时冻结 selection 与 renderer

每首 excerpt 起点是 model beat 整数。默认 `[16,32)`；若某首 horizon 不允许，必须在 formal 前通过 JSON 一次性冻结五首的起点，不能在看到输出后调整。

```bash
uv run python scripts/prepare_robustness_listening.py freeze-triangle-selection \
  --input-manifest <CAMPAIGN>/10_staging/input_manifest.json \
  --excerpt-start-beat 16 \
  --output <CAMPAIGN>/00_freeze/triangle_selection.json

uv run python scripts/prepare_robustness_listening.py freeze-triangle-renderer \
  --fluidsynth "$FS_BIN" \
  --fluidsynth-root "$FS_ROOT" \
  --soundfont "$SOUNDFONT" \
  --output <CAMPAIGN>/00_freeze/renderer_identity.json
```

selection exact validator 必须重建 95 scored trials、3 practice、285 scored presentations，以及 60 medium、10 high、5 sham sampling、6 identity、6 known-different、8 exact repeat。每个 19 题 prefix 的 song/control 覆盖、每 condition 的 six-pattern/odd-position/duplicated-arm 平衡也属于 frozen contract。

## 5. Fixed-20 qualification 与 C5

先提交代码并确认 `git status --short` 为空，再把字面 commit SHA 传入：

```bash
uv run python scripts/qualify_perturbation_campaign.py plan \
  --checkpoint <CHECKPOINT> \
  --input-manifest <CAMPAIGN>/10_staging/input_manifest.json \
  --code-identity <CLEAN_COMMIT_SHA> \
  --attestation-dir <CAMPAIGN>/00_freeze \
  --dense-song 2 \
  --tail-song 2 \
  --tail-song 4 \
  --output-dir <CAMPAIGN>/20_qualification

read -r QUAL_CONFIG_SHA _ \
  < "<CAMPAIGN>/20_qualification/qualification_config.json.sha256"
read -r QUAL_SCHEDULE_SHA _ \
  < "<CAMPAIGN>/20_qualification/qualification_manifest.jsonl.sha256"

uv run python scripts/run_perturbation_matrix.py run \
  --qualification \
  --config <CAMPAIGN>/20_qualification/qualification_config.json \
  --config-sha256 <QUAL_CONFIG_SHA> \
  --schedule <CAMPAIGN>/20_qualification/qualification_manifest.jsonl \
  --schedule-sha256 <QUAL_SCHEDULE_SHA> \
  --output-root <CAMPAIGN>/20_qualification/output

uv run python scripts/qualify_perturbation_campaign.py evaluate \
  --config <CAMPAIGN>/20_qualification/qualification_config.json \
  --config-sha256 <QUAL_CONFIG_SHA> \
  --schedule <CAMPAIGN>/20_qualification/qualification_manifest.jsonl \
  --schedule-sha256 <QUAL_SCHEDULE_SHA> \
  --output-root <CAMPAIGN>/20_qualification/output \
  --static-summary <CAMPAIGN>/10_staging/conversion_summary.json \
  --output <CAMPAIGN>/20_qualification/qualification_result.json

uv run python scripts/qualify_perturbation_campaign.py freeze \
  --candidate-config <CAMPAIGN>/20_qualification/qualification_config.json \
  --qualification-result <CAMPAIGN>/20_qualification/qualification_result.json \
  --listening-manifest <CAMPAIGN>/00_freeze/triangle_selection.json \
  --renderer-identity <CAMPAIGN>/00_freeze/renderer_identity.json \
  --output <CAMPAIGN>/00_freeze/campaign_config.json

read -r CONFIG_SHA _ < "<CAMPAIGN>/00_freeze/campaign_config.json.sha256"
```

`evaluate` 和 independent validator 都会从 20 个 immutable attempt artifacts 重推 determinism、static gate、tempo、tail 和最终 decision。`--development-only` result 永远不能 freeze。

## 6. Formal 160 runs 与 audit

```bash
uv run python scripts/run_perturbation_matrix.py schedule \
  --config <CAMPAIGN>/00_freeze/campaign_config.json \
  --config-sha256 <CONFIG_SHA> \
  --output <CAMPAIGN>/00_freeze/run_manifest.jsonl

read -r SCHEDULE_SHA _ < "<CAMPAIGN>/00_freeze/run_manifest.jsonl.sha256"

uv run python scripts/run_perturbation_matrix.py run \
  --config <CAMPAIGN>/00_freeze/campaign_config.json \
  --config-sha256 <CONFIG_SHA> \
  --schedule <CAMPAIGN>/00_freeze/run_manifest.jsonl \
  --schedule-sha256 <SCHEDULE_SHA> \
  --output-root <CAMPAIGN>/30_formal_runs

uv run python scripts/run_perturbation_matrix.py audit \
  --config <CAMPAIGN>/00_freeze/campaign_config.json \
  --config-sha256 <CONFIG_SHA> \
  --schedule <CAMPAIGN>/00_freeze/run_manifest.jsonl \
  --schedule-sha256 <SCHEDULE_SHA> \
  --output-root <CAMPAIGN>/30_formal_runs \
  --output <CAMPAIGN>/30_formal_runs/campaign_audit.json

read -r CAMPAIGN_AUDIT_SHA _ \
  < "<CAMPAIGN>/30_formal_runs/campaign_audit.json.sha256"
```

audit 必须是 160 expected、0 missing、0 invalid、无 extra run ID，并且 `listening_source_readiness.ready=true`。RT gate 会验证 full request lifecycle、raw-token/output-event digest、schedule trace、`theoretical_model.mid` 和 `combined.mid`；合法 empty accompaniment 会被记录为 empty-success，导出失败不会伪装成空成功。

## 7. Objective analysis

```bash
uv run python scripts/analyze_perturbation_robustness.py campaign \
  --config <CAMPAIGN>/00_freeze/campaign_config.json \
  --config-sha256 <CONFIG_SHA> \
  --schedule <CAMPAIGN>/00_freeze/run_manifest.jsonl \
  --schedule-sha256 <SCHEDULE_SHA> \
  --output-root <CAMPAIGN>/30_formal_runs \
  --report-dir <CAMPAIGN>/40_analysis

read -r CONTROL_REPORT_SHA _ \
  < "<CAMPAIGN>/40_analysis/control_report.json.sha256"
```

objective metrics、paired contrasts、coverage、factorial interaction、song-block bootstrap 和 controls 只用于报告及听测结果对齐，不能改变盲题。

当前 v3 campaign 的 harmonic/coverage/identity controls 通过，但冻结的 +1-tick rhythm control 为
0/5；因此 rhythm endpoint 是 assay-invalid，strict report 的 `targeted_controls=false`。不要事后降低
threshold 或覆盖 `40_analysis`。这不阻止 acc-only triangle 听测，但 objective rhythm 结论必须保持无效；
详见 [`2026-07-17 completion report`](../../developing-logs/reports/2026-07-17-melody-perturbation-generated-acc-listening-completion-report.md)。

## 8. 构建并审计 95 题 blind package

```bash
uv run python scripts/prepare_robustness_listening.py build-triangle \
  --config <CAMPAIGN>/00_freeze/campaign_config.json \
  --config-sha256 <CONFIG_SHA> \
  --selection <CAMPAIGN>/00_freeze/triangle_selection.json \
  --schedule <CAMPAIGN>/00_freeze/run_manifest.jsonl \
  --schedule-sha256 <SCHEDULE_SHA> \
  --output-root <CAMPAIGN>/30_formal_runs \
  --campaign-audit <CAMPAIGN>/30_formal_runs/campaign_audit.json \
  --campaign-audit-sha256 <CAMPAIGN_AUDIT_SHA> \
  --control-report <CAMPAIGN>/40_analysis/control_report.json \
  --control-report-sha256 <CONTROL_REPORT_SHA> \
  --package-dir <CAMPAIGN>/50_listening \
  --fluidsynth "$FS_BIN" \
  --fluidsynth-root "$FS_ROOT" \
  --soundfont "$SOUNDFONT"

uv run python scripts/prepare_robustness_listening.py audit-triangle \
  --package-dir <CAMPAIGN>/50_listening
```

只有 `accepted_final=true` 的 44.1 kHz/PCM16/WAV package 可用于正式听测。`--midi-only` 永远是 development-only。
`build-triangle` 会强制重验 `control_report.listening_known_different` 的 6 条记录，并要求 selection selector/recipe、formal 原始 MIDI、velocity-preserving comparator excerpt 与 velocity=96 synthetic excerpt 的 SHA-256 在 analyzer、canonical source 和 blind package 三方完全闭合；control report 的路径和哈希会写入 private key、render manifest 和 package audit。
若这是 QC retry，control report 仍绑定 immutable C5 base selection；package 另外记录 base/current selection hashes，并按稳定 `semantic_id` 对当前 retry 的 canonical material 重新闭环，不要求 retry 后的盲题号与 C5 题号相同。

只有一个**完整 95/95、已经 immutable snapshot 并完成 semantic unblind/summary、且
`qc_status=fail` / `retry_required=true`** 的 accepted-final package 才能授权下一轮 listening
attempt。partial、QC pending、QC pass 或尚未解盲的 snapshot 都不能 retry。先保留失败 package 原样，
再派生新 selection；不要手工换 seed 或编辑旧 selection：

```bash
FAILED_PACKAGE="<CAMPAIGN>/50_listening"
FAILED_SNAPSHOT="$FAILED_PACKAGE/snapshots/<FAILED_SNAPSHOT_ID>"
RETRY_SELECTION="<CAMPAIGN>/50_listening_attempt-002.selection.json"
RETRY_PACKAGE="<CAMPAIGN>/50_listening_attempt-002"

uv run python scripts/prepare_robustness_listening.py derive-triangle-retry \
  --failed-package "$FAILED_PACKAGE" \
  --failed-snapshot "$FAILED_SNAPSHOT" \
  --output "$RETRY_SELECTION"

uv run python scripts/prepare_robustness_listening.py build-triangle \
  --config <CAMPAIGN>/00_freeze/campaign_config.json \
  --config-sha256 <CONFIG_SHA> \
  --selection "$RETRY_SELECTION" \
  --schedule <CAMPAIGN>/00_freeze/run_manifest.jsonl \
  --schedule-sha256 <SCHEDULE_SHA> \
  --output-root <CAMPAIGN>/30_formal_runs \
  --campaign-audit <CAMPAIGN>/30_formal_runs/campaign_audit.json \
  --campaign-audit-sha256 <CAMPAIGN_AUDIT_SHA> \
  --control-report <CAMPAIGN>/40_analysis/control_report.json \
  --control-report-sha256 <CONTROL_REPORT_SHA> \
  --package-dir "$RETRY_PACKAGE" \
  --fluidsynth "$FS_BIN" \
  --fluidsynth-root "$FS_ROOT" \
  --soundfont "$SOUNDFONT"

uv run python scripts/prepare_robustness_listening.py audit-triangle \
  --package-dir "$RETRY_PACKAGE"
```

`derive-triangle-retry` 会在 failed package 的 `retry_authorizations/` 中原子写入并 hash-bind 失败
attempt 的 package、selection、audit、render/private manifests、response/sitting ledgers、snapshot、
unblind state 和 summary。attempt-002 只重新派生盲序、题号和 presentation；semantic
trial/source/excerpt/control/repeat 集合不变。`RETRY_PACKAGE` 在 build 前必须不存在，build 后两条 ledger
必须为空；后续 serve、snapshot、unblind 和 report 都改用该新 package。report 会沿 lineage 保留并验证
失败 attempt，不能只报告后来通过的一轮。

## 9. Flexible listening：答多少保存多少

推荐在服务器启动持久化播放器，并用 SSH port forwarding 在本地浏览器访问：

```bash
uv run python scripts/prepare_robustness_listening.py serve-triangle \
  --package-dir <CAMPAIGN>/50_listening \
  --host 127.0.0.1 \
  --port 8765

# 在本地终端：
ssh -L 8765:127.0.0.1:8765 <SERVER>
```

打开 `http://127.0.0.1:8765/`。三道 practice 不计分；正式题每段至少播放一次。开始正式作答前，
在页面填写设备和环境并点击“开始新的 sitting”；准备暂停时点击“结束当前 sitting”，可同时记录异常。
只有 active structured sitting 才允许提交 response。HTTP 模式只有在 sitting/response hash-chain ledger
`fsync` 和 progress atomic update 成功后才显示“已保存”。仅关闭页面不会伪造 sitting end event；同一浏览器
重开后可恢复仍 active 的 sitting，或先显式结束再开始新的 sitting。每次可答任意数量。

实际播放器文件是 `<CAMPAIGN>/50_listening/blind/player.html`。正式答题应优先使用上面的
`serve-triangle` HTTP mode；直接双击静态 HTML 只写浏览器 localStorage，不会自动写服务器 ledger。

若只下载 `blind/` 静态目录，播放器会使用浏览器 localStorage 并导出 `triangle-responses.json`；必须及时导回服务器：

```bash
uv run python scripts/prepare_robustness_listening.py import-triangle-responses \
  --package-dir <CAMPAIGN>/50_listening \
  --responses <DOWNLOADED>/triangle-responses.json
```

查看盲态进度不解盲。`--new-sitting` **只在输出中给出一个建议 ID**，不会创建、保留或开始
sitting：

```bash
uv run python scripts/prepare_robustness_listening.py resume-triangle \
  --package-dir <CAMPAIGN>/50_listening \
  --new-sitting
```

若使用 CLI 而不是页面按钮，必须显式写入 start/end。把下面的 `sitting-001` 替换为上一步建议值或
自己的 opaque ID；response 只能引用尚未 end 的 sitting：

```bash
uv run python scripts/prepare_robustness_listening.py start-triangle-sitting \
  --package-dir <CAMPAIGN>/50_listening \
  --sitting-id sitting-001 \
  --device "<HEADPHONES_OR_SPEAKERS>" \
  --environment "<LISTENING_ENVIRONMENT>" \
  --note "<OPTIONAL_START_NOTE>"

# 本次作答结束或准备暂停时：
uv run python scripts/prepare_robustness_listening.py end-triangle-sitting \
  --package-dir <CAMPAIGN>/50_listening \
  --sitting-id sitting-001 \
  --anomaly "<OPTIONAL_ANOMALY>" \
  --note "<OPTIONAL_END_NOTE>"
```

## 10. 任意 partial snapshot、解盲与 report

有至少一题后即可 snapshot；不要求答满：

```bash
uv run python scripts/prepare_robustness_listening.py snapshot-triangle \
  --package-dir <CAMPAIGN>/50_listening

uv run python scripts/prepare_robustness_listening.py unblind-triangle \
  --package-dir <CAMPAIGN>/50_listening \
  --snapshot-dir <CAMPAIGN>/50_listening/snapshots/<SNAPSHOT_ID>
```

第一次 semantic unblind 会冻结不可重置的 boundary。之后仍可答题，但新 rows 永久标记 `post_partial_unblind_exploratory`。解盲后会生成 `generated_acc_after_unblind/generated_acc_index.csv/json` 和带 semantic filename 的 MIDI/WAV；这些文件只用于听后自由复听，不回写 primary responses。

0 题时可生成 objective-only report。这里的 0 题必须是真正的空 response ledger、没有 snapshot、也没有
semantic-unblind state；不能通过省略参数来隐藏已经采集的数据。命令中**不传**
`--listening-snapshot`：

```bash
uv run python scripts/build_robustness_report.py \
  --config <CAMPAIGN>/00_freeze/campaign_config.json \
  --config-sha256 <CONFIG_SHA> \
  --schedule <CAMPAIGN>/00_freeze/run_manifest.jsonl \
  --schedule-sha256 <SCHEDULE_SHA> \
  --campaign-root <CAMPAIGN>/30_formal_runs \
  --campaign-audit <CAMPAIGN>/30_formal_runs/campaign_audit.json \
  --analysis-dir <CAMPAIGN>/40_analysis \
  --listening-package <CAMPAIGN>/50_listening \
  --output-dir <CAMPAIGN>/60_report/not_started
```

回答 1–94 或 95 题后，必须先对所选 immutable snapshot 执行上面的 `unblind-triangle`，再显式传入该
snapshot；report 只使用这个 sealed prefix，不会把其后的 response 混入：

```bash
uv run python scripts/build_robustness_report.py \
  --config <CAMPAIGN>/00_freeze/campaign_config.json \
  --config-sha256 <CONFIG_SHA> \
  --schedule <CAMPAIGN>/00_freeze/run_manifest.jsonl \
  --schedule-sha256 <SCHEDULE_SHA> \
  --campaign-root <CAMPAIGN>/30_formal_runs \
  --campaign-audit <CAMPAIGN>/30_formal_runs/campaign_audit.json \
  --analysis-dir <CAMPAIGN>/40_analysis \
  --listening-package <CAMPAIGN>/50_listening \
  --listening-snapshot <CAMPAIGN>/50_listening/snapshots/<SNAPSHOT_ID> \
  --output-dir <CAMPAIGN>/60_report/<SNAPSHOT_ID>
```

三个状态轴必须分开：

- `collection_status=not_started/partial/full` 只表示 0、1–94、95 题；
- `qc_status=not_started/pending/pass/fail` 分别表示尚无回答、QC 证据未完整、通过或失败；
- `blinding_status` 表示是否在继续收集期间看过 semantic result。

只有某 condition 的 20 题全在第一次解盲前完成且完整 QC pass，才能使用预注册 `confirmed/not confirmed`。其余情况仍按实际分母报告，并写 `partial — preregistered decision pending`。

## 11. 文件在哪里

- 每个 formal run 的 generated acc：`30_formal_runs/runs/<RUN_ID>/attempt-XXX/**/theoretical_model.mid`；
- blind player：`50_listening/blind/player.html`；
- 不泄义的盲听 WAV：`50_listening/blind/trials/Qxxx/clip_{1,2,3}.wav`；
- 解盲后的可读 MIDI/WAV：`50_listening/generated_acc_after_unblind/`；
- 当前人耳结果：所选 `snapshots/<SNAPSHOT_ID>/partial_discrimination_summary.json`；
- 综合报告：`60_report/<REPORT_ID>/report.md`；建议为 `not_started` 和每个 immutable snapshot 使用不同目录。
- QC retry selection/package：`50_listening_attempt-00N.selection.json` 与 `50_listening_attempt-00N/`；失败 package 及其 `retry_authorizations/` 永久保留。

所有 formal generated files 都保留。staging `acc` 只是原始 accompaniment 副本，不是模型对扰动 melody 新生成的 acc。
