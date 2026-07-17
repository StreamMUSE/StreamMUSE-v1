# Melody 扰动是否造成“可听的生成伴奏差异”——继续完成计划（2026-07-17）

> 状态：technical implementation 已落地并进入最终回归；clean frozen campaign 与 human listening 分阶段执行。
>
> 本计划基于 2026-07-16 implementation report 和 2026-07-17 的代码复核。它复用
> `2026-07-10-melody-perturbation-robustness-plan.md` 的 40-input、160-run、seed、指标和
> artifact binding 设计，但**替换旧 plan 的 Phase F / T10 听测目标**。旧的 24-clip、
> `overall quality 1–5` 方案不能回答“扰动后生成的 acc 本身是否听起来不一样”，因此不再是
> 本实验的完成判据；如以后仍要做质量评价，只能作为独立的 secondary study。
>
> 2026-07-17 flexible-listening update：95 题是可逐步积累的完整题库，不是一次性提交要求。播放器
> 必须逐题原子保存、可随时暂停/恢复，并能在任意已答题数上生成明确标注分母的 partial result。

## 1. 最终要回答的问题

本轮唯一 primary question 是：

> 对同一首歌、同一固定片段、同一 model checkpoint、同一 sample seed 和同一
> `rt_theoretical` generation pipeline，仅把 melody condition 从 sham 改成
> `pitch`、`onset` 或 `both` 后，模型生成的 accompaniment 是否能被用户本人在盲听中区分？

这里的“生成伴奏”特指 formal RT attempt 中由模型 token/trace 恢复出的
`theoretical_model.mid`，不是 staging 目录里从原数据复制过来的 source `acc`，也不是输入用的
perturbed melody。

本轮的结论边界：

- objective metrics 只能说明 token、note、onset、coverage 等在数值上变了多少；不能替代人耳判断
  “是否可听”。
- primary 听测只播放 accompaniment，不播放 clean/dirty melody、metronome，也不使用
  `rt_combined`，避免听者直接听到 melody 扰动或 playback scheduling 差异。
- triangle test 只回答“能不能区分”，不回答哪一版更好、是否更和谐、质量是否下降，也不定位具体
  模型机制。
- 单听者、5 首歌的结果只允许写成 fixed-package exploratory evidence；不能外推到其他听者、歌曲、
  真实人类演奏或总体模型鲁棒性。
- 未达到预注册阈值只能写“本轮未确认可听差异”，不能写“完全一样”“没有影响”或“等价”。

## 2. 当前项目状态与缺口

截至本计划写入时：

| 项目 | 当前状态 | 对本目标的意义 |
|---|---|---|
| 扰动/staging 工具 | 已实现并有自动测试 | `/tmp/.../mel` 的 40 个文件只是 formal input 候选 |
| staging `acc` | 是 source accompaniment 副本 | 不是模型根据扰动生成的新 acc，不能用于本听测 |
| offline/RT driver | 160-run、required-artifact、raw-token/MIDI reconciliation 已实现并有正负测试 | 真实 frozen campaign 待执行 |
| objective analyzer | metrics、targeted controls、selection/control hash 闭环已实现 | formal metrics/controls/analysis index 待真实 campaign 生成 |
| 当前 listening tool | v2 95 题 triangle、播放器、structured sitting、partial snapshot/report 和 QC retry 已实现 | 待生成 accepted-final WAV package |
| qualification | fixed-20 artifact re-derivation 与独立 validator 已实现 | 待在 clean commit 上执行并 freeze C5 |
| real checkpoint evidence | song 4/5 × 15/120 BPM consistency 已在当前实现上通过 | clean campaign 仍须归档同一 gate 与 runtime attestation |
| formal generated treatment acc | 不存在 | 当前没有 pitch/onset/both 对应的正式生成伴奏可听文件 |
| human scores/report | 不存在 | 还不能回答最终问题 |

2026-07-16 报告记录的 407 unit + 5 integration passed 是当时工作树的实现证据，不等于 formal
campaign 已通过。当前工作树仍有未提交变更；在 clean commit、真实 checkpoint qualification 和 C5
freeze 之前，不得把任何 `/tmp` 或 development smoke output 当正式结果。

## 3. 本计划冻结的关键决策

### 3.1 Formal generation 仍跑完整 160-run matrix

保持旧 plan 的冻结矩阵：

- 5 songs × 8 input variants = 40 inputs；
- 每个 input 使用 2 个 sample seeds；
- offline 80 + RT 80 = 160 formal model runs；
- pitch/onset/both 各有 2 个 perturb seeds；high-dose 有 1 个 perturb seed；
- formal analysis 仍保留 offline、`rt_theoretical`、`rt_combined`，但 primary human listening 只取
  `rt_theoretical`。

不能为了省时间只临时跑几个听测 run。完整矩阵用于验证生成完整性、seed 配对、pipeline 差异、
coverage 和 operational endpoints，也防止看过部分输出后挑样本。

### 3.2 Qualification 采用固定 20-run 设计

保留现有 canonical fixed-20 schedule：

- offline determinism：2 runs；
- RT determinism：2 runs；
- tempo：2 tempos × sham/both = 4 runs；
- tail：2 tempos × 2 songs × 3 tail candidates = 12 runs。

“determinism → static → tempo → tail”是**判定顺序**，不是执行 short-circuit。20 行全部预运行，
然后按冻结顺序从 artifacts 推导 decision；早期 gate 失败时整个 qualification 失败，后面的 artifacts
不能挽救它。

其他 qualification 决策：

- dense song 必须是 manifest 中稳定 ID `2`；tail songs 必须按稳定顺序为 `2, 4`；
- tempo 60 通过则选 60；否则只有 tempo 30 通过才选 30；两者都失败则停止；
- tail 8、16、24 在 analysis window 内全部一致才取 8；只有 16 与 24 一致才取 16；否则停止排查，
  **不能因为 24 最大就直接接受 24**；
- 20/20 只接受 `attempt-001`，qualification root 不得存在 retry；
- formal campaign 仍可按冻结的 retry policy 保留 immutable retries，但不得覆盖首 attempt。

### 3.3 Primary 听测使用 acc-only triangle discrimination

每个普通 trial 有三段盲化 WAV：其中两段是同一个 condition 文件的 literal byte copies，另一段来自
matched comparator。三段位置以及“sham 还是 treatment 被复制”均随机并平衡。用户回答：

1. odd clip 是 `1`、`2`、`3`，或 `no audible difference`；
2. confidence `1–5`；
3. 可选 difference tags：`pitch/harmony`、`rhythm/timing`、`density`、
   `texture/register`、`silence/coverage`、`other`；
4. 可选备注。

不询问“哪段是 treatment”或“哪段更好”。主 trial 中 `no audible difference` 计为没有成功区分；
identity catch 单独以该回答作为正确 response，不进入 1/3 chance 的 triangle hit rate。

### 3.4 Trial 数量和 seed 覆盖

| block | 数量 | 组成 | 用途 |
|---|---:|---|---|
| medium primary | 60 | 5 songs × 3 conditions × 2 pseed × 2 sseed | pitch/onset/both 各 20 个，回答主问题 |
| high-dose exploratory | 10 | 5 songs × 2 sseed | 看更强扰动是否更易造成可听差异；不是 QC gate |
| sham sampling baseline | 5 | 每歌 sham-sseed1 vs sham-sseed2 | 只作正常 sampling variation 的听觉量级参照 |
| identity catch | 6 | A/A/A literal copies | 检查 false-positive / 强行猜 odd 的倾向 |
| known-different control | 6 | sham vs 预冻结 synthetic known-bad | 检查听者/播放器/渲染链能否识别明显差异 |
| exact repeat | 8 | 从 medium primary 预冻结抽取并重新排列 | 检查 intra-listener consistency，不增加主 n |
| **合计题库** | **95** | 另有 3 个不计分 practice trials | 可分任意次数、逐题累计 |

medium primary 使用所有 `(pseed, sseed)` 四种组合。每个 treatment 与 sham comparator 必须使用相同
song、excerpt 和 sample seed；sham 没有 perturb seed。这样条件之间唯一有意改变的是输入 melody
condition。四个 trials/song/condition 不是 20 个独立歌曲样本，因此任何 exact-binomial 数值都只作为
fixed-package 描述，不作总体人群推断。

known-different control 的 song、sample seed 和 deterministic transform recipe 必须在看到 formal
output 前冻结。优先复用 analyzer 的 bar/rhythm shift、coverage dropout 等明显变换；high-dose 不能替代
known-different control，因为 high-dose 可能真实地没有造成可听输出差异。

### 3.5 Excerpt 与 flexible listening progress

- 每段固定 8 秒，即 120 BPM 下 16 model beats / 4 bars；默认窗口冻结为 model beats `[16, 32)`。
- 在任何 formal output 生成前，只用 clean input horizon 自动验证五首歌都覆盖该窗口；若某首不覆盖，
  只能根据 clean horizon 预先指定该歌的 16-beat window，不能听过生成结果后选“差异明显”的片段。
- 同一 song 的所有 condition、seed、control 必须使用完全相同的 excerpt。
- 合成时允许从更早位置带 pre-roll 渲染，再精确裁到冻结窗口，以保留窗口起点的 active notes；边界
  fade、pad 规则必须统一并写入 manifest。
- 95 题使用一个冻结的全局盲序；顺序要做 prefix/chunk balance，使较早停止时 pitch/onset/both、song、
  seed 和 controls 不会严重失衡。
- 用户每次有时间可以只答 1 题或任意多题，答完一题立即原子保存；关闭播放器后，下次从下一道
  unanswered trial 恢复。不得要求凑满固定 session 才能保存或出 summary。
- 每次实际作答前必须显式开始一个 listening sitting，记录 `sitting_id`、设备和环境；浏览器可自动生成
  opaque ID，CLI 可自行指定。结束时显式记录 end、异常和备注；仅关闭页面不会伪造 end event，下次可恢复
  active sitting 或显式结束后开始新的 sitting。sitting 数量和每次题数不冻结，原先的 31/32/32 只可作为
  可选进度提示，不能成为程序 gate。
- 每个 sitting 尽量使用相同耳机、音量和安静环境；practice 完成后固定音量，正式题不反馈答案。
- 用户可以随时暂停；连续听较久时建议每 25–40 分钟至少休息 10 分钟，但不强制一次完成指定数量。
- exact repeat 与原 trial 至少间隔 8 题；六种 triangle 排列
  `AAB/ABA/BAA/BBA/BAB/ABB` 在可行范围内全局及各 condition 平衡。

### 3.6 Canonical rendering 与盲化

- source artifact：`rt_theoretical/theoretical_model.mid`，只含 generated accompaniment；
- render BPM：120；sample rate：44.1 kHz；PCM：16 bit；soundfont、FluidSynth 版本、program、命令和
  平台全部冻结并 hash/记录；
- synth base gain 固定为 0.5；禁止逐 clip LUFS normalization；如 true-peak protection 触发，matched
  trial 内两种 condition 使用同一个 attenuation factor；limit 继续使用 `-0.1 dBTP` 和当前 4×
  inter-sample peak audit；
- 同 condition 的两份 presentation 必须从同一 canonical WAV 作字节级复制，不能分别 render；
- blind package 只能出现 opaque trial/clip IDs；不得泄露 song、condition、pipeline、seed、
  run ID、文件路径或已知 control 类型；
- private key 不直接暴露给 blind player。因为用户本人是唯一 listener，第一次 semantic-unblind snapshot
  前也不得试听按 condition 命名的 formal acc；开发预览只能使用明确标记为 development-only、不会进入
  formal selection 的输出。

### 3.7 Empty output 与 operational-invalid policy

- listening source 必须 content-valid、artifact-complete、hash-valid；`rt_theoretical` 不因 late/drop 等
  operational-invalid 自动换样或排除，operational flags 只进入 provenance 和报告。
- 模型成功生成合法 empty accompaniment 是实验结果，不是 renderer failure：该 clip 保留为 literal
  silence，并标记 `source_empty=true`；非空 MIDI 被错误 render 成 silence 才使 package audit 失败。
- 单边 empty/coverage collapse 造成的明显差异计入主结果，同时另外报告
  `coverage-driven difference`，并给出排除这些 pairs 后的敏感性摘要；不得事后替换 run/window。
- 如果 sham/treatment 的 canonical note events 或最终 WAV 完全相同，该 main trial 仍保留：
  `objective_identity=true`、triangle hit 记 0；用户选择 `no audible difference` 会作为一致证据单列。

### 3.8 预注册的听测 QC 与解释规则

以下 QC 是完整题库上的 final perceptual conclusion gate；partial snapshot 不要求相关题已经全部回答，
但必须把状态写成 `qc_status=pending`，并列出已答/总数。统一状态集合为
`not_started | pending | pass | fail`：

- identity catches：至少 5/6 回答 `no audible difference`；
- known-different controls：至少 5/6 正确定位 odd clip；
- exact repeats：至少 6/8 在重排后保持同一 semantic response（同一 underlying odd source，或两次均
  `no audible difference`）；
- 所有已提交 response 都有 hash-chain provenance、无事后删题/换题；未答题只标 `pending`，不能补 0、
  猜测或从分母中静默消失。

若 QC 失败，先 seal 并保留失败的 package/score，不允许只删除错题；按预先定义、由 base blind seed
domain-separated 派生的下一 listening attempt 重新生成完整盲序，休息后重做。最终报告必须披露前一
attempt 失败，不能只报告通过的一轮。

retry 只允许来自一个 accepted-final package 的 immutable 95-response snapshot，且该 snapshot 已完成
semantic unblind/summary、`qc_status=fail`、`retry_required=true`。`derive-triangle-retry` 必须原子生成
authorization，hash-bind 失败 attempt 的 selection/package/audit、response/sitting ledgers、snapshot、
unblind state 和 summary；partial、QC pending/pass 均不得授权。下一 attempt 只改变盲序、题号和
presentation，不改变 semantic trial/source/excerpt/control/repeat 集合，C5 继续绑定 attempt-001 base
selection。新 package 必须使用独立目录并从空 response/sitting ledger 开始。

每个 medium condition 的完整题库有 20 个主 trials。操作性判定为：

- `confirmed discriminable in this fixed listener/package`：该 condition 的 20 题都在第一次 semantic
  unblind 前完成、完整 QC 通过、正确定位至少 12/20，且至少 4/5 songs 各有不少于 2/4 正确；
- 该 condition 的 20 题都在第一次 semantic unblind 前完成但不满足上条：
  `not confirmed in this listening run`，报告准确 hit rate、
  每歌 0–4、confidence、
  `no_difference`、objective-identity 和 coverage-driven counts，不做等价结论。

condition 尚未答满 20 题时也必须立即可报告，但状态为
`partial — preregistered decision pending`，使用实际分母（例如 `pitch 3/7 correct`）并同时报告
answered/pending、涉及 songs/seeds、当前 controls/QC 覆盖和 confidence。partial 结果不能套用 12/20，
也不能因为当前比例较高/较低提前写 `confirmed` 或 `not confirmed`。

12/20 是比单个未校正 5% 阈值更保守的固定 operational threshold，方便同时查看三个 condition；
由于同一 listener、同一 song 和重复使用 sham sources 带来依赖，不能把它包装成 population-level
显著性。high-dose 的 10 个 trials 和 sham sampling baseline 只报告 raw rate；可把 high-dose
`≥7/10` 标记为强的 fixed-package indication，但它不决定主听测是否有效。

### 3.9 随时保存、随时出 partial result

- blind player 每答完一题就把 response 以 append-only/hash-chained record 原子写入
  `response_ledger.jsonl`，同时更新可恢复的 `progress_state.json`；进程退出或文件复制不能丢掉已答题。
- 任意 `answered_count >= 1` 都可以生成一个 immutable snapshot。snapshot 固定“截至哪个 ledger hash、
  哪些 trial 已答、哪些 pending”，后续回答不会改写旧 snapshot。
- `blind-progress summary` 不解盲，只显示完成数、sitting、漏答、response/QC presentation 覆盖，可在
  保持盲态时无限次查看并继续答题。
- 用户也可以在任意 snapshot 请求 `semantic partial summary`：只解盲已经回答的 rows，按
  pitch/onset/both/high/sham-noise 整理实际分母和结果，未答 rows 保持隐藏/pending。
- 一旦用户看过 semantic partial summary，系统记录 `first_semantic_unblind_at`。之后仍可继续回答，所有
  新 response 都保留并可汇总，但标记为 `post-partial-unblind exploratory`；最终报告同时给出
  pre-unblind、post-unblind 和 combined-descriptive 三个 view，不能把后续数据伪装成完全盲态结果。
- 未查看 semantic result 时可以不限次数暂停/恢复，不影响 blind status。用户若只想保持最严格的完整
  blind conclusion，就在最后一次作答前只查看 blind-progress summary。
- report builder 在 1–94 题时不得报错或要求补齐；它生成 `collection_status=partial` 的有效 report。
  95/95 时为 `collection_status=full`。0 题时仍可生成 objective-only report，但明确
  `collection_status=not_started`，不得产生 human perceptual conclusion；这里的 0 题必须是真正的空
  response ledger、无 snapshot、无 semantic-unblind state，不能靠省略 snapshot 隐藏已有数据。

## 4. 需要新增或修正的实现

### 4.1 Qualification fail-closed hardening

必须先关闭以下 blocker，之后才允许运行正式 qualification：

1. 从 `evaluate()` 抽出共享纯函数 `derive_qualification_decision(evidence)`；
2. `evaluate`、`validate_qualification_result`、`freeze` 和全部 downstream consumers 都从 pinned
   20-run artifacts 重算 determinism/static/tempo/tail decision，并与 result exact-compare；
3. 正式 qualification 禁止 `--allow-dirty`；development smoke 明确写
   `development_only=true`，`freeze` 无条件拒绝；
4. shared contract exact-freeze dense song `2`、tail songs `[2,4]`、fixed-20 schedule 和上述
   tempo/tail decision；
5. qualification verifier 只接受 `attempt-001` 且 run root 不得含 retry attempt；
6. evidence/result path 改为 campaign-root-relative 并支持 archive-root resolution，或把 durable campaign
   root 定为不可移动；本轮优先使用后者，避免运行后搬目录使验证失效。

必须新增负向测试：完整真实 evidence graph 不变，但篡改 `passed`、selected tempo、selected tail、
determinism/static decision 任一字段时，validator/freeze/formal consumer 全部 fail closed。

### 4.2 Formal RT required-artifact contract

当前 verdict 自建 artifact index 还不足以保证“听测需要的 generated acc 一定存在”。为 RT attempt 增加
required-artifact allowlist/schema，至少要求并 hash：

- `theoretical_model.mid`；
- `theoretical_model_summary.json`；
- `combined.mid`；
- `model_schedule_trace.jsonl`；
- `request_lifecycle.jsonl`；
- `validity.json`；
- full inference/raw-token trace；
- command/process/runtime/checkpoint/input/reset gates；
- immutable verdict 和 artifact index。

`SessionLogger._write_theoretical_model_midi()` 的异常不得再 silent-pass。post-run gate 要从 raw token/
schedule trace 独立重建 theoretical logical-tick notes，与 `theoretical_model.mid` 做 semantic equality；
combined accompaniment 也要与 scheduled ticks 对齐。formal audit 额外输出
`listening_source_readiness`，确保预冻结 selection 所引用的每个 RT source 都存在、attempt/verdict/artifact
hash 完整。

### 4.3 Listening contract v2

在 shared module 中新增并 strict-validate
`streammuse.melody_robustness.listening_triangle_selection.v2`，不要继续把所有概念塞进旧的
`pipeline` 字段。每个 source 至少包含：

- `formal_pipeline: rt`；
- `source_artifact: theoretical_model | control`；
- `presentation: acc_solo`；
- `question_id` 和 `semantics_version`；
- song/condition/pseed/sseed selector；
- attempt policy、excerpt model ticks、trial/order role；实际 `sitting_id` 只随答题进度记录，不冻结；
- source empty policy、render policy、control/repeat mapping；
- config/input/schedule/qualification/selection bindings。

更新 `campaign_config.json` listening section：schema v2、95-trial frozen pool、3 practice trials、
8 seconds、prefix-balanced global order、flexible sittings、per-trial persistence、partial snapshot semantics、
triangle response schema、QC/decision rules 以及 selection path/hash。C5 仍必须在 formal outputs 前绑定
selection hash。

### 4.4 Listening builder/auditor/scorer

扩展 `scripts/prepare_robustness_listening.py`，建议保留 v1 命令用于历史兼容，并新增：

- `freeze-triangle-selection`；
- `build-triangle`；
- `audit-triangle`；
- `derive-triangle-retry`；
- `serve-triangle`；
- `resume-triangle`；
- `start-triangle-sitting` / `end-triangle-sitting`；
- `record-triangle-response` / `import-triangle-responses`；
- `snapshot-triangle`；
- `seal-triangle-scores`；
- `unblind-triangle`；
- `summarize-triangle`。

builder 不得继续使用较弱的私有 `_verified_attempt()`；统一调用 shared immutable-attempt verifier，并验证
actual file set 等于 artifact index。private/render provenance 要增加：attempt ID、verdict path/hash、artifact
kind/path/hash、content/operational validity、source empty/note counts、excerpt ticks、program、rendered content
和 common attenuation。

blind public manifest/播放器必须显示不泄密的正式问题：

> “三段都只包含模型生成的伴奏。请选择听起来不同的一段；若确实无法听出差异，选择
> ‘no audible difference’。不要评价哪段更好。”

建议生成本地静态 blind player，按 frozen global order 播放、每题记录 play count/response time 后原子
autosave，并能从 progress state 恢复；“暂停”不是漏答或错误。若暂不做 UI，CLI/CSV workflow 也必须支持
逐题 append 和 exact-validate 全部 response 字段。任意 snapshot/seal 检查已答 rows 一一对应、无重复或
改写，允许 1–95 行并明确 pending rows；unblind/summarize 只处理 snapshot 中已答 rows，输出实际分母、
control QC 覆盖、per-condition/per-song、repeat consistency 和 exclusion-free/sensitivity views。

### 4.5 Final report builder

更新 `scripts/build_robustness_report.py` 的 DoD：旧的 24-clip overall-rating package 不再满足本计划。
report 必须验证 triangle selection/render/audit、append-only response ledger、所选 immutable snapshot/
seal、partial/full unblinded scores、QC coverage/summary、discrimination summary 和所有上游 hash。回答未满
95 题是合法的 `partial` 状态，不能使 builder fail；但 builder 必须禁止 partial 数据使用 full-sample
`confirmed/not confirmed` 措辞。

最终 report 要并排给出：

- raw token / canonical MIDI / rendered WAV 的 exact identity counts；
- objective sensitivity/quality/coverage metrics；
- pitch、onset、both、high-dose、sham-noise 的盲听结果；
- coverage-driven 与 objective-identical trials；
- operational-invalid provenance；
- QC、repeat consistency、listener/sitting limitations；
- 允许/不允许的结论措辞。

## 5. Durable campaign artifact layout

正式 root 一开始就创建在不会移动、不会自动清理的位置：

```text
task_runs/melody_robustness_<campaign_id>/
├── 00_freeze/
│   ├── code_identity.json
│   ├── checkpoint_identity.json
│   ├── environment.json
│   ├── qualification_spec.json
│   ├── renderer_identity.json       # binary/libs/soundfont/program/render policy
│   ├── triangle_selection.json      # immutable attempt-001 base selection
│   ├── campaign_config.json         # C5
│   ├── run_manifest.jsonl           # exact 160 rows
│   └── gold_consistency.junit.xml
├── 10_staging/
│   ├── mel/
│   ├── acc/                         # source copies，仅作模型输入配套
│   ├── npz/
│   ├── input_manifest.json
│   └── conversion_summary.json
├── 20_qualification/
│   ├── qualification_config.json
│   ├── qualification_manifest.jsonl # exact 20 rows
│   ├── qualification_plan.json
│   ├── output/runs/.../attempt-001/
│   ├── qualification_result.json
│   └── output/campaign_binding.json
├── 30_formal_runs/
│   ├── campaign_binding.json
│   ├── runs/<run_id>/<attempt-id>/
│   │   └── .../theoretical_model.mid # 真正 generated RT acc source
│   └── campaign_audit.json
├── 40_analysis/
│   ├── run_metrics.jsonl
│   ├── paired_contrasts.jsonl
│   ├── control_report.json
│   ├── bootstrap.json
│   └── analysis_index.json
├── 50_listening/                     # listening-attempt-001 package
│   ├── blind/
│   │   ├── trials/<trial_id>/{clip_1.wav,clip_2.wav,clip_3.wav}
│   │   ├── public_manifest.json
│   │   ├── response_ledger.jsonl    # 每题 append-only/hash-chain
│   │   ├── sitting_ledger.jsonl     # start/end/device/environment/anomalies
│   │   └── progress_state.json      # 随时暂停/恢复
│   ├── private/                     # semantic snapshot 前不要打开
│   │   └── private_key.json
│   ├── render_manifest.json
│   ├── package_audit.json
│   ├── unblind_state.json            # 第一次 semantic boundary；首次解盲后才存在
│   ├── retry_authorizations/         # 仅 full QC fail 后才存在
│   ├── snapshots/<snapshot_id>/
│   │   ├── sealed_responses.json
│   │   ├── blind_progress_summary.json
│   │   ├── partial_unblinded_scores.json
│   │   └── partial_discrimination_summary.json
│   └── generated_acc_after_unblind/ # 首次 semantic unblind 后的 immutable semantic export
│       ├── generated_acc_index.{json,csv}
│       └── .../*.mid, .../*.wav
├── 50_listening_attempt-002.selection.json # 仅 full QC fail 后由 authorization 派生
├── 50_listening_attempt-002.selection.json.sha256
├── 50_listening_attempt-002/         # 仅 QC retry；新盲序、空 response/sitting ledger
│   └── ...
└── 60_report/
    └── <report-id>/                  # not_started 或 immutable snapshot ID，避免覆盖旧报告
        ├── report.md
        ├── reproducibility_index.json
        └── generated_acc_after_unblind/ # n>=1 时复制并重验 package semantic export
            ├── generated_acc_index.{json,csv}
            └── .../*.mid, .../*.wav
```

用户保持盲态时只进入 `50_listening/blind/`。任意已答题数都可以建立 snapshot 并生成 partial report；
如果用户选择查看 semantic partial summary，则记录第一次部分解盲时间，之后的 response 标记为
post-partial-unblind。`generated_acc_after_unblind/` 在第一次 semantic unblind 后即可生成，但生成/试听
时间也必须记录，后续数据不能再声称是完全 blind primary result。

## 6. 分阶段执行计划

### Phase P0 — 关闭 formal blockers 和冻结新 spec

目标：任何伪造、缺 artifact、空输出或 operational flag 都按唯一合同处理，且新听测问题 machine-readable。

- 修 qualification re-derivation、clean/first-attempt/fixed-song/fixed-20/tail-stop contract；
- 修 RT required-artifact gate 和 theoretical export silent failure；
- 实现 listening schema v2、95-trial manifest、shared attempt verifier、empty policy、public prompt；
- 更新 analyzer/report builder 和 developer guide；
- 补所有正向/负向 unit/integration tests；
- 只用 synthetic fixtures 端到端演练 triangle freeze → build → audit → seal → unblind → summarize。

Gate P0：所有测试通过；对 decision/result/artifact/listening mapping 的任意篡改均 fail closed；不需要真实
treatment output 即可完成。

### Phase P1 — Clean freeze、真实模型 regression 和 fresh staging

目标：从一次 clean、可复现的代码/checkpoint/environment 状态开始，不复用 `/tmp` smoke artifacts。

- 提交本计划及相关代码/tests/docs，确认 `git status --short` 为空；
- 记录 code SHA、checkpoint SHA-256、依赖锁、GPU/CUDA/PyTorch/device/dtype；
- 配置真实 `LEKAI_CHECKPOINT_PATH`，运行 song 4/5 × tempo 15/120 gold consistency；
- 在 durable campaign root 重新生成 40 MIDI + sidecar + NPZ + source acc staging；
- 验证 40/40 input hash、sidecar replay、MIDI roll == NPZ part0 roll、analysis horizons；
- 在不查看任何 formal output 前冻结 excerpt、95 trials、control recipes、prefix-balanced global order 和
  soundfont/render identity，写 `triangle_selection.json` 及 hash。

Gate P1：clean attestation、real-checkpoint gold、40-input static gate 和 selection validator 全绿。

### Phase P2 — Canonical qualification 和 C5 freeze

目标：证明同 input/same seed determinism、runtime 可用和 tail 收敛，并从 evidence 唯一推导 C5。

- 生成 candidate config、qualification spec 和 exact 20-row schedule；
- clean worktree 运行 20/20 `attempt-001`，禁止 retry/allow-dirty；
- 对 offline/RT raw token、theoretical MIDI、dynamic input trace、tempo operational gate、tail convergence
  做 artifact-derived evaluation；
- 由独立 validator 再推导一次 result；
- 只有 result 完全一致且 passed 才运行 `freeze`，将 qualification result 和 triangle selection hashes
  绑定进 final `campaign_config.json`。

Gate P2：passed qualification、无 retry/dirty、C5 hash 唯一且所有 consumer 可复验。

### Phase P3 — Formal 160 runs 和完整性审计

目标：真正生成 pitch/onset/both/high 的模型 accompaniment。

- 从 C5 + 40-input manifest 重建 exact 160-row schedule；
- 先跑 offline 80，再跑 RT 80，避免两个 GPU model process 竞争；
- 每 run 做 checkpoint/config/input/reset/session gate，保存 immutable attempts；
- RT post-run 强制验证 required artifacts、trace↔theoretical semantic equality 和 empty-success；
- formal content retry 只能按冻结 policy 建新 attempt，首 attempt 永不覆盖；
- 生成 campaign audit，要求 160 expected、0 missing、0 extra、0 content-invalid/corrupt；
- operational-invalid 只标记不隐藏；额外要求所有 listening source `readiness=true`。

Gate P3：`campaign_audit.json` 完整通过。未通过前不得打开 treatment-level metrics 或试听 formal acc。

### Phase P4 — Objective analysis、controls 和 blind package

目标：生成辅助数值结果和一个不泄露条件的、可审计的 95-trial WAV 包。

- 运行 formal analyzer，生成 sensitivity、D_intended/D_actual、coverage、rhythm、factorial、bootstrap；
- 生成预冻结 known-different controls 并验证 endpoint control behavior；
- 在不改变 selection 的前提下解析 75 个 treatment/baseline comparison trials 及其 formal source、
  control、repeat mappings；
- canonical render 8-second acc-only WAV，构造 literal duplicates、prefix-balanced global blind order 和
  public prompt；
- audit 数量、时长、声道、采样率、bit depth、true peak、source hash、duplicate equality、blind leakage、
  expected silence、order/repeat/control mapping；
- package audit 通过后冻结 package hash；在用户听测前不生成带 semantic filename 的试听目录。

Gate P4：analysis index 和 final listening package audit 全绿；blind side 无任何 semantic leakage。

### Phase P5 — 用户按自己的时间逐步盲听

目标：在人耳不知道 condition 的情况下逐题积累 response；任意进度都可安全保存和整理结果。

- 先完成 3 个 practice，确认播放设备和问题理解；
- 之后每次按 frozen global order 答 1 题或任意多题；每题提交即原子保存，可以立刻关闭并以后恢复；
- 每次作答前显式写 sitting start（ID、设备、环境），只有 active sitting 可提交 response；暂停前显式写
  end、异常和备注，仅关闭页面不伪造 end event。可以重听，但不换 trial/window 或获取答案反馈；
- 任意进度可生成 blind progress snapshot；只要至少答 1 题，也可按用户要求生成 semantic partial summary；
- 每个 snapshot 使用当前 response-ledger hash seal，不要求 95 题全部完成；
- 查看 semantic partial summary 后仍可继续答，但后续 rows 自动标记 post-partial-unblind exploratory；
- controls/repeats 尚未全部出现时 QC 标 `pending`，不能把 pending 当 pass/fail。

Gate P5-partial：至少 1 个 response、ledger/snapshot seal 有效即可生成 partial human report。

Gate P5-full（可选继续积累）：95/95 responses、完整 QC 和 full seal 有效；它提供最强的预注册结论，
但不是“允许出结果”的前置条件。

### Phase P6 — Unblind、结论和最终可听文件导出

目标：给出严格受限但直接回答用户问题的结论，并交付可自由试听的生成 acc。

- 按 snapshot 实际 answered denominator 计算 pitch/onset/both 的 hit/no-difference/confidence；答满时再增加
  每 condition 20-trial、逐歌 4-trial full view；
- 单列已回答的 high-dose、sham sampling baseline、identity/known-different/repeat 和当前 QC coverage；
- 将 objective identity、coverage collapse、operational flags 与听测 row 对齐；
- condition 未答满时写 `partial — preregistered decision pending`；只有其 20 题均在第一次 semantic
  unblind 前完成且 full QC 可判定时，才按 3.8 写 `confirmed` 或 `not confirmed`；
- 生成带 semantic filenames 的 formal generated acc MIDI/WAV export 和 index；
- 构建带 `not_started/partial/full` collection status 的 `report.md` 与完整 reproducibility index；
- 用户可在 unblind 后进行自由 A/B 复听，但必须标作 post-unblinding qualitative follow-up，不能修改
  primary blind scores。

Gate P6：report builder fail-closed 验证所有 artifacts/hash/score workflow；无论结果阳性还是阴性，只要
实验有效并如实报告即可完成。

## 7. Detailed Todo List

> 勾选规则：只有“实现或运行完成 + 自动验收通过 + durable artifact 可审计”三项同时满足才可勾。
> 任何改变 trial 数、seed、excerpt、render、QC 或判定阈值的修改，都必须发生在 P1 selection freeze 前，
> 修改本 plan/schema/tests，并生成新 campaign ID；formal output 产生后不得原地改实验语义。

### T0. Spec 和合同

- [ ] **T0.1** 将本 plan 标记为旧 plan Phase F/T10 的 superseding contract；旧 24-clip quality package
  降为 optional secondary，不满足本轮 DoD。
- [ ] **T0.2** 在 shared config 写死 primary question、acc-only/rt-theoretical source、60 medium main、
  10 high、5 sham-noise、6 identity、6 known-different、8 repeat、3 practice。
- [ ] **T0.3** 写死 8 s / 16-beat excerpt、默认 `[16,32)`、prefix-balanced global order、flexible sittings
  和六种 presentation balancing 规则。
- [ ] **T0.4** 写死 response schema、partial snapshot semantics、QC thresholds、full-sample condition decision
  rule、empty/operational policies。
- [ ] **T0.5** 选定 fixed-20 qualification，更新所有与 staged short-circuit 或“直接取 tail 24”冲突的
  plan/docs/help text。
- [ ] **T0.6** 把 dense song `2`、tail songs `[2,4]` 和 stable manifest identities 加入 shared contract。
- [ ] **T0.7** 明确 formal blind 前禁止 semantic preview，development preview 永不进入 formal selection。

### T1. Qualification hardening

- [ ] **T1.1** 实现 shared `derive_qualification_decision()`，输入只允许 pinned spec/schedule/static summary/
  immutable attempt evidence。
- [ ] **T1.2** `evaluate()` 改为调用 shared derivation，不再维护第二套判定逻辑。
- [ ] **T1.3** `validate_qualification_result()` 重新读取 20 attempt trees 并 exact-compare 全部 derived fields。
- [ ] **T1.4** `freeze()` 和 `validate_frozen_qualification()` 重算 decision 后才接受 result。
- [ ] **T1.5** 正式 qualification CLI 拒绝 `--allow-dirty`；development-only result 永不能 freeze。
- [ ] **T1.6** verifier 强制 exact 20 runs、exact order、exact selectors、20/20 `attempt-001`、零 retry。
- [ ] **T1.7** 实现 tempo/tail 停止规则；不收敛时输出 failed reason 而不是选择 24。
- [ ] **T1.8** 增加 decision forgery、wrong songs、dirty、retry/latest、schedule reorder、artifact drift 负测。

### T2. Formal RT generated-acc artifact gate

- [ ] **T2.1** 定义 offline/RT attempt required-artifact schemas，不再只信 verdict 自建 index。
- [ ] **T2.2** `theoretical_model.mid` export 异常向 run gate 传播；删除 silent-pass。
- [ ] **T2.3** 空模型输出仍写合法 empty MIDI + summary，区分 empty-success 与 export/request failure。
- [ ] **T2.4** 从 raw token/model schedule trace 独立恢复 theoretical notes，与 MIDI semantic-equal。
- [ ] **T2.5** 验证 combined Accompaniment events 对应 scheduled ticks；保留 theoretical/combined 分层。
- [ ] **T2.6** required set 记录 lifecycle、validity、full trace、runtime/checkpoint/input/reset/command/process gates。
- [ ] **T2.7** shared attempt verifier 验 actual file set、index、size、hash、schema、campaign binding。
- [ ] **T2.8** campaign audit 新增 `listening_source_readiness` 和逐 source failure reasons。
- [ ] **T2.9** 增加 missing theoretical、tampered theoretical、trace mismatch、expected empty、unexpected silence
  测试。

### T3. Listening schema v2 和 selection

- [ ] **T3.1** 新增 `listening_triangle_selection.v2` builder/validator，保留 v1 read path 但禁止其满足新 DoD。
- [ ] **T3.2** 拆分 formal pipeline/source artifact/presentation/question semantics，不再复用 overloaded pipeline。
- [ ] **T3.3** 枚举 60 medium selectors，断言每 song/condition 精确覆盖 4 个 pseed×sseed pairs。
- [ ] **T3.4** 枚举 10 high 和 5 sham-noise selectors；high 不能标作 listener QC。
- [ ] **T3.5** 预冻结 6 identity、6 known-different recipes 和 8 exact repeat sources/间距。
- [ ] **T3.6** 实现六种 presentation order 与 odd-condition balancing，duplicate 必须 literal copy。
- [ ] **T3.7** 实现 deterministic prefix/chunk-balanced global order，使任意合理停止点的
  condition/song/seed/control 覆盖不过度偏斜；sitting 边界不进入 selection 语义。
- [ ] **T3.8** selection 写入 excerpt、render、empty、operational、score、per-trial persistence、partial
  snapshot、QC、retry-attempt 和 blind seed rules。
- [ ] **T3.9** validator exact-rebuild 全 selection；删行、换 seed、换 window、换 control/repeat/order 都失败。
- [ ] **T3.10** C5 config 绑定 selection path/hash/schema；formal driver 在运行前再校验 frozen-before-output。

### T4. Triangle package、播放器和 score workflow

- [ ] **T4.1** builder 改用 shared immutable-attempt verifier，禁止较弱的 `_verified_attempt()` 分叉逻辑。
- [ ] **T4.2** 只从 `theoretical_model.mid` 生成 primary acc-solo；自动审计不含 melody/metronome。
- [ ] **T4.3** 实现 120 BPM、44.1 kHz、16-bit、固定 program/soundfont/version/gain 的 canonical render。
- [ ] **T4.4** 实现 common pair attenuation、4× true-peak audit、pre-roll/crop/fade/pad 固定规则。
- [ ] **T4.5** expected empty 允许 literal silence；non-empty→silence 失败；provenance 写 source_empty/note counts。
- [ ] **T4.6** 为每 trial 从 unique canonical WAV 复制两份 duplicate，audit duplicate SHA-256 完全相同。
- [ ] **T4.7** private key/render manifest 记录 attempt/verdict/artifact/path/hash/validity/excerpt/program/gain。
- [ ] **T4.8** public manifest 和 blind player 显示正式 acc-only triangle prompt，不泄露 semantic fields；
  每题提交后显示“已保存”，支持关闭后从下一 unanswered trial 恢复。
- [ ] **T4.9** append-only score ledger exact-validate odd/no-difference、confidence、tags、note、play count、
  response time、sitting ID、previous-record hash；独立 sitting ledger 记录 start/end、device、environment、
  anomalies 和 previous-record hash；两条 ledger 采用同一 CAS lock、原子写/恢复，不能因中断丢失已答题。
- [ ] **T4.10** snapshot/seal 接受 1–95 个已答 rows，固定 response/sitting ledger prefix/head hash 和 pending
  set；拒绝 duplicate、edited、orphan rows，但不得仅因未答满而失败。
- [ ] **T4.11** summarize 支持 blind progress 与 semantic partial/full 两种模式；输出实际分母、QC coverage、
  per-condition/per-song、objective identity、coverage-driven、repeat views 和 first-semantic-unblind boundary。
- [ ] **T4.12** synthetic end-to-end fixture 覆盖 build/audit、答 1 题后中断恢复、任意 partial snapshot、
  0/1/94/95 题状态、seal/unblind/summarize 和全部篡改负测。
- [ ] **T4.13** full QC fail 后只允许从 immutable 95-response snapshot 派生 attempt-N+1；authorization
  hash-bind 失败 package/selection/audit/render/private key、response/sitting ledgers、snapshot/unblind/summary；
  retry 仅重排盲序和 presentation，保持 semantic trial/source/excerpt/control/repeat 集合，且新 package 两条
  ledger 必须从空开始。

### T5. Clean freeze 和 fresh staging

- [ ] **T5.1** 整理并提交当前 dirty worktree 中与本 campaign 有关的 code/plan/tests/docs；保留无关用户修改。
- [ ] **T5.2** 从独立 clean worktree/固定 durable root 启动；记录 code/environment/GPU identity。
- [ ] **T5.3** 提供真实 checkpoint，记录 SHA-256；`runtime_info.has_real_model=true`。
- [ ] **T5.4** 运行 song 4/5 × tempo 15/120 gold consistency，2 skipped 必须转为 passed。
- [ ] **T5.5** fresh 生成 40 inputs/sidecars/NPZ/source-acc copies，禁止复用 `/tmp` campaign-check。
- [ ] **T5.6** 40/40 replay、note universe、latent pairing、MIDI↔NPZ、source acc hash、horizon 全绿。
- [ ] **T5.7** 仅根据 clean inputs 确认/冻结每歌 excerpt；生成 selection 并计算 SHA-256。
- [ ] **T5.8** 安装/探测 FluidSynth 和 soundfont，冻结 binary/version/file SHA/program/render contract。

### T6. Qualification 和 C5

- [ ] **T6.1** 生成 candidate config、qualification spec、20-row schedule 及各自 hash/binding。
- [ ] **T6.2** 在 clean code 上运行 exact 20 first attempts；不允许 allow-dirty 或 retry。
- [ ] **T6.3** offline/RT determinism 逐 raw token/trace/theoretical notes 完全一致。
- [ ] **T6.4** static input gate 和 RT per-request dynamic input trace gate 通过。
- [ ] **T6.5** tempo 60/30 operational evidence 派生唯一 selected tempo；失败时停止。
- [ ] **T6.6** tail 8/16/24 convergence 派生唯一 selected tail；不收敛时停止。
- [ ] **T6.7** evaluate 和 independent validator derived decision exact-equal，qualification passed。
- [ ] **T6.8** freeze C5 config，绑定 code/checkpoint/input/result/selection/soundfont hashes。

### T7. Formal generation 和 objective analysis

- [ ] **T7.1** 从 frozen contract 重建 exact 160-row schedule，assert offline 80 + RT 80。
- [ ] **T7.2** 运行 offline 80；每 run same-seed reset、content/artifact gate 和 immutable attempt 完整。
- [ ] **T7.3** 运行 RT 80；每 run reset/session epoch、required artifacts、trace/MIDI gate 完整。
- [ ] **T7.4** formal retry 按 frozen matched-block policy 执行，首 attempt 永不覆盖且所有失败保留。
- [ ] **T7.5** campaign audit 确认 160 expected、0 missing/extra/content-invalid/corrupt。
- [ ] **T7.6** audit 确认所有 frozen listening sources ready；operational-invalid/empty 单列。
- [ ] **T7.7** 生成 targeted controls；control recipe/source hashes 与 selection 完全一致。
- [ ] **T7.8** 运行 analyzer，生成 metrics/contrasts/interactions/coverage/bootstrap/controls/index。
- [ ] **T7.9** objective 结果只用于后续对齐和报告，不能据此替换 blind trial/window。

### T8. Build、audit 和用户盲听

- [ ] **T8.1** 构建 95-trial/3-practice blind WAV package；生成 prefix-balanced global order、resume state
  template 和 private key。
- [ ] **T8.2** audit 285 scored presentations 的数量、8 s 长度、PCM、true peak、hash、blinding、mapping。
- [ ] **T8.3** audit 6 identity 是 A/A/A，所有 normal duplicate 是 literal copy，8 repeats 映射正确。
- [ ] **T8.4** audit expected silence 与 non-empty render，禁止自动替换 empty/operational-invalid source。
- [ ] **T8.5** 用户只打开 blind side，完成 practice 后固定设备/音量。
- [ ] **T8.6** 验证第一题提交后 response ledger、previous hash、progress state 均落盘；关闭/重开能从
  下一 unanswered trial 继续。
- [ ] **T8.7** 用户每次可答任意题数；每个 sitting 独立记录环境，暂停不触发失败或丢数据。
- [ ] **T8.8** 任意 `n>=1` 可建立 sealed snapshot，并生成 answered/pending/QC-coverage blind summary。
- [ ] **T8.9** 用户要求查看结果时，只解盲该 snapshot 已答 rows，生成带实际分母的 semantic partial
  summary；记录 `first_semantic_unblind_at`。
- [ ] **T8.10** partial unblind 后仍允许继续收集，但新 rows 标记 post-partial-unblind，报告分层而非隐藏。
- [ ] **T8.11** 若最终答满 95，生成 full snapshot 和完整 QC；QC 失败时保留本 attempt 并按预注册规则
  调用 `derive-triangle-retry` 生成新的 selection/package，报告保留全部失败 attempt history；QC pass 或
  partial snapshot 禁止授权 retry。答满不是 partial report 的前置条件。

### T9. Result、human-audible exports 和 report

- [ ] **T9.1** 对 pitch/onset/both 按实际 answered denominator 输出 hit/no-difference/confidence；答满时
  再输出 20-trial 和逐歌 4-trial full views。
- [ ] **T9.2** 按实际 answered denominator 输出 high-dose 和 sham-noise 参照，不混入 medium primary n。
- [ ] **T9.3** 对齐 raw token/MIDI/WAV identity、coverage collapse、empty、operational flags 和听测 response。
- [ ] **T9.4** 未答满的 condition 标 `partial — preregistered decision pending`；只有满足完整题数、QC 和
  blind-boundary 条件时才按预注册规则标 `confirmed` 或 `not confirmed`。
- [ ] **T9.5** 生成 `generated_acc_index.csv` 和 unblind 后按 semantic names 导出的 MIDI/WAV。
- [ ] **T9.6** report builder 在 0/1–94/95 题分别生成 `not_started/partial/full` 状态，验证当前 snapshot
  和所有 config/schedule/result/artifact/score hashes，不因 pending rows 拒绝 partial report；0 题只接受
  真正空 ledger 且不传 snapshot，任一已有 response/snapshot/unblind state 都必须 fail closed。
- [ ] **T9.7** limitations 明确单 listener、5 songs、fixed excerpts、source dependence、无质量/偏好结论。
- [ ] **T9.8** post-unblinding 自由复听单列记录，不回写或覆盖 primary blind scores。

## 8. Definition of Done 与可交付状态

首先完成不依赖用户答题数量的 technical campaign：

1. Qualification hardening 和 RT required-artifact gate 已实现并通过正负测试；
2. code/checkpoint/environment/soundfont 来自 clean、hash-pinned、durable campaign root；
3. fresh 40-input staging 和 static/dynamic input gates 通过；
4. fixed 20-run qualification 全为 first attempt，并由 artifacts 重新推导为 passed；
5. C5 config 绑定 qualification 与 triangle selection；
6. 160 formal runs 和 campaign/listening-readiness audit 完整通过；
7. formal objective analysis、controls 和 reproducibility index 完整；
8. 95-trial acc-only 题库通过 render/blind/empty/duplicate/provenance audit；
9. blind player 的逐题原子保存、resume、snapshot、partial unblind/report 已通过中断恢复和篡改测试；
10. report builder 对所有 artifact/hash/snapshot binding fail closed。

此后 human listening 按实际数据量报告状态，不强制答满才能出结果。三个状态轴分开记录：

- `collection_status` 只表示答题数量；
- `qc_status=not_started/pending/pass/fail` 表示尚无回答、证据未完整、通过或失败；
- `blinding_status=fully_blind/partially_unblinded_during_collection` 表示是否中途看过 semantic result。

| `collection_status` | 条件 | 必须交付的结果 |
|---|---|---|
| `not_started` | 0 题 | objective report + 完整可听题库；无 human conclusion |
| `partial` | 1–94 题 | immutable snapshot + 按实际分母的 partial semantic result + pending/QC/blind-boundary 说明 |
| `full` | 95/95 | full-count summary，同时单列 QC 和 blinding status |

只有相关 condition 的 20 题全部在第一次 semantic unblind 前完成且 `qc_status=pass` 时，才可使用预注册
`confirmed/not confirmed` 判定；否则仍按实际数据给 descriptive result，不丢弃任何回答。

因此，用户今天答 3 题、下次答 7 题时，都能分别得到 `n=3`、`n=10` 的可审计结果；旧 snapshot 不会
被新数据覆盖。95 题只是完整题库上最强结论的条件，不是生成结果文件或 partial report 的门槛。

DoD 不要求得到“有差异”的阳性结果。pitch、onset、both 全部 `not confirmed` 也可以是一个完整有效的
full listening result；用户停在 partial 也必须如实交付当前结果，而不能报错或隐藏数据。缺 formal
generated acc、用输入 melody 代替 acc、修改已答 response、或把 partial/QC-pending 数据冒充 full
confirmation，才算违反本计划。
