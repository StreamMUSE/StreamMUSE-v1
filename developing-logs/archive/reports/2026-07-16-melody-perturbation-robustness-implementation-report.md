# Melody 扰动鲁棒性实验详细实现报告（2026-07-16）

## 1. 结论摘要

本轮已经把 July 2026 melody perturbation robustness plan 所需的主要实验基础设施实现出来：冻结语义、40-input 扰动与 staging、offline/RT 输入和采样可追溯性、qualification、160-run driver、分析器、听测材料生成器以及最终 report builder 都已有代码和自动测试。

当前最准确的交付结论是：

- **实验工具链已大体实现，自动测试通过。** 当前工作树实测为 407 unit tests + 5 integration tests，共 412 passed；另有 2 个真实 checkpoint consistency tests 因未提供 `LEKAI_CHECKPOINT_PATH` 而按设计跳过。
- **正式实验尚未执行。** 当前没有 clean-worktree C5 freeze、真实 RT qualification、正式 offline 80 + RT 80、160-run audit、正式分析结果、最终 24-clip WAV 包、盲听评分或科学结果报告。
- **当前实现还不能直接进入正式 qualification。** 最后一轮独立复核发现 qualification 的 fail-closed contract 仍有 1 个高优先级和若干流程级缺口，详见第 8 节。最关键的是 frozen consumer 尚未从 20 个 run artifacts 重新计算 qualification decision，仍可能接受“证据结构完整但 decision fields 被完整伪造”的 result。
- **因此不能把本报告解释为实验结论。** 目前没有证据可以回答 H_input、H_generation_interaction 或 H_operational；H_realworld 本来就不在实验回答范围内。

本报告是 **implementation report**，不是 `build_robustness_report.py` 在 formal campaign 完成后生成的 scientific result report。

## 2. 项目与实验目标

StreamMUSE 是一个实时 AI 伴奏生成系统。它读取键盘、MIDI device 或 MIDI file 的旋律输入，将增量 melody context 发送给推理后端，并实时播放或记录模型生成的伴奏。

本实验要验证的不是“真人会不会弹错”，而是一个更窄、可控的前置问题：对同一组固定旋律施加可复现的 pitch/onset 合成扰动后，伴奏生成是否发生可测变化，以及 offline 与实时增量生成管线对这些扰动的反应是否不同。

冻结的假设边界是：

- `H_input`：指定合成扰动是否改变 fixed-clean-reference quality。
- `H_generation_interaction`：扰动对 `rt_theoretical` 的影响是否不同于 offline；这里只能解释为完整 generation pipeline 的差异，不能归因到单一组件。
- `H_operational`：`rt_theoretical` 与实际调度后的 `rt_combined` 有何差异，以及 late/drop/clamp/forced note-off 等运行现象。
- `H_realworld`：真人微时序或真实演奏失误能否解释 live gap；本实验明确不回答。

这是 5-song exploratory pilot。即使未来全部完成，也只应报告逐歌 effect、等权 overall estimator 和 descriptive song-block bootstrap，不能声称统计显著性、等价性、总体人群外推或组件级因果归因。

## 3. 冻结实验设计

### 3.1 Input matrix

每首 source song 固定生成 8 个 input variants：

| condition | pitch probability | onset probability | perturb seeds |
|---|---:|---:|---:|
| sham | 0 | 0 | 1 个 `null` seed |
| pitch | 0.05 | 0 | 2 |
| onset | 0 | 0.15 | 2 |
| both | 0.05 | 0.15 | 2 |
| high | 0.20 | 0.40 | 1 |

因此精确 run 数是：

```text
(1 + 2 + 2 + 2 + 1) variants
× 5 songs
= 40 distinct inputs

40 inputs × 2 sample seeds = 80 runs / pipeline
80 offline + 80 RT = 160 formal model runs
```

冻结 seeds：

- perturb：`2026071001`, `2026071002`
- sample：`2026071101`, `2026071102`
- run order：`2026071201`
- bootstrap：`2026071301`
- blind order：`2026071401`

### 3.2 Analysis semantics

Primary contrast 是 medium `both` vs `sham`。主要质量 endpoint 是：

- `D_intended`：伴奏相对 clean melody 的不协和度，表示是否偏离原曲目标。
- coverage guardrail：onset/beat、active pitch-tick/beat、empty-beat ratio、harmonic pair coverage。

Secondary/exploratory endpoints 包括：

- `D_actual`：伴奏相对实际 dirty melody 的 joint treatment effect。
- adaptation cross-reference：`D(acc_cond, dirty) - D(acc_sham, dirty)`。
- pitch-only、onset-only、high-dose、rhythmic endpoints。
- offline vs `rt_theoretical` interaction。
- `rt_theoretical` vs `rt_combined` operational difference。

完全空伴奏不会得到看似优秀的 `D=0`，而是 `D=NA` 加 coverage failure。单边 NA 也不会被静默 complete-case 删除。

## 4. 实现后的端到端数据流

```text
5 clean melody MIDI + 5 source accompaniment MIDI
        │
        ▼
perturb_melody.py
40 melody variants + 40 sidecars + input_manifest.json
        │
        ▼
midi_to_npz.py --strict
40 NPZ + conversion_summary.json + MIDI/NPZ roll gate
        │
        ▼
qualify_perturbation_campaign.py plan
candidate config + canonical 20-row qualification schedule
        │
        ▼
run_perturbation_matrix.py run --qualification
immutable attempts + lifecycle/token/input artifacts
        │
        ▼
qualify_perturbation_campaign.py evaluate
determinism + static gate + tempo + tail result
        │
        ▼
qualify_perturbation_campaign.py freeze
C5 campaign_config.json + qualification/listening hashes
        │
        ▼
run_perturbation_matrix.py schedule/run/audit
160 formal runs + campaign_audit.json
        │
        ├──► analyze_perturbation_robustness.py
        │    metrics / QC / controls / bootstrap / plots
        │
        └──► prepare_robustness_listening.py
             24 clips / audit / sealed scores / unblinding
                    │
                    ▼
             build_robustness_report.py
             result report + reproducibility index
```

设计意图是每一层都引用同一 config、schedule、input、checkpoint 和 qualification hash；不能通过复制后重新 hash 来默许语义改变。

## 5. 各子系统的详细实现

### 5.1 Shared campaign contract

核心实现位于 `src/streammuse/experiments/melody_robustness.py`。

已经实现：

- 条件表、固定 seeds、run counts、sampling 参数、runtime 参数、validity/retry 规则、estimands、bootstrap 和 listening contract 的统一 machine-readable schema。
- `default_campaign_config()` 默认只生成 `qualification_candidate`，不再默认伪装成已通过的 frozen campaign。
- candidate 与 `qualified_frozen` 两种状态分离。
- frozen config 对所有固定语义 subtree 做 exact comparison；不能手改 probability、seed、runtime、metric 或 retry 规则后只重新计算文件 hash。
- C5 config 必须绑定 candidate config、qualification result 和 listening selection 的 path + SHA-256。
- canonical 160-row schedule 由 config + 40-input manifest 确定性重建，run ID 唯一，offline/RT 各 80 行。
- campaign binding、attempt verdict 和 artifact index 使用 canonical JSON 与 SHA-256 交叉引用。

这个 shared module 让 perturbation、driver、analyzer、listening 和 report 不再各自解释 plan。

### 5.2 Offline pipeline

Offline 侧已实现：

- `PianoDataset mode="all"`：稳定排序、不切 train/test split，避免 40 个文件静默变 38 个。
- length cache 绑定 file list、顺序和长度，不能用旧 cache 覆盖新的 manifest ordering。
- `run_lekai_offline.py` 支持按 stem/path 精确寻址，正式 driver 不再依赖会随 shuffle 改变的整数 `condition_idx`。
- 每次生成使用独立、device-compatible `torch.Generator`；seed 显式传过完整 sampling call chain，不依赖 process-global `torch.manual_seed`。
- 每 run 保存 `run_config.json` 和 token trace，包括 NPZ/source MIDI/checkpoint hash、sampling config、device/dtype、raw sampled tokens、full interleaved sequence 与输出 MIDI hash。
- offline post-run gate 验证实际选择的 NPZ path/hash、part0 encode→decode、token artifact 和完整预期 artifact set。
- 模型条件 BPM 固定为 120；playback tempo 只影响 RT wall-clock，二者不再混为一个参数。

开发期真实 checkpoint smoke evidence 位于 `/tmp/streammuse-offdet-{1,2}`：同 checkpoint、NPZ 和 seed 的两次运行均有 333 个 sampled raw tokens，完整 interleaved sequence 和 generated MIDI hash 相同。该证据来自当前开发工作树，不替代 clean-worktree qualification 归档。

### 5.3 RT lifecycle、validity 与 sampling state

RT 主链已实现：

- service 状态拆成 accepting requests、draining、stopped；producer、worker 和 response consumer 不再由单一 `_running` 混合控制。
- `analysis_end_tick`、`request_cutoff_tick`、`last_input_note_off_tick` 和 `run_stop_tick` 分离并使用冻结公式。
- 到 cutoff 后停止产生新请求，但继续消费 queued/in-flight response。
- 使用显式 sentinel/join 和 hard drain timeout；stop 幂等，timeout 会进入 invalidity，而不是静默丢请求。
- lifecycle 记录 expected/enqueued/started/succeeded/failed/processed、HTTP error、pending-at-stop、stale drop、late、clamp、forced note-off、input digest 和 raw token digest。
- content gate 与 operational gate 分离。Operational invalid 不会被静默排除；正式 audit 的完成条件是 160 个 content-valid，operational-invalid 会保留为结果。
- 成功但空 response 仍写合法空 `theoretical_model.mid` 和空 token/event summary，并标记 `empty_success=true`；它与 request failure 不再混淆。
- 每个 session 有独立 Generator、session ID 和 epoch；原子 reset 会阻止新请求、处理 boundary work、清空 history、重置 seed 并返回 effective seed/epoch。
- stale epoch request 会被拒绝并记录，旧 future 不应污染下一次 run。
- `/runtime_info` 和 per-response metadata 扩展了 checkpoint、code identity、device/dtype、sampling、BPM、history/cap、seed/epoch 等信息。

### 5.4 RT actual-input trace boundary

HTTP metadata 新增 `part0_trace_available`，用于区分：

- 真实 encoder path：必须为 `true`，并提供非空 part0 roll digest、part0 token digest 和 context start tick。
- 开发 rule-stub：只允许明确返回 `false` 加空 trace。

service 的通用 metadata contract 可以接受明确“不适用”的 stub，但 formal driver 同时要求 real model，并逐 request 重建 expected input trace。因此 formal/qualification 中缺少真实 trace、digest 不一致或伪造 availability 都会使 dynamic input gate 失败。

负向测试覆盖了 `part0_trace_available=false` 不能用来隐藏非空 trace fields。

### 5.5 Melody perturbation tool

`scripts/perturb_melody.py` 已实现冻结的 perturbation semantics：

- 使用 raw-MIDI-derived canonical note table；可扰动 universe 限定为 matched、non-drum、pitch 21–108、model-visible notes。
- `source_note_id` 绑定 source hash、track ordinal 和 original note ordinal。
- dangling note-on、spurious off、drum/out-of-range notes、CC/pedal、pitch bend、tempo/key/time-signature、track/channel metadata 按统一 policy preserve/drop。
- 参与扰动的 start/end 必须落在 PPQ/4 grid，否则 fail-fast。
- keyed decision 使用 canonical、domain-separated BLAKE2b-128，personality 为 `SMusePerturbV1`；结果不依赖 Python hash seed 或全局 RNG。
- pitch selection、pitch candidate、onset selection、onset candidate 和 collision attempts 使用分离 domain。
- medium/high 共享 selection score，自动保证 medium mask 是 high-dose mask 的子集。
- pitch candidates 为 `{-2,-1,+1,+2}`，onset candidates 为 `{-1,+1}` model tick；每个 decision 最多尝试 3 个 keyed candidates。
- collision 使用半开区间，只拒绝新引入的 same-pitch overlap 或不可序列化 note-off ownership；giveup 保持原 note 并记录原因。
- factorial pairing 冻结的是 latent proposals，而不是强求各 arm effective edits 完全相同；effective mismatch/collision interaction 会显式记录。
- 每个输出都有 sidecar，记录 original/proposed/final state、selection score、candidate order、giveup/no-op、metadata policy 和 output hash。
- replay verifier 直接应用 sidecar final edits，不重新运行 PRF/collision，并比较 canonical semantic event 和 model roll。

Fixture 覆盖 p=0/p=1、pitch 21/108、start=0、duplicate onset、native overlap、chain collision、equal-start、dangling/off-grid/drum、giveup、both pairing、medium⊂high 和跨 `PYTHONHASHSEED` 复现。

### 5.6 Staging 与 strict NPZ conversion

已经实现：

- 必须使用 fresh staging root；stale MIDI、NPZ、sidecar、orphan、extra 或 missing artifact 会 fail。
- 每个 melody variant 配套同 stem accompaniment，acc 副本 hash 必须等于 source acc。
- `midi_to_npz.py --strict` 接收 expected manifest，任一 skip/conversion error/hash mismatch 都非零退出。
- NPZ stem set 必须精确等于 40-input manifest，而不只是 `converted == 40`。
- MIDI roll 与 NPZ part0 roll 在独立 validation horizon 上逐 cell 相等。
- validation horizon、analysis horizon 与 run-stop horizon 分离，不能通过 analysis crop 掩盖转换尾部错误。

开发期 `/tmp/streammuse-perturb-campaign-check5/conversion_summary.json` 显示：40 expected、40 converted、0 skipped、0 errors、exact stem set，并且 40 个 per-input roll gate 的 `differing_cells=0`。它仍是 `/tmp` smoke artifact，没有绑定 clean code SHA、正式 manifest freeze 和 archive，因此 T4.12/T5.9 不能据此完全关闭。

### 5.7 Qualification pipeline

`scripts/qualify_perturbation_campaign.py` 已实现 `plan → evaluate → freeze` 三步。

Canonical qualification schedule 是固定 20 行：

- offline determinism：2 runs。
- RT determinism：2 runs。
- tempo：2 tempos × sham/both = 4 runs。
- tail：2 tempos × 2 songs × 3 tail candidates = 12 runs。

`evaluate` 当前会：

- 验证 candidate config 和 schedule 的 pinned SHA-256。
- 从 input manifest 重建 canonical 20-row schedule，并逐行 exact compare。
- 验证 qualification output-root campaign binding。
- 验证每个 selected attempt 的 immutable verdict、artifact exact set、size 与 hash。
- 比较 offline raw tokens/MIDI determinism。
- 比较 RT per-request raw token/input/part0 trace 和 theoretical roll determinism。
- 检查 40-input static conversion summary。
- 从 sham/both operational validity 选择 campaign-wide tempo 60 或 30。
- 对两首歌比较 tail 8/16/24 在 analysis window 内的 trace/roll convergence。
- 只有所有 gate 通过才生成 `passed=true`。

C5 只能由 qualification `freeze` 产生；`run_perturbation_matrix.py` 的 direct freeze CLI 已移除，残留函数会明确抛错。Frozen config 会记录 candidate/result path+hash，final tempo/tail 必须等于 result 的 selected values。

第 8 节所列 blocker 表明这条链还需要继续加固，尤其是让 downstream validation 从 artifacts 重新推导 decision，而不是只验证 result 自洽。

### 5.8 Formal driver 与 campaign audit

`scripts/run_perturbation_matrix.py` 已实现：

- 只允许 dedicated loopback inference server；记录 server command、PID、cwd、environment allowlist、code identity 和 checkpoint hash。
- formal run 要求 exact clean code identity，禁止 `--allow-dirty`。
- schedule 必须等于 config + input manifest 重建出的 canonical 160 rows，并保持 offline 80 后 RT 80，避免两个 GPU model process 竞争。
- 每个 run 前验证 config/schedule/input/NPZ/acc/checkpoint/runtime/reset/session epoch。
- attempt 目录不可覆盖；`attempt-001` 永久保留，后续 retry 使用新 attempt ID。
- 完整且 hash 正确的 run 可幂等 skip；不完整/corrupt run 不能伪装成成功。
- offline run 后验证 run config、token trace、NPZ part0 identity 和 artifact set。
- RT run 后验证 lifecycle、actual input trace、raw token/event artifacts 和 empty-success semantics。
- verdict 包含 content/operational validity、artifact index 和 campaign binding fields。
- analyzer 不再盲信 mutable `latest_verdict.json`，而是复用 strict immutable-attempt verifier。
- audit 要求 160 expected run IDs、无 missing、无 content-invalid、无 extra run IDs；retry 和 operational-invalid 单列。

Formal campaign binding 会传播 `qualification_result_sha256` 到 verdict、audit、analysis、listening 和 report。

### 5.9 Analyzer 与 targeted controls

`src/streammuse/experiments/robustness_metrics.py` 与 `scripts/analyze_perturbation_robustness.py` 已实现：

- 所有 MIDI 恢复到公共 4 ticks/beat、exclusive analysis window。
- canonical roll loader 覆盖 retrigger、hanging notes、tempo header 差异和空 MIDI。
- sensitivity：sustain/onset Jaccard/F1 与 symmetric onset distance；both-empty/one-empty 显式标记。
- harmonic quality：interval classes `{1,2,6,10,11}`，报告 micro pair-tick `D` 与 macro coactive-tick `D`。
- coverage：harmonic pair coverage、onset/beat、active pitch-tick/beat、empty-beat ratio。
- `D_intended`、`D_actual` 和 adaptation cross-reference 分开计算，避免把 sensitivity 当 quality。
- paired contrast 先在相同 pseed/sseed 内计算，再按冻结顺序聚合到 song；sham 不会被复制成伪独立样本。
- factorial interaction、5-song raw effects、equal-weight overall、leave-one-song-out range 和 song-block bootstrap。
- lifecycle/content/operational endpoints 和 missing-schema fail-closed policy。
- targeted controls：identity、harmonic m2、harmonic tritone、fixed rhythm shift、coverage dropout、coverage empty。
- endpoint-specific control acceptance；control 失败会让对应 endpoint 标为 assay invalid。
- 输出 machine-readable metrics、CSV、QC、control report、bootstrap config/result、plot 和 artifact index。

这些输出路径已由 synthetic fixtures 测试，但尚未在正式 160-run campaign 上生成。

### 5.10 Listening package

`scripts/prepare_robustness_listening.py` 已实现 selection、build、audit、seal 和 unblind 流程。

冻结设计恰好 24 clips：

- 5 songs × `{sham,both}` ecological clips：10。
- 对应 accompaniment-solo clips：10。
- known-bad + high-dose anchors：2。
- 字面重复 consistency trials：2。

每段 25 秒，统一从 model-tick grid 按 120 BPM 重建，避免 RT playback tempo 泄露 condition/pipeline。Gain policy 使用同 pair 固定 gain，只在 true peak 超限时保护性衰减，不做逐 clip LUFS 放大。

True peak 使用 SciPy `resample_poly`、4× polyphase reconstruction、Kaiser beta 8.6；PCM16 写回后重新测量。Audit 会从 WAV 独立重算并验证 sample rate、channels、duration、hash、silence、peak policy、duplicate trials 和 blind-key bijection。

Blind scores 必须先 seal 再 unblind。`--midi-only` 只能做开发检查，不能作为最终听测包。

### 5.11 Final report builder

`scripts/build_robustness_report.py` 已实现 fail-closed scientific report 和 `reproducibility_index.json` builder。

它会交叉验证：

- frozen campaign config 与 qualification evidence。
- formal schedule/campaign binding/audit。
- analysis index、controls 和 artifact hashes。
- listening selection、render/audit、sealed scores 和 unblinded scores。
- `qualification_result_sha256` 在各阶段的一致性。

只有 staging、160 content-valid verdict、controls、analysis、final listening package 和 score workflow 全部通过时才能标记 complete。`qualification_passed_and_bound=true` 只是前置 validator 成功后的派生字段，不应被描述成独立的外部证明。

## 6. 已关闭的主要设计/绕过问题

| 原问题 | 当前处理 |
|---|---|
| default config 可直接表现成 frozen | 默认改为 `qualification_candidate` |
| driver 可手填 tempo/tail direct freeze | CLI 入口移除，函数显式拒绝 |
| qualification schedule 可改行/重排 | 重建 exact 20-row schedule 并逐行比较 |
| formal schedule 只看行数 | 重建 exact 160-row schedule 并逐行比较 |
| qualification output 可跨 campaign 混用 | candidate/schedule/root binding + verdict binding |
| 只信 mutable `latest_verdict.json` | 同时校验 immutable attempt verdict 和 artifact index |
| result/config hash 未传播到下游 | formal binding、verdict、audit、analysis、listening、report 全链传播 |
| generic stub 缺 encoder trace | `part0_trace_available` 明确区分；formal 必须真实 trace |
| 空伴奏可被当成优质 `D=0` | `D=NA` + coverage failure |
| RT operational failure 被 retry 隐藏 | content retry 与 operational outcome 分离，首 attempt 保留 |

已有负向测试覆盖 schedule reorder/rehash、cross-campaign qualification、minimal handwritten passed result、qualification-result hash drift、artifact drift 和 metadata trace hiding。

## 7. Plan T0–T11 实际状态映射

这里严格区分“实现路径存在”与“真实 campaign 已完成”。Plan 前半部分 Phase A–G 的 checkbox 是 phase completion gates；Detailed Todo 的 `[x]` 按 2026-07-16 状态说明只代表代码/测试完成，两套 checkbox 不应混读。

| Todo | 代码/测试状态 | 真实执行状态 |
|---|---|---|
| T0 | T0.1–T0.11 contract 已实现 | spec gate 完成 |
| T1 | T1.1–T1.5 完成；T1.6 有 dev evidence | clean-worktree determinism 需重跑归档 |
| T2 | T2.1–T2.11 lifecycle/validity 完成 | 需真实 RT qualification/formal 验证 |
| T3 | T3.1–T3.7 完成 | T3.8 真实 RT determinism 未完成 |
| T4 | T4.1–T4.11 完成 | T4.12 clean/frozen 五首验收未完成 |
| T5 | T5.1–T5.8 完成 | T5.9 正式 static + 每 RT run dynamic gate 未完成 |
| T6 | T6.1–T6.6 driver/audit 路径完成 | frozen schedule 与真实 run 未执行 |
| T7 | qualification tooling 存在；T7.1 tests 完成 | clean commit、gold、qualification、C5、预算均未完成 |
| T8 | runner/audit/control generator 已实现 | T8.1–T8.5 的 160-run campaign 全未执行 |
| T9 | metrics/analyzer/output generator 已实现 | 正式 metrics/control acceptance/artifacts 未生成 |
| T10 | selection/build/audit/seal/unblind 工具已实现；T10.2–T10.4 为 spec/code 完成 | 真实 selection、WAV、blind scores、seal/unblind 未完成 |
| T11 | fail-closed report builder 已实现 | T11.1–T11.7 的科学报告、归档和 DoD 未完成 |

T9.9/T9.10/T9.12 和 T10.3 的文字同时混入“代码能力”与“真实 artifact 完成”，与 plan 顶部 `[x]` 的实现状态说明存在语义张力。正式维护时最好把这些条目拆成 implementation checkbox 和 execution/artifact checkbox。

## 8. 正式运行前必须处理的剩余问题

### 8.1 P0：Qualification decisions 没有从 evidence 重新推导

`evaluate()` 会正确读取 artifacts 并计算：

- offline/RT determinism。
- static input gate。
- tempo checks/selection。
- tail equality/selection。

但是 `validate_qualification_result()` 当前只做：

- result schema/type 检查。
- result 内部 decision 自洽检查。
- candidate/schedule/binding path+hash 检查。
- 20 个 selected verdict 与 artifact index/hash 检查。

它没有从这些 artifacts 再计算上述四组 decisions，也没有重新运行 `_static_gate_errors()`。因此，一个包含真实完整 20-run evidence graph 的人仍可手写一份字段完整、内部自洽但结论为假的 `passed=true` result，再交给 `freeze()`。

建议修复：

1. 把 `evaluate()` 中 artifact → qualification decision 的逻辑抽成共享纯函数。
2. `evaluate()` 用它生成 result。
3. `validate_qualification_result()`、`freeze()` 和所有 formal consumers 用同一函数重算并 exact-compare。
4. 新增“完整 evidence graph + forged decision fields”负向测试，而不仅是 minimal result rejection。

在此修复前，不应执行或接受 C5 freeze。

### 8.2 P1：Qualification 仍允许 dirty worktree

Formal run 禁止 `--allow-dirty`，但 `run --qualification --allow-dirty` 目前仍被允许。Candidate 只记录声明的 `code_identity`；dirty diff 本身没有 attestation，所以 dirty qualification 可能被之后的 clean formal 接受。

建议：正式 qualification 禁止 `--allow-dirty`。如保留开发 smoke mode，应在 binding/result 中标记 `development_only=true`，并让 `freeze()` 无条件拒绝。

### 8.3 P1：Plan 指定 qualification songs 未成为共享常量

文档命令使用 dense song 2、tail songs 2 和 4，但 validator 只要求 dense song 存在、tail songs 是两个不同的现有 song。换成其他歌曲仍会生成合法 candidate。

建议把 stable song IDs/order 写入 shared contract 并 exact-validate，或新增独立 qualification-spec artifact 并冻结其 hash。

### 8.4 P1：Qualification 可以使用 retry 后的 latest attempt

Qualification schedule 写有 `expected_attempt_id="attempt-001"`，但 verifier 接受任意 `attempt-NNN` 的 latest verdict，也不审计之前失败的 attempt。这样 attempt-001 失败后可用 attempt-002 通过，违反 T7.7“qualification failure 回 Phase A/B”的纪律。

建议 qualification 强制 20 个 run 全部使用 attempt-001，且 run root 中不得存在 retry；formal campaign 的固定 content retry policy 可继续单独保留。

### 8.5 P1：声明的 decision order 不等于执行时 short-circuit

Config 冻结了 `determinism → static_input_gate → tempo → tail`，但 canonical runner 一次执行全部 20 rows，static summary 在 `evaluate` 阶段才检查。当前代码保证“失败不能 freeze”，不保证“早期 gate 失败就不继续花费后续 run”。

同时，代码预跑 tempo 60/30 和两种 tempo 下的全部 tail candidates，而 plan 正文读起来像 tempo 60 失败后才 fallback 30、只在最终 tempo 下测 tail。

需要在正式运行前选择并统一：

- 保持固定 20-row pre-run 设计，并更新 plan、预算和 decision-order wording；或
- 把 qualification 拆成阶段式 schedule，前一 gate 通过后才生成下一阶段。

### 8.6 P2：Archive relocation 与长期可验证性

Qualification evidence 使用绝对 path，result/reproducibility index 通过 path+hash 间接引用 schedule、binding、static summary 和 20 attempt trees。移动或删除 qualification root 后，validator 会失效。

SHA-256 能证明内容完整性与交叉关联，但不提供数字签名、可信身份或可信时间顺序。

建议最终 archive 使用不可移动的 campaign root，或实现相对路径 + archive-root remapping，并把 qualification tree 独立纳入 reproducibility artifact tree。

## 9. 自动测试与验证证据

### 9.1 当前实测结果

2026-07-16 在当前工作树执行：

```bash
uv run pytest -q tests/unit
# 407 passed, 1 warning

uv run pytest -q tests/integration
# 5 passed, 1 warning

uv run pytest -q -rs tests/consistency
# 2 skipped
# reason: LEKAI_CHECKPOINT_PATH was not provided
```

汇总：

```text
412 passed
2 skipped
1 unique warning category
```

Warning 来自 `pretty_midi` 依赖导入 deprecated `pkg_resources`，不是本实验测试失败。

此前同一最终代码路径还通过：

```bash
uv run python -m compileall -q src scripts tests
git diff --check
```

### 9.2 测试覆盖重点

自动测试覆盖：

- dataset all-mode、stable ordering/cache、stem/path selection。
- per-run Generator、offline token determinism contract。
- RT drain、timeout、empty response、HTTP error、stop reentry、pending future boundary。
- session reset/epoch/stale request/runtime metadata。
- `part0_trace_available` 的真实/stub 边界与负向伪装。
- perturbation PRF、collision、dose nesting、sidecar replay、hash-seed independence。
- strict staging、exact 40 stems、MIDI/NPZ roll equality。
- canonical 20-row qualification 与 160-row formal schedules。
- reordered/rehashed schedule、cross-campaign binding、minimal forged result、hash drift rejection。
- immutable verdict/artifact exact set、attempt idempotency、audit completeness。
- sensitivity、quality、coverage、NA、bootstrap、factorial 和 controls。
- exact 24-clip listening selection、true peak、blind-key、seal/unblind。
- final report binding 与 incomplete-package fail-closed。

未覆盖并且需要新增的重点正是第 8 节：完整 result decision forgery、dirty qualification、qualification retry/latest-attempt 和 fixed song-selector enforcement。

## 10. 当前尚不存在的正式 artifacts

仓库当前没有可作为正式 campaign 交付物的以下内容：

- clean/frozen code commit attestation。
- archived real-checkpoint gold regression。
- frozen 40-input manifest 与验收报告。
- passed `qualification_result.json`。
- final `campaign_config.json` 与 SHA-256。
- canonical formal `run_manifest.jsonl`。
- offline 80 + RT 80 attempt trees。
- 160-run `campaign_audit.json`。
- formal metrics、controls、bootstrap 和 plots。
- final soundfont-bound 24-clip WAV package。
- sealed blind scores 和 unblinded results。
- scientific `report.md` 与 final `reproducibility_index.json`。

`/tmp` evidence 只用于说明实现路径确实运行过，不是长期 archive，也不能支持研究结论。

## 11. 建议的后续执行顺序

在不查看任何 treatment-level formal result 的前提下，建议按以下顺序继续：

1. 修复第 8.1–8.5 节并补负向测试。
2. 统一 plan 中 fixed-20-row vs staged qualification、预算与 checkbox 语义。
3. 提交全部 plan/code/tests/docs，确认 worktree clean，记录新的 code SHA。
4. 设置真实 `LEKAI_CHECKPOINT_PATH`，运行 song 4/5 × tempo 15/120 gold consistency。
5. 从 clean worktree 重新生成并归档 40-input staging；不要复用 `/tmp` 作为正式 artifact root。
6. 在看 formal output 前冻结 exact 24-clip listening selection。
7. 运行 canonical qualification；20 个 run 必须 first-attempt、content/artifact/determinism/static/tempo/tail gate 全绿。
8. 用唯一合法 freeze path 生成 C5 config，并保存 config/result/listening hashes。
9. 生成 canonical 160-row schedule，执行 offline 80 和 RT 80。
10. 运行 campaign audit；确认 160 content-valid、无 missing/extra/corrupt。Operational-invalid 保留并报告。
11. 生成 targeted controls 和 formal analysis；任何 endpoint control failure 都使对应 assay invalid。
12. 构建并 audit 最终 WAV 包，完成 blind scoring、seal、unblind。
13. 运行 final report builder，检查 reproducibility index 后才允许标记 complete。

完整命令模板见 `docs/developer-guide/melody-perturbation-robustness.md`。

## 12. 主要实现文件

Shared contract 与指标：

- `src/streammuse/experiments/melody_robustness.py`
- `src/streammuse/experiments/robustness_metrics.py`

Perturbation/staging：

- `scripts/perturb_melody.py`
- `scripts/midi_to_npz.py`

Qualification、formal、analysis、listening、report：

- `scripts/qualify_perturbation_campaign.py`
- `scripts/run_perturbation_matrix.py`
- `scripts/analyze_perturbation_robustness.py`
- `scripts/prepare_robustness_listening.py`
- `scripts/build_robustness_report.py`

Offline/RT model execution：

- `scripts/run_lekai_offline.py`
- `src/streammuse/application/services/real_time_music_service.py`
- `src/streammuse/infrastructure/inference/lekai_http_backend.py`
- `src/streammuse/infrastructure/inference/server_lekai.py`
- `src/streammuse/infrastructure/inference/lekai_model/PianoDataset.py`
- `src/streammuse/infrastructure/inference/lekai_model/generation_utils.py`
- `src/streammuse/infrastructure/inference/lekai_model/model.py`
- `src/streammuse/infrastructure/output/session_logger.py`

主要测试：

- `tests/unit/experiments/test_melody_robustness.py`
- `tests/unit/experiments/test_robustness_metrics.py`
- `tests/unit/scripts/test_perturb_melody.py`
- `tests/unit/scripts/test_midi_to_npz.py`
- `tests/unit/scripts/test_qualify_perturbation_campaign.py`
- `tests/unit/scripts/test_run_perturbation_matrix.py`
- `tests/unit/scripts/test_robustness_analysis_report.py`
- `tests/unit/scripts/test_prepare_robustness_listening.py`
- `tests/unit/application/test_real_time_music_service.py`
- `tests/unit/infrastructure/inference/test_lekai_http_backend.py`
- `tests/integration/test_lekai_session_contract.py`

## 13. 最终交付判断

当前状态可以概括为：

```text
Experiment specification:       implemented
Core tooling:                   implemented
Synthetic/unit verification:    passed
Development smoke evidence:     available but non-frozen
Qualification hardening:        incomplete; formal blocker remains
Clean code/checkpoint freeze:    not done
Real qualification:             not done
Formal 160-run campaign:         not done
Listening study:                not done
Scientific conclusions:         none yet
```

所以，本轮工作已经把原先容易发生 dataset selection、RNG、RT drain、input trace、artifact drift、NA/coverage、listening leakage 等问题的实验流程，改造成了一条大体可审计的 pipeline；但在修复 qualification re-derivation 与 clean/first-attempt contract 之前，仍不应启动或接受正式 C5 campaign。
