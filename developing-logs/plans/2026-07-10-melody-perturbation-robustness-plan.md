# Melody 扰动鲁棒性实验（"人弹不稳"假设检验）— 计划（2026-07-10，rev3）

> rev2：第一轮评审 12 条采纳（factorial、seed 分离、theoretical/combined 分离、sensitivity/quality 分离等）。
> rev3（第二轮评审 12 条全采纳）：① 冻结顺序改为"先实现全部→提交→clean worktree→冻结 SHA"；② offline 加 `mode=all`（修 PianoDataset 40→38 静默丢文件，已核实）；③ RT graceful drain + tail 收敛验证（8/16/24）；④ 结构化 validity lifecycle 日志；⑤ sample seed 用 per-session `torch.Generator` + 原子 reset；⑥ **删除 IQR 等价判定，本轮定性 exploratory pilot**；⑦ 空伴奏 quality guardrail + D_actual/D_intended 命名 + known-bad output control；⑧ keyed RNG 实现真配对 factorial；⑨ note universe/collision/sidecar 验收重定义；⑩ campaign 级 tempo qualification + 统一 listening render BPM；⑪ Phase 0 拆解循环依赖；⑫ 听测子集预冻结。

## 假设（不变，rev2 拆分）

- **H_input**：指定合成扰动降低输出质量；
- **H_generation_interaction**：该扰动对 `rt_theoretical` generation pipeline 的影响强于 offline；只比较 offline vs RT theoretical，范围 = 增量逐拍生成 + 自回馈 + context 窗口等完整 generation pipeline，**不含 playback scheduling，也不单独归因**于其中某一项；
- **H_operational**：`rt_theoretical` vs `rt_combined` 的差异、late/drop/clamp/forced note-off 与 tempo qualification 单列，不与 generation interaction 混在一起；
- **H_realworld**（本实验不回答）：真实人类失误足以解释 live gap。

措辞纪律：只写"当前合成扰动的敏感性/效应量"；"未检测到效果"≠"无效应"。**本轮正式定性 exploratory pilot**（5 首歌 = 5 个独立 cluster），报告逐歌配对效应 + 描述性区间，**不做正式 equivalence 判定**（rev3 ⑥）。

## 冻结实验契约（T0 已关闭）

唯一 machine-readable contract 是 `src/streammuse/experiments/melody_robustness.py` 生成并校验的 `campaign_config.json`；staging、driver、analyzer、listening/report 工具必须引用同一 config hash。

### 条件、run 数与 seed

| condition | pitch p | onset p | pseed 数 |
|---|---:|---:|---:|
| sham | 0 | 0 | 1（`null`） |
| pitch | 0.05 | 0 | 2 |
| onset | 0 | 0.15 | 2 |
| both | 0.05 | 0.15 | 2 |
| high | 0.20 | 0.40 | 1 |

- 具体 seeds：`perturb=(2026071001,2026071002)`；`sample=(2026071101,2026071102)`；high 使用 `2026071001`；`run_order=2026071201`；`bootstrap=2026071301`；`blind_order=2026071401`。
- 5 songs × 8 input variants = 40 distinct inputs；40 × 2 sample seeds = 80 runs/pipeline；offline + RT = 160 model runs。known-bad controls 只从冻结输出派生，不计入 160。
- note selection = canonical、domain-separated BLAKE2b-128 PRF keyed Bernoulli（`score < p`，personalization=`SMusePerturbV1`）；分母固定为 matched、non-drum、pitch 21–108、model-visible canonical notes；同时报告 target/selected/proposed/effective/giveup/model-visible rate。medium/high 复用 selection score，保证 medium mask ⊂ high mask。

### Collision、非模型事件与 factorial pairing

- canonical notes 按 `(start_tick, track_index, track_note_ordinal)` 遍历；音符区间为半开 `[start,end)`；pitch candidates 为 keyed order 的 `{-2,-1,+1,+2}`，onset candidates 为独立 keyed order 的 `{-1,+1}`，最多 3 次 candidate attempt；giveup 保持原音并记录。
- 只拒绝候选新引入的 same-pitch overlap。原生 overlap 本身允许；若编辑会造成 raw MIDI note-off ownership 不可序列化，则拒绝该 candidate。start=0 的 -1 onset clamp 为 0，作为 boundary no-op 记录。
- 冻结 invariant 是 **latent proposal pairing**，不是 effective exact union：各 arm 独立 collision feasibility；effective mismatch 的 note IDs/count 和 `collision_interaction` 必须报告，factorial 解释同时披露该 mismatch。
- dangling note-on 与 spurious/zero-duration off pair 在所有 arms 一致 drop（它们不进入 `MidiConverter` model roll，禁止 onset edit 后意外配成可见 note）；matched drum/out-of-range spans、CC/pedal/pitch-bend、tempo/key/time-signature/track-name/channel 全部 preserve。

### Estimand、NA 与冻结顺序

- primary quality estimand = 固定 clean melody reference 的 `D_intended` + coverage guardrail；`D_actual(cond)-D_actual(sham)` 只叫 joint treatment effect。adaptation 必须用交叉参照：`D(acc_cond,dirty)-D(acc_sham,dirty)`。
- 完全空伴奏：D=NA + coverage failure；部分空：只报 conditional D 并同时报 coverage；cond/sham 单边 NA：paired contrast=NA 并保留 NA pattern；song 聚合和 bootstrap 不允许静默 complete-case 删除。bootstrap 单位固定为完整 song block。
- Phase B 只冻结 clean code SHA、checkpoint hash、qualification rules。先做 determinism/static input/tempo/tail qualification；C5 才写入最终 tempo/tail、listening selection hash 并冻结 `campaign_config.json`。qualification 导致代码变化必须回 Phase A/B。
- exact 24-clip listening selection manifest 必须在 formal output 和 objective analysis 之前生成、提交并 hash；选段不得看结果后调整。

## Phase A：实现全部代码改动（冻结之前，rev3 ①）

> 冻结顺序修正：**实现全部 → 测试绿 → 提交 plan+代码 → 从 clean worktree 启动 → 冻结 SHA/checkpoint/配置 → qualification → formal**。当前工作树的 bar-token 修复等未提交改动一并在此阶段收尾提交。

### A1. Offline 侧

- [ ] `run_lekai_offline.py` / `PianoDataset` 加 **`mode="all"`**：文件列表**排序**、**不切 train/test split**、按 manifest stem 精确寻址（已核实：现 `mode="train"` 对 40 个文件会静默切掉 2 个，且 condition_idx 依赖 seed shuffle，文件集合一变映射就变）。每 run 断言实际 NPZ stem + sha256 + 输出目录与 manifest 一致。
- [ ] `--seed`：offline 每首生成前重置采样 RNG（与 A3 同一套 Generator 机制）。

### A2. RT 侧：graceful drain + validity 日志

- [ ] **graceful drain**（已核实：现 max_ticks 到点直接 `_running=False`，worker 丢弃 pending 请求）：到 `run_stop_tick` 后停止**产生**新请求，等待 queued/in-flight response 全部消费再关停；明确 max_ticks 语义（exclusive/inclusive）写进文档。
- [ ] **三个 tick 概念分开配置/记录**：`analysis_end_tick`（分析窗口，= clean melody 预固定 horizon）、`last_input_note_off_tick`、`run_stop_tick`（= 尾部余量之后）。
- [ ] **结构化 validity lifecycle**（rev3 ④，现日志只记成功推理，"failed=0"可能是"失败从未被计入"）：每 session 写 `validity.json`/lifecycle JSONL——expected/enqueued/started/succeeded/failed 的 generation ticks、stale drops、HTTP 异常、pending-at-stop、analysis 窗口覆盖率。run 有效判据全部从这里读。
- [ ] **空伴奏也要有 theoretical 产物**：现 session_logger 无事件时不写 `theoretical_model.mid`——改为总是写合法空 MIDI，或 validity 里显式区分"模型成功返回空"vs"请求未完成"。

冻结 tick 公式：`analysis_end_tick` 是 exclusive；最后允许的 generation start 为
`floor((analysis_end_tick-1)/generation_interval_ticks)*generation_interval_ticks`
（inclusive，且严格小于 analysis end）；`run_stop_tick = max(last_input_note_off_tick,
request_cutoff_tick) + tail_beats*4`。

### A3. Sample seed：per-session Generator + 原子 reset（rev3 ⑤）

- [ ] 不用全局 `torch.manual_seed`（异步/遗留 future 可能先消耗新 seed）：每 session 独立 `torch.Generator`，显式传给 `torch.multinomial`（`generation_utils.sample_token` 加 generator 参数）；
- [ ] server 加 `POST /debug/reset_session`，**原子**执行：drain pending → clear history → reset generator(seed) → ack；同 server 禁止并行实验 session；validity 里记录**实际生效**的 seed；
- [ ] 确定性测试 ×2：同输入+同 sample seed，offline 连跑两次逐 token 一致；RT theoretical 连跑两次逐 token 一致。
- [ ] fallback（若 endpoint 方案受阻）：每 run 以 `LEKAI_RT_SEED` 重启 server——预算加 ~80 次加载 ≈ +40min，需重估。

### A4. 扰动工具 `scripts/perturb_melody.py`（rev3 ⑧⑨ 修订版）

**Note universe（显式定义）**：与 `MidiConverter.midi_to_notes()` 一致的 **matched、非鼓、pitch 21–108、model-visible** notes；每个 note 带稳定 `source_note_id`；unmatched note-on（五首现存 1/8/6/5/1 个悬空）/越界/鼓的处置策略写入 manifest；**roll 只作 postcondition，不从 roll 反解 notes**。

**Keyed RNG（latent 配对 factorial）**：每个决定用 canonical、domain-separated BLAKE2b-128 PRF（包含 schema、source hash、perturb seed、note id、decision/attempt）——保证 both 与单因素 arm 的 selection/candidate **latent proposals 相同**；collision feasibility 各 arm 独立，所以 effective edit 可不同，必须报告 mismatch/collision interaction 后才解释 `D_both − D_pitch − D_onset + D_sham`。

**扰动定义**（同 rev2，两处修订）：

- pitch：{±1,±2} 半音，合法域 21–108，越界**重抽**；log `requested/effective offset`；
- onset：±1 model tick（`step = PPQ//4`，断言整除，不硬编码），`effective_delta = max(requested, -start)`，on/off 同移，duration 不变，boundary no-op 入 log；
- **碰撞策略改为重抽**（rev3 ⑨）：按冻结的 candidate order 最多尝试 3 个合法候选；只拒绝新引入的 same-pitch overlap 或不可序列化的 raw note-off ownership。全部失败 → `resample_giveup` 并保持原样，不 trim/drop；
- sidecar：每输出一个 `<stem>.perturbation.json` + campaign manifest；**验收改为"重放校验"**（rev3 ⑨：rev2 的"sidecar 编辑数==roll diff 数"量纲不成立——一次 note edit 改变多个 cell）：`source MIDI + sidecar 重放 → 精确重建输出 MIDI 与模型域 roll`；note edits / onset-cell diff / sustain-cell diff 三个量分开报告。

**工具测试**：sham 模型域 identity（全 5 首）、同 seed 字节级可复现、p=1 fixture 精确 offset、21–108 恒成立、duration≥1、负 onset 不缩短、重放校验通过、latent proposal 配对与 effective mismatch 显式报告、N 输出 N sidecar、tempo/PPQ/拍号/program 保留。

### A5. Driver `scripts/run_perturbation_matrix.py` + 分析脚本（入库）

- [ ] 冻结配置表写死；per-run 调 `/debug/reset_session`；validity 全绿才计入；`/runtime_info` 前置断言 `has_real_model=true` + checkpoint hash；
- [ ] `midi_to_npz.py` 加 `--strict` + summary JSON：断言 **exactly 40 个 stem 全 converted、0 skipped**（同名 acc 副本 staging 照 rev2）。

### 冻结配置表（rev3 修订三处）

```
generation_interval_ticks = 4
generation_length_frames  = 4
model_condition_bpm       = 120     # server LEKAI_DEFAULT_BPM + offline --bpm
playback_tempo            = 60      # campaign 级 qualification 决定, 不做 per-run fallback
temperature=0.8  top_k=50  top_p=0.95  repetition_penalty=1.2
gt_prefix_beats           = 0
prompt_context_beats      = 128     # rev3: 对齐新默认与 server history retention;
                                    # 32 拍窗口已证实会使 river_flows 类歌曲从 beat 36 分叉
tail_beats                = 24      # rev3: 用金标准已验证的 24; qualification 中做 8/16/24 收敛实验,
                                    # 收敛才允许缩短
count_in_beats            = 0
artifact_tier             = debug
inference_log_detail      = full
listening_render_bpm      = 120     # rev3: 听测统一重定 tempo 渲染 —— offline MIDI 是 120、RT session
                                    # 是 60, 直接渲染会慢一倍并泄露 pipeline
```

## Phase B：提交与冻结（rev3 ①）

- [ ] B1. Phase A 全部测试绿 → **提交** plan + 全部实验代码（含现挂在工作树的 bar-token 修复、Token2Midi、model.py、midi_file.py、consistency tests 改动）；
- [ ] B2. 金标准一致性测试（songs 4,5 × tempo 15,120）在提交后的代码上全绿；
- [ ] B3. **从独立 clean worktree** 启动 qualification；此时只冻结并记录 git SHA、checkpoint sha256、qualification rules、GPU/dtype，最终 tempo/tail 尚不冻结。

## Phase C：Qualification（正式跑的资格赛）

- [ ] C1. **tempo qualification 用最稠密的 song 2**（不是 song 4——song 2 在 tempo 120 曾产生 36 drops）：tempo 60 下 sham+both 各一遍，validity 必须零 drop/late。失败 → **整个 campaign 统一降 30 并重估预算**（不做 per-run fallback，避免条件内混 tempo + 事后选择，rev3 ⑩）；
- [ ] C2. tail 收敛实验：song 2 + song 4，tail 8/16/24 三档各跑一遍，确认 analysis 窗口内输出一致 → 决定最终 tail；
- [ ] C3. 确定性测试（A3 的两条）通过；
- [ ] C4. Phase 0 静态部分：全部 40 个输入，**扰动 MIDI roll == NPZ part0 roll**（driver 前可做，rev3 ⑪ 拆解循环）；in-run 部分（expected roll == 实际 request/history roll）挪进 driver 的 per-run 校验，且**逐请求验证**而非依赖 session 末的 history（server history 有 retention 上限，不可无条件信任）。
- [ ] C5. C1–C4 全绿后，写入最终 tempo/tail、全部 seed/condition/window/validity/metric rules 与预冻结 listening selection hash，生成 canonical `campaign_config.json` + sha256；Phase D 只接受该 hash。

## Phase D：正式 pilot（160 run）

矩阵不变（sham/pitch/onset/both/high-dose × 2 pseed[单因素与 both] × 2 sseed 配对 × 5 歌 × 2 pipeline = 80/pipeline）。预算：offline ≈20min；RT tempo 60 ≈2.4h（graceful drain 每 run 加几秒，忽略）。

**命名修正（rev3 ⑦）**：20%/40% 档改称 **high-dose condition**（操纵检查 manipulation check），不再叫 positive control——它验证"重扰动可测"，但不保证一定产生退化。真正的**分析管线阳性对照**另加：**known-bad output control**——对若干 sham 输出做"伴奏整体 +1 半音移调"和"bar shuffle"两种合成破坏，分析指标必须能把它们判为显著更差；判不出 → 指标失效，全部质量结论作废。

## Phase E：分析（rev3 ⑥⑦ 重写）

### 指标

- **Sensitivity**（vs 同 sseed 的 sham）：tick 级 sustain/onset-roll Jaccard/F1、onset timing distance；空对空单独标记不计入"稳定"。
- **Quality**：
  - **D_actual**（vs 实际扰动旋律：模型有没有跟随输入）与 **D_intended**（vs 原 clean 旋律：有没有偏离原曲）**分开命名分开报**；
  - 不协和度精确定义：不协和 interval class = 半音距 {1,2,6,10,11}；sustain 对齐、duration 加权；分子 = 不协和 (acc_pitch, mel_pitch, tick) 加权计数，分母 = 全部有效 pair 加权计数；melody 静音 tick 不入分母；多声部逐 pair 计入；
  - **空伴奏 guardrail**（rev3 ⑦）：伴奏空 → D 记 **NA/failure**，绝不记 0%；报告 coverage（onset 数/拍、active tick 数/拍、empty-beat ratio）为共同主指标；**dissonance 降低但 coverage 同时塌缩 → 禁止称"改善"**；
  - **onset-only 的质量结论降级为 exploratory**（primary 是和声指标，测不了节奏问题）；rhythmic endpoint（onset 对齐网格偏差、与 melody 的 IOI 相关）列为 secondary；
  - 聚合顺序：每 run 先算 → 按歌等权 → 跨歌汇总；**禁止把所有 tick pool 在一起**。

### 统计（exploratory pilot 框架，删除 rev2 的 IQR 等价判定）

```
d_off = D(offline, cond) − D(offline, sham)     # 同 (song, sseed) 配对
d_rt  = D(rt_theoretical, cond) − D(rt_theoretical, sham)
interaction = d_rt − d_off                       # D 越高越差; 正值 = RT 额外退化
factorial 交互 = D_both − D_pitch − D_onset + D_sham
```

- **primary contrast = medium both vs sham**；pitch-only/onset-only = secondary；high-dose = manipulation check；
- bootstrap **只按完整 song block 重采样**（5 个 cluster；160 run 不是 160 个独立样本），区间标注 descriptive；
- sham 的 2 个 sseed 差值仅作为"采样噪声的量级参考"报告，**不做等价边界**（2 个值撑不起 IQR；事后补 seed 移边界的做法作废）；
- 结论模板："在 5 首 pilot 歌上，primary contrast 的逐歌配对效应为 …（区间 …）；不做正式显著性/等价判定。"

## Phase F：听测（rev3 ⑩⑫ 重写；**总时长控制在 ~30 分钟**，已确认）

**30 分钟预算倒推的子集**（看任何结果之前冻结）：

| 块 | 内容 | clips | 时长 |
|---|---|---|---|
| 主对比（ecological） | 5 首 × {sham, both}，实际 melody+acc，RT 管线，1 pseed × 1 sseed | 10 | 每段 **25s** 固定片段 |
| 主对比（acc-solo） | 同上 10 段的伴奏独奏版（只评 standalone coherence，不评 harmonic fit） | 10 | 25s |
| anchor | 1 个 known-bad（移调破坏）+ 1 个 high-dose | 2 | 25s |
| 重复 trial（一致性自检） | 从上面随机重复 | 2 | 25s |

合计 **24 clips ≈ 10 分钟纯音频**；按"听一遍→打分→个别重听"节奏 ≈ **30 分钟一场**。放弃的部分（预冻结记录在案，不算事后删减）：offline 管线的听测（只用客观指标覆盖）、clean-remix 版（反向偏差大、优先级最低）、pitch-only/onset-only（客观指标覆盖，听测只保 primary contrast）。若 30 分钟后仍有余力，备选块（offline 版 10 clips）作为可选加时，单独标注。

- **统一 `listening_render_bpm=120` 重定 tempo 后渲染 WAV**（否则 RT 按 session tempo 60 慢一倍直接泄露 pipeline）；响度统一、随机盲标、key 文件评后解盲；
- 评分维度精简为两个（30 分钟内评 4 个维度×24 clips 不现实）：**overall quality（1–5）** + **明显瑕疵备注（自由文本）**；
- 听评人 = 用户本人：先盲听 → 看客观数据 → 如需二轮，**标注 post-unblinding qualitative follow-up，不与盲听合并**；单听者结果 = exploratory qualitative judgement。

## 执行 Todo（rev3 重排）

- [ ] A. Phase A 全部（A1 offline mode=all+seed / A2 drain+validity+空产物 / A3 Generator+reset+确定性测试 / A4 扰动工具+测试 / A5 driver+strict 转换）。
- [ ] B. 提交全部 → 金标准 4,5 全绿 → clean worktree → 冻结 SHA/hash/配置。
- [ ] C. Qualification：song2 tempo 资格赛 → tail 8/16/24 收敛 → 确定性 ×2 → 40 输入静态三方一致性。
- [ ] D. 正式 160 run（RT ≈2.4h 后台）+ known-bad output control 生成。
- [ ] E. 分析（pilot 框架、guardrail、song-block bootstrap、factorial 交互）。
- [ ] F. 听测子集冻结 → 材料包（统一 tempo 渲染、盲标）→ 用户盲听 → 解盲补录 report。
- [ ] G. report：H_input / H_generation_interaction / H_operational 分节，H_realworld 标注未回答。

## 决策（已确认）

1. playback_tempo=60（song2 资格赛不过则**全 campaign** 降 30）✅
2. high-dose = pitch 20%/onset 40%（更名为 manipulation check；另加 known-bad output control 验证指标）✅
3. sample_seed=2、160 run ✅（等价判定已删除，2 seeds 仅作噪声量级参考——rev3 ⑥）
4. 听测由用户本人完成，先盲听后看数据 ✅（子集预冻结，**总时长 ~30 分钟**：24 clips × 25s，评分只留 overall+备注两项；offline 版听测砍掉、由客观指标覆盖，备选加时块单独标注）

## Detailed Todo List（执行级）

> 执行纪律：每项只有在“实现/文档完成 + 自动验收通过 + artifact 可审计”后才能勾选。任何会改变实验语义的调整，都必须先修改本 plan 并重新冻结；不得在看过 formal treatment 结果后修改 condition、seed、窗口、指标、选材或排除规则。
>
> 2026-07-16 实现状态说明：下列 `[x]` 只表示对应代码路径及其定向自动测试已经落地，不代表 Phase B 提交/clean-worktree freeze、qualification、formal 160-run、最终听测包或用户盲听已经完成。凡是同时要求真实 campaign artifact 或人工步骤的条目，即使工具已实现，也继续保持 `[ ]`。

### T0. Spec gate：先关闭剩余设计歧义

- [x] **T0.1 重排冻结时点**：Phase B 只冻结 code SHA、checkpoint hash 和 qualification 规则；C1/C2 决定 tempo/tail 后新增 C5，生成并 sha256 冻结最终 campaign_config.json，随后才允许 Phase D。qualification 若导致代码变化，必须回到 Phase A/B 重新提交。
- [x] **T0.2 补完整 condition 表**：明确 sham=(0,0)、pitch=(0.05,0)、onset=(0,0.15)、both=(0.05,0.15)、high=(0.20,0.40)；pseed 数分别为 1/2/2/2/1。
- [x] **T0.3 锁定 run 数**：5 首歌 × 每歌 8 个 input variant = 40 distinct inputs；40 × 2 sample seeds = 80 runs/pipeline；两条 pipeline 共 160 model runs。known-bad 是派生 artifact，不计入 160。
- [x] **T0.4 冻结具体 seed**：两个 perturb/sample seed、high pseed、run order、bootstrap、blind order 已写入正文和 shared contract。
- [x] **T0.5 冻结抽样语义**：使用 keyed Bernoulli threshold；分母和全部 dose fields 已冻结。
- [x] **T0.6 关闭 factorial/collision 矛盾**：选择“latent proposals 相同、effective edits 可不同；mismatch/collision interaction 强制报告”。
- [x] **T0.7 冻结 collision 规则**：半开区间、new-overlap scope、遍历/candidate/attempt/giveup/native-overlap serialization 均已冻结。
- [x] **T0.8 拆分 RT 假设**：H_generation_interaction = offline vs rt_theoretical；H_operational = rt_theoretical vs rt_combined。
- [x] **T0.9 冻结 quality estimand**：primary = fixed-clean D_intended + coverage；adaptation 使用交叉参照。
- [x] **T0.10 冻结 NA/coverage 规则**：完全空、部分空、单边 NA、song aggregation/bootstrap 规则已冻结且 analyzer fail-closed。
- [x] **T0.11 冻结听测选材时点**：selection manifest 必须在 formal/analysis 前生成并纳入 C5 hash。

### T1. Offline：all-mode、精确寻址与 RNG

- [x] **T1.1 实现 PianoDataset mode=all**：文件稳定排序，不切 train/test split；length cache 必须验证 file list、顺序和长度，不能覆盖当前排序。
- [x] **T1.2 实现 manifest 寻址**：run_lekai_offline.py 支持 condition stem/path；实验 driver 不依赖整数 condition_idx。
- [x] **T1.3 更新 gold fixture**：删除 tests/consistency/conftest.py 中旧 shuffled index mapping，gold runner 改为 stem/path 寻址。
- [x] **T1.4 实现 offline seed**：增加 --seed；warmup 后、每 run 前创建与 logits device 匹配的 torch.Generator，并显式传过完整 sampling call chain。
- [x] **T1.5 输出 run_config**：记录 run_id、NPZ path/stem/hash、source MIDI hash、checkpoint hash、seed、sampling 参数、device/dtype、输出 hash、token trace。
- [x] **T1.6 验收**：40 个 NPZ 时 dataset 长度严格为 40，manifest 每个 stem 恰好命中一次；同 input+seed 连跑两次 raw tokens 一致。development 实测证据为 `/tmp/streammuse-offdet-{1,2}`：同 checkpoint/NPZ/seed 的 333 个 sampled raw tokens、完整 interleaved sequence 和 generated MIDI hash 均一致；正式 qualification 仍须按 T7.4 从 clean worktree 归档。

### T2. RT 生命周期：request cutoff、drain 与双轴 validity

- [x] **T2.1 拆开状态机**：至少实现 accepting_requests、draining、stopped；不能继续用一个 _running 同时控制 producer、worker 和 response consumer。
- [x] **T2.2 增加 request_cutoff_tick**：最后一个可能影响 analysis window 的请求发出后停止 enqueue；run_stop_tick 只给 pending response 和 note-off 留消费时间。
- [x] **T2.3 写死 tick 公式**：analysis_end_tick 为 exclusive；明确 last_input_note_off_tick 来源、request cutoff 的 beat 对齐、run_stop 的 tail/取整规则。
- [x] **T2.4 实现 graceful drain**：停止生产后等待 request queue、in-flight、response queue 清空；使用显式 sentinel/join；stop 幂等；增加 hard timeout，超时记 invalid。
- [x] **T2.5 content lifecycle 日志**：逐 request 记录 session/request id、generation tick、expected/enqueued/started/succeeded/failed/processed、HTTP error、pending-at-stop、input increment/cumulative digest、raw token digest。
- [x] **T2.6 operational lifecycle 日志**：记录 stale merge/drop、logical/scheduled tick、late、clamp、dropped model event、forced note-off、orphan recovery 和 max lateness。
- [x] **T2.7 content gate**：expected request 集合精确等于 succeeded+processed 集合；input digest 正确；无 HTTP failure 和 pending。
- [x] **T2.8 operational gate**：在 content valid 基础上再检查预定义的 zero drop/late/clamp/forced 条件。
- [x] **T2.9 formal failure policy**：首 attempt 永久保留；content failure 按固定次数重跑 matched block；operational late/drop 作为结果保留，不能为了 theoretical 主分析静默排除。
- [x] **T2.10 空输出**：成功空响应仍写合法空 theoretical_model.mid、空 token/event summary 和 empty_success=true，与 request 失败明确区分。
- [x] **T2.11 生命周期测试**：覆盖最后一个 beat-tail request、空响应、HTTP exception、drain timeout、stop 重入、pending boundary future、cutoff 后到达的 response。

### T3. RT sample seed、session epoch 与 runtime contract

- [x] **T3.1 Generator 接口贯通**：sample_token 增加 generator 参数，并更新 backend、model.py、inference_adapter.py、inference_v2.py 等全部调用点。
- [x] **T3.2 实现 session epoch**：reset_session 返回 session_id/session_epoch/effective_seed；后续 generate 携带 epoch，旧 session 的迟到请求被拒绝并记录。
- [x] **T3.3 原子 reset**：gate 新请求 → drain/cancel boundary work → 获取 state/model lock → clear histories/token histories/active pitches/request BPM → reset Generator → ack；避免持锁等待需要同一锁的 future。
- [x] **T3.4 冻结异步 RNG 顺序**：若 boundary executor 和主 generation 共享 Generator 仍不稳定，改成确定顺序的单 generation queue，或按 request/boundary ID 派生 sub-generator。
- [x] **T3.5 保存 raw token artifact**：逐 response 保存 raw/structural tokens、request/session id、seed/state digest；不能以 MIDI 代替逐 token 验收。
- [x] **T3.6 扩展 runtime_info**：返回 checkpoint path+sha256、code/source identity、device/dtype、effective BPM、sampling 参数、prompt context、history retention、runtime caps、time-signature index、sample seed/session epoch。
- [x] **T3.7 fail-closed 检查**：driver 将 runtime_info 与 campaign config 逐字段比较；缺字段、fallback、hash 或值不一致均拒绝运行。
- [ ] **T3.8 确定性验收**：相同 input+seed 的 RT 至少连续两次逐 request raw tokens、theoretical roll、input digest 一致；失败不得继续 formal。

### T4. 扰动工具：canonical notes、稳定 PRF 与 shared edit plan

- [x] **T4.1 固定 parser/writer**：明确基于 mido raw events、PrettyMIDI notes 还是 canonical note table；量化 note 到 raw MIDI 的映射不能临时混用多个 universe。
- [x] **T4.2 建 canonical source-note table**：只包含 matched、non-drum、pitch 21–108、model-visible notes；source_note_id 基于 source hash + instrument/track ordinal + original note ordinal。
- [x] **T4.3 固定非 model-visible 事件策略**：明确 dangling note、drum、越界 pitch、CC/pedal、pitch bend、tempo/key/time-signature、track/channel/name 的 preserve/drop/fail 行为；所有 condition 一致。
- [x] **T4.4 网格 gate**：除 PPQ%4==0 外，断言参与扰动的 raw start/end 都落在 PPQ/4 网格；否则 fail-fast 或按预定义策略量化。
- [x] **T4.5 实现稳定 keyed PRF**：禁止 Python 内置 hash；使用 canonical serialization + BLAKE2b/SHA256，输入包含 schema version、source hash、perturb seed、note id、decision name、attempt index。
- [x] **T4.6 分离 decision streams**：pitch selection/offset、onset selection/delta、collision attempts 使用不同 domain key。
- [x] **T4.7 dose nesting**：high-dose 与 medium 复用 selection score，使 medium mask 为 high-dose mask 子集；记录 requested/effective/model-visible dose。
- [x] **T4.8 实现 T0.6 的 shared edit plan**：固定遍历顺序、candidate order、collision/giveup；自动断言约定的 cross-arm pairing invariant。
- [x] **T4.9 sidecar schema**：记录 schema/tool version、source hash、note id、original/proposed/final state、selection score、candidate order、effective edit、giveup/no-op reason、metadata policy、output hash。
- [x] **T4.10 replay verifier**：只应用 sidecar 已记录的 final edits，不重新运行 RNG/collision；明确验收是 canonical semantic event equality 还是 byte equality。
- [x] **T4.11 fixture 测试**：覆盖 p=0/p=1、pitch 21/108、start=0、duplicate onset、原生 overlap、chain collision、equal-start、dangling、off-grid、drum、giveup、both pairing、medium⊂high、跨 PYTHONHASHSEED 复现。
- [ ] **T4.12 五首验收**：40 个输出中 sham model-roll identity；MIDI/sidecar/manifest 一一对应；逐歌报告 selected/effective/giveup/roll diff。

### T5. Staging、NPZ 与模型实际输入一致性

- [x] **T5.1 使用新鲜 staging 目录**：发现 stale MIDI/NPZ/sidecar 立即 fail；manifest 使用相对路径和 sha256。
- [x] **T5.2 acc 完整性**：每个同名 acc 副本与 source acc hash 相同；检查 orphan、extra、missing。
- [x] **T5.3 midi_to_npz --strict**：读取 expected manifest；任一 skip、extra/missing stem、hash mismatch、转换失败均非零退出；输出 summary JSON。
- [x] **T5.4 exact stem set**：NPZ stem 集严格等于 40-input manifest，不能只检查 converted=40。
- [x] **T5.5 静态 two-way gate**：按 measure 拼接 NPZ channels 0:2；MIDI roll zero-pad 到 NPZ 的 T；断言 shape/cell 完全相同且无 note end 超界。
- [x] **T5.6 分开 horizon**：分别定义 validation_horizon、analysis_end_tick、run_stop_tick；不能用 analysis crop 掩盖转换尾部错误。
- [x] **T5.7 offline actual-input gate**：记录实际选择的 NPZ path/hash，并验证 part0 encode→decode；BAR/pad 和 prompt digest 可追溯。
- [x] **T5.8 RT per-request gate**：检查每个 input event 在 enqueue/merge/send 中恰好一次；比较 server 实际编码的 context start、part0 roll digest、part0 token digest。
- [ ] **T5.9 Phase 0 验收**：40/40 静态输入通过；qualification/formal 每个 RT run 的动态 input gate 通过；任一 mismatch fail-closed。

### T6. Driver、manifest 与执行隔离

- [x] **T6.1 dedicated server**：driver 从同一 clean worktree 启动专用 server subprocess，使用独立端口/log dir；记录命令、PID、cwd、env allowlist 和 source identity。
- [x] **T6.2 frozen run schedule**：生成 160 行 run_manifest.jsonl，固定 run_id、pipeline、song、condition、pseed、sseed、input/NPZ hash、顺序和预期 artifacts；按 run_order_seed 随机化/交错条件。
- [x] **T6.3 run 目录幂等**：完整 artifact 通过 hash 后可 skip；incomplete/corrupt run 使用新 attempt_id，不覆盖首 attempt。
- [x] **T6.4 每 run 前置检查**：clean SHA、campaign config hash、runtime_info、checkpoint、input/NPZ/acc hash、reset ack、session epoch 全通过。
- [x] **T6.5 每 run 后置检查**：输出 content/operational verdict、artifact exact set、token/event counts、empty-success、analysis coverage 和 hash index。
- [x] **T6.6 campaign completeness**：expected 160 run_id 全部有 verdict；missing、invalid、retried、empty、operational late 分栏汇总；重跑 attempt 不算独立样本。

### T7. 提交、clean worktree 与 qualification

- [ ] **T7.1 Phase A 测试**：offline all-mode、RT drain/validity、Generator/reset、perturb replay、strict staging、driver dry-run、analyzer fixture 全绿。
- [ ] **T7.2 提交并冻结代码**：提交 plan、production code、scripts、tests；记录 code SHA、checkpoint hash、lockfile/environment、GPU/CUDA/PyTorch、device/dtype；从独立 clean worktree 运行。
- [ ] **T7.3 gold regression**：song 4/5 × tempo 15/120 全绿；使用 stem 寻址并断言 real model/checkpoint hash。
- [ ] **T7.4 先做确定性 qualification**：先确认 offline/RT 同 seed 逐 token 一致，再比较 tempo/tail。
- [ ] **T7.5 tempo qualification**：使用冻结的 song/condition/pseed/sseed；tempo 60 不满足 operational gate 则全 campaign candidate 改 30；30 仍失败则停止，禁止 per-run 混 tempo。
- [ ] **T7.6 tail convergence**：相同 input+seed 比较 8/16/24，只看 analysis window 内 theoretical tokens/roll 和 coverage。8=16=24 取 8；仅 16=24 取 16；否则取 24 或停止排查。
- [ ] **T7.7 qualification 失败分支**：任何 input digest、determinism、drain、artifact、metric fixture 失败均回到 Phase A/B；不得边跑 formal 边修。
- [ ] **T7.8 C5 最终冻结**：写 campaign_config.json，包含最终 tempo/tail、seed、条件表、窗口、validity/retry、指标/统计/听测 manifest hash；计算 sha256，Phase D 只接受该 hash。
- [ ] **T7.9 更新预算**：tempo 60 + tail 24 的 formal RT 按约 2.8–3h，另加 qualification；tempo 30 按 5h 以上；报告实际 wall time 和重试成本。

### T8. Formal 160-run pilot

- [ ] **T8.1 冻结 40-input manifest**：40 个 MIDI/sidecar/NPZ/acc 映射和 hash 全通过 T4/T5。
- [ ] **T8.2 运行 offline 80**：按 frozen schedule；每 run 独立 reset Generator；完成 content/artifact gate。
- [ ] **T8.3 运行 RT 80**：每 run 原子 reset/session epoch；保存 theoretical、combined、raw tokens、schedule trace、lifecycle validity、input digests。
- [ ] **T8.4 生成 targeted controls**：从预冻结 artifacts 派生 identity、harmonic m2/TT、rhythm fixed shift、coverage dropout/empty；不修改 formal model run 数。
- [ ] **T8.5 完整性审计**：160 expected run verdict、attempt 历史、40-input hash、final config hash、runtime info 一致后，才允许打开 treatment-level 分析。

### T9. Analyzer：sensitivity、two-part quality 与统计

- [x] **T9.1 canonical roll loader**：offline/RT theoretical/combined 全部恢复到公共 4tpb grid，使用同一 exclusive analysis window；覆盖空 MIDI、retrigger、悬挂 note 和 tempo header 差异。
- [x] **T9.2 sensitivity**：实现 tick sustain/onset Jaccard/F1、onset distance；定义 both-empty、one-empty、无 onset 的返回值/flag；不得用 sensitivity 宣称质量改善。
- [x] **T9.3 精确 D 公式**：interval class = abs(acc_pitch-mel_pitch) mod 12；集合 {1,2,6,10,11}；正文“加权对数”改成“加权计数”；同时报告 micro pair-tick D 和可选 macro per-coactive-tick D。
- [x] **T9.4 two-part endpoint**：先计算 harmonic_pair_coverage、onset/beat、active pitch-tick/beat、empty-beat ratio，再在有效 coactive pairs 上计算 conditional D；零分母记 NA+coverage failure。
- [x] **T9.5 交叉参照分解**：
  - direct melody effect = D(acc_sham, dirty) - D(acc_sham, clean)；
  - adaptation effect = D(acc_cond, dirty) - D(acc_sham, dirty)；
  - intended-fidelity effect = D(acc_cond, clean) - D(acc_sham, clean)。
- [x] **T9.6 primary/secondary**：primary contrast = medium both vs sham；primary quality = D_intended + coverage guardrail；D_actual/adaptation、单因素、rhythm、high-dose、combined scheduling 标为 secondary/exploratory。
- [x] **T9.7 聚合 estimator**：相同 pseed/sseed 内先算 paired contrast；按冻结规则聚合 pseed、再聚合 sseed，得到每歌一个 effect；sham 重用不能复制成独立样本。
- [x] **T9.8 NA estimator**：每表同时给有效 block 数、NA pattern、coverage failure；禁止 treatment 为空时只保留幸存样本。
- [x] **T9.9 song-block summary**：列出 5 个 raw song effects、等权 overall estimator、leave-one-song-out range；bootstrap 只重采完整 song block，固定 interval、次数和 seed，并标注 descriptive。
- [x] **T9.10 control acceptance**：identity 指标不变；harmonic/rhythm/coverage controls 按预定义方向和最小变化通过；不用“显著更差”。失败时对应 endpoint 标 assay invalid。
- [x] **T9.11 analyzer fixture**：人工构造 consonant、m2/TT、全空、部分 coverage、密集和弦、节奏平移、identity，断言 metric/guardrail/NA 精确值。
- [x] **T9.12 输出**：生成 machine-readable metrics、per-song tables、run-level QC、control report、bootstrap config/result 和图表，每项带 campaign config hash。

### T10. 听测 manifest、渲染与盲评包

> selection/build/audit/seal/unblind 工具已有定向测试；T10.3/T10.4 的 canonical render 与 4× polyphase inter-sample true-peak 代码已通过定向测试。其余 checkbox 还要求冻结后的真实 artifact 或用户步骤，因此在正式 selection、soundfont/WAV 包和盲评完成前保持未勾。

- [ ] **T10.1 预冻结 selection manifest**：在 Phase D/E 前写死 pseed/sseed、每首 excerpt 起止 model beat、theoretical/combined 来源、known-bad/high-dose clip、重复 trial、呈现顺序和 seed。
- [ ] **T10.2 明确听测问题**：ecological 只回答 end-to-end joint quality；acc-solo 只回答 standalone coherence。若要测 accompaniment adaptation，用同一 dirty melody 下的 sham acc vs dirty-conditioned acc 交叉混音替换部分 acc-solo。
- [x] **T10.3 canonical render**：从 model-tick grid 重建 BPM=120 的 MIDI/WAV；冻结 soundfont hash、synth 版本、program、sample rate、bit depth、render command。
- [x] **T10.4 gain policy**：使用固定 synth gain 或同 song/pair 共用 gain，只做 true-peak protection；禁止逐 clip LUFS 拉齐后放大稀疏/近空伴奏。
- [ ] **T10.5 excerpt**：按 manifest 的 beat window 裁剪；sham/treatment 窗口完全相同；不足长度使用预定义 pad，不能听后挑片段。
- [ ] **T10.6 blind package**：生成不泄露 song/condition/pipeline/seed 的 sample ID、WAV、评分表、独立 key、manifest hash；检查 blind key 一一对应。
- [ ] **T10.7 评分流程**：overall 1–5 + flaw note，可增加同歌 paired preference；第一轮盲评封存后才解盲；第二轮单列 post-unblinding qualitative follow-up。
- [ ] **T10.8 包验收**：24 clips 的数量、时长、声道、采样率、峰值、静音/缺失、hash、重复 trial 全自动检查；offline 加时包单列。

### T11. Report、归档与完成判据

> report/reproducibility-index 的 fail-closed 生成器已有定向测试；本节要求的是绑定正式 campaign、analysis 和 listening artifacts 的最终报告，因此在这些输入齐备前保持未勾。

- [ ] **T11.1 H_input**：分开报告 D_intended、coverage、adaptation 和逐歌结果；不以 sensitivity 代替 quality，不把“未观察到”写成“无效应”。
- [ ] **T11.2 H_generation_interaction**：只用 offline vs RT theoretical；明确是完整 generation pipeline 差异，不归因到单一组件。
- [ ] **T11.3 H_operational**：单列 theoretical vs combined、late/drop/clamp、tempo qualification 和首 attempt failure；不得用重跑结果覆盖运行失败。
- [ ] **T11.4 H_realworld**：明确没有真人输入，不能解释 live gap；model-tick error 不冒充人类微时序误差。
- [ ] **T11.5 Limitations**：列出 5 首歌、2 sample seeds、单听者、描述性 bootstrap、空输出和 NA；不做显著性、等价或人群外推。
- [ ] **T11.6 可复现索引**：归档 code SHA、campaign config hash、checkpoint hash、40-input manifest、160-run manifest、attempt index、runtime info、metrics、controls、listening manifest/key、report 的相互引用。
- [ ] **T11.7 Definition of Done**：40-input staging、160 expected verdict、content validity、targeted controls、analysis artifacts、盲听包和复现索引全部通过后才标 complete；缺失项必须显式列为未完成/无效。
