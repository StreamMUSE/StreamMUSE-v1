# Melody perturbation generated-acc listening campaign：实施与运行报告（2026-07-17）

## 1. 结论先行

本轮已经完成所有**可由代码和服务器自动完成**的工作：qualification hardening、fresh staging、真实
checkpoint gate、20-run qualification、160-run formal generation、artifact audit、objective analysis、95 题
acc-only triangle package、播放器、逐题持久化/恢复、partial snapshot/unblind/report workflow，以及最终回归。

当前可交付状态是：

| 轴 | 状态 | 含义 |
|---|---|---|
| formal generation integrity | 通过 | 160/160 content-valid，0 missing、0 invalid、0 extra、0 retry |
| listening source readiness | 通过 | 80/80 RT theoretical sources ready |
| blind package | 通过 | 95 scored trials + 3 practice，accepted-final WAV package |
| objective analyzer ingestion | 通过 | 240 metric rows、210 paired rows、80 lifecycle rows，0 QC-invalid |
| overall objective technical DoD | **未通过** | 预注册 rhythm targeted assay 0/5；不能把 rhythm endpoint 当有效质量结论 |
| human collection | **未开始** | answered=0，pending=95，QC=not_started，保持 fully blind |

因此不能诚实地把整个实验写成“全部结束”：生成、审计和听测工具已经完成，真正回答“人耳能不能区分”
仍要由用户本人作答；另外当前 rhythm objective endpoint 存在 assay defect。听测可以正常开始，因为 primary
human test 比较的是冻结的 WAV，并不依赖 rhythm metric 是否通过；但最终报告必须继续把这个 objective
缺口单列，不能事后降低 threshold 或覆盖原分析。

## 2. 本轮要回答的问题

固定同一首歌、同一 8 秒 excerpt、同一 checkpoint、同一 sample seed 和同一 RT theoretical generation
pipeline，只改变输入 melody 的 `pitch`、`onset` 或 `both` condition 后，模型生成的 accompaniment 是否能
被同一位用户在盲听中区分。

本轮不回答：

- 哪一个 accompaniment 更好；
- 是否更和谐、质量是否下降；
- 真实人类演奏误差是否解释 live gap；
- 结果能否外推到其他听者、歌曲或模型。

Objective metrics 只能说明事件、覆盖率或和声距离发生了多少数值变化。最终的“可听差异”必须来自
acc-only blind triangle responses。

## 3. 代码实施范围

主实现提交及后续真实运行中发现的修复如下：

| commit | 内容 |
|---|---|
| `91f58f2c2e22d53c09232fdc580e16a156a09708` | 实现 generated-acc robustness/listening campaign 主体 |
| `6f7edf63a7eaa0bdea60c4b6e0bead7ccd477622` | 修复 sustain-only ghost note decoding；v3 formal inference 冻结于此提交 |
| `4202699239aded15b77f39ae3b38b1d9de5f1287` | 统一 output-event digest 的四字段 producer/consumer projection |
| `5b350f65eb5da4dd417ab2249a641453029454fe` | 将 expected-empty excerpt canonicalize 为 literal PCM silence |
| `4a01259fec8087325283d176d45a6b2f8ebc7985` | 收口 formal listening selector projection，修复严格 report 的 readiness 交叉验证 |

实现覆盖：

1. shared frozen contracts：40 inputs、fixed-20 qualification、160 formal runs、triangle selection v2；
2. fail-closed qualification：从 immutable attempts 重新推导 determinism、tempo、tail 和 final decision；
3. offline/RT required-artifact schemas、trace/MIDI semantic reconciliation、empty-success 语义；
4. 95 题 triangle selection、prefix/chunk balance、identity/known-different/repeat controls；
5. canonical MIDI/WAV rendering、common-pair gain、4× true-peak audit、literal duplicate；
6. append-only response/sitting ledgers、CAS/atomic persistence、resume、partial/full snapshot、semantic unblind；
7. objective analyzer、coverage/NA policy、song-block descriptive bootstrap、control report；
8. strict cross-linked report/reproducibility index 和 post-unblind semantic MIDI/WAV export。

## 4. Campaign provenance

Accepted technical campaign root：

```text
task_runs/melody_robustness_20260717_generated_acc_v3
```

核心冻结 identity：

| artifact | SHA-256 / identity |
|---|---|
| formal inference code | `6f7edf63a7eaa0bdea60c4b6e0bead7ccd477622` |
| checkpoint | `905819e7ac7ac4864e5cc0308b87b04eec10a9b552aa506cec534fdd7567558b` |
| input manifest | `b4e364f98879e9a0461d387b8393dbd19cd58661178149d0b44012288c2d7670` |
| qualification result | `b2d478de6f6e17bbe4d38cfdc6e9c514c1fc2f834ccd0791e21a8460d302e0e9` |
| triangle selection | `2ca4892c7a6b7b1c011c792d449fd98e5ee75ba714bdef62fed1e0a8939f794d` |
| renderer identity | `368a5cd6ea5a19c7959dd453c587289119bfa3661639102463ee038a8782d335` |
| C5 campaign config | `5095041ab3de8d078b111ebcc635b80ee93c590aec1803f6df305a7b30188416` |
| formal schedule | `f4d6fb3e4da78b36e07d5110bcbc9120ec870ce5f8b367457502283193a82999` |
| formal campaign binding | `9ee68fab9ba4a97ac14e66b582aeeda035feab9f571a9b1c046e19a7b7a5f389` |
| campaign audit | `50200e414ef2b70c29ec69a1f37968873e6a9151c11dd824d217f4e67c47bfa1` |
| analysis index | `d723fdb2c4eedde4bb7b9ed47f8206006a552ff094d56db5e17d390347935181` |
| control report | `f9811db4e57f3dd3269fbc2b960b15f0666213b90c79fdb7065b042897d90c00` |

Attestation 另外固定了 code/worktree、checkpoint size/hash、`uv.lock`、`pyproject.toml`、Python、Torch、
CUDA、cuDNN、Torch-visible GPU 和 nvidia-smi GPU/driver identity。真实 checkpoint consistency gate 在
song 4/5、tempo 15/120 上得到 `2 passed, 0 skipped`，运行时间 917.780 秒。

## 5. Qualification 与 fresh staging

### 5.1 v2 fail-closed 记录

第一次 durable qualification 保留在：

```text
task_runs/melody_robustness_20260717_generated_acc_v2
```

它完整运行了 20/20 attempts，但 tempo 60 和 30 都被判 operational failure，最终
`passed=false`、`no_operational_tempo`，没有生成 formal campaign。根因是 decoder 将只有 sustain、没有
对应 onset 的 ghost active note 变成固定 orphan note-off；两个 tempo 都出现 7 个 orphan note-offs。

该失败没有被删除、覆盖或通过放宽 gate 绕过。修复 decoder 并增加边界测试后，另建 v3，从 fresh
attestation/qualification 重新开始。

### 5.2 v3 accepted qualification

- fresh staging：40/40 converted，0 skipped，exact stem set；
- input replay、note universe、latent pairing、MIDI↔NPZ、source-acc hash 和 horizon gates 全部通过；
- qualification：20/20 均为 `attempt-001`，0 retry，`development_only=false`；
- offline determinism：通过；
- RT determinism：通过；
- static 和 per-request dynamic input gates：通过；
- playback tempo 60 与 30 BPM 都通过，按冻结规则选择 60 BPM；
- songs 2/4 的 tail 8、16、24 全部收敛，按冻结规则选择 8 beats；
- independent validator 与 evaluate 的 derived decision exact-equal；
- 最终 `passed=true`，然后才 freeze C5。

## 6. Formal generation 与 artifact integrity

Formal schedule 精确为 offline 80 + RT 80：

- 80 个 offline `*_generated.mid`；
- 80 个 RT `theoretical_model.mid`；
- 80 个 RT `combined.mid`；
- 每个 attempt 还保留 verdict、artifact index、runtime/checkpoint/input/reset records、inference trace、
  schedule trace、request lifecycle 和 validity artifacts。

最终 campaign audit：

| 项目 | 数值 |
|---|---:|
| expected / present / content-valid | 160 / 160 / 160 |
| missing / invalid / extra | 0 / 0 / 0 |
| content retries | 0 |
| listening sources ready | 80 / 80 |
| operational-invalid RT runs | 10 |
| fully empty RT theoretical outputs | 10 |

### 6.1 Operational-invalid 不是 content failure

10 个 RT attempts 因一 tick 的 late/clamp/drop 触发严格 operational endpoint，但 artifact 完整、内容有效、
listening-ready。冻结策略要求保留它们，不能换 seed、换 run 或重跑后隐藏首 attempt。

| run | song / condition | perturb/sample seed | late | clamp | drop |
|---|---|---|---:|---:|---:|
| `mr-7a8f4db8a0e9d4fd` | 2 / both | 1001 / 1101 | 7 | 3 | 2 |
| `mr-03939172815a3832` | 2 / both | 1002 / 1101 | 4 | 2 | 1 |
| `mr-216bcd6ffbca6cb0` | 3 / onset | 1001 / 1101 | 1 | 0 | 0 |
| `mr-3aa5a68b27bc3c3e` | 3 / onset | 1002 / 1102 | 10 | 5 | 1 |
| `mr-57e71be16ed869c7` | 2 / onset | 1002 / 1101 | 4 | 2 | 1 |
| `mr-958cd8d120600134` | 2 / pitch | 1001 / 1102 | 4 | 0 | 2 |
| `mr-c7b1b3e5f2f9e066` | 5 / both | 1002 / 1101 | 4 | 1 | 0 |
| `mr-c6278db77583d767` | 3 / both | 1002 / 1102 | 6 | 0 | 3 |
| `mr-834076fcae17f9ac` | 2 / high | 1001 / 1102 | 2 | 1 | 0 |
| `mr-df25df7b1beec56c` | 3 / both | 1001 / 1102 | 1 | 0 | 1 |

80 个 RT runs 的 lifecycle aggregate：7152/7152 requests processed and succeeded；late events 43、clamped
onsets 14、dropped model events 11、max lateness 1 tick。forced/orphan/stale/http/failed/pending 都为 0，
request coverage min/mean 都为 1.0。

### 6.2 Fully empty 是模型结果，不是导出失败

10 个完整 `theoretical_model.mid` 合法为空，均为 110 bytes，SHA-256 为
`394343c5fa59b7b93d1799ad3f8862a8702f6e2515afbcb6e32350febe0083e0`：

| run | song / condition | perturb/sample seed |
|---|---|---|
| `mr-a327fdcd1ec88870` | 1 / pitch | 1001 / 1102 |
| `mr-9100eadadad7b9cd` | 4 / sham | none / 1101 |
| `mr-6df4cf7440f8d201` | 5 / high | 1001 / 1102 |
| `mr-56667a6e469dee95` | 4 / pitch | 1001 / 1101 |
| `mr-322b9fa15c442ea8` | 4 / high | 1001 / 1101 |
| `mr-621f38be60145dfd` | 1 / high | 1001 / 1102 |
| `mr-25a4bfdb3fcb7477` | 4 / pitch | 1002 / 1101 |
| `mr-1c38ad0e71c694be` | 4 / onset | 1002 / 1101 |
| `mr-b48ab6ff011ea547` | 4 / onset | 1001 / 1101 |
| `mr-1d825f4d4f419fb4` | 4 / both | 1002 / 1101 |

它们在 quality metric 中是 `NA + coverage_failure`，不能填成 0，也不能描述为“质量改善”。固定
`[16,32)` excerpt 后还有一些完整 MIDI 非空但该窗口为空；因此 listening package 的 80 个 unique formal
excerpts 中共有 18 个 literal-silence excerpts。这里的 18 与“完整输出全空”的 10 是两个不同层级。

## 7. Objective results

Analyzer 生成了 240 metric rows、210 paired contrasts 和 80 lifecycle rows，`qc_invalid=0`、
`lifecycle_schema_invalid=0`。以下 interval 是 5-song block descriptive bootstrap，不是 population-level
显著性或等价性检验。

### 7.1 Offline `D_intended(condition) - D_intended(sham)`

| condition | estimate | descriptive 95% interval |
|---|---:|---:|
| onset | +0.00773 | [-0.01192, +0.02466] |
| pitch | +0.01557 | [-0.00892, +0.04280] |
| both | +0.02317 | [-0.00002, +0.04715] |

Sham mean `D_micro` 约为 0.14457。方向上是 `onset < pitch < both`，但三个 interval 都没有明确排除 0。

### 7.2 Offline `D_actual` joint-treatment effect

| condition | estimate | descriptive 95% interval |
|---|---:|---:|
| onset | +0.01111 | [-0.00863, +0.02638] |
| pitch | +0.02556 | [-0.00786, +0.06299] |
| both | +0.03587 | [+0.00764, +0.06878] |

只有 `both` 的 interval 排除 0，但这是“输入扰动 + 生成响应”的 joint treatment effect，不能称作模型
adaptation，也不能直接解释为听感变差。

### 7.3 RT aggregate、coverage 与 event branching

RT theoretical/combined 的完整 aggregate effect 和 interval 为 NA，不是效果为零。冻结的 5-song
complete-block 被 empty output 破坏：onset 4/5 songs 有效、pitch 3/5、both 4/5。

RT onset 相对 sham 的 coverage 描述显示输出稍稀疏：empty-beat ratio 约 +0.0580，descriptive interval
[+0.0217,+0.0943]；onsets/beat 约 -0.2101，interval [-0.4106,-0.0261]。

Matched event-stream overlap 说明生成确实会分叉：

| pipeline | onset F1 | pitch F1 | both F1 |
|---|---:|---:|---:|
| offline | 约 0.620 | 约 0.658 | 约 0.591 |
| RT theoretical | 约 0.100 | 约 0.179 | 约 0.070 |

可以据此说“扰动后生成 MIDI 事件流明显不同”，但不能据此说“人耳一定听得出”或“质量下降”。

## 8. Targeted rhythm assay 的已知问题

Controls 的真实结果：

- harmonic：5/5 pass；
- coverage dropout/empty：5/5 pass；
- identity：5/5 pass；
- listening known-different：6/6 recipe/source bound 且 non-identical；
- rhythm：**0/5 pass**。

冻结的 +1 tick rhythm control 在五首歌得到 0.667、0.224、0.820、0.919、0.254 ticks，均低于预注册
threshold `>=1`。

问题不在 control MIDI 平移：它确实将 onset 向后移动一 tick。问题在当前 `_onset_distance` 是
pitch-agnostic、双向 nearest-neighbour/Chamfer distance。稠密 onset 或和弦中，移动后的 onset 会匹配到
另一个相邻 onset，距离可能为 0；原 sparse unit fixture 没有暴露这个 aliasing。

处理原则：

1. 不事后把 threshold 调低；
2. 不覆盖 immutable `40_analysis`；
3. 当前 rhythm endpoint 标记 assay-invalid，overall report-builder technical DoD 保持 incomplete；
4. 如要修正，新增 versioned posthoc analysis amendment，绑定原 analysis hash 和新 analyzer code；
5. 新 endpoint 建议使用保留 onset multiplicity 的一维 Wasserstein 或 one-to-one assignment，并补 dense、
   adjacent、polyphonic、boundary regression tests。

该问题不阻止听测：triangle WAV 的选择、渲染和答题判定不使用 rhythm metric。

## 9. Blind listening package

Final package：

```text
task_runs/melody_robustness_20260717_generated_acc_v3/50_listening
```

Audit 状态：`valid=true`、`accepted_final=true`、`blinding_audited=true`、
`development_midi_only=false`、`errors=[]`。

| 项目 | 数值 |
|---|---:|
| scored trials | 95 |
| practice trials | 3 |
| scored WAV presentations | 285 |
| total blind WAV including practice | 294 |
| unique canonical MIDI/WAV sources | 88 |
| frozen renderer rerendered sources | 88 |
| matched pairs independently rebuilt | 90 |

Block counts 为 60 medium、10 high、5 sham sampling、6 identity、6 known-different、8 exact repeat。每段
8 秒、120 BPM、44.1 kHz、PCM16、固定 synth gain；normal trial 的重复 arm 是同一 canonical WAV 的
literal byte copy。

Package-level scored-trial diagnostics：

- objective-identical：19/95，其中 medium primary 9/60；
- coverage-driven：17/95，其中 medium primary 11/60；
- source-empty-either：28/95，其中 medium primary 18/60；
- operational-invalid-either：11/95，其中 medium primary 9/60。

这些题全部按预注册规则保留。Objective-identical 不删除；empty/collapse 不换 source；operational-invalid
不因播放调度 flag 被替换。

重要 hashes：

| artifact | SHA-256 |
|---|---|
| package audit | `76372c211d128319a33216d37aa94303a12bb18bef0d8b9fccd0c5131bea7fd8` |
| blind player | `f5992acc730c47a7e83c3ac17fcfa3c2b6665af89d749395fcb130d6897661b7` |
| public manifest | `fa9bce28af74eff74aeca1803a40df04516e2a2ff7b0f6e3f38d9ba0a6ac6396` |
| private key | `8143a3c6ad6d41a8fc2d704ae5322952e7d968b7d0d9ad8d26ba38da035587e4` |
| render manifest | `bddf1fc71ebf16e6744a36e03f4adea336269125f4263cfa5fbd7acf5c761bd1` |

构建 listening package 时发现并保留了两个 pre-final failure：

1. consumer 曾把额外 logging fields 纳入 output-event digest，导致 4877 个 non-empty inference rows 被
   错拒绝；修复为 producer 的 `{type,pitch,tick,velocity}` 四字段 projection，formal artifacts 不改；
2. FluidSynth 对 empty MIDI 会产生约 -87 dB noise；raw render 仍记录在 provenance，canonical empty
   excerpt 强制写 literal zero PCM，auditor 独立重渲染并重应用同一 policy。

第一次失败的 noisy-empty package 保留在：

```text
task_runs/melody_robustness_20260717_generated_acc_v3/50_listening_failed_pre_literal_silence
```

## 10. 当前 human status 与怎么开始听

当前 response ledger 为空：

- `collection_status=not_started`；
- answered=0，pending=95；
- `qc_status=not_started`；
- `blinding_status=fully_blind`；
- 无 snapshot、无 semantic unblind、无 human conclusion；
- `generated_acc_after_unblind/` 现在不存在是正确行为，避免听前泄露 condition。

播放器的真实文件是：

```text
task_runs/melody_robustness_20260717_generated_acc_v3/50_listening/blind/player.html
```

正式作答推荐从服务器启动持久化 HTTP mode：

```bash
CAMPAIGN="$PWD/task_runs/melody_robustness_20260717_generated_acc_v3"

uv run python scripts/prepare_robustness_listening.py serve-triangle \
  --package-dir "$CAMPAIGN/50_listening" \
  --host 127.0.0.1 \
  --port 8765
```

本地电脑另开终端：

```bash
ssh -L 8765:127.0.0.1:8765 <USER>@<SERVER>
```

浏览器打开 `http://127.0.0.1:8765/`。先做 3 道 practice；正式题开始前填写耳机/环境并开始 sitting。
每次有时间答 1 题或任意多题都可以，每题提交后服务器才显示“已保存”。关闭后可以继续下一道
unanswered trial；不要求一次或最终答满 95 题才有结果。

若希望保持最严格的 blind conclusion，在准备停止本轮收集前只看 blind progress，不先做 semantic
unblind。任意 `n>=1` 都可以 snapshot 和出 actual-denominator partial report；一旦查看 semantic partial
summary，后续回答仍保留，但永久标记为 post-partial-unblind exploratory。

## 11. n=0 strict report

当前 objective-only report 已生成：

```text
task_runs/melody_robustness_20260717_generated_acc_v3/60_report/not_started/report.md
task_runs/melody_robustness_20260717_generated_acc_v3/60_report/not_started/reproducibility_index.json
```

- report SHA-256：`a3dc34e9b273ade12853927be6cefe6da1ec595ac2211dfd4c7aeb2de017d415`；
- reproducibility index SHA-256：`b2a510244824adaa389f4d4853fdfdb73f0d7cc9fc5461fbc7240659b3583cb7`；
- status：`incomplete`；
- 唯一 false technical DoD check：`targeted_controls`；
- human state：0/95、not_started、fully_blind。

Strict builder 的 exit code 2 是预期行为：它确保 rhythm assay failure 不会被包装成 technical COMPLETE。
使用 `--allow-incomplete` 只会改变进程退出行为，不会改变报告状态，因此本轮没有用该 flag 掩盖问题。

## 12. 验证结果

最终代码回归：

```text
uv run pytest -q tests/unit tests/integration
530 passed, 1 warning in 30.13s
```

Warning 来自 `pretty_midi` 依赖使用已 deprecated 的 `pkg_resources`，不影响本实验验收。

另外完成：

- real checkpoint consistency：2 passed，0 skipped；
- formal audit：160/160 content-valid；
- triangle independent audit：accepted-final；
- campaign-wide output-event digest parity：7152 inference rows；
- package-wide frozen renderer re-audit：88 sources、90 pairs；
- loopback player smoke：`GET /` 200、`GET /api/progress` 200，返回 0/95、next `Q001`、
  `can_snapshot=false`、fully blind，且未创建 response/sitting/unblind ledger；
- final report exact campaign/listening re-verification：通过，随后按 rhythm DoD 预期 exit 2；
- `git diff --check`：通过。

## 13. Plan 完成边界

已完成：T0–T6、T7.1–T7.3、T7.5–T7.9、T8.1–T8.4、T9.6–T9.7。T7.4 为
`N/A — 0 content retry triggered`，但 retry policy 已实现并测试。T7.7 的 artifact/hash-bound 工作完成，
同时记录 rhythm endpoint 0/5，不能把“controls 已生成”误写为“所有 endpoints 有效”。

仍需真实用户行为后才能完成：T8.5–T8.11、T9.1–T9.5、T9.8。它们包括实际戴耳机作答、首条 ledger
持久化实测、partial/full snapshot、semantic result、human/objective 对齐、post-unblind export 和自由复听。
这些工作流的代码和 synthetic tests 已完成，但不能用 synthetic response 冒充人耳数据。

## 14. 限制与下一步

主要限制：5 首歌、2 个 sample seeds、固定 8 秒 excerpts、单听者、trial/source dependence、大量共享
sham source、RT empty output、描述性 bootstrap，以及 triangle discrimination 不等于质量或偏好。

下一步只有两条，且互不阻塞：

1. 用户开始 flexible blind listening；答多少就用多少 immutable data 出 partial result；
2. 若还需要可信的 objective rhythm displacement 结论，新建不可覆盖原分析的 corrected amendment，使用
   one-to-one/Wasserstein endpoint 并明确标作 posthoc。

在至少有一条真人 response 前，关于 onset/pitch/both 是否“听起来不一样”的唯一正确结论仍是：

> 生成 MIDI 事件流在 metric 上已经明显分叉，但人耳可辨别性尚无数据，不能下结论。
