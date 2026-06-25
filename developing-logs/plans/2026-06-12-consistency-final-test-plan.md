# Final Consistency Test 计划（2026-06-12，rev3）

> rev2 修正：realtime 侧是**真正的 `RealTimeMusicService`**（`streammuse-cli --input-mode midi_file`），不是 fake-rt 脚本。
> rev3 修正：① 用 **tempo 阶梯**（120/90/60/15）解决真实时的时序敏感问题；② 明确 **BPM 条件 token 一致性**约束——realtime 和 offline 的条件 BPM 必须相同，否则开头 BPM token 不同、后续生成全部分叉（历史上真踩过这个坑）。

## 目标

在确定性采样参数下（top_k=1、temperature=0、top_p=0），真实 realtime 链路（CLI → 3 线程 service → HTTP → server）与 Offline 一次性生成产出的 accompaniment 应**逐事件完全一致**（100% match，不是阈值匹配）。

这是系统的"金标准"端到端回归测试：覆盖 `MidiFileInput` 事件注入 → `_tick_loop` 触发语义（tick=0 全历史 + 每拍末尾增量）→ `_inference_worker` → `HttpInferenceClient` → `LekaiHttpBackend` 滑窗 → 录制链路，对照 offline 的 NPZ 一次性生成。任何破坏时序语义、tokenizer、滑窗、prompt 构造的改动都会让它变红。

## 核心设计 1：tempo 阶梯（解决真实时的时序敏感）

`--tempo` 只控制 wall-clock tick loop 的节奏，**不影响生成内容**（见下面 BPM 一节），所以可以自由放慢时钟来消除"推理赶不上 schedule"的干扰：

| wall-clock tempo | 256 ticks 实跑时长 | 含义 |
|---|---|---|
| 120 | ~32s | 真实演奏速度，最严格 |
| 90 | ~43s | |
| 60 | ~64s | |
| **15** | ~4.3min | 极慢，推理绝不可能掉队 → **一致性的金标准 rung** |

诊断语义：

- **tempo 15 红** → 系统一致性回归（与推理速度无关），真 bug；
- **tempo 15 绿、tempo 120 红** → 不是一致性问题，是推理速度跟不上实时 schedule（环境/性能问题）；
- 全绿 → 系统一致 且 当前环境能跑满 120 BPM 实时。

实现上每个 tempo 是独立的 parametrize case；同时保留**零丢弃前置检查**（session 日志无 "dropped … stale request"）——有丢弃时直接报 `run invalid: inference too slow`，把"速度红"和"一致性红"在报错信息层面也区分开。

默认跑哪些 rung：建议默认 `15 + 120` 两端（一个保证判别一致性、一个验证实时性能），`STREAMMUSE_CONSISTENCY_TEMPOS=120,90,60,15` 可扩全阶梯。

额外免费断言：既然 tempo 不影响生成，**不同 tempo 的 realtime 输出之间也应两两一致**——这本身就是"时钟速度不泄漏进生成结果"的回归测试。

## 关于确定性参数组合的说明（已核实代码）

realtime 与 offline 共用同一个 `lekai_model/generation_utils.py::sample_token`（`lekai_http_backend.py:407` 与 `model.py:267` 均 import 它），其中：

- `temperature == 0.0` 有专门分支（`generation_utils.py:22-26`）：直接 `torch.argmax` **提前 return**，不会除零；
- 该提前 return 意味着 temp=0 下 `top_k` / `top_p` / **`repetition_penalty` 全部不生效**——`LEKAI_RT_REPETITION_PENALTY=1.2` 在本测试中实为死参数（保留只为与历史命令一致）；
- **不要**用 "temp>0 + top_k=1" 替代 temp=0：那条路径会先施加 repetition_penalty（可能改变 argmax 结果，输出与 temp=0 不同），且 top_k=1 的过滤是严格小于、logit 平局时走 `multinomial` 随机二选一（fp16 下平局不罕见），并非严格确定。**temp=0 是唯一干净的确定性路径，两侧必须严格同为 temp=0。**

## 核心设计 2：BPM 条件 token 一致性（历史踩坑点）

BPM 是模型的条件输入（prompt 开头的 BPM token），realtime 和 offline 必须用**同一个条件 BPM**，否则第一个 token 就不同，后面全部分叉。

代码现状（已确认）：

- **CLI 链路不传 bpm**：`InferenceEngineFactory` 构造 `HttpInferenceClientConfig` 时没有 bpm 字段 → payload `bpm=None` → server 端 fallback：`effective_bpm = request_bpm or int(os.environ.get("LEKAI_DEFAULT_BPM", "120"))`。
- 所以 realtime 侧的条件 BPM 实际由 **server 的 `LEKAI_DEFAULT_BPM`** 决定，与 CLI `--tempo` 完全无关——这正好实现了"时钟与条件解耦"，tempo 阶梯才得以成立。
- offline 侧条件 BPM 来自 NPZ 数据本身。

另外两个已核实的事实：

- **BPM token 是分桶的**（`PianoDataset.py::encode_bpm`）：`None→UNK`、`<90→慢`、`90–200→中`、`>200→快`，只有 4 个值。所以"一致"实际只要求**同桶**——120 vs 150 同 token，但 89 vs 90 会翻桶，None 会走 UNK。历史上的"死活不一样"大概率就是跨桶导致。
- **offline 目前没有直接传 BPM 的入口**：`model.py:118` 硬读 NPZ 的 `metadata["bpm"]`，`run_lekai_offline.py` 无 flag。

测试方案（两侧显式钉死，不依赖 NPZ 考察）：

1. 给 offline 加 BPM 直传口子（小改动，向后兼容）：`model.generate_accompaniment` 加 `bpm_override: int | None = None`（`bpm_value = bpm_override if bpm_override is not None else metadata["bpm"]`），`run_lekai_offline.py` 加 `--bpm` 透传；
2. 测试统一用 **`LEKAI_DEFAULT_BPM=120`（server / realtime 侧）+ `--bpm 120`（offline 侧）**，两边永远落在同一个桶；
3. Phase 0 仍读一次 NPZ 的原生 BPM 做记录（确认 5 首歌原生值都在哪个桶，留档备查），但测试正确性不再依赖它。

## 两条路径与对比方式

| 环节 | 命令 | 关键参数 |
|---|---|---|
| 启 server | `python -m streammuse.infrastructure.inference.server_lekai` | `LEKAI_CHECKPOINT_PATH`、`LEKAI_DEVICE=cuda`、`LEKAI_DEFAULT_BPM=120`、**`LEKAI_RT_TEMPERATURE=0.0`、`LEKAI_RT_TOP_K=1`、`LEKAI_RT_TOP_P=0.0`、`LEKAI_RT_REPETITION_PENALTY=1.2`** |
| Realtime（真实时） | `uv run streammuse-cli --input-mode midi_file --midi-file-path prompts/inputs_lekai/mel/<id>.mid --model-name lekai --inference-type http --server-url http://127.0.0.1:<port>/generate_accompaniment --generation-interval-ticks 4 --generation-length-frames 4 --max-ticks 256 --tempo <rung> --output-type session --log-dir <tmp>` | `--tempo` 取阶梯值；session 输出供对比 + 有效性检查 |
| Offline | `scripts/run_lekai_offline.py 
  --temperature 0.0 --top-k 1 --top-p 0.0 --gt-prefix-beats 0 --bpm 120` | `--bpm` 为新增直传口子（见核心设计 2）；`condition_idx = id - 1` |
| 对比 | `debug_inference_consistency.py` 的对比逻辑 | realtime 取 session `combined.mid` 的 Accompaniment track，offline 取 `*_generated.mid`，按 `(tick, pitch, type)` 集合对比 |

素材已确认：`prompts/inputs_lekai/mel/{1..5}.mid` + `npz/{1..5}.npz` 配对齐全，checkpoint 在 `models/ModelLekai/epoch_4_1104_1204/model.safetensors`。

### 对比窗口

realtime 只跑 `--max-ticks 256`，offline 生成整首——直接全集合对比会把 offline 的尾部事件算成 mismatch。对比前需把两侧都截断到公共窗口（`tick < 256`，含 note_off 边界的处理）。实现期验证你之前手动比对时窗口是怎么对齐的，保持同样语义。

## 测试设计

### 位置与门控

- 新建 `tests/consistency/test_realtime_offline_consistency.py`。
- **默认 skip**：需要真模型 + GPU + 分钟级 wall-clock。门控：`LEKAI_CHECKPOINT_PATH` 已设置且存在，否则 skip。
- `pyproject.toml` 注册 `consistency` marker。运行方式：

```bash
LEKAI_CHECKPOINT_PATH=models/ModelLekai/epoch_4_1104_1204/model.safetensors \
  uv run pytest tests/consistency/ -v
```

日常 `uv run pytest tests/` 只显示 skipped，不受影响。

### 测试流程

```
1. 选空闲端口启动 server_lekai（确定性 env + LEKAI_DEFAULT_BPM + LEKAI_SERVER_PORT）
2. 轮询 /health 直到 ready（超时 fail 并 dump server log 尾部）
3. 预热请求一次（首次 CUDA kernel 编译最容易在快 tempo 下触发 drop）
4. 对每 (歌, tempo rung)：subprocess 跑 streammuse-cli → session 目录
5. 前置断言：session 日志零丢弃，违反报 "run invalid: inference too slow"
6. 终止 server（释放显存）
7. subprocess 跑 run_lekai_offline.py → *_generated.mid
8. 断言一：每个 (歌, rung) 与 offline 在公共窗口内 mismatched == 0
9. 断言二：同一首歌不同 rung 的 realtime 输出两两一致
```

两条路径全部走现有入口（CLI + 脚本），**不复制任何生成/调度逻辑**——它们本身就是被测对象，测试只做编排和断言。

### 对比逻辑落位

`debug_inference_consistency.py` 的 `load_track_events` + `compare_event_lists`（~60 行）：

- **方案 A（推荐）**：抽到可 import 的位置，debug 脚本和测试共用，单一事实来源。
- 方案 B：测试内复制最小实现，简单但会漂移。

### 断言强度与失败输出

- `mismatched == 0`，不留阈值——你已验证可达 100%，不等即回归。
- 失败时输出差异事件前 N 条 + 两侧文件路径；session 目录与 offline 输出保留到 `output/consistency/<timestamp>/` 供事后用现有 debug 脚本深挖。

### 默认规模

- 默认：**1 首歌（song 1）× 2 个 rung（15 + 120）** ≈ 5 分钟左右。
- `STREAMMUSE_CONSISTENCY_SONGS` / `STREAMMUSE_CONSISTENCY_TEMPOS` 扩到全量（5 首 × 4 rung，发版前完整校验，约 40 分钟）。

## 实现步骤

1. **手动验证一遍链路**（CLI 真实时跑通一次）：① 确认 `combined.mid` 的 Accompaniment track 完整记录模型输出（若受回放调度影响则改用 `inferences.json` 重建事件序列对比）；② 两侧显式 BPM=120（server `LEKAI_DEFAULT_BPM=120` + offline `--bpm 120`）后，从 `[PROMPT_DEBUG]` 日志确认两侧 BPM token 一致；③ 确认对比窗口语义；④ 记录单首耗时。
2. 抽取 MIDI 对比函数（方案 A），`debug_inference_consistency.py` 改 import，跑现有流程确认行为不变。
3. 写 `tests/consistency/conftest.py`：server fixture（空闲端口、确定性 env、health 轮询、预热、teardown、失败 dump log）。
4. 写测试本体：(歌 × rung) parametrize → 零丢弃前置 → 停 server → offline → 断言一/二。
5. `pyproject.toml` 注册 marker；目录补 `__init__.py`。
6. **验证判别力**：完整跑确认绿；临时改 `LEKAI_RT_TOP_K=2` 确认红；临时把 server `LEKAI_DEFAULT_BPM` 改成 80（**必须跨 encode_bpm 桶**，120→80 是 中→慢；改 150 不会翻 token、不会红）确认红——正好复现你当年的 BPM 坑；改回。
7. `docs/developer-guide/` 补"如何跑 consistency test"。

## 风险与注意点

- **共享 GPU**：别人占卡时快 rung 可能报 "run invalid"——这是设计内行为（与一致性红区分），但跑全阶梯时建议独占一张卡。
- **dtype/设备一致性**：两条路径统一 `LEKAI_DEVICE`/`LEKAI_DTYPE`（默认 cuda/auto，与你验证过的组合一致）。未来若翻车，第一排查项是 dtype。
- **tempo 15 的耗时**：256 ticks ≈ 4.3 分钟/首，是默认配置里最贵的 rung；如果太贵可考虑给慢 rung 降 `--max-ticks`（注意对比窗口同步缩小）。
- **素材是测试资产**：`prompts/inputs_lekai/` 和 checkpoint 成为测试依赖；换 checkpoint 时此测试守护的行为快照需同步确认。

## 决策点（已确认，按推荐方案执行）

1. **对比函数落位**：方案 A——抽公共模块，debug 脚本与测试共用。
2. **默认规模**：1 首（song 1）×（15 + 120）两 rung；env 扩全量。
3. **realtime 对比数据源**：优先 `combined.mid`；若实现期发现录制受回放调度影响，切到 `inferences.json` 重建。
4. **门控**：保持显式 `LEKAI_CHECKPOINT_PATH` env，未设置即 skip，不做自动探测。

## 执行 Todo List

> ✅ **全部执行完成（2026-06-25）**。完整结果、与计划的偏差、判别力证据见
> `developing-logs/reports/2026-06-12-consistency-final-test-report.md`。
> 默认配置（song 4 × tempo 15+120）真跑 PASSED（6m25s），日常套件 173 passed + 1 skipped。
>
> 下方各 Phase 子项的勾选状态**仅作历史参考**——实现中三处实质偏离计划原描述，以 report 为准：
> ① 对比改为 **pianoroll 归一化**（非计划写的 raw note 对比），落 `tests/consistency/midi_pianoroll.py`，未改 debug 脚本；
> ② 默认歌从 **song 1 改为 song 4**（song 1 贪心全空）；
> ③ server 与 offline **同卡共存**（143GB 显存够），省去"先停 server"编排。

### 附录 A：BPM 对照表（Phase 0.1 实测，2026-06-12）

| song | NPZ bpm | mel MIDI bpm | encode_bpm 桶 |
|---|---|---|---|
| 1 | 110 | 110.0 | 中 (90–200) |
| 2 | 74 | 74.0 | 慢 (<90) |
| 3 | 120 | 120.0 | 中 (90–200) |
| 4 | 57 | 57.0 | 慢 (<90) |
| 5 | 92 | 92.0 | 中 (90–200) |

NPZ 与 mel MIDI 两条数据源 BPM 完全一致。测试两侧统一钉 `120`（中桶），realtime 与 offline 用同一个 BPM token，与各歌原生桶无关——一致性只要求两侧相同。

### 附录 B：贪心下各歌伴奏非空程度（Phase 0 实测，2026-06-25）

贪心（top_k=1, temp=0, bpm=120）下 offline 生成的 part1 伴奏，多数歌塌缩为空，少数非空。**测试必须用非空的歌**：

| song | condition_idx | total beats | 非空拍数 | 真实 acc token | 适合做测试 |
|---|---|---|---|---|---|
| 4 | 4 | 76 | **56** | 112 | ✅ 最佳默认 |
| 5 | 1 | 116 | 19 | 76 | ✅ |
| 2 | 0 | 96 | 9 | 12 | ✅ |
| 3 | 2 | 151 | 1 | 1 | ⚠️ 几乎空 |
| 1 | 3 | 141 | 0 | 0 | ❌ 全空，勿用 |

→ **默认测试歌改为 song 4**（最初 plan 写的 song 1 恰好是全空的，不能用）。condition_idx 映射：song N → idx，见映射表 `{2:0, 5:1, 3:2, 1:3, 4:4}`。

### ✅ Phase 0 验证成功（2026-06-25）—— 前提成立，附正确对比方法

**song 4 实测：realtime 与 offline 伴奏在 pianoroll(beat,pitch)层 100% 一致**（截断到歌曲实际长度后，`只在 realtime=[]`、`只在 offline=[]`）。前提完全成立。三条必须遵守的方法论（都是踩坑换来的）：

1. **必须用非空歌**：贪心下 song 1 全空、song 3 几乎空；默认用 **song 4**（56/76 非空拍），辅以 song 5/2。见附录 B。
2. **必须在 pianoroll(beat,pitch active)层对比，不能用 raw MIDI note(tick,pitch,type)**：realtime 把持续音按拍**重触发**（每 4 拍 off+on），offline 保持**持续音**（一个长 note）。两者底层 pianoroll 相同，但 raw note 对比只有 **77.8%**（假性 mismatch）。归一化方法：把每个 note 区间展开成它覆盖的 `(beat, pitch)` 集合再比。→ **这推翻了 plan 原定的"复用 debug_inference_consistency.py 的 raw note 对比"，改为 pianoroll 归一化对比。**
3. **必须截断到公共窗口 = 歌曲实际内容长度**：melody MIDI（如 4.mid 旋律止于 beat 57）比 `--max-ticks` 短；realtime 跑过头后在无旋律区间继续挂音（pitch 43 持续到 beat 76），offline 在数据末尾就停。截断到 beat<58 → 100%。所以 realtime 的 `--max-ticks` 应贴合歌曲长度，且对比窗口取两侧公共部分。

> 注：context window 不是分叉源——`LEKAI_PROMPT_CONTEXT_BEATS=200`（全 context）与默认 32 结果完全相同，排除滑窗。分叉纯粹是"跑过头"。

### ⚠️ Phase 0 早期误判更正（2026-06-25）

> 最初用 song 1 验证，发现 offline/realtime 伴奏都空 → 一度误判"前提不成立"。**实为选歌问题**：song 1 在贪心下全空（很常见），而 song 2/4/5 非空。用户确认"有的歌全空很常见，有一两首正常"，与附录 B 实测吻合。前提成立，继续。那个到处出现的"263"是 GT（`save_gt_midi`）而非生成结果，注意别再被误导。

### ⚠️ Phase 0 阻塞记录（已解除，保留供参考）（2026-06-25）

手动验证撞上 plan 预设的"0.4 不到 100% 就停下排查"红线。核心前提（**当前单阶段 sliding-window realtime CLI 的输出 == offline**）在贪心参数下**不成立**：

- **realtime CLI（song 1, tempo 60, greedy, BPM=120）**：63 次推理**全部返回 0 个伴奏音符**，`combined.mid` 的 Accompaniment track 为空。
- **offline（song 1 = condition_idx=3, greedy, BPM=120）**：生成 **263 个音符**。BPM token 两侧一致（都是 265 = 中桶），排除 BPM 坑。
- **根因**：sliding-window 把自己生成的伴奏反馈进下一拍 context；第 0 拍贪心选了 empty marker(169)，之后每拍 context 里伴奏槽全是 169 → **自我强化的空循环**。offline 则一次性 condition 整个 part0(melody) 自回归生成整个 part1，二者是**架构上不等价的算法**。
- **历史佐证**：`developing-logs/2026-4-23/.../final_summary_report.md` 实测 offline vs fake-rt 总体 match rate **1.0%–1.15%**，"一致性未达标"，当年从未解决。报告原话："greedy decoding favors empty tokens... 模型行为问题，非代码 bug"。
- **真正达成一致的机制（晚于4月）**：commit `8250298` 的 `scripts/compare_lekai_offline_realtime_raw.py`（5/10）显示一致性是在**两阶段 prompt+continuation 架构**上、比对 streammuse-cli 存的 **`prompt_continuation_raw_history.json`**（raw acc 事件，非 audible MIDI）达成的，且依赖外部路径 `/data/home/yuanxin/RT-accompanimentV2/...`。该外部路径**在本机不存在**，`prompt_continuation_raw_history.json` 当前 streammuse-cli **不产出**，当前 server 只有单一 `sliding_window` mode。

**结论**：plan 假设的 setup（单 checkpoint / sliding-window / 比 combined.mid）与历史上真正验证过一致性的 setup（两阶段 / raw history JSON / 外部 codebase）不是同一套。需用户决策后才能继续（见与用户的对话）。

**已完成且保留的产物**：0.0 的 offline `--bpm` 口子（向后兼容，173 passed）、0.1 的 BPM 对照表（附录 A）、condition_idx→歌曲映射 `[2,5,3,1,4]`（idx 0→2.npz, 1→5, 2→3, 3→1, 4→4）。

### Phase 0：链路手动验证（写测试代码之前，把所有未知数消掉）

- [x] **0.0 给 offline 加 BPM 直传口子**（后续手动验证就要用）：`model.generate_accompaniment` 加 `bpm_override: int | None = None`（一行：`bpm_value = bpm_override if bpm_override is not None else metadata["bpm"]`），`run_lekai_offline.py` 加 `--bpm` 透传；不传时行为完全不变，`uv run pytest tests/ -q` 确认无破坏。
- [x] **0.1 留档原生 BPM**：见附录 A。NPZ 与 mel MIDI 两条数据源 BPM 完全一致（110/74/120/57/92），跨慢/中两桶；测试钉 120 不依赖此项。
- [ ] **0.2 手动跑一次 realtime 链路**：空闲端口起 server（确定性 env + `LEKAI_DEFAULT_BPM=120`），先发一次预热请求，然后 `streammuse-cli --input-mode midi_file ... --tempo 60 --max-ticks 256 --output-type session`，确认：
  - [ ] session 目录产出 `combined.mid` / `events.jsonl` / `inferences.json`；
  - [ ] `combined.mid` 的 Accompaniment track 事件数与 `inferences.json` 中各次响应合并后的事件数一致（即录制不受回放调度影响）；若不一致 → 改用 `inferences.json` 重建事件序列，并在本文件记录该决定；
  - [ ] session 日志中零丢弃（确认"dropped … stale request"的具体文案和落点：status 消息进的是哪个文件）；
  - [ ] 记录单首实际耗时（server 启动、预热、CLI 运行各多久）。
- [ ] **0.3 手动跑一次 offline**：`run_lekai_offline.py --temperature 0.0 --top-k 1 --top-p 0.0 --gt-prefix-beats 0 --bpm 120 --condition-idx 0`，确认产出 `000_1_generated.mid`，并从 `[PROMPT_DEBUG]` 日志确认第 3 个 initial token（BPM token）与 realtime 侧一致。
- [ ] **0.4 手动对比**：用 `debug_inference_consistency.py` 比 0.2 和 0.3 的输出，确认公共窗口（`tick < 256`）截断后 match 率 100%。对比脚本用 mido 在 tick 域比较是正确选择（pretty_midi 会换算成秒、把 tempo 重新卷进来），但需先确认两侧 MIDI 的 resolution（ticks per beat）相同，tick 才可直接比；不同则在对比层归一化。如果不是 100%，**停下来排查**（大概率是 BPM/窗口/参数没对齐），把原因记进本文件再继续——这一步绿了，后面才是纯工程化工作。

### Phase 1：对比逻辑抽取（方案 A）

- [ ] 1.1 新建 `src/streammuse/infrastructure/output/midi_compare.py`（或经评估后的更合适位置），迁入 `load_track_events`、`compare_event_lists`、`list_tracks`/`has_target_track`，加上**窗口截断参数**（`max_tick: int | None`）。
- [ ] 1.2 `scripts/debug_inference_consistency.py` 改为 import 公共模块，删除本地副本。
- [ ] 1.3 重跑 0.4 的对比，确认行为不变；`uv run pytest tests/ -q` 确认无破坏。

### Phase 2：测试基建

- [ ] 2.1 新建 `tests/consistency/__init__.py` + `tests/consistency/conftest.py`：
  - [ ] `lekai_server` fixture（session-scoped）：找空闲端口 → subprocess 启动 server（注入 `LEKAI_RT_TEMPERATURE=0.0 / TOP_K=1 / TOP_P=0.0 / REPETITION_PENALTY=1.2`、`LEKAI_DEFAULT_BPM`、`LEKAI_SERVER_PORT`、透传 `LEKAI_DEVICE`/`LEKAI_DTYPE`）→ 轮询 `/health`（超时 60s，失败 dump server log 尾部）→ 预热请求 → yield → teardown kill + 等待退出；
  - [ ] 门控逻辑：`LEKAI_CHECKPOINT_PATH` 未设置或文件不存在 → 整目录 skip；
  - [ ] 工件目录 fixture：`output/consistency/<timestamp>/`，测试失败时保留 session 目录与 offline 输出，通过时可清理（或一律保留，由 env 控制）。
- [ ] 2.2 `pyproject.toml` 注册 `consistency` marker（`[tool.pytest.ini_options] markers`，注意当前 pyproject 还没有 pytest 配置节，新建时确认不改变现有测试收集行为）。

### Phase 3：测试本体

- [ ] 3.1 `tests/consistency/test_realtime_offline_consistency.py`：
  - [ ] 参数化：歌曲来自 `STREAMMUSE_CONSISTENCY_SONGS`（默认 `1`），tempo rung 来自 `STREAMMUSE_CONSISTENCY_TEMPOS`（默认 `15,120`）；
  - [ ] 每个 (歌, rung)：subprocess 跑 `streammuse-cli`（参数照 plan 表格，`--log-dir` 指向工件目录），超时保护（按 rung 时长 × 2）；
  - [ ] 前置断言：CLI 退出码 0 + session 日志零丢弃，违反 → `pytest.fail("run invalid: inference too slow (N dropped)")`；
  - [ ] offline 运行：module-scoped fixture，在所有 realtime case 跑完、server 关停后执行一次（每首歌一次），缓存结果；
  - [ ] **断言一**：每个 (歌, rung) 与 offline 在公共窗口内 `mismatched == 0`，失败输出差异前 10 条 + 两侧文件路径；
  - [ ] **断言二**：同一首歌不同 rung 的 realtime 输出两两一致。
- [ ] 3.2 编排顺序确认：server fixture 必须在 offline 运行前 teardown（显存互斥）——用 fixture 依赖或显式阶段函数实现，不靠测试执行顺序的偶然性。

### Phase 4：真跑验证（需要 GPU）

- [ ] 4.1 完整跑默认配置（song 1 × 15+120），确认全绿，记录总耗时进测试 docstring。
- [ ] 4.2 判别力验证（三连）：
  - [ ] 临时 `LEKAI_RT_TOP_K=2` 且 `LEKAI_RT_TEMPERATURE=0.8` → 应红（temp=0 时 top_k 不生效，必须连温度一起改才能激活采样路径）；
  - [ ] 临时 server `LEKAI_DEFAULT_BPM=80` → 应红（跨 encode_bpm 桶，复现历史 BPM 坑；注意改 150 不跨桶、不会红）；
  - [ ] 恢复后重跑 → 绿。
- [ ] 4.3 （可选，时间允许）扩 `SONGS=all TEMPOS=120,90,60,15` 全量跑一次，记录结果。
- [ ] 4.4 `uv run pytest tests/ -q` 确认日常套件不受影响（consistency 显示 skipped）。

### Phase 5：收尾

- [ ] 5.1 `docs/developer-guide/` 新增"Consistency Test"页（怎么跑、tempo 阶梯的诊断语义、BPM 约束、常见红的排查顺序），加入 VitePress sidebar。
- [ ] 5.2 在本文件勾掉所有 todo，附上 Phase 0 的 BPM 对照表和实测耗时。
- [ ] 5.3 写执行 report 到 `developing-logs/reports/`。
