# StreamMUSE Test Suite Consolidation and Reliability Plan（2026-07-20）

## 目标

在不降低行为覆盖、不改变模型输出语义的前提下，整理合并后的测试体系：

- 删除已经被更严格测试覆盖、永久 skip、或实际可能假绿的测试；
- 合并同一边界上的重复 case 和重复 helper；
- 保留跨不同 trust boundary 的相似测试；
- 让 unit、integration、真实 checkpoint consistency、performance、全样本验证各自只承担一种职责；
- 统一 Offline vs Realtime 的比较语义和 runner 基础设施；
- 保证从仓库根目录直接运行 pytest 时只收集本项目测试。

本计划不是“为了减少数字而删测试”。测试数和文件数只作为观察指标，最终验收依据是：失败检测能力不下降、测试职责更清楚、环境错误不再被 skip 掩盖、真实模型验证不再有错误的 E2E 表述。

## 与旧计划的关系

[2026-06-11-test-cleanup-plan.md](2026-06-11-test-cleanup-plan.md) 已经完整实施，当时将 190 个测试整理到 173 个。此次审查针对之后新增的 Rap、任务系统、perturbation/listening、prompt-continuation 和 consistency 测试，以及两个分支合并后重新出现的内容。

旧计划曾明确删除没有生产调用的 MusicalSequence；Isochron 合并又把该源码和 6 个测试带了回来，但当前仍没有任何生产引用，也没有从 musical package 导出。本计划重新设置一个明确的删除决策门。

## 审查快照

审查日的仓库状态：

- tests/ 下共有 78 个 test_*.py 文件：72 unit、4 integration、2 consistency；
- tests/ 的严格 collection 共 650 items：639 unit、8 integration、3 consistency；
- 日常命令的用户实测结果为 646 passed、1 skipped、1 warning，耗时 31.79 秒；
- 唯一的日常 skip 来自一个硬编码其他用户目录的 RT MIDI integration test；
- unit tests 共约 16,851 行，其中 tests/unit/scripts 约 7,304 行；
- pyproject.toml 目前只注册 consistency marker，没有 testpaths、strict markers 或 warning policy；
- 从仓库根目录直接运行无参数 pytest 会额外收集 scripts/test_logging_system.py，并继续进入 vendored transformers tests，最终 collection error；
- uv 构建会刷新已跟踪的 src/streammuse.egg-info/SOURCES.txt，使纯测试命令污染工作树。

## 测试分层的最终定义

| 层级 | 允许的依赖 | 主要职责 | 默认 PR gate |
|---|---|---|---|
| Unit | CPU、tmp_path、deterministic fake | 单个类/函数、状态机、序列化和纯计算 | 是 |
| Hermetic integration | 线程、TestClient、本地临时 MIDI、真实 factory，但不依赖外部文件/GPU | 跨层 wiring、HTTP contract、MIDI 落盘 | 是 |
| Workflow contract | 较大的 campaign/listening artifact graph，仍不依赖 GPU | hash binding、tamper detection、完整工作流 contract | 是；可另提供 fast profile |
| Consistency gold | 真实 checkpoint + GPU，慢速 tempo | Offline 与 Realtime 内容一致性 | 发布前/模型改动时 |
| Realtime performance | 真实 checkpoint + GPU，真实速度 | drop、late、clamp 和吞吐能力 | 独立运行，不与 correctness 混报 |
| All-sample campaign | 真实 checkpoint、多样本、可多 GPU | 全 15 首、resume、per-sample artifact/report | 手动 comprehensive gate |

## 审查结论：可以取消的测试

### C1. 删除外部绝对路径 RT MIDI integration test

删除 tests/integration/test_lekai_prompt_continuation_rt_midi.py。

理由：

- fixture 硬编码 /data/home/yuanxin/RT-accompanimentV2/...；当前机器因为 PermissionError 永久 skip；
- 测试只使用 fake prompt/continuation engines，不是实际模型或 HTTP E2E；
- tests/unit/infrastructure/inference/test_lekai_prompt_continuation_scheduler.py::test_scheduler_uses_midi_converted_ticks_for_prompt_and_append_boundaries 已使用临时真实 MIDI，更严格地覆盖 resolution 转换、prompt/append boundary、catch-up 和 continuation 调用。

若以后证明该录音包含无法合成的独特 regression，应该提取脱敏的最小 MIDI 到 tests/fixtures/midi/，而不是继续读取其他用户目录。

### C2. 删除或再次确认 MusicalSequence 死代码

候选删除：

- src/streammuse/domain/musical/sequence.py；
- tests/unit/domain/musical/test_sequence.py；
- docs 中把它描述为当前架构组成部分的内容。

执行前门槛：全仓生产代码和公开 import 再确认一次没有引用；如果项目明确决定将其作为近期 public API，则保留源码和测试并补真实 consumer。不得只留下“未来可能用”的源码和 6 个自证测试。

### C3. 删除或合并约 18 个高置信重复 unit cases

以下操作必须先增强保留测试的断言，再删除被覆盖 case：

| 位置 | 处理 | 预计净减 |
|---|---|---:|
| presentation/test_cli_config_parser.py::test_args_to_config_mode_matrix | 删除 3 个仅重复 input/output type 的组合；对应默认、keyboard、midi_file 测试已覆盖 | 3 |
| domain/timing/test_scheduler.py | 删除 test_get_empty_tick；把 inclusive boundary 合入 clear_future_events_all | 2 |
| domain/events/test_generic_events.py | 删除弱于两个单向字段测试的 round-trip case | 1 |
| domain/musical/test_notes.py | 合并 from_events_valid 与 preserves_channel_program；前后向 round-trip 由增强后的两个方向测试覆盖 | 2 |
| application/factories/test_output_factory.py | 删除已被 factory session case 和 sink 单测覆盖的 test_session_logger_uses_passed_bpm_and_ticks | 1 |
| presentation/debug/test_server.py | 合并两个都 GET / 的 static app/orientation 测试 | 1 |
| inference/test_lekai_http_backend.py | 将 3 个 rule fallback 长度/interval/note-off velocity 测试合成一个表驱动 contract | 2 |
| application/test_real_time_music_service.py | 合并 3 个 beat-tail 状态测试；合并 2 个 count-in 测试 | 3 |
| application/test_prompt_continuation_realtime_service.py | 把 0/1/2 次 append 的 3 个 polling 测试合为一个状态序列 | 2 |
| scripts/test_robustness_analysis_report.py | 删除 report 层重复的 N=94 参数，保留 0/1/95；94/95 producer 边界已有专门测试 | 1 |

### C4. 删除弱的 CLI early-exit test

tests/integration/test_cli_entry_point.py::test_cli_main_early_exit 当前捕获任意 SystemExit 后直接 pass，错误退出也会被视为成功；它与 mocked-builder lifecycle test、RuntimeSessionBuilder unit tests 高度重叠。

先把真正需要的 lifecycle 断言并入标准 CLI smoke，再删除该 case。所有 CLI main 测试应 patch atexit.register 和 signal.signal，避免修改 pytest 进程的全局 handler。

### C5. 删除两个逻辑必然成立的 cross-tempo 尾部断言

单阶段和 two-stage consistency 中，每个 tempo 已经分别断言等于同一个 Offline reference，因此最后再断言 tempo A == tempo B 没有独立检测能力。

删除两个 *_by_tempo 缓存和最终循环；tempo ladder 本身仍保留为独立 test items 或 profile。

## 审查结论：不能删、但应该合并实现

### M1. 单阶段 pytest consistency 与 all-sample runner

以下两个入口都必须保留：

- tests/consistency/test_realtime_offline_consistency.py：少量非空歌曲的 pytest gold gate；
- scripts/run_all_sample_consistency.py：15 首全样本、多 GPU、resume、per-sample artifacts 和汇总报告。

目前它们重复实现 MIDI interval、active cells、server lifecycle、Offline/Realtime subprocess、dropped/late 检测和比较；而且比较精度不一致：pytest 用 (beat, pitch)，full runner 用每拍 4 格的 (model_timestep, pitch)。旧 beat 级比较会漏掉同一拍内部的 tick shift。

计划抽出 src/streammuse/experiments/consistency.py 作为唯一公共实现，至少提供：

- resolution-aware MIDI interval loader；
- active_cells(..., steps_per_beat=4)；
- comparison result dataclass；
- dropped request 和 late-policy trace parser；
- runtime validity、checkpoint identity 和采样配置校验。

随后删除 tests/consistency/midi_pianoroll.py 和 full runner 内部的重复版本。为公共模块新增纯 CPU unit tests，覆盖长音与逐拍 retrigger 等价、拍内移动 1 tick 必须被发现、不同 PPQ 归一化、hanging note、空伴奏、dropped gap 和 late policy。

### M2. 重复 helper 和 fixture

- 将三份 _build_tokenizer 合并到 lekai_model/conftest.py；
- 将外层 test_lekai_inference_adapter.py 的唯一 case 移入 lekai_model/test_inference_adapter.py，删除空文件；
- 将 prompt-continuation 测试中重复的 _note_on 和 event payload builder 放到局部 support/fakes 模块；
- 将 task runtime 两份相同的 response trace reader 放到 tests/unit/application/tasks/support.py；
- 将 consistency 的 _parse_int_list、songs、tempos 解析移到 support.py，并为环境变量解析写 CPU 单测；
- conftest.py 只保留 pytest fixtures；SongSpec、LekaiServer、路径常量和 subprocess env 移到 tests/consistency/support.py；禁止普通模块 import conftest；
- 两个 server fixture 共享一个 context-managed process lifecycle，但各自模型配置保持显式，不做难读的万能 fixture。

### M3. Robustness/listening 大 fixture

tests/unit/scripts/conftest.py 的 robustness_fixture 一次构造 campaign、attestation、20-run qualification、frozen config 和 listening package，schema 小改会波及 6 个文件。

计划把纯构造逻辑迁到 tests/support/robustness_campaign.py，拆成：

- campaign_inputs；
- attestation_bundle；
- qualification_evidence；
- frozen_campaign；
- listening_package。

迁移期间保留 robustness_fixture 作为兼容 facade，逐文件切换。相似的 hash/tamper tests 不删除：qualification、execution、listening、report 分别验证不同 artifact trust boundary。

### M4. 动态脚本 loader

至少 5 个文件在 collection 阶段自行 spec_from_file_location，部分还残留 sys.modules：voice_microbench、perturb_melody、midi_to_npz、run_lekai_offline、zzz_memory_sweep。

短期统一使用 tests/unit/scripts/conftest.py 的隔离 loader；长期将业务逻辑移入 src/streammuse，scripts/*.py 只保留 argparse 薄壳，测试直接 import 正式模块。

## 审查结论：看似重复但必须保留

- Inference factory 的 Lekai frame 校验与 server endpoint 校验分别保护 Python 和 HTTP 两个入口；
- CatchUpState、scheduler、backend、client、server 的相似 case 分别保护纯状态、线程编排和传输边界；
- input timing 单测和 RealTimeMusicService timing 集成测试边界不同；
- domain event pairing、MIDI converter roll 语义、token-to-MIDI delay 处理三种不同表示；
- test_lekai_session_contract.py 验证 service → HTTP client → FastAPI → backend 的 metadata digest contract，必须保留；
- test_simulator_midi_output.py 是 service thread → factory sink → MIDI 落盘 → 读回的 hermetic E2E，必须保留；
- robustness campaign 的 fail-closed、hash、retry、tamper tests 虽然形式相似，但对应不同信任边界；
- tokenizer 的 empty/sparse/dense/multi-beat cases 结构边界不同；
- pytest gold consistency 和 15-song runner 服务不同运行场景，不能二选一。

## 分阶段实施计划

### Phase 0：冻结基线和 collection 边界

- [ ] 保存 tests/ 下全部 node IDs、skip reasons、warning 和 durations；基线为 639 unit、8 integration、3 consistency。
- [ ] 在 pyproject.toml 设置 testpaths=["tests"]、norecursedirs、strict_markers、xfail_strict 和 --strict-config。
- [ ] 注册 consistency、gpu、performance、workflow markers；目录仍作为 unit/integration 的主要分层。
- [ ] 确认无参数 uv run pytest --collect-only -qq 只收集 tests/，不进入 scripts/ 或 transformers/。
- [ ] 将已跟踪的 src/streammuse.egg-info 从版本控制移除并添加 *.egg-info/ ignore，确保测试/构建不污染工作树。
- [ ] 建立 warning budget：项目自身 warning 视为错误；pretty_midi/pkg_resources 只允许精确、带后续清理说明的临时豁免，不做全局 ignore。

### Phase 1：先修“假绿”和并发脆弱测试

- [ ] 重写 scheduler thread_safety：用 barrier/futures 传播线程异常，并断言 consumed + remaining 等于所有唯一提交事件。
- [ ] 重写 MIDI delay test：精确断言第一次 sleep 包含 delay_ticks，而不是只断言出现过 sleep。
- [ ] prompt protocol worker 使用 threading.Event/condition，不再依赖固定 time.sleep(0.15)。
- [ ] RealTimeMusicService thread smoke 使用 stopped event/有界等待，不再固定 sleep(0.01)。
- [ ] simulator integration 在 finally 中 stop 清理，但在清理前断言服务确实由 max_ticks 自然结束。
- [ ] server_lekai 测试改用 function-scoped backend/TestClient fixture，移除跨 test mutable history；把 >=1 改成精确断言。
- [ ] backend delegation 测试让 fake engine 返回 sentinel，验证结果确实被透传，同时 backend 正确覆盖 model name。
- [ ] CLI main tests 精确断言 SystemExit code，并隔离 atexit/signal 全局状态。

### Phase 2：执行高置信 unit/integration 精简

- [ ] 按 C3 表逐模块先增强保留测试，再删除约 18 个重复 unit cases；每个模块单独提交。
- [ ] 删除外部 RT MIDI integration test，确认日常 integration 变为 0 skip。
- [ ] 合并标准 CLI lifecycle 和 Rap wiring；保留 injection validation/order，删除弱 early-exit case。
- [ ] 执行 MusicalSequence 决策门：无 consumer 则删除源码、测试和过期 docs；决定保留则补实际 consumer contract。
- [ ] 合并 tokenizer、event payload、task trace helpers；helper 只放到最小可见目录。
- [ ] 不以“大文件太长”为理由同时改断言和移动文件；机械移动与行为修改分开提交。

### Phase 3：修复伪测试和旧测试脚本

- [ ] 将 scripts/test_logging_system.py 中 logger 落盘和 compare_logs 场景迁到 tests/unit/infrastructure/inference/test_generation_logger.py，并改用 assert。
- [ ] 删除永远返回 True 的 environment-variable 伪测试及原 scripts/test_logging_system.py。
- [ ] 统一 generation_logger.py 与 compare_generation_logs.py 的 token diff 实现；目录模式按明确 identity 配对并拒绝两侧数量/identity 不一致。
- [ ] 从活跃 scripts/ 删除 run_all_tests.sh、compare_all.sh、run_debug_round2.sh；历史背景留在 developing-logs，不保留会被误认为权威 gate 的入口。
- [ ] 删除或归档 import 时读取固定日志的 analyze_token_diff.py 和 compare_melody_tokens.py；保留一个参数化、无硬编码路径的 token forensic CLI。
- [ ] 将 run_lekai_fake_realtime.py 和 raw-event debug comparator 明确标记为 forensic/debug 工具，不再称为生产 Realtime equivalence test。
- [ ] 更新 docs/developer-guide/generation-logging.md 和相关命令。

### Phase 4：统一 consistency correctness 基础设施

- [ ] 新建公共 model-timestep comparator/validity 模块及纯 CPU unit tests。
- [ ] single-stage pytest 和 all-sample runner 都切换到公共实现，删除本地重复 comparator。
- [ ] 显式 checkpoint env 未设置时允许 skip；env 已设置但路径不存在/不可读必须 fail。
- [ ] 增加 STREAMMUSE_REQUIRE_CONSISTENCY=1：正式 gate 中任何 checkpoint、requests、GPU 或 real-model 条件缺失都 fail。
- [ ] requests 作为正式依赖，缺失时直接环境失败，不再 skip。
- [ ] server fixture 从 session scope 改为 module scope；single-stage 与 two-stage 模型不跨模块同时驻留。
- [ ] 调整运行顺序：先生成 Offline reference 并退出模型进程，再启动 Realtime server，避免同 GPU 多份模型共驻。
- [ ] 真正发送一次最小 generate 请求进行 CUDA/model warmup，再 reset/clear RNG 和 history；删除“clear_history 等于 warmup”的错误注释/实现。
- [ ] artifact 目录使用足够唯一的 ID；记录 checkpoint hash、代码 identity、采样参数和 runtime_info。
- [ ] 将 song、tempo 参数化成独立 node，提供稳定 IDs，允许单独 rerun 和 JUnit 报告。
- [ ] 删除两个冗余 cross-tempo 尾部断言。

### Phase 5：拆分 correctness、performance 和 two-stage 实际语义

- [ ] single-stage gold profile 使用慢 tempo 验证内容一致性；120 BPM 归入 performance marker，drop/late 失败不再与内容 regression 混报。
- [ ] 将 two-stage 当前测试重命名为 test_two_stage_server_history_matches_offline，明确它只比较 prompt/raw server history。
- [ ] schedule_counts 从“仅失败信息”升级为实际 validity assertion，或明确该 history test 不声称 playback 有效。
- [ ] 新增 two-stage playback artifact test：比较 combined.mid、scheduled events、有效播放窗口和 dropped/clipped policy。
- [ ] 如果暂时无法定义可靠的 playable-window oracle，则将 playback test 标记为未完成，文档不得称当前 history test 为 comprehensive E2E。
- [ ] 对齐 two-stage server/offline repetition penalty 等全部采样配置，即使 greedy 模式下当前不影响结果。
- [ ] 更新 consistency 文档中的默认歌曲数、预计时长、checkpoint 要求和输出工件。

### Phase 6：加固全 15 首 runner

- [ ] run_all_sample_consistency.py 不再内置个人绝对 checkpoint 路径；要求 --checkpoint 或明确 env。
- [ ] 同一 GPU 上 Offline dataset generation 串行；并发度按唯一 GPU 数限制，避免单 GPU 同时加载两份模型。
- [ ] --resume 必须绑定 checkpoint SHA、代码 identity、tempo、tail、采样配置、源 MIDI/NPZ hash 和 comparison schema；任一不符拒绝复用。
- [ ] 恢复已有 result 前重新验证 artifact 存在性和 hash，不能只看 passed=true 或文件数量。
- [ ] --samples 精确选择一首时，只生成所需 Offline reference，不隐式跑完整 dataset。
- [ ] runtime_info 必须确认 real model、checkpoint identity 和采样参数。
- [ ] summary 将 empty-to-empty 单列为 uninformative consistent，不与 non-empty evidence 合并计数。
- [ ] 为 manifest、resume、GPU scheduling、summary 和 artifact validation 补纯 CPU tests。

### Phase 7：整理 robustness/workflow tests

- [ ] 拆分分层 robustness fixture builder，保留兼容 facade，逐文件迁移。
- [ ] 将 1,928 行 robustness analysis/report 文件按生产职责机械拆为 analyzer、report inputs、listening/report、retry/audit 等模块；移动提交不改变断言。
- [ ] 给 frozen-config 的 lambda 参数添加 pytest.param(..., id=...)，避免 <lambda>0...21 的失败名。
- [ ] 给完整 triangle package、95-response QC、retry lineage、local HTTP durability 标记 workflow；comprehensive gate 始终运行，另提供明确的 fast profile。
- [ ] 不删除跨 qualification/execution/listening/report trust boundary 的 tamper tests。

### Phase 8：最终验证和报告

- [ ] uv run pytest --collect-only -qq --strict-markers --strict-config 成功，且只收集 tests/。
- [ ] uv run pytest -q tests/unit tests/integration -rs：全部通过、0 unexpected skip、0 project warning。
- [ ] 对修改过的线程/全局状态测试重复运行至少 20 次，确认没有 flaky failure。
- [ ] 无 checkpoint 的 consistency 命令只产生预期 resource skips；require 模式下相同环境必须失败而不是 skip。
- [ ] 真实 checkpoint single-stage gold 通过。
- [ ] 真实 checkpoint single-stage performance 独立出报告，不混入 gold correctness。
- [ ] two-stage history 与 playback 分别通过；若 playback oracle 尚未完成，明确记录未完成而非把 history pass 当 E2E pass。
- [ ] 用单 GPU smoke 验证 full runner 不并发 OOM，并验证 resume 对配置/hash 漂移 fail closed。
- [ ] 运行 60 BPM 全 15 首 comprehensive runner，保存 summary.json、summary.md 和命令/环境记录。
- [ ] 测试后 git status 不出现 egg-info、cache 或输出工件变化。
- [ ] 写 developing-logs/reports/2026-07-20-test-suite-consolidation-reliability-report.md，记录 before/after node 数、耗时、skip、warning、真实模型结果和所有有意删除项。

## 验证命令

日常 hermetic gate：

    uv run pytest --collect-only -qq --strict-markers --strict-config
    uv run pytest -q tests/unit tests/integration -rs

无模型资源时检查 skip/fail 分类：

    uv run pytest -q tests/consistency -rs
    STREAMMUSE_REQUIRE_CONSISTENCY=1 uv run pytest -q tests/consistency -rs

单阶段真实模型 gold：

    STREAMMUSE_CONSISTENCY_GPU=0 \
    LEKAI_CHECKPOINT_PATH=/absolute/path/model.safetensors \
    STREAMMUSE_CONSISTENCY_SONGS=4,5,2 \
    STREAMMUSE_CONSISTENCY_TEMPOS=15 \
    uv run pytest tests/consistency/test_realtime_offline_consistency.py -v -s

单阶段 performance（独立报告）：

    STREAMMUSE_CONSISTENCY_GPU=0 \
    LEKAI_CHECKPOINT_PATH=/absolute/path/model.safetensors \
    STREAMMUSE_CONSISTENCY_SONGS=4,5,2 \
    STREAMMUSE_CONSISTENCY_TEMPOS=120 \
    uv run pytest tests/consistency/test_realtime_offline_consistency.py -v -s -m performance

60 BPM 全样本：

    .venv/bin/python scripts/run_all_sample_consistency.py \
      --checkpoint /absolute/path/model.safetensors \
      --output-dir output/consistency/all_samples_full60 \
      --gpus 0,1,2,3,4,5,6,7 \
      --tempo 60

## Definition of Done

- 根目录 pytest collection 稳定、只收集本项目 tests/；
- 日常 unit/integration 没有永久 skip、伪 pass 或未隔离全局状态；
- 删除项都有更强保留测试或明确的死代码证据；
- 不同 trust boundary 的测试没有因为表面相似被错误合并；
- Offline vs Realtime 只有一个 model-timestep comparator 和一套 validity 语义；
- single-stage gold、performance、two-stage history、two-stage playback、all-sample runner 的名称和实际断言一致；
- 显式写错 checkpoint、缺失依赖、加载到 stub model 都会 fail，而不是正常 skip；
- full runner 的 resume 不能复用跨代码、跨 checkpoint、跨输入或跨配置的旧结果；
- 所有命令执行后工作树不被生成 metadata 或测试 artifact 污染；
- 最终报告逐项列出删除、合并、保留和仍未完成的真实模型验证。

## 风险控制

- 不在同一个提交中同时移动测试、修改生产逻辑和修改断言；
- 比较器从 beat 升级到 model timestep 后若暴露新失败，先归因，不得放宽 oracle 以恢复绿色；
- 不直接启用 xdist/random-order；先解决 server 全局状态、GPU 独占、artifact 唯一性和线程异常传播；
- shared fixture 只抽完全同构的 setup，语义不同的同名 helper 保持局部；
- 参数化必须提供稳定 id，不能以减少函数数为代价降低失败定位；
- 测试数量下降不是验收目标；任何无法证明被覆盖的 case 默认保留。
