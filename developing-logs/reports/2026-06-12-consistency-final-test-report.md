# Final Consistency Test 执行 Report（2026-06-25）

对应计划：`developing-logs/plans/2026-06-12-consistency-final-test-plan.md`（含 rev3 + Phase 0 实测记录 + 执行 Todo List）。

## 一句话结论

实现了金标准端到端一致性测试 `tests/consistency/`：贪心参数下，真实实时链路（`streammuse-cli` → `RealTimeMusicService` → HTTP → Lekai server）与离线一次性生成在 pianoroll 层**逐拍一致**。**song 4 默认配置（tempo 15 + 120）全绿。**

## 关键转折：前提验证不是直线

Phase 0 手动验证经历了一次"误判 → 更正"，这是本次最有价值的部分，全部记录在案：

1. **初次用 song 1 验证 → 两边伴奏都空 → 一度误判"前提不成立"**。深挖发现贪心解码下 song 1 的伴奏天然全空（141 拍全是 empty marker），而那个到处出现的"263 notes"一直是 **GT**（`save_gt_midi`），不是生成结果。
2. **用户澄清**："有的歌全空很常见，有一两首正常"。实测 5 首歌非空程度（附录 B）：song 4 最丰富（56/76 非空拍），song 5/2 次之，song 1/3 几乎全空。**默认歌从计划原定的 song 1 改为 song 4。**
3. **改用 song 4 后**：realtime 与 offline 在 pianoroll 层 100% 一致。前提成立。

## 三条踩坑换来的对比方法论（写进了 docs 和测试注释）

1. **只用非空歌**：贪心对多数歌生成空伴奏（已知模型行为）。空对空比较无意义。
2. **pianoroll(beat,pitch) 层对比，不用 raw MIDI note**：realtime 把持续音每拍重触发，offline 保持长音符；raw note 对比只有 ~78% 假性 mismatch，pianoroll 归一化后 100%。→ **推翻了计划原定"复用 debug_inference_consistency.py 的 raw note 对比"**。
3. **截断到旋律窗口**：旋律 MIDI 比 max-ticks 短，realtime 跑过头会在无旋律区继续挂音；截断到 `melody_last_beat` 后一致。另：`max-ticks` 尾部余量要够（2 拍不够会丢窗口内最后一格 (56,43)，24 拍 OK）——余量大无害，窗口截断会忽略多余尾部。

附带排除：滑窗 `LEKAI_PROMPT_CONTEXT_BEATS` 不是分叉源（全 context 与默认 32 拍结果相同）。

## 交付物

### 生产代码改动（最小、向后兼容）
- `src/streammuse/infrastructure/inference/lekai_model/model.py`：`generate_accompaniment` 加 `bpm_override: Optional[int] = None`（None 时行为不变）。
- `scripts/run_lekai_offline.py`：加 `--bpm` 透传。
- 作用：让 offline 的条件 BPM 与 realtime 显式对齐（两侧钉 120，落同一 encode_bpm 桶），消除历史上"BPM token 不一致导致整体分叉"的坑。

### 测试代码（新增 `tests/consistency/`）
- `midi_pianoroll.py`：pianoroll `(beat,pitch)` 归一化对比 + 窗口截断（`compare_accompaniment`）。
- `conftest.py`：门控（`LEKAI_CHECKPOINT_PATH` 未设即 skip）、`lekai_server` session fixture（空闲端口 / 确定性 env / health 轮询 / 预热 / teardown / 失败 dump log）、歌曲元数据（`SONG_TO_CONDITION_IDX` 映射、旋律长度、max-ticks）、工件目录。
- `runners.py`：`run_realtime`（驱动真实 CLI）、`run_offline`（驱动离线脚本）、`count_dropped_requests`（从 `generation_start_tick` 连续性检测丢弃）。
- `test_realtime_offline_consistency.py`：按歌参数化，歌内遍历 tempo；零丢弃前置 + 断言一（vs offline 100%）+ 断言二（tempo 间两两一致）。
- `pyproject.toml`：注册 `consistency` marker（日常 `uv run pytest tests/` 不受影响，显示 skipped）。

### 文档
- `docs/developer-guide/consistency-test.md` + VitePress sidebar 条目：怎么跑、三条方法论、BPM 约束、tempo 阶梯诊断语义、红了怎么查。

## 测试设计

- **门控**：`LEKAI_CHECKPOINT_PATH` 指向存在的 checkpoint 才跑，否则整目录 skip。
- **默认规模**：song 4 × tempo {15, 120}。`STREAMMUSE_CONSISTENCY_SONGS` / `_TEMPOS` / `_GPU` 可扩展/指定。
- **tempo 阶梯诊断**：tempo 15 红 = 真回归；tempo 15 绿 + 120 红 = 推理太慢（也会被零丢弃前置抓到）；全绿 = 一致且能跑满实时。
- **断言二**（tempo 间一致）：免费验证"时钟速度不泄漏进生成内容"。

## 真跑结果（GPU 单卡，float16）

- **快速验证（song 4, tempo 120 单档）**：PASSED，61.8s。
- **完整默认配置（song 4, tempo 15 + 120，含跨 tempo 断言）**：**PASSED，6m25s**。
- **日常套件**：173 passed + 1 skipped（consistency 被正确跳过），无影响。

## 判别力证据（测试非平凡，能真红）

1. **非确定性采样（canonical）**：server 用 `temp=0.8, top_k=2` 跑 realtime，对比贪心 offline → **4.32% match，`is_consistent=False`**。破坏确定性必红。
2. **实战 RED**：实现期 max-ticks 余量不足导致窗口内单格 `(56,43)` 缺失，测试**真的 FAILED**（98.11%）——证明断言能抓住单格差异。
3. **pianoroll 模块自检**：不加窗口截断时给出 77.8%（`is_consistent=False`），能检测尾部 mismatch。
4. **BPM 跨桶（意外发现）**：offline `--bpm 80`（慢桶，token 264）vs realtime `bpm 120`（中桶，token 265）→ song 4 仍 **100% 一致**。即该歌贪心输出对 BPM 桶不敏感。BPM 钉死仍是正确的防御（历史上确实踩过跨桶坑），只是对 song 4 不是判别维度——换歌/换 checkpoint 时此结论需重核。

## 计划与实现的偏差

1. **对比层级**：计划原定复用 `debug_inference_consistency.py` 的 raw note 对比（方案 A）。实测该方法因持续音表示差异只有 78%，**改为 pianoroll 归一化对比**，落在 `tests/consistency/midi_pianoroll.py`（未改 debug 脚本，二者用途已分离）。
2. **默认歌**：计划原定 song 1，实测全空，**改为 song 4**。
3. **server/offline 显存**：计划原想"先停 server 再跑 offline 避免双份占卡"。实测卡有 143GB，两者各 ~325MB 可共存，**省去停启编排**，offline 与 server 同时在 GPU 上跑。

## 遗留 / 后续

- 默认只跑 song 4；发版前建议 `SONGS=4,5,2 TEMPOS=120,90,60,15` 全量。
- `4.mid`（旋律止于 beat 57）与 `4.npz`（内容到 ~beat 60）长度略有出入；当前靠旋律窗口截断规避，未深究数据源对齐。
- 计划附录 A/B（BPM 对照表、各歌非空程度）保留在 plan 文件，供换 checkpoint 时复核。
