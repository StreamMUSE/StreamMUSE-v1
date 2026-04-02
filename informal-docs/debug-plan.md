# Lekai Client-Server Mode Debug Plan

Lekai model + HTTP client-server 模式下发现的 7 个问题及修复方案。

---

## 详细实施计划 (Detailed Todo List)

### Phase 0 — 准备与基线建立

**目标**: 建立测试基线，确保改动可追溯

- [ ] **P0-1** 创建 bug 复现测试环境
  - [ ] P0-1.1 编写最小复现脚本 `tests/reproduce_bug1_velocity_null.py`
  - [ ] P0-1.2 编写最小复现脚本 `tests/reproduce_bug4_full_history.py`
  - [ ] P0-1.3 记录当前代码基线 commit hash

- [ ] **P0-2** 补充缺失的单元测试桩
  - [ ] P0-2.1 创建 `tests/unit/infrastructure/inference/test_lekai_http_backend.py`（空文件带 skip 标记）
  - [ ] P0-2.2 创建 `tests/integration/test_lekai_end_to_end.py`（空文件带 skip 标记）

- [ ] **P0-3** 配置 CI 支持
  - [ ] P0-3.1 确保 `pytest tests/ -k "lekai" --collect-only` 能发现所有 lekai 相关测试
  - [ ] P0-3.2 在 CI 中添加 lekai 测试 job（允许失败，直到修复完成）

**Exit Criteria**: 
- [ ] P0-E1 能稳定复现所有 7 个 bugs
- [ ] P0-E2 测试基础设施就位

---

### Phase 1 — Critical Fixes（让 lekai stub 能跑起来）

**目标**: 修复致命和严重 bug，使 lekai client-server 模式基本可用

#### Task 1.1: Bug #1 — `velocity: null` 崩溃修复

- [ ] **1.1.1** 客户端防御性修复
  - [ ] 1.1.1.1 修改 `serialization.py:event_from_dict()`，添加 None 保护
  - [ ] 1.1.1.2 修改 `serialization.py:event_from_dict()`，channel 和 program 同样添加 None 保护
  - [ ] 1.1.1.3 添加单元测试：`test_event_from_dict_handles_null_velocity()`

- [ ] **1.1.2** 服务端修复
  - [ ] 1.1.2.1 修改 `lekai_http_backend.py:_generate_rule_based()`，note_off 添加 velocity=0
  - [ ] 1.1.2.2 修改清除 active pitches 分支，添加 velocity=0
  - [ ] 1.1.2.3 搜索所有创建 note_off 事件的地方，确保都有 velocity

- [ ] **1.1.3** 回归测试
  - [ ] 1.1.3.1 运行复现脚本确认崩溃已修复
  - [ ] 1.1.3.2 全量测试 `pytest tests/unit/infrastructure/inference/ -v`

#### Task 1.2: Bug #2 短期 — 规则 stub 逻辑修复

- [ ] **1.2.1** 删除 `_active_model_pitches` 相关代码
  - [ ] 1.2.1.1 删除 `__init__` 中的 `_active_model_pitches` 初始化
  - [ ] 1.2.1.2 删除 `_build_active_pitches` 方法
  - [ ] 1.2.1.3 删除 `inject_history` 中的 `_active_model_pitches` 相关逻辑
  - [ ] 1.2.1.4 删除 `clear_history` 中的 `_active_model_pitches` 清理
  - [ ] 1.2.1.5 删除 `_generate_rule_based` 中的 `_active_model_pitches` 跟踪逻辑

- [ ] **1.2.2** 修复 stub 生成逻辑
  - [ ] 1.2.2.1 重写 `_generate_rule_based`，移除 next_active 跟踪
  - [ ] 1.2.2.2 确保同一批次的 note_on/note_off 正确闭合
  - [ ] 1.2.2.3 添加返回结果排序（note_off 在 note_on 之前）

- [ ] **1.2.3** 代码清理
  - [ ] 1.2.3.1 运行 `flake8`/`ruff` 检查删除代码后的格式问题
  - [ ] 1.2.3.2 运行 `mypy` 类型检查

#### Task 1.3: Bug #3 — 默认参数对齐与文档

- [ ] **1.3.1** 改善错误提示
  - [ ] 1.3.1.1 修改 `inference_factory.py`，为 `generation_interval_ticks` 添加详细错误信息
  - [ ] 1.3.1.2 修改 `inference_factory.py`，为 `generation_length_frames` 添加详细错误信息
  - [ ] 1.3.1.3 添加单元测试 `test_lekai_factory_rejects_invalid_interval()`

- [ ] **1.3.2** 补充 CLI 文档
  - [ ] 1.3.2.1 修改 `docs/reference/cli-reference.md`，添加 Lekai 约束说明章节
  - [ ] 1.3.2.2 修改 `docs/getting-started/configuration.md`，在参数表添加 lekai 注释
  - [ ] 1.3.2.3 修改 `docs/user-guide/running-realtime.md`，添加 lekai 启动示例

- [ ] **1.3.3** 考虑自动修正（可选）
  - [ ] 1.3.3.1 评估是否将默认值改为 4（lekai）/ 2（stanley）的自动检测
  - [ ] 1.3.3.2 如不做自动修正，确保错误信息足够清晰

#### Task 1.4: Bug #7 — Server 启动入口

- [ ] **1.4.1** 添加 main 函数
  - [ ] 1.4.1.1 在 `server_lekai.py` 末尾添加 `main()` 函数
  - [ ] 1.4.1.2 添加 `if __name__ == "__main__": main()`
  - [ ] 1.4.1.3 支持通过环境变量配置 host/port

- [ ] **1.4.2** 验证启动
  - [ ] 1.4.2.1 测试 `python -m src.streammuse.infrastructure.inference.server_lekai`
  - [ ] 1.4.2.2 测试 `uvicorn src.streammuse.infrastructure.inference.server_lekai:app`

#### Phase 1 集成测试

- [ ] **P1-INT-1** 端到端冒烟测试
  - [ ] P1-INT-1.1 启动 lekai server
  - [ ] P1-INT-1.2 运行 `uv run streammuse-cli --model-name lekai --generation-interval-ticks 4`
  - [ ] P1-INT-1.3 按键测试，确认无崩溃
  - [ ] P1-INT-1.4 确认有音频输出（或 console 输出事件）

**Exit Criteria**:
- [ ] P1-E1 Bug #1 修复：velocity null 不再导致崩溃
- [ ] P1-E2 Bug #2 短期修复：stub 逻辑正确，无冗余 note_off
- [ ] P1-E3 Bug #3 修复：错误提示清晰，文档完整
- [ ] P1-E4 Bug #7 修复：server 可以独立启动
- [ ] P1-E5 端到端测试通过（手动）

---

### Phase 2 — Stub 行为优化（让 lekai stub 行为合理）

**目标**: 修复中等和低优先级 bug，使 stub 行为更接近真实模型

#### Task 2.1: Bug #4 — Client/Server 历史同步修复

- [ ] **2.1.1** Client 端增量发送
  - [ ] 2.1.1.1 在 `RealTimeMusicService.__init__` 添加 `_last_sent_index` 字段
  - [ ] 2.1.1.2 修改 `_tick_loop`，使用切片 `self._melody_history[self._last_sent_index:]`
  - [ ] 2.1.1.3 更新 `_last_sent_index` 追踪逻辑
  - [ ] 2.1.1.4 添加单元测试 `test_incremental_event_sending()`

- [ ] **2.1.2** Server 端 extend 而非替换
  - [ ] 2.1.2.1 修改 `lekai_http_backend.py:generate()`，将赋值改为 `extend`
  - [ ] 2.1.2.2 确保 `inject_history` 后 `generate` 正确追加（不覆盖）
  - [ ] 2.1.2.3 添加单元测试 `test_history_extend_not_replace()`

- [ ] **2.1.3** Stanley 兼容性检查
  - [ ] 2.1.3.1 检查 `stanley_legacy.py` 的 extend 行为
  - [ ] 2.1.3.2 确认 client 改增量后 Stanley 仍正常工作
  - [ ] 2.1.3.3 如有问题，修复 Stanley 适配

- [ ] **2.1.4** 端到端验证
  - [ ] 2.1.4.1 长 session 测试（持续 5 分钟）
  - [ ] 2.1.4.2 验证请求体大小恒定（不随时间增长）
  - [ ] 2.1.4.3 验证 inject_history 后历史不丢失

#### Task 2.2: Bug #5 — `generation_length_frames` 前瞻

- [ ] **2.2.1** 修改规则生成器
  - [ ] 2.2.1.1 修改 `_generate_rule_based` 签名，添加 `generation_length_frames` 参数
  - [ ] 2.2.1.2 实现多 interval 循环生成
  - [ ] 2.2.1.3 确保时间偏移计算正确（`offset = i * interval`）
  - [ ] 2.2.1.4 更新 `generate()` 方法传递新参数

- [ ] **2.2.2** 测试
  - [ ] 2.2.2.1 添加单元测试 `test_generation_length_respected()`
  - [ ] 2.2.2.2 验证生成长度与 `generation_length_frames` 成正比
  - [ ] 2.2.2.3 边界测试：`generation_length_frames < interval` 的情况

#### Task 2.3: Bug #6 — History 裁剪

- [ ] **2.3.1** 实现裁剪逻辑
  - [ ] 2.3.1.1 参考 `stanley_legacy.py` 裁剪逻辑
  - [ ] 2.3.1.2 在 `generate()` 中添加 melody_history 裁剪
  - [ ] 2.3.1.3 在 `generate()` 中添加 accompaniment_history 裁剪
  - [ ] 2.3.1.4 确定合适的 `max_history_ticks`（建议 2x generation_length_frames）

- [ ] **2.3.2** 内存测试
  - [ ] 2.3.2.1 编写内存监控脚本
  - [ ] 2.3.2.2 长 session（10 分钟）内存使用测试
  - [ ] 2.3.2.3 验证内存使用平稳，无泄漏

#### Phase 2 集成测试

- [ ] **P2-INT-1** 综合行为测试
  - [ ] P2-INT-1.1 完整 session 测试（inject + 多轮 generate）
  - [ ] P2-INT-1.2 验证前瞻缓冲正确（generation_length_frames 被尊重）
  - [ ] P2-INT-1.3 长 session 稳定性测试（10 分钟无崩溃）

**Exit Criteria**:
- [ ] P2-E1 Bug #4 修复：client 发增量，server 正确追加
- [ ] P2-E2 Bug #5 修复：generation_length_frames 生效
- [ ] P2-E3 Bug #6 修复：history 裁剪，内存稳定
- [ ] P2-E4 所有单元测试通过
- [ ] P2-E5 长 session 稳定性测试通过

---

### Phase 3 — 真实 PianoLLaMA 模型集成（长期方案）

**目标**: 接入真实模型，彻底替代规则 stub

#### Task 3.1: 模型接口改造

- [ ] **3.1.1** 改造 `PianoLLaMA.generate_accompaniment()`
  - [ ] 3.1.1.1 分析当前接口（从 NPZ/Dataset 读取）
  - [ ] 3.1.1.2 设计新的 `part0_beats` 参数接口
  - [ ] 3.1.1.3 实现支持 beat tokens 列表作为输入的重载
  - [ ] 3.1.1.4 保持向后兼容（原有接口仍可用）
  - [ ] 3.1.1.5 添加单元测试验证两种输入方式结果一致

- [ ] **3.1.2** 增量生成支持（KV Cache）
  - [ ] 3.1.2.1 分析当前 `generate_accompaniment` 的 KV cache 使用
  - [ ] 3.1.2.2 设计增量生成接口（接受 past_key_values）
  - [ ] 3.1.2.3 实现流式/增量生成模式
  - [ ] 3.1.2.4 性能测试：对比全量 vs 增量生成速度

#### Task 3.2: LekaiHttpBackend 集成模型

- [ ] **3.2.1** 初始化与加载
  - [ ] 3.2.1.1 在 `__init__` 中添加模型组件初始化
  - [ ] 3.2.1.2 实现 `_load_model()` 方法
  - [ ] 3.2.1.3 支持 checkpoint 路径配置（环境变量/参数）
  - [ ] 3.2.1.4 优雅处理模型加载失败（fallback 到规则 stub）

- [ ] **3.2.2** 转换管道实现
  - [ ] 3.2.2.1 events → pianoroll: 使用 `MidiConverter.events_to_pianoroll()`
  - [ ] 3.2.2.2 pianoroll → beat tokens: 实现 `_pianoroll_to_beat_tokens()`
  - [ ] 3.2.2.3 tokens → 模型推理: 调用 `PianoLLaMA.generate_accompaniment()`
  - [ ] 3.2.2.4 tokens → pianoroll: 使用 `process_part_beats_to_pianoroll()`
  - [ ] 3.2.2.5 pianoroll → events: 使用 `MidiConverter.pianoroll_to_events()`

- [ ] **3.2.3** 持续音跟踪
  - [ ] 3.2.3.1 使用 `_active_pitches` 跟踪跨请求的持续音
  - [ ] 3.2.3.2 确保 `pianoroll_to_events` 的 `active_pitches` 参数正确使用
  - [ ] 3.2.3.3 处理 retrigger 情况（同一 pitch 的新 onset）

- [ ] **3.2.4** Timing 记录
  - [ ] 3.2.4.1 确保所有阶段时间戳正确记录
  - [ ] 3.2.4.2 preprocess_start / inference_start / inference_end / postprocess_start
  - [ ] 3.2.4.3 与 HTTP response 的 timings 字段对齐

#### Task 3.3: 集成测试与验证

- [ ] **3.3.1** 单元测试
  - [ ] 3.3.1.1 测试转换管道各阶段输出 shape 正确
  - [ ] 3.3.1.2 测试模型生成结果非空且格式正确
  - [ ] 3.3.1.3 测试持续音跨请求保持

- [ ] **3.3.2** 端到端测试
  - [ ] 3.3.2.1 使用真实 checkpoint（如可用）进行端到端测试
  - [ ] 3.3.2.2 对比规则 stub 与真实模型的输出差异
  - [ ] 3.3.2.3 时序正确性：音符时值由模型决定，非写死

- [ ] **3.3.3** 性能基准
  - [ ] 3.3.3.1 测量单次推理延迟（p95/p99）
  - [ ] 3.3.3.2 确认延迟满足实时性要求（< generation_interval_ticks）
  - [ ] 3.3.3.3 GPU 内存使用监控

#### Task 3.4: 配置与文档

- [ ] **3.4.1** 配置支持
  - [ ] 3.4.1.1 添加 `LEKAI_CHECKPOINT_PATH` 环境变量支持
  - [ ] 3.4.1.2 添加 CLI 参数 `--lekai-checkpoint`
  - [ ] 3.4.1.3 更新 `InferenceConfig` 添加 checkpoint_path 字段

- [ ] **3.4.2** 文档更新
  - [ ] 3.4.2.1 更新 `docs/developer-guide/adding-inference-engine.md`
  - [ ] 3.4.2.2 添加模型集成架构图
  - [ ] 3.4.2.3 编写模型下载与配置指南

**Exit Criteria**:
- [ ] P3-E1 `PianoLLaMA.generate_accompaniment()` 支持 beat tokens 输入
- [ ] P3-E2 LekaiHttpBackend 可选择使用真实模型或规则 stub
- [ ] P3-E3 端到端测试通过：events → pianoroll → tokens → 模型 → tokens → pianoroll → events
- [ ] P3-E4 音符时值由模型输出决定（非写死）
- [ ] P3-E5 实时性满足要求（推理延迟 < interval）

---

### Phase 4 — 回归测试与发布准备

#### Task 4.1: 全面回归测试

- [ ] **4.1.1** Stanley 模式回归
  - [ ] 4.1.1.1 运行 Stanley 全量单元测试
  - [ ] 4.1.1.2 Stanley client-server 端到端测试
  - [ ] 4.1.1.3 确认 client 改增量发送后 Stanley 仍正常

- [ ] **4.1.2** Lekai 模式测试
  - [ ] 4.1.2.1 规则 stub 模式全量测试
  - [ ] 4.1.2.2 真实模型模式测试（如 checkpoint 可用）
  - [ ] 4.1.2.3 混合场景测试（inject + generate + clear）

- [ ] **4.1.3** 边界情况
  - [ ] 4.1.3.1 空输入处理
  - [ ] 4.1.3.2 超长音符（跨多个 generation window）
  - [ ] 4.1.3.3 快速按键（短时间内大量事件）

#### Task 4.2: 代码质量与文档

- [ ] **4.2.1** 代码审查
  - [ ] 4.2.1.1 类型注解完整性检查（`mypy --strict`）
  - [ ] 4.2.1.2 代码格式检查（`ruff format --check`）
  - [ ] 4.2.1.3 导入排序检查（`isort --check-only`）

- [ ] **4.2.2** 文档完整性
  - [ ] 4.2.2.1 更新 CHANGELOG.md
  - [ ] 4.2.2.2 更新 README.md 中的 lekai 使用说明
  - [ ] 4.2.2.3 确认所有 TODO/FIXME 已处理或转为 issue

- [ ] **4.2.3** 性能报告
  - [ ] 4.2.3.1 生成 Phase 1/2/3 性能对比报告
  - [ ] 4.2.3.2 记录已知限制（显存需求、最大序列长度等）

#### Task 4.3: 发布检查

- [ ] **4.3.1** 版本标记
  - [ ] 4.3.1.1 确定版本号（建议 minor bump：lekai 支持是大功能）
  - [ ] 4.3.1.2 创建 git tag

- [ ] **4.3.2** 回滚方案
  - [ ] 4.3.2.1 记录回滚步骤
  - [ ] 4.3.2.2 确认旧版本（stanley-only）仍可运行

**Exit Criteria**:
- [ ] P4-E1 所有现有测试通过（无回归）
- [ ] P4-E2 代码质量检查通过
- [ ] P4-E3 文档完整更新
- [ ] P4-E4 性能基准达标

---

## 附录：测试矩阵

| 测试场景 | Phase 1 | Phase 2 | Phase 3 | Phase 4 |
|---------|---------|---------|---------|---------|
| Bug #1 velocity null 崩溃 | ✅ | - | - | ✅ 回归 |
| Bug #2 stub 逻辑 | ✅ 短期 | ✅ | ✅ 长期 | ✅ 回归 |
| Bug #3 参数约束 | ✅ | - | - | ✅ 回归 |
| Bug #4 历史同步 | - | ✅ | - | ✅ 回归 |
| Bug #5 generation_length | - | ✅ | - | ✅ 回归 |
| Bug #6 history 裁剪 | - | ✅ | - | ✅ 回归 |
| Bug #7 server 入口 | ✅ | - | - | ✅ 回归 |
| Stanley 兼容性 | - | ✅ | - | ✅ 回归 |
| 真实模型推理 | - | - | ✅ | ✅ 回归 |
| 长 session 稳定性 | - | ✅ | ✅ | ✅ |

---

## 附录：风险与缓解

| 风险 | 概率 | 影响 | 缓解措施 |
|-----|------|-----|---------|
| Phase 1 改动影响 Stanley | 中 | 高 | 4.1.1 Stanley 专项回归测试 |
| client 改增量后其他 server 不兼容 | 低 | 高 | 保持 HTTP 协议不变，server 端已经是 extend |
| PianoLLaMA 接口改造破坏原有功能 | 中 | 高 | 3.1.1.4 向后兼容，3.1.1.5 结果一致性测试 |
| 真实模型推理太慢不满足实时 | 中 | 高 | 3.1.2 KV cache 优化，3.3.3 性能基准 |
| checkpoint 加载失败 | 低 | 中 | 3.2.1.4 fallback 到规则 stub |

---

## Bug #1 (致命): `velocity: null` 导致客户端 `int(None)` 崩溃

### 问题定位

**服务端** `lekai_http_backend.py:185-189` 生成 note_off 时不带 velocity:

```python
accompaniment.append(
    {
        "type": "note_off",
        "pitch": int(model_pitch),
        "tick": note_off_tick,
        # 没有 velocity 字段
    }
)
```

**服务端** `server_lekai.py:21-22` Pydantic model 定义:

```python
class AccompanimentNoteEvent(BaseModel):
    type: str
    pitch: int
    tick: int
    velocity: Optional[int] = None  # None → JSON null
```

FastAPI 默认 `response_model_exclude_none=False`，序列化结果:
```json
{"type": "note_off", "pitch": 48, "tick": 6, "velocity": null}
```

**客户端** `serialization.py:35` 反序列化:

```python
velocity=int(d.get("velocity", 100))
# d.get("velocity", 100) → key 存在但值为 None → 返回 None（不走默认值 100）
# int(None) → TypeError!
```

### 修复方案

双端修复，防御性编程:

**A. 客户端 `serialization.py:event_from_dict()`** — 加 None 保护:

```python
def event_from_dict(d: Dict[str, Any]) -> MusicalEvent:
    return MusicalEvent(
        tick=int(d.get("tick", 0)),
        pitch=int(d.get("pitch", -1)),
        event_type=EventType(str(d.get("type"))),
        velocity=int(d["velocity"]) if d.get("velocity") is not None else 100,
        channel=int(d["channel"]) if d.get("channel") is not None else 0,
        program=int(d["program"]) if d.get("program") is not None else 0,
        is_placeholder=bool(d.get("is_placeholder", False)),
    )
```

**B. 服务端 `lekai_http_backend.py:_generate_rule_based()`** — note_off 也带 velocity:

```python
accompaniment.append(
    {
        "type": "note_off",
        "pitch": int(model_pitch),
        "tick": note_off_tick,
        "velocity": 0,  # MIDI 标准: note_off velocity = 0
    }
)
```

同样修复清除 active pitches 的分支 (第 158-164 行):

```python
events.append(
    {
        "type": "note_off",
        "pitch": int(pitch),
        "tick": int(generation_start_tick),
        "velocity": 0,
    }
)
```

---

## Bug #2 (严重): Lekai 模型未集成，规则 stub 写死音符时值

### 问题定位

`lekai_http_backend.py:_generate_rule_based()` **根本不是模型推理**，而是一个硬编码的规则 stub：

```python
# lekai_http_backend.py:170-171
chord_root_tick = int(generation_start_tick)
note_off_tick = int(generation_start_tick + interval)  # 写死 = 1 个 interval

for pitch in unique:
    model_pitch = max(36, pitch - 12)            # 写死: 旋律下移八度
    accompaniment.append({"type": "note_on",  "tick": chord_root_tick})
    accompaniment.append({"type": "note_off", "tick": note_off_tick})  # 写死: 时值 = interval
```

三个根本问题：
1. **音符时值写死** — 每个音的 note_off = `generation_start_tick + interval`，不是模型决定的
2. **音高写死** — `max(36, pitch - 12)`，只是旋律下移八度，不是模型生成的和声/伴奏
3. **`_active_model_pitches` 跟踪错误** — 同一批次的 note_on/note_off 仍被标记为 active，导致下一次调用产生冗余 note_off

但实际上，lekai 模型（PianoLLaMA）和所有转换工具**已经在仓库中了**，只是没有接入 `LekaiHttpBackend`：

| 组件 | 文件 | 状态 |
|------|------|------|
| PianoLLaMA 模型 | `lekai_model/model.py` | 已实现，`generate_accompaniment()` 可用 |
| Token 编解码 | `lekai_model/my_tokenizer.py` | 已实现 |
| Token → pianoroll → MIDI | `lekai_model/Token2Midi.py` | 已实现 |
| Event ↔ Pianoroll 转换 | `lekai_model/MidiConverter.py` | **已实现**，关键桥梁 |
| 采样策略 | `lekai_model/generation_utils.py` | 已实现 |
| 配置 | `lekai_model/config.py` | 已实现 |
| **HTTP 后端** | `lekai_http_backend.py` | **未接入模型，用 stub** |

### 修复方案：接入真实模型

分两步走：短期修复 stub 的附带 bug，长期集成真模型。

#### 短期：修复 stub 自身的逻辑错误

在真模型集成之前，让 stub 至少行为正确（不产生冗余 note_off、不崩溃）：

```python
def _generate_rule_based(self, generation_start_tick: int, generation_interval_ticks: int) -> List[EventPayload]:
    interval = max(1, int(generation_interval_ticks))
    window_start = generation_start_tick - interval * 2
    recent_on_pitches: List[int] = []

    for event in self._melody_history:
        event_type = str(event.get("type", ""))
        tick = int(event.get("tick", 0))
        pitch = int(event.get("pitch", -1))
        if pitch < 0:
            continue
        if event_type == "note_on" and window_start <= tick <= generation_start_tick:
            recent_on_pitches.append(pitch)

    unique: List[int] = []
    seen: Set[int] = set()
    for pitch in reversed(recent_on_pitches):
        if pitch in seen:
            continue
        seen.add(pitch)
        unique.append(pitch)
        if len(unique) >= 4:
            break

    if not unique:
        return []  # 删除 _active_model_pitches 清理分支，因为不再跟踪

    accompaniment: List[EventPayload] = []
    chord_root_tick = int(generation_start_tick)
    note_off_tick = int(generation_start_tick + interval)

    for pitch in unique:
        model_pitch = max(36, pitch - 12)
        accompaniment.append(
            {"type": "note_on", "pitch": int(model_pitch), "tick": chord_root_tick, "velocity": 80}
        )
        accompaniment.append(
            {"type": "note_off", "pitch": int(model_pitch), "tick": note_off_tick, "velocity": 0}
        )
        # 删除 next_active.add() — 同一批次已闭合，不需要跟踪

    # 删除 self._active_model_pitches 赋值
    accompaniment.sort(key=lambda e: (int(e["tick"]), 0 if str(e["type"]) == "note_off" else 1))
    return accompaniment
```

同时删除类中所有 `_active_model_pitches` 相关代码（`__init__`、`inject_history`、`clear_history`、`_build_active_pitches`），因为 stub 不需要这个概念。

#### 长期：集成真实 PianoLLaMA 模型

核心转换管道：

```
客户端 note_on/note_off 事件
    ↓  MidiConverter.events_to_pianoroll()
(2, 88, T) pianoroll (sustain + onset)
    ↓  PianoRollTokenizer.encode()
beat-level token sequence
    ↓  PianoLLaMA.generate_accompaniment()  (part0 注入, part1 生成)
生成的 part1 beat tokens
    ↓  PianoRollTokenizer.decode() / process_part_beats_to_pianoroll()
(2, 88, T) pianoroll
    ↓  MidiConverter.pianoroll_to_events()
note_on/note_off 事件 (带模型决定的时值!)
```

**关键：`MidiConverter` 已经实现了 event ↔ pianoroll 的精确双向转换**，而且 `pianoroll_to_events()` 通过 sustain 通道的下降沿来确定 note_off 时刻——音符的时值完全由模型的 pianoroll 输出决定，不是写死的。

具体实现：

```python
# lekai_http_backend.py — 集成真模型后的 generate()

import torch
from streammuse.infrastructure.inference.lekai_model.model import PianoLLaMA
from streammuse.infrastructure.inference.lekai_model.config import ModelConfig
from streammuse.infrastructure.inference.lekai_model.MidiConverter import MidiConverter
from streammuse.infrastructure.inference.lekai_model.my_tokenizer import PianoRollTokenizer
from streammuse.infrastructure.inference.lekai_model.PianoDataset import process_measure_with_beat_interleaving


class LekaiHttpBackend:
    def __init__(self, checkpoint_path: str | None = None) -> None:
        self._melody_history: List[EventPayload] = []
        self._accompaniment_history: List[EventPayload] = []
        self._injection_length_ticks: int = 0
        self._runtime_config: Optional[BackendRuntimeConfig] = None

        # 模型组件
        self._converter = MidiConverter(ticks_per_beat=4)
        self._tokenizer = PianoRollTokenizer(patch_h=1, patch_w=4)
        self._model: PianoLLaMA | None = None
        self._active_pitches: set[int] = set()  # 跨 beat 的持续音跟踪

        if checkpoint_path:
            self._load_model(checkpoint_path)

    def _load_model(self, checkpoint_path: str) -> None:
        """加载 PianoLLaMA 模型"""
        from streammuse.infrastructure.inference.lekai_model.inference import load_model
        model_config = ModelConfig()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = load_model(
            model_path=checkpoint_path,
            model_config=model_config,
            device=device,
        )

    def generate(self, melody_events, generation_start_tick, generation_length_frames,
                 generation_interval_ticks, ...) -> tuple[List[EventPayload], TimingPayload]:
        request_arrival = time.perf_counter()

        if self._model is None:
            # 模型未加载时 fallback 到规则生成（保持向后兼容）
            return self._generate_rule_based(...)

        # ——— 真模型推理路径 ———

        # 1. 用户事件 → pianoroll (melody = part0)
        #    MidiConverter.events_to_pianoroll() 会正确处理跨窗口的持续音
        prompt_start = max(0, generation_start_tick - generation_length_frames)
        melody_pianoroll = self._converter.events_to_pianoroll(
            events=self._melody_history,
            start_tick=prompt_start,
            end_tick=generation_start_tick,
            active_pitches=None,
        )
        # melody_pianoroll shape: (2, 88, prompt_length)

        # 2. pianoroll → beat tokens (part0)
        #    每 4 个时间步 = 1 beat，tokenizer 编码
        part0_beats = self._pianoroll_to_beat_tokens(melody_pianoroll)

        inference_start = time.perf_counter()

        # 3. PianoLLaMA 生成 part1 tokens
        #    模型逐 token 自回归生成，part0 注入，part1 预测
        device = "cuda" if torch.cuda.is_available() else "cpu"
        with torch.no_grad():
            result = self._model.generate_accompaniment(
                dataset=None,  # 不从 dataset 读取，直接用 beat tokens
                part0_beats=part0_beats,
                temperature=0.8,
                top_k=50,
                top_p=0.95,
                device=device,
            )

        inference_end = time.perf_counter()

        # 4. part1 tokens → pianoroll
        part1_pianoroll = process_part_beats_to_pianoroll(
            result['part1_beats'],
            tokenizer=self._tokenizer,
        )
        # part1_pianoroll shape: (2, 88, T_generated)

        # 5. pianoroll → note_on/note_off 事件
        #    关键：pianoroll_to_events() 用 sustain 通道的下降沿确定 note_off
        #    音符时值由模型输出决定，不是写死的
        events, self._active_pitches = self._converter.pianoroll_to_events(
            pianoroll=part1_pianoroll,
            start_tick=generation_start_tick,  # 绝对 tick 偏移
            close_at_end=False,                # 不在窗口末尾强制关闭
            active_pitches=self._active_pitches,
        )

        response_output = time.perf_counter()
        timings = {
            "request_arrival_time": request_arrival,
            "response_output_time": response_output,
            "preprocess_start_time": request_arrival,
            "inference_start_time": inference_start,
            "inference_end_time": inference_end,
            "postprocess_start_time": inference_end,
        }
        return events, timings
```

#### MidiConverter 作为桥梁的关键作用

`MidiConverter`（已实现）提供了事件 ↔ pianoroll 的精确双向转换：

**events → pianoroll** (`events_to_pianoroll`, 第 185-278 行):
```python
# 输入: note_on/note_off 事件流 + 窗口范围 + 跨窗口持续音
# 输出: (2, 88, T) pianoroll
# 自动处理:
#   - 时间窗口裁剪 (只取 start_tick ~ end_tick)
#   - active_pitches 跨窗口持续 (上一拍还在响的音)
#   - note_on 标记 onset 通道, sustain 从 note_on 延续到 note_off
```

**pianoroll → events** (`pianoroll_to_events`, 第 280-375 行):
```python
# 输入: (2, 88, T) pianoroll + start_tick 偏移 + 上一拍的 active_pitches
# 输出: (事件列表, 新的 active_pitches)
# 自动处理:
#   - sustain 下降沿 → note_off (音符时值由 sustain 通道决定!)
#   - onset 通道 → note_on
#   - 跨 beat 的持续音: active_pitches 传入/传出
#   - retrigger: 同一 pitch 的新 onset 前先发 note_off
```

这样，音符的时值完全由模型输出的 pianoroll sustain 通道决定，而不是写死 `generation_start_tick + interval`。

#### 集成时需要注意的问题

1. **`generate_accompaniment()` 接口适配**: 当前 `model.py` 的 `generate_accompaniment()` 从 NPZ 文件/Dataset 读取输入，需要改造为接受 beat tokens 列表作为 part0 输入
2. **时间分辨率对齐**: lekai 模型的 1 timestep = 1/16 音符，StreamMUSE 的 1 tick = 1/4 beat（默认 `ticks_per_beat=4`），两者恰好匹配（4/4 拍下 1 beat = 4 个 16 分音符 = 4 ticks）
3. **实时性**: `generate_accompaniment()` 是全量生成（几百个 beat），实时场景需要改造为增量生成（只生成未来几个 beat），利用 KV cache
4. **`_active_pitches` 跨调用持续**: `pianoroll_to_events()` 返回的 `new_active_pitches` 需要在 backend 中保存，下次调用时传入，确保跨请求的持续音正确处理

---

## Bug #3 (中等): 默认参数不满足 lekai 约束，且文档未说明

### 问题定位

`application/config/models.py:50`:
```python
generation_interval_ticks: int = 2  # 默认 2
```

`application/factories/inference_factory.py:22-25`:
```python
if cfg.model_name == "lekai":
    if int(cfg.generation_interval_ticks) % 4 != 0:
        raise ValueError("lekai requires generation_interval_ticks to be a multiple of 4")
```

用户运行 `uv run streammuse-cli --model-name lekai` 直接报错，没有任何提示该用什么值。

**文档缺失**：检查了 `docs/` 下所有相关文件：
- `docs/reference/cli-reference.md` — 列出了 `--model-name lekai` 和示例命令（恰好用了 `--generation-interval-ticks 4`），但**没有解释为什么必须是 4 的倍数**
- `docs/getting-started/configuration.md` — 推理参数表中列出 `--generation-interval-ticks` 默认值 2，**完全没提 lekai 约束**
- `docs/user-guide/running-realtime.md` — 没有 lekai 相关内容
- `docs/developer-guide/adding-inference-engine.md` — 以 lekai 为例但没提参数约束
- `docs/architecture/application/factories.md` — InferenceEngineFactory 说明中没提 lekai 校验逻辑

### 修复方案

**两件事**：改善错误提示 + 补文档。

#### A. 改善 factory 中的错误信息

保留校验报错（不自动修正，让用户显式指定），但给出清晰的提示：

```python
# inference_factory.py

if cfg.type == "http":
    if cfg.model_name == "lekai":
        if int(cfg.generation_interval_ticks) % 4 != 0:
            raise ValueError(
                f"lekai model requires --generation-interval-ticks to be a multiple of 4 "
                f"(got {cfg.generation_interval_ticks}). "
                f"Recommended: --generation-interval-ticks 4"
            )
        if int(cfg.generation_length_frames) % 4 != 0:
            raise ValueError(
                f"lekai model requires --generation-length-frames to be a multiple of 4 "
                f"(got {cfg.generation_length_frames}). "
                f"Recommended: --generation-length-frames 20"
            )
    ...
```

#### B. 补充文档

**B1. `docs/reference/cli-reference.md`** — 在推理参数表下方加 lekai 约束说明：

在推理参数表后面添加：

```markdown
### Lekai 模型参数约束

使用 `--model-name lekai` 时，以下参数必须是 **4 的倍数**（lekai 模型以 4 个 timestep 为一拍进行 tokenization）：

| 参数 | 约束 | 推荐值 |
|---|---|---|
| `--generation-interval-ticks` | 必须是 4 的倍数 | `4` |
| `--generation-length-frames` | 必须是 4 的倍数 | `20` |

不满足约束时 CLI 会报错并提示推荐值。
```

**B2. `docs/getting-started/configuration.md`** — 在推理配置表中加注释：

在推理配置表的 `--generation-interval-ticks` 和 `--generation-length-frames` 行的说明列追加注释，并在表后添加：

```markdown
> **注意**：使用 `--model-name lekai` 时，`--generation-interval-ticks` 和 `--generation-length-frames` 必须是 4 的倍数。详见 [CLI 参考](../reference/cli-reference.md#lekai-模型参数约束)。
```

**B3. `docs/user-guide/running-realtime.md`** — 添加 lekai server 启动说明：

```markdown
## 使用 Lekai 模型

```bash
# 终端 1：启动 Lekai 推理服务器
uvicorn src.streammuse.infrastructure.inference.server_lekai:app \
    --host 0.0.0.0 --port 8000

# 终端 2：启动 CLI 客户端
# 注意：lekai 要求 generation-interval-ticks 和 generation-length-frames 为 4 的倍数
uv run streammuse-cli \
    --input-mode keyboard \
    --model-name lekai \
    --generation-interval-ticks 4 \
    --generation-length-frames 20
```
```

---

## Bug #4 (中等): client 发全量历史 + server 替换而非追加，与设计意图不符

### 设计意图

> Server 端保存 history，client 每次请求只发送**新增**的事件。

### 问题定位：两端都不符合设计

**问题 A — Client 端发送全量历史而非增量**

`real_time_music_service.py:206-209`：

```python
# _tick_loop 触发推理时
with self._melody_history_lock:
    melody_snapshot = self._melody_history.copy()    # ← 每次发全量累积的事件
if melody_snapshot:
    self._inference_request_queue.put((tick, melody_snapshot))
```

`_melody_history` 从 session 开始就 append，从不清理，所以每次请求都把**所有历史事件**发给 server。随着 session 进行，请求体越来越大。

**问题 B — Server 端替换而非追加**

`lekai_http_backend.py:58`：

```python
def generate(self, melody_events, ...):
    self._melody_history = list(melody_events)    # ← 替换！不是 extend
```

如果 client 只发增量，这里替换就会丢失之前的历史。如果 client 发全量（当前行为），替换倒是"凑巧"正确——但两端都偏离了设计意图。

**问题 C — inject_history 后被 generate 覆盖**

```python
def inject_history(self, melody_events, ...):
    self._melody_history = list(melody_events)       # 注入基线历史

def generate(self, melody_events, ...):
    self._melody_history = list(melody_events)       # 第一次 generate 就覆盖了注入数据
```

注入的旋律历史在第一次 `generate()` 调用时就丢失了。

### 修复方案

Client 和 Server 两端同时修改，对齐到 "server 保存 history，client 只发增量" 的设计。

#### A. Client 端：只发上次请求之后的新增事件

```python
# real_time_music_service.py

class RealTimeMusicService:
    def __init__(self, ...):
        ...
        self._melody_history: List[MusicalEvent] = []
        self._melody_history_lock = threading.Lock()
        self._last_sent_index: int = 0    # 新增：追踪上次发送到哪个位置

    def _tick_loop(self, *, max_ticks: Optional[int]) -> None:
        ...
        while self._running:
            ...
            # Trigger inference at generation intervals.
            if tick - last_generation_tick >= self._generation_interval_ticks:
                with self._melody_history_lock:
                    # 只发送上次请求之后新增的事件
                    new_events = self._melody_history[self._last_sent_index:]
                    self._last_sent_index = len(self._melody_history)
                if new_events:
                    self._inference_request_queue.put((tick, new_events))
                    last_generation_tick = tick
            ...
```

这样每次请求只包含自上次请求以来新增的事件，请求体大小恒定（不随 session 增长）。

#### B. Server 端：extend 而非替换

```python
# lekai_http_backend.py

def generate(self, melody_events, ...):
    request_arrival = time.perf_counter()
    preprocess_start = request_arrival

    self.configure(...)

    # 追加新事件到已有历史（包括 inject 的基线数据）
    self._melody_history.extend(melody_events)

    inference_start = time.perf_counter()
    accompaniment = self._generate_rule_based(
        generation_start_tick=int(generation_start_tick),
        generation_interval_ticks=int(generation_interval_ticks),
    )
    inference_end = time.perf_counter()

    self._accompaniment_history.extend(accompaniment)
    response_output = time.perf_counter()

    timings = { ... }
    return accompaniment, timings
```

#### C. inject_history 保持现有语义（设置基线）

`inject_history` 的行为不变——它替换整个历史作为基线。后续 `generate()` 调用只追加，不再覆盖：

```python
def inject_history(self, melody_events, accompaniment_events, injection_length_ticks):
    # 设置基线历史（替换是正确的——这是初始化）
    self._melody_history = list(melody_events)
    self._accompaniment_history = list(accompaniment_events)
    self._injection_length_ticks = int(injection_length_ticks)
    ...
```

调用时序：
```
inject_history(baseline_melody)    → _melody_history = [baseline]
generate(new_events_1)             → _melody_history = [baseline, new_1, new_2, ...]
generate(new_events_2)             → _melody_history = [baseline, new_1, ..., new_N, ...]
```

#### D. 对 Stanley 的影响检查

Stanley 的 `LegacyInferenceEngineStanley` 也用 extend：

```python
# stanley_legacy.py:147
self.melody_history.extend(absolute_melody)   # ← 已经是 extend
```

所以 Stanley 本身就符合增量追加的设计。但 client 当前发全量，Stanley 端会重复追加相同的事件。修复 client 后 Stanley 也要检查兼容性：

```python
# stanley_legacy.py:generate_accompaniment()
# 当前代码直接 extend，如果 client 从全量改为增量，这里不需要改动
# 但要确认注入 offset 的处理逻辑仍然正确
```

#### E. HttpInferenceClient 无需修改

`HttpInferenceClient` 只是透传 `melody_events`，不关心是全量还是增量：

```python
# http_client.py:48-49
payload = {
    "melody_notes": [event_to_dict(e) for e in melody_events],
    ...
}
```

client 传什么它就发什么，不需要改。

---

## Bug #5 (低): `generation_length_frames` 被规则生成器忽略

### 问题定位

`lekai_http_backend.py:131`:

```python
def _generate_rule_based(self, generation_start_tick, generation_interval_ticks):
    note_off_tick = int(generation_start_tick + interval)
    # 只生成 1 个 interval 的伴奏，完全忽略 generation_length_frames
```

Stanley 生成 20 frames ≈ 10 ticks 的前瞻缓冲，lekai 只生成 1 interval (4 ticks)，无延迟容忍。

### 修复方案

将 `generation_length_frames` 传入规则生成器，生成多个 interval 的伴奏:

```python
def _generate_rule_based(
    self,
    generation_start_tick: int,
    generation_interval_ticks: int,
    generation_length_frames: int,  # 新增参数
) -> List[EventPayload]:
    interval = max(1, int(generation_interval_ticks))
    # 计算需要覆盖多少个 interval
    # lekai 的 frame 概念: 1 frame = 1 tick (与 Stanley 不同)
    # 生成足够的 interval 来覆盖 generation_length_frames
    num_intervals = max(1, generation_length_frames // interval)

    window_start = generation_start_tick - interval * 2
    recent_on_pitches = self._collect_recent_pitches(window_start, generation_start_tick)

    unique = self._deduplicate_pitches(recent_on_pitches)

    if not unique:
        return []

    accompaniment: List[EventPayload] = []
    for i in range(num_intervals):
        offset = i * interval
        chord_tick = generation_start_tick + offset
        off_tick = chord_tick + interval

        for pitch in unique:
            model_pitch = max(36, pitch - 12)
            accompaniment.append(
                {"type": "note_on", "pitch": model_pitch, "tick": chord_tick, "velocity": 80}
            )
            accompaniment.append(
                {"type": "note_off", "pitch": model_pitch, "tick": off_tick, "velocity": 0}
            )

    accompaniment.sort(key=lambda e: (int(e["tick"]), 0 if str(e["type"]) == "note_off" else 1))
    return accompaniment
```

`generate()` 中传递参数:

```python
accompaniment = self._generate_rule_based(
    generation_start_tick=int(generation_start_tick),
    generation_interval_ticks=int(generation_interval_ticks),
    generation_length_frames=int(generation_length_frames),  # 传入
)
```

---

## Bug #6 (低): `_accompaniment_history` 无限增长

### 问题定位

`lekai_http_backend.py:67`:

```python
self._accompaniment_history.extend(accompaniment)  # 只增不减
```

### 修复方案

参考 Stanley 的裁剪逻辑 (`stanley_legacy.py:219-220`)，在每次 generate 后裁剪旧数据:

```python
def generate(self, melody_events, ...):
    ...
    self._accompaniment_history.extend(accompaniment)

    # 裁剪：只保留 generation_start_tick 附近的历史
    max_history_ticks = int(generation_length_frames) * 2  # 保留 2 倍生成长度的历史
    cutoff_tick = generation_start_tick - max_history_ticks
    if cutoff_tick > 0:
        self._accompaniment_history = [
            e for e in self._accompaniment_history
            if int(e.get("tick", 0)) >= cutoff_tick
        ]
        self._melody_history = [
            e for e in self._melody_history
            if int(e.get("tick", 0)) >= cutoff_tick
        ]

    ...
```

---

## Bug #7 (低): `server_lekai.py` 缺少启动入口

### 修复方案

在 `server_lekai.py` 末尾添加:

```python
def main() -> None:
    import uvicorn
    host = "0.0.0.0"
    port = 8000
    print("StreamMUSE Lekai Inference Server (rule-based placeholder)")
    print(f"Listening on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
```

---

## 修复优先级和依赖关系

```
修复顺序:

Phase 1 — 让 lekai stub 能跑起来（不崩溃）:
  #1 (velocity null 崩溃)      ← 必须首先修，否则完全无法运行
  #2 短期 (stub 逻辑修复)       ← 删除 _active_model_pitches，修复 stub 附带 bug
  #3 (默认参数对齐)              ← 让 --model-name lekai 开箱即用
  #7 (server 启动入口)           ← 顺手

Phase 2 — 让 lekai stub 行为合理:
  #4 (inject melody 覆盖)       ← 需要先确定 #3 的 interval 值
  #5 (generation_length 前瞻)   ← stub 多 interval 生成
  #6 (history 裁剪)              ← 防止长 session 内存泄漏

Phase 3 — 集成真实 PianoLLaMA 模型 (#2 长期方案):
  a. 改造 model.generate_accompaniment() 接受 beat tokens 而非 Dataset
  b. LekaiHttpBackend 接入模型: events → pianoroll → tokens → 模型 → tokens → pianoroll → events
  c. 利用 MidiConverter 的 events_to_pianoroll / pianoroll_to_events 做双向转换
  d. 实现增量生成（KV cache），而非全量重新生成
  e. _active_pitches 跨请求持续音跟踪
```

建议分三个 commit:
1. **Critical fix**: Phase 1（#1 #2短期 #3 #7）— 让 lekai client-server 能跑
2. **Enhancement**: Phase 2（#4 #5 #6）— 让 stub 行为合理
3. **Model integration**: Phase 3（#2长期）— 接入真实 PianoLLaMA，彻底解决时值写死问题
