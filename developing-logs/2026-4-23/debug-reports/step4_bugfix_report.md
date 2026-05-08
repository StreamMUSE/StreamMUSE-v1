# Step 4: Bug 修复与回归测试 执行报告

**执行日期**: 2026-04-24  
**执行人**: Claude Code  

---

## 1. 执行内容

- [x] Bug 1 修复：`generation_start_tick <= 0` 守卫错误
- [x] Bug 2 修复：BPM 未传递给服务器
- [x] Bug 3 分析：FakeRT 缺少 `acc_{-1}` 先行生成（结构性 prompt 不匹配）
- [x] FakeRT v2 重新运行（Bug1+Bug2 修复后）
- [ ] Bug 3 修复实现（在本报告后执行）
- [ ] 回归测试（`uv run pytest tests/`）

---

## 2. Bug 修复详情

### 2.1 Bug 1 修复（已完成）

**文件**：`src/streammuse/infrastructure/inference/lekai_http_backend.py`

**问题**：
```python
# 修复前（L716）
if int(generation_start_tick) <= 0:
    return self._generate_rule_based(...)
```

当 `generation_start_tick=0` 时（FakeRT 第一次请求总是从 tick=0 开始），条件 `<= 0` 为 True，
导致第一次请求总是走 rule-based stub，而非真实模型。

**修复**：
```python
# 修复后
if int(generation_start_tick) < 0:
    return self._generate_rule_based(...)
```

只有负数 tick（不合法值）才触发 rule-based 回退，tick=0 现在走真实模型。

---

### 2.2 Bug 2 修复（已完成）

**问题**：FakeRT 脚本未将 MIDI 文件中的实际 BPM 传递给服务器，服务器使用默认的 `LEKAI_DEFAULT_BPM=120`。  
5 首曲子的实际 BPM 分别为：110、74、120、57、92。其中 4 首 BPM 与默认值不符。

**修复涉及 3 个文件**：

**a) `scripts/run_lekai_fake_realtime.py`**：
- 添加 `import mido` 和 BPM 提取逻辑
- 从 MIDI 文件的 `set_tempo` 消息计算 BPM
- 将 `bpm=midi_bpm` 传入 `HttpInferenceClientConfig`

**b) `src/streammuse/infrastructure/inference/http_client.py`**：
- `HttpInferenceClientConfig` 添加 `bpm: Optional[int] = None` 字段
- `generate_accompaniment()` payload 中添加 `"bpm": self._config.bpm`

**c) `src/streammuse/infrastructure/inference/server_lekai.py`**：
- `InferenceRequest` 添加 `bpm: Optional[int] = None` 字段
- `backend.generate()` 调用中传入 `bpm=request.bpm`

**d) `src/streammuse/infrastructure/inference/lekai_http_backend.py`**：
- 添加 `_request_bpm: Optional[int] = None` 实例变量
- `generate()` 方法接受 `bpm` 参数并设置 `self._request_bpm`
- `_generate_with_interleaved_prompt()` 使用 `self._request_bpm` 替代环境变量默认值
- `clear_history()` 中重置 `self._request_bpm = None`

---

### 2.3 Bug 3 分析（修复待实现）

**问题**：FakeRT 的 prompt 结构与训练时不匹配。

训练时（Offline，`delay_beats=-1`）的 prompt 序列：
```
[BOS, ts, bpm, pad, acc_{-1}, bar, bar, mel_0, acc_0, mel_1, acc_1, ...]
```

FakeRT 当前的 prompt 序列（第一次请求，current_beat=0）：
```
[BOS, ts, bpm, pad, bar, bar, mel_0, acc_0, mel_1, acc_1, ...]
```

**缺少的 token**：`acc_{-1}`（在任何旋律之前生成的伴奏 token）。

**影响**：模型所有后续 token 的位置编码偏移 1，导致模型处于分布外（OOD）状态，
以 top_k=1 时几乎总是选择 empty marker（最安全的默认选项），使伴奏极度稀疏。

**FakeRT v2 vs Offline 比较**（修复 Bug1+Bug2 后）：

| 歌曲 | Offline acc note_on | FakeRT v2 acc note_on | 比率 |
|------|--------------------|-----------------------|------|
| 1    | 233                | 6                     | 2.6% |
| 2    | 0                  | 19                    | N/A  |
| 3    | 0                  | 11                    | N/A  |
| 4    | 59                 | 9                     | 15%  |
| 5    | 57                 | 29                    | 51%  |

Song 1 仅 2.6% 的比率，印证了 Bug 3 的严重性。

**修复方案**：
在 `_generate_with_interleaved_prompt()` 中，当 `start_beat == 0` 时（第一次生成），
先从 `[BOS, ts, bpm, pad]` 生成一次 `acc_{-1}` token，并将其附加到 seq 后，
再开始 context loop 和 generation loop。

---

## 3. FakeRT v2 重新运行结果

运行目录：`output/debug/fake_rt_equivalent_v2/`  
命令：同 v1，但服务器已重启（含 Bug1+Bug2 修复），BPM 从 MIDI 提取并传递。

结果见上表。整体匹配率仍远低于预期，Bug 3 是主要根因。

---

## 4. 回归测试

待 Bug 3 修复实现后执行：
```bash
uv run pytest tests/ -q --tb=short
```

---

## 5. 结论与下一步

- **结论**: Bug 1（start_tick 守卫）和 Bug 2（BPM 传递）已修复并重新验证。FakeRT v2 相比 v1 有所改善（尤其 Song 2/3/4 有更多伴奏），但与 Offline 的匹配率仍极低。Bug 3（结构性 prompt 不匹配）是剩余差异的主要根因。
- **下一步行动**: 实现 Bug 3 修复（在 `_generate_with_interleaved_prompt` 中添加 `acc_{-1}` 先行生成），然后重跑 FakeRT v3 并对比
- **阻塞项**: 无

---

## 6. 附件

- Bug 1/2 修复差异：见对应文件 git diff
- FakeRT v2 输出：`output/debug/fake_rt_equivalent_v2/`
- 修复后测试日志：`output/debug/logs/`
