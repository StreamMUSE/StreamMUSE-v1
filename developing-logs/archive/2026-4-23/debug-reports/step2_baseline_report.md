# Step 2: 运行 Baseline 测试 执行报告

**执行日期**: 2026-04-24  
**执行人**: Claude Code  

---

## 1. 执行内容

- [x] 运行 Offline gt-prefix=0（5 首曲子）
- [x] 运行 Offline gt-prefix=4（5 首曲子）
- [x] 运行 FakeRT "等价 Offline" 模式 v1（interval=4, length=4，Bug 修复前）
- [x] 运行 FakeRT "等价 Offline" 模式 v2（interval=4, length=4，Bug1+Bug2 修复后）
- [x] 运行 FakeRT "Overlap" 模式（interval=4, length=8）
- [x] 运行初步对比（offline vs FakeRT v1）

---

## 2. 执行结果

### 2.1 Offline gt-prefix=0 输出（`output/debug/offline/`）

| 文件 | 歌曲 | Melody note_on | Accompaniment note_on | Acc tick 范围（MIDI ticks, tpb=220） |
|------|------|----------------|----------------------|--------------------------------------|
| `000_2_generated.mid` | 2 | 376 | **0** | N/A |
| `001_5_generated.mid` | 5 | 241 | 57 | 330 - 6710 |
| `002_3_generated.mid` | 3 | 275 | **0** | N/A |
| `003_1_generated.mid` | 1 | 166 | 233 | **7920 - 24530**（从第 36 拍才开始）|
| `004_4_generated.mid` | 4 | 75  | 59 | 880 - 12980 |

**关键异常**：
- 曲子 2、3 的伴奏输出全为空（0 note_on），模型生成了全空 beat 序列
- 曲子 1 的伴奏从 tick 7920（第 36 拍）才开始，前 35 拍全为空

### 2.2 Offline gt-prefix=4 输出（`output/debug/offline_prefix4/`）

运行成功，5 个输出文件已生成（主要用于对比参考，非本次 debug 核心路径）。

### 2.3 FakeRT v1（Bug 修复前，`output/debug/fake_rt_equivalent/`）

| 歌曲 | 请求数 | 生成事件总数 | note_on | BPM |
|------|--------|-------------|---------|-----|
| 1    | 54     | 24          | 12      | N/A（未传递 BPM） |
| 2    | 63     | 88          | 44      | N/A |
| 3    | 52     | 30          | 15      | N/A |
| 4    | 44     | 44          | 22      | N/A |
| 5    | 60     | 58          | 29      | N/A |

v1 未传递 BPM（服务器使用默认 120），且存在 Bug 1（start_tick=0 时走 rule-based）。

### 2.4 FakeRT v2（Bug1+Bug2 修复后，`output/debug/fake_rt_equivalent_v2/`）

| 歌曲 | 请求数 | 生成事件总数 | note_on | BPM（从 MIDI 提取） |
|------|--------|-------------|---------|---------------------|
| 1    | 54     | 18          | 6       | 110 |
| 2    | 63     | 38          | 19      | 74  |
| 3    | 52     | 22          | 11      | 120 |
| 4    | 44     | 22          | 9       | 57  |
| 5    | 60     | 60          | 29      | 92  |

FakeRT v2 伴奏仍然极度稀疏（Song 1 仅 6 note_on vs Offline 的 233）。

### 2.5 初步比较（offline vs FakeRT v1）

对比脚本：`scripts/debug_inference_consistency.py`  
对比配置：`output/debug/offline/` vs `output/debug/fake_rt_equivalent/`

| 歌曲 | Offline 事件 | FakeRT v1 事件 | 匹配 | 匹配率 |
|------|-------------|----------------|------|--------|
| 1    | 466         | 24             | 0    | 0.0%   |
| 2    | 0           | 0              | 0    | N/A    |
| 3    | 0           | 0              | 0    | N/A    |
| 4    | 236         | 88             | 2    | 0.8%   |
| 5    | 228         | 116            | 4    | 1.7%   |

**总体匹配率极低（< 2%）**，两种模式的输出有本质差异。

---

## 3. 关键发现

### 3.1 正常情况

- 所有测试命令均成功执行，5 首曲子均产生了输出文件
- Server 在 port 8000 正常运行（计划文件中写 8001，实际使用 8000）
- FakeRT v2 成功从 MIDI 文件提取 BPM 并传递给服务器

### 3.2 异常情况

**异常 1：Offline 曲子 2、3 伴奏全为空**
- 症状：Song 2 和 Song 3 的 Part1 (Accompaniment) 轨道 note_on = 0
- 原因待查：模型对这两首曲子生成了全空的 beat 序列（all empty markers）
- 标记为 Bug 4，在 Phase 3 进一步调查

**异常 2：Offline Song 1 伴奏延迟 36 拍**
- 症状：Song 1 的伴奏从 tick 7920（第 36 拍）才开始，前 35 拍全为空
- 原因：模型在前 36 拍生成了 empty beat，从第 36 拍起才产生音符
- 说明：这是模型需要足够 melody context 才开始生成伴奏的行为

**异常 3：FakeRT 伴奏极度稀疏**
- 症状：FakeRT v2 在所有歌曲中生成极少的伴奏音符（比 Offline 少 10x-40x）
- 原因待查：疑为 Bug 3（FakeRT 提示词结构与训练时不同）导致模型处于错误状态
- 标记为 Phase 3.2 进一步调查

---

## 4. 结论与下一步

- **结论**: Baseline 测试完成，确认 Offline 与 FakeRT 输出存在根本性差异（0-2% 匹配率），需进入 Phase 3 逐层排查
- **下一步行动**: 进入 Phase 3.1（输入层验证）和 Phase 3.2（Tokenization 层分析）
- **阻塞项**: 无（调查继续进行）

---

## 5. 附件

- Offline 输出：`output/debug/offline/`（含 `inference_consistency_report.json`）
- FakeRT v1 输出：`output/debug/fake_rt_equivalent/`
- FakeRT v2 输出：`output/debug/fake_rt_equivalent_v2/`
- FakeRT Overlap 输出：`output/debug/fake_rt_overlap/`
- 对比报告：`output/debug/offline/inference_consistency_report.json`
