# Step 5: 最终验证与总结 执行报告

**执行日期**: 2026-04-24  
**执行人**: GitHub Copilot

---

## 1. 执行内容

- [x] 重启 Lekai 服务器，确保加载包含 Bug 3 修复的最新代码
- [x] 运行 FakeRT v3（5 首曲子，interval=4，length=4）
- [x] 运行 Offline vs FakeRT v3 一致性对比
- [x] 修正对比脚本对“空伴奏轨道”的误报（允许 0 事件参与比较）
- [x] 运行 Offline vs FakeRT overlap 一致性对比（复用既有 overlap 结果）
- [x] 汇总验证结果并形成最终结论

---

## 2. 最终测试结果

### 2.1 配置级结果

| 配置 | 平均 Match Rate（事件级） | 最低单曲 Match Rate | 通过标准 | 结果 |
|------|---------------------------|---------------------|----------|------|
| FakeRT equiv v3 (4,4) | **1.15%** | **0.0%** | >= 95% | ❌ 未通过 |
| FakeRT overlap (4,8) | **1.00%** | **0.0%** | >= 80% | ❌ 未通过 |

### 2.2 FakeRT equiv v3（4,4）逐曲对比

| 歌曲 | Offline 事件 | FakeRT v3 事件 | Matched | Match Rate |
|------|--------------|----------------|---------|------------|
| 1 | 466 | 28  | 0 | 0.0% |
| 2 | 0   | 74  | 0 | 0.0% |
| 3 | 0   | 34  | 0 | 0.0% |
| 4 | 118 | 32  | 0 | 0.0% |
| 5 | 114 | 140 | 8 | 5.93% |
| **总计** | **698** | **308** | **8** | **1.15%** |

### 2.3 FakeRT 产出密度（v2 vs v3，model note_on）

| 歌曲 | FakeRT v2 | FakeRT v3 | 变化 |
|------|-----------|-----------|------|
| 1 | 6  | 14 | +8 |
| 2 | 19 | 37 | +18 |
| 3 | 11 | 17 | +6 |
| 4 | 9  | 16 | +7 |
| 5 | 29 | 70 | +41 |

**观察**：v3 的伴奏事件数量显著增加，说明 Bug 3 修复对“生成稀疏”问题有正向影响；但和 Offline 的结构一致性仍极差（总体 1.15%）。

---

## 3. 发现的 Bug 总结

| Bug ID | 严重程度 | 描述 | 状态 |
|--------|----------|------|------|
| Bug 1 | 高 | `start_tick <= 0` 导致 tick=0 时回退 rule-based | ✅ 已修复并纳入 v3 验证 |
| Bug 2 | 高 | FakeRT 未传 MIDI BPM 给服务端 | ✅ 已修复并纳入 v3 验证 |
| Bug 3 | 高 | FakeRT 缺少 `acc_{-1}` 先行 token | ✅ 已实现修复；⚠️ 仍未达到等价一致性 |
| Bug 4 | 高 | Offline Song 2/3 伴奏为空，Song 1 前 36 拍为空 | ❌ 未修复（当前主要阻塞） |

补充：对比脚本 `scripts/debug_inference_consistency.py` 已修复空事件误报（之前 Song 2/3 会被错误标记为“找不到伴奏轨道”）。

---

## 4. 结论与建议

- **总体结论**: 本轮 Step 5 验证未通过。尽管 Bug 1/2/3 修复后 FakeRT 生成密度提升，但与 Offline 的等价性仍远低于目标门槛。
- **核心阻塞**: Offline 参考结果本身存在异常（Bug 4），导致“用 Offline 当金标准”这一步存在系统性偏差。
- **建议下一步**:
  1. 优先单独定位并修复 Bug 4（为什么 Song 2/3 全空，Song 1 前 36 拍全空）。
  2. 修复 Bug 4 后重跑 Offline 基线，再进行 FakeRT v4 对比，避免在异常基线上迭代。
  3. 增加 Prompt 级对齐日志（同一 beat 的 mel/acc token 序列）用于定位剩余结构差异。

---

## 5. 附件

- FakeRT v3 输出目录: `output/debug/fake_rt_equivalent_v3/`
- Offline vs FakeRT v3 对比报告: `output/debug/offline/inference_consistency_report.json`
- Offline vs FakeRT overlap 对比报告: `output/debug/fake_rt_overlap/inference_consistency_report.json`
- 会话进度记录: `developing-logs/2026-4-23/debug-reports/session_progress_2026-04-24.md`
