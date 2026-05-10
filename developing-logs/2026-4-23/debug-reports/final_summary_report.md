# Offline vs Fake Realtime Debug 最终总结报告

**测试周期**: 2026-04-23 ~ 2026-04-24  
**最终更新**: 2026-04-24  
**状态**: 部分完成（核心问题未完全闭环）

---

## 1. 项目概述

- **目标**: 验证并定位 Offline 与 Fake Realtime 在“等价配置（interval=4, length=4）”下不一致的根因。
- **测试数据**: `prompts/inputs_lekai/` 中 5 首曲子（mel + npz）。
- **核心标准**: 在 deterministic 设置下，FakeRT equivalent 配置应尽可能逼近 Offline（计划阈值 >=95%）。

---

## 2. 已完成工作总览

### 2.1 报告与实验链路

已完成并沉淀以下报告：
- `step2_baseline_report.md`
- `step3_phase1_input_layer.md`
- `step3_phase2_tokenization.md`
- `step4_bugfix_report.md`
- `step5_verification_report.md`

### 2.2 Bug 修复进展

| Bug | 描述 | 状态 |
|-----|------|------|
| Bug 1 | `start_tick <= 0` 误回退 rule-based | ✅ 已修复 |
| Bug 2 | FakeRT 未传 BPM 给服务端 | ✅ 已修复 |
| Bug 3 | FakeRT 缺少 `acc_{-1}` 先行 token | ✅ 已修复（效果部分） |
| Bug 4 | Offline Song 2/3 伴奏为空、Song 1 前 36 拍为空 | ❌ 未修复 |

### 2.3 本轮新增执行

- 重启服务端并加载最新修复代码
- 运行 FakeRT v3（5 首）
- 完成 Offline vs FakeRT v3 对比
- 修复对比脚本对空事件的误报
- 完成 overlap 配置对比结果归档

---

## 3. 关键结果

### 3.1 FakeRT equivalent v3（4,4）

- 总体 Match Rate: **1.15%**
- 逐曲：`[0.0%, 0.0%, 0.0%, 0.0%, 5.93%]`
- 结论：**远低于 >=95% 目标**，未通过。

### 3.2 FakeRT overlap（4,8）

- 总体 Match Rate: **1.00%**
- 结论：**远低于 >=80% 目标**，未通过。

### 3.3 修复效果（v2 -> v3）

虽然一致性未达标，但 FakeRT 的伴奏生成密度明显提升：
- Song1 note_on: 6 -> 14
- Song2 note_on: 19 -> 37
- Song3 note_on: 11 -> 17
- Song4 note_on: 9 -> 16
- Song5 note_on: 29 -> 70

说明 Bug 3 修复方向是有效的，但不足以解释并消除全部差异。

---

## 4. 根因判断（当前阶段）

1. FakeRT 侧结构问题（Bug 1/2/3）已被部分收敛。  
2. Offline 基线本身存在异常（Bug 4），使“Offline 作为金标准”的对齐工作受阻。  
3. 目前最可能的主阻塞已转移到 **Offline 端生成异常** 与剩余 prompt/state 对齐细节。

---

## 5. 结论

- 本轮调试实现了从“可运行”到“可定位”的关键跃迁：
  - 核心链路修复已落地（Bug 1/2/3）。
  - v3 回归验证和报告体系已补齐。
- 但目标尚未完成：
  - `FakeRT equivalent` 与 Offline 的一致性仍显著不足。
  - 必须优先处理 Bug 4 后再进行下一轮等价验证。

---

## 6. 后续行动（建议执行顺序）

1. **优先调查 Bug 4**：定位 Offline 在 Song 2/3 全空、Song 1 前 36 拍全空的触发机制。  
2. 在修复 Bug 4 后，重跑 Offline 基线（生成新的 gold reference）。  
3. 基于新 Offline 基线重跑 FakeRT（v4）并复用现有对比脚本。  
4. 若仍不一致，增加逐 beat 的 prompt token 对齐日志（Offline vs FakeRT 同步输出）。

---

## 7. 数据归档

- Offline 基线输出: `output/debug/offline/`
- FakeRT v3 输出: `output/debug/fake_rt_equivalent_v3/`
- FakeRT overlap 输出: `output/debug/fake_rt_overlap/`
- v3 对比报告: `output/debug/offline/inference_consistency_report.json`
- overlap 对比报告: `output/debug/fake_rt_overlap/inference_consistency_report.json`
- 全部过程报告: `developing-logs/2026-4-23/debug-reports/`
