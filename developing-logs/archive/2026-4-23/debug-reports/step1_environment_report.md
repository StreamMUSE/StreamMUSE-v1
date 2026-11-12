# Step 1: 环境准备与数据验证 执行报告

**执行日期**: 2026-04-24  
**执行人**: Claude Code  

---

## 1. 执行内容

- [x] 验证测试数据完整性 (MIDI + NPZ)
- [x] 创建输出目录结构
- [x] 确认 Checkpoint 路径与 CUDA 设备
- [x] 编写自动化脚本 (`run_all_tests.sh`, `compare_all.sh`, `debug_inference_consistency.py`)

---

## 2. 执行结果

### 2.1 MIDI 文件验证

| 文件 | 大小 | Note 数量 | ticks_per_beat |
|------|------|-----------|----------------|
| 1.mid | 1,145 B | 167 | 220 |
| 2.mid | 2,374 B | 384 | 220 |
| 3.mid | 1,787 B | 281 | 220 |
| 4.mid |   551 B |  80 | 220 |
| 5.mid | 1,545 B | 242 | 220 |

### 2.2 NPZ 文件验证

| 文件 | 大小 | Measures | BPM |
|------|------|----------|-----|
| 1.npz | 623 KB | 28 | 110 |
| 2.npz | 423 KB | 19 |  74 |
| 3.npz | 668 KB | 30 | 120 |
| 4.npz | 334 KB | 15 |  57 |
| 5.npz | 512 KB | 23 |  92 |

### 2.3 输出目录

| 目录 | 状态 |
|------|------|
| `output/debug/offline/` | ✅ 已创建 |
| `output/debug/offline_prefix4/` | ✅ 已创建 |
| `output/debug/fake_rt_equivalent/` | ✅ 已创建 |
| `output/debug/fake_rt_overlap/` | ✅ 已创建 |
| `output/debug/logs/` | ✅ 已创建 |

### 2.4 环境配置

| 检查项 | 状态 | 备注 |
|--------|------|------|
| Checkpoint | ✅ | `models/ModelLekai/epoch_4_1104_1204/model.safetensors` (681 MB) |
| CUDA 设备 | ✅ | 6× NVIDIA H200 NVL (GPU 0-5) |
| `run_lekai_offline.py` | ✅ | 支持 `--condition-idx all`, `--temperature`, `--top-k`, `--top-p` |
| `run_lekai_fake_realtime.py` | ✅ | 支持 `--generation-interval-ticks`, `--generation-length-frames` |

### 2.5 自动化脚本

| 脚本 | 状态 | 功能 |
|------|------|------|
| `scripts/run_all_tests.sh` | ✅ 已创建，语法正确 | 批量运行 Offline + FakeRT (2 种配置) |
| `scripts/compare_all.sh` | ✅ 已创建，语法正确 | 批量对比 Offline vs FakeRT |
| `scripts/debug_inference_consistency.py` | ✅ 已创建 | 自动对比 MIDI，生成 JSON 报告 |

---

## 3. 关键发现

### 3.1 正常情况

- 所有 5 个 MIDI 和 NPZ 文件均存在且非空
- MIDI 文件统一使用 `ticks_per_beat=220`
- NPZ BPM 各不相同 (57~120)，每首曲子长度不一
- Checkpoint 完整 (681 MB)
- CUDA 6×H200 NVL 可用，将使用 `CUDA_VISIBLE_DEVICES=4`

### 3.2 注意事项

- `ticks_per_beat=220` (MIDI) 与系统的 `ticks_per_beat=4` (StreamMUSE 内部) 不同，需确认 `run_lekai_fake_realtime.py` 中的 tick 对齐逻辑在读取 MIDI 时做了正确的转换
- NPZ BPM 各不相同，但 NPZ 的 generation_length_frames 由 `measure` 数量决定，应与 `--max-ticks` 配合验证是否覆盖完整旋律
- `os.listdir` 顺序不保证，plan 中已使用 `--condition-idx all` 并通过 stem 配对，比对脚本实现了此逻辑

---

## 4. 结论与下一步

- **结论**: 环境准备完成，所有测试数据和脚本就绪，可以进行 Step 2 测试
- **下一步行动**: 
  1. 启动 Lekai 推理服务器（deterministic 参数）
  2. 运行 `bash scripts/run_all_tests.sh 4` 执行全量测试
  3. 运行 `bash scripts/compare_all.sh` 生成对比报告
  4. 根据对比结果填写 `step2_baseline_report.md`
- **阻塞项**: 无

---

## 5. 附件

- 自动化脚本: `scripts/run_all_tests.sh`, `scripts/compare_all.sh`, `scripts/debug_inference_consistency.py`
- 输出目录: `output/debug/`
