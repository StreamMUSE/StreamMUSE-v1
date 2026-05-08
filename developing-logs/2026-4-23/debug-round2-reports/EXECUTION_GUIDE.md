# Debug Round 2 - 执行指南

## 快速执行

### 方式 1: 自动执行脚本（推荐）

```bash
cd /data/home/bowenzheng/mbzuai-projects/StreamMUSE-v1

# 给脚本执行权限
chmod +x scripts/run_debug_round2.sh

# 执行完整 debug 流程
./scripts/run_debug_round2.sh
```

**注意**: 此脚本会自动执行所有步骤，包括：
1. 启动 Server
2. 运行 FakeRT 测试（5 首歌曲）
3. 运行 Offline 测试（5 首歌曲）
4. 提取并对比结果
5. 生成最终报告

执行时间约 30-60 分钟。

---

### 方式 2: 分步手动执行（调试用）

如果你需要手动控制每一步，或者脚本执行出错，可以使用以下分步命令：

#### Step 1: 创建目录

```bash
cd /data/home/bowenzheng/mbzuai-projects/StreamMUSE-v1

mkdir -p output/debug_round2/{offline,fake_rt_v4,comparison}
mkdir -p logs
```

#### Step 2: 启动 Server

```bash
# Terminal 1
LEKAI_CHECKPOINT_PATH=models/ModelLekai/epoch_4_1104_1204/model.safetensors \
LEKAI_DEVICE=cuda \
LEKAI_DTYPE=auto \
LEKAI_RT_TEMPERATURE=0.0 \
LEKAI_RT_TOP_K=1 \
LEKAI_RT_TOP_P=0.0 \
LEKAI_RT_REPETITION_PENALTY=1.2 \
uv run python -m streammuse.infrastructure.inference.server_lekai

# 验证 Server 启动
# 应该看到: "Listening on http://0.0.0.0:8000"
```

#### Step 3: 运行 FakeRT 测试

```bash
# Terminal 2 (Server 保持运行)

# Song 1
uv run python scripts/run_lekai_fake_realtime.py \
  --midi-file-path prompts/inputs_lekai/mel/1.mid \
  --output-dir output/debug_round2/fake_rt_v4 \
  --server-url http://127.0.0.1:8000/generate_accompaniment \
  --generation-interval-ticks 4 \
  --generation-length-frames 4 \
  --max-ticks 256 \
  2>&1 | tee logs/fakert_1.log

# Song 2-5 (类似，可选)
for i in 2 3 4 5; do
  uv run python scripts/run_lekai_fake_realtime.py \
    --midi-file-path prompts/inputs_lekai/mel/${i}.mid \
    --output-dir output/debug_round2/fake_rt_v4 \
    --server-url http://127.0.0.1:8000/generate_accompaniment \
    --generation-interval-ticks 4 \
    --generation-length-frames 4 \
    --max-ticks 256 \
    2>&1 | tee logs/fakert_${i}.log
done
```

#### Step 4: 停止 Server 并运行 Offline

```bash
# Terminal 1: 按 Ctrl+C 停止 Server

# Terminal 2: 运行 Offline

# Song 1 (condition_idx=0 对应 1.npz)
CUDA_VISIBLE_DEVICES=4 uv run python scripts/run_lekai_offline.py \
  --checkpoint models/ModelLekai/epoch_4_1104_1204/model.safetensors \
  --npz-dir prompts/inputs_lekai/npz \
  --output-dir output/debug_round2/offline \
  --device cuda \
  --dtype auto \
  --temperature 0.0 \
  --top-k 1 \
  --top-p 0.0 \
  --condition-idx 0 \
  --gt-prefix-beats 0 \
  2>&1 | tee logs/offline_1.log

# Song 2-5 (condition_idx=1-4，可选)
for i in 1 2 3 4; do
  song_id=$((i+1))
  CUDA_VISIBLE_DEVICES=4 uv run python scripts/run_lekai_offline.py \
    --checkpoint models/ModelLekai/epoch_4_1104_1204/model.safetensors \
    --npz-dir prompts/inputs_lekai/npz \
    --output-dir output/debug_round2/offline \
    --device cuda \
    --dtype auto \
    --temperature 0.0 \
    --top-k 1 \
    --top-p 0.0 \
    --condition-idx ${i} \
    --gt-prefix-beats 0 \
    2>&1 | tee logs/offline_${song_id}.log
done
```

#### Step 5: 提取 Prompt 日志

```bash
# 提取 [PROMPT_DEBUG] 日志
for i in 1 2 3 4 5; do
  grep "\[PROMPT_DEBUG\]" logs/fakert_${i}.log > logs/fakert_${i}_prompt.log 2>/dev/null || true
  grep "\[PROMPT_DEBUG\]" logs/offline_${i}.log > logs/offline_${i}_prompt.log 2>/dev/null || true
done
```

#### Step 6: 对比结果

```bash
# 创建对比脚本并执行
cat > scripts/compare_outputs_round2.py << 'PYTHON_EOF'
import os
import sys
import json
import glob
import mido

def count_notes(mid_path):
    if not os.path.exists(mid_path):
        return -1
    mid = mido.MidiFile(mid_path)
    return sum(1 for track in mid.tracks for msg in track if msg.type == 'note_on' and msg.velocity > 0)

output_dir = sys.argv[1] if len(sys.argv) > 1 else "output/debug_round2"
results = []

for song_id in [1, 2, 3, 4, 5]:
    offline_files = glob.glob(f"{output_dir}/offline/*_{song_id}_generated.mid")
    offline_file = offline_files[0] if offline_files else None
    fakert_file = f"{output_dir}/fake_rt_v4/{song_id}_fake_realtime_combined.mid"
    
    offline_notes = count_notes(offline_file) if offline_file else -1
    fakert_notes = count_notes(fakert_file)
    
    match_rate = 0
    if offline_notes > 0 and fakert_notes > 0:
        match_rate = min(offline_notes, fakert_notes) / max(offline_notes, fakert_notes) * 100
    
    results.append({
        "song_id": song_id,
        "offline_notes": offline_notes,
        "fakert_notes": fakert_notes,
        "match_rate": match_rate
    })
    
    print(f"Song {song_id}: Offline={offline_notes}, FakeRT={fakert_notes}, Match={match_rate:.1f}%")

with open(f"{output_dir}/comparison/round2_results.json", "w") as f:
    json.dump(results, f, indent=2)

avg = sum(r["match_rate"] for r in results) / len(results)
print(f"\nAverage Match Rate: {avg:.1f}%")
PYTHON_EOF

uv run python scripts/compare_outputs_round2.py output/debug_round2
```

---

## 结果分析

### 成功标准

| 指标 | 目标 | 说明 |
|------|------|------|
| Match Rate | >= 95% | FakeRT 和 Offline 的 note_on 数量一致 |
| Prompt 结构 | 完全一致 | 前 30 个 token 完全相同 |

### 查看详细日志

```bash
# 对比 Song 1 的 prompt 日志
diff logs/fakert_1_prompt.log logs/offline_1_prompt.log

# 查看 FakeRT 完整日志
cat logs/fakert_1.log | grep -A5 -B5 "PROMPT_DEBUG"

# 查看 Offline 完整日志
cat logs/offline_1.log | grep -A5 -B5 "PROMPT_DEBUG"
```

---

## 常见问题

### Q1: Server 启动失败

**症状**: "Address already in use" 或端口冲突

**解决**:
```bash
# 查找占用 8000 端口的进程
lsof -i :8000

# 杀死进程
kill -9 <PID>

# 或使用不同端口
# 修改 --server-url 为 http://127.0.0.1:8001/...
```

### Q2: CUDA out of memory

**症状**: RuntimeError: CUDA out of memory

**解决**:
```bash
# 使用 CPU
LEKAI_DEVICE=cpu ...

# 或减少 batch size (修改 generation-length-frames)
```

### Q3: 找不到 MIDI 文件

**症状**: FileNotFoundError: MIDI file not found

**解决**:
```bash
# 检查文件是否存在
ls prompts/inputs_lekai/mel/1.mid

# 确认路径正确
```

---

## 输出文件

执行完成后，以下文件将生成：

```
output/debug_round2/
├── offline/                    # Offline 生成的 MIDI
│   ├── 000_1_generated.mid
│   ├── 001_2_generated.mid
│   └── ...
├── fake_rt_v4/                 # FakeRT 生成的 MIDI
│   ├── 1_fake_realtime_combined.mid
│   ├── 2_fake_realtime_combined.mid
│   └── ...
└── comparison/
    └── round2_results.json     # 对比结果

logs/
├── fakert_1.log               # FakeRT 完整日志
├── offline_1.log              # Offline 完整日志
├── fakert_1_prompt.log        # 提取的 prompt 日志
└── offline_1_prompt.log       # 提取的 prompt 日志

developing-logs/2026-4-23/debug-round2-reports/
├── round2-final-report.md     # 最终报告
└── ...
```

---

**祝调试顺利！**
