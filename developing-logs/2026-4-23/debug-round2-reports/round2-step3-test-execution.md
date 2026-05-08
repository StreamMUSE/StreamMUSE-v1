# Debug Round 2 - Step 3: 测试执行报告

**日期**: 2026-04-24  
**状态**: 准备执行  

---

## 1. 测试环境准备

### 1.1 输出目录
```
output/debug_round2/
├── offline/          # Offline 输出
├── fake_rt_v4/       # FakeRT (修复后) 输出
└── comparison/       # 对比结果

logs/
├── offline_round2.log    # Offline 日志
├── fakert_round2.log     # FakeRT 日志
└── prompt_comparison.txt # Prompt 对比结果
```

### 1.2 测试数据
- Song 1: `prompts/inputs_lekai/mel/1.mid` + `prompts/inputs_lekai/npz/1.npz`
- Song 2-5: 待全量验证时使用

---

## 2. 执行步骤

### Step 3.1: 重启 Server (Terminal 1)

```bash
cd /data/home/bowenzheng/mbzuai-projects/StreamMUSE-v1

LEKAI_CHECKPOINT_PATH=models/ModelLekai/epoch_4_1104_1204/model.safetensors \
LEKAI_DEVICE=cuda \
LEKAI_DTYPE=auto \
LEKAI_RT_TEMPERATURE=0.0 \
LEKAI_RT_TOP_K=1 \
LEKAI_RT_TOP_P=0.0 \
LEKAI_RT_REPETITION_PENALTY=1.0 \
uv run python -m streammuse.infrastructure.inference.server_lekai
```

**验证 Server 启动成功**:
- 检查日志显示 `Listening on http://0.0.0.0:8000`
- 检查 checkpoint 加载成功

### Step 3.2: 运行 FakeRT (Terminal 2)

```bash
cd /data/home/bowenzheng/mbzuai-projects/StreamMUSE-v1

# 创建输出目录
mkdir -p output/debug_round2/fake_rt_v4

# 运行 FakeRT (仅 Song 1)
uv run python scripts/run_lekai_fake_realtime.py \
  --midi-file-path prompts/inputs_lekai/mel/1.mid \
  --output-dir output/debug_round2/fake_rt_v4 \
  --server-url http://127.0.0.1:8000/generate_accompaniment \
  --generation-interval-ticks 4 \
  --generation-length-frames 4 \
  --max-ticks 256 \
  2>&1 | tee logs/fakert_round2.log
```

**预期输出**:
- 日志中包含 `[PROMPT_DEBUG]` 标记的日志
- 生成文件 `output/debug_round2/fake_rt_v4/1_fake_realtime_combined.mid`

### Step 3.3: 运行 Offline (Terminal 2, Server 停止后)

```bash
cd /data/home/bowenzheng/mbzuai-projects/StreamMUSE-v1

# 创建输出目录
mkdir -p output/debug_round2/offline

# 运行 Offline (Song 1, condition-idx=0 对应 1.npz)
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
  2>&1 | tee logs/offline_round2.log
```

**预期输出**:
- 日志中包含 `[PROMPT_DEBUG]` 标记的日志
- 生成文件 `output/debug_round2/offline/XXX_1_generated.mid`

---

## 3. 日志提取与对比

### 3.1 提取 Prompt 日志

```bash
# 提取 FakeRT prompt 日志
grep "\[PROMPT_DEBUG\]" logs/fakert_round2.log > logs/fakert_prompt.log

# 提取 Offline prompt 日志
grep "\[PROMPT_DEBUG\]" logs/offline_round2.log > logs/offline_prompt.log
```

### 3.2 对比 Prompt 结构

```python
# scripts/compare_prompt_logs.py
import re

def parse_prompt_log(log_path):
    with open(log_path) as f:
        content = f.read()
    
    # 提取 First 30 tokens
    match = re.search(r'First 30 tokens: \[(.*?)\]', content)
    if match:
        tokens_str = match.group(1)
        tokens = [int(t.strip()) for t in tokens_str.split(',')]
        return tokens
    return None

fakert_tokens = parse_prompt_log('logs/fakert_prompt.log')
offline_tokens = parse_prompt_log('logs/offline_prompt.log')

print("FakeRT first 30 tokens:", fakert_tokens)
print("Offline first 30 tokens:", offline_tokens)

if fakert_tokens and offline_tokens:
    for i, (f, o) in enumerate(zip(fakert_tokens, offline_tokens)):
        if f != o:
            print(f"Mismatch at position {i}: FakeRT={f}, Offline={o}")
```

---

## 4. 结果验证

### 4.1 MIDI 文件对比

```bash
# 提取 note 数量
echo "=== FakeRT ==="
python -c "import mido; mid=mido.MidiFile('output/debug_round2/fake_rt_v4/1_fake_realtime_combined.mid'); print('Tracks:', len(mid.tracks))"

echo "=== Offline ==="
python -c "import mido; import glob; f=glob.glob('output/debug_round2/offline/*_generated.mid')[0]; mid=mido.MidiFile(f); print('Tracks:', len(mid.tracks))"
```

### 4.2 详细对比

```bash
python scripts/debug_inference_consistency.py \
  output/debug_round2/offline \
  output/debug_round2/fake_rt_v4
```

---

## 5. 预期结果

### 成功标准
- [ ] FakeRT 和 Offline 的 prompt 前 30 个 token 完全一致
- [ ] Song 1 的 note_on 数量一致 (Offline: 233)
- [ ] Match rate >= 95%

### 如果失败
- 提取差异位置
- 分析原因
- 回到 Step 2 继续修复

---

## 6. 执行状态

| 步骤 | 状态 | 备注 |
|------|------|------|
| Server 重启 | ⏳ 待执行 | 需要 Terminal 1 |
| FakeRT 运行 | ⏳ 待执行 | Song 1 only |
| Offline 运行 | ⏳ 待执行 | Song 1 only |
| 日志对比 | ⏳ 待执行 | 提取 [PROMPT_DEBUG] |
| MIDI 对比 | ⏳ 待执行 | note_on 数量 |

---

**下一步**: 执行上述命令，收集结果
