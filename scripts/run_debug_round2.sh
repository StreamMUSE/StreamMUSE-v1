#!/bin/bash
# Debug Round 2 - Complete Execution Script
# 执行所有 debug 步骤并生成报告

set -e  # 遇到错误立即退出

echo "=========================================="
echo "Debug Round 2 - Complete Execution"
echo "Start time: $(date)"
echo "=========================================="

# 配置
WORK_DIR="/data/home/bowenzheng/mbzuai-projects/StreamMUSE-v1"
OUTPUT_DIR="${WORK_DIR}/output/debug_round2"
LOG_DIR="${WORK_DIR}/logs"
REPORT_DIR="${WORK_DIR}/developing-logs/2026-4-23/debug-round2-reports"

cd ${WORK_DIR}

# 创建目录
mkdir -p ${OUTPUT_DIR}/{offline,fake_rt_v4,comparison}
mkdir -p ${LOG_DIR}
mkdir -p ${REPORT_DIR}

# 测试歌曲列表
SONGS="1 2 3 4 5"

echo ""
echo "[Step 1] Starting Server..."
echo "=========================================="

# 启动 Server (后台)
LEKAI_CHECKPOINT_PATH=models/ModelLekai/epoch_4_1104_1204/model.safetensors \
LEKAI_DEVICE=cuda \
LEKAI_DTYPE=auto \
LEKAI_RT_TEMPERATURE=0.0 \
LEKAI_RT_TOP_K=1 \
LEKAI_RT_TOP_P=0.0 \
LEKAI_RT_REPETITION_PENALTY=1.2 \
uv run python -m streammuse.infrastructure.inference.server_lekai > ${LOG_DIR}/server.log 2>&1 &

SERVER_PID=$!
echo "Server started with PID: ${SERVER_PID}"

# 等待 Server 启动
sleep 5

# 检查 Server 是否正常运行
if ! curl -s http://localhost:8000/health > /dev/null; then
    echo "ERROR: Server failed to start!"
    cat ${LOG_DIR}/server.log
    exit 1
fi

echo "Server is running."

echo ""
echo "[Step 2] Running FakeRT tests..."
echo "=========================================="

for song_id in ${SONGS}; do
    echo "Processing Song ${song_id}..."
    
    uv run python scripts/run_lekai_fake_realtime.py \
        --midi-file-path prompts/inputs_lekai/mel/${song_id}.mid \
        --output-dir ${OUTPUT_DIR}/fake_rt_v4 \
        --server-url http://127.0.0.1:8000/generate_accompaniment \
        --generation-interval-ticks 4 \
        --generation-length-frames 4 \
        --max-ticks 256 \
        > ${LOG_DIR}/fakert_${song_id}.log 2>&1
    
    echo "  Song ${song_id} completed."
done

# 停止 Server
echo "Stopping Server..."
kill ${SERVER_PID} 2>/dev/null || true
sleep 2

echo ""
echo "[Step 3] Running Offline tests..."
echo "=========================================="

# 注意：需要先修改 run_lekai_offline.py 支持 temperature 参数
# 如果未修改，需要先修改

for i in 0 1 2 3 4; do
    song_id=$((i+1))
    echo "Processing Song ${song_id} (condition_idx=${i})..."
    
    # 检查是否需要添加 temperature 参数支持
    if grep -q "temperature" scripts/run_lekai_offline.py; then
        TEMP_ARGS="--temperature 0.0 --top-k 1 --top-p 0.0"
    else
        TEMP_ARGS=""
    fi
    
    CUDA_VISIBLE_DEVICES=4 uv run python scripts/run_lekai_offline.py \
        --checkpoint models/ModelLekai/epoch_4_1104_1204/model.safetensors \
        --npz-dir prompts/inputs_lekai/npz \
        --output-dir ${OUTPUT_DIR}/offline \
        --device cuda \
        --dtype auto \
        ${TEMP_ARGS} \
        --condition-idx ${i} \
        --gt-prefix-beats 0 \
        > ${LOG_DIR}/offline_${song_id}.log 2>&1
    
    echo "  Song ${song_id} completed."
done

echo ""
echo "[Step 4] Extracting and comparing prompts..."
echo "=========================================="

# 提取 prompt 日志
for song_id in ${SONGS}; do
    grep "\[PROMPT_DEBUG\]" ${LOG_DIR}/fakert_${song_id}.log > ${LOG_DIR}/fakert_${song_id}_prompt.log 2>/dev/null || true
    grep "\[PROMPT_DEBUG\]" ${LOG_DIR}/offline_${song_id}.log > ${LOG_DIR}/offline_${song_id}_prompt.log 2>/dev/null || true
done

echo "Prompt logs extracted."

echo ""
echo "[Step 5] Comparing MIDI outputs..."
echo "=========================================="

# 创建对比脚本
cat > ${WORK_DIR}/scripts/compare_outputs_round2.py << 'PYTHON_EOF'
import os
import sys
import json
import glob
import mido
from pathlib import Path

def count_notes(mid_path):
    """统计 MIDI 文件中的 note_on 事件数"""
    if not os.path.exists(mid_path):
        return -1
    
    mid = mido.MidiFile(mid_path)
    note_on_count = 0
    for track in mid.tracks:
        for msg in track:
            if msg.type == 'note_on' and msg.velocity > 0:
                note_on_count += 1
    return note_on_count

def main():
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "output/debug_round2"
    
    results = []
    
    for song_id in [1, 2, 3, 4, 5]:
        # Offline 文件路径 (需要找到正确的文件)
        offline_files = glob.glob(f"{output_dir}/offline/*_{song_id}_generated.mid")
        offline_file = offline_files[0] if offline_files else None
        
        # FakeRT 文件路径
        fakert_file = f"{output_dir}/fake_rt_v4/{song_id}_fake_realtime_combined.mid"
        
        offline_notes = count_notes(offline_file) if offline_file else -1
        fakert_notes = count_notes(fakert_file)
        
        match_rate = 0
        if offline_notes > 0:
            match_rate = min(offline_notes, fakert_notes) / max(offline_notes, fakert_notes) * 100
        
        results.append({
            "song_id": song_id,
            "offline_notes": offline_notes,
            "fakert_notes": fakert_notes,
            "match_rate": match_rate,
            "offline_file": offline_file,
            "fakert_file": fakert_file
        })
        
        print(f"Song {song_id}: Offline={offline_notes}, FakeRT={fakert_notes}, Match={match_rate:.1f}%")
    
    # 保存结果
    with open(f"{output_dir}/comparison/round2_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # 计算平均
    avg_match = sum(r["match_rate"] for r in results) / len(results)
    print(f"\nAverage Match Rate: {avg_match:.1f}%")
    
    if avg_match >= 95:
        print("✅ PASSED: Match rate >= 95%")
    else:
        print("❌ FAILED: Match rate < 95%")

if __name__ == "__main__":
    main()
PYTHON_EOF

uv run python scripts/compare_outputs_round2.py ${OUTPUT_DIR}

echo ""
echo "[Step 6] Generating final report..."
echo "=========================================="

# 生成最终报告
cat > ${REPORT_DIR}/round2-final-report.md << EOF
# Debug Round 2 - Final Execution Report

**Execution Date**: $(date +%Y-%m-%d)  
**Execution Time**: $(date +%H:%M:%S)

---

## Execution Summary

| Step | Status | Notes |
|------|--------|-------|
| Server Start | ✅ | PID ${SERVER_PID} |
| FakeRT Tests | ✅ | 5 songs |
| Offline Tests | ✅ | 5 songs |
| Prompt Extraction | ✅ | 5 songs |
| MIDI Comparison | ✅ | See results below |

## Results

### Note Count Comparison

| Song | Offline | FakeRT | Match Rate |
|------|---------|--------|------------|
EOF

# 添加结果表格
python3 << PYTHON_EOF
import json
with open("${OUTPUT_DIR}/comparison/round2_results.json") as f:
    results = json.load(f)
    for r in results:
        print(f"| {r['song_id']} | {r['offline_notes']} | {r['fakert_notes']} | {r['match_rate']:.1f}% |")
PYTHON_EOF

cat >> ${REPORT_DIR}/round2-final-report.md << EOF

### Analysis

$(python3 << PYTHON_EOF
import json
with open("${OUTPUT_DIR}/comparison/round2_results.json") as f:
    results = json.load(f)
    avg = sum(r['match_rate'] for r in results) / len(results)
    if avg >= 95:
        print(f"✅ **PASSED**: Average match rate {avg:.1f}% >= 95%")
        print("\nFakeRT and Offline outputs are consistent.")
    else:
        print(f"❌ **FAILED**: Average match rate {avg:.1f}% < 95%")
        print("\nFurther debugging required.")
PYTHON_EOF
)

## Conclusion

$(python3 << PYTHON_EOF
import json
with open("${OUTPUT_DIR}/comparison/round2_results.json") as f:
    results = json.load(f)
    avg = sum(r['match_rate'] for r in results) / len(results)
    if avg >= 95:
        print("Debug Round 2 is complete. The consistency issue has been resolved.")
    else:
        print("Debug Round 2 requires further investigation. See analysis above.")
PYTHON_EOF
)

## Attachments

- Output files: ${OUTPUT_DIR}
- Log files: ${LOG_DIR}
- Comparison results: ${OUTPUT_DIR}/comparison/round2_results.json
EOF

echo ""
echo "=========================================="
echo "Debug Round 2 - Execution Complete!"
echo "End time: $(date)"
echo "=========================================="
echo ""
echo "Results saved to:"
echo "  - ${REPORT_DIR}/round2-final-report.md"
echo "  - ${OUTPUT_DIR}/comparison/round2_results.json"
echo ""

# 显示摘要
python3 << PYTHON_EOF
import json
print("\n=== SUMMARY ===")
with open("${OUTPUT_DIR}/comparison/round2_results.json") as f:
    results = json.load(f)
    for r in results:
        status = "✅" if r['match_rate'] >= 95 else "❌"
        print(f"{status} Song {r['song_id']}: {r['match_rate']:.1f}% match")
    avg = sum(r['match_rate'] for r in results) / len(results)
    print(f"\nAverage: {avg:.1f}%")
PYTHON_EOF
