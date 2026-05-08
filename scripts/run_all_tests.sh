#!/bin/bash
# Batch test runner: Offline + FakeRT equivalent + FakeRT overlap
# Usage: bash scripts/run_all_tests.sh [CUDA_DEVICE]
# Requires: Lekai server already running on port 8001

set -e

CUDA_DEVICE=${1:-4}
CHECKPOINT="models/ModelLekai/epoch_4_1104_1204/model.safetensors"
NPZ_DIR="prompts/inputs_lekai/npz"
MEL_DIR="prompts/inputs_lekai/mel"

echo "=== Step 2.1.1: Offline gt-prefix-beats=0 ==="
CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} uv run python scripts/run_lekai_offline.py \
  --checkpoint "${CHECKPOINT}" \
  --npz-dir "${NPZ_DIR}" \
  --output-dir output/debug/offline \
  --device cuda \
  --dtype auto \
  --condition-idx all \
  --gt-prefix-beats 0 \
  --temperature 1.0 \
  --top-k 1 \
  --top-p 1.0 \
  --repetition-penalty 1.2

echo ""
echo "=== Step 2.1.2: Offline gt-prefix-beats=4 ==="
CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} uv run python scripts/run_lekai_offline.py \
  --checkpoint "${CHECKPOINT}" \
  --npz-dir "${NPZ_DIR}" \
  --output-dir output/debug/offline_prefix4 \
  --device cuda \
  --dtype auto \
  --condition-idx all \
  --gt-prefix-beats 4 \
  --temperature 1.0 \
  --top-k 1 \
  --top-p 1.0 \
  --repetition-penalty 1.2

echo ""
echo "=== Step 2.2: Fake Realtime equivalent (interval=4, length=4) ==="
for i in 1 2 3 4 5; do
  echo "  Processing: ${MEL_DIR}/${i}.mid"
  uv run python scripts/run_lekai_fake_realtime.py \
    --midi-file-path "${MEL_DIR}/${i}.mid" \
    --output-dir output/debug/fake_rt_equivalent \
    --server-url http://127.0.0.1:8001/generate_accompaniment \
    --generation-interval-ticks 4 \
    --generation-length-frames 4 \
    --max-ticks 256
done

echo ""
echo "=== Step 2.3: Fake Realtime overlap (interval=4, length=8) ==="
for i in 1 2 3 4 5; do
  echo "  Processing: ${MEL_DIR}/${i}.mid"
  uv run python scripts/run_lekai_fake_realtime.py \
    --midi-file-path "${MEL_DIR}/${i}.mid" \
    --output-dir output/debug/fake_rt_overlap \
    --server-url http://127.0.0.1:8001/generate_accompaniment \
    --generation-interval-ticks 4 \
    --generation-length-frames 8 \
    --max-ticks 256
done

echo ""
echo "=== All tests complete ==="
echo "Output directories:"
ls output/debug/offline/ output/debug/offline_prefix4/ output/debug/fake_rt_equivalent/ output/debug/fake_rt_overlap/ 2>/dev/null
