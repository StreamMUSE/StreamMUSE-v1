#!/bin/bash
# Batch comparison: Offline vs FakeRT equivalent, then Offline vs FakeRT overlap
# Usage: bash scripts/compare_all.sh

set -e

REPORT_DIR="developing-logs/2026-4-23/debug-reports"
mkdir -p "${REPORT_DIR}"

echo "=== Comparing Offline vs FakeRT Equivalent (interval=4, length=4) ==="
uv run python scripts/debug_inference_consistency.py \
  output/debug/offline \
  output/debug/fake_rt_equivalent \
  output/debug/offline

echo ""
echo "JSON report: output/debug/offline/inference_consistency_report.json"

echo ""
echo "=== Comparing Offline vs FakeRT Overlap (interval=4, length=8) ==="
uv run python scripts/debug_inference_consistency.py \
  output/debug/offline \
  output/debug/fake_rt_overlap \
  output/debug/offline_prefix4

echo ""
echo "All comparisons done."
