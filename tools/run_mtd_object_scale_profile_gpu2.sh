#!/usr/bin/env bash
set -euo pipefail

source /home/zhouchongtian/miniconda3/etc/profile.d/conda.sh
export PYTHONPATH="${PYTHONPATH:-}"
conda activate vbench

ROOT=/home/zhouchongtian/quantization/qvdit_flash_bf16
OUT="$ROOT/logs_fp16_flash/mtd_object_scale_search_gpu2_0731"
FEATURES="$ROOT/logs_fp16_flash/mtd_seed42_motion_profile_0730/features"
FP16="$ROOT/logs_fp16_flash/vbench_fp16_reference_seed42_motion_profile_0731/fp16_subject_opensora"
MANIFEST="$ROOT/tools/mtd_seed42_manifest.json"

mkdir -p "$OUT"
cd /home/zhouchongtian/quantization/eval/Vbench
export PYTHONPATH="/home/zhouchongtian/quantization/eval/Vbench:$ROOT:$ROOT/t2v"
export CUDA_VISIBLE_DEVICES=2

python "$ROOT/tools/profile_mtd_object_scale_search.py" \
  --manifest "$MANIFEST" \
  --profile-dir "$FEATURES" \
  --fp16-videos "$FP16" \
  --output "$OUT/analysis" \
  --feature-view cond \
  --device cuda:0 \
  2>&1 | tee "$OUT/profile.log"

echo "COMPLETED $(date '+%F %T')" > "$OUT/status.txt"
