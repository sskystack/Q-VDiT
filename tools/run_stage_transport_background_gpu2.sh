#!/usr/bin/env bash
set -euo pipefail

source /home/zhouchongtian/miniconda3/etc/profile.d/conda.sh

ROOT=/home/zhouchongtian/quantization/qvdit_flash_bf16
OUT="$ROOT/logs_fp16_flash/stage_transport_background_profile_gpu2_0731"
FEATURES="$ROOT/logs_fp16_flash/mtd_seed42_motion_profile_0730/features"
FP16="$ROOT/logs_fp16_flash/vbench_fp16_reference_seed42_motion_profile_0731/fp16_subject_opensora"
BASELINE="$ROOT/logs_fp16_flash/vbench_baseline_fp16gs_final10000/subject_opensora"
MTD="$ROOT/logs_fp16_flash/vbench_mtdfp16gs_final10000/subject_opensora"

mkdir -p "$OUT/stage_transport" "$OUT/background_motion"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES=2

conda activate qvdit
export PYTHONPATH="$ROOT:$ROOT/t2v"
python tools/profile_stage_sparse_transport.py \
  --profile-dir "$FEATURES" \
  --output "$OUT/stage_transport" \
  --feature-view cond \
  --device cuda:0 \
  2>&1 | tee "$OUT/stage_transport.log"

conda activate vbench
cd /home/zhouchongtian/quantization/eval/Vbench
export PYTHONPATH=/home/zhouchongtian/quantization/eval/Vbench
python "$ROOT/tools/profile_foreground_background_motion.py" \
  --variant "fp16=$FP16" \
  --variant "baseline=$BASELINE" \
  --variant "mtd=$MTD" \
  --prompts \
    "a giraffe taking a peaceful walk" \
    "a dog enjoying a peaceful walk" \
    "a horse taking a peaceful walk" \
    "a sheep taking a peaceful walk" \
    "a zebra taking a peaceful walk" \
    "an elephant taking a peaceful walk" \
  --output "$OUT/background_motion" \
  --device cuda:0 \
  2>&1 | tee "$OUT/background_motion.log"

echo "COMPLETED $(date '+%F %T')" > "$OUT/status.txt"
