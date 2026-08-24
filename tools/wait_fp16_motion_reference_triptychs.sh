#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/home/zhouchongtian/quantization/qvdit_flash_bf16
RUN="$ROOT/logs_fp16_flash/vbench_fp16_reference_seed42_motion_profile_0731"
MANIFEST="$RUN/analysis/inspection_manifest.txt"

while [[ ! -s "$MANIFEST" ]]; do
    sleep 60
done

python "$ROOT/tools/make_motion_reference_triptychs.py" \
    --manifest "$MANIFEST" \
    --fp16-videos "$RUN/fp16_subject_opensora" \
    --baseline-videos "$ROOT/logs_fp16_flash/vbench_baseline_fp16gs_final10000/subject_opensora" \
    --mtd-videos "$ROOT/logs_fp16_flash/vbench_mtdfp16gs_final10000/subject_opensora" \
    --output "$RUN/manual_triptychs" \
    2>&1 | tee "$RUN/triptych_generation.log"
