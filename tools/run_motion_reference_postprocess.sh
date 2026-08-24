#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/home/zhouchongtian/quantization/qvdit_flash_bf16
VBENCH_ROOT=/home/zhouchongtian/quantization/eval/Vbench
RUN="$ROOT/logs_fp16_flash/vbench_fp16_reference_seed42_motion_profile_0731"
FP16_VIDEOS="$RUN/fp16_subject_opensora"
BASELINE_ROOT="$ROOT/logs_fp16_flash/vbench_baseline_fp16gs_final10000"
MTD_ROOT="$ROOT/logs_fp16_flash/vbench_mtdfp16gs_final10000"
FP16_TEMPORAL="$RUN/fp16_temporal_custom_eval"
BASELINE_TEMPORAL="$RUN/baseline_temporal_custom_eval"
MTD_TEMPORAL="$RUN/mtd_temporal_custom_eval"
STATUS="$RUN/postprocess_status.txt"

exec > >(tee -a "$RUN/postprocess.log") 2>&1
export PYTHONPATH="${PYTHONPATH:-}"
source /home/zhouchongtian/miniconda3/etc/profile.d/conda.sh
conda activate vbench
cd "$VBENCH_ROOT"
export CUDA_VISIBLE_DEVICES=2
export VBENCH_CACHE_DIR=/home/zhouchongtian/quantization/models/vbench

run_temporal() {
    local variant=$1
    local videos=$2
    local output=$3
    if [[ -s "$output/temporal_flickering_eval_results.json" ]]; then
        echo "SKIP temporal variant=$variant"
        return
    fi
    mkdir -p "$output"
    python evaluate.py \
        --videos_path "$videos" \
        --output_path "$output" \
        --dimension temporal_flickering \
        --load_ckpt_from_local True \
        --custom_input \
        2>&1 | tee "$RUN/${variant}_temporal_custom.log"
}

echo "RUNNING_TEMPORAL $(date '+%F %T')" | tee "$STATUS"
run_temporal fp16 "$FP16_VIDEOS" "$FP16_TEMPORAL" &
pid_fp16=$!
run_temporal baseline "$BASELINE_ROOT/subject_opensora" "$BASELINE_TEMPORAL" &
pid_baseline=$!
run_temporal mtd "$MTD_ROOT/subject_opensora" "$MTD_TEMPORAL" &
pid_mtd=$!
wait "$pid_fp16" "$pid_baseline" "$pid_mtd"

python - "$FP16_TEMPORAL" "$BASELINE_TEMPORAL" "$MTD_TEMPORAL" <<'PY'
import json
import math
import sys
from pathlib import Path

for root in map(Path, sys.argv[1:]):
    path = root / "temporal_flickering_eval_results.json"
    value, details = json.loads(path.read_text())["temporal_flickering"]
    if not math.isfinite(float(value)) or len(details) != 72:
        raise RuntimeError(f"Invalid custom temporal result: {path}, value={value}, n={len(details)}")
PY

echo "ANALYZING $(date '+%F %T')" | tee "$STATUS"
python "$ROOT/tools/analyze_fp16_motion_reference.py" \
    --fp16-eval "$RUN/fp16_eval" "$FP16_TEMPORAL" \
    --baseline-eval "$BASELINE_ROOT/subject_eval" "$BASELINE_TEMPORAL" \
    --mtd-eval "$MTD_ROOT/subject_eval" "$MTD_TEMPORAL" \
    --output "$RUN/analysis" \
    2>&1 | tee "$RUN/analysis_fixed.log"

echo "COMPLETED $(date '+%F %T')" | tee "$STATUS"
