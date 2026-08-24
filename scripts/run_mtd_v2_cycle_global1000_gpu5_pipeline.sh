#!/usr/bin/env bash
set -Eeuo pipefail

REPO=${REPO:-/home/zhouchongtian/quantization/Q-VDiT-mtd-v2-20260810}
GPU_ID=${GPU_ID:-5}
CALIB_RUN_ROOT=${CALIB_RUN_ROOT:-/home/zhouchongtian/quantization/experiments/mtd_v2_cycle_global1000_gpu5_20260813_r1}
VBENCH_RUN_ROOT=${VBENCH_RUN_ROOT:-/home/zhouchongtian/quantization/experiments/mtd_v2_cycle_global1000_gpu5_vbench_20260813_r1}
SHARED_CALIB_DATA=${SHARED_CALIB_DATA:-/home/zhouchongtian/quantization/experiments/mtd_v2_20260810_retry1/shared_calibration/calib_data.pt}
CALIB_CONFIG=t2v/configs/quant/opensora/w4a6_mtd_v2_cycle_global1000.yaml
QUANT_CKPT=$CALIB_RUN_ROOT/calibration/ckpt.pth
BASELINE_RUNNER=/home/zhouchongtian/quantization/qvdit_flash_bf16/tools/run_baseline_fp32t5_vbench_gpu2.sh
EMBED_ROOT=/home/zhouchongtian/quantization/qvdit_flash_bf16/logs_fp16_flash/vbench_official_embeds_fp32t5_seed42
PIPELINE_ROOT=${PIPELINE_ROOT:-/home/zhouchongtian/quantization/experiments/mtd_v2_cycle_global1000_gpu5_pipeline_20260813_r1}

for path in "$SHARED_CALIB_DATA" "$REPO/$CALIB_CONFIG" "$BASELINE_RUNNER"; do
    [[ -s "$path" ]] || { echo "missing or empty file: $path" >&2; exit 2; }
done
[[ ! -e "$PIPELINE_ROOT" ]] || { echo "pipeline root exists: $PIPELINE_ROOT" >&2; exit 2; }
[[ ! -e "$CALIB_RUN_ROOT" ]] || { echo "calibration root exists: $CALIB_RUN_ROOT" >&2; exit 2; }
[[ ! -e "$VBENCH_RUN_ROOT" ]] || { echo "VBench root exists: $VBENCH_RUN_ROOT" >&2; exit 2; }
mkdir -p "$PIPELINE_ROOT"
exec > >(tee -a "$PIPELINE_ROOT/pipeline.log") 2>&1
stamp() { printf '[%s] %s\n' "$(date -Is)" "$*"; }
fail() { stamp "FAILED: $*"; printf 'FAILED %s: %s\n' "$(date -Is)" "$*" > "$PIPELINE_ROOT/status.txt"; exit 1; }
trap 'fail "command failed at line $LINENO: $BASH_COMMAND"' ERR

{
    date -Is
    printf 'experiment=mtd_v2_cycle_global_weight_x1000\n'
    printf 'controlled_change=global_relation_weight_0.1_to_100.0\n'
    printf 'gpu=%s\nrepo=%s\ncalibration_root=%s\nvbench_root=%s\n' \
        "$GPU_ID" "$REPO" "$CALIB_RUN_ROOT" "$VBENCH_RUN_ROOT"
    printf 'shared_calibration=%s\ncalibration_config=%s\n' "$SHARED_CALIB_DATA" "$CALIB_CONFIG"
    printf 'vbench=FP16_FlashAttention_DDIM100_CFG4_seed42_batch1_FP32T5\n'
    sha256sum "$REPO/$CALIB_CONFIG" "$SHARED_CALIB_DATA"
} > "$PIPELINE_ROOT/experiment_manifest.txt"

printf 'CALIBRATION_PENDING %s\n' "$(date -Is)" > "$PIPELINE_ROOT/status.txt"
stamp "Starting controlled 10k calibration on GPU $GPU_ID"
cd "$REPO"
GPU_ID="$GPU_ID" WAIT_FOR_GPU_IDLE=1 GPU_POLL_SECONDS=30 \
RUN_ROOT="$CALIB_RUN_ROOT" SHARED_CALIB_DATA="$SHARED_CALIB_DATA" \
CALIB_CONFIG="$CALIB_CONFIG" \
bash scripts/run_mtd_v2_formal_calibration_gpu6.sh
[[ -s "$QUANT_CKPT" ]] || fail "final calibration checkpoint is missing"
printf 'CALIBRATION_COMPLETE_VBENCH_PENDING %s\n' "$(date -Is)" > "$PIPELINE_ROOT/status.txt"

stamp "Calibration complete; starting baseline-equivalent VBench workflow on GPU $GPU_ID"
REPO="$REPO" GPU_INDEX="$GPU_ID" WAIT_FOR_GPU_IDLE=1 GPU_POLL_SECONDS=30 \
QUANT_CFG="$REPO/$CALIB_CONFIG" QUANT_CKPT="$QUANT_CKPT" \
EMBED_ROOT="$EMBED_ROOT" RUN_ROOT="$VBENCH_RUN_ROOT" \
LOCK_FILE="$PIPELINE_ROOT/vbench_gpu${GPU_ID}.lock" \
EXPERIMENT_NAME="MTD_V2_cycle_global1000_W4A6_FP16_FlashAttention_DDIM100_CFG4_seed42" \
bash "$BASELINE_RUNNER"

[[ -s "$VBENCH_RUN_ROOT/vbench_8metrics_summary.json" ]] || fail "VBench summary is missing"
printf 'COMPLETED %s\n' "$(date -Is)" > "$PIPELINE_ROOT/status.txt"
stamp "ALL_COMPLETE"
