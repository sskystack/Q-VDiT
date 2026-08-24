#!/usr/bin/env bash
# Controlled TQE-only rank ablation for OpenSora W4A6.
# The baseline quant config is intentional: it sets frame_axis=BASELINE, so
# neither legacy MTD, MTD-v2, nor official TMD is part of this experiment.
set -Eeuo pipefail

REPO=${REPO:-/home/zhouchongtian/quantization/Q-VDiT-mtd-v2-20260810}
GPU_ID=${GPU_ID:?GPU_ID must be set}
TQE_RANK=${TQE_RANK:?TQE_RANK must be set}
CALIB_RUN_ROOT=${CALIB_RUN_ROOT:?CALIB_RUN_ROOT must be set}
VBENCH_RUN_ROOT=${VBENCH_RUN_ROOT:?VBENCH_RUN_ROOT must be set}
PIPELINE_ROOT=${PIPELINE_ROOT:?PIPELINE_ROOT must be set}
SHARED_CALIB_DATA=${SHARED_CALIB_DATA:-/home/zhouchongtian/quantization/experiments/mtd_v2_20260810_retry1/shared_calibration/calib_data.pt}

PYTHON=${PYTHON:-/home/zhouchongtian/miniconda3/envs/qvdit/bin/python}
MODEL_CKPT=/home/zhouchongtian/quantization/Q-VDiT/logs/split_ckpt/OpenSora-v1-HQ-16x512x512-split.pth
MODEL_CONFIG=t2v/configs/quant/opensora/16x512x512_mtd_v2_fp16_flash.py
QUANT_CONFIG=t2v/configs/quant/opensora/w4a6_baseline.yaml
WEIGHT_MP=t2v/configs/quant/opensora/mixed_precision/weight_4_mp.yaml
ACT_MP=t2v/configs/quant/opensora/mixed_precision/act_6_mp.yaml
TEXT_ENCODER_BYPASS_EMBEDS=t2v/utils_files/text_embeds.pth
VBENCH_RUNNER=/home/zhouchongtian/quantization/qvdit_flash_bf16/tools/run_baseline_fp32t5_vbench_gpu2.sh
EMBED_ROOT=/home/zhouchongtian/quantization/qvdit_flash_bf16/logs_fp16_flash/vbench_official_embeds_fp32t5_seed42

[[ "$TQE_RANK" =~ ^[1-9][0-9]*$ ]] || { echo "TQE_RANK must be a positive integer" >&2; exit 2; }
for path in "$REPO/$MODEL_CONFIG" "$REPO/$QUANT_CONFIG" "$SHARED_CALIB_DATA" "$MODEL_CKPT" "$VBENCH_RUNNER"; do
    [[ -s "$path" ]] || { echo "missing or empty file: $path" >&2; exit 2; }
done
grep -q 'frame_axis: "BASELINE"' "$REPO/$QUANT_CONFIG" || {
    echo "TQE-only run requires frame_axis=BASELINE; refusing non-baseline config" >&2
    exit 2
}
for root in "$PIPELINE_ROOT" "$CALIB_RUN_ROOT" "$VBENCH_RUN_ROOT"; do
    [[ ! -e "$root" ]] || { echo "refusing to reuse existing output root: $root" >&2; exit 2; }
done

mkdir -p "$PIPELINE_ROOT" "$CALIB_RUN_ROOT/shared_calibration" "$CALIB_RUN_ROOT/calibration"
exec > >(tee -a "$PIPELINE_ROOT/pipeline.log") 2>&1

stamp() { printf '[%s] %s\n' "$(date -Is)" "$*"; }
fail() {
    stamp "FAILED: $*"
    printf 'FAILED %s: %s\n' "$(date -Is)" "$*" > "$PIPELINE_ROOT/status.txt"
    exit 1
}
trap 'fail "command failed at line $LINENO: $BASH_COMMAND"' ERR

wait_for_gpu_idle() {
    while true; do
        local pids
        pids=$(nvidia-smi --id="$GPU_ID" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
            | sed '/^[[:space:]]*$/d' | paste -sd, -)
        if [[ -z "$pids" ]]; then
            stamp "GPU $GPU_ID is idle"
            return
        fi
        stamp "GPU $GPU_ID occupied by compute PID(s): $pids; retrying in 30 seconds"
        sleep 30
    done
}

{
    date -Is
    printf 'experiment=TQE_only_rank_%s\n' "$TQE_RANK"
    printf 'method=token_axis_TQE_only\n'
    printf 'explicit_exclusions=legacy_MTD,MTD_v2,official_TMD\n'
    printf 'quantization=W4A6\nframe_axis=BASELINE\n'
    printf 'tqe_rank=%s\n' "$TQE_RANK"
    printf 'gpu=%s\nrepo=%s\n' "$GPU_ID" "$REPO"
    printf 'shared_calibration=%s\n' "$SHARED_CALIB_DATA"
    printf 'calibration=FP16_FlashAttention_DDIM50_CFG4_seed42_batch1_samples10_iters10000_GradScaler\n'
    printf 'vbench=FP16_FlashAttention_FP32T5_DDIM100_CFG4_seed42_per_group_batch1\n'
    printf 'model_checkpoint=%s\nmodel_config=%s\nquant_config=%s\n' "$MODEL_CKPT" "$MODEL_CONFIG" "$QUANT_CONFIG"
    git -C "$REPO" rev-parse HEAD 2>/dev/null || printf 'git_head=unavailable\n'
    git -C "$REPO" status --short 2>/dev/null || printf 'git_status=unavailable\n'
    sha256sum "$REPO/$MODEL_CONFIG" "$REPO/$QUANT_CONFIG" "$SHARED_CALIB_DATA"
    nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
} > "$PIPELINE_ROOT/experiment_manifest.txt"

ln -s "$SHARED_CALIB_DATA" "$CALIB_RUN_ROOT/shared_calibration/calib_data.pt"
printf 'CALIBRATION_PENDING %s\n' "$(date -Is)" > "$PIPELINE_ROOT/status.txt"
printf 'TQE_ONLY_CALIBRATION_STARTED %s\n' "$(date -Is)" > "$CALIB_RUN_ROOT/status.txt"

wait_for_gpu_idle
stamp "Starting TQE-only rank-$TQE_RANK calibration on GPU $GPU_ID"
cd "$REPO"
export PYTHONPATH="$REPO:$REPO/t2v"
export QVDIT_TQE_RANK="$TQE_RANK"
CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON" -u t2v/scripts/calib.py "$MODEL_CONFIG" \
    --gpu 0 \
    --ckpt_path "$MODEL_CKPT" \
    --calib_config "$QUANT_CONFIG" \
    --precompute_text_embeds "$TEXT_ENCODER_BYPASS_EMBEDS" \
    --outdir "$CALIB_RUN_ROOT/calibration" \
    --calib_data "$CALIB_RUN_ROOT/shared_calibration/calib_data.pt" \
    --part_fp \
    --time_mp_config_weight "$WEIGHT_MP" \
    --time_mp_config_act "$ACT_MP" \
    --use_grad_scaler \
    2>&1 | tee "$CALIB_RUN_ROOT/calibration/run.log"

[[ -s "$CALIB_RUN_ROOT/calibration/ckpt.pth" ]] || fail "final calibration checkpoint is missing"
printf 'TQE_ONLY_CALIBRATION_COMPLETE %s\n' "$(date -Is)" > "$CALIB_RUN_ROOT/status.txt"
printf 'CALIBRATION_COMPLETE_VBENCH_PENDING %s\n' "$(date -Is)" > "$PIPELINE_ROOT/status.txt"

stamp "Calibration complete; starting VBench on GPU $GPU_ID"
QVDIT_TQE_RANK="$TQE_RANK" \
REPO="$REPO" GPU_INDEX="$GPU_ID" WAIT_FOR_GPU_IDLE=1 GPU_POLL_SECONDS=30 \
QUANT_CFG="$REPO/$QUANT_CONFIG" QUANT_CKPT="$CALIB_RUN_ROOT/calibration/ckpt.pth" \
EMBED_ROOT="$EMBED_ROOT" RUN_ROOT="$VBENCH_RUN_ROOT" \
LOCK_FILE="$PIPELINE_ROOT/vbench_gpu${GPU_ID}.lock" \
EXPERIMENT_NAME="TQE_only_rank${TQE_RANK}_W4A6_FP16_FlashAttention_DDIM100_CFG4_seed42" \
bash "$VBENCH_RUNNER"

[[ -s "$VBENCH_RUN_ROOT/vbench_8metrics_summary.json" ]] || fail "VBench summary is missing"
printf 'COMPLETED %s\n' "$(date -Is)" > "$PIPELINE_ROOT/status.txt"
stamp "ALL_COMPLETE"
