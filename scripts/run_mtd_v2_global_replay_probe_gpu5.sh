#!/usr/bin/env bash
set -Eeuo pipefail

REPO=${REPO:-/home/zhouchongtian/quantization/Q-VDiT-mtd-v2-20260810}
PYTHON=${PYTHON:-/home/zhouchongtian/miniconda3/envs/qvdit/bin/python}
GPU_ID=${GPU_ID:-5}
RUN_ROOT=${RUN_ROOT:-/home/zhouchongtian/quantization/experiments/mtd_v2_global_probe_gpu5_20260812}
SOURCE_ROOT=/home/zhouchongtian/quantization/experiments/mtd_v2_gpu6_20260811_retry7/calibration
RESUME_STATE=$SOURCE_ROOT/reconstruction_state_iter_00009500.pth
CALIB_DATA=/home/zhouchongtian/quantization/experiments/mtd_v2_20260810_retry1/shared_calibration/calib_data.pt
MODEL_CKPT=/home/zhouchongtian/quantization/Q-VDiT/logs/split_ckpt/OpenSora-v1-HQ-16x512x512-split.pth
MODEL_CONFIG=t2v/configs/quant/opensora/16x512x512_mtd_v2_fp16_flash.py
CALIB_CONFIG=t2v/configs/quant/opensora/w4a6_mtd_v2_replay_probe.yaml
WEIGHT_MP=t2v/configs/quant/opensora/mixed_precision/weight_4_mp.yaml
ACT_MP=t2v/configs/quant/opensora/mixed_precision/act_6_mp.yaml

[[ ! -e "$RUN_ROOT" ]] || { echo "Refusing existing probe root: $RUN_ROOT" >&2; exit 2; }
for path in "$PYTHON" "$RESUME_STATE" "$CALIB_DATA" "$MODEL_CKPT"; do
    [[ -s "$path" ]] || { echo "missing: $path" >&2; exit 2; }
done

gpu_uuid=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits |
    awk -F', ' -v gpu="$GPU_ID" '$1 == gpu {print $2}')
while true; do
    pids=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits 2>/dev/null |
        awk -F', ' -v uuid="$gpu_uuid" '$1 == uuid {print $2}' | paste -sd, -)
    [[ -n "$pids" ]] || break
    printf '[%s] GPU%s occupied by PID(s) %s; waiting 30s\n' "$(date -Is)" "$GPU_ID" "$pids"
    sleep 30
done

mkdir -p "$RUN_ROOT"
cd "$REPO"
export PYTHONPATH="$REPO:$REPO/t2v"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
{
    date -Is
    printf 'purpose=replay_iteration_9500_to_10000_for_unformatted_mtd_v2_global_loss_and_gradient\n'
    printf 'gpu=%s\nresume=%s\ncalib_data=%s\nmodel=%s\nconfig=%s\n' \
        "$GPU_ID" "$RESUME_STATE" "$CALIB_DATA" "$MODEL_CKPT" "$CALIB_CONFIG"
    sha256sum "$RESUME_STATE" "$CALIB_DATA" "$CALIB_CONFIG"
} > "$RUN_ROOT/experiment_manifest.txt"
printf 'RUNNING %s\n' "$(date -Is)" > "$RUN_ROOT/status.txt"

on_error() { printf 'FAILED %s\n' "$(date -Is)" > "$RUN_ROOT/status.txt"; }
trap on_error ERR

CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON" t2v/scripts/calib.py "$MODEL_CONFIG" \
    --gpu 0 \
    --ckpt_path "$MODEL_CKPT" \
    --calib_config "$CALIB_CONFIG" \
    --precompute_text_embeds t2v/utils_files/text_embeds.pth \
    --outdir "$RUN_ROOT" \
    --calib_data "$CALIB_DATA" \
    --resume_reconstruction "$RESUME_STATE" \
    --reconstruction_checkpoint_interval 0 \
    --numeric_monitor_interval 100 \
    --numeric_monitor_detailed_interval 100 \
    --part_fp \
    --time_mp_config_weight "$WEIGHT_MP" \
    --time_mp_config_act "$ACT_MP" \
    --use_grad_scaler \
    2>&1 | tee "$RUN_ROOT/pipeline.log"

[[ -s "$RUN_ROOT/numeric_monitor.jsonl" ]]
[[ -s "$RUN_ROOT/mtd_v2_gradients.jsonl" ]]
printf 'COMPLETED %s\n' "$(date -Is)" > "$RUN_ROOT/status.txt"
