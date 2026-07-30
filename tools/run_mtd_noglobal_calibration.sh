#!/usr/bin/env bash
set -Eeuo pipefail

GPU_ID=${1:?GPU index required}
CONFIG_NAME=${2:?config filename required}
EXPERIMENT_NAME=${3:?experiment name required}
ALLOW_SHARED=${4:-0}

ROOT=/home/zhouchongtian/quantization/qvdit_flash_bf16
OUT="$ROOT/logs_fp16_flash/$EXPERIMENT_NAME"
CALIB="$OUT/calibration"
CONFIG="$ROOT/t2v/configs/quant/opensora/$CONFIG_NAME"
CALIB_DATA="$ROOT/logs_fp16_flash/formal_w4a6_mtd_samples10_gpu5_0724/calib_data_ddim50_cfg4/calib_data.pt"
TEXT_EMBEDS=/home/zhouchongtian/quantization/new/logs_50steps/text_embeds_samples_10.pt

mkdir -p "$OUT"
exec > >(tee -a "$OUT/console.log") 2>&1
trap 'echo "FAILED gpu=$GPU_ID $(date "+%F %T")" | tee "$OUT/status.txt"' ERR

[[ -s "$CONFIG" ]]
[[ -s "$CALIB_DATA" ]]
[[ -s "$TEXT_EMBEDS" ]]
if [[ -e "$CALIB/run.log" ]]; then
    echo "Refusing to overwrite existing calibration: $CALIB" >&2
    exit 1
fi

GPU_UUID=$(nvidia-smi --id="$GPU_ID" --query-gpu=uuid --format=csv,noheader)
if [[ "$ALLOW_SHARED" != "1" ]]; then
    echo "WAITING gpu=$GPU_ID uuid=$GPU_UUID $(date '+%F %T')" | tee "$OUT/status.txt"
    idle_checks=0
    while (( idle_checks < 3 )); do
        compute_uuids=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null || true)
        if grep -Fxq "$GPU_UUID" <<<"$compute_uuids"; then
            idle_checks=0
        else
            ((idle_checks += 1))
        fi
        sleep 10
    done
else
    echo "SHARED_START gpu=$GPU_ID uuid=$GPU_UUID $(date '+%F %T')" | tee "$OUT/status.txt"
fi

echo "RUNNING gpu=$GPU_ID config=$CONFIG_NAME $(date '+%F %T')" | tee "$OUT/status.txt"
source /home/zhouchongtian/miniconda3/etc/profile.d/conda.sh
conda activate qvdit
cd "$ROOT"
export PYTHONPATH="$ROOT:$ROOT/t2v"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$CALIB"
python t2v/scripts/calib.py \
  ./t2v/configs/quant/opensora/16x512x512_fp16_flash_50steps.py \
  --ckpt_path /home/zhouchongtian/quantization/Q-VDiT/logs/split_ckpt/OpenSora-v1-HQ-16x512x512-split.pth \
  --calib_config "$CONFIG" \
  --calib_data "$CALIB_DATA" \
  --precompute_text_embeds "$TEXT_EMBEDS" \
  --outdir "$CALIB" \
  --part_fp \
  --time_mp_config_weight ./t2v/configs/quant/opensora/mixed_precision/weight_4_mp.yaml \
  --time_mp_config_act ./t2v/configs/quant/opensora/mixed_precision/act_6_mp.yaml \
  --use_grad_scaler \
  --numeric_monitor_interval 25 \
  --numeric_monitor_detailed_interval 100 \
  --reconstruction_checkpoint_interval 1000

python tools/summarize_mtd_subterm_gradients.py \
  "$CALIB/loss_component_gradients.jsonl" \
  --output-prefix "$CALIB/mtd_subterm_gradient_summary" \
  2>&1 | tee "$OUT/gradient_summary.log"

echo "CALIBRATION_COMPLETED gpu=$GPU_ID $(date '+%F %T')" | tee "$OUT/status.txt"
