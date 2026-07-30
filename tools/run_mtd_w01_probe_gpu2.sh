#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/home/zhouchongtian/quantization/qvdit_flash_bf16
OUT="$ROOT/logs_fp16_flash/formal_w4a6_mtd_w01_gradprobe_samples10_gpu2_0728"
GPU_ID=2

mkdir -p "$OUT"
if [[ -e "$OUT/calibration/run.log" ]]; then
    echo "Refusing to overwrite an existing calibration: $OUT/calibration" >&2
    exit 1
fi

GPU_UUID=$(nvidia-smi --id="$GPU_ID" --query-gpu=uuid --format=csv,noheader)
echo "WAITING gpu=$GPU_ID uuid=$GPU_UUID $(date '+%F %T')" | tee "$OUT/status.txt"
idle_checks=0
while (( idle_checks < 6 )); do
    compute_uuids=$(nvidia-smi \
        --query-compute-apps=gpu_uuid --format=csv,noheader)
    if grep -Fxq "$GPU_UUID" <<<"$compute_uuids"; then
        idle_checks=0
        echo "WAITING gpu=$GPU_ID busy $(date '+%F %T')" > "$OUT/status.txt"
    else
        ((idle_checks += 1))
        echo "WAITING gpu=$GPU_ID idle_check=$idle_checks/6 $(date '+%F %T')" \
            > "$OUT/status.txt"
    fi
    sleep 10
done

echo "RUNNING gpu=$GPU_ID $(date '+%F %T')" | tee "$OUT/status.txt"
source /home/zhouchongtian/miniconda3/etc/profile.d/conda.sh
conda activate qvdit
cd "$ROOT"
export PYTHONPATH="$ROOT:$ROOT/t2v"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

mkdir -p "$OUT/calibration"
python t2v/scripts/calib.py \
  ./t2v/configs/quant/opensora/16x512x512_fp16_flash_50steps.py \
  --ckpt_path /home/zhouchongtian/quantization/Q-VDiT/logs/split_ckpt/OpenSora-v1-HQ-16x512x512-split.pth \
  --calib_config ./t2v/configs/quant/opensora/w4a6_mtd_w01_probe.yaml \
  --calib_data "$ROOT/logs_fp16_flash/formal_w4a6_mtd_samples10_gpu5_0724/calib_data_ddim50_cfg4/calib_data.pt" \
  --precompute_text_embeds /home/zhouchongtian/quantization/new/logs_50steps/text_embeds_samples_10.pt \
  --outdir "$OUT/calibration" \
  --part_fp \
  --time_mp_config_weight ./t2v/configs/quant/opensora/mixed_precision/weight_4_mp.yaml \
  --time_mp_config_act ./t2v/configs/quant/opensora/mixed_precision/act_6_mp.yaml \
  --use_grad_scaler \
  --numeric_monitor_interval 25 \
  --numeric_monitor_detailed_interval 100 \
  --reconstruction_checkpoint_interval 500 \
  2>&1 | tee "$OUT/console.log"

python tools/summarize_mtd_subterm_gradients.py \
  "$OUT/calibration/loss_component_gradients.jsonl" \
  --output-prefix "$OUT/calibration/mtd_subterm_gradient_summary" \
  2>&1 | tee "$OUT/gradient_summary.log"

echo "COMPLETED gpu=$GPU_ID $(date '+%F %T')" | tee "$OUT/status.txt"
