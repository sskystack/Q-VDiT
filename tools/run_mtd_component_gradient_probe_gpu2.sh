#!/usr/bin/env bash
set -Eeuo pipefail

source /home/zhouchongtian/miniconda3/etc/profile.d/conda.sh
conda activate qvdit

ROOT=/home/zhouchongtian/quantization/qvdit_flash_bf16
OUT="$ROOT/logs_fp16_flash/mtd_component_gradient_probe_resume9500_gpu2_0728"
cd "$ROOT"
export PYTHONPATH="$ROOT:$ROOT/t2v"
export CUDA_VISIBLE_DEVICES=2

mkdir -p "$OUT/calibration"
python t2v/scripts/calib.py \
  ./t2v/configs/quant/opensora/16x512x512_fp16_flash_50steps.py \
  --ckpt_path /home/zhouchongtian/quantization/Q-VDiT/logs/split_ckpt/OpenSora-v1-HQ-16x512x512-split.pth \
  --calib_config ./t2v/configs/quant/opensora/w4a6_mtd_gradient_probe.yaml \
  --calib_data "$ROOT/logs_fp16_flash/formal_w4a6_mtd_samples10_gpu5_0724/calib_data_ddim50_cfg4/calib_data.pt" \
  --precompute_text_embeds /home/zhouchongtian/quantization/new/logs_50steps/text_embeds_samples_10.pt \
  --outdir "$OUT/calibration" \
  --part_fp \
  --time_mp_config_weight ./t2v/configs/quant/opensora/mixed_precision/weight_4_mp.yaml \
  --time_mp_config_act ./t2v/configs/quant/opensora/mixed_precision/act_6_mp.yaml \
  --use_grad_scaler \
  --resume_reconstruction "$ROOT/logs_fp16_flash/formal_w4a6_mtd_gradscaler_samples10_gpu5_0725/calibration/reconstruction_state_iter_00009500.pth" \
  --reconstruction_checkpoint_interval 500 \
  2>&1 | tee "$OUT/console.log"
