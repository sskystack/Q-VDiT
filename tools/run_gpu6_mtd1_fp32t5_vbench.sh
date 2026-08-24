#!/usr/bin/env bash
set -Eeuo pipefail

REPO=${REPO:-/home/zhouchongtian/quantization/qvdit_flash_bf16}

export REPO
export GPU_INDEX=6
export WAIT_FOR_GPU_IDLE=0
export QUANT_CFG="$REPO/t2v/configs/quant/opensora/w4a6_mtd.yaml"
export QUANT_CKPT="$REPO/logs_fp16_flash/formal_w4a6_mtd_gradscaler_samples10_gpu5_0725/calibration/ckpt.pth"
export EMBED_ROOT="$REPO/logs_fp16_flash/vbench_official_embeds_fp32t5_seed42"
export RUN_ROOT="$REPO/logs_fp16_flash/vbench_mtd1_fp32t5_fp16flash_ddim100_cfg4_seed42_gpu6_0803"
export LOCK_FILE="$REPO/logs_fp16_flash/.vbench_mtd1_fp32t5_gpu6.lock"
export EXPERIMENT_NAME="FP16_FlashAttention_W4A6_MTD1_with_FP32_T5_embeddings_GPU6"

exec bash "$REPO/tools/run_baseline_fp32t5_vbench_gpu2.sh"
