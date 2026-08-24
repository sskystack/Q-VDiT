#!/usr/bin/env bash
set -Eeuo pipefail

REPO=${REPO:-/home/zhouchongtian/quantization/qvdit_flash_bf16}

export EXPERIMENT_NAME=mtd1
export QUANT_CFG="$REPO/t2v/configs/quant/opensora/w4a6_mtd.yaml"
export MODEL_CKPT=/home/zhouchongtian/quantization/Q-VDiT/logs/split_ckpt/OpenSora-v1-HQ-16x512x512-split.pth
export CALIB_DATA_DIR="$REPO/logs/formal_baseline_calib_cfg4_ddim50_gpu2_20260806/data"
export CALIB_ROOT="$REPO/logs/formal_mtd1_calib_cfg4_ddim50_gpu5_20260806"
export CALIB_OUT="$CALIB_ROOT/calibration"
export RUN_ROOT="$REPO/logs/formal_vbench_mtd1_cfg4_ddim100_seed42_gpu5_20260806"

exec bash "$REPO/tools/run_formal_calib_vbench_pipeline.sh" 5
