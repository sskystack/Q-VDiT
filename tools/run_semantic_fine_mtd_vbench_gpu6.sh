#!/usr/bin/env bash
set -Eeuo pipefail

# Semantic Fine-MTD experiment on GPU 6.
# The canonical pipeline enforces FP32 T5, FP16 DiT/VAE, GradScaler,
# seed-42 replay, calibration-data reuse, and separate VBench groups.
if [[ -z "${TMUX:-}" ]]; then
    echo "Run this experiment from a new dedicated tmux window." >&2
    exit 2
fi

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-semantic_fine_mtd}
export MODEL_CKPT=${MODEL_CKPT:-/home/zhouchongtian/quantization/Q-VDiT/logs/split_ckpt/OpenSora-v1-HQ-16x512x512-split.pth}
export QUANT_CFG=${QUANT_CFG:-$ROOT/t2v/configs/quant/opensora/w4a6_mtd_semantic_fine.yaml}
export CALIB_ROOT=${CALIB_ROOT:-$ROOT/logs/formal_semantic_fine_mtd_calib_cfg4_ddim50}
export CALIB_DATA_DIR=${CALIB_DATA_DIR:-$ROOT/logs_fp16_flash/formal_w4a6_mtd_samples10_gpu5_0724/calib_data_ddim50_cfg4}
export RUN_ROOT=${RUN_ROOT:-$ROOT/logs/formal_vbench_semantic_fine_mtd_cfg4_ddim100_seed42_gpu6}

exec bash "$ROOT/tools/run_formal_calib_vbench_pipeline.sh" 6
