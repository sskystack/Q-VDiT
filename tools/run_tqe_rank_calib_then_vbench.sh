#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 GPU_INDEX TQE_RANK" >&2
    exit 2
fi

GPU_INDEX=$1
TQE_RANK=$2
REPO=${REPO:-/home/zhouchongtian/quantization/qvdit_flash_bf16}
CONDA_SH=${CONDA_SH:-/home/zhouchongtian/miniconda3/etc/profile.d/conda.sh}
DATE_TAG=${DATE_TAG:-0804}

[[ "$GPU_INDEX" =~ ^[0-9]+$ ]] || { echo "GPU_INDEX must be numeric" >&2; exit 2; }
[[ "$TQE_RANK" =~ ^[1-9][0-9]*$ ]] || { echo "TQE_RANK must be positive" >&2; exit 2; }

CALIB_ROOT="$REPO/logs_fp16_flash/formal_w4a6_mtd_tqerank${TQE_RANK}_samples10_gpu${GPU_INDEX}_${DATE_TAG}"
CALIB_DIR="$CALIB_ROOT/calibration"
VBENCH_ROOT="$REPO/logs_fp16_flash/vbench_mtd_tqerank${TQE_RANK}_fp32t5_fp16flash_ddim100_cfg4_seed42_gpu${GPU_INDEX}_${DATE_TAG}"
PIPELINE_LOG="$CALIB_ROOT/pipeline.log"
STATUS_FILE="$CALIB_ROOT/status.txt"
LOCK_FILE="$REPO/logs_fp16_flash/.tqe_rank${TQE_RANK}_gpu${GPU_INDEX}_${DATE_TAG}.lock"

MODEL_CFG="$REPO/t2v/configs/quant/opensora/16x512x512_fp16_flash_50steps.py"
MODEL_CKPT=/home/zhouchongtian/quantization/Q-VDiT/logs/split_ckpt/OpenSora-v1-HQ-16x512x512-split.pth
QUANT_CFG="$REPO/t2v/configs/quant/opensora/w4a6_mtd.yaml"
CALIB_DATA="$REPO/logs_fp16_flash/formal_w4a6_mtd_samples10_gpu5_0724/calib_data_ddim50_cfg4/calib_data.pt"
CALIB_EMBEDS=/home/zhouchongtian/quantization/new/logs_50steps/text_embeds_samples_10.pt
MP_WEIGHT="$REPO/t2v/configs/quant/opensora/mixed_precision/weight_4_mp.yaml"
MP_ACT="$REPO/t2v/configs/quant/opensora/mixed_precision/act_6_mp.yaml"

mkdir -p "$CALIB_DIR"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    echo "Another rank-$TQE_RANK pipeline already holds $LOCK_FILE" >&2
    exit 1
fi
exec > >(tee -a "$PIPELINE_LOG") 2>&1

stamp() {
    printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

fail() {
    local message=$1
    stamp "FAILED: $message"
    printf 'FAILED %s: %s\n' "$(date '+%F %T')" "$message" > "$STATUS_FILE"
    exit 1
}

trap 'fail "command failed at line $LINENO: $BASH_COMMAND"' ERR

for path in "$MODEL_CFG" "$MODEL_CKPT" "$QUANT_CFG" "$CALIB_DATA" "$CALIB_EMBEDS" "$MP_WEIGHT" "$MP_ACT"; do
    [[ -s "$path" ]] || fail "missing or empty file: $path"
done

export QVDIT_TQE_RANK="$TQE_RANK"
export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
export PYTHONPATH="$REPO:$REPO/t2v"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python - "$CALIB_ROOT/experiment_manifest.json" <<'PY'
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

repo = Path(os.environ["PYTHONPATH"].split(":", 1)[0])
rank = int(os.environ["QVDIT_TQE_RANK"])
gpu = int(os.environ["CUDA_VISIBLE_DEVICES"])
payload = {
    "experiment": "TQE output-compensation rank ablation",
    "tqe_rank": rank,
    "physical_gpu": gpu,
    "controlled_factors": {
        "quantization": "W4A6",
        "frame_axis": "MTD",
        "calibration_samples": 10,
        "calibration_steps": 50,
        "calibration_iterations": 10000,
        "inference": "FP16 FlashAttention, DDIM100, CFG4, seed42",
        "evaluation": "VBench official 8 metrics",
    },
    "quant_layer_sha256": hashlib.sha256((repo / "qdiff/models/quant_layer.py").read_bytes()).hexdigest(),
    "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
}
Path(sys.argv[1]).write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(json.dumps(payload, indent=2))
PY

if [[ ! -s "$CALIB_DIR/ckpt.pth" ]]; then
    if [[ -s "$CALIB_DIR/run.log" ]]; then
        fail "partial calibration exists without final checkpoint: $CALIB_DIR"
    fi
    printf 'CALIBRATING %s\n' "$(date '+%F %T')" > "$STATUS_FILE"
    stamp "Starting W4A6 MTD calibration with TQE rank=$TQE_RANK on physical GPU $GPU_INDEX"
    source "$CONDA_SH"
    conda activate qvdit
    cd "$REPO"
    python -u t2v/scripts/calib.py \
        "$MODEL_CFG" \
        --ckpt_path "$MODEL_CKPT" \
        --calib_config "$QUANT_CFG" \
        --calib_data "$CALIB_DATA" \
        --precompute_text_embeds "$CALIB_EMBEDS" \
        --outdir "$CALIB_DIR" \
        --part_fp \
        --time_mp_config_weight "$MP_WEIGHT" \
        --time_mp_config_act "$MP_ACT" \
        --use_grad_scaler \
        --numeric_monitor_interval 25 \
        --numeric_monitor_detailed_interval 100 \
        --reconstruction_checkpoint_interval 500 \
        2>&1 | tee "$CALIB_ROOT/calibration_console.log"
    [[ -s "$CALIB_DIR/ckpt.pth" ]] || fail "calibration did not produce ckpt.pth"
else
    stamp "Using existing completed calibration checkpoint: $CALIB_DIR/ckpt.pth"
fi

source "$CONDA_SH"
conda activate qvdit
python - "$CALIB_DIR/ckpt.pth" "$TQE_RANK" <<'PY'
import sys
import torch

payload = torch.load(sys.argv[1], map_location="cpu")
rank = int(sys.argv[2])
items = payload.items() if isinstance(payload, dict) else []
a_shapes = [tuple(value.shape) for key, value in items if key.endswith("loraA_out.weight")]
b_shapes = [tuple(value.shape) for key, value in items if key.endswith("loraB_out.weight")]
if not a_shapes or not b_shapes:
    raise SystemExit("checkpoint contains no TQE loraA_out/loraB_out tensors")
if any(shape[0] != rank for shape in a_shapes):
    raise SystemExit(f"loraA_out rank mismatch: expected {rank}, examples={a_shapes[:5]}")
if any(shape[1] != rank for shape in b_shapes):
    raise SystemExit(f"loraB_out rank mismatch: expected {rank}, examples={b_shapes[:5]}")
print({"rank": rank, "loraA_out_count": len(a_shapes), "loraB_out_count": len(b_shapes)})
PY

printf 'VBENCH %s\n' "$(date '+%F %T')" > "$STATUS_FILE"
stamp "Calibration complete; starting chained VBench inference and evaluation"
export GPU_INDEX
export WAIT_FOR_GPU_IDLE=0
export QUANT_CFG
export QUANT_CKPT="$CALIB_DIR/ckpt.pth"
export EMBED_ROOT="$REPO/logs_fp16_flash/vbench_official_embeds_fp32t5_seed42"
export RUN_ROOT="$VBENCH_ROOT"
export LOCK_FILE="$REPO/logs_fp16_flash/.vbench_mtd_tqerank${TQE_RANK}_gpu${GPU_INDEX}_${DATE_TAG}.lock"
export EXPERIMENT_NAME="FP16_FlashAttention_W4A6_MTD_TQE_rank${TQE_RANK}_FP32T5_GPU${GPU_INDEX}"
bash "$REPO/tools/run_baseline_fp32t5_vbench_gpu2.sh"

[[ -s "$VBENCH_ROOT/vbench_8metrics_summary.json" ]] \
    || fail "VBench workflow did not produce the 8-metric summary"
printf 'COMPLETED %s\n' "$(date '+%F %T')" > "$STATUS_FILE"
stamp "ALL_COMPLETE rank=$TQE_RANK gpu=$GPU_INDEX"
