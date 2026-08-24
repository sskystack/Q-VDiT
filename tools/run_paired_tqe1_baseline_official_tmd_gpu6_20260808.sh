#!/usr/bin/env bash
set -Eeuo pipefail

# Paired Q-VDiT experiment: TQE-only baseline versus official TMD.
# TMD here is the Q-VDiT time-relation reconstruction term, not MTD.

if [[ -z "${TMUX:-}" ]]; then
  echo "Run this paired experiment from a dedicated tmux session." >&2
  exit 2
fi

REPO=/home/zhouchongtian/quantization/qvdit_flash_bf16
GPU_INDEX=6
TQE_RANK=1
DATE_TAG=20260808
PAIR_ROOT="$REPO/logs/formal_tqe1_baseline_vs_official_tmd_gpu6_${DATE_TAG}"
CALIB_DATA="$REPO/logs_fp16_flash/formal_w4a6_mtd_samples10_gpu5_0724/calib_data_ddim50_cfg4/calib_data.pt"
MODEL_CFG="$REPO/t2v/configs/quant/opensora/16x512x512_fp16_flash_50steps.py"
MODEL_CKPT=/home/zhouchongtian/quantization/Q-VDiT/logs/split_ckpt/OpenSora-v1-HQ-16x512x512-split.pth
MP_WEIGHT="$REPO/t2v/configs/quant/opensora/mixed_precision/weight_4_mp.yaml"
MP_ACT="$REPO/t2v/configs/quant/opensora/mixed_precision/act_6_mp.yaml"
CONDA_SH=/home/zhouchongtian/miniconda3/etc/profile.d/conda.sh
LOCK_FILE="$REPO/logs/.tqe1_baseline_official_tmd_gpu6_${DATE_TAG}.lock"

[[ ! -e "$PAIR_ROOT" ]] || { echo "Refusing to reuse pair output root: $PAIR_ROOT" >&2; exit 1; }
for f in "$CALIB_DATA" "$MODEL_CFG" "$MODEL_CKPT" "$MP_WEIGHT" "$MP_ACT"; do
  [[ -s "$f" ]] || { echo "Missing required file: $f" >&2; exit 1; }
done
mkdir -p "$PAIR_ROOT"
exec 9>"$LOCK_FILE"
flock -n 9 || { echo "Another paired TQE/TMD workflow holds $LOCK_FILE" >&2; exit 1; }
exec > >(tee -a "$PAIR_ROOT/pipeline.log") 2>&1

stamp() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }
fail() { stamp "FAILED: $*"; printf 'FAILED %s: %s\n' "$(date '+%F %T')" "$*" > "$PAIR_ROOT/status.txt"; exit 1; }
trap 'fail "line=$LINENO command=$BASH_COMMAND"' ERR

export CUDA_VISIBLE_DEVICES=$GPU_INDEX
export PYTHONPATH="$REPO:$REPO/t2v"
export QVDIT_TQE_RANK=$TQE_RANK
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CALIB_DATA

python - "$PAIR_ROOT/experiment_manifest.json" <<'PY'
import json, os, subprocess, sys
from pathlib import Path
repo = Path(os.environ['PYTHONPATH'].split(':', 1)[0])
payload = {
  'experiment': 'paired Q-VDiT TQE rank-1 baseline vs official TMD',
  'physical_gpu': int(os.environ['CUDA_VISIBLE_DEVICES']),
  'tqe_rank': int(os.environ['QVDIT_TQE_RANK']),
  'shared_calibration_data': os.environ['CALIB_DATA'],
  'controlled_factors': {
    'quantization': 'W4A6', 'calibration_samples': 10,
    'calibration_sampling': 'FP16 FlashAttention, DDIM50, CFG4, seed42',
    'calibration_iterations': 10000,
    'inference': 'FP16 FlashAttention, DDIM100, CFG4, seed42, FP32-T5 embeddings',
    'evaluation': 'three-group smoke then official VBench 8 metrics',
  },
  'baseline': {'config': 'w4a6_baseline.yaml', 'frame_axis': 'BASELINE'},
  'official_tmd': {'config': 'w4a6_official_tmd_probe.yaml', 'frame_axis': 'BASELINE', 'objective': 'MSE + 100*time-relation loss'},
  'git_head': subprocess.check_output(['git','rev-parse','HEAD'], cwd=repo, text=True).strip(),
}
Path(sys.argv[1]).write_text(json.dumps(payload, indent=2), encoding='utf-8')
print(json.dumps(payload, indent=2))
PY

run_one() {
  local name=$1 config=$2 calib_root=$3 vbench_root=$4
  local calib_dir="$calib_root/calibration"
  [[ -s "$config" ]] || fail "missing config: $config"
  mkdir -p "$calib_dir"
  stamp "${name}: calibration start (TQE rank=$TQE_RANK, GPU=$GPU_INDEX)"
  printf 'CALIBRATING_%s %s\n' "${name^^}" "$(date '+%F %T')" > "$PAIR_ROOT/status.txt"
  source "$CONDA_SH"
  conda activate qvdit
  cd "$REPO"
  python -u t2v/scripts/calib.py "$MODEL_CFG" \
    --ckpt_path "$MODEL_CKPT" --calib_config "$config" --calib_data "$CALIB_DATA" \
    --outdir "$calib_dir" --seed 42 --batch_size 1 --gpu 0 \
    --part_fp --time_mp_config_weight "$MP_WEIGHT" --time_mp_config_act "$MP_ACT" \
    --use_grad_scaler --numeric_monitor_interval 25 --numeric_monitor_detailed_interval 100 \
    --reconstruction_checkpoint_interval 500 2>&1 | tee "$calib_root/calibration_console.log"
  [[ -s "$calib_dir/ckpt.pth" ]] || fail "$name calibration did not create ckpt.pth"
  stamp "${name}: calibration complete; starting formal smoke + VBench"
  printf 'VBENCH_%s %s\n' "${name^^}" "$(date '+%F %T')" > "$PAIR_ROOT/status.txt"
  REPO="$REPO" GPU_INDEX="$GPU_INDEX" WAIT_FOR_GPU_IDLE=0 QUANT_CFG="$config" \
    QUANT_CKPT="$calib_dir/ckpt.pth" \
    EMBED_ROOT="$REPO/logs_fp16_flash/vbench_official_embeds_fp32t5_seed42" \
    RUN_ROOT="$vbench_root" LOCK_FILE="$REPO/logs/.${name}_tqe1_gpu6_${DATE_TAG}.lock" \
    EXPERIMENT_NAME="FP16_FlashAttention_W4A6_${name}_TQE_rank1_FP32T5_GPU6" \
    bash "$REPO/tools/run_baseline_fp32t5_vbench_gpu2.sh"
  [[ -s "$vbench_root/vbench_8metrics_summary.json" ]] || fail "$name VBench summary missing"
  stamp "${name}: complete"
}

run_one baseline_tqe "$REPO/t2v/configs/quant/opensora/w4a6_baseline.yaml" \
  "$REPO/logs/formal_tqe1_baseline_calib_gpu6_${DATE_TAG}" \
  "$REPO/logs/formal_vbench_tqe1_baseline_cfg4_ddim100_seed42_gpu6_${DATE_TAG}"
run_one official_tmd_tqe "$REPO/t2v/configs/quant/opensora/w4a6_official_tmd_probe.yaml" \
  "$REPO/logs/formal_tqe1_official_tmd_calib_gpu6_${DATE_TAG}" \
  "$REPO/logs/formal_vbench_tqe1_official_tmd_cfg4_ddim100_seed42_gpu6_${DATE_TAG}"

printf 'COMPLETE %s\n' "$(date '+%F %T')" > "$PAIR_ROOT/status.txt"
stamp 'COMPLETE: paired baseline TQE vs official TMD TQE'
