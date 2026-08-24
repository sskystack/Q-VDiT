#!/usr/bin/env bash
set -Eeuo pipefail

source /home/zhouchongtian/miniconda3/etc/profile.d/conda.sh
conda activate qvdit

CODE_ROOT=${CODE_ROOT:-/home/zhouchongtian/quantization/qvdit_adaptive_mtd_calibration_v1}
DATA_ROOT=${DATA_ROOT:-/home/zhouchongtian/quantization/qvdit_flash_bf16}
GPU_INDEX=${GPU_INDEX:-2}
OUT_ROOT=${OUT_ROOT:-$DATA_ROOT/logs_fp16_flash/adaptive_mtd_calibration_v1_200}
CALIB_DATA="$DATA_ROOT/logs_fp16_flash/formal_w4a6_mtd_samples10_gpu5_0724/calib_data_ddim50_cfg4/calib_data.pt"
TEXT_EMBEDS=/home/zhouchongtian/quantization/new/logs_50steps/text_embeds_samples_10.pt
MODEL_CKPT=/home/zhouchongtian/quantization/Q-VDiT/logs/split_ckpt/OpenSora-v1-HQ-16x512x512-split.pth

arms=(fixed weighted random centered)
configs=(
  w4a6_mtd_adaptive_v1_fixed_200.yaml
  w4a6_mtd_adaptive_v1_weighted_200.yaml
  w4a6_mtd_adaptive_v1_random_200.yaml
  w4a6_mtd_adaptive_v1_centered_200.yaml
)

mkdir -p "$OUT_ROOT"
[[ -s "$CALIB_DATA" ]]
[[ -s "$TEXT_EMBEDS" ]]
[[ -s "$MODEL_CKPT" ]]

current_arm="startup"
current_arm_root=""
on_error() {
  code=$?
  message="FAILED arm=$current_arm gpu=$GPU_INDEX exit=$code $(date '+%F %T')"
  echo "$message" | tee "$OUT_ROOT/status.txt"
  if [[ -n "$current_arm_root" ]]; then
    echo "$message" > "$current_arm_root/status.txt"
  fi
  exit "$code"
}
trap on_error ERR

gpu_uuid=$(nvidia-smi --id="$GPU_INDEX" --query-gpu=uuid --format=csv,noheader)
idle_checks=0
while (( idle_checks < 3 )); do
  compute_uuids=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null || true)
  if grep -Fxq "$gpu_uuid" <<<"$compute_uuids"; then
    idle_checks=0
  else
    ((idle_checks += 1))
  fi
  sleep 10
done

cd "$CODE_ROOT"
export PYTHONPATH="$CODE_ROOT:$CODE_ROOT/t2v"
export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export QVDIT_MEMORY_PROFILE=1
export QVDIT_MEMORY_HISTORY=0

echo "RUNNING gpu=$GPU_INDEX $(date '+%F %T')" | tee "$OUT_ROOT/status.txt"
for index in "${!arms[@]}"; do
  arm=${arms[$index]}
  current_arm="$arm"
  config_name=${configs[$index]}
  config="$CODE_ROOT/t2v/configs/quant/opensora/$config_name"
  arm_root="$OUT_ROOT/$arm"
  current_arm_root="$arm_root"
  calib="$arm_root/calibration"
  mkdir -p "$arm_root"

  if [[ -s "$arm_root/status.txt" ]] && grep -q '^COMPLETED ' "$arm_root/status.txt"; then
    echo "SKIP_COMPLETED arm=$arm"
    continue
  fi
  if [[ -e "$calib/run.log" ]]; then
    echo "Refusing to overwrite incomplete calibration: $calib" >&2
    exit 1
  fi

  echo "RUNNING arm=$arm gpu=$GPU_INDEX $(date '+%F %T')" | tee "$arm_root/status.txt"
  mkdir -p "$calib"
  /usr/bin/time -v python t2v/scripts/calib.py \
    "$CODE_ROOT/t2v/configs/quant/opensora/16x512x512_fp16_flash_50steps.py" \
    --ckpt_path "$MODEL_CKPT" \
    --calib_config "$config" \
    --calib_data "$CALIB_DATA" \
    --precompute_text_embeds "$TEXT_EMBEDS" \
    --outdir "$calib" \
    --part_fp \
    --time_mp_config_weight "$CODE_ROOT/t2v/configs/quant/opensora/mixed_precision/weight_4_mp.yaml" \
    --time_mp_config_act "$CODE_ROOT/t2v/configs/quant/opensora/mixed_precision/act_6_mp.yaml" \
    --use_grad_scaler \
    --numeric_monitor_interval 1 \
    --numeric_monitor_detailed_interval 100 \
    --reconstruction_checkpoint_interval 200 \
    2>&1 | tee "$arm_root/console.log"

  python tools/summarize_mtd_subterm_gradients.py \
    "$calib/loss_component_gradients.jsonl" \
    --output-prefix "$calib/mtd_subterm_gradient_summary" \
    2>&1 | tee "$arm_root/gradient_summary.log"
  echo "COMPLETED arm=$arm gpu=$GPU_INDEX $(date '+%F %T')" | tee "$arm_root/status.txt"
done

echo "COMPLETED gpu=$GPU_INDEX $(date '+%F %T')" | tee "$OUT_ROOT/status.txt"
