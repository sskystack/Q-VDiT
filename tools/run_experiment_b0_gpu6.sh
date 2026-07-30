#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/zhouchongtian/quantization/qvdit_flash_bf16
PYTHON=/home/zhouchongtian/miniconda3/envs/qvdit/bin/python
CFG="$ROOT/t2v/configs/quant/opensora/16x512x512_fp16_flash_100steps.py"
MODEL_CKPT=/home/zhouchongtian/quantization/Q-VDiT/logs/split_ckpt/OpenSora-v1-HQ-16x512x512-split.pth
CALIB_CFG="$ROOT/t2v/configs/quant/opensora/w4a6_baseline.yaml"
QUANT_CKPT="$ROOT/logs_fp16_flash/formal_w4a6_baseline_fp16_gradscaler_samples10_gpu5_0726/calibration/ckpt.pth"
PROMPTS="$ROOT/t2v/assets/texts/t2v_samples_10.txt"
TEXT_EMBEDS=/home/zhouchongtian/quantization/new/logs_50steps/text_embeds_samples_10.pt
MP_WEIGHT="$ROOT/t2v/configs/quant/opensora/mixed_precision/weight_4_mp.yaml"
MP_ACT="$ROOT/t2v/configs/quant/opensora/mixed_precision/act_6_mp.yaml"
OUT_ROOT="$ROOT/logs_fp16_flash/stage_quant_profiling/b0_prompt0_seed42_p21_30"

mkdir -p "$OUT_ROOT"
cd "$ROOT"
export PYTHONPATH="$ROOT:$ROOT/t2v"
export CUDA_VISIBLE_DEVICES=6

while nvidia-smi --id=6 --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | grep -Eq '[0-9]'; do
  echo "$(date -Is) GPU6 busy; Experiment B0 is waiting."
  sleep 60
done

run_mode() {
  local mode="$1"
  shift
  local outdir="$OUT_ROOT/$mode/runtime"
  local savedir="$OUT_ROOT/$mode/video"
  mkdir -p "$outdir" "$savedir"
  echo "$(date -Is) starting mode=$mode"
  "$PYTHON" t2v/scripts/quant_txt2video.py "$CFG" \
    --ckpt_path "$MODEL_CKPT" \
    --calib_config "$CALIB_CFG" \
    --quant_ckpt "$QUANT_CKPT" \
    --outdir "$outdir" \
    --save_dir "$savedir" \
    --prompt_path "$PROMPTS" \
    --precompute_text_embeds "$TEXT_EMBEDS" \
    --num_videos 1 \
    --batch_size 1 \
    --num_sampling_steps 100 \
    --cfg_scale 4.0 \
    --sampler ddim \
    --seed 42 \
    --dataset_type opensora \
    --part_fp \
    --time_mp_config_weight "$MP_WEIGHT" \
    --time_mp_config_act "$MP_ACT" \
    --save_final_latent \
    --save_init_noise \
    "$@" \
    2>&1 | tee "$OUT_ROOT/$mode/console.log"
  echo "$(date -Is) finished mode=$mode"
}

# Full-precision control, constructed through the same QuantModel wrapper.
run_mode fp --skip_quant_weight --skip_quant_act

# Quantize only progress steps 21..30; all other steps remain full precision.
COMMON_WINDOW=(
  --timestep_wise_quant
  --quant_progress_start 21
  --quant_progress_end 30
  --save_quant_trace
)
run_mode w4 --skip_quant_act "${COMMON_WINDOW[@]}"
run_mode a6 --skip_quant_weight "${COMMON_WINDOW[@]}"
run_mode w4a6 "${COMMON_WINDOW[@]}"

"$PYTHON" "$ROOT/tools/analyze_experiment_b0.py" --root "$OUT_ROOT"
echo "$(date -Is) Experiment B0 complete."
