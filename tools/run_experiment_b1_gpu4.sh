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
OUT_ROOT="$ROOT/logs_fp16_flash/stage_quant_profiling/b1_prompt0_seed42_10windows"
B0_FP="$ROOT/logs_fp16_flash/stage_quant_profiling/b0_prompt0_seed42_p21_30/fp"

mkdir -p "$OUT_ROOT"
cd "$ROOT"
export PYTHONPATH="$ROOT:$ROOT/t2v"
export CUDA_VISIBLE_DEVICES=4

while nvidia-smi --id=4 --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | grep -Eq '[0-9]'; do
  echo "$(date -Is) GPU4 busy; Experiment B1 is waiting."
  sleep 60
done

# Reuse the already validated full-precision control from B0.
mkdir -p "$OUT_ROOT/fp"
ln -sfn "$B0_FP/runtime" "$OUT_ROOT/fp/runtime"
ln -sfn "$B0_FP/video_opensora" "$OUT_ROOT/fp/video_opensora"

run_window() {
  local mode="$1"
  local start="$2"
  local end="$3"
  local tag
  tag=$(printf "p%03d_%03d" "$start" "$end")
  local base="$OUT_ROOT/$mode/$tag"
  local outdir="$base/runtime"
  local savedir="$base/video"
  mkdir -p "$outdir" "$savedir"
  if [[ -f "$outdir/final_latents/final_latent_0000.pt" ]]; then
    echo "$(date -Is) skipping completed mode=$mode window=$start-$end"
    return
  fi

  local skip_args=()
  if [[ "$mode" == "w4" ]]; then
    skip_args=(--skip_quant_act)
  elif [[ "$mode" == "a6" ]]; then
    skip_args=(--skip_quant_weight)
  fi

  echo "$(date -Is) starting mode=$mode window=$start-$end"
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
    --save_quant_trace \
    --timestep_wise_quant \
    --quant_progress_start "$start" \
    --quant_progress_end "$end" \
    "${skip_args[@]}" \
    2>&1 | tee "$base/console.log"
  echo "$(date -Is) finished mode=$mode window=$start-$end"
}

for start in 1 11 21 31 41 51 61 71 81 91; do
  end=$((start + 9))
  for mode in w4 a6 w4a6; do
    run_window "$mode" "$start" "$end"
  done
done

"$PYTHON" "$ROOT/tools/analyze_experiment_b1.py" --root "$OUT_ROOT"
echo "$(date -Is) Experiment B1 complete."
