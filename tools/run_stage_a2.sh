#!/usr/bin/env bash
set -uo pipefail

source /home/zhouchongtian/miniconda3/etc/profile.d/conda.sh
conda activate qvdit
cd /home/zhouchongtian/quantization/qvdit_flash_bf16
export PYTHONPATH="$PWD:$PWD/t2v"
export CUDA_VISIBLE_DEVICES=6

ROOT=/home/zhouchongtian/quantization/qvdit_flash_bf16/logs_fp16_flash/stage_profiling
CONFIG=t2v/configs/quant/opensora/16x512x512_fp16_flash_100steps.py
PROMPTS=t2v/assets/texts/t2v_samples_10.txt
EMBEDS=/home/zhouchongtian/quantization/new/logs_50steps/text_embeds_samples_10.pt
MASTER_LOG="$ROOT/a2_batch_10prompts_2seeds.log"

mkdir -p "$ROOT"
echo "A2 batch started: $(date --iso-8601=seconds)" | tee -a "$MASTER_LOG"

for seed in 42 123; do
  for prompt_index in $(seq 0 9); do
    out="$ROOT/fp16_fp_trajectory_prompt${prompt_index}_seed${seed}"
    if [[ -f "$out/config.json" ]] && grep -q '"status": "completed"' "$out/config.json"; then
      echo "SKIP completed prompt=$prompt_index seed=$seed: $(date --iso-8601=seconds)" | tee -a "$MASTER_LOG"
      continue
    fi

    mkdir -p "$out"
    echo "START prompt=$prompt_index seed=$seed: $(date --iso-8601=seconds)" | tee -a "$MASTER_LOG"
    python tools/profile_diffusion_stages.py \
      --config "$CONFIG" \
      --prompt-path "$PROMPTS" \
      --prompt-index "$prompt_index" \
      --text-embeds "$EMBEDS" \
      --output-dir "$out" \
      --seed "$seed" \
      >"$out/run.log" 2>&1
    status=$?
    echo "END prompt=$prompt_index seed=$seed status=$status: $(date --iso-8601=seconds)" | tee -a "$MASTER_LOG"
  done
done

echo "A2 batch finished: $(date --iso-8601=seconds)" | tee -a "$MASTER_LOG"
