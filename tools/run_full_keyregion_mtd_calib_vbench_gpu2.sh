#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/home/zhouchongtian/quantization/qvdit_flash_bf16
CALIB_ROOT="$ROOT/logs_fp16_flash/formal_w4a6_mtd_keyregion_reliable25_full10000_gpu2_0804"
CALIB_DIR="$CALIB_ROOT/calibration"
RUN_ROOT="$ROOT/logs_fp16_flash/vbench_mtd_keyregion_reliable25_full10000"
CURRENT_SUMMARY="$ROOT/logs_fp16_flash/vbench_mtdfp16gs_final10000/vbench_8metrics_summary.json"
MODEL_CKPT=/home/zhouchongtian/quantization/Q-VDiT/logs/split_ckpt/OpenSora-v1-HQ-16x512x512-split.pth
CALIB_DATA="$ROOT/logs_fp16_flash/formal_w4a6_mtd_samples10_gpu5_0724/calib_data_ddim50_cfg4/calib_data.pt"
CALIB_EMBEDS=/home/zhouchongtian/quantization/new/logs_50steps/text_embeds_samples_10.pt
CALIB_MODEL_CFG="$ROOT/t2v/configs/quant/opensora/16x512x512_fp16_flash_50steps.py"
INFER_MODEL_CFG="$ROOT/t2v/configs/quant/opensora/16x512x512_fp16_flash_100steps.py"
QUANT_CFG="$ROOT/t2v/configs/quant/opensora/w4a6_mtd_keyregion_reliable25_full10000.yaml"
MP_WEIGHT="$ROOT/t2v/configs/quant/opensora/mixed_precision/weight_4_mp.yaml"
MP_ACT="$ROOT/t2v/configs/quant/opensora/mixed_precision/act_6_mp.yaml"
SUBJECT_PROMPTS="$ROOT/t2v/assets/texts/vbench_official/subject_consistency.txt"
SCENE_PROMPTS="$ROOT/t2v/assets/texts/vbench_official/scene.txt"
OVERALL_PROMPTS="$ROOT/t2v/assets/texts/vbench_official/overall_consistency.txt"
VBENCH_REPO=/home/zhouchongtian/quantization/eval/Vbench

source /home/zhouchongtian/miniconda3/etc/profile.d/conda.sh
conda activate qvdit
cd "$ROOT"
export PYTHONPATH="$ROOT:$ROOT/t2v"
export CUDA_VISIBLE_DEVICES=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$CALIB_ROOT" "$RUN_ROOT"
exec > >(tee -a "$CALIB_ROOT/driver.log") 2>&1
trap 'echo "FAILED $(date -Is)" > "$CALIB_ROOT/status.txt"' ERR

for required in "$MODEL_CKPT" "$CALIB_DATA" "$CALIB_EMBEDS" "$QUANT_CFG" \
  "$SUBJECT_PROMPTS" "$SCENE_PROMPTS" "$OVERALL_PROMPTS" "$CURRENT_SUMMARY"; do
  test -s "$required"
done
[[ "$(wc -l < "$SUBJECT_PROMPTS")" -eq 72 ]]
[[ "$(wc -l < "$SCENE_PROMPTS")" -eq 86 ]]
[[ "$(wc -l < "$OVERALL_PROMPTS")" -eq 93 ]]

if [[ ! -s "$CALIB_DIR/ckpt.pth" ]]; then
  if [[ -e "$CALIB_DIR/run.log" ]]; then
    echo "Incomplete calibration requires manual audit: $CALIB_DIR" >&2
    exit 1
  fi
  mkdir -p "$CALIB_DIR"
  echo "CALIBRATING $(date -Is)" > "$CALIB_ROOT/status.txt"
  python t2v/scripts/calib.py "$CALIB_MODEL_CFG" \
    --ckpt_path "$MODEL_CKPT" \
    --calib_config "$QUANT_CFG" \
    --calib_data "$CALIB_DATA" \
    --precompute_text_embeds "$CALIB_EMBEDS" \
    --outdir "$CALIB_DIR" \
    --part_fp \
    --time_mp_config_weight "$MP_WEIGHT" \
    --time_mp_config_act "$MP_ACT" \
    --use_grad_scaler \
    --numeric_monitor_interval 100 \
    --numeric_monitor_detailed_interval 500 \
    --reconstruction_checkpoint_interval 500 \
    2>&1 | tee "$CALIB_ROOT/console.log"
fi
test -s "$CALIB_DIR/ckpt.pth"

count_videos() {
  find "$1" -maxdepth 1 -type f -name '*.mp4' 2>/dev/null | wc -l
}

generate_subset() {
  local name=$1
  local prompts=$2
  local expected=$3
  local runtime="$RUN_ROOT/${name}_runtime"
  local save_base="$RUN_ROOT/$name"
  local videos="${save_base}_opensora"
  if [[ "$(count_videos "$videos")" -eq "$expected" ]]; then
    return
  fi
  rm -rf "$runtime" "$videos"
  mkdir -p "$runtime"
  echo "GENERATING_${name^^} $(date -Is)" > "$CALIB_ROOT/status.txt"
  python t2v/scripts/quant_txt2video.py "$INFER_MODEL_CFG" \
    --ckpt_path "$MODEL_CKPT" \
    --calib_config "$QUANT_CFG" \
    --quant_ckpt "$CALIB_DIR/ckpt.pth" \
    --outdir "$runtime" \
    --save_dir "$save_base" \
    --prompt_path "$prompts" \
    --prompt_as_path \
    --num_videos "$expected" \
    --batch_size 1 \
    --num_sampling_steps 100 \
    --cfg_scale 4.0 \
    --sampler ddim \
    --seed 42 \
    --dataset_type opensora \
    --part_fp \
    --time_mp_config_weight "$MP_WEIGHT" \
    --time_mp_config_act "$MP_ACT" \
    2>&1 | tee "$RUN_ROOT/generate_${name}.log"
  [[ "$(count_videos "$videos")" -eq "$expected" ]]
}

generate_subset subject "$SUBJECT_PROMPTS" 72
generate_subset scene "$SCENE_PROMPTS" 86
generate_subset overall "$OVERALL_PROMPTS" 93

conda activate vbench
cd "$VBENCH_REPO"
export VBENCH_CACHE_DIR=/home/zhouchongtian/quantization/models/vbench

evaluate_subset() {
  local name=$1
  shift
  local output="$RUN_ROOT/${name}_eval"
  rm -rf "$output"
  mkdir -p "$output"
  echo "EVALUATING_${name^^} $(date -Is)" > "$CALIB_ROOT/status.txt"
  python evaluate.py \
    --videos_path "$RUN_ROOT/${name}_opensora" \
    --output_path "$output" \
    --dimension "$@" \
    --load_ckpt_from_local True \
    2>&1 | tee "$RUN_ROOT/evaluate_${name}.log"
}

evaluate_subset subject subject_consistency dynamic_degree motion_smoothness
evaluate_subset scene scene background_consistency
evaluate_subset overall overall_consistency aesthetic_quality imaging_quality

export RUN_ROOT CURRENT_SUMMARY
python - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["RUN_ROOT"])
sources = [
    root / "subject_eval" / "subject_consistency_eval_results.json",
    root / "scene_eval" / "scene_eval_results.json",
    root / "overall_eval" / "overall_consistency_eval_results.json",
]
order = [
    "subject_consistency", "dynamic_degree", "motion_smoothness", "scene",
    "background_consistency", "overall_consistency", "aesthetic_quality",
    "imaging_quality",
]
scores = {}
for path in sources:
    result = json.loads(path.read_text())
    for metric, value in result.items():
        scores[metric] = float(value[0] if isinstance(value, list) else value)
candidate = {metric: scores[metric] for metric in order}
current = json.loads(Path(os.environ["CURRENT_SUMMARY"]).read_text())["raw_scores"]
summary = {
    "experiment": "MTD_WEIGHT1_RELIABLE_KEYREGION_ERROR_X_CONFIDENCE25_FULL10000",
    "candidate_raw_scores": candidate,
    "current_mtd_raw_scores": {metric: current[metric] for metric in order},
    "candidate_minus_current": {
        metric: candidate[metric] - current[metric] for metric in order
    },
}
(root / "comparison_vs_current_mtd.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
PY

echo "COMPLETED $(date -Is)" > "$CALIB_ROOT/status.txt"
