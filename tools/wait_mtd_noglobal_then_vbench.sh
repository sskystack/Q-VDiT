#!/usr/bin/env bash
set -Eeuo pipefail

GPU_ID=${1:?GPU index required}
CONFIG_NAME=${2:?config filename required}
EXPERIMENT_NAME=${3:?experiment name required}

REPO=/home/zhouchongtian/quantization/qvdit_flash_bf16
EXP="$REPO/logs_fp16_flash/$EXPERIMENT_NAME"
CALIB="$EXP/calibration"
RUN="$EXP/vbench"
CONFIG="$REPO/t2v/configs/quant/opensora/$CONFIG_NAME"
MODEL_CFG="$REPO/t2v/configs/quant/opensora/16x512x512_fp16_flash_100steps.py"
MODEL_CKPT=/home/zhouchongtian/quantization/Q-VDiT/logs/split_ckpt/OpenSora-v1-HQ-16x512x512-split.pth
MPW="$REPO/t2v/configs/quant/opensora/mixed_precision/weight_4_mp.yaml"
MPA="$REPO/t2v/configs/quant/opensora/mixed_precision/act_6_mp.yaml"
VBENCH=/home/zhouchongtian/quantization/eval/Vbench

mkdir -p "$RUN"
exec > >(tee -a "$RUN/automation.log") 2>&1
trap 'echo "FAILED $(date "+%F %T")" | tee "$RUN/status.txt"' ERR

echo "WAITING_FOR_CALIBRATION $(date '+%F %T')" | tee "$RUN/status.txt"
while true; do
    if grep -q '^CALIBRATION_COMPLETED ' "$EXP/status.txt" 2>/dev/null; then
        break
    fi
    if grep -q '^FAILED ' "$EXP/status.txt" 2>/dev/null; then
        echo "Calibration failed or was manually stopped; VBench will not start" >&2
        exit 1
    fi
    sleep 60
done

[[ -s "$CALIB/ckpt.pth" ]]
grep -q 'count=10000' "$EXP/console.log"

source /home/zhouchongtian/miniconda3/etc/profile.d/conda.sh
conda activate qvdit
cd "$REPO"
export PYTHONPATH="$REPO:$REPO/t2v"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

generate_group() {
    local name=$1 prompts=$2 embeds=$3 expected=$4
    local videos="$RUN/${name}_opensora"
    if [[ "$(find "$videos" -maxdepth 1 -type f -name '*.mp4' 2>/dev/null | wc -l)" -eq "$expected" ]]; then
        return
    fi
    rm -rf "$RUN/${name}_runtime" "$videos"
    mkdir -p "$RUN/${name}_runtime"
    echo "GENERATING_${name^^} $(date '+%F %T')" | tee "$RUN/status.txt"
    python t2v/scripts/quant_txt2video.py "$MODEL_CFG" \
      --ckpt_path "$MODEL_CKPT" \
      --calib_config "$CONFIG" \
      --quant_ckpt "$CALIB/ckpt.pth" \
      --outdir "$RUN/${name}_runtime" \
      --save_dir "$RUN/$name" \
      --prompt_path "$prompts" \
      --precompute_text_embeds "$embeds" \
      --prompt_as_path \
      --num_videos "$expected" \
      --batch_size 1 \
      --num_sampling_steps 100 \
      --cfg_scale 4.0 \
      --sampler ddim \
      --seed 42 \
      --dataset_type opensora \
      --part_fp \
      --time_mp_config_weight "$MPW" \
      --time_mp_config_act "$MPA"
    [[ "$(find "$videos" -maxdepth 1 -type f -name '*.mp4' | wc -l)" -eq "$expected" ]]
}

generate_group subject \
  "$REPO/t2v/assets/texts/vbench_official/subject_consistency.txt" \
  "$REPO/logs_bf16_flash/vbench_mtd_iter5000/subject_consistency_embeds.pth" 72
generate_group scene \
  "$REPO/t2v/assets/texts/vbench_official/scene.txt" \
  /home/zhouchongtian/quantization/Q-VDiT/logs_50steps/early_stop_cfg01_proxy/vbench_scene_embeds.pth 86
generate_group overall \
  "$REPO/t2v/assets/texts/vbench_official/overall_consistency.txt" \
  /home/zhouchongtian/quantization/Q-VDiT/logs_50steps/early_stop_cfg01_proxy/vbench_overall_embeds.pth 93

conda activate vbench
cd "$VBENCH"
export VBENCH_CACHE_DIR=/home/zhouchongtian/quantization/models/vbench

evaluate_group() {
    local name=$1 result_file=$2
    shift 2
    rm -rf "$RUN/${name}_eval"
    mkdir -p "$RUN/${name}_eval"
    echo "EVALUATING_${name^^} $(date '+%F %T')" | tee "$RUN/status.txt"
    python evaluate.py \
      --videos_path "$RUN/${name}_opensora" \
      --output_path "$RUN/${name}_eval" \
      --dimension "$@" \
      --load_ckpt_from_local True
    [[ -s "$RUN/${name}_eval/$result_file" ]]
}

evaluate_group subject subject_consistency_eval_results.json \
  subject_consistency dynamic_degree motion_smoothness
evaluate_group scene scene_eval_results.json scene background_consistency
evaluate_group overall overall_consistency_eval_results.json \
  overall_consistency aesthetic_quality imaging_quality

python - "$RUN" "$EXPERIMENT_NAME" <<'PY'
import csv, json, sys
from pathlib import Path
root, name = Path(sys.argv[1]), sys.argv[2]
sources = [
    root / "subject_eval/subject_consistency_eval_results.json",
    root / "scene_eval/scene_eval_results.json",
    root / "overall_eval/overall_consistency_eval_results.json",
]
order = ["imaging_quality", "aesthetic_quality", "motion_smoothness",
         "dynamic_degree", "background_consistency", "subject_consistency",
         "scene", "overall_consistency"]
scores = {}
for path in sources:
    for key, value in json.loads(path.read_text()).items():
        scores[key] = float(value[0] if isinstance(value, list) else value)
missing = [x for x in order if x not in scores]
if missing: raise RuntimeError(f"Missing metrics: {missing}")
data = {"experiment": name, "raw_scores": {k:scores[k] for k in order},
        "percentage_scores": {k:scores[k]*100 for k in order}}
(root / "vbench_8metrics_summary.json").write_text(json.dumps(data, indent=2))
with (root / "vbench_8metrics_summary.csv").open("w", newline="") as f:
    w=csv.writer(f); w.writerow(["metric","raw_score","percentage_score"])
    for k in order: w.writerow([k,f"{scores[k]:.8f}",f"{scores[k]*100:.4f}"])
lines=[name,"",f"{'Metric':<28}{'Raw':>12}{'x100':>12}","-"*52]
lines += [f"{k:<28}{scores[k]:>12.6f}{scores[k]*100:>12.2f}" for k in order]
(root / "vbench_8metrics_summary.txt").write_text("\n".join(lines)+"\n")
print("\n".join(lines))
PY

echo "COMPLETED $(date '+%F %T')" | tee "$RUN/status.txt"
