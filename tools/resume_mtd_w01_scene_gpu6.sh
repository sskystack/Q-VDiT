#!/usr/bin/env bash
set -Eeuo pipefail

REPO=/home/zhouchongtian/quantization/qvdit_flash_bf16
ROOT="$REPO/logs_fp16_flash/vbench_mtd_w01_fp16gs_final10000"
CALIB="$REPO/logs_fp16_flash/formal_w4a6_mtd_w01_gradprobe_samples10_gpu2_0728/calibration"
CFG="$REPO/t2v/configs/quant/opensora/16x512x512_fp16_flash_100steps.py"
QCFG="$REPO/t2v/configs/quant/opensora/w4a6_mtd_w01_probe.yaml"
MODEL=/home/zhouchongtian/quantization/Q-VDiT/logs/split_ckpt/OpenSora-v1-HQ-16x512x512-split.pth
PROMPTS="$REPO/t2v/assets/texts/vbench_official/scene.txt"
EMBEDS=/home/zhouchongtian/quantization/Q-VDiT/logs_50steps/early_stop_cfg01_proxy/vbench_scene_embeds.pth
MPW="$REPO/t2v/configs/quant/opensora/mixed_precision/weight_4_mp.yaml"
MPA="$REPO/t2v/configs/quant/opensora/mixed_precision/act_6_mp.yaml"
LOG="$ROOT/scene_gpu6_second_resume.log"

mkdir -p "$ROOT/scene_resume2_runtime"
exec > >(tee -a "$LOG") 2>&1
trap 'echo "[$(date "+%F %T")] FAILED line=$LINENO command=$BASH_COMMAND"' ERR

source /home/zhouchongtian/miniconda3/etc/profile.d/conda.sh
conda activate qvdit
cd "$REPO"
export PYTHONPATH="$REPO:$REPO/t2v"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=6

COUNT=$(find "$ROOT/scene_opensora" -maxdepth 1 -type f -name '*.mp4' | wc -l)
if [[ "$COUNT" -lt 86 ]]; then
    echo "[$(date '+%F %T')] Resuming Scene at prompt index 70"
    python t2v/scripts/quant_txt2video.py "$CFG" \
        --ckpt_path "$MODEL" \
        --calib_config "$QCFG" \
        --quant_ckpt "$CALIB/ckpt.pth" \
        --outdir "$ROOT/scene_resume2_runtime" \
        --save_dir "$ROOT/scene" \
        --prompt_path "$PROMPTS" \
        --precompute_text_embeds "$EMBEDS" \
        --prompt_start_index 70 \
        --prompt_as_path \
        --num_videos 16 \
        --batch_size 1 \
        --num_sampling_steps 100 \
        --cfg_scale 4.0 \
        --sampler ddim \
        --seed 112 \
        --dataset_type opensora \
        --part_fp \
        --time_mp_config_weight "$MPW" \
        --time_mp_config_act "$MPA"
fi
[[ "$(find "$ROOT/scene_opensora" -maxdepth 1 -type f -name '*.mp4' | wc -l)" -eq 86 ]]

conda activate vbench
cd /home/zhouchongtian/quantization/eval/Vbench
export VBENCH_CACHE_DIR=/home/zhouchongtian/quantization/models/vbench
rm -rf "$ROOT/scene_eval"
mkdir -p "$ROOT/scene_eval"
python evaluate.py \
    --videos_path "$ROOT/scene_opensora" \
    --output_path "$ROOT/scene_eval" \
    --dimension scene background_consistency \
    --load_ckpt_from_local True

python - "$ROOT" <<'PY'
import csv, json, sys
from pathlib import Path

root = Path(sys.argv[1])
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
    data = json.loads(path.read_text(encoding="utf-8"))
    for key, value in data.items():
        scores[key] = float(value[0] if isinstance(value, list) else value)
missing = [x for x in order if x not in scores]
if missing:
    raise RuntimeError(f"Missing metrics: {missing}")
summary = {
    "experiment": "FP16_FLASH_MTD_W01_W4A6_DDIM100_CFG4_SEED42",
    "raw_scores": {k: scores[k] for k in order},
    "percentage_scores": {k: scores[k] * 100 for k in order},
}
(root / "vbench_8metrics_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
with (root / "vbench_8metrics_summary.csv").open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["metric", "raw_score", "percentage_score"])
    for k in order: w.writerow([k, f"{scores[k]:.8f}", f"{scores[k]*100:.4f}"])
lines = ["FP16 FlashAttention MTD-0.1 W4A6 - DDIM 100 - CFG 4.0 - seed 42", "",
         f"{'Metric':<28}{'Raw':>12}{'x100':>12}", "-"*52]
lines += [f"{k:<28}{scores[k]:>12.6f}{scores[k]*100:>12.2f}" for k in order]
(root / "vbench_8metrics_summary.txt").write_text("\n".join(lines)+"\n", encoding="utf-8")
print("\n".join(lines))
PY

echo "[$(date '+%F %T')] ALL_COMPLETE" | tee "$ROOT/status.txt"
