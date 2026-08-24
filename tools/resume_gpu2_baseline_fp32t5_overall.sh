#!/usr/bin/env bash
set -Eeuo pipefail

REPO=/home/zhouchongtian/quantization/qvdit_flash_bf16
CONDA_SH=/home/zhouchongtian/miniconda3/etc/profile.d/conda.sh
GPU_INDEX=2
RUN_ROOT="$REPO/logs_fp16_flash/vbench_baseline_fp32t5_fp16flash_ddim100_cfg4_seed42_0802"
STATUS_FILE="$RUN_ROOT/status.txt"
LOG_FILE="$RUN_ROOT/resume_overall_gpu2.log"
LOCK_FILE="$REPO/logs_fp16_flash/.resume_baseline_fp32t5_overall_gpu2.lock"

MODEL_CFG="$REPO/t2v/configs/quant/opensora/16x512x512_fp16_flash_100steps.py"
QUANT_CFG="$REPO/t2v/configs/quant/opensora/w4a6_baseline.yaml"
MODEL_CKPT=/home/zhouchongtian/quantization/Q-VDiT/logs/split_ckpt/OpenSora-v1-HQ-16x512x512-split.pth
QUANT_CKPT="$REPO/logs_fp16_flash/formal_w4a6_baseline_fp16_gradscaler_samples10_gpu5_0726/calibration/ckpt.pth"
MP_WEIGHT="$REPO/t2v/configs/quant/opensora/mixed_precision/weight_4_mp.yaml"
MP_ACT="$REPO/t2v/configs/quant/opensora/mixed_precision/act_6_mp.yaml"
PROMPTS="$REPO/t2v/assets/texts/vbench_official/overall_consistency.txt"
EMBEDS="$REPO/logs_fp16_flash/vbench_official_embeds_fp32t5_seed42/overall_fp32t5.pth"

VBENCH_REPO=/home/zhouchongtian/quantization/eval/Vbench
VBENCH_CACHE=/home/zhouchongtian/quantization/models/vbench
VBENCH_BERT_DIR="$VBENCH_CACHE/bert-base-uncased"

mkdir -p "$RUN_ROOT"
exec 9>"$LOCK_FILE"
flock -n 9 || { echo "GPU2 Overall resume workflow already running" >&2; exit 1; }
exec > >(tee -a "$LOG_FILE") 2>&1

stamp() {
    printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

fail() {
    stamp "FAILED: $*"
    printf 'FAILED %s: %s\n' "$(date '+%F %T')" "$*" > "$STATUS_FILE"
    exit 1
}

trap 'fail "command failed at line $LINENO: $BASH_COMMAND"' ERR

for path in "$MODEL_CFG" "$QUANT_CFG" "$MODEL_CKPT" "$QUANT_CKPT" \
    "$MP_WEIGHT" "$MP_ACT" "$PROMPTS" "$EMBEDS"; do
    [[ -s "$path" ]] || fail "missing or empty file: $path"
done

python - "$PROMPTS" "$RUN_ROOT/overall_opensora" <<'PY'
import sys
from pathlib import Path

prompt_path = Path(sys.argv[1])
video_dir = Path(sys.argv[2])
prompts = [line.strip() for line in prompt_path.read_text(encoding="utf-8").splitlines() if line.strip()]
if len(prompts) != 93:
    raise SystemExit(f"expected 93 prompts, found {len(prompts)}")
actual = {path.name for path in video_dir.glob("*.mp4")}
expected = {prompt + ".mp4" for prompt in prompts[:38]}
if actual != expected:
    raise SystemExit(
        f"existing videos are not exactly prompt indices 0..37: "
        f"missing={sorted(expected-actual)[:5]}, extra={sorted(actual-expected)[:5]}"
    )
print("Validated exact completed prefix: Overall prompt indices 0..37")
PY

mapfile -t PROMPT_INDICES < <(seq 38 92)

source "$CONDA_SH"
conda activate qvdit
cd "$REPO"
export PYTHONPATH="$REPO:$REPO/t2v"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

printf 'RESUME_OVERALL_38_TO_92 %s\n' "$(date '+%F %T')" > "$STATUS_FILE"
stamp "Resuming Overall prompt indices 38..92 on GPU2 with full RNG replay"
CUDA_VISIBLE_DEVICES="$GPU_INDEX" python -u t2v/scripts/quant_txt2video.py "$MODEL_CFG" \
    --ckpt_path "$MODEL_CKPT" \
    --calib_config "$QUANT_CFG" \
    --quant_ckpt "$QUANT_CKPT" \
    --outdir "$RUN_ROOT/overall_resume_runtime" \
    --save_dir "$RUN_ROOT/overall" \
    --prompt_path "$PROMPTS" \
    --precompute_text_embeds "$EMBEDS" \
    --prompt_as_path \
    --prompt_indices "${PROMPT_INDICES[@]}" \
    --replay_original_prompt_rng \
    --num_videos 55 \
    --batch_size 1 \
    --num_sampling_steps 100 \
    --cfg_scale 4.0 \
    --sampler ddim \
    --seed 42 \
    --dataset_type opensora \
    --part_fp \
    --time_mp_config_weight "$MP_WEIGHT" \
    --time_mp_config_act "$MP_ACT" \
    2>&1 | tee "$RUN_ROOT/overall_generation_resume_38_92.log"

python - "$PROMPTS" "$RUN_ROOT/overall_opensora" <<'PY'
import sys
from pathlib import Path

prompts = [line.strip() for line in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines() if line.strip()]
actual = {path.name for path in Path(sys.argv[2]).glob("*.mp4")}
expected = {prompt + ".mp4" for prompt in prompts}
if actual != expected:
    raise SystemExit(
        f"final Overall video set mismatch: missing={sorted(expected-actual)[:5]}, "
        f"extra={sorted(actual-expected)[:5]}"
    )
print("Validated complete Overall video set: 93/93")
PY

conda activate vbench
cd "$VBENCH_REPO"
export PYTHONPATH="$VBENCH_REPO:${PYTHONPATH:-}"
export VBENCH_CACHE_DIR="$VBENCH_CACHE"
export VBENCH_BERT_DIR
mkdir -p "$RUN_ROOT/overall_eval"
printf 'EVAL_OVERALL %s\n' "$(date '+%F %T')" > "$STATUS_FILE"
stamp "Starting Overall VBench evaluation"
CUDA_VISIBLE_DEVICES="$GPU_INDEX" python -u evaluate.py \
    --videos_path "$RUN_ROOT/overall_opensora" \
    --output_path "$RUN_ROOT/overall_eval" \
    --dimension overall_consistency aesthetic_quality imaging_quality \
    --load_ckpt_from_local True \
    2>&1 | tee "$RUN_ROOT/evaluate_overall_resume.log"

[[ -s "$RUN_ROOT/overall_eval/overall_consistency_eval_results.json" ]] \
    || fail "Overall evaluation result missing"

python - "$RUN_ROOT" "$QUANT_CKPT" <<'PY'
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
quant_checkpoint = sys.argv[2]
sources = [
    root / "scene_eval" / "scene_eval_results.json",
    root / "subject_eval" / "subject_consistency_eval_results.json",
    root / "overall_eval" / "overall_consistency_eval_results.json",
]
order = [
    "imaging_quality", "aesthetic_quality", "motion_smoothness",
    "dynamic_degree", "background_consistency", "subject_consistency",
    "scene", "overall_consistency",
]
scores = {}
for path in sources:
    result = json.loads(path.read_text(encoding="utf-8"))
    for metric, value in result.items():
        scores[metric] = float(value[0] if isinstance(value, list) else value)
missing = [metric for metric in order if metric not in scores]
if missing:
    raise SystemExit(f"missing VBench metrics: {missing}")
summary = {
    "experiment": "FP16_FlashAttention_W4A6_baseline_with_FP32_T5_embeddings",
    "seed_protocol": "Each group starts at seed 42; Overall indices 38..92 resumed with full RNG replay",
    "quant_checkpoint": quant_checkpoint,
    "percentage_scores": {metric: scores[metric] * 100.0 for metric in order},
}
(root / "vbench_8metrics_summary.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
)
with (root / "vbench_8metrics_summary.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.writer(handle)
    writer.writerow(["metric", "percentage_score"])
    for metric in order:
        writer.writerow([metric, f"{scores[metric] * 100.0:.4f}"])
print(json.dumps(summary, indent=2, ensure_ascii=False))
PY

printf 'COMPLETED %s\n' "$(date '+%F %T')" > "$STATUS_FILE"
stamp "ALL_COMPLETE: Overall resumed, evaluated, and 8-metric summary generated"
