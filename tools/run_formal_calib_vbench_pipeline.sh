#!/usr/bin/env bash
set -Eeuo pipefail

# Canonical baseline template. New methods should copy this script and change
# only the method-specific quantization configuration and output paths.

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 GPU_ID" >&2
    exit 2
fi
if [[ -z "${TMUX:-}" ]]; then
    echo "Run this experiment from a new dedicated tmux window." >&2
    exit 2
fi

GPU_ID=$1
EXPERIMENT_NAME=${EXPERIMENT_NAME:-baseline}
ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
CONDA_SH=${CONDA_SH:-/home/zhouchongtian/miniconda3/etc/profile.d/conda.sh}
QVEDIT_ENV=${QVEDIT_ENV:-qvdit}
VBENCH_ENV=${VBENCH_ENV:-vbench}
VBENCH_ROOT=${VBENCH_ROOT:-/home/zhouchongtian/quantization/eval/Vbench}
MODEL_CKPT=${MODEL_CKPT:-$ROOT/logs/split_ckpt/OpenSora-v1-HQ-16x512x512-split.pth}
CALIB_CFG=${CALIB_CFG:-$ROOT/t2v/configs/quant/opensora/16x512x512_fp16_flash_50steps.py}
INFER_CFG=${INFER_CFG:-$ROOT/t2v/configs/quant/opensora/16x512x512_fp16_flash_100steps.py}
QUANT_CFG=${QUANT_CFG:-$ROOT/t2v/configs/quant/opensora/w4a6_baseline.yaml}
PROMPTS_10=${PROMPTS_10:-$ROOT/t2v/assets/texts/t2v_samples_10.txt}
CALIB_ROOT=${CALIB_ROOT:-$ROOT/logs/formal_baseline_calib_cfg4_ddim50}
CALIB_DATA_DIR=${CALIB_DATA_DIR:-$CALIB_ROOT/data}
CALIB_OUT=${CALIB_OUT:-$CALIB_ROOT/calibration}
RUN_ROOT=${RUN_ROOT:-$ROOT/logs/formal_vbench_${EXPERIMENT_NAME}_cfg4_ddim100_seed42}
QUANT_CKPT=$CALIB_OUT/ckpt.pth
MP_WEIGHT=${MP_WEIGHT:-$ROOT/t2v/configs/quant/opensora/mixed_precision/weight_4_mp.yaml}
MP_ACT=${MP_ACT:-$ROOT/t2v/configs/quant/opensora/mixed_precision/act_6_mp.yaml}
export CUDA_VISIBLE_DEVICES=$GPU_ID PYTHONPATH="$ROOT:$ROOT/t2v"
export QVDIT_TQE_RANK=1

[[ -s "$MODEL_CKPT" && -s "$CALIB_CFG" && -s "$INFER_CFG" && -s "$QUANT_CFG" && -s "$PROMPTS_10" && -s "$MP_WEIGHT" && -s "$MP_ACT" ]] || { echo "Missing model/config/prompt input" >&2; exit 1; }
mkdir -p "$CALIB_DATA_DIR" "$CALIB_OUT" "$RUN_ROOT"
exec > >(tee -a "$RUN_ROOT/pipeline.log") 2>&1
stamp() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }
require_file() { [[ -s "$1" ]] || { stamp "Missing: $1"; exit 1; }; }
on_error() {
    local status=$1 line=$2 command=$3
    trap - ERR
    stamp "FAILED: exit=$status line=$line command=$command"
    exit "$status"
}
trap 'on_error "$?" "$LINENO" "$BASH_COMMAND"' ERR

source "$CONDA_SH"
conda activate "$QVEDIT_ENV"
cd "$ROOT"

{
    printf 'started_at: %s\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')"
    printf 'command: bash %q %q\n' "$0" "$GPU_ID"
    printf 'experiment: %s\n' "$EXPERIMENT_NAME"
    printf 'git_commit: %s\n' "$(git rev-parse HEAD)"
    printf 'git_status:\n'
    git status --short
    printf 'cuda_visible_devices: %s\n' "$CUDA_VISIBLE_DEVICES"
    printf 'conda_environment: %s\n' "$CONDA_DEFAULT_ENV"
    printf 'python: %s\n' "$(python --version 2>&1)"
    printf 'model_checkpoint: %s\n' "$MODEL_CKPT"
    printf 'calibration_config: %s\n' "$CALIB_CFG"
    printf 'inference_config: %s\n' "$INFER_CFG"
    printf 'quantization_config: %s\n' "$QUANT_CFG"
    printf 'calibration_data: %s\n' "$CALIB_DATA_DIR/calib_data.pt"
    printf 'mixed_precision_weight: %s\n' "$MP_WEIGHT"
    printf 'mixed_precision_activation: %s\n' "$MP_ACT"
} > "$RUN_ROOT/experiment_manifest.txt"

CALIB_DATA="$CALIB_DATA_DIR/calib_data.pt"
if [[ ! -s "$CALIB_DATA" ]]; then
    stamp "Generating calibration data: samples_10, DDIM, CFG 4.0, 50 steps"
    CUDA_VISIBLE_DEVICES="$GPU_ID" python -u t2v/scripts/get_calib_data.py "$CALIB_CFG" \
        --ckpt_path "$MODEL_CKPT" --prompt_path "$PROMPTS_10" --data_num 10 \
        --num_sampling_steps 50 --cfg_scale 4.0 --sampler ddim \
        --seed 42 --batch_size 1 --gpu 0 --outdir "$CALIB_DATA_DIR" --save_dir "$CALIB_DATA_DIR"
else
    stamp "Using existing calibration data: $CALIB_DATA"
fi
require_file "$CALIB_DATA"

if [[ ! -s "$QUANT_CKPT" ]]; then
    stamp "Running $EXPERIMENT_NAME calibration"
    CUDA_VISIBLE_DEVICES="$GPU_ID" python -u t2v/scripts/calib.py "$CALIB_CFG" \
        --ckpt_path "$MODEL_CKPT" --calib_config "$QUANT_CFG" --calib_data "$CALIB_DATA" \
        --prompt_path "$PROMPTS_10" --num_sampling_steps 50 --cfg_scale 4.0 --sampler ddim \
        --outdir "$CALIB_OUT" --seed 42 --batch_size 1 --gpu 0 \
        --part_fp --time_mp_config_weight "$MP_WEIGHT" --time_mp_config_act "$MP_ACT" \
        --use_grad_scaler
else
    stamp "Using existing baseline checkpoint: $QUANT_CKPT"
fi
require_file "$QUANT_CKPT"

count_videos() { find "$1" -maxdepth 1 -type f -name '*.mp4' | wc -l | tr -d ' '; }

generate_group() {
    local name=$1 prompts=$2 expected=$3
    local runtime="$RUN_ROOT/${name}_runtime" save_base="$RUN_ROOT/$name"
    local video_dir="${save_base}_opensora"
    require_file "$prompts"
    mkdir -p "$video_dir"
    local start=0
    while [[ $start -lt $expected ]]; do
        local path
        path=$(sed -n "$((start + 1))p" "$prompts")
        [[ -s "$video_dir/${path}.mp4" ]] || break
        start=$((start + 1))
    done
    for ((i=start + 1; i<expected; i++)); do
        local later_path
        later_path=$(sed -n "$((i + 1))p" "$prompts")
        if [[ -s "$video_dir/${later_path}.mp4" ]]; then
            stamp "$name has a non-contiguous result: index $start is missing but index $i exists"
            exit 1
        fi
    done
    if [[ $start -eq $expected ]]; then
        stamp "$name generation already complete"
        return
    fi
    stamp "Generating $name from original prompt index $start"
    local -a indices=()
    for ((i=start; i<expected; i++)); do indices+=("$i"); done
    CUDA_VISIBLE_DEVICES="$GPU_ID" python -u t2v/scripts/quant_txt2video.py "$INFER_CFG" \
        --ckpt_path "$MODEL_CKPT" --calib_config "$QUANT_CFG" --quant_ckpt "$QUANT_CKPT" \
        --outdir "$runtime" --save_dir "$save_base" --prompt_path "$prompts" \
        --prompt_indices "${indices[@]}" --replay_original_prompt_rng \
        --prompt_as_path --num_videos "$expected" --batch_size 1 \
        --num_sampling_steps 100 --cfg_scale 4.0 --sampler ddim --seed 42 \
        --dataset_type opensora --part_fp --time_mp_config_weight "$MP_WEIGHT" \
        --time_mp_config_act "$MP_ACT"
    [[ "$(count_videos "$video_dir")" -eq "$expected" ]] || { stamp "$name generation incomplete"; exit 1; }
}

evaluate_group() {
    local name=$1 result=$2; shift 2
    local out="$RUN_ROOT/${name}_eval"
    [[ -s "$out/$result" ]] && { stamp "$name evaluation already complete"; return; }
    rm -rf "$out"; mkdir -p "$out"
    conda activate "$VBENCH_ENV"; cd "$VBENCH_ROOT"
    CUDA_VISIBLE_DEVICES="$GPU_ID" python evaluate.py --videos_path "$RUN_ROOT/${name}_opensora" \
        --output_path "$out" --dimension "$@" --load_ckpt_from_local True
    require_file "$out/$result"
    conda activate "$QVEDIT_ENV"; cd "$ROOT"
}

generate_group subject "$ROOT/t2v/assets/texts/vbench_official/subject_consistency.txt" 72
generate_group scene "$ROOT/t2v/assets/texts/vbench_official/scene.txt" 86
generate_group overall "$ROOT/t2v/assets/texts/vbench_official/overall_consistency.txt" 93
evaluate_group subject subject_consistency_eval_results.json subject_consistency dynamic_degree motion_smoothness
evaluate_group scene scene_eval_results.json scene background_consistency
evaluate_group overall overall_consistency_eval_results.json overall_consistency aesthetic_quality imaging_quality
export RUN_ROOT EXPERIMENT_NAME QUANT_CKPT
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
    "imaging_quality", "aesthetic_quality", "motion_smoothness",
    "dynamic_degree", "background_consistency", "subject_consistency",
    "scene", "overall_consistency",
]
scores = {}
for source in sources:
    payload = json.loads(source.read_text(encoding="utf-8"))
    for key, value in payload.items():
        scores[key] = float(value[0] if isinstance(value, list) else value)
missing = [key for key in order if key not in scores]
if missing:
    raise RuntimeError(f"Missing VBench metrics: {missing}")
summary = {
    "experiment": os.environ["EXPERIMENT_NAME"],
    "quant_checkpoint": os.environ["QUANT_CKPT"],
    "raw_scores": {key: scores[key] for key in order},
    "percentage_scores": {key: 100.0 * scores[key] for key in order},
}
(root / "vbench_8metrics_summary.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)
PY
stamp "COMPLETE: $EXPERIMENT_NAME"
