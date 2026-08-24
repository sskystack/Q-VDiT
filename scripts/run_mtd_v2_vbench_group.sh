#!/usr/bin/env bash
set -Eeuo pipefail

# Run exactly one official VBench group while reusing the baseline runner's
# functions and command line.  This permits independent GPU scheduling without
# concurrent jobs sharing status files, locks, or output directories.
GROUP=${GROUP:?set GROUP to scene, subject, or overall}
BASELINE_RUNNER=${BASELINE_RUNNER:-/home/zhouchongtian/quantization/qvdit_flash_bf16/tools/run_baseline_fp32t5_vbench_gpu2.sh}
COMMON_ROOT=${COMMON_ROOT:?set COMMON_ROOT}
REPO=${REPO:?set REPO}
QUANT_CKPT=${QUANT_CKPT:?set QUANT_CKPT}
GPU_INDEX=${GPU_INDEX:?set GPU_INDEX}
EMBED_ROOT=${EMBED_ROOT:?set EMBED_ROOT}
RUN_ROOT="$COMMON_ROOT/groups/$GROUP"
LOCK_FILE="$COMMON_ROOT/locks/gpu${GPU_INDEX}_${GROUP}.lock"
EXPERIMENT_NAME=${EXPERIMENT_NAME:-MTD_V2_W4A6_FP16_FlashAttention_DDIM100_CFG4_seed42}
QUANT_CFG=${QUANT_CFG:-$REPO/t2v/configs/quant/opensora/w4a6_mtd_v2.yaml}
WAIT_FOR_GPU_IDLE=${WAIT_FOR_GPU_IDLE:-1}
GPU_POLL_SECONDS=${GPU_POLL_SECONDS:-30}

[[ -s "$BASELINE_RUNNER" ]] || { echo "missing baseline runner: $BASELINE_RUNNER" >&2; exit 1; }
mkdir -p "$RUN_ROOT" "$COMMON_ROOT/locks"

# Load the baseline implementation through its function definitions, stopping
# before its top-level three-group workflow. Environment overrides above keep
# this job isolated while preserving baseline inference/evaluation semantics.
source <(awk '/^trap /{exit} {print}' "$BASELINE_RUNNER")

case "$GROUP" in
    scene)
        prompts=$SCENE_PROMPTS; embeds=$SCENE_EMBEDS; expected=86
        result_file=scene_eval_results.json
        metrics=(scene background_consistency)
        ;;
    subject)
        prompts=$SUBJECT_PROMPTS; embeds=$SUBJECT_EMBEDS; expected=72
        result_file=subject_consistency_eval_results.json
        metrics=(subject_consistency dynamic_degree motion_smoothness)
        ;;
    overall)
        prompts=$OVERALL_PROMPTS; embeds=$OVERALL_EMBEDS; expected=93
        result_file=overall_consistency_eval_results.json
        metrics=(overall_consistency aesthetic_quality imaging_quality)
        ;;
    *) fail "unknown GROUP=$GROUP" ;;
esac

trap 'fail "command failed at line $LINENO: $BASH_COMMAND"' ERR

stamp "Preflight for $GROUP on physical GPU $GPU_INDEX"
for path in "$MODEL_CFG" "$QUANT_CFG" "$MODEL_CKPT" "$QUANT_CKPT" "$MP_WEIGHT" "$MP_ACT"; do
    require_file "$path"
done
validate_prompt_file "$prompts" "$expected"
wait_for_gpu_idle

source "$CONDA_SH"
conda activate qvdit
cd "$REPO"
export PYTHONPATH="$REPO:$REPO/t2v"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
generate_embed "$GROUP" "$prompts" "$embeds" "$expected"

conda activate vbench
cd "$VBENCH_REPO"
export PYTHONPATH="$VBENCH_REPO:${PYTHONPATH:-}"
export VBENCH_CACHE_DIR="$VBENCH_CACHE"
export VBENCH_BERT_DIR
python -c 'import torch, decord; import vbench.scene, vbench.subject_consistency, vbench.overall_consistency; print(torch.__version__)'
conda activate qvdit
cd "$REPO"

stamp "Starting $GROUP two-prompt smoke inference and evaluation"
run_smoke_group "$GROUP" "$prompts" "$embeds" "$result_file" "${metrics[@]}"
printf 'SMOKE_COMPLETE %s\n' "$(date '+%F %T')" > "$STATUS_FILE"

stamp "Starting formal $GROUP inference and evaluation"
run_inference "$GROUP" "$prompts" "$embeds" "$expected" "$RUN_ROOT"
conda activate vbench
cd "$VBENCH_REPO"
run_eval "$GROUP" "$result_file" "$RUN_ROOT" "${metrics[@]}"
printf 'COMPLETED %s\n' "$(date '+%F %T')" > "$STATUS_FILE"
stamp "GROUP_COMPLETE: $GROUP"
