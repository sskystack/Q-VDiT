#!/usr/bin/env bash
set -Eeuo pipefail

# Full-precision FP16 VBench reference using the same inference/evaluation
# protocol as the formal baseline and MTD-1 runs.  QuantModel is retained only
# because quant_txt2video.py requires its config/checkpoint inputs; both
# quantizers are explicitly bypassed.

if [[ $# -ne 1 || -z "${TMUX:-}" ]]; then
  echo "Run from a dedicated tmux session: $0 GPU_ID" >&2
  exit 2
fi

GPU_ID=$1
ROOT=/home/zhouchongtian/quantization/qvdit_flash_bf16
CONDA_SH=/home/zhouchongtian/miniconda3/etc/profile.d/conda.sh
MODEL_CFG="$ROOT/t2v/configs/quant/opensora/16x512x512_fp16_flash_100steps.py"
MODEL_CKPT=/home/zhouchongtian/quantization/Q-VDiT/logs/split_ckpt/OpenSora-v1-HQ-16x512x512-split.pth
# These satisfy quant_txt2video.py's required inputs only; --skip_quant_* below
# ensures neither weight nor activation quantization is applied.
COMPAT_QUANT_CFG="$ROOT/t2v/configs/quant/opensora/w4a6_baseline.yaml"
COMPAT_QUANT_CKPT="$ROOT/logs/formal_baseline_calib_cfg4_ddim50_gpu2_scaler_20260806/calibration/ckpt.pth"
RUN_ROOT="$ROOT/logs/formal_vbench_fp16_cfg4_ddim100_seed42_gpu5_20260808_r3"
STATUS="$RUN_ROOT/status.txt"
VBENCH_ROOT=/home/zhouchongtian/quantization/eval/Vbench
MP_WEIGHT="$ROOT/t2v/configs/quant/opensora/mixed_precision/weight_4_mp.yaml"
MP_ACT="$ROOT/t2v/configs/quant/opensora/mixed_precision/act_6_mp.yaml"

[[ ! -e "$RUN_ROOT" ]] || { echo "Refusing to reuse output root: $RUN_ROOT" >&2; exit 1; }
for f in "$MODEL_CFG" "$MODEL_CKPT" "$COMPAT_QUANT_CFG" "$COMPAT_QUANT_CKPT" "$MP_WEIGHT" "$MP_ACT"; do
  [[ -s "$f" ]] || { echo "Missing required input: $f" >&2; exit 1; }
done
mkdir -p "$RUN_ROOT"
exec > >(tee -a "$RUN_ROOT/pipeline.log") 2>&1

stamp() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }
fail() { stamp "FAILED: $*"; printf 'FAILED %s: %s\n' "$(date '+%F %T')" "$*" > "$STATUS"; exit 1; }
trap 'fail "exit=$? line=$LINENO command=$BASH_COMMAND"' ERR

source "$CONDA_SH"
conda activate qvdit
cd "$ROOT"
export CUDA_VISIBLE_DEVICES=$GPU_ID PYTHONPATH="$ROOT:$ROOT/t2v"

{
  printf 'started_at: %s\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')"
  printf 'experiment: fp16_full_precision\n'
  printf 'gpu: %s\n' "$GPU_ID"
  printf 'git_commit: %s\n' "$(git rev-parse HEAD)"
  printf 'model_config: %s\nmodel_checkpoint: %s\n' "$MODEL_CFG" "$MODEL_CKPT"
  printf 'precision: fp16 inference; skip_quant_weight=true; skip_quant_act=true\n'
  printf 'sampling: DDIM, 100 steps, CFG 4.0, seed 42, batch size 1\n'
  printf 'prompt_protocol: official VBench groups; original indices; RNG replay\n'
  printf 'smoke_protocol: two prompts per group followed by VBench evaluation\n'
  printf 'compat_quant_config: %s\ncompat_quant_checkpoint: %s\n' "$COMPAT_QUANT_CFG" "$COMPAT_QUANT_CKPT"
} > "$RUN_ROOT/experiment_manifest.txt"

count_videos() { find "$1" -maxdepth 1 -type f -name '*.mp4' 2>/dev/null | wc -l | tr -d ' '; }
validate_prompt_count() {
  local path=$1 expected=$2
  [[ "$(grep -cve '^[[:space:]]*$' "$path")" -eq "$expected" ]] || fail "$path does not contain $expected prompts"
}
validate_videos() {
  local prompts=$1 videos=$2 expected=$3
  [[ "$(count_videos "$videos")" -eq "$expected" ]] || fail "video count in $videos is not $expected"
  python - "$prompts" "$videos" <<'PY'
import sys
from pathlib import Path
p = Path(sys.argv[1]); d = Path(sys.argv[2])
expected = {x.strip()+'.mp4' for x in p.read_text().splitlines() if x.strip()}
actual = {x.name for x in d.glob('*.mp4')}
if expected != actual:
    raise SystemExit(f'prompt/video mismatch: missing={sorted(expected-actual)[:3]}, extra={sorted(actual-expected)[:3]}')
PY
}

generate() {
  local name=$1 prompts=$2 expected=$3
  local runtime="$RUN_ROOT/${name}_runtime"
  local save="$RUN_ROOT/$name"
  local videos="${save}_opensora"
  [[ -s "$prompts" ]] || fail "missing prompts: $prompts"
  mkdir -p "$videos"
  local existing
  existing=$(count_videos "$videos")
  [[ "$existing" -eq 0 ]] || fail "unexpected pre-existing videos in fresh output root"
  local -a indices=()
  for ((i=0; i<expected; i++)); do indices+=("$i"); done
  stamp "INFER_${name^^}: 0/$expected"
  printf 'INFER_%s %s\n' "${name^^}" "$(date '+%F %T')" > "$STATUS"
  python -u t2v/scripts/quant_txt2video.py "$MODEL_CFG" \
    --ckpt_path "$MODEL_CKPT" --calib_config "$COMPAT_QUANT_CFG" --quant_ckpt "$COMPAT_QUANT_CKPT" \
    --outdir "$runtime" --save_dir "$save" --prompt_path "$prompts" \
    --prompt_indices "${indices[@]}" --replay_original_prompt_rng --prompt_as_path \
    --num_videos "$expected" --batch_size 1 --num_sampling_steps 100 --cfg_scale 4.0 \
    --sampler ddim --seed 42 --dataset_type opensora --skip_quant_weight --skip_quant_act \
    --time_mp_config_weight "$MP_WEIGHT" --time_mp_config_act "$MP_ACT" \
    2>&1 | tee "$RUN_ROOT/${name}_generation.log"
  validate_videos "$prompts" "$videos" "$expected"
  stamp "INFER_${name^^}_COMPLETE: $expected/$expected"
}

evaluate() {
  local name=$1 result=$2; shift 2
  local videos="$RUN_ROOT/${name}_opensora" out="$RUN_ROOT/${name}_eval"
  stamp "EVAL_${name^^}: $*"
  printf 'EVAL_%s %s\n' "${name^^}" "$(date '+%F %T')" > "$STATUS"
  conda activate vbench
  cd "$VBENCH_ROOT"
  CUDA_VISIBLE_DEVICES=$GPU_ID python -u evaluate.py --videos_path "$videos" --output_path "$out" \
    --dimension "$@" --load_ckpt_from_local True 2>&1 | tee "$RUN_ROOT/evaluate_${name}.log"
  [[ -s "$out/$result" ]] || fail "missing evaluation result: $out/$result"
  conda activate qvdit
  cd "$ROOT"
}

stamp 'PREFLIGHT: validate prompts, environments, and GPU'
validate_prompt_count "$ROOT/t2v/assets/texts/vbench_official/subject_consistency.txt" 72
validate_prompt_count "$ROOT/t2v/assets/texts/vbench_official/scene.txt" 86
validate_prompt_count "$ROOT/t2v/assets/texts/vbench_official/overall_consistency.txt" 93
nvidia-smi --query-gpu=index --format=csv,noheader,nounits | awk '{$1=$1; print}' | grep -qx "$GPU_ID" || fail "GPU $GPU_ID does not exist"
python -c 'import torch, transformers; print(torch.__version__, transformers.__version__)'
conda activate vbench; cd "$VBENCH_ROOT"
python -c 'import torch, decord; import vbench.scene, vbench.subject_consistency, vbench.overall_consistency; print(torch.__version__)'
conda activate qvdit; cd "$ROOT"

stamp 'SMOKE: two prompts per VBench group'
mkdir -p "$RUN_ROOT/smoke_prompts"
for group in subject scene overall; do
  case "$group" in
    subject) prompts="$ROOT/t2v/assets/texts/vbench_official/subject_consistency.txt"; result=subject_consistency_eval_results.json; metrics=(subject_consistency dynamic_degree motion_smoothness);;
    scene) prompts="$ROOT/t2v/assets/texts/vbench_official/scene.txt"; result=scene_eval_results.json; metrics=(scene background_consistency);;
    overall) prompts="$ROOT/t2v/assets/texts/vbench_official/overall_consistency.txt"; result=overall_consistency_eval_results.json; metrics=(overall_consistency aesthetic_quality imaging_quality);;
  esac
  smoke="$RUN_ROOT/smoke_prompts/${group}_two.txt"
  head -n 2 "$prompts" > "$smoke"
  generate "smoke_$group" "$smoke" 2
  evaluate "smoke_$group" "$result" "${metrics[@]}"
done
printf 'SMOKE_COMPLETE %s\n' "$(date '+%F %T')" > "$STATUS"
stamp 'SMOKE_COMPLETE'

stamp 'FORMAL: Subject -> Scene -> Overall'
generate subject "$ROOT/t2v/assets/texts/vbench_official/subject_consistency.txt" 72
evaluate subject subject_consistency_eval_results.json subject_consistency dynamic_degree motion_smoothness
generate scene "$ROOT/t2v/assets/texts/vbench_official/scene.txt" 86
evaluate scene scene_eval_results.json scene background_consistency
generate overall "$ROOT/t2v/assets/texts/vbench_official/overall_consistency.txt" 93
evaluate overall overall_consistency_eval_results.json overall_consistency aesthetic_quality imaging_quality

export RUN_ROOT
python - <<'PY'
import json, os
from pathlib import Path
root = Path(os.environ['RUN_ROOT'])
sources = [root/'subject_eval/subject_consistency_eval_results.json', root/'scene_eval/scene_eval_results.json', root/'overall_eval/overall_consistency_eval_results.json']
order = ['imaging_quality','aesthetic_quality','motion_smoothness','dynamic_degree','background_consistency','subject_consistency','scene','overall_consistency']
scores = {}
for f in sources:
    for k,v in json.loads(f.read_text()).items(): scores[k] = float(v[0] if isinstance(v,list) else v)
missing = [k for k in order if k not in scores]
if missing: raise RuntimeError(f'Missing metrics: {missing}')
(root/'vbench_8metrics_summary.json').write_text(json.dumps({'experiment':'fp16_full_precision','raw_scores':{k:scores[k] for k in order},'percentage_scores':{k:100*scores[k] for k in order}}, indent=2)+'\n')
PY
printf 'COMPLETE %s\n' "$(date '+%F %T')" > "$STATUS"
stamp 'COMPLETE: fp16_full_precision'
