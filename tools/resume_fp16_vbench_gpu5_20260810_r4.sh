#!/usr/bin/env bash
set -Eeuo pipefail

# Resume the interrupted r3 FP16 full-precision VBench run without changing it.
if [[ -z "${TMUX:-}" ]]; then
  echo "Run from a dedicated tmux session." >&2
  exit 2
fi

ROOT=/home/zhouchongtian/quantization/qvdit_flash_bf16
SRC="$ROOT/logs/formal_vbench_fp16_cfg4_ddim100_seed42_gpu5_20260808_r3"
RUN="$ROOT/logs/formal_vbench_fp16_cfg4_ddim100_seed42_gpu5_20260808_r4"
GPU=5
MODEL_CFG="$ROOT/t2v/configs/quant/opensora/16x512x512_fp16_flash_100steps.py"
MODEL_CKPT=/home/zhouchongtian/quantization/Q-VDiT/logs/split_ckpt/OpenSora-v1-HQ-16x512x512-split.pth
COMPAT_CFG="$ROOT/t2v/configs/quant/opensora/w4a6_baseline.yaml"
COMPAT_CKPT="$ROOT/logs/formal_baseline_calib_cfg4_ddim50_gpu2_scaler_20260806/calibration/ckpt.pth"
MP_WEIGHT="$ROOT/t2v/configs/quant/opensora/mixed_precision/weight_4_mp.yaml"
MP_ACT="$ROOT/t2v/configs/quant/opensora/mixed_precision/act_6_mp.yaml"
CONDA_SH=/home/zhouchongtian/miniconda3/etc/profile.d/conda.sh
VBENCH_ROOT=/home/zhouchongtian/quantization/eval/Vbench

[[ -d "$SRC" && ! -e "$RUN" ]] || { echo "Invalid source or pre-existing resume root" >&2; exit 1; }
for f in "$MODEL_CFG" "$MODEL_CKPT" "$COMPAT_CFG" "$COMPAT_CKPT" "$MP_WEIGHT" "$MP_ACT"; do [[ -s "$f" ]] || exit 1; done
mkdir -p "$RUN"
exec > >(tee -a "$RUN/pipeline.log") 2>&1
stamp() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }
fail() { stamp "FAILED: $*"; printf 'FAILED %s: %s\n' "$(date '+%F %T')" "$*" > "$RUN/status.txt"; exit 1; }
trap 'fail "line=$LINENO command=$BASH_COMMAND"' ERR

printf 'RESUMED_FROM: %s\nscene_resume_index: 29\n' "$SRC" > "$RUN/experiment_manifest.txt"
source "$CONDA_SH"; conda activate qvdit; cd "$ROOT"
export CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH="$ROOT:$ROOT/t2v"

# Preserve r3 unchanged. Reuse only validated complete outputs.
ln -s "$SRC/subject_opensora" "$RUN/subject_opensora"
ln -s "$SRC/subject_eval" "$RUN/subject_eval"
for name in smoke_subject smoke_scene smoke_overall; do
  ln -s "$SRC/${name}_opensora" "$RUN/${name}_opensora"
  ln -s "$SRC/${name}_eval" "$RUN/${name}_eval"
done
mkdir -p "$RUN/scene_opensora"
mapfile -t scene_prompts < <(sed '/^[[:space:]]*$/d' "$ROOT/t2v/assets/texts/vbench_official/scene.txt")
for ((i=0; i<29; i++)); do
  name=${scene_prompts[$i]}
  f="$SRC/scene_opensora/$name.mp4"
  [[ -s "$f" ]] || fail "r3 scene prefix invalid at index $i"
  ln "$f" "$RUN/scene_opensora/$name.mp4"
done

infer_remaining() {
  local group=$1 prompts=$2 start=$3 total=$4
  local -a indices=()
  for ((i=start; i<total; i++)); do indices+=("$i"); done
  stamp "INFER_${group^^}: $start/$total"
  printf 'INFER_%s %s\n' "${group^^}" "$(date '+%F %T')" > "$RUN/status.txt"
  python -u t2v/scripts/quant_txt2video.py "$MODEL_CFG" \
    --ckpt_path "$MODEL_CKPT" --calib_config "$COMPAT_CFG" --quant_ckpt "$COMPAT_CKPT" \
    --outdir "$RUN/${group}_runtime" --save_dir "$RUN/$group" --prompt_path "$prompts" \
    --prompt_indices "${indices[@]}" --replay_original_prompt_rng --prompt_as_path \
    --num_videos "$total" --batch_size 1 --num_sampling_steps 100 --cfg_scale 4.0 --sampler ddim --seed 42 \
    --dataset_type opensora --skip_quant_weight --skip_quant_act \
    --time_mp_config_weight "$MP_WEIGHT" --time_mp_config_act "$MP_ACT" 2>&1 | tee "$RUN/${group}_generation.log"
}
validate_group() {
  local group=$1 prompts=$2 total=$3
  python - "$RUN/${group}_opensora" "$prompts" "$total" <<'PY'
import sys
from pathlib import Path
d,p,n=Path(sys.argv[1]),Path(sys.argv[2]),int(sys.argv[3])
expected={x.strip()+'.mp4' for x in p.read_text().splitlines() if x.strip()}
actual={x.name for x in d.glob('*.mp4') if x.stat().st_size>1024}
if len(actual)!=n or actual!=expected: raise SystemExit(f'video validation failed: {len(actual)}/{n}, missing={sorted(expected-actual)[:3]}')
PY
}
evaluate() {
  local group=$1 result=$2; shift 2
  printf 'EVAL_%s %s\n' "${group^^}" "$(date '+%F %T')" > "$RUN/status.txt"
  conda activate vbench; cd "$VBENCH_ROOT"
  CUDA_VISIBLE_DEVICES=$GPU python -u evaluate.py --videos_path "$RUN/${group}_opensora" --output_path "$RUN/${group}_eval" --dimension "$@" --load_ckpt_from_local True 2>&1 | tee "$RUN/evaluate_${group}.log"
  [[ -s "$RUN/${group}_eval/$result" ]] || fail "missing $group result"
  conda activate qvdit; cd "$ROOT"
}

SCENE="$ROOT/t2v/assets/texts/vbench_official/scene.txt"
OVERALL="$ROOT/t2v/assets/texts/vbench_official/overall_consistency.txt"
infer_remaining scene "$SCENE" 29 86
validate_group scene "$SCENE" 86
evaluate scene scene_eval_results.json scene background_consistency
infer_remaining overall "$OVERALL" 0 93
validate_group overall "$OVERALL" 93
evaluate overall overall_consistency_eval_results.json overall_consistency aesthetic_quality imaging_quality

export RUN SRC
python - <<'PY'
import json, os
from pathlib import Path
r=Path(os.environ['RUN'])
order=['imaging_quality','aesthetic_quality','motion_smoothness','dynamic_degree','background_consistency','subject_consistency','scene','overall_consistency']
scores={}
for f in [r/'subject_eval/subject_consistency_eval_results.json',r/'scene_eval/scene_eval_results.json',r/'overall_eval/overall_consistency_eval_results.json']:
 for k,v in json.loads(f.read_text()).items(): scores[k]=float(v[0] if isinstance(v,list) else v)
missing=[k for k in order if k not in scores]
if missing: raise RuntimeError(missing)
(r/'vbench_8metrics_summary.json').write_text(json.dumps({'experiment':'fp16_full_precision_resumed','resumed_from':os.environ.get('SRC','r3'),'percentage_scores':{k:100*scores[k] for k in order}},indent=2)+'\n')
PY
printf 'COMPLETE %s\n' "$(date '+%F %T')" > "$RUN/status.txt"
stamp 'COMPLETE: FP16 full-precision VBench resume'
