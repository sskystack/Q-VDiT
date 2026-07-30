#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/zhouchongtian/quantization/qvdit_flash_bf16
PROMPT_ROOT="$ROOT/logs_fp16_flash/stage_numeric_profiling/h_calibration_stability_gpu6"
PHASE_ROOT="$ROOT/logs_fp16_flash/stage_numeric_profiling/j_phase_calibration_stability_gpu6"
SINGLE_ROOT="$ROOT/logs_fp16_flash/stage_numeric_profiling/k_single_layer_sensitivity_gpu6"
B3_ROOT="$ROOT/logs_fp16_flash/stage_quant_profiling/b3_seed42_cross_prompt_w4a6_fine_windows"

source /home/zhouchongtian/miniconda3/etc/profile.d/conda.sh
conda activate qvdit
cd "$ROOT"
export PYTHONPATH="$ROOT:$ROOT/t2v"

for required in \
  "$PROMPT_ROOT/status.txt" \
  "$PHASE_ROOT/status.txt" \
  "$SINGLE_ROOT/status.txt" \
  "$B3_ROOT/status.txt"; do
  test -s "$required"
  test "$(tr -d '[:space:]' < "$required")" = complete
done

python tools/analyze_calibration_outcomes.py \
  --labels subset_0 subset_1 subset_2 \
  --heldout-metrics \
    "$PROMPT_ROOT/subset_0/heldout_prompt0_seed42/metrics.json" \
    "$PROMPT_ROOT/subset_1/heldout_prompt0_seed42/metrics.json" \
    "$PROMPT_ROOT/subset_2/heldout_prompt0_seed42/metrics.json" \
  --pairwise-aggregate "$PROMPT_ROOT/checkpoint_comparison/pairwise_aggregate.csv" \
  --variance-aggregate "$PROMPT_ROOT/checkpoint_comparison/checkpoint_variance_aggregate.csv" \
  --output-dir "$PROMPT_ROOT/outcome_analysis" \
  --title "Prompt-subset calibration stability" \
  2>&1 | tee "$PROMPT_ROOT/outcome_analysis.log"

python tools/analyze_calibration_outcomes.py \
  --labels uniform early_heavy late_heavy \
  --heldout-metrics \
    "$PROMPT_ROOT/subset_0/heldout_prompt0_seed42/metrics.json" \
    "$PHASE_ROOT/early_heavy/heldout_prompt0_seed42/metrics.json" \
    "$PHASE_ROOT/late_heavy/heldout_prompt0_seed42/metrics.json" \
  --pairwise-aggregate "$PHASE_ROOT/checkpoint_comparison/pairwise_aggregate.csv" \
  --variance-aggregate "$PHASE_ROOT/checkpoint_comparison/checkpoint_variance_aggregate.csv" \
  --output-dir "$PHASE_ROOT/outcome_analysis" \
  --title "Phase-biased calibration" \
  2>&1 | tee "$PHASE_ROOT/outcome_analysis.log"

if [[ ! -s "$SINGLE_ROOT/analysis/single_layer_summary.json" ]]; then
  python tools/analyze_single_layer_sensitivity.py \
    --input "$SINGLE_ROOT/single_layer_sensitivity.jsonl" \
    --output-dir "$SINGLE_ROOT/analysis" \
    2>&1 | tee "$SINGLE_ROOT/analysis.log"
fi

conda activate vbench
export CUDA_VISIBLE_DEVICES=6
if [[ ! -s "$ROOT/logs_fp16_flash/stage_quant_profiling/b2_seed42_cross_prompt_w4a6/perceptual_analysis/b2_perceptual_summary.json" ]]; then
  python tools/analyze_b2_perceptual_quality.py \
    --root "$ROOT/logs_fp16_flash/stage_quant_profiling/b2_seed42_cross_prompt_w4a6" \
    --prompt-file "$ROOT/t2v/assets/texts/t2v_samples_10.txt" \
    --clip-checkpoint /home/zhouchongtian/quantization/models/vbench/clip_model/ViT-L-14.pt \
    --output-dir "$ROOT/logs_fp16_flash/stage_quant_profiling/b2_seed42_cross_prompt_w4a6/perceptual_analysis" \
    2>&1 | tee "$ROOT/logs_fp16_flash/stage_quant_profiling/b2_seed42_cross_prompt_w4a6/perceptual_analysis.log"
fi

python - <<'PY'
import json
from pathlib import Path

root = Path('/home/zhouchongtian/quantization/qvdit_flash_bf16')
artifacts = {
    'prompt_calibration': root / 'logs_fp16_flash/stage_numeric_profiling/h_calibration_stability_gpu6/outcome_analysis/calibration_outcome_summary.json',
    'phase_calibration': root / 'logs_fp16_flash/stage_numeric_profiling/j_phase_calibration_stability_gpu6/outcome_analysis/calibration_outcome_summary.json',
    'single_layer': root / 'logs_fp16_flash/stage_numeric_profiling/k_single_layer_sensitivity_gpu6/analysis/single_layer_summary.json',
    'fine_windows': root / 'logs_fp16_flash/stage_quant_profiling/b3_seed42_cross_prompt_w4a6_fine_windows/b3_summary.json',
    'b2_perceptual': root / 'logs_fp16_flash/stage_quant_profiling/b2_seed42_cross_prompt_w4a6/perceptual_analysis/b2_perceptual_summary.json',
}
audit = {'artifacts': {}, 'all_present': True, 'all_numerically_valid': True}
for label, path in artifacts.items():
    present = path.is_file() and path.stat().st_size > 0
    item = {'path': str(path), 'present': present}
    if present:
        data = json.loads(path.read_text())
        if label == 'fine_windows':
            valid = all(data.get(key, False) for key in (
                'all_latents_finite', 'all_initial_noise_exact', 'all_traces_valid'
            ))
        else:
            valid = bool(data.get('all_values_finite', False))
        item['numerically_valid'] = valid
        audit['all_numerically_valid'] = audit['all_numerically_valid'] and valid
    else:
        audit['all_present'] = False
        audit['all_numerically_valid'] = False
    audit['artifacts'][label] = item
output = root / 'logs_fp16_flash/stage_numeric_profiling/remaining_profiling_completion_audit.json'
output.write_text(json.dumps(audit, indent=2))
print(json.dumps(audit, indent=2))
if not audit['all_present'] or not audit['all_numerically_valid']:
    raise SystemExit(1)
PY

echo complete > "$ROOT/logs_fp16_flash/stage_numeric_profiling/remaining_profiling_postprocess_status.txt"
