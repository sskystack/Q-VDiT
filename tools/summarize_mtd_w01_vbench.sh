#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/home/zhouchongtian/quantization/qvdit_flash_bf16/logs_fp16_flash/vbench_mtd_w01_fp16gs_final10000
LOG="$ROOT/summary_waiter.log"
mkdir -p "$ROOT"
exec > >(tee -a "$LOG") 2>&1

FILES=(
  "$ROOT/subject_eval/subject_consistency_eval_results.json"
  "$ROOT/scene_eval/scene_eval_results.json"
  "$ROOT/overall_eval/overall_consistency_eval_results.json"
)

echo "[$(date '+%F %T')] Waiting for all three evaluations"
while true; do
    ready=1
    for f in "${FILES[@]}"; do
        [[ -s "$f" ]] || ready=0
    done
    [[ "$ready" -eq 1 ]] && break
    sleep 60
done

python - "$ROOT" <<'PY'
import csv, json, sys
from pathlib import Path

root = Path(sys.argv[1])
paths = [
    root / "subject_eval/subject_consistency_eval_results.json",
    root / "scene_eval/scene_eval_results.json",
    root / "overall_eval/overall_consistency_eval_results.json",
]
order = [
    "imaging_quality", "aesthetic_quality", "motion_smoothness",
    "dynamic_degree", "background_consistency", "subject_consistency",
    "scene", "overall_consistency",
]
scores = {}
for path in paths:
    data = json.loads(path.read_text(encoding="utf-8"))
    for key, value in data.items():
        scores[key] = float(value[0] if isinstance(value, list) else value)
missing = [key for key in order if key not in scores]
if missing:
    raise RuntimeError(f"Missing metrics: {missing}")

summary = {
    "experiment": "FP16_FLASH_MTD_W01_W4A6_DDIM100_CFG4_SEED42",
    "raw_scores": {k: scores[k] for k in order},
    "percentage_scores": {k: scores[k] * 100 for k in order},
}
(root / "vbench_8metrics_summary.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
)
with (root / "vbench_8metrics_summary.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["metric", "raw_score", "percentage_score"])
    for key in order:
        writer.writerow([key, f"{scores[key]:.8f}", f"{scores[key] * 100:.4f}"])
lines = [
    "FP16 FlashAttention MTD-0.1 W4A6 - DDIM 100 - CFG 4.0 - seed 42",
    "", f"{'Metric':<28}{'Raw':>12}{'x100':>12}", "-" * 52,
]
for key in order:
    lines.append(f"{key:<28}{scores[key]:>12.6f}{scores[key] * 100:>12.2f}")
text = "\n".join(lines) + "\n"
(root / "vbench_8metrics_summary.txt").write_text(text, encoding="utf-8")
print(text)
PY

echo "[$(date '+%F %T')] ALL_COMPLETE" | tee "$ROOT/status.txt"
