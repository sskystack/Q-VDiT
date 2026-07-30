#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/home/zhouchongtian/quantization/qvdit_flash_bf16
EXP01="$ROOT/logs_fp16_flash/formal_w4a6_mtd_noglobal_w01_gpu6_0730"
EXP1="$ROOT/logs_fp16_flash/formal_w4a6_mtd_noglobal_w1_gpu5_0730"
OUT="$ROOT/logs_fp16_flash/mtd_noglobal_1000_audit"
mkdir -p "$OUT"
exec > >(tee -a "$OUT/watcher.log") 2>&1

ready() {
    local exp=$1
    [[ -s "$exp/calibration/reconstruction_state_iter_00001000.pth" ]] &&
    grep -q '"iteration": 1000' "$exp/calibration/loss_component_gradients.jsonl" 2>/dev/null
}

echo "[$(date '+%F %T')] Waiting for both 1000-step probes"
until ready "$EXP01" && ready "$EXP1"; do
    sleep 30
done

source /home/zhouchongtian/miniconda3/etc/profile.d/conda.sh
conda activate qvdit
cd "$ROOT"
for exp in "$EXP01" "$EXP1"; do
    python tools/summarize_mtd_subterm_gradients.py \
      "$exp/calibration/loss_component_gradients.jsonl" \
      --output-prefix "$exp/calibration/mtd_subterm_gradient_summary_1000"
done

python - "$EXP01" "$EXP1" "$OUT" <<'PY'
import json, sys
from pathlib import Path

experiments = [("weight_0.1", Path(sys.argv[1])), ("weight_1.0", Path(sys.argv[2]))]
out = Path(sys.argv[3])
report = {}
lines = ["MTD no-global gradient audit through iteration 1000", ""]
for name, root in experiments:
    records = [json.loads(x) for x in (root / "calibration/loss_component_gradients.jsonl").read_text().splitlines()]
    records = [r for r in records if int(r["iteration"]) <= 1000]
    rows = []
    for r in records:
        def pair_cos(total_norm, first_norm, second_norm):
            denominator = 2.0 * first_norm * second_norm
            if denominator == 0.0:
                return None
            value = (
                total_norm ** 2 - first_norm ** 2 - second_norm ** 2
            ) / denominator
            return max(-1.0, min(1.0, value))

        local_norm = r["subterms"]["local"]["grad_norm"]
        motion_norm = r["subterms"]["motion"]["grad_norm"]
        mtd_norm = r["gradient"]["mtd_norm"]
        local_motion_cosine = pair_cos(mtd_norm, local_norm, motion_norm)
        category_pair_cosines = {}
        for category in ("lora", "delta"):
            category_pair_cosines[category] = pair_cos(
                r["categories"]["mtd"][category]["norm"],
                r["subterms"]["local"]["categories"][category]["norm"],
                r["subterms"]["motion"]["categories"][category]["norm"],
            )
        row = {
            "iteration": r["iteration"],
            "mtd_ratio": r["gradient"]["mtd_to_rec_ratio"],
            "mtd_cosine": r["gradient"]["cosine"],
            "local_ratio": r["subterms"]["local"]["ratio_to_rec"],
            "local_cosine": r["subterms"]["local"]["cosine_with_rec"],
            "motion_ratio": r["subterms"]["motion"]["ratio_to_rec"],
            "motion_cosine": r["subterms"]["motion"]["cosine_with_rec"],
            "global_ratio": r["subterms"]["global"]["ratio_to_rec"],
            "local_motion_cosine": local_motion_cosine,
            "local_motion_cosine_lora": category_pair_cosines["lora"],
            "local_motion_cosine_delta": category_pair_cosines["delta"],
        }
        rows.append(row)
    report[name] = rows
    lines.append(name)
    lines.append(
        "iter  mtd_r  mtd_cos  local_r  local_cos  motion_r  motion_cos  "
        "L-M_cos  L-M_lora  L-M_delta  global_r"
    )
    for x in rows:
        lines.append(
            f'{x["iteration"]:4d}  {x["mtd_ratio"]:6.3f}  {x["mtd_cosine"]:+7.3f}  '
            f'{x["local_ratio"]:7.3f}  {x["local_cosine"]:+9.3f}  '
            f'{x["motion_ratio"]:8.3f}  {x["motion_cosine"]:+10.3f}  '
            f'{x["local_motion_cosine"]:+7.3f}  '
            f'{x["local_motion_cosine_lora"]:+8.3f}  '
            f'{x["local_motion_cosine_delta"]:+9.3f}  '
            f'{x["global_ratio"]:8.3f}'
        )
    lines.append("")
(out / "audit_1000.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
(out / "audit_1000.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
PY

echo "READY_FOR_MANUAL_REVIEW $(date '+%F %T')" | tee "$OUT/status.txt"
