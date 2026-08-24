#!/usr/bin/env python3
"""Compare FP16, W4A6, and W4A6+MTD motion-related VBench results."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


METRICS = (
    "dynamic_degree",
    "motion_smoothness",
    "temporal_flickering",
    "subject_consistency",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp16-eval", type=Path, nargs="+", required=True)
    parser.add_argument("--baseline-eval", type=Path, nargs="+", required=True)
    parser.add_argument("--mtd-eval", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_eval_tree(roots):
    scores = {}
    per_video = {}
    for root in roots:
        for path in sorted(root.rglob("*eval_results.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            for metric, value in payload.items():
                if metric not in METRICS:
                    continue
                aggregate = value[0] if isinstance(value, list) else value
                details = value[1] if isinstance(value, list) and len(value) > 1 else []
                scores[metric] = float(aggregate)
                per_video[metric] = {
                    Path(item["video_path"]).name: item["video_results"] for item in details
                }
    missing = [metric for metric in METRICS if metric not in scores]
    if missing:
        raise RuntimeError(f"Missing metrics under {roots}: {missing}")
    return scores, per_video


def exact_two_sided_binomial(left: int, right: int):
    total = left + right
    if total == 0:
        return 1.0
    cutoff = min(left, right)
    probability = sum(math.comb(total, k) for k in range(cutoff + 1)) / (2**total)
    return min(1.0, 2.0 * probability)


def mean(values):
    return sum(values) / len(values) if values else float("nan")


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    roots = {
        "fp16": args.fp16_eval,
        "baseline": args.baseline_eval,
        "mtd": args.mtd_eval,
    }
    aggregate = {}
    per_video = {}
    for variant, root in roots.items():
        aggregate[variant], per_video[variant] = load_eval_tree(root)

    names = sorted(set(per_video["fp16"]["dynamic_degree"]))
    for variant in roots:
        current = set(per_video[variant]["dynamic_degree"])
        if current != set(names):
            raise RuntimeError(
                f"Dynamic Degree video set mismatch for {variant}: "
                f"missing={sorted(set(names) - current)}, extra={sorted(current - set(names))}"
            )

    dynamic_rows = []
    repaired = 0
    broken = 0
    baseline_disagreements = 0
    mtd_disagreements = 0
    for name in names:
        fp = bool(per_video["fp16"]["dynamic_degree"][name])
        baseline = bool(per_video["baseline"]["dynamic_degree"][name])
        mtd = bool(per_video["mtd"]["dynamic_degree"][name])
        baseline_wrong = baseline != fp
        mtd_wrong = mtd != fp
        repaired_here = baseline_wrong and not mtd_wrong
        broken_here = not baseline_wrong and mtd_wrong
        repaired += int(repaired_here)
        broken += int(broken_here)
        baseline_disagreements += int(baseline_wrong)
        mtd_disagreements += int(mtd_wrong)
        dynamic_rows.append(
            {
                "video": name,
                "fp16": int(fp),
                "baseline": int(baseline),
                "mtd": int(mtd),
                "baseline_vs_fp16": "match" if not baseline_wrong else "mismatch",
                "mtd_vs_fp16": "match" if not mtd_wrong else "mismatch",
                "mtd_effect": (
                    "repair" if repaired_here else "break" if broken_here else "unchanged"
                ),
            }
        )

    continuous = {}
    for metric in ("motion_smoothness", "temporal_flickering", "subject_consistency"):
        common = set(per_video["fp16"][metric])
        common &= set(per_video["baseline"][metric])
        common &= set(per_video["mtd"][metric])
        if not common:
            raise RuntimeError(f"No common per-video records for {metric}")
        baseline_errors = [
            abs(float(per_video["baseline"][metric][name]) - float(per_video["fp16"][metric][name]))
            for name in common
        ]
        mtd_errors = [
            abs(float(per_video["mtd"][metric][name]) - float(per_video["fp16"][metric][name]))
            for name in common
        ]
        improved = sum(mtd < baseline for baseline, mtd in zip(baseline_errors, mtd_errors))
        worsened = sum(mtd > baseline for baseline, mtd in zip(baseline_errors, mtd_errors))
        continuous[metric] = {
            "video_count": len(common),
            "baseline_mean_absolute_error_to_fp16": mean(baseline_errors),
            "mtd_mean_absolute_error_to_fp16": mean(mtd_errors),
            "relative_mae_reduction": (
                mean(baseline_errors) - mean(mtd_errors)
            ) / max(mean(baseline_errors), 1.0e-12),
            "mtd_closer_video_count": improved,
            "mtd_farther_video_count": worsened,
            "tie_video_count": len(common) - improved - worsened,
        }

    summary = {
        "comparison_contract": {
            "fp16_is_reference_not_monotonic_target_for_dynamic_degree": True,
            "dynamic_degree_interpretation": (
                "Compare per-video agreement and absolute deviation from FP16; do not "
                "treat a larger aggregate Dynamic Degree as automatically better."
            ),
        },
        "aggregate_scores": aggregate,
        "aggregate_absolute_deviation_from_fp16": {
            variant: {
                metric: abs(aggregate[variant][metric] - aggregate["fp16"][metric])
                for metric in METRICS
            }
            for variant in ("baseline", "mtd")
        },
        "dynamic_degree": {
            "video_count": len(names),
            "baseline_disagreement_with_fp16": baseline_disagreements,
            "mtd_disagreement_with_fp16": mtd_disagreements,
            "mtd_repairs_baseline_error": repaired,
            "mtd_breaks_baseline_match": broken,
            "mcnemar_exact_two_sided_p": exact_two_sided_binomial(repaired, broken),
        },
        "per_video_distance_to_fp16": continuous,
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (args.output / "dynamic_degree_per_video.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dynamic_rows[0]))
        writer.writeheader()
        writer.writerows(dynamic_rows)
    (args.output / "inspection_manifest.txt").write_text(
        "\n".join(row["video"] for row in dynamic_rows if row["mtd_effect"] != "unchanged")
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
