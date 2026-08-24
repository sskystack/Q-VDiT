#!/usr/bin/env python3
"""Compare flow-free adaptive-MTD held-out captures across checkpoints."""

import argparse
import csv
import json
from pathlib import Path

import torch


SCHEME = "teacher_centered_3x3"
VIEWS = {
    "cond": ("fp_cond_pooled", "quant_cond_pooled"),
    "guided": ("fp_guided_pooled", "quant_guided_pooled"),
}
METRICS = ("local_kl", "motion_error")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--random", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-improvement", type=float, default=0.05)
    return parser.parse_args()


def read_json(path):
    return json.loads(path.read_text())


def read_rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def capture_key(row):
    return int(row["prompt_index"]), int(row["sampling_progress"])


def improvement(baseline, value):
    return (baseline - value) / baseline


def compare_analysis(baseline_root, other_root, minimum):
    result = {"views": {}, "all_pooled_gates_pass": True}
    for view in VIEWS:
        baseline_dir = baseline_root / "analysis" / view
        other_dir = other_root / "analysis" / view
        baseline_summary = read_json(baseline_dir / "summary.json")
        other_summary = read_json(other_dir / "summary.json")
        baseline_rows = {capture_key(row): row for row in read_rows(baseline_dir / "per_capture.csv")}
        other_rows = {capture_key(row): row for row in read_rows(other_dir / "per_capture.csv")}
        if baseline_rows.keys() != other_rows.keys():
            raise RuntimeError(f"capture mismatch for {view}")

        pooled = {}
        for metric in METRICS:
            baseline_value = float(baseline_summary["schemes"][SCHEME][metric])
            other_value = float(other_summary["schemes"][SCHEME][metric])
            gain = improvement(baseline_value, other_value)
            pooled[metric] = {
                "baseline": baseline_value,
                "value": other_value,
                "relative_improvement": gain,
                "passes": gain >= minimum,
            }
            result["all_pooled_gates_pass"] &= gain >= minimum

        captures = []
        prompt_values = {}
        for key in sorted(baseline_rows):
            baseline_row = baseline_rows[key]
            other_row = other_rows[key]
            row = {"prompt_index": key[0], "sampling_progress": key[1]}
            for metric in METRICS:
                field = f"{SCHEME}_{metric}"
                baseline_value = float(baseline_row[field])
                other_value = float(other_row[field])
                gain = improvement(baseline_value, other_value)
                row[metric] = {
                    "baseline": baseline_value,
                    "value": other_value,
                    "relative_improvement": gain,
                }
                prompt_values.setdefault(key[0], {}).setdefault(metric, []).append(
                    (baseline_value, other_value)
                )
            captures.append(row)

        prompts = {}
        for prompt_index, values in sorted(prompt_values.items()):
            prompts[str(prompt_index)] = {}
            for metric, pairs in values.items():
                baseline_value = sum(pair[0] for pair in pairs) / len(pairs)
                other_value = sum(pair[1] for pair in pairs) / len(pairs)
                prompts[str(prompt_index)][metric] = {
                    "baseline": baseline_value,
                    "value": other_value,
                    "relative_improvement": improvement(baseline_value, other_value),
                }
        result["views"][view] = {"pooled": pooled, "prompts": prompts, "captures": captures}
    return result


def feature_files(root):
    values = {}
    for path in sorted((root / "features").rglob("step_*.pt")):
        payload = torch.load(path, map_location="cpu")
        key = int(payload["prompt_index"]), int(payload["sampling_progress"])
        values[key] = payload
    return values


def compare_fp_references(baseline_root, other_root):
    baseline = feature_files(baseline_root)
    other = feature_files(other_root)
    if baseline.keys() != other.keys():
        raise RuntimeError("feature capture keys differ")
    result = {"all_exact": True, "max_abs_difference": 0.0, "captures": []}
    for key in sorted(baseline):
        row = {"prompt_index": key[0], "sampling_progress": key[1], "views": {}}
        for view, (teacher_key, _) in VIEWS.items():
            difference = baseline[key][teacher_key].float() - other[key][teacher_key].float()
            maximum = float(difference.abs().max())
            exact = bool(torch.equal(baseline[key][teacher_key], other[key][teacher_key]))
            row["views"][view] = {"exact": exact, "max_abs_difference": maximum}
            result["all_exact"] &= exact
            result["max_abs_difference"] = max(result["max_abs_difference"], maximum)
        result["captures"].append(row)
    return result


def main():
    args = parse_args()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    candidate = compare_analysis(args.baseline, args.candidate, args.minimum_improvement)
    candidate_fp = compare_fp_references(args.baseline, args.candidate)
    result = {
        "scheme": SCHEME,
        "minimum_improvement": args.minimum_improvement,
        "baseline": str(args.baseline.resolve()),
        "candidate": str(args.candidate.resolve()),
        "candidate_vs_baseline": candidate,
        "candidate_fp_reference_consistency": candidate_fp,
        "run_random_control": bool(candidate["all_pooled_gates_pass"] and candidate_fp["all_exact"]),
    }
    if args.random:
        random_result = compare_analysis(args.baseline, args.random, args.minimum_improvement)
        random_fp = compare_fp_references(args.baseline, args.random)
        result["random"] = str(args.random.resolve())
        result["random_vs_baseline"] = random_result
        result["random_fp_reference_consistency"] = random_fp
        result["candidate_localization_confirmed"] = bool(
            result["run_random_control"]
            and random_fp["all_exact"]
            and not random_result["all_pooled_gates_pass"]
        )
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
