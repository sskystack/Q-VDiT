#!/usr/bin/env python3
"""Video-level statistics for the seed-42 MTD motion profiling experiment."""

import argparse
import csv
import itertools
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METRICS = (
    "outside_mass",
    "top1_outside_rate",
    "teacher_displacement_mean",
    "spearman_magnitude",
    "direction_cosine",
    "entropy_gap",
    "quant_teacher_correspondence_kl",
    "quant_teacher_displacement_mae",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap", type=int, default=20000)
    return parser.parse_args()


def bootstrap_mean(values, rng, iterations):
    values = np.asarray(values, dtype=np.float64)
    samples = rng.choice(values, (iterations, len(values)), replace=True).mean(axis=1)
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def bootstrap_difference(left, right, rng, iterations):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    samples = (
        rng.choice(left, (iterations, len(left)), replace=True).mean(axis=1)
        - rng.choice(right, (iterations, len(right)), replace=True).mean(axis=1)
    )
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def exact_permutation_p(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    pooled = np.concatenate([left, right])
    observed = abs(float(left.mean() - right.mean()))
    exceed = 0
    total = 0
    for selected in itertools.combinations(range(len(pooled)), len(left)):
        mask = np.zeros(len(pooled), dtype=bool)
        mask[list(selected)] = True
        difference = abs(float(pooled[mask].mean() - pooled[~mask].mean()))
        exceed += difference >= observed - 1.0e-15
        total += 1
    return float(exceed / total)


def uniform_outside_probability(height=16, width=16, radius=4):
    probabilities = []
    for y in range(height):
        for x in range(width):
            valid = []
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    if 0 <= y + dy < height and 0 <= x + dx < width:
                        valid.append(max(abs(dx), abs(dy)) > 1)
            probabilities.append(sum(valid) / len(valid))
    return float(np.mean(probabilities))


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with args.input.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in list(row):
            if key not in {"group", "prompt"}:
                try:
                    row[key] = float(row[key])
                except ValueError:
                    pass
        row["entropy_gap"] = row["entropy_high_flow"] - row["entropy_low_flow"]

    prompt_rows = []
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["group"], row["prompt"])].append(row)
    for (group, prompt), values in grouped.items():
        item = {"group": group, "prompt": prompt}
        for metric in METRICS:
            item[metric] = float(np.mean([row[metric] for row in values]))
        prompt_rows.append(item)

    chance = uniform_outside_probability()
    rng = np.random.default_rng(args.seed)
    result = {
        "independent_unit": "video/prompt (five diffusion steps averaged first)",
        "videos_per_group": 8,
        "uniform_9x9_outside_3x3_probability": chance,
        "metrics": {},
        "per_step_group_means": {},
    }
    for metric in METRICS:
        left = [row[metric] for row in prompt_rows if row["group"] == "flip_1to0"]
        right = [row[metric] for row in prompt_rows if row["group"] == "control_both_true"]
        result["metrics"][metric] = {
            "flip_mean": float(np.mean(left)),
            "flip_bootstrap_95ci": bootstrap_mean(left, rng, args.bootstrap),
            "control_mean": float(np.mean(right)),
            "control_bootstrap_95ci": bootstrap_mean(right, rng, args.bootstrap),
            "flip_minus_control": float(np.mean(left) - np.mean(right)),
            "difference_bootstrap_95ci": bootstrap_difference(
                left, right, rng, args.bootstrap
            ),
            "exact_two_sided_permutation_p": exact_permutation_p(left, right),
        }

    for progress in sorted({int(row["sampling_progress"]) for row in rows}):
        result["per_step_group_means"][str(progress)] = {}
        for group in ("flip_1to0", "control_both_true"):
            subset = [
                row for row in rows
                if row["group"] == group and int(row["sampling_progress"]) == progress
            ]
            result["per_step_group_means"][str(progress)][group] = {
                metric: float(np.mean([row[metric] for row in subset]))
                for metric in METRICS
            }

    (args.output / "video_level_statistics.json").write_text(
        json.dumps(result, indent=2)
    )
    with (args.output / "prompt_level_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(prompt_rows[0]))
        writer.writeheader()
        writer.writerows(sorted(prompt_rows, key=lambda row: (row["group"], row["prompt"])))

    colors = {"flip_1to0": "#d95f02", "control_both_true": "#1b9e77"}
    groups = list(colors)
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.0))
    x = np.arange(2)
    outside_means = [
        np.mean([row["outside_mass"] for row in prompt_rows if row["group"] == group])
        for group in groups
    ]
    axes[0].bar(x, outside_means, color=[colors[group] for group in groups], width=0.6)
    axes[0].axhline(chance, color="black", linestyle="--", label="uniform 9x9")
    axes[0].set_xticks(x, ["flip", "control"])
    axes[0].set_ylabel("Mass outside original 3x3")
    axes[0].set_ylim(chance - 0.01, chance + 0.01)
    axes[0].legend(frameon=False)

    for offset, group in enumerate(groups):
        values = [row["spearman_magnitude"] for row in prompt_rows if row["group"] == group]
        axes[1].scatter(np.full(len(values), offset), values, color=colors[group], alpha=0.85)
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set_xticks(x, ["flip", "control"])
    axes[1].set_ylabel("Feature displacement vs RAFT (Spearman)")

    for offset, group in enumerate(groups):
        values = [row["entropy_gap"] for row in prompt_rows if row["group"] == group]
        axes[2].scatter(np.full(len(values), offset), values, color=colors[group], alpha=0.85)
    axes[2].axhline(0, color="black", linewidth=1)
    axes[2].set_xticks(x, ["flip", "control"])
    axes[2].set_ylabel("Entropy(high-flow) - Entropy(low-flow)")
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.output / "04_video_level_diagnostics.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
