#!/usr/bin/env python3
"""Summarize checkpoint variation and common held-out trajectory quality."""

import argparse
import csv
import json
import math
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


FAMILIES = (
    "weight_delta",
    "lora_effective",
    "lora_effective_out",
    "loraA",
    "loraB",
    "loraA_out",
    "loraB_out",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--heldout-metrics", nargs="+", required=True)
    parser.add_argument("--pairwise-aggregate", required=True)
    parser.add_argument("--variance-aggregate")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--title", default="Calibration stability")
    return parser.parse_args()


def read_csv(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    if len(args.labels) != len(args.heldout_metrics):
        raise ValueError("--labels and --heldout-metrics must have equal length")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    heldout = []
    final_latents = {}
    for label, path in zip(args.labels, args.heldout_metrics):
        metrics_path = Path(path)
        row = json.loads(metrics_path.read_text())
        heldout.append({"label": label, **row})
        latent_path = metrics_path.with_name("final_latents.pt")
        payload = torch.load(latent_path, map_location="cpu")
        final_latents[label] = {
            "fp": payload["fp_final_latent"].double(),
            "quant": payload["quant_final_latent"].double(),
        }
    write_csv(output / "heldout_metrics.csv", heldout)

    reference_label = args.labels[0]
    reference_fp = final_latents[reference_label]["fp"]
    fp_consistency = []
    for label in args.labels:
        value = final_latents[label]["fp"]
        difference = value - reference_fp
        fp_consistency.append({
            "reference_label": reference_label,
            "label": label,
            "exact_equal": bool(torch.equal(value, reference_fp)),
            "max_abs_difference": float(difference.abs().max()),
            "relative_l2": float(
                torch.linalg.vector_norm(difference)
                / torch.linalg.vector_norm(reference_fp).clamp_min(1e-30)
            ),
        })
    write_csv(output / "heldout_fp_reference_consistency.csv", fp_consistency)

    quant_pairwise = []
    for label_a, label_b in combinations(args.labels, 2):
        value_a = final_latents[label_a]["quant"]
        value_b = final_latents[label_b]["quant"]
        difference = value_a - value_b
        denominator = (
            torch.linalg.vector_norm(value_a) + torch.linalg.vector_norm(value_b)
        ).clamp_min(1e-30)
        quant_pairwise.append({
            "checkpoint_a": label_a,
            "checkpoint_b": label_b,
            "symmetric_relative_l2": float(2 * torch.linalg.vector_norm(difference) / denominator),
            "rmse": float(torch.sqrt(difference.square().mean())),
            "max_abs_difference": float(difference.abs().max()),
        })
    write_csv(output / "heldout_quant_output_pairwise.csv", quant_pairwise)

    metric_names = ("relative_l2", "nmse", "cosine", "rmse")
    variation = {}
    for metric in metric_names:
        values = np.asarray([float(row[metric]) for row in heldout], dtype=np.float64)
        variation[metric] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
            "min": float(values.min()),
            "max": float(values.max()),
            "range": float(values.max() - values.min()),
            "coefficient_of_variation": float(values.std(ddof=1) / abs(values.mean()))
            if values.size > 1 and abs(values.mean()) > 1e-30 else 0.0,
        }

    pairwise = read_csv(args.pairwise_aggregate)
    selected = [
        row for row in pairwise
        if row["family"] in FAMILIES and row["region"] == "all"
    ]
    pair_labels = sorted({f'{row["checkpoint_a"]} vs {row["checkpoint_b"]}' for row in selected})
    lookup = {
        (f'{row["checkpoint_a"]} vs {row["checkpoint_b"]}', row["family"]): float(row["symmetric_relative_l2"])
        for row in selected
    }
    fig, axis = plt.subplots(figsize=(10.2, 5.2))
    x = np.arange(len(FAMILIES), dtype=float)
    width = 0.8 / max(1, len(pair_labels))
    for index, pair in enumerate(pair_labels):
        values = [lookup.get((pair, family), np.nan) for family in FAMILIES]
        axis.bar(x - 0.4 + width / 2 + index * width, values, width, label=pair)
    axis.set_xticks(x, FAMILIES)
    axis.set_ylabel("Symmetric relative L2 between checkpoints")
    axis.set_title(f"{args.title}: learned quantization parameter variation")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "checkpoint_parameter_variation.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2))
    labels = [row["label"] for row in heldout]
    for axis, metric in zip(axes, ("relative_l2", "nmse", "cosine")):
        values = [float(row[metric]) for row in heldout]
        axis.bar(labels, values, color="#4C78A8")
        axis.set_title(metric)
        axis.tick_params(axis="x", rotation=25)
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle(f"{args.title}: common held-out prompt trajectory")
    fig.tight_layout()
    fig.savefig(output / "heldout_trajectory_comparison.png", dpi=220)
    plt.close(fig)

    family_variation = {}
    for family in FAMILIES:
        values = [float(row["symmetric_relative_l2"]) for row in selected if row["family"] == family]
        if values:
            family_variation[family] = {
                "mean_pairwise_symmetric_relative_l2": float(np.mean(values)),
                "max_pairwise_symmetric_relative_l2": float(np.max(values)),
            }
    direct_variance = {}
    if args.variance_aggregate:
        variance_rows = read_csv(args.variance_aggregate)
        selected_variance = [
            row for row in variance_rows
            if row["family"] in FAMILIES and row["region"] == "all"
        ]
        direct_variance = {
            row["family"]: {
                "relative_rms_std": float(row["relative_rms_std"]),
                "population_mean_element_variance": float(row["population_mean_element_variance"]),
                "mean_element_second_moment": float(row["mean_element_second_moment"]),
            }
            for row in selected_variance
        }
        if selected_variance:
            fig, axis = plt.subplots(figsize=(8.8, 4.8))
            axis.bar(
                [row["family"] for row in selected_variance],
                [float(row["relative_rms_std"]) for row in selected_variance],
                color="#F58518",
            )
            axis.set_ylabel("Population relative RMS std across checkpoints")
            axis.set_title(f"{args.title}: direct calibration-set variance")
            axis.grid(axis="y", alpha=0.25)
            fig.tight_layout()
            fig.savefig(output / "checkpoint_direct_variance.png", dpi=220)
            plt.close(fig)
    summary = {
        "labels": args.labels,
        "heldout_metric_variation": variation,
        "checkpoint_family_variation": family_variation,
        "checkpoint_family_direct_variance": direct_variance,
        "heldout_fp_reference_consistency": fp_consistency,
        "heldout_fp_references_exactly_identical": all(
            row["exact_equal"] for row in fp_consistency
        ),
        "heldout_quant_output_pairwise": quant_pairwise,
        "all_values_finite": all(
            math.isfinite(float(row[metric])) for row in heldout for metric in metric_names
        )
        and all(math.isfinite(float(row["symmetric_relative_l2"])) for row in selected)
        and all(
            math.isfinite(float(row[key]))
            for row in fp_consistency
            for key in ("max_abs_difference", "relative_l2")
        )
        and all(
            math.isfinite(float(row[key]))
            for row in quant_pairwise
            for key in ("symmetric_relative_l2", "rmse", "max_abs_difference")
        ),
    }
    (output / "calibration_outcome_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
