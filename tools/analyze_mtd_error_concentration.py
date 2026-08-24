#!/usr/bin/env python3
"""Measure whether per-position MTD error is spatially concentrated."""

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qdiff.mtd import _local_transport_distribution


COMPONENTS = ("local", "motion", "total")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--views", nargs="+", default=("cond", "guided"))
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--transport-size", type=int, default=16)
    parser.add_argument("--ratios", nargs="+", type=float, default=(0.10, 0.25, 0.50))
    parser.add_argument("--block-size", type=int, default=4)
    parser.add_argument("--block-ratio", type=float, default=0.25)
    parser.add_argument("--local-weight", type=float, default=1.0)
    parser.add_argument("--motion-weight", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--top25-threshold", type=float, default=0.50)
    parser.add_argument("--block-top25-threshold", type=float, default=0.40)
    return parser.parse_args()


def prompt_groups(manifest_path):
    manifest = json.loads(manifest_path.read_text())
    groups = {}
    for group in ("flip_1to0", "control_both_true"):
        for item in manifest[group]:
            groups[item["prompt"]] = group
    return manifest, groups


def ratio_key(ratio):
    return f"top{int(round(100 * ratio))}_coverage"


def top_coverage(error, ratio):
    count = max(1, math.ceil(error.shape[-1] * ratio))
    denominator = error.sum(dim=-1)
    numerator = torch.topk(error, count, dim=-1).values.sum(dim=-1)
    return torch.where(
        denominator > 1.0e-20,
        numerator / denominator,
        torch.full_like(denominator, float("nan")),
    )


def block_top_coverage(error, size, block_size, block_ratio):
    if size % block_size:
        raise ValueError(f"transport size {size} is not divisible by block size {block_size}")
    error = error.reshape(-1, size, size)
    selected = error.new_zeros(error.shape[0])
    count = max(1, math.ceil(block_size * block_size * block_ratio))
    for y in range(0, size, block_size):
        for x in range(0, size, block_size):
            block = error[:, y : y + block_size, x : x + block_size].flatten(1)
            selected += torch.topk(block, count, dim=-1).values.sum(dim=-1)
    denominator = error.flatten(1).sum(dim=-1)
    return torch.where(
        denominator > 1.0e-20,
        selected / denominator,
        torch.full_like(denominator, float("nan")),
    )


def per_position_errors(pred, target, size, temperature, local_weight, motion_weight):
    pred_probs, pred_current, pred_neighbours = _local_transport_distribution(
        pred, size, temperature
    )
    target_probs, target_current, target_neighbours = _local_transport_distribution(
        target, size, temperature
    )
    local = F.kl_div(
        pred_probs.clamp_min(1.0e-8).log(), target_probs, reduction="none"
    ).sum(dim=-1)
    pred_transport = (pred_probs[..., None] * pred_neighbours).sum(-2) - pred_current
    target_transport = (
        target_probs[..., None] * target_neighbours
    ).sum(-2) - target_current
    motion = F.smooth_l1_loss(
        pred_transport, target_transport, reduction="none"
    ).mean(dim=-1)
    return {
        "local": local_weight * local,
        "motion": motion_weight * motion,
        "total": local_weight * local + motion_weight * motion,
    }


def bootstrap_ci(values, rng, iterations):
    values = np.asarray(values, dtype=np.float64)
    samples = rng.choice(values, size=(iterations, len(values)), replace=True).mean(axis=1)
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def aggregate(rows, args):
    metric_names = [ratio_key(ratio) for ratio in args.ratios]
    metric_names.append("block_top_coverage")
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["view"], row["component"], row["group"], row["prompt"])].append(row)

    prompt_rows = []
    for (view, component, group, prompt), values in grouped.items():
        item = {
            "view": view,
            "component": component,
            "group": group,
            "prompt": prompt,
        }
        for metric in metric_names:
            item[metric] = float(np.mean([value[metric] for value in values]))
        prompt_rows.append(item)

    rng = np.random.default_rng(args.seed)
    summary = {}
    for view in args.views:
        summary[view] = {}
        for component in COMPONENTS:
            summary[view][component] = {}
            for group in ("all", "flip_1to0", "control_both_true"):
                subset = [
                    row for row in prompt_rows
                    if row["view"] == view
                    and row["component"] == component
                    and (group == "all" or row["group"] == group)
                ]
                result = {"prompt_count": len(subset)}
                for metric in metric_names:
                    values = [row[metric] for row in subset]
                    result[metric] = {
                        "mean": float(np.mean(values)),
                        "median": float(np.median(values)),
                        "bootstrap_95ci": bootstrap_ci(values, rng, args.bootstrap),
                        "minimum": float(np.min(values)),
                        "maximum": float(np.max(values)),
                    }
                summary[view][component][group] = result
    return prompt_rows, summary


def decision(summary, args):
    primary = summary["cond"]["total"]["all"]
    top25 = primary["top25_coverage"]["bootstrap_95ci"]
    block = primary["block_top_coverage"]["bootstrap_95ci"]
    if top25[0] >= args.top25_threshold and block[0] >= args.block_top25_threshold:
        outcome = "go"
    elif top25[1] < args.top25_threshold or block[1] < args.block_top25_threshold:
        outcome = "kill"
    else:
        outcome = "revise"
    return {
        "outcome": outcome,
        "primary_view": "cond",
        "primary_component": "total",
        "rule": (
            "go when both 95% CI lower bounds pass; kill when either 95% CI "
            "upper bound fails; otherwise revise"
        ),
        "thresholds": {
            "top25_coverage": args.top25_threshold,
            "block_top25_coverage": args.block_top25_threshold,
        },
        "observed_95ci": {
            "top25_coverage": top25,
            "block_top25_coverage": block,
        },
    }


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    if "cond" not in args.views:
        raise ValueError("the frozen decision contract requires the cond view")
    if 0.25 not in args.ratios:
        raise ValueError("the frozen decision contract requires ratio 0.25")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest, groups = prompt_groups(args.manifest)
    device = torch.device(args.device)

    rows = []
    files = sorted(args.features_root.glob("*/step_*.pt"))
    if not files:
        raise FileNotFoundError(f"no step_*.pt files under {args.features_root}")
    with torch.inference_mode():
        for index, path in enumerate(files, start=1):
            payload = torch.load(path, map_location="cpu")
            prompt = payload["prompt"]
            if prompt not in groups:
                raise KeyError(f"prompt absent from manifest: {prompt}")
            print(f"[{index:03d}/{len(files):03d}] {path.parent.name}/{path.name}", flush=True)
            for view in args.views:
                pred = payload[f"quant_{view}"].float().to(device)
                target = payload[f"fp_{view}"].float().to(device)
                errors = per_position_errors(
                    pred,
                    target,
                    args.transport_size,
                    args.temperature,
                    args.local_weight,
                    args.motion_weight,
                )
                for component, error in errors.items():
                    metrics = {
                        ratio_key(ratio): top_coverage(error, ratio).cpu().numpy()
                        for ratio in args.ratios
                    }
                    metrics["block_top_coverage"] = block_top_coverage(
                        error,
                        args.transport_size,
                        args.block_size,
                        args.block_ratio,
                    ).cpu().numpy()
                    totals = error.sum(dim=-1).cpu().numpy()
                    for frame_pair in range(error.shape[0]):
                        row = {
                            "view": view,
                            "component": component,
                            "group": groups[prompt],
                            "prompt": prompt,
                            "sampling_progress": int(payload["sampling_progress"]),
                            "model_timestep": int(payload["model_timestep"]),
                            "frame_pair": frame_pair,
                            "total_error": float(totals[frame_pair]),
                        }
                        for metric, values in metrics.items():
                            row[metric] = float(values[frame_pair])
                        rows.append(row)

    prompt_rows, summary = aggregate(rows, args)
    result = {
        "experiment": {
            "hypothesis": "per-position MTD error is concentrated enough to justify key-region refinement",
            "independent_unit": "prompt/video after averaging five diffusion steps and frame pairs",
            "feature_files": len(files),
            "prompt_count": len({row["prompt"] for row in rows}),
            "views": list(args.views),
            "transport_size": args.transport_size,
            "temperature": args.temperature,
            "ratios": list(args.ratios),
            "block_selection": {
                "block_size": args.block_size,
                "ratio_per_block": args.block_ratio,
            },
            "component_weights": {
                "local": args.local_weight,
                "motion": args.motion_weight,
            },
            "device": str(device),
            "cuda_device_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
            "seed": args.seed,
            "bootstrap_iterations": args.bootstrap,
            "manifest_seed": manifest["seed"],
        },
        "uniform_error_baselines": {
            ratio_key(ratio): ratio for ratio in args.ratios
        }
        | {"block_top_coverage": args.block_ratio},
        "summary": summary,
    }
    result["decision"] = decision(summary, args)
    write_csv(args.output / "frame_pair_metrics.csv", rows)
    write_csv(args.output / "prompt_metrics.csv", prompt_rows)
    (args.output / "summary.json").write_text(json.dumps(result, indent=2))

    primary = summary["cond"]["total"]["all"]
    lines = [
        "MTD error concentration diagnostic",
        "",
        f"decision: {result['decision']['outcome']}",
        f"prompts: {result['experiment']['prompt_count']}",
    ]
    for metric in ("top10_coverage", "top25_coverage", "top50_coverage", "block_top_coverage"):
        item = primary[metric]
        lines.append(
            f"{metric}: mean={item['mean']:.6f}, "
            f"95% CI=[{item['bootstrap_95ci'][0]:.6f}, {item['bootstrap_95ci'][1]:.6f}]"
        )
    (args.output / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
