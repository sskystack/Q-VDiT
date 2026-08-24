#!/usr/bin/env python3
"""Cross-validated profiling for phase-aware sparse MTD correspondence supports.

The profiler consumes paired FP/quant feature captures produced by
``qdiff.mtd_feature_profiler``.  It fits nine-offset correspondence supports on
all but one prompt and evaluates the held-out prompt.  This distinguishes the
claim that a fixed 3x3 window is too restrictive from the stronger claim that
different diffusion phases need different correspondence directions.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--feature-view", choices=("cond", "guided"), default="cond")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--search-radius", type=int, default=4)
    parser.add_argument("--support-size", type=int, default=9)
    parser.add_argument(
        "--phase-groups",
        default="early:1,25;middle:50;late:75,100",
    )
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def parse_phases(raw):
    phases = {}
    progress_to_phase = {}
    for entry in raw.split(";"):
        name, values = entry.split(":", 1)
        points = [int(value) for value in values.split(",") if value.strip()]
        phases[name.strip()] = points
        for point in points:
            if point in progress_to_phase:
                raise ValueError(f"Progress {point} occurs in multiple phases")
            progress_to_phase[point] = name.strip()
    return phases, progress_to_phase


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def offset_table(radius, device):
    yy, xx = torch.meshgrid(
        torch.arange(-radius, radius + 1, device=device),
        torch.arange(-radius, radius + 1, device=device),
        indexing="ij",
    )
    return torch.stack([xx.flatten(), yy.flatten()], dim=-1)


def correspondence_aggregate(feature, temperature, radius, device):
    feature = feature.to(device=device, dtype=torch.float32)
    if feature.ndim == 5:
        feature = feature[0]
    channels, frames, height, width = feature.shape
    feature = F.normalize(feature, dim=0, eps=1.0e-6)
    current = feature[:, :-1].permute(1, 2, 3, 0).reshape(
        frames - 1, height * width, channels
    )
    following = feature[:, 1:].permute(1, 0, 2, 3)
    kernel = 2 * radius + 1
    neighbours = F.unfold(following, kernel_size=kernel, padding=radius)
    neighbours = neighbours.reshape(
        frames - 1, channels, kernel * kernel, height * width
    ).permute(0, 3, 2, 1)
    logits = (current[:, :, None, :] * neighbours).sum(-1) / temperature

    valid_grid = following.new_ones(frames - 1, 1, height, width)
    valid = F.unfold(valid_grid, kernel_size=kernel, padding=radius)
    valid = valid.reshape(frames - 1, kernel * kernel, height * width)
    valid = valid.permute(0, 2, 1).bool()
    logits = logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
    probabilities = torch.softmax(logits, dim=-1)
    top1 = probabilities.argmax(-1)
    candidate_count = kernel * kernel
    return {
        "probability_mass": probabilities.sum(dim=(0, 1)).cpu(),
        "top1_count": torch.bincount(
            top1.flatten(), minlength=candidate_count
        ).cpu().to(torch.float64),
        "token_count": int(top1.numel()),
    }


def add_aggregate(target, source):
    if target is None:
        return {
            "probability_mass": source["probability_mass"].clone(),
            "top1_count": source["top1_count"].clone(),
            "token_count": source["token_count"],
        }
    target["probability_mass"] += source["probability_mass"]
    target["top1_count"] += source["top1_count"]
    target["token_count"] += source["token_count"]
    return target


def select_top_support(aggregate, size):
    return tuple(
        sorted(
            torch.topk(aggregate["probability_mass"], k=size).indices.tolist()
        )
    )


def support_metrics(aggregate, support):
    indices = torch.tensor(support, dtype=torch.long)
    denominator = max(aggregate["token_count"], 1)
    return {
        "captured_probability_mass": float(
            aggregate["probability_mass"].index_select(0, indices).sum() / denominator
        ),
        "top1_coverage": float(
            aggregate["top1_count"].index_select(0, indices).sum() / denominator
        ),
    }


def support_for_offsets(offsets, wanted):
    lookup = {tuple(int(x) for x in offset): index for index, offset in enumerate(offsets)}
    return tuple(sorted(lookup[tuple(offset)] for offset in wanted))


def dilation_support(offsets, dilation):
    wanted = [
        (dx, dy)
        for dy in (-dilation, 0, dilation)
        for dx in (-dilation, 0, dilation)
    ]
    return support_for_offsets(offsets, wanted)


def support_offsets(support, offsets):
    return [
        [int(offsets[index][0]), int(offsets[index][1])]
        for index in support
    ]


def mean_ci(values, rng, iterations):
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return float("nan"), [float("nan"), float("nan")]
    samples = rng.choice(values, (iterations, len(values)), replace=True).mean(axis=1)
    return float(values.mean()), [
        float(np.quantile(samples, 0.025)),
        float(np.quantile(samples, 0.975)),
    ]


def jaccard(left, right):
    left, right = set(left), set(right)
    return len(left & right) / max(len(left | right), 1)


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    phases, progress_to_phase = parse_phases(args.phase_groups)
    device = torch.device(args.device)
    offsets = offset_table(args.search_radius, device="cpu").tolist()
    fixed_support = dilation_support(offsets, 1)

    records = []
    feature_key = f"fp_{args.feature_view}_pooled"
    for prompt_dir in sorted(args.profile_dir.iterdir()):
        if not prompt_dir.is_dir():
            continue
        for feature_path in sorted(prompt_dir.glob("step_*.pt")):
            payload = torch.load(feature_path, map_location="cpu")
            progress = int(payload["sampling_progress"])
            if progress not in progress_to_phase:
                continue
            aggregate = correspondence_aggregate(
                payload[feature_key], args.temperature, args.search_radius, device
            )
            records.append({
                "prompt": str(payload["prompt"]),
                "prompt_index": int(payload["prompt_index"]),
                "progress": progress,
                "phase": progress_to_phase[progress],
                "aggregate": aggregate,
            })
    if not records:
        raise RuntimeError(f"No feature records found under {args.profile_dir}")

    prompts = sorted({record["prompt"] for record in records})
    expected_points = set(progress_to_phase)
    for prompt in prompts:
        points = {record["progress"] for record in records if record["prompt"] == prompt}
        if points != expected_points:
            raise RuntimeError(
                f"Prompt {prompt!r} has progress {sorted(points)}, expected {sorted(expected_points)}"
            )

    evaluation_rows = []
    fitted_supports = defaultdict(list)
    for holdout in prompts:
        training = [record for record in records if record["prompt"] != holdout]
        shared_aggregate = None
        phase_aggregates = {phase: None for phase in phases}
        point_aggregates = {point: None for point in progress_to_phase}
        for record in training:
            shared_aggregate = add_aggregate(shared_aggregate, record["aggregate"])
            phase = record["phase"]
            phase_aggregates[phase] = add_aggregate(
                phase_aggregates[phase], record["aggregate"]
            )
            point = record["progress"]
            point_aggregates[point] = add_aggregate(
                point_aggregates[point], record["aggregate"]
            )

        shared_support = select_top_support(shared_aggregate, args.support_size)
        phase_supports = {
            phase: select_top_support(aggregate, args.support_size)
            for phase, aggregate in phase_aggregates.items()
        }
        point_supports = {
            point: select_top_support(aggregate, args.support_size)
            for point, aggregate in point_aggregates.items()
        }
        phase_dilations = {}
        for phase, aggregate in phase_aggregates.items():
            candidates = []
            for dilation in range(1, args.search_radius + 1):
                support = dilation_support(offsets, dilation)
                candidates.append((
                    support_metrics(aggregate, support)["captured_probability_mass"],
                    -dilation,
                    dilation,
                    support,
                ))
            _, _, dilation, support = max(candidates)
            phase_dilations[phase] = (dilation, support)
            fitted_supports[("phase_sparse", phase)].append(phase_supports[phase])
            fitted_supports[("phase_dilation", phase)].append(support)

        for record in records:
            if record["prompt"] != holdout:
                continue
            methods = {
                "fixed_3x3": fixed_support,
                "shared_sparse9": shared_support,
                "phase_sparse9": phase_supports[record["phase"]],
                "point_sparse9": point_supports[record["progress"]],
                "phase_dilated3x3": phase_dilations[record["phase"]][1],
            }
            for method, support in methods.items():
                metrics = support_metrics(record["aggregate"], support)
                evaluation_rows.append({
                    "holdout_prompt": holdout,
                    "progress": record["progress"],
                    "phase": record["phase"],
                    "method": method,
                    "selected_dilation": (
                        phase_dilations[record["phase"]][0]
                        if method == "phase_dilated3x3" else ""
                    ),
                    **metrics,
                    "support_offsets": json.dumps(support_offsets(support, offsets)),
                })

    write_csv(args.output / "leave_one_prompt_out.csv", evaluation_rows)

    rng = np.random.default_rng(args.seed)
    methods = sorted({row["method"] for row in evaluation_rows})
    prompt_method = defaultdict(list)
    for row in evaluation_rows:
        prompt_method[(row["holdout_prompt"], row["method"])].append(row)

    metric_summary = {}
    for method in methods:
        metric_summary[method] = {}
        for metric in ("captured_probability_mass", "top1_coverage"):
            values = [
                np.mean([row[metric] for row in prompt_method[(prompt, method)]])
                for prompt in prompts
            ]
            mean, ci = mean_ci(values, rng, args.bootstrap)
            metric_summary[method][metric] = {
                "prompt_mean": mean,
                "bootstrap_95ci": ci,
            }

    paired_differences = {}
    for metric in ("captured_probability_mass", "top1_coverage"):
        paired_differences[metric] = {}
        for left, right in (
            ("phase_sparse9", "fixed_3x3"),
            ("phase_sparse9", "shared_sparse9"),
            ("point_sparse9", "phase_sparse9"),
            ("phase_dilated3x3", "fixed_3x3"),
        ):
            values = []
            for prompt in prompts:
                left_value = np.mean([
                    row[metric] for row in prompt_method[(prompt, left)]
                ])
                right_value = np.mean([
                    row[metric] for row in prompt_method[(prompt, right)]
                ])
                values.append(left_value - right_value)
            mean, ci = mean_ci(values, rng, args.bootstrap)
            paired_differences[metric][f"{left}_minus_{right}"] = {
                "prompt_mean_difference": mean,
                "bootstrap_95ci": ci,
                "positive_prompt_fraction": float(np.mean(np.asarray(values) > 0)),
            }

    support_stability = {}
    for (method, phase), supports in sorted(fitted_supports.items()):
        similarities = [
            jaccard(supports[i], supports[j])
            for i in range(len(supports))
            for j in range(i + 1, len(supports))
        ]
        support_stability[f"{method}:{phase}"] = {
            "mean_leave_one_out_jaccard": float(np.mean(similarities)),
            "minimum_leave_one_out_jaccard": float(np.min(similarities)),
        }

    full_phase_aggregates = {phase: None for phase in phases}
    for record in records:
        full_phase_aggregates[record["phase"]] = add_aggregate(
            full_phase_aggregates[record["phase"]], record["aggregate"]
        )
    full_phase_supports = {
        phase: select_top_support(aggregate, args.support_size)
        for phase, aggregate in full_phase_aggregates.items()
    }
    phase_names = list(phases)
    cross_phase_jaccard = {}
    for index, left in enumerate(phase_names):
        for right in phase_names[index + 1:]:
            cross_phase_jaccard[f"{left}_vs_{right}"] = jaccard(
                full_phase_supports[left], full_phase_supports[right]
            )

    mass_gain_fixed = paired_differences["captured_probability_mass"][
        "phase_sparse9_minus_fixed_3x3"
    ]
    mass_gain_shared = paired_differences["captured_probability_mass"][
        "phase_sparse9_minus_shared_sparse9"
    ]
    mean_stability = float(np.mean([
        support_stability[f"phase_sparse:{phase}"]["mean_leave_one_out_jaccard"]
        for phase in phases
    ]))
    fixed_window_inadequate = (
        mass_gain_fixed["prompt_mean_difference"] >= 0.15
        and mass_gain_fixed["bootstrap_95ci"][0] > 0
    )
    phase_specific_supported = (
        fixed_window_inadequate
        and mass_gain_shared["prompt_mean_difference"] >= 0.02
        and mass_gain_shared["bootstrap_95ci"][0] > 0
        and mass_gain_shared["positive_prompt_fraction"] >= 0.70
        and mean_stability >= 0.60
    )
    if phase_specific_supported:
        outcome = "go_phase_specific_sparse_transport"
    elif fixed_window_inadequate:
        outcome = "revise_to_shared_sparse_transport"
    else:
        outcome = "kill_sparse_support_hypothesis"

    summary = {
        "experiment": {
            "prompt_count": len(prompts),
            "record_count": len(records),
            "feature_view": args.feature_view,
            "search_radius": args.search_radius,
            "support_size": args.support_size,
            "phases": phases,
            "validation": "leave-one-prompt-out",
        },
        "metric_summary": metric_summary,
        "paired_differences": paired_differences,
        "support_stability": support_stability,
        "full_data_phase_supports": {
            phase: support_offsets(support, offsets)
            for phase, support in full_phase_supports.items()
        },
        "cross_phase_support_jaccard": cross_phase_jaccard,
        "decision": {
            "outcome": outcome,
            "fixed_window_inadequate": fixed_window_inadequate,
            "phase_specific_supported": phase_specific_supported,
            "mean_leave_one_out_phase_support_jaccard": mean_stability,
            "thresholds": {
                "phase_minus_fixed_probability_mass": 0.15,
                "phase_minus_shared_probability_mass": 0.02,
                "phase_minus_shared_positive_prompt_fraction": 0.70,
                "minimum_support_stability": 0.60,
            },
            "interpretation": (
                "A gain over fixed 3x3 proves only that the local support is too small. "
                "A held-out gain over shared sparse9 is additionally required to support "
                "diffusion-phase-specific correspondence topology."
            ),
        },
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
