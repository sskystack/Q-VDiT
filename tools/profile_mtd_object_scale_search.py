#!/usr/bin/env python3
"""Profile MTD correspondence search by foreground/background and object scale.

FP16 rendered frames provide semantic object masks through a COCO Mask R-CNN.
Those masks are pooled to the 16x16 MTD grid and used only for profiling.  The
test compares fixed local windows, dilated 3x3 supports, a shared sparse support,
foreground/background supports, and object-scale-specific supports under
leave-one-prompt-out validation.  No diffusion-stage specialization is used.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models.detection import (
    MaskRCNN_ResNet50_FPN_V2_Weights,
    maskrcnn_resnet50_fpn_v2,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--profile-dir", type=Path, required=True)
    parser.add_argument("--fp16-videos", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--feature-view", choices=("cond", "guided"), default="cond")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--search-radius", type=int, default=4)
    parser.add_argument("--score-threshold", type=float, default=0.35)
    parser.add_argument("--support-size", type=int, default=9)
    return parser.parse_args()


def safe_name(value):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def manifest_items(path):
    payload = json.loads(path.read_text())
    items = []
    for group in ("flip_1to0", "control_both_true"):
        for item in payload[group]:
            items.append({"group": group, **item})
    return sorted(items, key=lambda item: int(item["index"]))


def prompt_class(prompt, categories):
    lowered = prompt.lower()
    candidates = [
        category for category in categories
        if category != "__background__" and re.search(
            rf"\b{re.escape(category.lower())}\b", lowered
        )
    ]
    if not candidates:
        raise ValueError(f"No COCO class name found in prompt: {prompt}")
    return max(candidates, key=len)


def decode_video(path):
    capture = cv2.VideoCapture(str(path))
    frames = []
    while capture.isOpened():
        success, frame = capture.read()
        if not success:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from {path}")
    return frames


def detect_masks(model, transforms, frames, label_index, device, threshold):
    masks = []
    scores = []
    with torch.no_grad():
        for frame in frames:
            tensor = transforms(torch.from_numpy(frame).permute(2, 0, 1)).to(device)
            prediction = model([tensor])[0]
            correct = prediction["labels"] == label_index
            indices = torch.nonzero(correct, as_tuple=False).flatten()
            if len(indices) == 0:
                masks.append(torch.zeros(frame.shape[:2], dtype=torch.float32))
                scores.append(0.0)
                continue
            correct_scores = prediction["scores"].index_select(0, indices)
            best_position = int(correct_scores.argmax())
            selected = int(indices[best_position])
            score = float(prediction["scores"][selected])
            if score < threshold:
                masks.append(torch.zeros(frame.shape[:2], dtype=torch.float32))
                scores.append(score)
                continue
            masks.append(prediction["masks"][selected, 0].detach().float().cpu())
            scores.append(score)
    return torch.stack(masks), scores


def pool_masks(masks, size):
    return F.interpolate(
        masks[:, None], size=(size, size), mode="area"
    )[:, 0].clamp(0, 1)


def offset_table(radius, device):
    yy, xx = torch.meshgrid(
        torch.arange(-radius, radius + 1, device=device),
        torch.arange(-radius, radius + 1, device=device),
        indexing="ij",
    )
    return torch.stack([xx.flatten(), yy.flatten()], dim=-1)


def correspondence(feature, temperature, radius, device):
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
    entropy = -(
        probabilities.clamp_min(1.0e-12)
        * probabilities.clamp_min(1.0e-12).log()
    ).sum(-1)
    return probabilities, top1, entropy


def weighted_aggregate(probabilities, top1, entropy, weights, offsets, radius):
    weights = weights.to(probabilities.device, dtype=torch.float32).reshape(
        probabilities.shape[0], probabilities.shape[1]
    )
    total_weight = weights.sum().clamp_min(1.0e-12)
    candidate_count = probabilities.shape[-1]
    probability_mass = (probabilities * weights[..., None]).sum(dim=(0, 1))
    top1_count = torch.zeros(candidate_count, device=probabilities.device)
    top1_count.scatter_add_(0, top1.flatten(), weights.flatten())
    offset_tensor = torch.tensor(
        offsets, device=probabilities.device, dtype=torch.long
    )
    top_offsets = offset_tensor.index_select(0, top1.flatten()).reshape(
        *top1.shape, 2
    )
    centers = top_offsets.clamp(min=-(radius - 1), max=radius - 1)
    dynamic_support = (
        (offset_tensor[None, None, :, :] - centers[:, :, None, :])
        .abs().amax(dim=-1) <= 1
    )
    dynamic_mass = (probabilities * dynamic_support).sum(-1)
    center_magnitude = torch.linalg.vector_norm(centers.float(), dim=-1)
    return {
        "probability_mass": probability_mass.cpu(),
        "top1_count": top1_count.cpu(),
        "weight": float(total_weight),
        "mean_entropy": float((entropy * weights).sum() / total_weight),
        "mean_top1_probability": float(
            (probabilities.max(-1).values * weights).sum() / total_weight
        ),
        "teacher_centered_mass_weighted": float(
            (dynamic_mass * weights).sum()
        ),
        "teacher_center_magnitude_weighted": float(
            (center_magnitude * weights).sum()
        ),
    }


def add_aggregate(target, source):
    source_entropy_weighted = source.get(
        "entropy_weighted", source.get("mean_entropy", 0.0) * source["weight"]
    )
    source_top1_probability_weighted = source.get(
        "top1_probability_weighted",
        source.get("mean_top1_probability", 0.0) * source["weight"],
    )
    source_teacher_centered_mass_weighted = source.get(
        "teacher_centered_mass_weighted", 0.0
    )
    source_teacher_center_magnitude_weighted = source.get(
        "teacher_center_magnitude_weighted", 0.0
    )
    if target is None:
        return {
            "probability_mass": source["probability_mass"].clone(),
            "top1_count": source["top1_count"].clone(),
            "weight": source["weight"],
            "entropy_weighted": source_entropy_weighted,
            "top1_probability_weighted": source_top1_probability_weighted,
            "teacher_centered_mass_weighted": source_teacher_centered_mass_weighted,
            "teacher_center_magnitude_weighted": source_teacher_center_magnitude_weighted,
        }
    target["probability_mass"] += source["probability_mass"]
    target["top1_count"] += source["top1_count"]
    target["weight"] += source["weight"]
    target["entropy_weighted"] += source_entropy_weighted
    target["top1_probability_weighted"] += source_top1_probability_weighted
    target["teacher_centered_mass_weighted"] += source_teacher_centered_mass_weighted
    target["teacher_center_magnitude_weighted"] += source_teacher_center_magnitude_weighted
    return target


def support_metrics(aggregate, support):
    indices = torch.tensor(sorted(support), dtype=torch.long)
    denominator = max(aggregate["weight"], 1.0e-12)
    return {
        "captured_probability_mass": float(
            aggregate["probability_mass"].index_select(0, indices).sum() / denominator
        ),
        "top1_coverage": float(
            aggregate["top1_count"].index_select(0, indices).sum() / denominator
        ),
    }


def top_support(aggregate, size):
    return tuple(sorted(torch.topk(
        aggregate["probability_mass"], k=size
    ).indices.tolist()))


def offset_support(offsets, predicate):
    return tuple(
        index for index, offset in enumerate(offsets)
        if predicate(int(offset[0]), int(offset[1]))
    )


def dilation_support(offsets, dilation):
    values = {-dilation, 0, dilation}
    return offset_support(offsets, lambda x, y: x in values and y in values)


def pyramid_support(offsets, dilations):
    support = set()
    for dilation in dilations:
        support.update(dilation_support(offsets, dilation))
    return tuple(sorted(support))


def scale_group(area_fraction):
    if area_fraction < 0.10:
        return "small"
    if area_fraction < 0.25:
        return "medium"
    return "large"


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    categories = weights.meta["categories"]
    transforms = weights.transforms()
    detector = maskrcnn_resnet50_fpn_v2(weights=weights).to(device).eval()
    items = manifest_items(args.manifest)

    mask_metadata = []
    prompt_masks = {}
    for item_index, item in enumerate(items, start=1):
        prompt = item["prompt"]
        category = prompt_class(prompt, categories)
        frames = decode_video(args.fp16_videos / f"{prompt}.mp4")
        masks, scores = detect_masks(
            detector, transforms, frames, categories.index(category),
            device, args.score_threshold,
        )
        pooled = pool_masks(masks, 16)
        area = float(pooled.mean())
        prompt_masks[prompt] = pooled
        mask_metadata.append({
            "prompt": prompt,
            "prompt_index": item["index"],
            "category": category,
            "object_area_fraction": area,
            "scale_group": scale_group(area),
            "mean_detection_score": float(np.mean(scores)),
            "detected_frame_fraction": float(np.mean(np.asarray(scores) >= args.score_threshold)),
        })
        print(
            f"[mask {item_index}/{len(items)}] {prompt}: {category}, "
            f"area={area:.3f}, detected={mask_metadata[-1]['detected_frame_fraction']:.2f}",
            flush=True,
        )
    write_csv(args.output / "mask_metadata.csv", mask_metadata)

    offsets = offset_table(args.search_radius, device="cpu").tolist()
    feature_key = f"fp_{args.feature_view}_pooled"
    per_prompt = {}
    for item_index, item in enumerate(items, start=1):
        prompt = item["prompt"]
        profile_dir = args.profile_dir / f"{int(item['index']):03d}_{safe_name(prompt)}"
        files = sorted(profile_dir.glob("step_*.pt"))
        if not files:
            raise FileNotFoundError(profile_dir)
        foreground = None
        background = None
        pooled_mask = prompt_masks[prompt]
        for feature_path in files:
            payload = torch.load(feature_path, map_location="cpu")
            probabilities, top1, entropy = correspondence(
                payload[feature_key], args.temperature, args.search_radius, device
            )
            current_mask = pooled_mask[:-1].reshape(
                probabilities.shape[0], probabilities.shape[1]
            )
            fg = weighted_aggregate(
                probabilities, top1, entropy, current_mask.to(device),
                offsets, args.search_radius,
            )
            bg = weighted_aggregate(
                probabilities, top1, entropy, (1.0 - current_mask).to(device),
                offsets, args.search_radius,
            )
            foreground = add_aggregate(foreground, fg)
            background = add_aggregate(background, bg)
        metadata = next(row for row in mask_metadata if row["prompt"] == prompt)
        per_prompt[prompt] = {
            "foreground": foreground,
            "background": background,
            "scale_group": metadata["scale_group"],
        }
        print(f"[feature {item_index}/{len(items)}] {prompt}", flush=True)

    fixed = offset_support(offsets, lambda x, y: max(abs(x), abs(y)) <= 1)
    rows = []
    fitted = []
    prompts = sorted(per_prompt)
    for holdout in prompts:
        training = [prompt for prompt in prompts if prompt != holdout]
        shared = None
        region = {"foreground": None, "background": None}
        scales = {"small": None, "medium": None, "large": None}
        for prompt in training:
            for region_name in region:
                aggregate = per_prompt[prompt][region_name]
                shared = add_aggregate(shared, aggregate)
                region[region_name] = add_aggregate(region[region_name], aggregate)
            group = per_prompt[prompt]["scale_group"]
            scales[group] = add_aggregate(scales[group], per_prompt[prompt]["foreground"])
        shared_support = top_support(shared, args.support_size)
        region_support = {
            name: top_support(aggregate, args.support_size)
            for name, aggregate in region.items()
        }
        scale_support = {
            name: top_support(aggregate, args.support_size)
            for name, aggregate in scales.items() if aggregate is not None
        }
        scale_dilation = {}
        for name, aggregate in scales.items():
            if aggregate is None:
                continue
            candidates = []
            for dilation in range(1, args.search_radius + 1):
                support = dilation_support(offsets, dilation)
                candidates.append((
                    support_metrics(aggregate, support)["captured_probability_mass"],
                    -dilation, dilation, support,
                ))
            scale_dilation[name] = max(candidates)[2:]

        holdout_scale = per_prompt[holdout]["scale_group"]
        for region_name in ("foreground", "background"):
            aggregate = per_prompt[holdout][region_name]
            methods = {
                "fixed_3x3": fixed,
                "shared_sparse9": shared_support,
                f"{region_name}_sparse9": region_support[region_name],
                "pyramid_d1_d2": pyramid_support(offsets, (1, 2)),
                "pyramid_d1_d2_d4": pyramid_support(offsets, (1, 2, 4)),
            }
            if region_name == "foreground" and holdout_scale in scale_support:
                methods["scale_sparse9"] = scale_support[holdout_scale]
                methods["scale_dilated3x3"] = scale_dilation[holdout_scale][1]
                methods["scale_pyramid"] = pyramid_support(
                    offsets,
                    {
                        "small": (1,),
                        "medium": (1, 2),
                        "large": (1, 2, 4),
                    }[holdout_scale],
                )
            for radius in range(1, args.search_radius + 1):
                methods[f"dense_radius_{radius}"] = offset_support(
                    offsets, lambda x, y, radius=radius: max(abs(x), abs(y)) <= radius
                )
            for method, support in methods.items():
                values = support_metrics(aggregate, support)
                rows.append({
                    "holdout_prompt": holdout,
                    "scale_group": holdout_scale,
                    "region": region_name,
                    "method": method,
                    **values,
                    "candidate_count": len(support),
                    "probability_mass_per_candidate": (
                        values["captured_probability_mass"] / len(support)
                    ),
                    "lift_over_uniform": (
                        values["captured_probability_mass"]
                        / (len(support) / len(offsets))
                    ),
                    "mean_entropy": aggregate["entropy_weighted"] / aggregate["weight"],
                    "mean_top1_probability": aggregate["top1_probability_weighted"] / aggregate["weight"],
                })
            dynamic_mass = (
                aggregate["teacher_centered_mass_weighted"] / aggregate["weight"]
            )
            rows.append({
                "holdout_prompt": holdout,
                "scale_group": holdout_scale,
                "region": region_name,
                "method": "teacher_centered_3x3",
                "captured_probability_mass": dynamic_mass,
                "top1_coverage": 1.0,
                "candidate_count": 9,
                "probability_mass_per_candidate": dynamic_mass / 9,
                "lift_over_uniform": dynamic_mass / (9 / len(offsets)),
                "mean_entropy": aggregate["entropy_weighted"] / aggregate["weight"],
                "mean_top1_probability": aggregate["top1_probability_weighted"] / aggregate["weight"],
            })
            fitted.append({
                "holdout_prompt": holdout,
                "scale_group": holdout_scale,
                "foreground_weight": per_prompt[holdout]["foreground"]["weight"],
                "background_weight": per_prompt[holdout]["background"]["weight"],
                "selected_scale_dilation": (
                    scale_dilation[holdout_scale][0]
                    if holdout_scale in scale_dilation else None
                ),
            })
    write_csv(args.output / "leave_one_prompt_out.csv", rows)
    write_csv(args.output / "holdout_metadata.csv", fitted)

    summary = {"scale_counts": defaultdict(int), "groups": {}, "decisions": {}}
    for prompt in prompts:
        summary["scale_counts"][per_prompt[prompt]["scale_group"]] += 1
    summary["scale_counts"] = dict(summary["scale_counts"])
    for region_name in ("foreground", "background"):
        summary["groups"][region_name] = {}
        current = [row for row in rows if row["region"] == region_name]
        for method in sorted({row["method"] for row in current}):
            method_rows = [row for row in current if row["method"] == method]
            summary["groups"][region_name][method] = {
                "prompt_count": len(method_rows),
                "candidate_count": int(method_rows[0]["candidate_count"]),
                "captured_probability_mass": float(np.mean([
                    row["captured_probability_mass"] for row in method_rows
                ])),
                "top1_coverage": float(np.mean([
                    row["top1_coverage"] for row in method_rows
                ])),
                "probability_mass_per_candidate": float(np.mean([
                    row["probability_mass_per_candidate"] for row in method_rows
                ])),
                "lift_over_uniform": float(np.mean([
                    row["lift_over_uniform"] for row in method_rows
                ])),
            }
        summary["groups"][region_name]["matching_confidence"] = {
            "mean_entropy": float(np.mean([row["mean_entropy"] for row in current])),
            "mean_top1_probability": float(np.mean([
                row["mean_top1_probability"] for row in current
            ])),
            "maximum_entropy_radius4": math.log((2 * args.search_radius + 1) ** 2),
        }

    foreground = [row for row in rows if row["region"] == "foreground"]
    paired = defaultdict(dict)
    for row in foreground:
        paired[row["holdout_prompt"]][row["method"]] = row
    comparisons = {}
    for left, right in (
        ("scale_sparse9", "fixed_3x3"),
        ("scale_sparse9", "shared_sparse9"),
        ("foreground_sparse9", "fixed_3x3"),
        ("scale_dilated3x3", "fixed_3x3"),
        ("scale_pyramid", "fixed_3x3"),
        ("pyramid_d1_d2_d4", "fixed_3x3"),
        ("teacher_centered_3x3", "fixed_3x3"),
    ):
        values = [
            methods[left]["captured_probability_mass"]
            - methods[right]["captured_probability_mass"]
            for methods in paired.values() if left in methods and right in methods
        ]
        comparisons[f"{left}_minus_{right}"] = {
            "prompt_count": len(values),
            "mean_probability_mass_difference": float(np.mean(values)) if values else None,
            "positive_prompt_fraction": float(np.mean(np.asarray(values) > 0)) if values else None,
        }
    summary["decisions"] = {
        "comparisons": comparisons,
        "interpretation_rule": (
            "Object-scale search is supported only if scale-specific sparse or dilated "
            "supports improve held-out foreground mass over both fixed 3x3 and a shared "
            "sparse support. High entropy near log(81) indicates that feature identity, "
            "not search range, is the primary bottleneck."
        ),
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
