#!/usr/bin/env python3
"""Decompose video optical flow into robust background and foreground motion.

RAFT estimates dense adjacent-frame flow.  A RANSAC affine model fitted to a
regularly sampled flow grid represents the dominant background/camera motion;
inlier flow measures background speed while outlier residual measures local
foreground motion.  The method is segmentation-free and reports fit quality so
that failed background dominance is visible instead of silently trusted.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from argparse import Namespace
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from vbench.third_party.RAFT.core.raft import RAFT
from vbench.third_party.RAFT.core.utils_core.utils import InputPadder


DEFAULT_RAFT = Path(
    "/home/zhouchongtian/quantization/models/vbench/raft_model/models/raft-things.pth"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant", action="append", required=True,
        help="NAME=/absolute/path/to/video/directory; repeat for each variant",
    )
    parser.add_argument("--prompts", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raft-model", type=Path, default=DEFAULT_RAFT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--grid-stride", type=int, default=8)
    parser.add_argument("--ransac-threshold", type=float, default=1.5)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def parse_variants(values):
    variants = {}
    for value in values:
        name, root = value.split("=", 1)
        variants[name] = Path(root)
    if "fp16" not in variants:
        raise ValueError("One --variant must be named fp16")
    return variants


def safe_name(value):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_raft(model_path, device):
    args = Namespace(
        model=str(model_path), small=False,
        mixed_precision=False, alternate_corr=False,
    )
    model = RAFT(args)
    checkpoint = torch.load(model_path, map_location="cpu")
    checkpoint = {
        key.replace("module.", ""): value for key, value in checkpoint.items()
    }
    model.load_state_dict(checkpoint)
    return model.to(device).eval()


def load_frames(video_path, device):
    capture = cv2.VideoCapture(str(video_path))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames = []
    while capture.isOpened():
        success, frame = capture.read()
        if not success:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(
            torch.from_numpy(frame).permute(2, 0, 1).float()[None].to(device)
        )
    capture.release()
    if len(frames) < 2:
        raise RuntimeError(f"Fewer than two frames decoded from {video_path}")
    return frames, fps


def raft_flows(model, video_path, device):
    frames, fps = load_frames(video_path, device)
    flows = []
    with torch.no_grad():
        for first, second in zip(frames[:-1], frames[1:]):
            padder = InputPadder(first.shape)
            padded_first, padded_second = padder.pad(first, second)
            _, flow = model(
                padded_first, padded_second, iters=20, test_mode=True
            )
            flows.append(
                flow[0, :, :first.shape[-2], :first.shape[-1]].float().cpu().numpy()
            )
    return flows, fps


def robust_affine_metrics(flow, stride, threshold):
    _, height, width = flow.shape
    yy, xx = np.mgrid[0:height:stride, 0:width:stride]
    source = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=-1).astype(np.float32)
    sampled_flow = flow[:, yy, xx].transpose(1, 2, 0).reshape(-1, 2).astype(np.float32)
    target = source + sampled_flow
    affine, inliers = cv2.estimateAffine2D(
        source, target, method=cv2.RANSAC,
        ransacReprojThreshold=threshold, maxIters=5000,
        confidence=0.995, refineIters=20,
    )
    if affine is None or inliers is None or int(inliers.sum()) < 12:
        median_flow = np.median(sampled_flow, axis=0)
        affine = np.array([
            [1.0, 0.0, median_flow[0]],
            [0.0, 1.0, median_flow[1]],
        ], dtype=np.float64)
        residual = np.linalg.norm(sampled_flow - median_flow[None], axis=1)
        robust_scale = 1.4826 * np.median(np.abs(residual - np.median(residual)))
        cutoff = max(threshold, np.median(residual) + 2.5 * robust_scale)
        inlier_mask = residual <= cutoff
        fallback = True
    else:
        inlier_mask = inliers.reshape(-1).astype(bool)
        fallback = False

    homogeneous = np.concatenate([
        source.astype(np.float64), np.ones((len(source), 1), dtype=np.float64)
    ], axis=1)
    predicted_target = homogeneous @ affine.T
    predicted_flow = predicted_target - source
    residual_vector = sampled_flow - predicted_flow
    residual_magnitude = np.linalg.norm(residual_vector, axis=1)
    actual_magnitude = np.linalg.norm(sampled_flow, axis=1)
    predicted_magnitude = np.linalg.norm(predicted_flow, axis=1)
    outlier_mask = ~inlier_mask
    if not np.any(outlier_mask):
        outlier_mask = residual_magnitude >= np.quantile(residual_magnitude, 0.80)

    linear = affine[:, :2]
    singular = np.linalg.svd(linear, compute_uv=False)
    scale = float(np.sqrt(max(np.linalg.det(linear), 0.0)))
    rotation = math.degrees(math.atan2(linear[1, 0] - linear[0, 1], linear[0, 0] + linear[1, 1]))
    background_vectors = sampled_flow[inlier_mask]
    background_mean = background_vectors.mean(axis=0)
    background_mean_norm = float(np.linalg.norm(background_mean))
    background_mean_magnitude = float(actual_magnitude[inlier_mask].mean())
    coherence = background_mean_norm / max(background_mean_magnitude, 1.0e-12)
    return {
        "background_inlier_ratio": float(inlier_mask.mean()),
        "background_speed_median": float(np.median(actual_magnitude[inlier_mask])),
        "background_speed_mean": background_mean_magnitude,
        "background_speed_p90": float(np.quantile(actual_magnitude[inlier_mask], 0.90)),
        "background_mean_dx": float(background_mean[0]),
        "background_mean_dy": float(background_mean[1]),
        "background_direction_coherence": coherence,
        "background_model_speed_median": float(np.median(predicted_magnitude)),
        "background_residual_median": float(np.median(residual_magnitude[inlier_mask])),
        "foreground_actual_speed_median": float(np.median(actual_magnitude[outlier_mask])),
        "foreground_residual_median": float(np.median(residual_magnitude[outlier_mask])),
        "foreground_residual_p90": float(np.quantile(residual_magnitude[outlier_mask], 0.90)),
        "affine_scale": scale,
        "affine_anisotropy": float(singular.max() / max(singular.min(), 1.0e-12)),
        "affine_rotation_degrees": float(rotation),
        "affine_fallback": fallback,
    }


def bootstrap_ci(values, rng, iterations):
    values = np.asarray(values, dtype=np.float64)
    samples = rng.choice(values, (iterations, len(values)), replace=True).mean(axis=1)
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def summarize_pairs(pair_rows, rng, iterations):
    metrics = [
        "background_inlier_ratio",
        "background_speed_median",
        "background_speed_mean",
        "background_speed_p90",
        "background_direction_coherence",
        "background_model_speed_median",
        "background_residual_median",
        "foreground_actual_speed_median",
        "foreground_residual_median",
        "foreground_residual_p90",
        "affine_scale",
        "affine_anisotropy",
        "affine_rotation_degrees",
    ]
    result = {}
    for metric in metrics:
        values = [float(row[metric]) for row in pair_rows]
        result[metric] = {
            "mean": float(np.mean(values)),
            "bootstrap_95ci": bootstrap_ci(values, rng, iterations),
        }
    vectors = np.array([
        [row["background_mean_dx"], row["background_mean_dy"]]
        for row in pair_rows
    ], dtype=np.float64)
    speeds = np.linalg.norm(vectors, axis=1)
    if len(vectors) > 1:
        accelerations = np.linalg.norm(np.diff(vectors, axis=0), axis=1)
        speed_changes = np.abs(np.diff(speeds))
        result["background_vector_acceleration"] = {
            "mean": float(accelerations.mean()),
            "bootstrap_95ci": bootstrap_ci(accelerations, rng, iterations),
        }
        result["background_speed_change"] = {
            "mean": float(speed_changes.mean()),
            "bootstrap_95ci": bootstrap_ci(speed_changes, rng, iterations),
        }
    result["affine_fallback_pair_count"] = int(sum(row["affine_fallback"] for row in pair_rows))
    result["frame_pair_count"] = len(pair_rows)
    return result


def plot_prompt(rows, prompt, output):
    subset = [row for row in rows if row["prompt"] == prompt]
    variants = sorted({row["variant"] for row in subset})
    fig, axes = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True)
    for variant in variants:
        current = sorted(
            (row for row in subset if row["variant"] == variant),
            key=lambda row: row["frame_pair"],
        )
        x = [row["frame_pair"] for row in current]
        axes[0].plot(x, [row["background_speed_median"] for row in current], marker="o", label=variant)
        axes[1].plot(x, [row["foreground_residual_median"] for row in current], marker="o", label=variant)
    axes[0].set_ylabel("background flow (px/frame)")
    axes[1].set_ylabel("foreground residual (px/frame)")
    axes[1].set_xlabel("frame pair")
    axes[0].set_title(prompt)
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    variants = parse_variants(args.variant)
    args.output.mkdir(parents=True, exist_ok=True)
    flow_dir = args.output / "raft_flows"
    flow_dir.mkdir(exist_ok=True)
    device = torch.device(args.device)
    model = load_raft(args.raft_model, device)

    pair_rows = []
    for prompt_index, prompt in enumerate(args.prompts, start=1):
        filename = f"{prompt}.mp4"
        for variant, root in variants.items():
            video_path = root / filename
            if not video_path.exists():
                raise FileNotFoundError(video_path)
            print(
                f"[{prompt_index}/{len(args.prompts)}] {variant}: {prompt}",
                flush=True,
            )
            flows, fps = raft_flows(model, video_path, device)
            torch.save(
                torch.from_numpy(np.stack(flows)),
                flow_dir / f"{safe_name(prompt)}_{variant}.pt",
            )
            for pair_index, flow in enumerate(flows):
                pair_rows.append({
                    "prompt": prompt,
                    "variant": variant,
                    "frame_pair": pair_index,
                    "fps": fps,
                    **robust_affine_metrics(
                        flow, args.grid_stride, args.ransac_threshold
                    ),
                })
    write_csv(args.output / "frame_pair_metrics.csv", pair_rows)

    rng = np.random.default_rng(args.seed)
    summaries = {}
    for prompt in args.prompts:
        summaries[prompt] = {}
        for variant in variants:
            current = [
                row for row in pair_rows
                if row["prompt"] == prompt and row["variant"] == variant
            ]
            summaries[prompt][variant] = summarize_pairs(
                current, rng, args.bootstrap
            )
        fp16 = summaries[prompt]["fp16"]
        for variant in variants:
            if variant == "fp16":
                continue
            ratio = {}
            for metric in (
                "background_speed_median",
                "background_vector_acceleration",
                "background_speed_change",
                "foreground_actual_speed_median",
                "foreground_residual_median",
            ):
                if metric in summaries[prompt][variant] and metric in fp16:
                    ratio[metric] = (
                        summaries[prompt][variant][metric]["mean"]
                        / max(fp16[metric]["mean"], 1.0e-12)
                    )
            summaries[prompt][variant]["ratio_to_fp16"] = ratio
        plot_prompt(
            pair_rows, prompt,
            args.output / f"{safe_name(prompt)}_motion_decomposition.png",
        )

    giraffe_prompt = "a giraffe taking a peaceful walk"
    diagnostic = None
    if giraffe_prompt in summaries:
        diagnostic = {
            variant: summaries[giraffe_prompt][variant].get("ratio_to_fp16", {})
            for variant in variants if variant != "fp16"
        }
    summary = {
        "method": {
            "dense_flow": "RAFT things checkpoint, adjacent generated frames",
            "background": "RANSAC affine inliers on an 8-pixel grid",
            "foreground": "flow residual outside the dominant affine inlier set",
            "limitation": (
                "The decomposition is segmentation-free. Low background inlier ratios or "
                "many affine fallbacks indicate that object masks are needed before a causal claim."
            ),
        },
        "prompts": summaries,
        "giraffe_ratio_to_fp16": diagnostic,
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
