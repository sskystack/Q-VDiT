#!/usr/bin/env python3
"""Analyze whether inference-aligned quantization residuals drive spatial drift.

For each selected DDIM step, the exact CFG-combined quantized and FP outputs are
compared on the same latent.  The residual is decomposed into spatial frequency,
frame-shared energy, and the local translation tangent spanned by spatial
derivatives of x_t.  A larger tangent projection indicates that quantization
error can act like a coherent image translation rather than independent jitter.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-profile", type=Path, required=True)
    parser.add_argument("--mtd-profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-existing-video", type=Path)
    parser.add_argument("--baseline-rerun-video", type=Path)
    parser.add_argument("--mtd-existing-video", type=Path)
    parser.add_argument("--mtd-rerun-video", type=Path)
    parser.add_argument("--low-frequency-radius", type=float, default=0.125)
    return parser.parse_args()


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def video_frame_mae(left, right):
    if left is None or right is None or not left.exists() or not right.exists():
        return None
    first = cv2.VideoCapture(str(left))
    second = cv2.VideoCapture(str(right))
    values = []
    while True:
        ok_first, frame_first = first.read()
        ok_second, frame_second = second.read()
        if not ok_first or not ok_second:
            break
        values.append(float(np.abs(
            frame_first.astype(np.float32) - frame_second.astype(np.float32)
        ).mean()))
    first.release()
    second.release()
    return float(np.mean(values)) if values else None


def adjacent_cosine(tensor):
    frames = tensor[0].permute(1, 0, 2, 3).reshape(tensor.shape[2], -1)
    frames = torch.nn.functional.normalize(frames, dim=1, eps=1.0e-12)
    return float((frames[:-1] * frames[1:]).sum(1).mean())


def spatial_frequency_metrics(residual, low_radius):
    spectrum = torch.fft.fft2(residual.float(), dim=(-2, -1), norm="ortho")
    energy = spectrum.abs().square()
    height, width = residual.shape[-2:]
    fy = torch.fft.fftfreq(height, device=residual.device)
    fx = torch.fft.fftfreq(width, device=residual.device)
    yy, xx = torch.meshgrid(fy, fx, indexing="ij")
    radius = torch.sqrt(xx.square() + yy.square())
    low = radius <= low_radius
    dc = radius == 0
    total = energy.sum().clamp_min(1.0e-20)
    return {
        "low_frequency_energy_fraction": float(energy[..., low].sum() / total),
        "dc_energy_fraction": float(energy[..., dc].sum() / total),
    }


def translation_tangent_metrics(latent, residual):
    latent = latent.float()
    residual = residual.float()
    grad_x = 0.5 * (latent[..., 1:-1, 2:] - latent[..., 1:-1, :-2])
    grad_y = 0.5 * (latent[..., 2:, 1:-1] - latent[..., :-2, 1:-1])
    target = residual[..., 1:-1, 1:-1]
    coefficients = []
    fractions = []
    cosines = []
    for frame in range(target.shape[2]):
        gx = grad_x[0, :, frame].reshape(-1)
        gy = grad_y[0, :, frame].reshape(-1)
        value = target[0, :, frame].reshape(-1)
        design = torch.stack([gx, gy], dim=1)
        gram = design.T @ design
        scale = torch.trace(gram).clamp_min(1.0e-12)
        regularized = gram + torch.eye(2) * (scale * 1.0e-6)
        coefficient = torch.linalg.solve(regularized, design.T @ value)
        fitted = design @ coefficient
        target_energy = value.square().sum().clamp_min(1.0e-20)
        fitted_energy = fitted.square().sum()
        cosine = (fitted @ value) / (
            fitted.square().sum().sqrt() * target_energy.sqrt() + 1.0e-20
        )
        coefficients.append(coefficient)
        fractions.append(float(fitted_energy / target_energy))
        cosines.append(float(cosine))
    coefficients = torch.stack(coefficients)
    magnitudes = torch.linalg.vector_norm(coefficients, dim=1)
    if len(coefficients) > 1:
        acceleration = torch.linalg.vector_norm(
            coefficients[1:] - coefficients[:-1], dim=1
        ).mean()
    else:
        acceleration = coefficients.new_tensor(0.0)
    return {
        "translation_tangent_energy_fraction": float(np.mean(fractions)),
        "translation_tangent_cosine": float(np.mean(cosines)),
        "translation_coefficient_mean": float(magnitudes.mean()),
        "translation_coefficient_p95": float(torch.quantile(magnitudes, 0.95)),
        "translation_coefficient_frame_acceleration": float(acceleration),
        "translation_coefficient_mean_x": float(coefficients[:, 0].mean()),
        "translation_coefficient_mean_y": float(coefficients[:, 1].mean()),
    }


def analyze_file(path, variant, low_radius):
    payload = torch.load(path, map_location="cpu")
    quant = payload["quant_guided"].float()[:, :4]
    fp = payload["fp_guided"].float()[:, :4]
    latent = payload["latent"].float()[:, :4]
    residual = quant - fp
    residual_energy = residual.square().sum()
    fp_energy = fp.square().sum().clamp_min(1.0e-20)
    frame_mean = residual.mean(dim=2, keepdim=True)
    frame_shared_energy = frame_mean.square().sum() * residual.shape[2]
    return {
        "variant": variant,
        "sampling_progress": int(payload["sampling_progress"]),
        "model_timestep": int(payload["model_timestep"]),
        "relative_l2": float(torch.sqrt(residual_energy / fp_energy)),
        "residual_rms": float(residual.square().mean().sqrt()),
        "frame_shared_energy_fraction": float(
            frame_shared_energy / residual_energy.clamp_min(1.0e-20)
        ),
        "adjacent_frame_residual_cosine": adjacent_cosine(residual),
        **spatial_frequency_metrics(residual, low_radius),
        **translation_tangent_metrics(latent, residual),
    }


def phase(progress):
    if progress <= 25:
        return "early"
    if progress <= 65:
        return "middle"
    return "late"


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for variant, root in (
        ("baseline", args.baseline_profile),
        ("mtd", args.mtd_profile),
    ):
        files = sorted(root.glob("*/step_*.pt"))
        if not files:
            raise FileNotFoundError(f"No step profiles found under {root}")
        for path in files:
            row = analyze_file(path, variant, args.low_frequency_radius)
            row["phase"] = phase(row["sampling_progress"])
            rows.append(row)
    write_csv(args.output / "step_metrics.csv", rows)

    metrics = [
        "relative_l2",
        "residual_rms",
        "frame_shared_energy_fraction",
        "adjacent_frame_residual_cosine",
        "low_frequency_energy_fraction",
        "dc_energy_fraction",
        "translation_tangent_energy_fraction",
        "translation_tangent_cosine",
        "translation_coefficient_mean",
        "translation_coefficient_p95",
        "translation_coefficient_frame_acceleration",
    ]
    aggregate = {}
    for variant in ("baseline", "mtd"):
        aggregate[variant] = {"all": {}, "phases": {}}
        current = [row for row in rows if row["variant"] == variant]
        for metric in metrics:
            aggregate[variant]["all"][metric] = float(np.mean([
                row[metric] for row in current
            ]))
        for phase_name in ("early", "middle", "late"):
            phase_rows = [row for row in current if row["phase"] == phase_name]
            aggregate[variant]["phases"][phase_name] = {
                metric: float(np.mean([row[metric] for row in phase_rows]))
                for metric in metrics
            }

    ratios = {
        metric: aggregate["mtd"]["all"][metric]
        / max(abs(aggregate["baseline"]["all"][metric]), 1.0e-12)
        for metric in metrics
        if metric not in ("adjacent_frame_residual_cosine", "translation_tangent_cosine")
    }
    alignment = {
        "baseline_existing_vs_rerun_frame_mae": video_frame_mae(
            args.baseline_existing_video, args.baseline_rerun_video
        ),
        "mtd_existing_vs_rerun_frame_mae": video_frame_mae(
            args.mtd_existing_video, args.mtd_rerun_video
        ),
    }
    summary = {
        "comparison_contract": {
            "input_pairing": "FP and quantized outputs use the same x_t within each variant",
            "inference_alignment": "quant_guided/fp_guided reproduce the repository's exact 3-channel CFG rule",
            "translation_test": "least-squares projection onto spatial derivatives of the current latent",
            "limitation": "Baseline and MTD follow different generated trajectories; compare mechanism strength, not pointwise residual identity.",
        },
        "rerun_alignment": alignment,
        "aggregate": aggregate,
        "mtd_to_baseline_ratio": ratios,
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
