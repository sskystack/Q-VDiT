#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


MODES = ("w4", "a6", "w4a6")
WINDOWS = tuple((start, start + 9) for start in range(1, 100, 10))


def read_video(path):
    capture = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    if not frames:
        raise RuntimeError(f"could not decode {path}")
    # [C,T,H,W], normalized to [0,1].
    return torch.from_numpy(np.stack(frames)).permute(3, 0, 1, 2).float() / 255.0


def cosine(x, ref):
    a = x.double().reshape(-1)
    b = ref.double().reshape(-1)
    value = torch.dot(a, b) / (torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b) + 1e-12)
    return float(value.clamp(-1.0, 1.0))


def nmse(x, ref):
    return float(torch.mean((x - ref).square()) / (torch.mean(ref.square()) + 1e-12))


def lowpass(x, factor=16):
    y = x.permute(1, 0, 2, 3)
    y = F.avg_pool2d(y, kernel_size=factor, stride=factor)
    return y.permute(1, 0, 2, 3).contiguous()


def highpass(x, factor=16):
    _, _, height, width = x.shape
    y = x.permute(1, 0, 2, 3)
    low = F.avg_pool2d(y, kernel_size=factor, stride=factor)
    low = F.interpolate(low, size=(height, width), mode="bilinear", align_corners=False)
    return (y - low).permute(1, 0, 2, 3).contiguous()


def metrics(x, ref):
    low_x, low_ref = lowpass(x), lowpass(ref)
    high_x, high_ref = highpass(x), highpass(ref)
    dx, dref = x[:, 1:] - x[:, :-1], ref[:, 1:] - ref[:, :-1]
    mse = float(torch.mean((x - ref).square()))
    return {
        "pixel_nmse": nmse(x, ref),
        "pixel_cosine": cosine(x, ref),
        "psnr_db": float(-10.0 * np.log10(max(mse, 1e-12))),
        "layout_nmse": nmse(low_x, low_ref),
        "layout_cosine": cosine(low_x, low_ref),
        "detail_nmse": nmse(high_x, high_ref),
        "detail_cosine": cosine(high_x, high_ref),
        "temporal_difference_nmse": nmse(dx, dref),
        "temporal_difference_cosine": cosine(dx, dref),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    ref = read_video(root / "fp/video_opensora/sample_0.mp4")
    rows = []
    for start, end in WINDOWS:
        tag = f"p{start:03d}_{end:03d}"
        for mode in MODES:
            video = read_video(root / mode / tag / "video_opensora/sample_0.mp4")
            rows.append({
                "mode": mode,
                "progress_start": start,
                "progress_end": end,
                "progress_center": (start + end) / 2,
                "frames": int(video.shape[1]),
                **metrics(video, ref),
            })

    with (root / "b1_video_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (root / "b1_video_metrics.json").write_text(json.dumps(rows, indent=2))

    panels = (
        ("pixel_nmse", "Pixel NMSE"),
        ("layout_nmse", "Low-pass layout NMSE"),
        ("detail_nmse", "High-pass detail NMSE"),
        ("temporal_difference_nmse", "Temporal-difference NMSE"),
    )
    colors = {"w4": "#377eb8", "a6": "#e41a1c", "w4a6": "#4daf4a"}
    labels = {"w4": "W4-only", "a6": "A6-only", "w4a6": "W4A6"}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (key, title) in zip(axes.reshape(-1), panels):
        for mode in MODES:
            selected = [x for x in rows if x["mode"] == mode]
            ax.plot(
                [x["progress_center"] for x in selected],
                [x[key] for x in selected],
                marker="o",
                color=colors[mode],
                label=labels[mode],
            )
        ax.set_title(title)
        ax.set_xlabel("Sampling progress window center")
        ax.grid(alpha=0.25)
    axes[0, 0].legend()
    fig.suptitle("Experiment B1: decoded-video sensitivity to timestep-window quantization")
    fig.tight_layout()
    fig.savefig(root / "b1_video_sensitivity_curves.png", dpi=180)
    plt.close(fig)
    print(json.dumps({"rows": len(rows), "all_frames_16": all(x["frames"] == 16 for x in rows)}, indent=2))


if __name__ == "__main__":
    main()
