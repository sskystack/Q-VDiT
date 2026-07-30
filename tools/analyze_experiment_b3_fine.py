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


PROMPTS = (0, 2, 6)
WINDOWS = ((11, 15), (16, 20), (21, 25), (26, 30))


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
        raise RuntimeError(f"Could not decode {path}")
    return torch.from_numpy(np.stack(frames)).permute(3, 0, 1, 2).float() / 255.0


def nmse(value, reference):
    return float(
        torch.mean((value - reference).square())
        / torch.mean(reference.square()).clamp_min(1e-12)
    )


def latent_metrics(value, reference):
    value = value.double()
    reference = reference.double()
    difference = value - reference
    flat_value = value.reshape(-1)
    flat_reference = reference.reshape(-1)
    return {
        "latent_relative_l2": float(
            torch.linalg.vector_norm(difference)
            / torch.linalg.vector_norm(reference).clamp_min(1e-12)
        ),
        "latent_rmse": float(torch.sqrt(torch.mean(difference.square()))),
        "latent_cosine": float(
            torch.dot(flat_value, flat_reference)
            / (
                torch.linalg.vector_norm(flat_value)
                * torch.linalg.vector_norm(flat_reference)
            ).clamp_min(1e-12)
        ),
    }


def video_metrics(value, reference):
    value_2d = value.permute(1, 0, 2, 3)
    reference_2d = reference.permute(1, 0, 2, 3)
    low_value = F.avg_pool2d(value_2d, 16, 16)
    low_reference = F.avg_pool2d(reference_2d, 16, 16)
    up_value = F.interpolate(low_value, size=value.shape[-2:], mode="bilinear", align_corners=False)
    up_reference = F.interpolate(low_reference, size=reference.shape[-2:], mode="bilinear", align_corners=False)
    temporal_value = value[:, 1:] - value[:, :-1]
    temporal_reference = reference[:, 1:] - reference[:, :-1]
    return {
        "pixel_nmse": nmse(value, reference),
        "layout_nmse": nmse(low_value, low_reference),
        "detail_nmse": nmse(value_2d - up_value, reference_2d - up_reference),
        "temporal_nmse": nmse(temporal_value, temporal_reference),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--reference-root", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    reference_root = Path(args.reference_root)
    rows = []

    for prompt in PROMPTS:
        fp_runtime = reference_root / f"prompt{prompt}/fp/runtime"
        fp_latent = torch.load(
            fp_runtime / "final_latents/final_latent_0000.pt", map_location="cpu"
        )
        fp_noise = torch.load(fp_runtime / "init_noise/init_noise_0000.pt", map_location="cpu")
        fp_video = read_video(reference_root / f"prompt{prompt}/fp/video_opensora/sample_0.mp4")
        for start, end in WINDOWS:
            tag = f"p{start:03d}_{end:03d}"
            base = root / f"prompt{prompt}/w4a6/{tag}"
            runtime = base / "runtime"
            latent = torch.load(
                runtime / "final_latents/final_latent_0000.pt", map_location="cpu"
            )
            noise = torch.load(runtime / "init_noise/init_noise_0000.pt", map_location="cpu")
            video = read_video(base / "video_opensora/sample_0.mp4")
            trace = json.loads((runtime / "quant_trace_batch_0000.json").read_text())
            active = [item for item in trace if item["in_quant_window"]]
            inactive = [item for item in trace if not item["in_quant_window"]]
            rows.append(
                {
                    "prompt_index": prompt,
                    "progress_start": start,
                    "progress_end": end,
                    "progress_center": (start + end) / 2,
                    "latent_finite": bool(torch.isfinite(latent).all()),
                    "initial_noise_exact": bool(torch.equal(noise, fp_noise)),
                    "trace_valid": len(trace) == 100
                    and [item["sampling_progress"] for item in active]
                    == list(range(start, end + 1))
                    and all(item["weight_quant"] and item["act_quant"] for item in active)
                    and all(
                        not item["weight_quant"] and not item["act_quant"]
                        for item in inactive
                    ),
                    **latent_metrics(latent, fp_latent),
                    **video_metrics(video, fp_video),
                }
            )

    aggregates = []
    for start, end in WINDOWS:
        values = [row["latent_relative_l2"] for row in rows if row["progress_start"] == start]
        aggregates.append(
            {
                "progress_start": start,
                "progress_end": end,
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)),
            }
        )
    summary = {
        "prompts": list(PROMPTS),
        "seed": 42,
        "windows": [list(window) for window in WINDOWS],
        "rows": rows,
        "latent_relative_l2_aggregate": aggregates,
        "all_latents_finite": all(row["latent_finite"] for row in rows),
        "all_initial_noise_exact": all(row["initial_noise_exact"] for row in rows),
        "all_traces_valid": all(row["trace_valid"] for row in rows),
    }
    (root / "b3_summary.json").write_text(json.dumps(summary, indent=2))
    with (root / "b3_rows.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    fig, axis = plt.subplots(figsize=(8.2, 4.8))
    for prompt in PROMPTS:
        subset = [row for row in rows if row["prompt_index"] == prompt]
        axis.plot(
            [row["progress_center"] for row in subset],
            [row["latent_relative_l2"] for row in subset],
            marker="o",
            alpha=0.7,
            label=f"prompt {prompt}",
        )
    axis.errorbar(
        [(item["progress_start"] + item["progress_end"]) / 2 for item in aggregates],
        [item["mean"] for item in aggregates],
        yerr=[item["std"] for item in aggregates],
        color="black",
        linewidth=2.5,
        marker="s",
        capsize=3,
        label="mean ± std",
    )
    axis.set_xlabel("Sampling progress sub-window center")
    axis.set_ylabel("Final latent relative L2")
    axis.set_title("Fine-grained W4A6 sensitivity in progress 11–30")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(root / "b3_fine_window_sensitivity.png", dpi=200)
    plt.close(fig)
    print(json.dumps({key: summary[key] for key in summary if key.startswith("all_")}, indent=2))


if __name__ == "__main__":
    main()
