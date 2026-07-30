#!/usr/bin/env python3
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


PROMPTS = (0, 2, 6)
WINDOWS = ((1, 10), (11, 20), (21, 30), (41, 50), (61, 70), (81, 90), (91, 100))


def read_video(path):
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"could not decode {path}")
    return torch.from_numpy(np.stack(frames)).permute(3, 0, 1, 2).float() / 255.0


def cosine(x, ref):
    a, b = x.double().reshape(-1), ref.double().reshape(-1)
    return float((torch.dot(a, b) / (torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b) + 1e-12)).clamp(-1, 1))


def nmse(x, ref):
    return float(torch.mean((x - ref).square()) / (torch.mean(ref.square()) + 1e-12))


def lowpass(x, factor=16):
    return F.avg_pool2d(x.permute(1, 0, 2, 3), factor, factor).permute(1, 0, 2, 3).contiguous()


def highpass(x, factor=16):
    _, _, h, w = x.shape
    y = x.permute(1, 0, 2, 3)
    low = F.interpolate(F.avg_pool2d(y, factor, factor), size=(h, w), mode="bilinear", align_corners=False)
    return (y - low).permute(1, 0, 2, 3).contiguous()


def latent_metrics(x, ref):
    diff = (x.double() - ref.double()).reshape(-1)
    ref_flat = ref.double().reshape(-1)
    return {
        "latent_relative_l2": float(torch.linalg.vector_norm(diff) / torch.linalg.vector_norm(ref_flat).clamp_min(1e-12)),
        "latent_rmse": float(torch.sqrt(torch.mean(diff.square()))),
        "latent_cosine": cosine(x, ref),
    }


def video_metrics(x, ref):
    lx, lr = lowpass(x), lowpass(ref)
    hx, hr = highpass(x), highpass(ref)
    dx, dr = x[:, 1:] - x[:, :-1], ref[:, 1:] - ref[:, :-1]
    return {
        "pixel_nmse": nmse(x, ref),
        "layout_nmse": nmse(lx, lr),
        "detail_nmse": nmse(hx, hr),
        "temporal_nmse": nmse(dx, dr),
    }


def aggregate(rows, key):
    output = []
    for start, end in WINDOWS:
        values = np.asarray([r[key] for r in rows if r["progress_start"] == start], dtype=np.float64)
        output.append({
            "progress_start": start,
            "progress_end": end,
            "progress_center": (start + end) / 2,
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)),
            "min": float(values.min()),
            "max": float(values.max()),
        })
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    rows = []

    for prompt_index in PROMPTS:
        fp_base = root / f"prompt{prompt_index}/fp/runtime"
        fp_latent = torch.load(fp_base / "final_latents/final_latent_0000.pt", map_location="cpu")
        fp_noise = torch.load(fp_base / "init_noise/init_noise_0000.pt", map_location="cpu")
        fp_video = read_video(root / f"prompt{prompt_index}/fp/video_opensora/sample_0.mp4")
        for start, end in WINDOWS:
            tag = f"p{start:03d}_{end:03d}"
            base = root / f"prompt{prompt_index}/w4a6/{tag}/runtime"
            latent = torch.load(base / "final_latents/final_latent_0000.pt", map_location="cpu")
            noise = torch.load(base / "init_noise/init_noise_0000.pt", map_location="cpu")
            video = read_video(root / f"prompt{prompt_index}/w4a6/{tag}/video_opensora/sample_0.mp4")
            trace = json.loads((base / "quant_trace_batch_0000.json").read_text())
            active = [x for x in trace if x["in_quant_window"]]
            rows.append({
                "prompt_index": prompt_index,
                "progress_start": start,
                "progress_end": end,
                "progress_center": (start + end) / 2,
                "latent_finite": bool(torch.isfinite(latent).all()),
                "initial_noise_exact": bool(torch.equal(noise, fp_noise)),
                "trace_valid": len(trace) == 100
                and [x["sampling_progress"] for x in active] == list(range(start, end + 1))
                and all(x["weight_quant"] and x["act_quant"] for x in active),
                **latent_metrics(latent, fp_latent),
                **video_metrics(video, fp_video),
            })

    keys = ("latent_relative_l2", "pixel_nmse", "layout_nmse", "detail_nmse", "temporal_nmse")
    aggregates = {key: aggregate(rows, key) for key in keys}
    peak_windows = {
        str(prompt): max(
            (r for r in rows if r["prompt_index"] == prompt),
            key=lambda r: r["latent_relative_l2"],
        )["progress_start"]
        for prompt in PROMPTS
    }
    early_late_ratios = {}
    for prompt in PROMPTS:
        selected = [r for r in rows if r["prompt_index"] == prompt]
        early = np.mean([r["latent_relative_l2"] for r in selected if r["progress_start"] in (1, 11, 21)])
        late = np.mean([r["latent_relative_l2"] for r in selected if r["progress_start"] in (81, 91)])
        early_late_ratios[str(prompt)] = float(early / max(late, 1e-12))

    summary = {
        "prompts": list(PROMPTS),
        "seed": 42,
        "windows": [list(x) for x in WINDOWS],
        "rows": rows,
        "aggregates": aggregates,
        "all_latents_finite": all(r["latent_finite"] for r in rows),
        "all_initial_noise_exact": all(r["initial_noise_exact"] for r in rows),
        "all_traces_valid": all(r["trace_valid"] for r in rows),
        "peak_window_start_by_prompt": peak_windows,
        "peak_window_counts": dict(Counter(peak_windows.values())),
        "early_to_late_ratio_by_prompt": early_late_ratios,
    }
    (root / "b2_summary.json").write_text(json.dumps(summary, indent=2))
    with (root / "b2_rows.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(PROMPTS)))
    for color, prompt in zip(colors, PROMPTS):
        selected = [r for r in rows if r["prompt_index"] == prompt]
        xs = [r["progress_center"] for r in selected]
        axes[0].plot(xs, [r["latent_relative_l2"] for r in selected], marker="o", alpha=0.65, color=color, label=f"prompt {prompt}")
        axes[1].plot(xs, [r["detail_nmse"] for r in selected], marker="o", alpha=0.65, color=color, label=f"prompt {prompt}")
    for ax, key in zip(axes, ("latent_relative_l2", "detail_nmse")):
        agg = aggregates[key]
        xs = [x["progress_center"] for x in agg]
        means = [x["mean"] for x in agg]
        stds = [x["std"] for x in agg]
        ax.errorbar(xs, means, yerr=stds, color="black", linewidth=2.5, marker="s", capsize=3, label="mean ± std")
        ax.set_xlabel("Sampling progress window center")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Final latent relative L2")
    axes[1].set_ylabel("Decoded high-pass detail NMSE")
    axes[0].legend(ncol=2, fontsize=8)
    fig.suptitle("Experiment B2: cross-prompt W4A6 timestep sensitivity, seed 42")
    fig.tight_layout()
    fig.savefig(root / "b2_cross_prompt_curves.png", dpi=180)
    plt.close(fig)

    print(json.dumps({
        "rows": len(rows),
        "all_latents_finite": summary["all_latents_finite"],
        "all_initial_noise_exact": summary["all_initial_noise_exact"],
        "all_traces_valid": summary["all_traces_valid"],
        "peak_window_counts": summary["peak_window_counts"],
        "early_to_late_ratio_by_prompt": summary["early_to_late_ratio_by_prompt"],
    }, indent=2))


if __name__ == "__main__":
    main()
