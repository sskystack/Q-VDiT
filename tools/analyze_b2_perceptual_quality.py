#!/usr/bin/env python3
"""Add CLIP and LPIPS evidence to the completed B2 causal-window videos."""

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import lpips
import matplotlib.pyplot as plt
import numpy as np
import open_clip
import torch
from PIL import Image


PROMPTS = (0, 2, 6)
WINDOWS = ((1, 10), (11, 20), (21, 30), (41, 50), (61, 70), (81, 90), (91, 100))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--clip-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def read_video(path):
    capture = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    if len(frames) != 16:
        raise RuntimeError(f"Expected 16 frames, found {len(frames)} in {path}")
    return np.stack(frames)


@torch.no_grad()
def clip_score(frames, prompt, model, preprocess, tokenizer, device):
    images = torch.stack([preprocess(Image.fromarray(frame)) for frame in frames]).to(device)
    text = tokenizer([prompt]).to(device)
    image_features = model.encode_image(images)
    text_features = model.encode_text(text)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return float((image_features @ text_features.T).mean())


@torch.no_grad()
def lpips_score(frames, reference, network, device):
    value = torch.from_numpy(frames).permute(0, 3, 1, 2).float().to(device) / 127.5 - 1.0
    target = torch.from_numpy(reference).permute(0, 3, 1, 2).float().to(device) / 127.5 - 1.0
    values = []
    for start in range(0, len(value), 4):
        values.append(network(value[start:start + 4], target[start:start + 4]).flatten())
    return float(torch.cat(values).mean())


def main():
    args = parse_args()
    root = Path(args.root)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    row_path = output / "b2_perceptual_rows.jsonl"
    rows = [
        json.loads(line)
        for line in row_path.read_text().splitlines()
        if line.strip()
    ] if row_path.exists() else []
    completed = {
        (int(row["prompt_index"]), int(row["progress_start"]))
        for row in rows
    }
    prompts = [line.strip() for line in Path(args.prompt_file).read_text().splitlines() if line.strip()]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained=args.clip_checkpoint, device=device
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer("ViT-L-14")
    perceptual = lpips.LPIPS(net="vgg", verbose=False).to(device).eval()

    montage = []
    for prompt_index in PROMPTS:
        prompt = prompts[prompt_index]
        fp_path = root / f"prompt{prompt_index}/fp/video_opensora/sample_0.mp4"
        fp_frames = read_video(fp_path)
        fp_clip = clip_score(fp_frames, prompt, model, preprocess, tokenizer, device)
        for start, end in WINDOWS:
            tag = f"p{start:03d}_{end:03d}"
            video_path = root / f"prompt{prompt_index}/w4a6/{tag}/video_opensora/sample_0.mp4"
            frames = read_video(video_path)
            if (prompt_index, start) not in completed:
                score = clip_score(frames, prompt, model, preprocess, tokenizer, device)
                perceptual_distance = lpips_score(frames, fp_frames, perceptual, device)
                row = {
                    "prompt_index": prompt_index,
                    "progress_start": start,
                    "progress_end": end,
                    "progress_center": (start + end) / 2,
                    "fp_clip_score": fp_clip,
                    "quant_clip_score": score,
                    "clip_delta_vs_fp": score - fp_clip,
                    "clip_absolute_delta_vs_fp": abs(score - fp_clip),
                    "lpips_to_fp": perceptual_distance,
                }
                rows.append(row)
                with row_path.open("a") as handle:
                    handle.write(json.dumps(row) + "\n")
                    handle.flush()
                completed.add((prompt_index, start))
                print(json.dumps({
                    "completed": len(completed),
                    "total": len(PROMPTS) * len(WINDOWS),
                    **row,
                }), flush=True)
            if prompt_index == 0 and (start, end) in ((1, 10), (41, 50), (91, 100)):
                montage.append((f"W4A6 {start}–{end}", frames[7]))
        if prompt_index == 0:
            montage.insert(0, ("Full precision", fp_frames[7]))

    rows.sort(key=lambda row: (int(row["prompt_index"]), int(row["progress_start"])))
    if len(rows) != len(PROMPTS) * len(WINDOWS):
        raise RuntimeError(f"Expected {len(PROMPTS) * len(WINDOWS)} rows, found {len(rows)}")
    with (output / "b2_perceptual_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    aggregates = []
    for start, end in WINDOWS:
        subset = [row for row in rows if row["progress_start"] == start]
        aggregate = {"progress_start": start, "progress_end": end}
        for metric in ("lpips_to_fp", "clip_delta_vs_fp", "clip_absolute_delta_vs_fp"):
            values = np.asarray([row[metric] for row in subset], dtype=np.float64)
            aggregate[f"{metric}_mean"] = float(values.mean())
            aggregate[f"{metric}_std"] = float(values.std(ddof=1))
        aggregates.append(aggregate)

    centers = [(item["progress_start"] + item["progress_end"]) / 2 for item in aggregates]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4))
    for axis, metric, ylabel in (
        (axes[0], "lpips_to_fp", "LPIPS to full-precision video"),
        (axes[1], "clip_absolute_delta_vs_fp", "Absolute CLIP-score change"),
    ):
        axis.errorbar(
            centers,
            [item[f"{metric}_mean"] for item in aggregates],
            yerr=[item[f"{metric}_std"] for item in aggregates],
            marker="o",
            linewidth=2,
            capsize=3,
        )
        axis.set_xlabel("Quantized sampling-window center")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
    fig.suptitle("B2 perceptual and semantic effects across causal W4A6 windows")
    fig.tight_layout()
    fig.savefig(output / "b2_perceptual_quality_curves.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(montage), figsize=(4.2 * len(montage), 4.2))
    for axis, (title, frame) in zip(np.atleast_1d(axes), montage):
        axis.imshow(frame)
        axis.set_title(title)
        axis.axis("off")
    fig.suptitle("Prompt 0, frame 8: causal-window visual comparison")
    fig.tight_layout()
    fig.savefig(output / "b2_prompt0_visual_comparison.png", dpi=220)
    plt.close(fig)

    summary = {
        "prompts": list(PROMPTS),
        "seed": 42,
        "windows": [list(window) for window in WINDOWS],
        "runtime_device": str(device),
        "rows": len(rows),
        "aggregate": aggregates,
        "all_values_finite": all(
            math.isfinite(float(row[key]))
            for row in rows
            for key in ("fp_clip_score", "quant_clip_score", "clip_delta_vs_fp", "lpips_to_fp")
        ),
    }
    (output / "b2_perceptual_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"rows": summary["rows"], "all_values_finite": summary["all_values_finite"]}, indent=2))


if __name__ == "__main__":
    main()
