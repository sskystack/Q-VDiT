#!/usr/bin/env python3
"""Analyze MTD feature correspondence against VBench RAFT optical flow."""

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from argparse import Namespace

from vbench.third_party.RAFT.core.raft import RAFT
from vbench.third_party.RAFT.core.utils_core.utils import InputPadder


DEFAULT_RAFT = Path(
    "/home/zhouchongtian/quantization/models/vbench/raft_model/models/raft-things.pth"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--profile-dir", type=Path, required=True)
    parser.add_argument("--baseline-videos", type=Path, required=True)
    parser.add_argument("--mtd-videos", type=Path, required=True)
    parser.add_argument("--rerun-videos", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raft-model", type=Path, default=DEFAULT_RAFT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--search-radius", type=int, default=4)
    parser.add_argument(
        "--feature-view",
        choices=("cond", "guided"),
        default="cond",
        help=(
            "cond analyzes the pre-CFG conditional branch that is directly seen "
            "by the MTD reconstruction loss; guided analyzes the final CFG-combined "
            "denoising output"
        ),
    )
    return parser.parse_args()


def safe_name(value):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


def load_manifest(path):
    data = json.loads(path.read_text())
    rows = []
    for group in ("flip_1to0", "control_both_true"):
        for item in data[group]:
            rows.append({"group": group, **item})
    return data, rows


def load_frames(video_path, device):
    capture = cv2.VideoCapture(str(video_path))
    fps = capture.get(cv2.CAP_PROP_FPS)
    interval = max(1, round(fps / 8))
    frames = []
    uint8_frames = []
    while capture.isOpened():
        success, frame = capture.read()
        if not success:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        uint8_frames.append(frame)
        tensor = torch.from_numpy(frame).permute(2, 0, 1).float()[None].to(device)
        frames.append(tensor)
    capture.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from {video_path}")
    return frames[::interval], uint8_frames[::interval]


def load_raft(model_path, device):
    args = Namespace(
        model=str(model_path), small=False,
        mixed_precision=False, alternate_corr=False,
    )
    model = RAFT(args)
    checkpoint = torch.load(model_path, map_location="cpu")
    checkpoint = {key.replace("module.", ""): value for key, value in checkpoint.items()}
    model.load_state_dict(checkpoint)
    return model.to(device).eval()


def raft_video(model, video_path, device):
    frames, uint8_frames = load_frames(video_path, device)
    flows = []
    pair_scores = []
    with torch.no_grad():
        for image1, image2 in zip(frames[:-1], frames[1:]):
            padder = InputPadder(image1.shape)
            padded1, padded2 = padder.pad(image1, image2)
            _, flow = model(padded1, padded2, iters=20, test_mode=True)
            flow = flow[0, :, : image1.shape[-2], : image1.shape[-1]]
            magnitude = torch.linalg.vector_norm(flow.float(), dim=0)
            cut = max(1, int(magnitude.numel() * 0.05))
            pair_scores.append(float(torch.topk(magnitude.flatten(), cut).values.mean()))
            flows.append(flow.detach().float().cpu())
    return torch.stack(flows), pair_scores, uint8_frames


def downsample_flow(flow, height=16, width=16):
    original_height, original_width = flow.shape[-2:]
    resized = F.interpolate(flow, size=(height, width), mode="bilinear", align_corners=False)
    resized[:, 0] /= original_width / width
    resized[:, 1] /= original_height / height
    return resized


def correspondence_statistics(feature, temperature, radius):
    """Return large-window teacher/quant correspondence statistics."""
    feature = feature.detach().float()
    if feature.ndim == 5:
        feature = feature[0]
    channels, frames, height, width = feature.shape
    feature = F.normalize(feature, dim=0, eps=1.0e-6)
    current = feature[:, :-1].permute(1, 2, 3, 0).reshape(-1, height * width, channels)
    following = feature[:, 1:].permute(1, 0, 2, 3)
    kernel = radius * 2 + 1
    neighbours = F.unfold(following, kernel_size=kernel, padding=radius)
    neighbours = neighbours.reshape(
        frames - 1, channels, kernel * kernel, height * width
    ).permute(0, 3, 2, 1)
    logits = (current[:, :, None, :] * neighbours).sum(-1) / temperature

    valid_grid = following.new_ones(frames - 1, 1, height, width)
    valid = F.unfold(valid_grid, kernel_size=kernel, padding=radius)
    valid = valid.reshape(frames - 1, kernel * kernel, height * width).permute(0, 2, 1).bool()
    logits = logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
    probs = torch.softmax(logits, dim=-1)

    offsets_y, offsets_x = torch.meshgrid(
        torch.arange(-radius, radius + 1, device=feature.device),
        torch.arange(-radius, radius + 1, device=feature.device),
        indexing="ij",
    )
    offsets = torch.stack([offsets_x.flatten(), offsets_y.flatten()], dim=-1).float()
    outside = offsets.abs().amax(dim=-1) > 1
    outside_mass = probs[..., outside].sum(-1)
    top_index = probs.argmax(-1)
    top_outside = outside[top_index]
    displacement = torch.einsum("bpk,kd->bpd", probs, offsets)
    second_moment = torch.einsum(
        "bpk,k->bp", probs, offsets.square().sum(-1)
    )
    entropy = -(probs.clamp_min(1.0e-12) * probs.clamp_min(1.0e-12).log()).sum(-1)

    shape = (frames - 1, height, width)
    return {
        "probabilities": probs.reshape(frames - 1, height, width, -1),
        "outside_mass": outside_mass.reshape(shape),
        "top1_outside": top_outside.reshape(shape),
        "displacement": displacement.reshape(frames - 1, height, width, 2),
        "second_moment": second_moment.reshape(shape),
        "entropy": entropy.reshape(shape),
    }


def rankdata(values):
    values = np.asarray(values)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1
        start = end
    return ranks


def spearman(x, y):
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)
    if len(x) < 3 or np.std(x) < 1.0e-12 or np.std(y) < 1.0e-12:
        return float("nan")
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])


def direction_cosine(displacement, flow):
    displacement = np.asarray(displacement).reshape(-1, 2)
    flow = np.asarray(flow).reshape(-1, 2)
    denominator = np.linalg.norm(displacement, axis=1) * np.linalg.norm(flow, axis=1)
    valid = denominator > 1.0e-6
    if not np.any(valid):
        return float("nan")
    cosine = (displacement[valid] * flow[valid]).sum(-1) / denominator[valid]
    return float(np.mean(cosine))


def mean_or_nan(values):
    values = [value for value in values if np.isfinite(value)]
    return float(np.mean(values)) if values else float("nan")


def bootstrap_mean_ci(values, seed=42, iterations=5000):
    values = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    if len(values) == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(iterations, len(values)), replace=True).mean(axis=1)
    return [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))]


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate_rerun(existing_path, rerun_path):
    if rerun_path is None or not rerun_path.exists():
        return None
    old_capture = cv2.VideoCapture(str(existing_path))
    new_capture = cv2.VideoCapture(str(rerun_path))
    differences = []
    while True:
        ok_old, old = old_capture.read()
        ok_new, new = new_capture.read()
        if not ok_old or not ok_new:
            break
        differences.append(float(np.abs(old.astype(np.float32) - new.astype(np.float32)).mean()))
    old_capture.release()
    new_capture.release()
    return float(np.mean(differences)) if differences else None


def plot_outside(rows, output):
    steps = sorted({row["sampling_progress"] for row in rows})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    colors = {"flip_1to0": "#d95f02", "control_both_true": "#1b9e77"}
    for group in colors:
        subset = [row for row in rows if row["group"] == group]
        axes[0].plot(
            steps,
            [mean_or_nan([r["outside_mass"] for r in subset if r["sampling_progress"] == step]) for step in steps],
            marker="o", label=group, color=colors[group],
        )
        axes[1].plot(
            steps,
            [mean_or_nan([r["top1_outside_rate"] for r in subset if r["sampling_progress"] == step]) for step in steps],
            marker="o", label=group, color=colors[group],
        )
    axes[0].set_ylabel("Probability mass outside 3x3")
    axes[1].set_ylabel("Top-1 outside 3x3 rate")
    for axis in axes:
        axis.set_xlabel("DDIM sampling progress")
        axis.grid(alpha=0.25)
        axis.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_correlation(rows, output):
    steps = sorted({row["sampling_progress"] for row in rows})
    groups = ["flip_1to0", "control_both_true"]
    matrix = np.array([
        [mean_or_nan([r["spearman_magnitude"] for r in rows if r["group"] == group and r["sampling_progress"] == step]) for step in steps]
        for group in groups
    ])
    fig, ax = plt.subplots(figsize=(8, 3.2))
    image = ax.imshow(matrix, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(steps)), steps)
    ax.set_yticks(range(len(groups)), groups)
    ax.set_xlabel("DDIM sampling progress")
    ax.set_title("Feature displacement vs. RAFT magnitude (Spearman)")
    for y in range(matrix.shape[0]):
        for x in range(matrix.shape[1]):
            ax.text(x, y, f"{matrix[y, x]:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(image, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_entropy(rows, output):
    groups = ["flip_1to0", "control_both_true"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for axis, group in zip(axes, groups):
        subset = [row for row in rows if row["group"] == group]
        axis.boxplot(
            [[row["entropy_high_flow"] for row in subset], [row["entropy_low_flow"] for row in subset]],
            labels=["high-flow", "low-flow"], showfliers=False,
        )
        axis.set_title(group)
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Teacher correspondence entropy")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    flow_dir = args.output / "raft_flows"
    flow_dir.mkdir(exist_ok=True)
    manifest, items = load_manifest(args.manifest)
    device = torch.device(args.device)
    raft = load_raft(args.raft_model, device)

    raft_data = {}
    alignment_rows = []
    for item_index, item in enumerate(items, start=1):
        prompt = item["prompt"]
        filename = f"{prompt}.mp4"
        print(f"[RAFT {item_index}/{len(items)}] {prompt}", flush=True)
        raft_data[prompt] = {}
        for variant, root in (("baseline", args.baseline_videos), ("mtd", args.mtd_videos)):
            flow, scores, _ = raft_video(raft, root / filename, device)
            torch.save(flow, flow_dir / f"{safe_name(prompt)}_{variant}.pt")
            raft_data[prompt][variant] = {"flow": flow, "scores": scores}
        rerun = None
        if args.rerun_videos is not None:
            rerun = args.rerun_videos / filename
        alignment_rows.append({
            "group": item["group"],
            "prompt": prompt,
            "mtd_existing_vs_profile_rerun_frame_mae": validate_rerun(
                args.mtd_videos / filename, rerun
            ),
        })

    rows = []
    token_rows = []
    for item_index, item in enumerate(items, start=1):
        prompt = item["prompt"]
        prompt_dir = args.profile_dir / f"{int(item['index']):03d}_{safe_name(prompt)}"
        files = sorted(prompt_dir.glob("step_*.pt"))
        if not files:
            raise FileNotFoundError(f"No profile files found in {prompt_dir}")
        mtd_flow = downsample_flow(raft_data[prompt]["mtd"]["flow"])
        flow_vector = mtd_flow.permute(0, 2, 3, 1).numpy()
        flow_magnitude = np.linalg.norm(flow_vector, axis=-1)
        high_threshold = np.quantile(flow_magnitude, 0.80)
        low_threshold = np.quantile(flow_magnitude, 0.50)
        high_mask = flow_magnitude >= high_threshold
        low_mask = flow_magnitude <= low_threshold

        for feature_path in files:
            payload = torch.load(feature_path, map_location="cpu")
            progress = int(payload["sampling_progress"])
            fp_feature_key = f"fp_{args.feature_view}_pooled"
            quant_feature_key = f"quant_{args.feature_view}_pooled"
            fp_stats = correspondence_statistics(
                payload[fp_feature_key], args.temperature, args.search_radius
            )
            quant_stats = correspondence_statistics(
                payload[quant_feature_key], args.temperature, args.search_radius
            )
            displacement = fp_stats["displacement"].numpy()
            displacement_norm = np.linalg.norm(displacement, axis=-1)
            probabilities_fp = fp_stats["probabilities"].numpy()
            probabilities_q = quant_stats["probabilities"].numpy()
            correspondence_kl = np.sum(
                probabilities_fp * (
                    np.log(np.clip(probabilities_fp, 1.0e-12, None))
                    - np.log(np.clip(probabilities_q, 1.0e-12, None))
                ), axis=-1,
            )
            row = {
                "group": item["group"],
                "prompt_index": item["index"],
                "prompt": prompt,
                "sampling_progress": progress,
                "model_timestep": int(payload["model_timestep"]),
                "outside_mass": float(fp_stats["outside_mass"].mean()),
                "top1_outside_rate": float(fp_stats["top1_outside"].float().mean()),
                "teacher_displacement_mean": float(displacement_norm.mean()),
                "teacher_displacement_p95": float(np.quantile(displacement_norm, 0.95)),
                "teacher_second_moment": float(fp_stats["second_moment"].mean()),
                "spearman_magnitude": spearman(displacement_norm, flow_magnitude),
                "direction_cosine": direction_cosine(displacement, flow_vector),
                "entropy_high_flow": float(fp_stats["entropy"].numpy()[high_mask].mean()),
                "entropy_low_flow": float(fp_stats["entropy"].numpy()[low_mask].mean()),
                "quant_teacher_correspondence_kl": float(correspondence_kl.mean()),
                "quant_teacher_displacement_mae": float(np.abs(
                    quant_stats["displacement"].numpy() - displacement
                ).mean()),
                "baseline_mean_raft_score": float(np.mean(raft_data[prompt]["baseline"]["scores"])),
                "mtd_mean_raft_score": float(np.mean(raft_data[prompt]["mtd"]["scores"])),
            }
            rows.append(row)
            for frame_pair in range(displacement.shape[0]):
                token_rows.append({
                    "group": item["group"],
                    "prompt": prompt,
                    "sampling_progress": progress,
                    "frame_pair": frame_pair,
                    "outside_mass": float(fp_stats["outside_mass"][frame_pair].mean()),
                    "top1_outside_rate": float(fp_stats["top1_outside"][frame_pair].float().mean()),
                    "teacher_displacement_mean": float(displacement_norm[frame_pair].mean()),
                    "raft_magnitude_mean": float(flow_magnitude[frame_pair].mean()),
                    "entropy_high_flow": float(fp_stats["entropy"].numpy()[frame_pair][high_mask[frame_pair]].mean()),
                    "entropy_low_flow": float(fp_stats["entropy"].numpy()[frame_pair][low_mask[frame_pair]].mean()),
                })

    write_csv(args.output / "video_step_summary.csv", rows)
    write_csv(args.output / "frame_pair_summary.csv", token_rows)
    write_csv(args.output / "rerun_alignment.csv", alignment_rows)

    summary = {
        "seed": manifest["seed"],
        "feature_view": args.feature_view,
        "groups": {},
    }
    for group in ("flip_1to0", "control_both_true"):
        group_rows = [row for row in rows if row["group"] == group]
        summary["groups"][group] = {}
        for metric in (
            "outside_mass", "top1_outside_rate", "teacher_displacement_mean",
            "spearman_magnitude", "direction_cosine", "entropy_high_flow",
            "entropy_low_flow", "quant_teacher_correspondence_kl",
        ):
            values = [row[metric] for row in group_rows]
            summary["groups"][group][metric] = {
                "mean": mean_or_nan(values),
                "bootstrap_95ci": bootstrap_mean_ci(values),
            }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))

    plot_outside(rows, args.output / "01_outside_search_range.png")
    plot_correlation(rows, args.output / "02_feature_raft_correlation.png")
    plot_entropy(rows, args.output / "03_high_low_flow_entropy.png")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
