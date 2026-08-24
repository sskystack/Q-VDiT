#!/usr/bin/env python3
"""Profile teacher correspondence, RAFT reliability, and RAFT-guided MTD cost."""

import argparse
import json
import math
import re
import time
from argparse import Namespace
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from qdiff.mtd import motion_transport_distillation
from vbench.third_party.RAFT.core.raft import RAFT
from vbench.third_party.RAFT.core.utils_core.utils import InputPadder


def safe_name(value):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--flow-dir", type=Path, required=True)
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--raft-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--feature-view", choices=("cond", "guided"), default="cond")
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--timing-batch", type=int, default=4)
    parser.add_argument("--timing-warmup", type=int, default=3)
    parser.add_argument("--timing-repeats", type=int, default=12)
    return parser.parse_args()


def manifest_items(path):
    data = json.loads(path.read_text())
    return [
        {"group": group, **item}
        for group in ("flip_1to0", "control_both_true")
        for item in data[group]
    ]


def downsample_flow(flow, height=16, width=16):
    old_h, old_w = flow.shape[-2:]
    resized = F.interpolate(flow.float(), size=(height, width), mode="bilinear", align_corners=False)
    resized[:, 0] /= old_w / width
    resized[:, 1] /= old_h / height
    return resized


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


def spearman(left, right):
    left = np.asarray(left).reshape(-1)
    right = np.asarray(right).reshape(-1)
    if np.std(left) < 1.0e-12 or np.std(right) < 1.0e-12:
        return float("nan")
    return float(np.corrcoef(rankdata(left), rankdata(right))[0, 1])


def direction_cosine(left, right):
    left = np.asarray(left).reshape(-1, 2)
    right = np.asarray(right).reshape(-1, 2)
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    valid = denominator > 1.0e-6
    if not valid.any():
        return float("nan")
    return float(np.mean((left[valid] * right[valid]).sum(-1) / denominator[valid]))


def fixed_distribution(feature, temperature):
    feature = F.normalize(feature.detach().float(), dim=1, eps=1.0e-6)
    _, channels, frames, height, width = feature.shape
    current = feature[0, :, :-1].permute(1, 2, 3, 0).reshape(frames - 1, height * width, channels)
    following = feature[0, :, 1:].permute(1, 0, 2, 3)
    neighbours = F.unfold(following, kernel_size=3, padding=1)
    neighbours = neighbours.reshape(frames - 1, channels, 9, height * width).permute(0, 3, 2, 1)
    valid = F.unfold(following.new_ones(frames - 1, 1, height, width), 3, padding=1)
    valid = valid.reshape(frames - 1, 9, height * width).permute(0, 2, 1).bool()
    logits = (current[:, :, None] * neighbours).sum(-1) / temperature
    probs = torch.softmax(logits.masked_fill(~valid, torch.finfo(logits.dtype).min), dim=-1)
    yy, xx = torch.meshgrid(torch.arange(-1, 2), torch.arange(-1, 2), indexing="ij")
    offsets = torch.stack([xx.flatten(), yy.flatten()], dim=-1).to(probs).float()
    expected = torch.einsum("bpk,kd->bpd", probs, offsets)
    return probs.reshape(frames - 1, height, width, 9), expected.reshape(frames - 1, height, width, 2)


def flow_centered_distribution(feature, flow, temperature):
    feature = F.normalize(feature.detach().float(), dim=1, eps=1.0e-6)
    _, channels, frames, height, width = feature.shape
    current = feature[0, :, :-1].permute(1, 2, 3, 0)
    following = feature[0, :, 1:].permute(1, 0, 2, 3)
    yy, xx = torch.meshgrid(
        torch.arange(height, device=feature.device),
        torch.arange(width, device=feature.device), indexing="ij"
    )
    base = torch.stack([xx, yy], dim=-1).float()[None].expand(frames - 1, -1, -1, -1)
    oy, ox = torch.meshgrid(
        torch.arange(-1, 2, device=feature.device),
        torch.arange(-1, 2, device=feature.device), indexing="ij"
    )
    offsets = torch.stack([ox.flatten(), oy.flatten()], dim=-1).float()
    coords = base[:, None] + flow.permute(0, 2, 3, 1)[:, None] + offsets[None, :, None, None]
    valid = (
        (coords[..., 0] >= 0) & (coords[..., 0] <= width - 1)
        & (coords[..., 1] >= 0) & (coords[..., 1] <= height - 1)
    )
    grid = coords.clone()
    grid[..., 0] = 2.0 * grid[..., 0] / max(width - 1, 1) - 1.0
    grid[..., 1] = 2.0 * grid[..., 1] / max(height - 1, 1) - 1.0
    sampled = F.grid_sample(
        following[:, None].expand(-1, 9, -1, -1, -1).reshape(-1, channels, height, width),
        grid.reshape(-1, height, width, 2), mode="bilinear", padding_mode="zeros", align_corners=True,
    ).reshape(frames - 1, 9, channels, height, width).permute(0, 3, 4, 1, 2)
    logits = (current[..., None, :] * sampled).sum(-1) / temperature
    probs = torch.softmax(logits.masked_fill(~valid.permute(0, 2, 3, 1), torch.finfo(logits.dtype).min), dim=-1)
    residual = torch.einsum("bhwk,kd->bhwd", probs, offsets)
    top_residual = offsets[probs.argmax(-1)]
    entropy = -(probs.clamp_min(1.0e-12) * probs.clamp_min(1.0e-12).log()).sum(-1)
    return probs, residual, top_residual, entropy


def load_raft(path, device):
    args = Namespace(model=str(path), small=False, mixed_precision=False, alternate_corr=False)
    model = RAFT(args)
    state = torch.load(path, map_location="cpu")
    model.load_state_dict({key.replace("module.", ""): value for key, value in state.items()})
    return model.to(device).eval()


def load_video_frames(path, device):
    capture = cv2.VideoCapture(str(path))
    fps = capture.get(cv2.CAP_PROP_FPS)
    interval = max(1, round(fps / 8))
    frames = []
    while capture.isOpened():
        ok, frame = capture.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(torch.from_numpy(frame).permute(2, 0, 1).float()[None])
    capture.release()
    return [frame.to(device) for frame in frames[::interval]]


def sample_vector(field, coords):
    height, width = field.shape[-2:]
    grid = coords.clone()
    grid[..., 0] = 2.0 * grid[..., 0] / max(width - 1, 1) - 1.0
    grid[..., 1] = 2.0 * grid[..., 1] / max(height - 1, 1) - 1.0
    return F.grid_sample(field, grid[None], mode="bilinear", padding_mode="zeros", align_corners=True)[0]


def raft_reliability(model, frames, forward_flow):
    cycle_errors, reliable, valid_ratios, photo_errors, elapsed = [], [], [], [], []
    for index, (first, second) in enumerate(zip(frames[:-1], frames[1:])):
        padder = InputPadder(first.shape)
        second_pad, first_pad = padder.pad(second, first)
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            _, backward = model(second_pad, first_pad, iters=20, test_mode=True)
        torch.cuda.synchronize()
        elapsed.append(time.perf_counter() - start)
        backward = backward[0, :, : first.shape[-2], : first.shape[-1]].float()
        forward = forward_flow[index].to(first.device).float()
        height, width = forward.shape[-2:]
        yy, xx = torch.meshgrid(torch.arange(height, device=first.device), torch.arange(width, device=first.device), indexing="ij")
        base = torch.stack([xx, yy], dim=-1).float()
        endpoint = base + forward.permute(1, 2, 0)
        valid = (
            (endpoint[..., 0] >= 0) & (endpoint[..., 0] <= width - 1)
            & (endpoint[..., 1] >= 0) & (endpoint[..., 1] <= height - 1)
        )
        backward_at_endpoint = sample_vector(backward[None], endpoint).permute(1, 2, 0)
        cycle = forward.permute(1, 2, 0) + backward_at_endpoint
        cycle2 = cycle.square().sum(-1)
        scale = forward.permute(1, 2, 0).square().sum(-1) + backward_at_endpoint.square().sum(-1)
        good = valid & (cycle2 <= 0.01 * scale + 0.5)
        warped_second = sample_vector((second / 255.0), endpoint)
        photo = (first[0] / 255.0 - warped_second).abs().mean(0)
        cycle_errors.append(cycle.square().sum(-1).sqrt()[valid].mean().item())
        reliable.append(good.float().mean().item())
        valid_ratios.append(valid.float().mean().item())
        photo_errors.append(photo[good].mean().item() if good.any() else float("nan"))
    return {
        "cycle_error_px_mean": float(np.mean(cycle_errors)),
        "forward_backward_reliable_fraction": float(np.mean(reliable)),
        "in_frame_endpoint_fraction": float(np.mean(valid_ratios)),
        "photometric_mae_on_reliable": float(np.nanmean(photo_errors)),
        "backward_raft_seconds_per_pair": float(np.mean(elapsed)),
    }


def guided_mtd(pred, target, flow, temperature=0.07):
    pred = pred.float()[:, :4]
    target = target.detach().float()[:, :4]
    batch, _, frames, _, _ = pred.shape
    pred_pool = F.adaptive_avg_pool2d(pred.permute(0, 2, 1, 3, 4).reshape(batch * frames, 4, 64, 64), (16, 16)).reshape(batch, frames, 4, 16, 16).permute(0, 2, 1, 3, 4)
    target_pool = F.adaptive_avg_pool2d(target.permute(0, 2, 1, 3, 4).reshape(batch * frames, 4, 64, 64), (16, 16)).reshape(batch, frames, 4, 16, 16).permute(0, 2, 1, 3, 4)
    local_losses, motion_losses = [], []
    for batch_index in range(batch):
        target_probs, target_residual, _, _ = flow_centered_distribution(target_pool[batch_index:batch_index + 1], flow[batch_index], temperature)
        pred_probs, pred_residual, _, _ = flow_centered_distribution(pred_pool[batch_index:batch_index + 1], flow[batch_index], temperature)
        local_losses.append(F.kl_div(pred_probs.clamp_min(1e-8).log(), target_probs, reduction="batchmean") / (15 * 16 * 16))
        motion_losses.append(F.smooth_l1_loss(pred_residual, target_residual))
    pred_summary = F.normalize(pred.mean(dim=(-1, -2)).transpose(1, 2), dim=-1, eps=1e-6)
    target_summary = F.normalize(target.mean(dim=(-1, -2)).transpose(1, 2), dim=-1, eps=1e-6)
    global_loss = F.kl_div(
        F.log_softmax(pred_summary @ pred_summary.transpose(1, 2), dim=-1),
        F.softmax(target_summary @ target_summary.transpose(1, 2), dim=-1), reduction="batchmean"
    )
    return torch.stack(local_losses).mean() + torch.stack(motion_losses).mean() + 0.1 * global_loss


def benchmark_loss(payload, flow, device, batch, warmup, repeats):
    target = payload["fp_cond"].to(device).repeat(batch, 1, 1, 1, 1)
    source = payload["quant_cond"].to(device).repeat(batch, 1, 1, 1, 1)
    flow = flow.to(device).repeat(batch, 1, 1, 1, 1)
    config = {"enabled": True, "noise_channels": 4, "transport_size": 16, "temperature": 0.07,
              "total_weight": 1.0, "local_transport_weight": 1.0,
              "motion_residual_weight": 1.0, "global_relation_weight": 0.1}

    def run(kind):
        pred = source.detach().clone().requires_grad_(True)
        loss = motion_transport_distillation(pred, target, config) if kind == "fixed" else guided_mtd(pred, target, flow)
        loss.backward()

    result = {}
    for kind in ("fixed", "raft_guided"):
        for _ in range(warmup):
            run(kind)
        torch.cuda.synchronize()
        start_event, end_event = torch.cuda.Event(True), torch.cuda.Event(True)
        wall_start = time.perf_counter()
        start_event.record()
        for _ in range(repeats):
            run(kind)
        end_event.record()
        torch.cuda.synchronize()
        result[kind] = {
            "cuda_ms_forward_backward": start_event.elapsed_time(end_event) / repeats,
            "wall_ms_forward_backward": (time.perf_counter() - wall_start) * 1000.0 / repeats,
        }
    return result


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    items = manifest_items(args.manifest)
    rows = []
    first_payload = first_flow = None
    for item in items:
        prompt = item["prompt"]
        prompt_dir = args.feature_dir / f"{int(item['index']):03d}_{safe_name(prompt)}"
        flow_path = args.flow_dir / f"{safe_name(prompt)}_mtd.pt"
        flow = downsample_flow(torch.load(flow_path, map_location="cpu")).to(device)
        for feature_path in sorted(prompt_dir.glob("step_*.pt")):
            payload = torch.load(feature_path, map_location="cpu")
            feature = payload[f"fp_{args.feature_view}_pooled"].to(device)
            _, expected = fixed_distribution(feature, args.temperature)
            _, centered_residual, centered_top, centered_entropy = flow_centered_distribution(feature, flow, args.temperature)
            expected_np = expected.cpu().numpy()
            flow_np = flow.permute(0, 2, 3, 1).cpu().numpy()
            rows.append({
                "group": item["group"], "prompt": prompt,
                "sampling_progress": int(payload["sampling_progress"]),
                "fixed_expected_vs_raft_epe": float(np.linalg.norm(expected_np - flow_np, axis=-1).mean()),
                "fixed_direction_cosine": direction_cosine(expected_np, flow_np),
                "fixed_magnitude_spearman": spearman(np.linalg.norm(expected_np, axis=-1), np.linalg.norm(flow_np, axis=-1)),
                "raft_endpoint_inside_original_3x3": float((np.abs(flow_np).max(axis=-1) <= 1.0).mean()),
                "flow_centered_expected_residual": float(centered_residual.square().sum(-1).sqrt().mean()),
                "flow_centered_top1_residual": float(centered_top.square().sum(-1).sqrt().mean()),
                "flow_centered_center_top1_rate": float((centered_top == 0).all(-1).float().mean()),
                "flow_centered_entropy": float(centered_entropy.mean()),
            })
            if first_payload is None:
                first_payload, first_flow = payload, flow.cpu()

    feature_summary = {
        key: float(np.nanmean([row[key] for row in rows]))
        for key in rows[0] if key not in {"group", "prompt", "sampling_progress"}
    }
    raft = load_raft(args.raft_model, device)
    reliability_rows = []
    for item in items:
        prompt = item["prompt"]
        frames = load_video_frames(args.video_dir / f"{prompt}.mp4", device)
        forward = torch.load(args.flow_dir / f"{safe_name(prompt)}_mtd.pt", map_location="cpu")
        reliability_rows.append({"group": item["group"], "prompt": prompt, **raft_reliability(raft, frames, forward)})
    reliability_summary = {
        key: float(np.nanmean([row[key] for row in reliability_rows]))
        for key in reliability_rows[0] if key not in {"group", "prompt"}
    }
    timing = benchmark_loss(first_payload, first_flow, device, args.timing_batch, args.timing_warmup, args.timing_repeats)
    output = {
        "experiment": {"device": args.device, "video_count": len(items), "feature_records": len(rows),
                       "feature_view": args.feature_view, "timing_batch": args.timing_batch},
        "teacher_feature_vs_raft": feature_summary,
        "raft_reliability": reliability_summary,
        "loss_timing": timing,
        "loss_cuda_overhead_ms": timing["raft_guided"]["cuda_ms_forward_backward"] - timing["fixed"]["cuda_ms_forward_backward"],
        "per_feature": rows,
        "per_video_raft": reliability_rows,
    }
    (args.output / "summary.json").write_text(json.dumps(output, indent=2))
    print(json.dumps({key: value for key, value in output.items() if not key.startswith("per_")}, indent=2))


if __name__ == "__main__":
    main()
