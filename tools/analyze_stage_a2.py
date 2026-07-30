#!/usr/bin/env python3
"""Aggregate experiment-A diffusion-stage curves with CLIP and RAFT.

Run this in the server's ``vbench`` environment after all trajectory directories
have completed. Existing layout/detail metrics are read from the original tensor
analysis; semantic and motion curves are computed from decoded stage videos.
"""

import argparse
import csv
import json
import re
import sys
from argparse import Namespace
from collections import defaultdict
from pathlib import Path

import clip
import cv2
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt


RUN_PATTERN = re.compile(r"fp16_fp_trajectory_prompt(?P<prompt>\d+)_seed(?P<seed>\d+)$")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--prompt-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--clip-model", default="/home/zhouchongtian/quantization/models/vbench/clip_model/ViT-L-14.pt")
    parser.add_argument("--raft-root", default="/home/zhouchongtian/quantization/eval/multi_aspects_metrics/third_party/RAFT")
    parser.add_argument("--raft-ckpt", default="/home/zhouchongtian/quantization/models/vbench/raft_model/models/raft-things.pth")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--raft-iters", type=int, default=20)
    parser.add_argument("--raft-batch-size", type=int, default=2)
    return parser.parse_args()


def read_video(path):
    capture = cv2.VideoCapture(str(path))
    frames = []
    while capture.isOpened():
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    if len(frames) < 2:
        raise RuntimeError(f"Could not read enough frames from {path}")
    return frames


def load_raft(raft_root, ckpt, device):
    sys.path.insert(0, str(Path(raft_root) / "core"))
    from raft import RAFT
    from utils.utils import InputPadder

    model_args = Namespace(small=False, mixed_precision=False, alternate_corr=False, dropout=0)
    wrapped = torch.nn.DataParallel(RAFT(model_args))
    wrapped.load_state_dict(torch.load(ckpt, map_location=device))
    model = wrapped.module.to(device).eval()
    model.args.mixed_precision = False
    return model, InputPadder


def clip_video_score(frames, prompt, model, preprocess, device, batch_size=16):
    from PIL import Image

    text = clip.tokenize([prompt], truncate=True).to(device)
    with torch.no_grad():
        text_feature = model.encode_text(text).float()
        text_feature /= text_feature.norm(dim=-1, keepdim=True)
        frame_scores = []
        frame_features = []
        for start in range(0, len(frames), batch_size):
            batch = torch.stack([preprocess(Image.fromarray(frame)) for frame in frames[start : start + batch_size]]).to(device)
            image_features = model.encode_image(batch).float()
            image_features /= image_features.norm(dim=-1, keepdim=True)
            frame_scores.extend((image_features @ text_feature.T).squeeze(1).cpu().tolist())
            frame_features.append(image_features.cpu())
        mean_feature = torch.cat(frame_features).mean(dim=0, keepdim=True)
        mean_feature /= mean_feature.norm(dim=-1, keepdim=True)
        pooled_score = float(mean_feature @ text_feature.cpu().T)
    return {
        "clip_frame_mean": float(np.mean(frame_scores)),
        "clip_frame_std": float(np.std(frame_scores)),
        "clip_pooled_video": pooled_score,
    }


def compute_flows(frames, model, input_padder, device, iters, batch_size):
    images = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float()
    all_flows = []
    with torch.no_grad():
        for start in range(0, len(frames) - 1, batch_size):
            end = min(start + batch_size, len(frames) - 1)
            image1 = images[start:end].to(device)
            image2 = images[start + 1 : end + 1].to(device)
            padder = input_padder(image1.shape)
            image1, image2 = padder.pad(image1, image2)
            _, flow = model(image1, image2, iters=iters, test_mode=True)
            all_flows.append(flow.float().cpu())
    return torch.cat(all_flows, dim=0)


def flow_metrics(flow, final_flow):
    x = flow.double().reshape(-1)
    ref = final_flow.double().reshape(-1)
    cosine = torch.dot(x, ref) / (torch.linalg.vector_norm(x) * torch.linalg.vector_norm(ref) + 1e-12)
    diff = flow.float() - final_flow.float()
    magnitude = torch.linalg.vector_norm(flow.float(), dim=1)
    final_magnitude = torch.linalg.vector_norm(final_flow.float(), dim=1)
    return {
        "flow_cosine_to_final": float(cosine.clamp(-1, 1)),
        "flow_nmse_to_final": float(torch.mean(diff.square()) / (torch.mean(final_flow.float().square()) + 1e-12)),
        "flow_epe_to_final": float(torch.linalg.vector_norm(diff, dim=1).mean()),
        "flow_mean_magnitude": float(magnitude.mean()),
        "flow_magnitude_ratio_to_final": float(magnitude.mean() / (final_magnitude.mean() + 1e-12)),
    }


def load_existing_metrics(run_dir):
    return {
        row["sampling_progress"]: row
        for row in (json.loads(line) for line in (run_dir / "decoded_metrics.jsonl").read_text().splitlines())
    }


def discover_runs(root):
    runs = []
    for path in sorted(Path(root).iterdir()):
        match = RUN_PATTERN.match(path.name)
        if not match or not (path / "config.json").exists() or not (path / "decoded_metrics.jsonl").exists():
            continue
        config = json.loads((path / "config.json").read_text())
        if config.get("status") != "completed":
            continue
        runs.append((path, int(match.group("prompt")), int(match.group("seed")), config))
    return runs


def aggregate(rows):
    numeric_keys = [
        "clip_frame_mean",
        "clip_pooled_video",
        "layout_lowpass_cosine_to_final",
        "detail_highpass_cosine_to_final",
        "flow_cosine_to_final",
        "flow_nmse_to_final",
        "flow_epe_to_final",
        "flow_mean_magnitude",
        "flow_magnitude_ratio_to_final",
    ]
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["sampling_progress"]].append(row)
    output = []
    for progress in sorted(grouped):
        item = {"sampling_progress": progress, "n": len(grouped[progress])}
        for key in numeric_keys:
            values = np.asarray([row[key] for row in grouped[progress]], dtype=np.float64)
            item[f"{key}_mean"] = float(values.mean())
            item[f"{key}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            item[f"{key}_sem"] = float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
        output.append(item)
    return output


def first_threshold(rows, key, threshold=0.9):
    values = sorted((row["sampling_progress"], row[key]) for row in rows)
    for progress, value in values:
        if value >= threshold:
            return progress
    return None


def stability_summary(rows):
    per_run = defaultdict(list)
    for row in rows:
        per_run[(row["prompt_index"], row["seed"])].append(row)
    thresholds = {"layout_p90": [], "detail_p90": [], "flow_p90": []}
    monotonic = defaultdict(list)
    for run_rows in per_run.values():
        run_rows = sorted(run_rows, key=lambda row: row["sampling_progress"])
        thresholds["layout_p90"].append(first_threshold(run_rows, "layout_lowpass_cosine_to_final"))
        thresholds["detail_p90"].append(first_threshold(run_rows, "detail_highpass_cosine_to_final"))
        thresholds["flow_p90"].append(first_threshold(run_rows, "flow_cosine_to_final"))
        for key in ["clip_frame_mean", "layout_lowpass_cosine_to_final", "detail_highpass_cosine_to_final", "flow_cosine_to_final"]:
            sequence = [row[key] for row in run_rows]
            monotonic[key].append(float(np.mean(np.diff(sequence) >= 0)))

    result = {"num_runs": len(per_run), "thresholds": {}, "monotonic_step_fraction": {}}
    for key, values in thresholds.items():
        valid = np.asarray([value for value in values if value is not None], dtype=np.float64)
        result["thresholds"][key] = {
            "found": int(len(valid)),
            "mean": float(valid.mean()) if len(valid) else None,
            "std": float(valid.std(ddof=1)) if len(valid) > 1 else 0.0 if len(valid) else None,
            "min": int(valid.min()) if len(valid) else None,
            "max": int(valid.max()) if len(valid) else None,
            "values": [int(value) if value is not None else None for value in values],
        }
    for key, values in monotonic.items():
        result["monotonic_step_fraction"][key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        }
    return result


def largest_changes(aggregate_rows, key, count=3):
    x = [row["sampling_progress"] for row in aggregate_rows]
    y = np.asarray([row[f"{key}_mean"] for row in aggregate_rows])
    changes = np.diff(y)
    order = np.argsort(np.abs(changes))[::-1][:count]
    return [
        {"from": x[index], "to": x[index + 1], "delta": float(changes[index])}
        for index in order
    ]


def plot_aggregate(agg, output):
    x = np.asarray([row["sampling_progress"] for row in agg])
    panels = [
        ("clip_frame_mean", "Semantic: CLIP video-text score", None),
        ("layout_lowpass_cosine_to_final", "Layout: low-pass similarity to final", (0, 1.03)),
        ("flow_cosine_to_final", "Motion: RAFT-flow similarity to final", (-0.05, 1.03)),
        ("detail_highpass_cosine_to_final", "Detail: high-pass similarity to final", (-0.05, 1.03)),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), dpi=180)
    for ax, (key, title, ylim) in zip(axes.flat, panels):
        mean = np.asarray([row[f"{key}_mean"] for row in agg])
        std = np.asarray([row[f"{key}_std"] for row in agg])
        ax.plot(x, mean, marker="o", linewidth=2)
        ax.fill_between(x, mean - std, mean + std, alpha=0.2, label="±1 std across prompt/seed")
        ax.set_title(title)
        ax.set_xlabel("DDIM update progress")
        ax.grid(alpha=0.25)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.legend()
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    root = Path(args.root)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    prompts = [line.strip() for line in Path(args.prompt_path).read_text().splitlines() if line.strip()]
    runs = discover_runs(root)
    if not runs:
        raise RuntimeError(f"No completed profiling runs found under {root}")

    clip_model, clip_preprocess = clip.load(args.clip_model, device=args.device, jit=False)
    clip_model.eval()
    raft_model, input_padder = load_raft(args.raft_root, args.raft_ckpt, args.device)

    all_rows = []
    for run_number, (run_dir, prompt_index, seed, config) in enumerate(runs, 1):
        print(f"[{run_number}/{len(runs)}] prompt={prompt_index} seed={seed}: {run_dir}", flush=True)
        existing = load_existing_metrics(run_dir)
        stage_files = sorted((run_dir / "decoded_x0").glob("progress_*.mp4"))
        if len(stage_files) != len(existing):
            raise RuntimeError(f"Stage-video count mismatch in {run_dir}: {len(stage_files)} vs {len(existing)}")

        parsed_stages = []
        for video_path in stage_files:
            match = re.search(r"progress_(\d+)_t(\d+)", video_path.stem)
            parsed_stages.append((int(match.group(1)), int(match.group(2)), video_path))
        final_progress, _, final_path = max(parsed_stages)
        final_frames = read_video(final_path)
        final_flow = compute_flows(final_frames, raft_model, input_padder, args.device, args.raft_iters, args.raft_batch_size)

        for stage_number, (progress, original_timestep, video_path) in enumerate(parsed_stages, 1):
            print(f"  stage {stage_number}/{len(parsed_stages)} progress={progress}", flush=True)
            frames = final_frames if progress == final_progress else read_video(video_path)
            flow = final_flow if progress == final_progress else compute_flows(
                frames, raft_model, input_padder, args.device, args.raft_iters, args.raft_batch_size
            )
            row = {
                "run_dir": str(run_dir),
                "prompt_index": prompt_index,
                "seed": seed,
                "sampling_progress": progress,
                "original_timestep": original_timestep,
                "prompt": prompts[prompt_index],
            }
            row.update(existing[progress])
            row.update(clip_video_score(frames, prompts[prompt_index], clip_model, clip_preprocess, args.device))
            row.update(flow_metrics(flow, final_flow))
            all_rows.append(row)
            del flow

        del final_flow
        torch.cuda.empty_cache()

    with (outdir / "per_run_stage_metrics.jsonl").open("w") as handle:
        for row in all_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    agg = aggregate(all_rows)
    write_csv(outdir / "aggregate_curves.csv", agg)
    plot_aggregate(agg, outdir / "four_formation_curves_mean_std.png")
    stability = stability_summary(all_rows)
    changes = {
        key: largest_changes(agg, key)
        for key in ["clip_frame_mean", "layout_lowpass_cosine_to_final", "flow_cosine_to_final", "detail_highpass_cosine_to_final"]
    }
    summary = {
        "num_runs": len(runs),
        "num_prompts": len({prompt_index for _, prompt_index, _, _ in runs}),
        "num_seeds": len({seed for _, _, seed, _ in runs}),
        "runtime_dtype": "fp16",
        "quantization": "disabled",
        "clip_model": args.clip_model,
        "raft_checkpoint": args.raft_ckpt,
        "raft_iterations": args.raft_iters,
        "stability": stability,
        "largest_adjacent_changes": changes,
        "limitations": [
            "CLIP is an image-text proxy averaged over frames, not a complete video-semantic metric.",
            "RAFT on very noisy early pred_xstart frames can be unreliable; early flow values must be interpreted with the decoded montage.",
            "The ten prompts come from one OpenSora sample file and are predominantly natural-scene prompts.",
        ],
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
