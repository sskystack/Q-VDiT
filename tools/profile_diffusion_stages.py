#!/usr/bin/env python3
"""Profile the full-precision diffusion trajectory without changing sampling.

This script intentionally uses the repository's existing IDDPM/DDIM implementation
and records selected ``pred_xstart`` tensors returned by the scheduler.  It is a
data-collection tool, not an implementation of a phase-aware quantization method.
"""

import argparse
import json
import math
import os
import subprocess
import time
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from mmengine.config import Config
from mmengine.runner import set_random_seed
from PIL import Image, ImageDraw

from opensora.datasets import save_sample
from opensora.registry import MODELS, SCHEDULERS, build_module
from opensora.schedulers.iddpm import forward_with_cfg
from opensora.utils.misc import to_torch_dtype


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--prompt-path", required=True)
    parser.add_argument("--prompt-index", type=int, default=0)
    parser.add_argument("--text-embeds", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--selected-progress",
        default="1,10,20,30,40,50,60,70,80,90,95,100",
        help="1-based DDIM update numbers to preserve and decode",
    )
    parser.add_argument("--no-decode", action="store_true")
    return parser.parse_args()


def tensor_stats(x):
    x = x.detach().float()
    flat = x.abs().reshape(-1)
    return {
        "mean": float(x.mean()),
        "std": float(x.std()),
        "rms": float(torch.sqrt(torch.mean(x.square()))),
        "abs_max": float(flat.max()),
        "abs_p99": float(torch.quantile(flat, 0.99)),
        "abs_p999": float(torch.quantile(flat, 0.999)),
        "finite_ratio": float(torch.isfinite(x).float().mean()),
    }


def relative_l2(a, b, eps=1e-12):
    """Return ||a-b||_2 / ||b||_2."""
    a = a.detach().float()
    b = b.detach().float()
    return float(torch.linalg.vector_norm(a - b) / (torch.linalg.vector_norm(b) + eps))


def cosine(a, b, eps=1e-12):
    # Accumulate in FP64. These videos contain >10M values and FP32 dot
    # accumulation can otherwise produce a self-cosine slightly above one.
    a = a.detach().double().reshape(-1)
    b = b.detach().double().reshape(-1)
    value = torch.dot(a, b) / (torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b) + eps)
    return float(value.clamp(-1.0, 1.0))


def normalized_mse(a, b, eps=1e-12):
    a = a.detach().float()
    b = b.detach().float()
    return float(torch.mean((a - b).square()) / (torch.mean(b.square()) + eps))


def spatial_lowpass(x, factor=16):
    # x: [C,T,H,W]. Pool only spatial dimensions and keep a compact layout map.
    c, t, h, w = x.shape
    y = x.permute(1, 0, 2, 3)
    kernel = max(1, min(factor, h, w))
    y = F.avg_pool2d(y, kernel_size=kernel, stride=kernel)
    return y.permute(1, 0, 2, 3).contiguous()


def spatial_highpass(x, factor=16):
    c, t, h, w = x.shape
    y = x.permute(1, 0, 2, 3)
    kernel = max(1, min(factor, h, w))
    low = F.avg_pool2d(y, kernel_size=kernel, stride=kernel)
    low = F.interpolate(low, size=(h, w), mode="bilinear", align_corners=False)
    return (y - low).permute(1, 0, 2, 3).contiguous()


def decoded_metrics(decoded, final_decoded):
    x = decoded.detach().float().cpu()
    ref = final_decoded.detach().float().cpu()
    low_x, low_ref = spatial_lowpass(x), spatial_lowpass(ref)
    high_x, high_ref = spatial_highpass(x), spatial_highpass(ref)
    dx, dref = x[:, 1:] - x[:, :-1], ref[:, 1:] - ref[:, :-1]
    return {
        "pixel_cosine_to_final": cosine(x, ref),
        "pixel_nmse_to_final": normalized_mse(x, ref),
        "layout_lowpass_cosine_to_final": cosine(low_x, low_ref),
        "layout_lowpass_nmse_to_final": normalized_mse(low_x, low_ref),
        "detail_highpass_cosine_to_final": cosine(high_x, high_ref),
        "detail_highpass_nmse_to_final": normalized_mse(high_x, high_ref),
        "high_frequency_energy": float(torch.mean(high_x.square())),
        "high_frequency_energy_ratio_to_final": float(
            torch.mean(high_x.square()) / (torch.mean(high_ref.square()) + 1e-12)
        ),
        "temporal_difference_cosine_to_final": cosine(dx, dref),
        "temporal_difference_nmse_to_final": normalized_mse(dx, dref),
        "temporal_difference_energy": float(torch.mean(dx.square())),
        "temporal_difference_energy_ratio_to_final": float(
            torch.mean(dx.square()) / (torch.mean(dref.square()) + 1e-12)
        ),
    }


def to_uint8_frame(video, frame_idx):
    frame = video[:, frame_idx].detach().float().cpu().clamp(-1, 1)
    frame = ((frame + 1.0) * 127.5).round().clamp(0, 255).byte()
    return frame.permute(1, 2, 0).numpy()


def save_montage(thumbnails, labels, frame_numbers, path, thumb_width=192):
    # thumbnails: list of lists; outer dimension is selected progress, inner is frames.
    rows = len(thumbnails[0])
    cols = len(thumbnails)
    src_h, src_w = thumbnails[0][0].shape[:2]
    thumb_height = round(src_h * thumb_width / src_w)
    label_height = 28
    canvas = Image.new("RGB", (cols * thumb_width, rows * (thumb_height + label_height)), "white")
    draw = ImageDraw.Draw(canvas)
    for col, (column_frames, label) in enumerate(zip(thumbnails, labels)):
        for row, frame in enumerate(column_frames):
            image = Image.fromarray(frame).resize((thumb_width, thumb_height), Image.Resampling.LANCZOS)
            y = row * (thumb_height + label_height)
            canvas.paste(image, (col * thumb_width, y + label_height))
            draw.text((col * thumb_width + 4, y + 5), f"{label} | frame {frame_numbers[row]}", fill="black")
    canvas.save(path)


def save_curve_plots(decoded_records, trajectory_records, plots_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = [r["sampling_progress"] for r in decoded_records]
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=180)
    for key, label in [
        ("layout_lowpass_cosine_to_final", "layout: low-pass cosine"),
        ("detail_highpass_cosine_to_final", "detail: high-pass cosine"),
        ("temporal_difference_cosine_to_final", "temporal-difference cosine"),
        ("pixel_cosine_to_final", "pixel cosine"),
    ]:
        ax.plot(x, [r[key] for r in decoded_records], marker="o", linewidth=2, label=label)
    ax.set(xlabel="DDIM update progress (1 = first denoising update)", ylabel="Similarity to final video", ylim=(-0.03, 1.03))
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(plots_dir / "decoded_proxy_similarity_curves.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=180)
    ax.semilogy(x, [r["high_frequency_energy_ratio_to_final"] for r in decoded_records], marker="o", label="high-frequency energy / final")
    ax.semilogy(x, [r["temporal_difference_energy_ratio_to_final"] for r in decoded_records], marker="o", label="temporal-difference energy / final")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set(xlabel="DDIM update progress", ylabel="Energy ratio (log scale)")
    ax.grid(alpha=0.25, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "decoded_proxy_energy_curves.png")
    plt.close(fig)

    tx = [r["sampling_progress"] for r in trajectory_records]
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=180)
    ax.plot(tx, [r["relative_latent_update"] for r in trajectory_records], label=r"$||x_{t-1}-x_t||/||x_t||$")
    ax.plot(
        tx[1:],
        [r["pred_xstart_change_from_previous_step"] for r in trajectory_records[1:]],
        label=r"change in predicted $x_0$",
    )
    ax.set(xlabel="DDIM update progress", ylabel="Relative L2 change")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "latent_update_curves.png")
    plt.close(fig)


def git_metadata(repo):
    def run(*args):
        try:
            return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()
        except Exception as exc:
            return f"unavailable: {exc}"

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "status_porcelain": run("status", "--short"),
    }


def prepare_conditioning(embeds_path, prompt_index, device, dtype):
    loaded = torch.load(embeds_path, map_location="cpu")
    model_args = loaded.copy()
    y = model_args["y"]
    mask = model_args["mask"]
    if prompt_index >= y.shape[0]:
        raise IndexError(f"prompt-index {prompt_index} exceeds embedding batch {y.shape[0]}")
    # Same indexing/reshape used by IDDPM.sample in this repository.
    model_args["y"] = y[prompt_index : prompt_index + 1].permute(1, 0, 2, 3, 4).reshape(
        -1, y.shape[2], y.shape[3], y.shape[4]
    )
    model_args["mask"] = mask[prompt_index : prompt_index + 1]
    model_args["y"] = model_args["y"].to(device=device, dtype=dtype)
    model_args["mask"] = model_args["mask"].to(device=device)
    return model_args


def main():
    args = parse_args()
    start_time = time.time()
    outdir = Path(args.output_dir).resolve()
    latent_dir = outdir / "pred_xstart"
    decoded_dir = outdir / "decoded_x0"
    plots_dir = outdir / "plots"
    videos_dir = outdir / "videos"
    for directory in (outdir, latent_dir, decoded_dir, plots_dir, videos_dir):
        directory.mkdir(parents=True, exist_ok=True)

    selected_progress = sorted({int(x) for x in args.selected_progress.split(",") if x.strip()})
    cfg = Config.fromfile(args.config)
    cfg.multi_resolution = cfg.get("multi_resolution", False)
    dtype = to_torch_dtype(cfg.dtype)
    device = torch.device("cuda")
    set_random_seed(args.seed)
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    prompts = [line.strip() for line in Path(args.prompt_path).read_text().splitlines() if line.strip()]
    prompt = prompts[args.prompt_index]
    scheduler = build_module(cfg.scheduler, SCHEDULERS)
    input_size = (cfg.num_frames, *cfg.image_size)
    vae = build_module(cfg.vae, MODELS)
    latent_size = vae.get_latent_size(input_size)
    model = build_module(
        cfg.model,
        MODELS,
        input_size=latent_size,
        in_channels=vae.out_channels,
        caption_channels=4096,
        model_max_length=cfg.text_encoder.model_max_length,
        dtype=dtype,
    )
    vae = vae.to(device, dtype).eval()
    model = model.to(device, dtype).eval()

    model_args = prepare_conditioning(args.text_embeds, args.prompt_index, device, dtype)
    if cfg.multi_resolution:
        image_size = cfg.image_size
        model_args["data_info"] = {
            "ar": torch.tensor([[image_size[0] / image_size[1]]], device=device, dtype=dtype),
            "hw": torch.tensor([image_size], device=device, dtype=dtype),
        }

    z_size = (vae.out_channels, *latent_size)
    init_noise_single = torch.randn(1, *z_size, device=device, dtype=dtype)
    z = torch.cat([init_noise_single, init_noise_single], dim=0)
    forward = partial(forward_with_cfg, model, cfg_scale=cfg.scheduler.cfg_scale, return_trajectory=False)
    generator = scheduler.ddim_sample_loop_progressive(
        forward,
        z.shape,
        noise=z,
        clip_denoised=False,
        model_kwargs=model_args,
        progress=True,
        return_trajectory=False,
        device=device,
    )

    trajectory_path = outdir / "trajectory.jsonl"
    trajectory_records = []
    selected_latents = {}
    previous_pred = None
    current_x = z
    total_steps = scheduler.num_timesteps
    with trajectory_path.open("w") as handle:
        for sampling_index, result in enumerate(generator):
            progress = sampling_index + 1
            respaced_timestep = total_steps - progress
            original_timestep = int(scheduler.timestep_map[respaced_timestep])
            x_t = current_x[:1]
            x_next = result["sample"][:1]
            pred_xstart = result["pred_xstart"][:1]
            record = {
                "sampling_progress": progress,
                "sampling_progress_fraction": progress / total_steps,
                "respaced_timestep": respaced_timestep,
                "original_timestep": original_timestep,
                "x_t": tensor_stats(x_t),
                "pred_xstart": tensor_stats(pred_xstart),
                "relative_latent_update": relative_l2(x_next, x_t),
                "pred_xstart_change_from_previous_step": (
                    None if previous_pred is None else relative_l2(pred_xstart, previous_pred)
                ),
                "gpu_memory_allocated_gib": torch.cuda.memory_allocated() / (1024**3),
                "gpu_memory_reserved_gib": torch.cuda.memory_reserved() / (1024**3),
            }
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            trajectory_records.append(record)
            if progress in selected_progress:
                cpu_pred = pred_xstart.detach().float().cpu()
                selected_latents[progress] = cpu_pred
                torch.save(
                    {
                        "pred_xstart": cpu_pred,
                        "sampling_progress": progress,
                        "respaced_timestep": respaced_timestep,
                        "original_timestep": original_timestep,
                    },
                    latent_dir / f"progress_{progress:03d}_t{original_timestep:04d}.pt",
                )
            previous_pred = pred_xstart.detach().clone()
            current_x = result["sample"]

    final_latent = current_x[:1].detach().float().cpu()
    torch.save(final_latent, outdir / "final_latent.pt")

    decoded_records = []
    if not args.no_decode:
        # Free the 1.1B denoiser before decoding all selected estimates.
        del generator, forward, model, model_args, current_x, previous_pred
        torch.cuda.empty_cache()

        final_progress = max(selected_latents)
        with torch.no_grad():
            final_decoded = vae.decode(selected_latents[final_progress].to(device=device, dtype=dtype))[0].float().cpu()
        save_sample(final_decoded.clone(), fps=cfg.fps, save_path=str(videos_dir / "final_full_precision"))

        thumbnails = []
        labels = []
        montage_frame_numbers = None
        for progress in sorted(selected_latents):
            latent = selected_latents[progress]
            with torch.no_grad():
                decoded = vae.decode(latent.to(device=device, dtype=dtype))[0].float().cpu()
            respaced_timestep = total_steps - progress
            original_timestep = int(scheduler.timestep_map[respaced_timestep])
            save_sample(
                decoded.clone(),
                fps=cfg.fps,
                save_path=str(decoded_dir / f"progress_{progress:03d}_t{original_timestep:04d}"),
            )
            metrics = decoded_metrics(decoded, final_decoded)
            metrics.update(
                {
                    "sampling_progress": progress,
                    "sampling_progress_fraction": progress / total_steps,
                    "respaced_timestep": respaced_timestep,
                    "original_timestep": original_timestep,
                }
            )
            decoded_records.append(metrics)
            frame_ids = [0, decoded.shape[1] // 2, decoded.shape[1] - 1]
            montage_frame_numbers = [frame_id + 1 for frame_id in frame_ids]
            thumbnails.append([to_uint8_frame(decoded, frame_id) for frame_id in frame_ids])
            labels.append(f"p={progress}, t={original_timestep}")
            del decoded

        with (outdir / "decoded_metrics.jsonl").open("w") as handle:
            for record in decoded_records:
                handle.write(json.dumps(record) + "\n")
        save_montage(thumbnails, labels, montage_frame_numbers, plots_dir / "pred_xstart_stage_montage.png")
        save_curve_plots(decoded_records, trajectory_records, plots_dir)

    metadata = {
        "experiment": "A1_full_precision_fp16_trajectory_smoke",
        "started_at": datetime.fromtimestamp(start_time, timezone.utc).astimezone().isoformat(),
        "finished_at": datetime.now().astimezone().isoformat(),
        "elapsed_seconds": time.time() - start_time,
        "repository": str(Path.cwd()),
        "git": git_metadata(Path.cwd()),
        "config": str(Path(args.config).resolve()),
        "model_checkpoint": cfg.model.from_pretrained,
        "runtime_dtype": cfg.dtype,
        "flash_attention": bool(cfg.model.get("enable_flashattn", False)),
        "quantization": "disabled (unquantized model)",
        "scheduler": cfg.scheduler.type,
        "sampler": "ddim",
        "sampling_steps": total_steps,
        "cfg_scale": cfg.scheduler.cfg_scale,
        "prompt_path": str(Path(args.prompt_path).resolve()),
        "prompt_index": args.prompt_index,
        "prompt": prompt,
        "text_embeds": str(Path(args.text_embeds).resolve()),
        "seed": args.seed,
        "visible_cuda_device": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "physical_gpu": 6,
        "selected_sampling_progress": selected_progress,
        "prediction_type": "epsilon",
        "status": "completed",
        "outputs": {
            "trajectory": str(trajectory_path),
            "decoded_metrics": str(outdir / "decoded_metrics.jsonl"),
            "montage": str(plots_dir / "pred_xstart_stage_montage.png"),
            "final_video": str(videos_dir / "final_full_precision.mp4"),
        },
    }
    with (outdir / "config.json").open("w") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
