#!/usr/bin/env python3
"""Evaluate a calibration checkpoint on one held-out prompt trajectory."""

import argparse
import json
import os
import subprocess
from datetime import datetime
from functools import partial
from pathlib import Path

import torch
import yaml
from mmengine.config import Config
from mmengine.runner import set_random_seed

from opensora.schedulers.iddpm import forward_with_cfg
from opensora.utils.misc import to_torch_dtype
from tools.profile_stage_numeric_mechanism import build_quant_model, prepare_conditioning, set_quant_mode


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--calib-config", required=True)
    parser.add_argument("--quant-ckpt", required=True)
    parser.add_argument("--text-embeds", required=True)
    parser.add_argument("--prompt-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--time-mp-config-weight", required=True)
    parser.add_argument("--time-mp-config-act", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def metrics(value, reference):
    value = value.detach().double()
    reference = reference.detach().double()
    diff = value - reference
    value_flat, reference_flat = value.reshape(-1), reference.reshape(-1)
    return {
        "relative_l2": float(torch.linalg.vector_norm(diff) / torch.linalg.vector_norm(reference).clamp_min(1e-30)),
        "nmse": float(diff.square().sum() / reference.square().sum().clamp_min(1e-30)),
        "cosine": float(
            torch.dot(value_flat, reference_flat)
            / (torch.linalg.vector_norm(value_flat) * torch.linalg.vector_norm(reference_flat)).clamp_min(1e-30)
        ),
        "rmse": float(torch.sqrt(diff.square().mean())),
        "max_abs_error": float(diff.abs().max()),
    }


def git_value(*args):
    try:
        return subprocess.check_output(
            ["git", *args], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


@torch.no_grad()
def sample(scheduler, qnn, z, conditioning, cfg_scale, weight_quant, act_quant):
    set_quant_mode(qnn, weight_quant, act_quant)
    model = partial(forward_with_cfg, qnn, cfg_scale=cfg_scale, return_trajectory=False)
    return scheduler.ddim_sample_loop(
        model,
        z.shape,
        noise=z.clone(),
        clip_denoised=False,
        model_kwargs=conditioning,
        progress=True,
        device=z.device,
    )


def main():
    args = parse_args()
    started_at = datetime.now().astimezone().isoformat()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    cfg = Config.fromfile(args.config)
    with Path(args.calib_config).open() as handle:
        quant_config = yaml.safe_load(handle)
    cfg.multi_resolution = cfg.get("multi_resolution", False)
    dtype = to_torch_dtype(cfg.dtype)
    device = torch.device("cuda")
    set_random_seed(args.seed)
    torch.set_grad_enabled(False)

    scheduler, latent_size, qnn = build_quant_model(args, cfg, device, dtype)
    conditioning = prepare_conditioning(args.text_embeds, args.prompt_index, device, dtype)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    init_noise = torch.randn(1, qnn.in_channels, *latent_size, device=device, generator=generator)
    z = torch.cat([init_noise, init_noise], dim=0)
    fp = sample(scheduler, qnn, z, conditioning, float(cfg.scheduler.cfg_scale), False, False)
    quant = sample(scheduler, qnn, z, conditioning, float(cfg.scheduler.cfg_scale), True, True)
    fp, _ = fp.chunk(2, dim=0)
    quant, _ = quant.chunk(2, dim=0)
    ended_at = datetime.now().astimezone().isoformat()
    result = {
        "started_at": started_at,
        "ended_at": ended_at,
        "completed": True,
        "git_commit": git_value("rev-parse", "HEAD"),
        "git_branch": git_value("branch", "--show-current"),
        "config": str(Path(args.config).resolve()),
        "calibration_config": str(Path(args.calib_config).resolve()),
        "model_checkpoint": str(Path(cfg.model.from_pretrained).resolve()),
        "prompt_index": args.prompt_index,
        "seed": args.seed,
        "gpu_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "runtime_dtype": str(dtype),
        "flash_attention": bool(cfg.model.get("enable_flashattn", False)),
        "weight_bits": int(quant_config["quant"]["weight"]["quantizer"]["n_bits"]),
        "activation_bits": int(quant_config["quant"]["activation"]["quantizer"]["n_bits"]),
        "scheduler": str(cfg.scheduler.type),
        "sampling_steps": int(cfg.scheduler.num_sampling_steps),
        "cfg_scale": float(cfg.scheduler.cfg_scale),
        "quant_checkpoint": str(Path(args.quant_ckpt).resolve()),
        "time_mp_config_weight": str(Path(args.time_mp_config_weight).resolve()),
        "time_mp_config_activation": str(Path(args.time_mp_config_act).resolve()),
        "profiling_operation": "paired FP and W4A6 DDIM trajectories from identical initial noise",
        "output_directory": str(output),
        **metrics(quant, fp),
        "fp_all_finite": bool(torch.isfinite(fp).all()),
        "quant_all_finite": bool(torch.isfinite(quant).all()),
        "initial_noise_checksum": float(init_noise.double().sum()),
    }
    torch.save({"fp_final_latent": fp.cpu(), "quant_final_latent": quant.cpu()}, output / "final_latents.pt")
    (output / "metrics.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
