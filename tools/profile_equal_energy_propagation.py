#!/usr/bin/env python3
"""Measure timestep propagation gain with equal-energy perturbations."""

import argparse
import json
import math
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from mmengine.config import Config
from mmengine.runner import set_random_seed

from opensora.schedulers.iddpm import forward_with_cfg
from opensora.utils.misc import to_torch_dtype
from tools.profile_stage_numeric_mechanism import (
    build_quant_model,
    error_metrics,
    prepare_conditioning,
    set_quant_mode,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--calib-config", required=True)
    parser.add_argument("--quant-ckpt", required=True)
    parser.add_argument("--prompt-path", required=True)
    parser.add_argument("--text-embeds", required=True)
    parser.add_argument("--prompt-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--selected-progress", default="5,15,25,45,65,85,95")
    parser.add_argument("--relative-energy", type=float, default=0.01)
    parser.add_argument("--directions", default="random,w4a6")
    parser.add_argument("--time-mp-config-weight", required=True)
    parser.add_argument("--time-mp-config-act", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def run_step(scheduler, model_fn, x, internal_timestep, conditioning):
    timestep = torch.tensor(
        [internal_timestep] * x.shape[0], device=x.device, dtype=torch.long
    )
    return scheduler.ddim_sample(
        model_fn,
        x,
        timestep,
        clip_denoised=False,
        model_kwargs=conditioning,
        eta=0.0,
    )["sample"]


def normalize_direction(direction, reference, rho):
    direction = direction.double()
    reference = reference.double()
    scale = rho * torch.linalg.vector_norm(reference) / torch.linalg.vector_norm(direction).clamp_min(1e-12)
    return (direction * scale).to(reference.dtype), float(scale)


class PerturbedForward:
    def __init__(self, qnn, cfg_scale, direction_kind, rho, seed, progress):
        self.qnn = qnn
        self.cfg_scale = cfg_scale
        self.direction_kind = direction_kind
        self.rho = rho
        self.seed = seed
        self.progress = progress
        self.used = False
        self.local = None

    def __call__(self, x, timestep, y, **kwargs):
        if self.used:
            set_quant_mode(self.qnn, False, False)
            return forward_with_cfg(
                self.qnn, x, timestep, y, self.cfg_scale, return_trajectory=False, **kwargs
            )

        set_quant_mode(self.qnn, False, False)
        fp_out = forward_with_cfg(
            self.qnn, x, timestep, y, self.cfg_scale, return_trajectory=False, **kwargs
        )
        fp_eps_half = fp_out[:1, :3]

        if self.direction_kind == "w4a6":
            set_quant_mode(self.qnn, True, True)
            quant_out = forward_with_cfg(
                self.qnn, x, timestep, y, self.cfg_scale, return_trajectory=False, **kwargs
            )
            raw_half = quant_out[:1, :3] - fp_eps_half
        elif self.direction_kind == "random":
            generator = torch.Generator(device=x.device)
            generator.manual_seed(self.seed + 10000 + self.progress)
            raw_half = torch.randn(
                fp_eps_half.shape,
                device=x.device,
                dtype=fp_eps_half.dtype,
                generator=generator,
            )
        else:
            raise ValueError(f"Unknown direction: {self.direction_kind}")

        perturb_half_double, scale = normalize_direction(raw_half, fp_eps_half, self.rho)
        perturb_half = perturb_half_double.to(fp_eps_half.dtype)
        perturb = torch.cat([perturb_half, perturb_half], dim=0)
        perturbed = fp_out.clone()
        perturbed[:, :3] = perturbed[:, :3] + perturb
        achieved = float(
            torch.linalg.vector_norm(perturb_half.double())
            / torch.linalg.vector_norm(fp_eps_half.double()).clamp_min(1e-12)
        )
        self.local = {
            "direction": self.direction_kind,
            "progress": self.progress,
            "original_timestep": int(timestep[0]),
            "rho_requested": self.rho,
            "rho_achieved": achieved,
            "raw_direction_relative_l2": float(
                torch.linalg.vector_norm(raw_half.double())
                / torch.linalg.vector_norm(fp_eps_half.double()).clamp_min(1e-12)
            ),
            "normalization_scale": scale,
            "fp_eps_norm": float(torch.linalg.vector_norm(fp_eps_half.double())),
            "perturb_norm": float(torch.linalg.vector_norm(perturb_half.double())),
        }
        self.used = True
        set_quant_mode(self.qnn, False, False)
        return perturbed


def main():
    args = parse_args()
    outdir = Path(args.output_dir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    selected = sorted({int(x) for x in args.selected_progress.split(",") if x.strip()})
    directions = [x.strip() for x in args.directions.split(",") if x.strip()]
    cfg = Config.fromfile(args.config)
    cfg.multi_resolution = cfg.get("multi_resolution", False)
    dtype = to_torch_dtype(cfg.dtype)
    device = torch.device("cuda")
    set_random_seed(args.seed)
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    prompts = [line.strip() for line in Path(args.prompt_path).read_text().splitlines() if line.strip()]
    scheduler, latent_size, qnn = build_quant_model(args, cfg, device, dtype)
    conditioning = prepare_conditioning(args.text_embeds, args.prompt_index, device, dtype)
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    init_noise = torch.randn(1, qnn.in_channels, *latent_size, device=device, generator=generator)
    initial = torch.cat([init_noise, init_noise], dim=0)
    set_quant_mode(qnn, False, False)
    fp_model = partial(
        forward_with_cfg, qnn, cfg_scale=cfg.scheduler.cfg_scale, return_trajectory=False
    )

    snapshots = {}
    x = initial
    for internal_timestep in range(scheduler.num_timesteps - 1, -1, -1):
        progress = scheduler.num_timesteps - internal_timestep
        if progress in selected:
            snapshots[progress] = x.detach().clone()
        x = run_step(scheduler, fp_model, x, internal_timestep, conditioning)
    fp_final = x[:1].detach().clone()
    torch.save(fp_final.float().cpu(), outdir / "fp_final_latent.pt")

    rows = []
    for progress in selected:
        target_internal = scheduler.num_timesteps - progress
        for direction in directions:
            x = snapshots[progress].clone()
            perturbed_forward = PerturbedForward(
                qnn=qnn,
                cfg_scale=cfg.scheduler.cfg_scale,
                direction_kind=direction,
                rho=args.relative_energy,
                seed=args.seed,
                progress=progress,
            )
            for internal_timestep in range(target_internal, -1, -1):
                model_fn = perturbed_forward if internal_timestep == target_internal else fp_model
                x = run_step(scheduler, model_fn, x, internal_timestep, conditioning)
            if not perturbed_forward.used or perturbed_forward.local is None:
                raise RuntimeError(f"Perturbation was not applied at progress {progress}")
            final = x[:1]
            metrics = error_metrics(final, fp_final)
            row = {
                **perturbed_forward.local,
                **{f"final_{key}": value for key, value in metrics.items()},
            }
            row["propagation_gain_relative_l2"] = row["final_relative_l2"] / row["rho_achieved"]
            rows.append(row)
            torch.save(
                final.float().cpu(), outdir / f"final_{direction}_progress_{progress:03d}.pt"
            )
            print(json.dumps(row), flush=True)

    with (outdir / "propagation_results.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    metadata = {
        "prompt_index": args.prompt_index,
        "prompt": prompts[args.prompt_index],
        "seed": args.seed,
        "selected_progress": selected,
        "directions": directions,
        "relative_energy": args.relative_energy,
        "runtime_dtype": str(dtype),
        "cfg_scale": cfg.scheduler.cfg_scale,
        "paired_fp_prefix": True,
        "fp_suffix_after_injection": True,
        "all_values_finite": all(
            math.isfinite(v) for row in rows for v in row.values() if isinstance(v, float)
        ),
    }
    (outdir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    fig, ax = plt.subplots(figsize=(8, 5))
    for direction in directions:
        subset = sorted((r for r in rows if r["direction"] == direction), key=lambda r: r["progress"])
        ax.plot(
            [r["progress"] for r in subset],
            [r["propagation_gain_relative_l2"] for r in subset],
            marker="o",
            label=direction,
        )
    ax.set_xlabel("Sampling progress")
    ax.set_ylabel("Propagation gain: final relative L2 / injected relative energy")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "propagation_gain_curve.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
