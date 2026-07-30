#!/usr/bin/env python3
"""Check whether timestep propagation gain is stable across perturbation sizes."""

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
from tools.profile_equal_energy_propagation import PerturbedForward, run_step
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
    parser.add_argument("--selected-progress", default="5,15,25")
    parser.add_argument("--relative-energies", default="0.005,0.01,0.02")
    parser.add_argument("--directions", default="random,w4a6")
    parser.add_argument("--time-mp-config-weight", required=True)
    parser.add_argument("--time-mp-config-act", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.output_dir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    selected = sorted({int(x) for x in args.selected_progress.split(",") if x.strip()})
    rhos = sorted({float(x) for x in args.relative_energies.split(",") if x.strip()})
    directions = [x.strip() for x in args.directions.split(",") if x.strip()]
    cfg = Config.fromfile(args.config)
    cfg.multi_resolution = cfg.get("multi_resolution", False)
    dtype = to_torch_dtype(cfg.dtype)
    device = torch.device("cuda")
    set_random_seed(args.seed)
    torch.set_grad_enabled(False)

    scheduler, latent_size, qnn = build_quant_model(args, cfg, device, dtype)
    conditioning = prepare_conditioning(args.text_embeds, args.prompt_index, device, dtype)
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    init_noise = torch.randn(1, qnn.in_channels, *latent_size, device=device, generator=generator)
    x = torch.cat([init_noise, init_noise], dim=0)
    set_quant_mode(qnn, False, False)
    fp_model = partial(
        forward_with_cfg, qnn, cfg_scale=cfg.scheduler.cfg_scale, return_trajectory=False
    )
    snapshots = {}
    for internal_timestep in range(scheduler.num_timesteps - 1, -1, -1):
        progress = scheduler.num_timesteps - internal_timestep
        if progress in selected:
            snapshots[progress] = x.detach().clone()
        x = run_step(scheduler, fp_model, x, internal_timestep, conditioning)
    fp_final = x[:1].detach().clone()

    rows = []
    for progress in selected:
        target_internal = scheduler.num_timesteps - progress
        for direction in directions:
            for rho in rhos:
                x = snapshots[progress].clone()
                perturb_model = PerturbedForward(
                    qnn, cfg.scheduler.cfg_scale, direction, rho, args.seed, progress
                )
                for internal_timestep in range(target_internal, -1, -1):
                    model_fn = perturb_model if internal_timestep == target_internal else fp_model
                    x = run_step(scheduler, model_fn, x, internal_timestep, conditioning)
                metrics = error_metrics(x[:1], fp_final)
                row = {
                    **perturb_model.local,
                    **{f"final_{key}": value for key, value in metrics.items()},
                }
                row["propagation_gain_relative_l2"] = row["final_relative_l2"] / row["rho_achieved"]
                rows.append(row)
                print(json.dumps(row), flush=True)

    with (outdir / "linearity_results.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    metadata = {
        "prompt_index": args.prompt_index,
        "seed": args.seed,
        "selected_progress": selected,
        "relative_energies": rhos,
        "directions": directions,
        "all_values_finite": all(
            math.isfinite(v) for row in rows for v in row.values() if isinstance(v, float)
        ),
    }
    (outdir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    fig, axes = plt.subplots(1, len(directions), figsize=(6 * len(directions), 4.5), squeeze=False)
    for axis, direction in zip(axes[0], directions):
        for progress in selected:
            subset = sorted(
                (r for r in rows if r["direction"] == direction and r["progress"] == progress),
                key=lambda r: r["rho_requested"],
            )
            axis.plot(
                [100 * r["rho_requested"] for r in subset],
                [r["propagation_gain_relative_l2"] for r in subset],
                marker="o",
                label=f"progress {progress}",
            )
        axis.set_title(direction)
        axis.set_xlabel("Injected relative energy (%)")
        axis.set_ylabel("Propagation gain")
        axis.grid(alpha=0.25)
        axis.legend()
    fig.tight_layout()
    fig.savefig(outdir / "propagation_linearity.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
