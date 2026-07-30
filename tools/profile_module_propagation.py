#!/usr/bin/env python3
"""Measure equal-energy propagation of real module-group W4A6 errors."""

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
from tools.profile_equal_energy_propagation import normalize_direction, run_step
from tools.profile_stage_numeric_mechanism import (
    build_quant_model,
    error_metrics,
    prepare_conditioning,
    quantizable_module_groups,
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
    parser.add_argument("--selected-progress", default="5,15,25,45")
    parser.add_argument("--module-groups", default="ffn,late_blocks,spatial_attention")
    parser.add_argument("--relative-energy", type=float, default=0.01)
    parser.add_argument("--time-mp-config-weight", required=True)
    parser.add_argument("--time-mp-config-act", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


class ModulePerturbedForward:
    def __init__(self, qnn, cfg_scale, module_group, module_names, rho, progress):
        self.qnn = qnn
        self.cfg_scale = cfg_scale
        self.module_group = module_group
        self.module_names = module_names
        self.rho = rho
        self.progress = progress
        self.used = False
        self.local = None

    def __call__(self, x, timestep, y, **kwargs):
        set_quant_mode(self.qnn, False, False)
        fp_out = forward_with_cfg(
            self.qnn, x, timestep, y, self.cfg_scale, return_trajectory=False, **kwargs
        )
        if self.used:
            return fp_out

        self.qnn.set_layer_quant(
            model=self.qnn,
            module_name_list=self.module_names,
            quant_level="per_layer",
            weight_quant=True,
            act_quant=True,
            prefix="",
        )
        quant_out = forward_with_cfg(
            self.qnn, x, timestep, y, self.cfg_scale, return_trajectory=False, **kwargs
        )
        set_quant_mode(self.qnn, False, False)

        fp_eps_half = fp_out[:1, :3]
        raw_half = quant_out[:1, :3] - fp_eps_half
        perturb_half_double, scale = normalize_direction(raw_half, fp_eps_half, self.rho)
        perturb_half = perturb_half_double.to(fp_eps_half.dtype)
        perturb = torch.cat([perturb_half, perturb_half], dim=0)
        perturbed = fp_out.clone()
        perturbed[:, :3] += perturb
        achieved = float(
            torch.linalg.vector_norm(perturb_half.double())
            / torch.linalg.vector_norm(fp_eps_half.double()).clamp_min(1e-12)
        )
        self.local = {
            "module_group": self.module_group,
            "num_quant_layers": len(self.module_names),
            "progress": self.progress,
            "original_timestep": int(timestep[0]),
            "rho_requested": self.rho,
            "rho_achieved": achieved,
            "raw_direction_relative_l2": float(
                torch.linalg.vector_norm(raw_half.double())
                / torch.linalg.vector_norm(fp_eps_half.double()).clamp_min(1e-12)
            ),
            "normalization_scale": scale,
        }
        self.used = True
        return perturbed


def main():
    args = parse_args()
    outdir = Path(args.output_dir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    selected = sorted({int(x) for x in args.selected_progress.split(",") if x.strip()})
    requested_groups = [x.strip() for x in args.module_groups.split(",") if x.strip()]
    cfg = Config.fromfile(args.config)
    cfg.multi_resolution = cfg.get("multi_resolution", False)
    dtype = to_torch_dtype(cfg.dtype)
    device = torch.device("cuda")
    set_random_seed(args.seed)
    torch.set_grad_enabled(False)

    prompts = [line.strip() for line in Path(args.prompt_path).read_text().splitlines() if line.strip()]
    scheduler, latent_size, qnn = build_quant_model(args, cfg, device, dtype)
    all_groups = quantizable_module_groups(qnn)
    missing = [name for name in requested_groups if name not in all_groups]
    if missing:
        raise KeyError(f"Unknown or empty module groups: {missing}")
    groups = {name: all_groups[name] for name in requested_groups}
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
    torch.save(fp_final.float().cpu(), outdir / "fp_final_latent.pt")

    rows = []
    for progress in selected:
        target_internal = scheduler.num_timesteps - progress
        for group_name, module_names in groups.items():
            x = snapshots[progress].clone()
            perturb_model = ModulePerturbedForward(
                qnn, cfg.scheduler.cfg_scale, group_name, module_names, args.relative_energy, progress
            )
            for internal_timestep in range(target_internal, -1, -1):
                model_fn = perturb_model if internal_timestep == target_internal else fp_model
                x = run_step(scheduler, model_fn, x, internal_timestep, conditioning)
            if perturb_model.local is None:
                raise RuntimeError(f"No perturbation for {group_name} at {progress}")
            metrics = error_metrics(x[:1], fp_final)
            row = {
                **perturb_model.local,
                **{f"final_{key}": value for key, value in metrics.items()},
            }
            row["propagation_gain_relative_l2"] = row["final_relative_l2"] / row["rho_achieved"]
            row["linear_risk_proxy"] = (
                row["raw_direction_relative_l2"] * row["propagation_gain_relative_l2"]
            )
            rows.append(row)
            print(json.dumps(row), flush=True)

    with (outdir / "module_propagation_results.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    metadata = {
        "prompt_index": args.prompt_index,
        "prompt": prompts[args.prompt_index],
        "seed": args.seed,
        "selected_progress": selected,
        "module_groups": requested_groups,
        "module_group_sizes": {name: len(names) for name, names in groups.items()},
        "relative_energy": args.relative_energy,
        "all_values_finite": all(
            math.isfinite(v) for row in rows for v in row.values() if isinstance(v, float)
        ),
    }
    (outdir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    fig, ax = plt.subplots(figsize=(8, 5))
    for group_name in requested_groups:
        subset = sorted((r for r in rows if r["module_group"] == group_name), key=lambda r: r["progress"])
        ax.plot(
            [r["progress"] for r in subset],
            [r["linear_risk_proxy"] for r in subset],
            marker="o",
            label=group_name,
        )
    ax.set_xlabel("Sampling progress")
    ax.set_ylabel("Module propagation-risk proxy")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "module_propagation_risk_curve.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
