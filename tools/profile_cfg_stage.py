#!/usr/bin/env python3
"""Profile CFG contribution and branch-error alignment across diffusion stages."""

import argparse
import json
import math
from functools import partial
from pathlib import Path

import torch
from mmengine.config import Config
from mmengine.runner import set_random_seed

from opensora.schedulers.iddpm import forward_with_cfg
from opensora.utils.misc import to_torch_dtype
from qdiff.models.quant_layer import QuantLayer
from tools.profile_stage_numeric_mechanism import (
    build_quant_model,
    prepare_conditioning,
    set_quant_mode,
)


def assert_full_precision_state(qnn):
    bad = [
        name
        for name, module in qnn.named_modules()
        if isinstance(module, QuantLayer)
        and (getattr(module, "weight_quant", False) or getattr(module, "act_quant", False))
    ]
    if bad:
        raise RuntimeError(f"Trajectory state leak: quantization remains enabled for {bad[:8]}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--calib-config", required=True)
    parser.add_argument("--quant-ckpt", required=True)
    parser.add_argument("--text-embeds", required=True)
    parser.add_argument("--prompt-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--selected-progress", default="5,15,25,45,65,85,95")
    parser.add_argument("--time-mp-config-weight", required=True)
    parser.add_argument("--time-mp-config-act", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def norm(x):
    return torch.linalg.vector_norm(x.detach().double())


def cosine(a, b):
    a, b = a.detach().double().reshape(-1), b.detach().double().reshape(-1)
    return float(torch.dot(a, b) / (norm(a) * norm(b)).clamp_min(1e-30))


def relative_l2(value, reference):
    return float(norm(value - reference) / norm(reference).clamp_min(1e-30))


@torch.no_grad()
def branch_outputs(qnn, x, timestep, conditioning):
    y = conditioning["y"]
    y = y.reshape([2, y.shape[0] // 2] + list(y.shape[1:]))
    t = timestep.reshape([2, -1])
    y_cond, y_uncond = y.unbind(0)
    t_cond, t_uncond = t.unbind(0)
    half = x[: len(x) // 2]
    outputs = {}
    for name, branch_y, branch_t in (
        ("conditional", y_cond, t_cond),
        ("unconditional", y_uncond, t_uncond),
    ):
        value = qnn.forward(half, branch_t, branch_y, mask=conditioning["mask"])
        value = value["x"] if isinstance(value, dict) else value
        outputs[name] = value[:, :3].detach()
    return outputs


def main():
    args = parse_args()
    selected = sorted({int(x) for x in args.selected_progress.split(",") if x.strip()})
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    cfg = Config.fromfile(args.config)
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
    scale = float(cfg.scheduler.cfg_scale)

    set_quant_mode(qnn, False, False)
    fp_model = partial(forward_with_cfg, qnn, cfg_scale=scale, return_trajectory=False)
    trajectory = scheduler.ddim_sample_loop_progressive(
        fp_model, z.shape, noise=z, clip_denoised=False,
        model_kwargs=conditioning, progress=True, device=device,
    )
    rows = []
    current_x = z
    for sampling_index, result in enumerate(trajectory):
        progress = sampling_index + 1
        x_t = current_x.detach()
        current_x = result["sample"]
        if progress not in selected:
            continue
        internal_timestep = scheduler.num_timesteps - progress
        original_timestep = int(scheduler.timestep_map[internal_timestep])
        timestep = torch.tensor([original_timestep] * x_t.shape[0], device=device, dtype=torch.long)

        set_quant_mode(qnn, False, False)
        fp = branch_outputs(qnn, x_t, timestep, conditioning)
        set_quant_mode(qnn, True, True)
        quant = branch_outputs(qnn, x_t, timestep, conditioning)
        cond_fp, uncond_fp = fp["conditional"], fp["unconditional"]
        cond_q, uncond_q = quant["conditional"], quant["unconditional"]
        cfg_fp = uncond_fp + scale * (cond_fp - uncond_fp)
        cfg_q = uncond_q + scale * (cond_q - uncond_q)
        cond_error = cond_q - cond_fp
        uncond_error = uncond_q - uncond_fp
        cfg_error = cfg_q - cfg_fp
        weighted_cond_error = scale * cond_error
        weighted_uncond_error = (1.0 - scale) * uncond_error
        rss = torch.sqrt(norm(weighted_cond_error) ** 2 + norm(weighted_uncond_error) ** 2)
        rows.append(
            {
                "progress": progress,
                "original_timestep": original_timestep,
                "cfg_scale": scale,
                "fp_guidance_ratio": float(norm(cond_fp - uncond_fp) / norm(uncond_fp).clamp_min(1e-30)),
                "fp_cond_uncond_cosine": cosine(cond_fp, uncond_fp),
                "conditional_quant_relative_l2": relative_l2(cond_q, cond_fp),
                "unconditional_quant_relative_l2": relative_l2(uncond_q, uncond_fp),
                "cfg_quant_relative_l2": relative_l2(cfg_q, cfg_fp),
                "branch_error_cosine": cosine(cond_error, uncond_error),
                "weighted_branch_error_cosine": cosine(weighted_cond_error, weighted_uncond_error),
                "cfg_error_over_weighted_branch_rss": float(norm(cfg_error) / rss.clamp_min(1e-30)),
                "cfg_error_l2": float(norm(cfg_error)),
                "all_finite": bool(
                    all(torch.isfinite(tensor).all() for tensor in (cond_fp, uncond_fp, cond_q, uncond_q))
                ),
            }
        )
        # The DDIM progressive iterator is lazy.  Its next step must see FP,
        # not the W4A6 state used by the final local comparison above.
        set_quant_mode(qnn, False, False)
        assert_full_precision_state(qnn)

    with (output / "cfg_stage_metrics.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    metadata = {
        "prompt_index": args.prompt_index,
        "seed": args.seed,
        "selected_progress": selected,
        "rows": len(rows),
        "all_values_finite": all(
            row["all_finite"] and all(math.isfinite(v) for v in row.values() if isinstance(v, float))
            for row in rows
        ),
        "fp_state_restored_after_each_selected_progress": True,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
