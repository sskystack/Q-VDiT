#!/usr/bin/env python3
"""Paired timestep profiling for Experiment C/D.

The full-precision DDIM trajectory supplies one shared x_t. FP, W4, A6 and
W4A6 are then evaluated on that exact x_t, preventing accumulated trajectory
drift from contaminating the local quantization-error measurement.
"""

import argparse
import json
import math
from collections import defaultdict
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from mmengine.config import Config
from mmengine.runner import set_random_seed
from omegaconf import OmegaConf

from opensora.registry import MODELS, SCHEDULERS, build_module
from opensora.schedulers.iddpm import forward_with_cfg
from opensora.utils.misc import to_torch_dtype
from qdiff.models.quant_layer import QuantLayer
from qdiff.models.quant_model import QuantModel
from qdiff.utils import load_quant_params


MODES = {
    "fp": (False, False),
    "w4": (True, False),
    "a6": (False, True),
    "w4a6": (True, True),
}


def set_quant_mode(qnn, weight_quant, act_quant):
    """Match official --part_fp inference semantics for every mode switch."""
    qnn.set_quant_state(weight_quant, act_quant)
    if qnn.fp_layer_list:
        qnn.set_layer_quant(
            model=qnn,
            module_name_list=qnn.fp_layer_list,
            quant_level="per_layer",
            weight_quant=False,
            act_quant=False,
            prefix="",
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
    parser.add_argument("--time-mp-config-weight", required=True)
    parser.add_argument("--time-mp-config-act", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-stat-values", type=int, default=65536)
    parser.add_argument("--skip-module-sensitivity", action="store_true")
    return parser.parse_args()


def prepare_conditioning(path, prompt_index, device, dtype):
    loaded = torch.load(path, map_location="cpu")
    y, mask = loaded["y"], loaded["mask"]
    if prompt_index >= y.shape[0]:
        raise IndexError(f"prompt {prompt_index} exceeds embedding batch {y.shape[0]}")
    return {
        "y": y[prompt_index : prompt_index + 1]
        .permute(1, 0, 2, 3, 4)
        .reshape(-1, y.shape[2], y.shape[3], y.shape[4])
        .to(device=device, dtype=dtype),
        "mask": mask[prompt_index : prompt_index + 1].to(device=device),
    }


def cosine(x, ref):
    a = x.detach().double().reshape(-1)
    b = ref.detach().double().reshape(-1)
    return float(
        (torch.dot(a, b) / (torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b) + 1e-12))
        .clamp(-1, 1)
    )


def error_metrics(x, ref):
    a = x.detach().double()
    b = ref.detach().double()
    diff = a - b
    ref_energy = torch.sum(b.square()).clamp_min(1e-12)
    return {
        "relative_l2": float(torch.linalg.vector_norm(diff) / torch.linalg.vector_norm(b).clamp_min(1e-12)),
        "nmse": float(torch.sum(diff.square()) / ref_energy),
        "cosine": cosine(a, b),
        "rmse": float(torch.sqrt(torch.mean(diff.square()))),
        "max_abs_error": float(diff.abs().max()),
    }


def sample_flat(x, max_values):
    flat = x.detach().float().reshape(-1)
    if flat.numel() <= max_values:
        return flat
    stride = max(1, flat.numel() // max_values)
    return flat[::stride][:max_values]


def classify_module(name):
    if ".attn_temp." in name:
        return "temporal_attention"
    if ".cross_attn." in name:
        return "cross_attention"
    if ".attn." in name:
        return "spatial_attention"
    if ".mlp.fc1" in name:
        return "ffn_fc1"
    if ".mlp.fc2" in name:
        return "ffn_fc2"
    if "embedder" in name:
        return "embedder"
    if "final" in name:
        return "final"
    return "other_linear"


def block_index(name):
    parts = name.split(".")
    if "blocks" in parts:
        pos = parts.index("blocks")
        if pos + 1 < len(parts) and parts[pos + 1].isdigit():
            return int(parts[pos + 1])
    return None


def dynamic_per_token_quant(x, bits):
    # Exact reproduction of BaseQuantizer.init_quant_params(per_group="token"):
    # for x=[B,N,C], each of the N token positions shares one range over B*C.
    # B is often frames (spatial attention) or spatial locations (temporal
    # attention), so treating every B,N pair independently would be incorrect.
    if x.ndim != 3:
        raise ValueError(f"Dynamic token quantization expects [B,N,C], got {tuple(x.shape)}")
    source = x.detach().float()
    n_token = source.shape[1]
    groups = source.permute(1, 0, 2).reshape(n_token, -1)
    row_min = groups.min(dim=-1).values
    row_max = groups.max(dim=-1).values
    row_min = torch.minimum(row_min, torch.zeros_like(row_min))
    row_max = torch.maximum(row_max, torch.zeros_like(row_max))
    levels = 2**bits
    delta = ((row_max - row_min) / (levels - 1)).clamp_min(1e-6)
    zero_point = torch.round(-row_min / delta)
    delta_view = delta.reshape(1, n_token, 1)
    zero_point_view = zero_point.reshape(1, n_token, 1)
    q_int = torch.clamp(torch.round(source / delta_view) + zero_point_view, 0, levels - 1)
    dequant = (q_int - zero_point_view) * delta_view
    return (
        source,
        dequant,
        delta,
        row_min.reshape(1, n_token, 1),
        row_max.reshape(1, n_token, 1),
    )


class ActivationCollector:
    def __init__(self, max_values):
        self.max_values = max_values
        self.rows = []
        self.active = False
        self.mode = None
        self.branch = None
        self.progress = None
        self.original_timestep = None
        self.handles = []

    def register(self, qnn):
        for name, module in qnn.named_modules():
            if isinstance(module, QuantLayer):
                self.handles.append(module.register_forward_pre_hook(self._make_hook(name)))

    def _make_hook(self, name):
        def hook(module, inputs):
            if not self.active or not inputs or not torch.is_tensor(inputs[0]):
                return
            x = inputs[0].detach()
            sampled = sample_flat(x, self.max_values)
            abs_sampled = sampled.abs()
            finite = torch.isfinite(sampled)
            sampled = sampled[finite]
            abs_sampled = abs_sampled[finite]
            if sampled.numel() == 0:
                return
            rows = x.float().reshape(-1, x.shape[-1])
            if x.ndim == 3:
                quant_groups = x.float().permute(1, 0, 2).reshape(x.shape[1], -1)
                token_range = quant_groups.max(dim=-1).values - quant_groups.min(dim=-1).values
            else:
                token_range = rows.max(dim=-1).values - rows.min(dim=-1).values
            channel_range = rows.max(dim=0).values - rows.min(dim=0).values
            token_sample = sample_flat(token_range, self.max_values)
            channel_sample = sample_flat(channel_range, self.max_values)
            p999 = torch.quantile(abs_sampled, 0.999)
            record = {
                "mode": self.mode,
                "branch": self.branch,
                "progress": self.progress,
                "original_timestep": self.original_timestep,
                "module": name,
                "module_group": classify_module(name),
                "block_index": block_index(name),
                "shape": list(x.shape),
                "numel": x.numel(),
                "min": float(sampled.min()),
                "max": float(sampled.max()),
                "mean": float(sampled.mean()),
                "std": float(sampled.std(unbiased=False)),
                "abs_max": float(abs_sampled.max()),
                "abs_p99": float(torch.quantile(abs_sampled, 0.99)),
                "abs_p999": float(p999),
                "abs_p9999": float(torch.quantile(abs_sampled, 0.9999)),
                "tail_ratio_max_over_p999": float(abs_sampled.max() / p999.clamp_min(1e-12)),
                "zero_ratio": float((sampled == 0).float().mean()),
                "fp16_overflow_ratio": float((abs_sampled > 65504).float().mean()),
                "fp16_subnormal_or_underflow_ratio": float(
                    ((abs_sampled > 0) & (abs_sampled < 2**-14)).float().mean()
                ),
                "token_range_mean": float(token_sample.mean()),
                "token_range_std": float(token_sample.std(unbiased=False)),
                "token_range_max": float(token_sample.max()),
                "channel_range_mean": float(channel_sample.mean()),
                "channel_range_std": float(channel_sample.std(unbiased=False)),
                "channel_range_max": float(channel_sample.max()),
                "act_bits": int(module.act_quantizer.n_bits) if hasattr(module, "act_quantizer") else None,
            }
            if (
                self.mode in ("a6", "w4a6")
                and getattr(module, "act_quant", False)
                and hasattr(module, "act_quantizer")
                and not module.disable_act_quant
            ):
                bits = int(module.act_quantizer.n_bits)
                source, quantized, delta, qmin, qmax = dynamic_per_token_quant(x, bits)
                qdiff = quantized - source
                signal = torch.mean(source.square()).clamp_min(1e-12)
                q_mse = torch.mean(qdiff.square())
                # Dynamic min-max quantization has no intended clipping. Keep
                # the explicit field so this can be verified rather than assumed.
                clipping = ((source < qmin) | (source > qmax)).float().mean()
                record.update(
                    {
                        "quant_nmse": float(q_mse / signal),
                        "quant_sqnr_db": float(10 * torch.log10(signal / q_mse.clamp_min(1e-20))),
                        "rounding_mse": float(q_mse),
                        "clipping_ratio": float(clipping),
                        "delta_mean": float(delta.mean()),
                        "delta_std": float(delta.std(unbiased=False)),
                        "delta_max": float(delta.max()),
                    }
                )
            else:
                record.update(
                    {
                        "quant_nmse": None,
                        "quant_sqnr_db": None,
                        "rounding_mse": None,
                        "clipping_ratio": None,
                        "delta_mean": None,
                        "delta_std": None,
                        "delta_max": None,
                    }
                )
            self.rows.append(record)

        return hook

    def set_context(self, mode, branch, progress, original_timestep):
        self.mode = mode
        self.branch = branch
        self.progress = progress
        self.original_timestep = original_timestep

    def close(self):
        for handle in self.handles:
            handle.remove()


def profiled_forward_with_cfg(qnn, x, timestep, model_args, cfg_scale, collector=None):
    y = model_args["y"]
    mask = model_args["mask"]
    y_shape = y.shape
    y = y.reshape([2, y_shape[0] // 2] + list(y_shape[1:]))
    timestep = timestep.reshape([2, -1])
    y_cond, y_uncond = y.unbind(0)
    t_cond, t_uncond = timestep.unbind(0)
    half = x[: len(x) // 2]

    if collector is not None:
        collector.branch = "conditional"
    cond = qnn.forward(half, t_cond, y_cond, mask=mask)
    if collector is not None:
        collector.branch = "unconditional"
    uncond = qnn.forward(half, t_uncond, y_uncond, mask=mask)
    cond = cond["x"] if isinstance(cond, dict) else cond
    uncond = uncond["x"] if isinstance(uncond, dict) else uncond
    cond_eps = cond[:, :3]
    uncond_eps = uncond[:, :3]
    cfg_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
    return {"cfg": cfg_eps, "conditional": cond_eps, "unconditional": uncond_eps}


def build_quant_model(args, cfg, device, dtype):
    calib = OmegaConf.load(args.calib_config)
    scheduler = build_module(cfg.scheduler, SCHEDULERS)
    vae = build_module(cfg.vae, MODELS)
    latent_size = vae.get_latent_size((cfg.num_frames, *cfg.image_size))
    model = build_module(
        cfg.model,
        MODELS,
        input_size=latent_size,
        in_channels=vae.out_channels,
        caption_channels=4096,
        model_max_length=cfg.text_encoder.model_max_length,
        dtype=dtype,
        enable_sequence_parallelism=False,
    ).to(device, dtype).eval()

    wq_params = calib.quant.weight.quantizer
    aq_params = calib.quant.activation.quantizer
    if calib.get("mixed_precision", False):
        wq_params["mixed_precision"] = calib.mixed_precision
    qnn = QuantModel(model=model, weight_quant_params=wq_params, act_quant_params=aq_params)
    qnn.cuda().eval()
    qnn.cfg_split = bool(calib.get("cfg_split", False))
    fp_layers = [line.strip() for line in Path(calib.part_fp_list).read_text().splitlines() if line.strip()]
    qnn.fp_layer_list = fp_layers
    qnn.set_quant_state(False, False)
    qnn.set_smooth_quant(smooth_quant=False, smooth_quant_running_stat=False)

    with open(args.time_mp_config_weight) as handle:
        qnn.load_bitwidth_config(qnn, yaml.safe_load(handle), bit_type="weight")
    with open(args.time_mp_config_act) as handle:
        qnn.load_bitwidth_config(qnn, yaml.safe_load(handle), bit_type="act")
    qnn.set_quant_init_done("weight")
    qnn.set_quant_init_done("activation")
    load_quant_params(qnn, args.quant_ckpt)
    qnn.cuda()
    return scheduler, latent_size, qnn


def quantizable_module_groups(qnn):
    fp_roots = set(qnn.fp_layer_list)
    names = []
    for name, module in qnn.model.named_modules():
        if not isinstance(module, QuantLayer):
            continue
        if any(name == root or name.startswith(root + ".") for root in fp_roots):
            continue
        names.append(name)

    groups = {
        "spatial_attention": [n for n in names if ".attn." in n and ".cross_attn." not in n and ".attn_temp." not in n],
        "temporal_attention": [n for n in names if ".attn_temp." in n],
        "cross_attention": [n for n in names if ".cross_attn." in n],
        "ffn": [n for n in names if ".mlp." in n],
        "early_blocks": [n for n in names if block_index(n) is not None and block_index(n) <= 8],
        "middle_blocks": [n for n in names if block_index(n) is not None and 9 <= block_index(n) <= 18],
        "late_blocks": [n for n in names if block_index(n) is not None and block_index(n) >= 19],
    }
    return {key: value for key, value in groups.items() if value}


def plot_results(outdir, output_rows, activation_rows, module_rows):
    modes = ("w4", "a6", "w4a6")
    progress_values = sorted({row["progress"] for row in output_rows})
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for mode in modes:
        rows = sorted((r for r in output_rows if r["mode"] == mode), key=lambda r: r["progress"])
        axes[0].plot([r["progress"] for r in rows], [r["cfg_nmse"] for r in rows], marker="o", label=mode)
        axes[1].plot([r["progress"] for r in rows], [1 - r["cfg_cosine"] for r in rows], marker="o", label=mode)
        axes[2].plot([r["progress"] for r in rows], [r["cfg_relative_l2"] for r in rows], marker="o", label=mode)
    axes[0].set_ylabel("Local noise-prediction NMSE")
    axes[1].set_ylabel("1 - local direction cosine")
    axes[2].set_ylabel("Local relative L2")
    for ax in axes:
        ax.set_xlabel("Sampling progress")
        ax.grid(alpha=0.25)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(outdir / "local_output_error_curves.png", dpi=180)
    plt.close(fig)

    groups = sorted({r["module_group"] for r in activation_rows if r["quant_nmse"] is not None})
    for mode in ("a6", "w4a6"):
        matrix = np.full((len(groups), len(progress_values)), np.nan)
        for gi, group in enumerate(groups):
            for pi, progress in enumerate(progress_values):
                values = [
                    r["quant_sqnr_db"]
                    for r in activation_rows
                    if r["mode"] == mode and r["module_group"] == group and r["progress"] == progress and r["quant_sqnr_db"] is not None
                ]
                if values:
                    matrix[gi, pi] = float(np.mean(values))
        fig, ax = plt.subplots(figsize=(10, max(4, 0.5 * len(groups))))
        image = ax.imshow(matrix, aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(progress_values)), progress_values)
        ax.set_yticks(range(len(groups)), groups)
        ax.set_xlabel("Sampling progress")
        ax.set_title(f"{mode.upper()} activation SQNR (module-group mean)")
        fig.colorbar(image, ax=ax, label="SQNR (dB)")
        fig.tight_layout()
        fig.savefig(outdir / f"activation_sqnr_heatmap_{mode}.png", dpi=180)
        plt.close(fig)

    if module_rows:
        module_groups = sorted({r["module_group"] for r in module_rows})
        matrix = np.full((len(module_groups), len(progress_values)), np.nan)
        for gi, group in enumerate(module_groups):
            for pi, progress in enumerate(progress_values):
                values = [r["cfg_nmse"] for r in module_rows if r["module_group"] == group and r["progress"] == progress]
                if values:
                    matrix[gi, pi] = float(np.mean(values))
        fig, ax = plt.subplots(figsize=(10, max(4, 0.55 * len(module_groups))))
        image = ax.imshow(matrix, aspect="auto", cmap="magma")
        ax.set_xticks(range(len(progress_values)), progress_values)
        ax.set_yticks(range(len(module_groups)), module_groups)
        ax.set_xlabel("Sampling progress")
        ax.set_title("Module-group × timestep W4A6 sensitivity")
        fig.colorbar(image, ax=ax, label="Noise-prediction NMSE")
        fig.tight_layout()
        fig.savefig(outdir / "module_timestep_sensitivity_heatmap.png", dpi=180)
        plt.close(fig)


def main():
    args = parse_args()
    outdir = Path(args.output_dir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    selected = sorted({int(x) for x in args.selected_progress.split(",") if x.strip()})
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
    scheduler, latent_size, qnn = build_quant_model(args, cfg, device, dtype)
    conditioning = prepare_conditioning(args.text_embeds, args.prompt_index, device, dtype)
    collector = ActivationCollector(args.max_stat_values)
    collector.register(qnn)
    module_groups = quantizable_module_groups(qnn)

    generator_rng = torch.Generator(device=device)
    generator_rng.manual_seed(args.seed)
    init_noise = torch.randn(1, qnn.in_channels, *latent_size, device=device, generator=generator_rng)
    z = torch.cat([init_noise, init_noise], dim=0)
    set_quant_mode(qnn, False, False)
    trajectory_forward = partial(forward_with_cfg, qnn, cfg_scale=cfg.scheduler.cfg_scale, return_trajectory=False)
    trajectory = scheduler.ddim_sample_loop_progressive(
        trajectory_forward,
        z.shape,
        noise=z,
        clip_denoised=False,
        model_kwargs=conditioning,
        progress=True,
        return_trajectory=False,
        device=device,
    )

    output_rows = []
    module_rows = []
    current_x = z
    total_steps = scheduler.num_timesteps
    for sampling_index, result in enumerate(trajectory):
        progress = sampling_index + 1
        x_t = current_x.detach()
        current_x = result["sample"]
        if progress not in selected:
            continue
        respaced_timestep = total_steps - progress
        original_timestep = int(scheduler.timestep_map[respaced_timestep])
        timestep = torch.tensor([original_timestep] * x_t.shape[0], device=device, dtype=torch.long)
        mode_outputs = {}
        for mode, (weight_quant, act_quant) in MODES.items():
            set_quant_mode(qnn, weight_quant, act_quant)
            collector.set_context(mode, None, progress, original_timestep)
            collector.active = True
            mode_outputs[mode] = profiled_forward_with_cfg(
                qnn, x_t, timestep, conditioning, cfg.scheduler.cfg_scale, collector
            )
            collector.active = False
        set_quant_mode(qnn, False, False)
        fp = mode_outputs["fp"]
        for mode in ("w4", "a6", "w4a6"):
            row = {
                "mode": mode,
                "progress": progress,
                "respaced_timestep": respaced_timestep,
                "original_timestep": original_timestep,
            }
            for branch in ("cfg", "conditional", "unconditional"):
                metrics = error_metrics(mode_outputs[mode][branch], fp[branch])
                for key, value in metrics.items():
                    row[f"{branch}_{key}"] = value
            cond_error = torch.linalg.vector_norm(
                mode_outputs[mode]["conditional"].double() - fp["conditional"].double()
            )
            uncond_error = torch.linalg.vector_norm(
                mode_outputs[mode]["unconditional"].double() - fp["unconditional"].double()
            )
            cfg_error = torch.linalg.vector_norm(mode_outputs[mode]["cfg"].double() - fp["cfg"].double())
            row["cfg_error_over_branch_error_sum"] = float(
                cfg_error / (cond_error + uncond_error).clamp_min(1e-12)
            )
            output_rows.append(row)

        if not args.skip_module_sensitivity:
            for group_name, names in module_groups.items():
                set_quant_mode(qnn, False, False)
                qnn.set_layer_quant(
                    model=qnn,
                    module_name_list=names,
                    quant_level="per_layer",
                    weight_quant=True,
                    act_quant=True,
                    prefix="",
                )
                out = profiled_forward_with_cfg(
                    qnn, x_t, timestep, conditioning, cfg.scheduler.cfg_scale, collector=None
                )
                metrics = error_metrics(out["cfg"], fp["cfg"])
                module_rows.append(
                    {
                        "module_group": group_name,
                        "num_quant_layers": len(names),
                        "progress": progress,
                        "original_timestep": original_timestep,
                        **{f"cfg_{key}": value for key, value in metrics.items()},
                    }
                )
            set_quant_mode(qnn, False, False)

    collector.close()
    metadata = {
        "prompt_index": args.prompt_index,
        "prompt": prompt,
        "seed": args.seed,
        "selected_progress": selected,
        "runtime_dtype": str(dtype),
        "cfg_scale": cfg.scheduler.cfg_scale,
        "quant_checkpoint": args.quant_ckpt,
        "fp_layers": qnn.fp_layer_list,
        "module_group_sizes": {key: len(value) for key, value in module_groups.items()},
        "paired_input_design": True,
    }
    (outdir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    with (outdir / "local_output_errors.jsonl").open("w") as handle:
        for row in output_rows:
            handle.write(json.dumps(row) + "\n")
    with (outdir / "activation_stats.jsonl").open("w") as handle:
        for row in collector.rows:
            handle.write(json.dumps(row) + "\n")
    with (outdir / "module_sensitivity.jsonl").open("w") as handle:
        for row in module_rows:
            handle.write(json.dumps(row) + "\n")
    plot_results(outdir, output_rows, collector.rows, module_rows)
    summary = {
        "output_rows": len(output_rows),
        "activation_rows": len(collector.rows),
        "module_sensitivity_rows": len(module_rows),
        "all_output_values_finite": all(
            math.isfinite(value)
            for row in output_rows
            for key, value in row.items()
            if isinstance(value, float)
        ),
        "all_activation_values_finite": all(
            math.isfinite(value)
            for row in collector.rows
            for key, value in row.items()
            if isinstance(value, float)
        ),
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
