#!/usr/bin/env python3
"""Profile internal operators that QuantLayer-input statistics cannot explain."""

import argparse
import json
import math
from collections import defaultdict
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
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
    parser.add_argument("--prompt-path", required=True)
    parser.add_argument("--text-embeds", required=True)
    parser.add_argument("--prompt-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--selected-progress", default="5,15,25,45,65,85,95")
    parser.add_argument("--selected-blocks", default="0,9,17,27")
    parser.add_argument("--max-stat-values", type=int, default=65536)
    parser.add_argument("--attention-query-samples", type=int, default=32)
    parser.add_argument("--attention-batch-samples", type=int, default=2)
    parser.add_argument("--attention-head-samples", type=int, default=2)
    parser.add_argument("--tqe-token-samples", type=int, default=128)
    parser.add_argument("--time-mp-config-weight", required=True)
    parser.add_argument("--time-mp-config-act", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def evenly_spaced_indices(length, count, device):
    count = min(length, count)
    if count == length:
        return torch.arange(length, device=device)
    return torch.linspace(0, length - 1, count, device=device).round().long().unique()


def tensor_stats(x, max_values):
    flat = x.detach().float().reshape(-1)
    if flat.numel() > max_values:
        stride = max(1, flat.numel() // max_values)
        flat = flat[::stride][:max_values]
    flat = flat[torch.isfinite(flat)]
    if flat.numel() == 0:
        return None
    mean = flat.mean()
    std = flat.std(unbiased=False).clamp_min(1e-12)
    centered = (flat - mean) / std
    abs_flat = flat.abs()
    return {
        "numel": int(x.numel()),
        "shape": list(x.shape),
        "min": float(flat.min()),
        "max": float(flat.max()),
        "mean": float(mean),
        "std": float(std),
        "abs_max": float(abs_flat.max()),
        "abs_p99": float(torch.quantile(abs_flat, 0.99)),
        "abs_p999": float(torch.quantile(abs_flat, 0.999)),
        "skewness": float(centered.pow(3).mean()),
        "kurtosis": float(centered.pow(4).mean()),
        "zero_ratio": float((flat == 0).float().mean()),
        "fp16_overflow_ratio": float((abs_flat > 65504).float().mean()),
        "fp16_subnormal_ratio": float(((abs_flat > 0) & (abs_flat < 2**-14)).float().mean()),
    }


def sampled_attention(q, k, num_heads, scale, kind, args):
    # q/k are linear outputs [B,N,C]. Cross-attention k has already been split.
    bsz, nq, channels = q.shape
    nk = k.shape[1]
    head_dim = channels // num_heads
    q = q.float().reshape(bsz, nq, num_heads, head_dim).permute(0, 2, 1, 3)
    k = k.float().reshape(k.shape[0], nk, num_heads, head_dim).permute(0, 2, 1, 3)
    b_idx = evenly_spaced_indices(bsz, args.attention_batch_samples, q.device)
    h_idx = evenly_spaced_indices(num_heads, args.attention_head_samples, q.device)
    q_idx = evenly_spaced_indices(nq, args.attention_query_samples, q.device)
    q_sel = q.index_select(0, b_idx).index_select(1, h_idx).index_select(2, q_idx)
    if k.shape[0] == 1 and q_sel.shape[0] > 1:
        k_sel = k.index_select(1, h_idx).expand(q_sel.shape[0], -1, -1, -1)
    else:
        k_sel = k.index_select(0, b_idx).index_select(1, h_idx)
    logits = torch.matmul(q_sel * scale, k_sel.transpose(-2, -1))
    probs = torch.softmax(logits, dim=-1)
    entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1)
    top1 = probs.topk(1, dim=-1).values.sum(dim=-1)
    topk = probs.topk(min(5, nk), dim=-1).values.sum(dim=-1)
    locality = None
    key_positions = torch.arange(nk, device=probs.device, dtype=torch.float32)
    if kind == "spatial" and int(math.isqrt(nk)) ** 2 == nk and nq == nk:
        side = int(math.isqrt(nk))
        qy = (q_idx // side).float().view(1, 1, -1, 1)
        qx = (q_idx % side).float().view(1, 1, -1, 1)
        ky = (key_positions // side).view(1, 1, 1, -1)
        kx = (key_positions % side).view(1, 1, 1, -1)
        distance = (qy - ky).abs() + (qx - kx).abs()
        locality = float((probs * distance).sum(dim=-1).mean() / max(1, 2 * (side - 1)))
    elif kind == "temporal" and nq == nk:
        distance = (q_idx.float().view(1, 1, -1, 1) - key_positions.view(1, 1, 1, -1)).abs()
        locality = float((probs * distance).sum(dim=-1).mean() / max(1, nk - 1))
    return {
        "entropy": float(entropy.mean()),
        "normalized_entropy": float(entropy.mean() / math.log(max(2, nk))),
        "top1_mass": float(top1.mean()),
        "top5_mass": float(topk.mean()),
        "locality_distance": locality,
        "sampled_query_count": int(q_sel.shape[0] * q_sel.shape[1] * q_sel.shape[2]),
        "num_keys": nk,
    }, probs.detach().cpu().half()


class InternalCollector:
    def __init__(self, qnn, blocks, args):
        self.qnn = qnn
        self.blocks = blocks
        self.args = args
        self.active = False
        self.mode = None
        self.branch = None
        self.progress = None
        self.original_timestep = None
        self.rows = []
        self.attention_rows = []
        self.attention_probs = {}
        self.cache = {}
        self.handles = []
        self.tqe_matrix_cache = {}
        self._register()

    def context(self):
        return {
            "mode": self.mode,
            "branch": self.branch,
            "progress": self.progress,
            "original_timestep": self.original_timestep,
        }

    def _register(self):
        selected_prefixes = tuple(f"model.blocks.{i}." for i in self.blocks)
        for name, module in self.qnn.named_modules():
            if name in {f"model.blocks.{i}" for i in self.blocks}:
                self.handles.append(module.register_forward_hook(self._block_hook(name)))
            if any(name == f"model.blocks.{i}.norm1" or name == f"model.blocks.{i}.norm2" for i in self.blocks):
                self.handles.append(module.register_forward_hook(self._tensor_hook(name, "layernorm_output")))
            if any(name == f"model.blocks.{i}.mlp.act" for i in self.blocks):
                self.handles.append(module.register_forward_hook(self._tensor_hook(name, "post_gelu")))
            if isinstance(module, QuantLayer) and name.startswith(selected_prefixes):
                if any(token in name for token in (".attn.", ".attn_temp.", ".cross_attn.", ".mlp.")):
                    self.handles.append(module.register_forward_hook(self._quant_hook(name, module)))

    def _tensor_hook(self, name, operator):
        def hook(module, inputs, output):
            if not self.active or not torch.is_tensor(output):
                return
            stats = tensor_stats(output, self.args.max_stat_values)
            if stats:
                self.rows.append({**self.context(), "operator": operator, "module": name, **stats})
        return hook

    def _block_hook(self, name):
        def hook(module, inputs, output):
            if not self.active or not inputs or not torch.is_tensor(inputs[0]) or not torch.is_tensor(output):
                return
            residual = output.detach() - inputs[0].detach()
            stats = tensor_stats(residual, self.args.max_stat_values)
            if stats:
                input_norm = torch.linalg.vector_norm(inputs[0].detach().float()).clamp_min(1e-12)
                stats["residual_over_input_l2"] = float(
                    torch.linalg.vector_norm(residual.float()) / input_norm
                )
                self.rows.append({**self.context(), "operator": "block_total_residual", "module": name, **stats})
        return hook

    def _get_tqe_matrices(self, name, module):
        if name in self.tqe_matrix_cache:
            return self.tqe_matrix_cache[name]
        with torch.no_grad():
            eye = torch.eye(module.weight.shape[1], device=module.weight.device, dtype=module.loraB.weight.dtype)
            lora_in = module.loraB(module.loraA(eye)).T
            lora_out = module.loraB_out(module.loraA_out(eye)).T
            q_with = module.weight_quantizer(module.weight + lora_in)
            q_base = module.weight_quantizer(module.weight)
            inside_effect = q_with - q_base
        cached = (inside_effect.detach(), lora_out.detach())
        self.tqe_matrix_cache[name] = cached
        return cached

    def _quant_hook(self, name, module):
        def hook(mod, inputs, output):
            if not self.active or not inputs or not torch.is_tensor(inputs[0]) or not torch.is_tensor(output):
                return
            stats = tensor_stats(output, self.args.max_stat_values)
            if stats:
                self.rows.append({**self.context(), "operator": "linear_output", "module": name, **stats})
            self.cache[name] = output.detach()
            if self.mode == "w4a6" and getattr(module, "weight_quant", False):
                inside_weight, outside_weight = self._get_tqe_matrices(name, module)
                x = inputs[0].detach().reshape(-1, inputs[0].shape[-1])
                if x.shape[0] > self.args.tqe_token_samples:
                    idx = evenly_spaced_indices(x.shape[0], self.args.tqe_token_samples, x.device)
                    x = x.index_select(0, idx)
                x_inside = x.to(inside_weight.dtype)
                inside = F.linear(x_inside, inside_weight)
                outside = F.linear(x.to(outside_weight.dtype), outside_weight)
                total = inside.to(torch.float32) + outside.to(torch.float32)
                out_sample = output.detach().reshape(-1, output.shape[-1])
                if out_sample.shape[0] > x.shape[0]:
                    idx = evenly_spaced_indices(out_sample.shape[0], x.shape[0], out_sample.device)
                    out_sample = out_sample.index_select(0, idx)
                out_norm = torch.linalg.vector_norm(out_sample.float()).clamp_min(1e-12)
                self.rows.append(
                    {
                        **self.context(),
                        "operator": "tqe_compensation_output",
                        "module": name,
                        "sampled_tokens": int(x.shape[0]),
                        "inside_quantizer_l2": float(torch.linalg.vector_norm(inside.float())),
                        "outside_quantizer_l2": float(torch.linalg.vector_norm(outside.float())),
                        "total_compensation_l2": float(torch.linalg.vector_norm(total)),
                        "compensation_over_layer_output": float(torch.linalg.vector_norm(total) / out_norm),
                    }
                )
        return hook

    def finalize_attention(self):
        for block in self.blocks:
            module = self.qnn.model.blocks[block]
            specs = [
                ("spatial", f"model.blocks.{block}.attn.q", f"model.blocks.{block}.attn.k", module.attn),
                ("temporal", f"model.blocks.{block}.attn_temp.q", f"model.blocks.{block}.attn_temp.k", module.attn_temp),
            ]
            for kind, q_name, k_name, attn_module in specs:
                if q_name not in self.cache or k_name not in self.cache:
                    continue
                summary, probs = sampled_attention(
                    self.cache[q_name], self.cache[k_name], attn_module.num_heads, attn_module.scale, kind, self.args
                )
                key = (self.progress, self.branch, self.mode, block, kind)
                self.attention_probs[key] = probs
                self.attention_rows.append({**self.context(), "block": block, "attention_kind": kind, **summary})

            q_name = f"model.blocks.{block}.cross_attn.q_linear"
            kv_name = f"model.blocks.{block}.cross_attn.kv_linear"
            if q_name in self.cache and kv_name in self.cache:
                q = self.cache[q_name]
                kv = self.cache[kv_name]
                k = kv[..., : kv.shape[-1] // 2]
                cross = module.cross_attn
                summary, probs = sampled_attention(q, k, cross.num_heads, cross.head_dim**-0.5, "cross", self.args)
                key = (self.progress, self.branch, self.mode, block, "cross")
                self.attention_probs[key] = probs
                self.attention_rows.append({**self.context(), "block": block, "attention_kind": "cross", **summary})
        self.cache.clear()

    def close(self):
        for handle in self.handles:
            handle.remove()


def split_forward(qnn, x, timestep, conditioning, cfg_scale, collector):
    y = conditioning["y"]
    mask = conditioning["mask"]
    y_shape = y.shape
    y = y.reshape([2, y_shape[0] // 2] + list(y_shape[1:]))
    timestep = timestep.reshape([2, -1])
    y_cond, y_uncond = y.unbind(0)
    t_cond, t_uncond = timestep.unbind(0)
    half = x[: len(x) // 2]
    outputs = {}
    for branch, branch_y, branch_t in (
        ("conditional", y_cond, t_cond),
        ("unconditional", y_uncond, t_uncond),
    ):
        collector.branch = branch
        out = qnn.forward(half, branch_t, branch_y, mask=mask)
        out = out["x"] if isinstance(out, dict) else out
        outputs[branch] = out[:, :3]
        collector.finalize_attention()
    outputs["cfg"] = outputs["unconditional"] + cfg_scale * (
        outputs["conditional"] - outputs["unconditional"]
    )
    return outputs


def main():
    args = parse_args()
    outdir = Path(args.output_dir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    selected = sorted({int(x) for x in args.selected_progress.split(",") if x.strip()})
    blocks = sorted({int(x) for x in args.selected_blocks.split(",") if x.strip()})
    cfg = Config.fromfile(args.config)
    cfg.multi_resolution = cfg.get("multi_resolution", False)
    dtype = to_torch_dtype(cfg.dtype)
    device = torch.device("cuda")
    set_random_seed(args.seed)
    torch.set_grad_enabled(False)

    scheduler, latent_size, qnn = build_quant_model(args, cfg, device, dtype)
    conditioning = prepare_conditioning(args.text_embeds, args.prompt_index, device, dtype)
    collector = InternalCollector(qnn, blocks, args)
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    init_noise = torch.randn(1, qnn.in_channels, *latent_size, device=device, generator=generator)
    z = torch.cat([init_noise, init_noise], dim=0)
    set_quant_mode(qnn, False, False)
    fp_model = partial(
        forward_with_cfg, qnn, cfg_scale=cfg.scheduler.cfg_scale, return_trajectory=False
    )
    trajectory = scheduler.ddim_sample_loop_progressive(
        fp_model,
        z.shape,
        noise=z,
        clip_denoised=False,
        model_kwargs=conditioning,
        progress=True,
        device=device,
    )
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
        for mode, state in (("fp", (False, False)), ("w4a6", (True, True))):
            set_quant_mode(qnn, *state)
            collector.mode = mode
            collector.progress = progress
            collector.original_timestep = original_timestep
            collector.active = True
            split_forward(qnn, x_t, timestep, conditioning, cfg.scheduler.cfg_scale, collector)
            collector.active = False
        # The progressive generator is lazy: its next denoising step executes
        # only after this loop body returns.  Restore FP explicitly so the
        # reference trajectory cannot inherit the final W4A6 comparison mode.
        set_quant_mode(qnn, False, False)
        assert_full_precision_state(qnn)
    collector.close()

    kl_rows = []
    keys = {(p, b, block, kind) for p, b, mode, block, kind in collector.attention_probs}
    for progress, branch, block, kind in sorted(keys):
        fp_key = (progress, branch, "fp", block, kind)
        q_key = (progress, branch, "w4a6", block, kind)
        if fp_key not in collector.attention_probs or q_key not in collector.attention_probs:
            continue
        fp = collector.attention_probs[fp_key].float()
        quant = collector.attention_probs[q_key].float()
        kl = (fp * (torch.log(fp.clamp_min(1e-12)) - torch.log(quant.clamp_min(1e-12)))).sum(dim=-1)
        top_fp = fp.argmax(dim=-1)
        top_q = quant.argmax(dim=-1)
        kl_rows.append(
            {
                "progress": progress,
                "branch": branch,
                "block": block,
                "attention_kind": kind,
                "fp_to_w4a6_kl": float(kl.mean()),
                "top1_change_ratio": float((top_fp != top_q).float().mean()),
            }
        )

    def write_jsonl(name, rows):
        with (outdir / name).open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")

    write_jsonl("internal_operator_stats.jsonl", collector.rows)
    write_jsonl("attention_stats.jsonl", collector.attention_rows)
    write_jsonl("attention_kl.jsonl", kl_rows)
    metadata = {
        "prompt_index": args.prompt_index,
        "seed": args.seed,
        "selected_progress": selected,
        "selected_blocks": blocks,
        "operator_rows": len(collector.rows),
        "attention_rows": len(collector.attention_rows),
        "attention_kl_rows": len(kl_rows),
        "all_values_finite": all(
            math.isfinite(v)
            for rows in (collector.rows, collector.attention_rows, kl_rows)
            for row in rows
            for v in row.values()
            if isinstance(v, float)
        ),
        "fp_state_restored_after_each_selected_progress": True,
    }
    (outdir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    kinds = ["spatial", "temporal", "cross"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, kind in zip(axes, kinds):
        for block in blocks:
            rows = sorted(
                (r for r in kl_rows if r["attention_kind"] == kind and r["block"] == block),
                key=lambda r: r["progress"],
            )
            grouped = defaultdict(list)
            for row in rows:
                grouped[row["progress"]].append(row["fp_to_w4a6_kl"])
            if grouped:
                xs = sorted(grouped)
                ax.plot(xs, [float(np.mean(grouped[x])) for x in xs], marker="o", label=f"block {block}")
        ax.set_title(kind)
        ax.set_xlabel("Sampling progress")
        ax.set_ylabel("FP→W4A6 attention KL")
        ax.grid(alpha=0.25)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(outdir / "attention_kl_curves.png", dpi=180)
    plt.close(fig)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
