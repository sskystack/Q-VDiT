"""Motivation-validation experiments for TARQ.

Tests the core hypothesis behind TARQ: "quantization error *direction* is
correlated with a token's motion pattern (beyond what content/magnitude
already explains)".

Experiment 1 (qualitative): per-layer, per-token W4A6 error-energy heatmaps,
rendered next to the latent content and the transport motion magnitude.

Experiment 2 (quantitative), per layer x sampling step:
  - Spearman correlations of error energy vs motion magnitude / content norm,
    plus the partial correlation motion|content.
  - Group tokens by (a) motion descriptor, (b) content statistics, (c) random,
    with k-means (k=4); measure within-group error-direction coherence
    (resultant length of normalized error vectors) and the fraction of error
    energy a per-group rank-1 expert can explain.  A single global rank-1 fit
    (the TQE regime) is the reference.

Uses the existing calibration cache (FP trajectories) and the FP checkpoint;
no PTQ run is needed.  Quantizers are simulated offline with the W4A6 layout
from the formal configs (4-bit per-output-channel asymmetric weights, 6-bit
per-token dynamic asymmetric activations, min-max scales).

Run on the server:
  CUDA_VISIBLE_DEVICES=3 python tools/verify_motion_hypothesis.py
"""
import json
import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "t2v"))

from opensora.models.stdit.stdit import STDiT_XL_2  # noqa: E402
from qdiff.research import _compact_transport_features  # noqa: E402

CKPT = "/home/zhouchongtian/quantization/Q-VDiT/logs/split_ckpt/OpenSora-v1-HQ-16x512x512-split.pth"
CALIB = "/home/zhouchongtian/quantization/new/logs_50steps/calib_data_ddim50_cfg4/calib_data.pt"
OUT_DIR = "/home/zhouchongtian/quantization/new/runs/motion_hypothesis"
STEPS = [5, 25, 45]          # early / mid / late sampling positions (of 50)
SAMPLES = [0, 1]             # two cached samples per step
B, T, S, SIDE = 2, 16, 1024, 32
K_GROUPS = 4
SVD_TOKENS = 4096            # token subsample for rank-1 fits
SEED = 0


def quantize_weight_w4(weight):
    """4-bit asymmetric per-output-channel min-max (rows of [out, in])."""
    w_min = weight.amin(dim=1, keepdim=True)
    w_max = weight.amax(dim=1, keepdim=True)
    delta = (w_max - w_min).clamp_min(1e-8) / 15.0
    zp = torch.round(-w_min / delta)
    q = torch.clamp(torch.round(weight / delta) + zp, 0, 15)
    return (q - zp) * delta


def quantize_act_a6(x):
    """6-bit asymmetric per-token dynamic min-max (rows of [tokens, C])."""
    x_min = x.amin(dim=-1, keepdim=True)
    x_max = x.amax(dim=-1, keepdim=True)
    delta = (x_max - x_min).clamp_min(1e-8) / 63.0
    zp = torch.round(-x_min / delta)
    q = torch.clamp(torch.round(x / delta) + zp, 0, 63)
    return (q - zp) * delta


def to_grid(inputs, layout):
    """Return canonical [B, T, S, C] from a captured layer input."""
    C = inputs.shape[-1]
    if layout == "spatial":      # [B*T, S, C]
        return inputs.reshape(B, T, S, C)
    if layout == "temporal":     # [B*S, T, C]
        return inputs.reshape(B, S, T, C).permute(0, 2, 1, 3).contiguous()
    if layout == "sequence":     # [B, T*S, C]
        return inputs.reshape(B, T, S, C)
    raise ValueError(layout)


def motion_descriptor(grid):
    """[B, T, S, 5] descriptor from FP features via a fixed orthogonal proj."""
    C = grid.shape[-1]
    gen = torch.Generator(device="cpu").manual_seed(SEED)
    proj = torch.linalg.qr(torch.randn(C, 8, generator=gen))[0].to(grid.device, grid.dtype)
    desc = (grid.reshape(B, T, SIDE, SIDE, C) @ proj)
    feats = _compact_transport_features(desc, 16, 8, 0.07)  # [B, T, 16, 16, 5]
    feats = feats.permute(0, 1, 4, 2, 3).reshape(B * T, 5, 16, 16)
    feats = F.interpolate(feats, size=(SIDE, SIDE), mode="bilinear", align_corners=False)
    return feats.reshape(B, T, 5, S).permute(0, 1, 3, 2).contiguous()


def kmeans(feats, k, iters=25):
    """Plain torch k-means on standardized features. Returns labels."""
    feats = (feats - feats.mean(0)) / feats.std(0).clamp_min(1e-6)
    gen = torch.Generator(device=feats.device.type if feats.device.type == "cpu" else "cpu").manual_seed(SEED)
    idx = torch.randperm(feats.shape[0], generator=gen)[:k].to(feats.device)
    centers = feats[idx].clone()
    labels = None
    for _ in range(iters):
        dists = torch.cdist(feats, centers)
        labels = dists.argmin(dim=1)
        for j in range(k):
            mask = labels == j
            if mask.any():
                centers[j] = feats[mask].mean(0)
    return labels


def coherence_and_rank1(delta, labels, k):
    """Weighted intra-group direction coherence and grouped rank-1 explained
    energy for one grouping of the error matrix delta [N, d_out]."""
    total_energy = delta.square().sum()
    unit = F.normalize(delta, dim=-1, eps=1e-8)
    coh_num, rank1_energy = 0.0, 0.0
    gen = torch.Generator(device="cpu").manual_seed(SEED)
    for j in range(k):
        mask = labels == j
        n = int(mask.sum())
        if n < 8:
            continue
        group_unit = unit[mask]
        coh_num += n * group_unit.mean(0).norm().item()
        group = delta[mask]
        if n > SVD_TOKENS:
            sel = torch.randperm(n, generator=gen)[:SVD_TOKENS].to(group.device)
            sub = group[sel]
            scale = group.square().sum() / sub.square().sum().clamp_min(1e-12)
        else:
            sub, scale = group, torch.tensor(1.0)
        u, s_vals, v = torch.svd_lowrank(sub.float(), q=6)
        rank1_energy += (s_vals[0] ** 2 * scale).item()
    return coh_num / delta.shape[0], rank1_energy / total_energy.item()


def spearman(a, b):
    ra = a.argsort().argsort().float()
    rb = b.argsort().argsort().float()
    ra = (ra - ra.mean()) / ra.std().clamp_min(1e-8)
    rb = (rb - rb.mean()) / rb.std().clamp_min(1e-8)
    return (ra * rb).mean().item()


def save_heatmap_png(path, panels, titles):
    """panels: list of [T_show, H, W] tensors -> grid png (rows=panels)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        t_show = panels[0].shape[0]
        fig, axes = plt.subplots(len(panels), t_show, figsize=(2.1 * t_show, 2.2 * len(panels)))
        for r, (panel, title) in enumerate(zip(panels, titles)):
            for c in range(t_show):
                ax = axes[r][c] if len(panels) > 1 else axes[c]
                ax.imshow(panel[c].cpu().numpy(), cmap="magma")
                ax.set_xticks([]); ax.set_yticks([])
                if c == 0:
                    ax.set_ylabel(title, fontsize=9)
                if r == 0:
                    ax.set_title(f"frame {c * 5}", fontsize=9)
        fig.tight_layout()
        fig.savefig(path, dpi=110)
        plt.close(fig)
    except Exception as err:  # matplotlib missing: dump raw tensors instead
        torch.save({"panels": panels, "titles": titles}, path + ".pt")
        print(f"matplotlib unavailable ({err}); saved raw tensors")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = "cuda"
    torch.manual_seed(SEED)

    model = STDiT_XL_2(from_pretrained=CKPT, input_size=(16, 64, 64), in_channels=4,
                       caption_channels=4096, model_max_length=120)
    model = model.to(device).eval()

    layer_specs = {
        "b00.spatial_q": (model.blocks[0].attn.q, "spatial"),
        "b00.ffn_fc1": (model.blocks[0].mlp.fc1, "sequence"),
        "b14.spatial_q": (model.blocks[14].attn.q, "spatial"),
        "b14.temporal_q": (model.blocks[14].attn_temp.q, "temporal"),
        "b14.ffn_fc1": (model.blocks[14].mlp.fc1, "sequence"),
        "b27.spatial_q": (model.blocks[27].attn.q, "spatial"),
        "b27.ffn_fc1": (model.blocks[27].mlp.fc1, "sequence"),
    }
    quant_weights = {
        name: (module.weight.data.float(), quantize_weight_w4(module.weight.data.float()))
        for name, (module, _) in layer_specs.items()
    }

    captured = {}
    hooks = []
    for name, (module, _) in layer_specs.items():
        def make_hook(key):
            def hook(_m, args):
                captured[key] = args[0].detach()
            return hook
        hooks.append(module.register_forward_pre_hook(make_hook(name)))

    data = torch.load(CALIB, map_location="cpu")
    results = []
    for step in STEPS:
        xs = data["xs"][step][: 2 * len(SAMPLES)].float()
        # cached rows are [cond..., uncond...]; take the first two entries
        x = xs[list(SAMPLES)].to(device)
        ts = data["ts"][step][list(SAMPLES)].to(device)
        y = data["cond_emb"][step][list(SAMPLES)].to(device).float()
        mask = data["mask"][step][list(SAMPLES)].to(device)
        with torch.no_grad():
            model(x, ts, y, mask=mask)

        content_ref = x.abs().mean(dim=1)  # [B, T, 64, 64]
        content_ref = F.avg_pool2d(content_ref.reshape(B * T, 1, 64, 64), 2).reshape(B, T, SIDE, SIDE)

        for name, (_, layout) in layer_specs.items():
            grid = to_grid(captured[name].float(), layout)         # [B, T, S, C]
            motion = motion_descriptor(grid)                        # [B, T, S, 5]
            weight, weight_q = quant_weights[name]

            tokens = grid.reshape(B * T * S, -1)
            tokens_q = quantize_act_a6(tokens)
            delta = tokens @ weight.T - tokens_q @ weight_q.T       # [N, d_out]
            energy = delta.norm(dim=-1)

            m = motion.reshape(B * T * S, 5)
            motion_mag = m[:, 0]
            content_norm = tokens.norm(dim=-1)
            outlier = tokens.abs().amax(dim=-1) / tokens.square().mean(dim=-1).sqrt().clamp_min(1e-8)

            rho_e_motion = spearman(energy, motion_mag)
            rho_e_content = spearman(energy, content_norm)
            rho_m_c = spearman(motion_mag, content_norm)
            partial = (rho_e_motion - rho_e_content * rho_m_c) / max(
                math.sqrt((1 - rho_e_content ** 2) * (1 - rho_m_c ** 2)), 1e-8)

            motion_labels = kmeans(m.cpu(), K_GROUPS).to(device)
            content_feats = torch.stack([content_norm.log(), outlier], dim=-1)
            content_labels = kmeans(content_feats.cpu(), K_GROUPS).to(device)
            gen = torch.Generator().manual_seed(SEED)
            random_labels = torch.randint(0, K_GROUPS, (delta.shape[0],), generator=gen).to(device)

            row = {"layer": name, "step": step,
                   "rho_energy_motion": round(rho_e_motion, 4),
                   "rho_energy_content": round(rho_e_content, 4),
                   "rho_motion_content": round(rho_m_c, 4),
                   "partial_motion_given_content": round(partial, 4)}
            for tag, labels in [("motion", motion_labels), ("content", content_labels),
                                ("random", random_labels),
                                ("global", torch.zeros_like(random_labels))]:
                coh, ev = coherence_and_rank1(delta, labels, K_GROUPS if tag != "global" else 1)
                row[f"coh_{tag}"] = round(coh, 4)
                row[f"rank1ev_{tag}"] = round(ev, 4)
            results.append(row)
            print(row, flush=True)

            if name in ("b14.spatial_q", "b14.ffn_fc1") and step == 25:
                e_map = energy.reshape(B, T, SIDE, SIDE)[0][::5]
                m_map = motion_mag.reshape(B, T, SIDE, SIDE)[0][::5]
                c_map = content_ref[0][::5]
                save_heatmap_png(
                    os.path.join(OUT_DIR, f"heatmap_{name}_step{step}.png"),
                    [c_map, m_map, e_map],
                    ["latent |x|", "motion mag", "quant err"])

        # heatmaps across steps for one representative layer
        name = "b14.spatial_q"
        grid = to_grid(captured[name].float(), layer_specs[name][1])
        motion = motion_descriptor(grid)
        weight, weight_q = quant_weights[name]
        tokens = grid.reshape(B * T * S, -1)
        delta = tokens @ weight.T - quantize_act_a6(tokens) @ weight_q.T
        e_map = delta.norm(dim=-1).reshape(B, T, SIDE, SIDE)[0][::5]
        m_map = motion.reshape(B * T * S, 5)[:, 0].reshape(B, T, SIDE, SIDE)[0][::5]
        c_map = content_ref[0][::5]
        save_heatmap_png(os.path.join(OUT_DIR, f"heatmap_{name}_step{step}.png"),
                         [c_map, m_map, e_map], ["latent |x|", "motion mag", "quant err"])

    for h in hooks:
        h.remove()
    with open(os.path.join(OUT_DIR, "stats.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved {len(results)} rows -> {OUT_DIR}/stats.json")


if __name__ == "__main__":
    main()
