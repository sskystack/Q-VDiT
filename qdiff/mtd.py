"""Motion Transport Distillation (MTD) for video reconstruction.

This module is intentionally independent from the other research directions.
It only contains MTD configuration parsing and the three MTD loss terms.
"""

from collections.abc import Mapping

import torch
import torch.nn.functional as F


def _plain_dict(value):
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return {
            key: _plain_dict(item) if isinstance(item, Mapping) else item
            for key, item in value.items()
        }
    if hasattr(value, "items"):
        return {
            key: _plain_dict(item) if hasattr(item, "items") else item
            for key, item in value.items()
        }
    return dict(value)


def normalize_mtd_config(config=None):
    """Normalize either a full calibration config or an MTD-only mapping."""
    config = _plain_dict(config)
    already_normalized = "enabled" in config and "transport_size" in config
    method = config if "frame_axis" in config else _plain_dict(config.get("method"))
    mtd = config if already_normalized else _plain_dict(config.get("mtd"))

    normalized = {
        "enabled": bool(config["enabled"]) if already_normalized else (
            str(method.get("frame_axis", "BASELINE")).upper() == "MTD"
        ),
        # OpenSora stacks predicted noise and variance on the channel axis.
        # MTD supervises only the denoising signal.
        "noise_channels": int(config.get("noise_channels", 4)),
        "transport_size": int(mtd.get("transport_size", 16)),
        "temperature": float(mtd.get("temperature", 0.07)),
        "total_weight": float(mtd.get("total_weight", 1.0)),
        "local_transport_weight": float(mtd.get("local_transport_weight", 1.0)),
        "motion_residual_weight": float(mtd.get("motion_residual_weight", 1.0)),
        "global_relation_weight": float(mtd.get("global_relation_weight", 0.1)),
    }
    if normalized["noise_channels"] <= 0:
        raise ValueError("noise_channels must be positive")
    if normalized["transport_size"] <= 0:
        raise ValueError("mtd.transport_size must be positive")
    if normalized["temperature"] <= 0:
        raise ValueError("mtd.temperature must be positive")
    if normalized["total_weight"] < 0:
        raise ValueError("mtd.total_weight must be non-negative")
    return normalized


def _local_transport_distribution(features, size, temperature):
    """Build a 3x3 correspondence distribution for every adjacent-frame token."""
    batch, channels, frames, height, width = features.shape
    pooled = features.permute(0, 2, 1, 3, 4).reshape(
        batch * frames, channels, height, width
    )
    pooled = F.adaptive_avg_pool2d(pooled, (size, size)).reshape(
        batch, frames, channels, size, size
    )
    pooled = F.normalize(pooled, dim=2, eps=1.0e-6)

    current = pooled[:, :-1]
    following = pooled[:, 1:]
    current_flat = current.permute(0, 1, 3, 4, 2).reshape(
        batch * (frames - 1), size * size, channels
    )
    following_flat = following.reshape(
        batch * (frames - 1), channels, size, size
    )
    neighbours = F.unfold(following_flat, kernel_size=3, padding=1)
    neighbours = neighbours.reshape(
        batch * (frames - 1), channels, 9, size * size
    ).permute(0, 3, 2, 1)

    logits = (current_flat[:, :, None, :] * neighbours).sum(-1) / temperature

    # Padding introduced by unfold is not a real candidate. Mask it so corner,
    # edge and interior tokens have exactly 4, 6 and 9 valid neighbours.
    valid_grid = following_flat.new_ones(
        batch * (frames - 1), 1, size, size
    )
    valid_neighbours = F.unfold(valid_grid, kernel_size=3, padding=1)
    valid_neighbours = valid_neighbours.reshape(
        batch * (frames - 1), 1, 9, size * size
    ).permute(0, 3, 2, 1).squeeze(-1).bool()
    logits = logits.masked_fill(
        ~valid_neighbours, torch.finfo(logits.dtype).min
    )
    return F.softmax(logits, dim=-1), current_flat, neighbours


def motion_transport_distillation(
    pred, target, config=None, return_components=False
):
    """Compute local-distribution, transport-residual and global-relation MTD."""
    mtd = normalize_mtd_config(config)
    if not mtd["enabled"] or pred.ndim != 5 or pred.shape[2] < 2:
        zero = pred.new_tensor(0.0)
        if return_components:
            return zero, {"local": zero, "motion": zero, "global": zero}
        return zero

    # Run the terminal loss in FP32 for BF16/FP16 FlashAttention stability.
    # The cast remains differentiable and gradients return in the model dtype.
    pred = pred.float()[:, :mtd["noise_channels"]]
    target = target.detach().float()[:, :mtd["noise_channels"]]
    size = min(mtd["transport_size"], pred.shape[-2], pred.shape[-1])

    pred_probs, pred_current, pred_neighbours = _local_transport_distribution(
        pred, size, mtd["temperature"]
    )
    with torch.no_grad():
        target_probs, target_current, target_neighbours = _local_transport_distribution(
            target, size, mtd["temperature"]
        )

    # Sum over the 3x3 categorical distribution, then average over frame pairs
    # and spatial tokens. This avoids a loss scale proportional to size**2.
    local_kl = F.kl_div(
        pred_probs.clamp_min(1.0e-8).log(),
        target_probs,
        reduction="none",
    ).sum(dim=-1).mean()

    pred_transport = (
        pred_probs[..., None] * pred_neighbours
    ).sum(-2) - pred_current
    target_transport = (
        target_probs[..., None] * target_neighbours
    ).sum(-2) - target_current
    motion_residual = F.smooth_l1_loss(pred_transport, target_transport)

    pred_summary = F.normalize(
        pred.mean(dim=(-1, -2)).transpose(1, 2), dim=-1, eps=1.0e-6
    )
    target_summary = F.normalize(
        target.mean(dim=(-1, -2)).transpose(1, 2), dim=-1, eps=1.0e-6
    )
    pred_relation = pred_summary @ pred_summary.transpose(1, 2)
    target_relation = target_summary @ target_summary.transpose(1, 2)
    global_relation = F.kl_div(
        F.log_softmax(pred_relation, dim=-1),
        F.softmax(target_relation, dim=-1),
        reduction="batchmean",
    )

    # Return final weighted terms.  Each component already includes both its
    # internal coefficient and the global MTD coefficient, so its gradient is
    # exactly lambda_k * grad(L_k), matching the diagnostic definition.
    weighted_components = {
        "local": (
            mtd["total_weight"]
            * mtd["local_transport_weight"]
            * local_kl
        ),
        "motion": (
            mtd["total_weight"]
            * mtd["motion_residual_weight"]
            * motion_residual
        ),
        "global": (
            mtd["total_weight"]
            * mtd["global_relation_weight"]
            * global_relation
        ),
    }
    total = sum(weighted_components.values())
    if return_components:
        return total, weighted_components
    return total
