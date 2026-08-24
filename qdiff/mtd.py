"""Motion Transport Distillation (MTD) for video reconstruction.

This module is intentionally independent from the other research directions.
It only contains MTD configuration parsing and the three MTD loss terms.
"""

import hashlib
import math
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

    fine_weight = float(mtd.get("fine_weight", 0.0))
    fine_selection = str(mtd.get("fine_selection", "error")).lower()
    semantic_weighting = bool(mtd.get("semantic_weighting", False))
    semantic_importance = bool(
        mtd.get(
            "semantic_importance",
            semantic_weighting or fine_selection == "semantic",
        )
    )
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
        "semantic_importance": semantic_importance,
        "semantic_weighting": semantic_weighting,
        "weight_min": float(mtd.get("weight_min", 0.5)),
        "weight_max": float(mtd.get("weight_max", 1.5)),
        "semantic_percentile_low": float(
            mtd.get("semantic_percentile_low", 1.0)
        ),
        "semantic_percentile_high": float(
            mtd.get("semantic_percentile_high", 99.0)
        ),
        "semantic_pair_batch_size": int(
            mtd.get("semantic_pair_batch_size", 2)
        ),
        "attention_chunk_size": int(mtd.get("attention_chunk_size", 256)),
        "candidate_mode": str(
            mtd.get("candidate_mode", "fixed_3x3")
        ).lower(),
        "spatial_weighting": str(
            mtd.get("spatial_weighting", "uniform")
        ).lower(),
        "search_radius": int(mtd.get("search_radius", 2)),
        "topk": int(mtd.get("topk", 9)),
        "random_seed": int(mtd.get("random_seed", 42)),
        "importance_eps": float(mtd.get("importance_eps", 1.0e-8)),
        # The reconstruction loop sets this on a per-call copy.  Keeping it in
        # the normalized mapping makes random controls exactly resumable.
        "iteration": int(mtd.get("iteration", 0)),
        # Optional calibration-only coarse-to-fine key-region refinement.
        "fine_enabled": bool(mtd.get("fine_enabled", fine_weight > 0.0)),
        "fine_weight": fine_weight,
        "fine_transport_size": int(mtd.get("fine_transport_size", 32)),
        "fine_kernel_size": int(mtd.get("fine_kernel_size", 5)),
        "fine_selection": fine_selection,
        "fine_motion_weight": float(mtd.get("fine_motion_weight", 1.0)),
        "key_region_ratio": float(mtd.get("key_region_ratio", 0.25)),
        "key_region_block_size": int(mtd.get("key_region_block_size", 4)),
        "key_region_confidence": str(
            mtd.get("key_region_confidence", "none")
        ).lower(),
        "key_region_confidence_power": float(
            mtd.get("key_region_confidence_power", 1.0)
        ),
        "fine_selection_seed": int(mtd.get("fine_selection_seed", 42)),
    }
    if normalized["noise_channels"] <= 0:
        raise ValueError("noise_channels must be positive")
    if normalized["transport_size"] <= 0:
        raise ValueError("mtd.transport_size must be positive")
    if normalized["temperature"] <= 0:
        raise ValueError("mtd.temperature must be positive")
    if normalized["total_weight"] < 0:
        raise ValueError("mtd.total_weight must be non-negative")
    if normalized["weight_min"] < 0:
        raise ValueError("mtd.weight_min must be non-negative")
    if normalized["weight_max"] <= normalized["weight_min"]:
        raise ValueError("mtd.weight_max must exceed mtd.weight_min")
    if not (
        0.0 <= normalized["semantic_percentile_low"]
        < normalized["semantic_percentile_high"] <= 100.0
    ):
        raise ValueError("semantic percentiles must satisfy 0 <= low < high <= 100")
    if normalized["semantic_pair_batch_size"] <= 0:
        raise ValueError("mtd.semantic_pair_batch_size must be positive")
    if normalized["attention_chunk_size"] <= 0:
        raise ValueError("mtd.attention_chunk_size must be positive")
    if normalized["candidate_mode"] not in {
        "fixed_3x3", "random_topk", "teacher_centered_3x3"
    }:
        raise ValueError(
            "mtd.candidate_mode must be fixed_3x3, random_topk, or "
            "teacher_centered_3x3"
        )
    if normalized["spatial_weighting"] not in {
        "uniform", "fp_quant_error"
    }:
        raise ValueError(
            "mtd.spatial_weighting must be uniform or fp_quant_error"
        )
    if normalized["search_radius"] < 1:
        raise ValueError("mtd.search_radius must be at least 1")
    if normalized["topk"] <= 0:
        raise ValueError("mtd.topk must be positive")
    if normalized["importance_eps"] <= 0:
        raise ValueError("mtd.importance_eps must be positive")
    if normalized["fine_weight"] < 0:
        raise ValueError("mtd.fine_weight must be non-negative")
    if normalized["fine_transport_size"] <= 0:
        raise ValueError("mtd.fine_transport_size must be positive")
    if normalized["fine_kernel_size"] <= 0 or normalized["fine_kernel_size"] % 2 == 0:
        raise ValueError("mtd.fine_kernel_size must be a positive odd number")
    if normalized["fine_selection"] not in {"error", "random", "all", "semantic"}:
        raise ValueError(
            "mtd.fine_selection must be one of: error, random, all, semantic"
        )
    if normalized["fine_motion_weight"] < 0:
        raise ValueError("mtd.fine_motion_weight must be non-negative")
    if not 0.0 < normalized["key_region_ratio"] <= 1.0:
        raise ValueError("mtd.key_region_ratio must lie in (0, 1]")
    if normalized["key_region_block_size"] <= 0:
        raise ValueError("mtd.key_region_block_size must be positive")
    if normalized["key_region_confidence"] not in {"none", "entropy"}:
        raise ValueError(
            "mtd.key_region_confidence must be one of: none, entropy"
        )
    if normalized["key_region_confidence_power"] < 0:
        raise ValueError(
            "mtd.key_region_confidence_power must be non-negative"
        )
    if (
        normalized["candidate_mode"] != "fixed_3x3"
        and normalized["spatial_weighting"] != "uniform"
    ):
        raise ValueError(
            "adaptive candidates and spatial importance weighting are separate "
            "v1 ablation arms"
        )
    if normalized["semantic_weighting"] and (
        normalized["candidate_mode"] != "fixed_3x3"
        or normalized["spatial_weighting"] != "uniform"
    ):
        raise ValueError(
            "semantic weighting requires fixed_3x3 candidates and uniform "
            "spatial weighting"
        )
    if normalized["fine_selection"] == "semantic" and not normalized["semantic_importance"]:
        raise ValueError("semantic fine selection requires semantic_importance")
    if normalized["fine_enabled"] and (
        normalized["candidate_mode"] != "fixed_3x3"
        or normalized["spatial_weighting"] != "uniform"
    ):
        raise ValueError(
            "fine key-region refinement requires fixed_3x3 candidates and "
            "uniform coarse weighting in the controlled L2 ablation"
        )
    return normalized


def _local_transport_distribution(
    features, size, temperature, kernel_size=3, return_valid=False
):
    """Build a local correspondence distribution for every adjacent-frame token."""
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd number")
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
    radius = kernel_size // 2
    neighbours = F.unfold(
        following_flat, kernel_size=kernel_size, padding=radius
    )
    neighbours = neighbours.reshape(
        batch * (frames - 1), channels, kernel_size * kernel_size, size * size
    ).permute(0, 3, 2, 1)

    logits = (current_flat[:, :, None, :] * neighbours).sum(-1) / temperature

    # Padding introduced by unfold is not a real candidate. Mask it so corner,
    # edge and interior tokens have exactly 4, 6 and 9 valid neighbours.
    valid_grid = following_flat.new_ones(
        batch * (frames - 1), 1, size, size
    )
    valid_neighbours = F.unfold(
        valid_grid, kernel_size=kernel_size, padding=radius
    )
    valid_neighbours = valid_neighbours.reshape(
        batch * (frames - 1), 1, kernel_size * kernel_size, size * size
    ).permute(0, 3, 2, 1).squeeze(-1).bool()
    logits = logits.masked_fill(
        ~valid_neighbours, torch.finfo(logits.dtype).min
    )
    probabilities = F.softmax(logits, dim=-1)
    if return_valid:
        return probabilities, current_flat, neighbours, valid_neighbours
    return probabilities, current_flat, neighbours


def _teacher_matching_confidence(probabilities, valid, eps=1.0e-8):
    """Detached normalized inverse entropy of the teacher correspondence."""
    probabilities = probabilities.detach()
    valid = valid.detach()
    safe_probabilities = probabilities.masked_fill(~valid, 0.0)
    entropy = -(
        safe_probabilities
        * safe_probabilities.clamp_min(eps).log()
    ).sum(dim=-1)
    candidate_count = valid.sum(dim=-1).clamp_min(2).to(entropy.dtype)
    confidence = 1.0 - entropy / candidate_count.log()
    return confidence.clamp(0.0, 1.0)


def _importance_weights(pred_current, target_current, eps):
    """Detached mean-one FP/Q-error weights for each frame-pair query."""
    error = (pred_current - target_current).square().mean(dim=-1).detach()
    count = error.shape[-1]
    return count * (error + eps) / (
        error.sum(dim=-1, keepdim=True) + count * eps
    )


def _weighted_query_mean(values, weights):
    return (values * weights).mean(dim=-1).mean()


def _block_topk_mask(scores, size, block_size, ratio):
    """Select the same fraction of tokens from each coarse spatial block."""
    if size % block_size:
        raise ValueError(
            "mtd.key_region_block_size must divide the coarse transport size"
        )
    count = max(1, math.ceil(block_size * block_size * ratio))
    grid = scores.reshape(-1, size, size)
    mask = torch.zeros_like(grid, dtype=torch.bool)
    for y in range(0, size, block_size):
        for x in range(0, size, block_size):
            block = grid[:, y : y + block_size, x : x + block_size].reshape(
                grid.shape[0], -1
            )
            indices = block.topk(count, dim=-1).indices
            selected = torch.zeros_like(block, dtype=torch.bool)
            selected.scatter_(1, indices, True)
            mask[:, y : y + block_size, x : x + block_size] = selected.reshape(
                -1, block_size, block_size
            )
    return mask.flatten(1)


def _deterministic_random_scores(shape, seed, device):
    """Create a fixed random control without advancing the CUDA RNG stream."""
    indices = torch.arange(
        math.prod(shape), device=device, dtype=torch.int64
    ).reshape(shape)
    values = (indices * 1103515245 + int(seed) * 12345) % 2147483647
    return values.float()


def _key_region_mask(
    local_error,
    motion_error,
    size,
    mtd,
    teacher_probs=None,
    teacher_valid=None,
):
    selection = mtd["fine_selection"]
    if selection == "all":
        return torch.ones_like(local_error, dtype=torch.bool)
    if selection == "random":
        scores = _deterministic_random_scores(
            local_error.shape, mtd["fine_selection_seed"], local_error.device
        )
    else:
        # Only the discrete selector is detached.  The original coarse losses
        # below retain their normal gradients.
        local = local_error.detach()
        motion = motion_error.detach()
        local_scale = local.mean(dim=-1, keepdim=True).clamp_min(1.0e-12)
        motion_scale = motion.mean(dim=-1, keepdim=True).clamp_min(1.0e-12)
        scores = local / local_scale + motion / motion_scale
        if mtd["key_region_confidence"] == "entropy":
            if teacher_probs is None or teacher_valid is None:
                raise ValueError(
                    "entropy-guided key-region selection requires teacher "
                    "probabilities and valid-candidate masks"
                )
            confidence = _teacher_matching_confidence(
                teacher_probs, teacher_valid, mtd["importance_eps"]
            )
            scores = scores * confidence.pow(
                mtd["key_region_confidence_power"]
            )
    return _block_topk_mask(
        scores,
        size,
        mtd["key_region_block_size"],
        mtd["key_region_ratio"],
    )


def _masked_mean(values, mask):
    weights = mask.to(values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def _fixed_transport_terms(
    pred,
    target,
    size,
    temperature,
    weighting,
    eps,
    semantic_weights=None,
):
    pred_probs, pred_current, pred_neighbours = _local_transport_distribution(
        pred, size, temperature
    )
    with torch.no_grad():
        target_probs, target_current, target_neighbours, target_valid = (
            _local_transport_distribution(
                target, size, temperature, return_valid=True
            )
        )

    local_per_query = F.kl_div(
        pred_probs.clamp_min(1.0e-8).log(),
        target_probs,
        reduction="none",
    ).sum(dim=-1)
    pred_transport = (
        pred_probs[..., None] * pred_neighbours
    ).sum(-2) - pred_current
    target_transport = (
        target_probs[..., None] * target_neighbours
    ).sum(-2) - target_current
    motion_per_query = F.smooth_l1_loss(
        pred_transport, target_transport, reduction="none"
    ).mean(dim=-1)

    diagnostics = {}
    if semantic_weights is not None:
        weights = semantic_weights.reshape_as(local_per_query).detach().float()
        if not torch.isfinite(weights).all() or torch.any(weights < 0):
            raise ValueError("MTD semantic importance weights must be finite and non-negative")
        denominator = weights.sum().clamp_min(eps)
        local_kl = (local_per_query * weights).sum() / denominator
        motion_residual = F.smooth_l1_loss(
            pred_transport, target_transport
        )
        diagnostics = {
            "semantic_weight_mean": weights.mean().detach(),
            "semantic_weight_min": weights.min().detach(),
            "semantic_weight_max": weights.max().detach(),
        }
    elif weighting == "fp_quant_error":
        weights = _importance_weights(pred_current, target_current, eps)
        local_kl = _weighted_query_mean(local_per_query, weights)
        motion_residual = _weighted_query_mean(motion_per_query, weights)
        diagnostics = {
            "importance_weight_mean": weights.mean().detach(),
            "importance_weight_max": weights.max().detach(),
            "importance_weight_top25_mass": (
                weights.topk(max(1, (weights.shape[-1] + 3) // 4), dim=-1)
                .values.sum(dim=-1).mean() / weights.shape[-1]
            ).detach(),
        }
    else:
        # Keep the released reduction exactly unchanged for arm A.
        local_kl = local_per_query.mean()
        motion_residual = F.smooth_l1_loss(
            pred_transport, target_transport
        )
    return (
        local_kl,
        motion_residual,
        diagnostics,
        local_per_query,
        motion_per_query,
        target_probs,
        target_valid,
    )


def _fine_transport_terms(
    pred,
    target,
    coarse_local_error,
    coarse_motion_error,
    coarse_teacher_probs,
    coarse_teacher_valid,
    coarse_size,
    mtd,
    fine_selection_scores=None,
):
    """Refine selected coarse regions at a denser but equally local scale."""
    fine_size = min(
        mtd["fine_transport_size"], pred.shape[-2], pred.shape[-1]
    )
    if fine_size <= coarse_size:
        raise ValueError(
            "mtd.fine_transport_size must exceed the effective coarse size"
        )
    coarse_mask = None
    if mtd["fine_enabled"] and mtd["fine_selection"] == "semantic":
        expected_shape = (
            pred.shape[0] * (pred.shape[2] - 1),
            fine_size * fine_size,
        )
        if fine_selection_scores is None:
            raise ValueError(
                "semantic fine selection requires fine_selection_scores"
            )
        if tuple(fine_selection_scores.shape) != expected_shape:
            raise ValueError(
                "fine_selection_scores must have shape "
                f"{expected_shape}, got {tuple(fine_selection_scores.shape)}"
            )
        scale = fine_size // coarse_size
        if fine_size % coarse_size or scale <= 0:
            raise ValueError("fine transport size must be an integer multiple of coarse size")
        fine_mask = _block_topk_mask(
            fine_selection_scores.detach().float(),
            fine_size,
            mtd["key_region_block_size"] * scale,
            mtd["key_region_ratio"],
        )
    else:
        coarse_mask = _key_region_mask(
            coarse_local_error,
            coarse_motion_error,
            coarse_size,
            mtd,
            teacher_probs=coarse_teacher_probs,
            teacher_valid=coarse_teacher_valid,
        )
        fine_mask = F.interpolate(
            coarse_mask.reshape(-1, 1, coarse_size, coarse_size).float(),
            size=(fine_size, fine_size),
            mode="nearest",
        ).reshape(coarse_mask.shape[0], fine_size * fine_size).bool()

    pred_probs, pred_current, pred_neighbours = _local_transport_distribution(
        pred,
        fine_size,
        mtd["temperature"],
        kernel_size=mtd["fine_kernel_size"],
    )
    with torch.no_grad():
        target_probs, target_current, target_neighbours = (
            _local_transport_distribution(
                target,
                fine_size,
                mtd["temperature"],
                kernel_size=mtd["fine_kernel_size"],
            )
        )
    fine_local_per_query = F.kl_div(
        pred_probs.clamp_min(1.0e-8).log(),
        target_probs,
        reduction="none",
    ).sum(dim=-1)
    pred_transport = (
        pred_probs[..., None] * pred_neighbours
    ).sum(-2) - pred_current
    target_transport = (
        target_probs[..., None] * target_neighbours
    ).sum(-2) - target_current
    fine_motion_per_query = F.smooth_l1_loss(
        pred_transport, target_transport, reduction="none"
    ).mean(dim=-1)
    fine_local = _masked_mean(fine_local_per_query, fine_mask)
    fine_motion = _masked_mean(fine_motion_per_query, fine_mask)
    diagnostics = {
        "fine_selected_fraction": fine_mask.float().mean().detach(),
        "fine_local_unweighted": fine_local.detach(),
        "fine_motion_unweighted": fine_motion.detach(),
    }
    if (
        coarse_mask is not None
        and coarse_teacher_probs is not None
        and coarse_teacher_valid is not None
    ):
        confidence = _teacher_matching_confidence(
            coarse_teacher_probs,
            coarse_teacher_valid,
            mtd["importance_eps"],
        )
        diagnostics.update(
            {
                "teacher_confidence_mean": confidence.mean().detach(),
                "teacher_confidence_selected": _masked_mean(
                    confidence, coarse_mask
                ).detach(),
                "teacher_confidence_unselected": _masked_mean(
                    confidence, ~coarse_mask
                ).detach(),
            }
        )
    return fine_local, fine_motion, diagnostics


def _prepare_candidate_bank(features, size, radius):
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
    pairs = batch * (frames - 1)
    tokens = size * size
    kernel = 2 * radius + 1
    current_flat = current.permute(0, 1, 3, 4, 2).reshape(
        pairs, tokens, channels
    )
    following_flat = following.reshape(pairs, channels, size, size)
    neighbours = F.unfold(
        following_flat, kernel_size=kernel, padding=radius
    ).reshape(pairs, channels, kernel * kernel, tokens).permute(0, 3, 2, 1)
    valid_grid = following_flat.new_ones(pairs, 1, size, size)
    valid = F.unfold(
        valid_grid, kernel_size=kernel, padding=radius
    ).reshape(pairs, kernel * kernel, tokens).permute(0, 2, 1).bool()
    offsets = torch.stack(
        torch.meshgrid(
            torch.arange(-radius, radius + 1, device=features.device),
            torch.arange(-radius, radius + 1, device=features.device),
            indexing="ij",
        ),
        dim=-1,
    ).reshape(-1, 2)
    return current_flat, neighbours, valid, offsets


def _masked_softmax(logits, mask):
    floor = torch.finfo(logits.dtype).min
    return F.softmax(logits.masked_fill(~mask, floor), dim=-1).masked_fill(
        ~mask, 0.0
    )


def _random_candidate_mask(valid, topk, seed, iteration):
    generator = torch.Generator(device=valid.device)
    generator.manual_seed(
        (int(seed) + int(iteration) * 1_000_003) % (2**63 - 1)
    )
    scores = torch.rand(
        valid.shape, generator=generator, device=valid.device
    ).masked_fill(~valid, torch.finfo(torch.float32).min)
    k = min(int(topk), scores.shape[-1])
    selected = torch.zeros_like(valid)
    selected.scatter_(-1, scores.topk(k, dim=-1).indices, True)
    return selected & valid


def _teacher_centered_mask(valid, offsets, teacher_logits, wide_mask):
    centre_indices = teacher_logits.masked_fill(
        ~wide_mask, torch.finfo(teacher_logits.dtype).min
    ).argmax(dim=-1)
    centres = offsets[centre_indices]
    delta = offsets.view(1, 1, -1, 2) - centres.unsqueeze(-2)
    return valid & (delta.abs().amax(dim=-1) <= 1), centres


def _adaptive_transport_terms(pred, target, size, mtd):
    search_radius = mtd["search_radius"]
    bank_radius = search_radius + 1
    pred_current, pred_neighbours, valid, offsets = _prepare_candidate_bank(
        pred, size, bank_radius
    )
    with torch.no_grad():
        target_current, target_neighbours, target_valid, _ = (
            _prepare_candidate_bank(target, size, bank_radius)
        )
    if not torch.equal(valid, target_valid):
        raise RuntimeError("prediction/target candidate validity differs")

    pred_logits = (
        pred_current[:, :, None, :] * pred_neighbours
    ).sum(dim=-1) / mtd["temperature"]
    with torch.no_grad():
        target_logits = (
            target_current[:, :, None, :] * target_neighbours
        ).sum(dim=-1) / mtd["temperature"]
    wide_mask = valid & (
        offsets.abs().amax(dim=-1).view(1, 1, -1) <= search_radius
    )
    centres = None
    if mtd["candidate_mode"] == "random_topk":
        candidate_mask = _random_candidate_mask(
            wide_mask, mtd["topk"], mtd["random_seed"], mtd["iteration"]
        )
    elif mtd["candidate_mode"] == "teacher_centered_3x3":
        candidate_mask, centres = _teacher_centered_mask(
            valid, offsets, target_logits, wide_mask
        )
    else:
        raise RuntimeError("adaptive transport called for fixed candidate mode")

    pred_probs = _masked_softmax(pred_logits, candidate_mask)
    target_probs = _masked_softmax(target_logits, candidate_mask)
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

    with torch.no_grad():
        full_target_probs = _masked_softmax(target_logits, wide_mask)
        full_top1 = full_target_probs.argmax(dim=-1)
        diagnostics = {
            "candidate_count": candidate_mask.sum(dim=-1).float().mean(),
            "teacher_mass_covered": (
                full_target_probs * candidate_mask
            ).sum(dim=-1).mean(),
            "teacher_top1_retained": candidate_mask.gather(
                -1, full_top1.unsqueeze(-1)
            ).squeeze(-1).float().mean(),
        }
        if centres is not None:
            width = 2 * search_radius + 1
            flat_centres = (
                (centres[..., 0] + search_radius) * width
                + centres[..., 1] + search_radius
            ).flatten()
            diagnostics["coarse_offset_histogram"] = torch.bincount(
                flat_centres, minlength=width * width
            ).reshape(width, width)
    return local_kl, motion_residual, diagnostics


def motion_transport_distillation(
    pred,
    target,
    config=None,
    return_components=False,
    return_diagnostics=False,
    importance_weights=None,
    fine_selection_scores=None,
):
    """Compute local-distribution, transport-residual and global-relation MTD."""
    mtd = normalize_mtd_config(config)
    if not mtd["enabled"] or pred.ndim != 5 or pred.shape[2] < 2:
        zero = pred.new_tensor(0.0)
        components = {
            "local": zero,
            "motion": zero,
            "global": zero,
            "fine": zero,
        }
        if return_components and return_diagnostics:
            return zero, components, {}
        if return_components:
            return zero, components
        if return_diagnostics:
            return zero, {}
        return zero

    # Run the terminal loss in FP32 for BF16/FP16 FlashAttention stability.
    # The cast remains differentiable and gradients return in the model dtype.
    pred = pred.float()[:, :mtd["noise_channels"]]
    target = target.detach().float()[:, :mtd["noise_channels"]]
    size = min(mtd["transport_size"], pred.shape[-2], pred.shape[-1])
    semantic_weights = None
    if mtd["semantic_weighting"]:
        if importance_weights is None:
            raise ValueError(
                "semantic MTD is enabled but no importance_weights were provided"
            )
        expected_shape = (pred.shape[0], pred.shape[2] - 1, size * size)
        if tuple(importance_weights.shape) != expected_shape:
            raise ValueError(
                "MTD importance_weights must have shape "
                f"{expected_shape}, got {tuple(importance_weights.shape)}"
            )
        semantic_weights = importance_weights.to(device=pred.device)
    if mtd["fine_enabled"] and mtd["fine_selection"] == "semantic":
        if fine_selection_scores is None:
            raise ValueError(
                "semantic fine selection is enabled but no fine_selection_scores "
                "were provided"
            )
        effective_fine_size = min(
            mtd["fine_transport_size"], pred.shape[-2], pred.shape[-1]
        )
        expected_fine_shape = (
            pred.shape[0], pred.shape[2] - 1,
            effective_fine_size ** 2,
        )
        if tuple(fine_selection_scores.shape) != expected_fine_shape:
            raise ValueError(
                "fine_selection_scores must have shape "
                f"{expected_fine_shape}, got {tuple(fine_selection_scores.shape)}"
            )
        fine_selection_scores = fine_selection_scores.reshape(
            pred.shape[0] * (pred.shape[2] - 1), -1
        ).to(device=pred.device)

    if mtd["candidate_mode"] == "fixed_3x3":
        (
            local_kl,
            motion_residual,
            diagnostics,
            local_per_query,
            motion_per_query,
            coarse_teacher_probs,
            coarse_teacher_valid,
        ) = _fixed_transport_terms(
            pred,
            target,
            size,
            mtd["temperature"],
            mtd["spatial_weighting"],
            mtd["importance_eps"],
            semantic_weights=semantic_weights,
        )
    else:
        local_kl, motion_residual, diagnostics = _adaptive_transport_terms(
            pred, target, size, mtd
        )
        local_per_query = motion_per_query = None
        coarse_teacher_probs = coarse_teacher_valid = None

    fine_refinement = pred.new_tensor(0.0)
    if mtd["fine_enabled"]:
        fine_local, fine_motion, fine_diagnostics = _fine_transport_terms(
            pred,
            target,
            local_per_query,
            motion_per_query,
            coarse_teacher_probs,
            coarse_teacher_valid,
            size,
            mtd,
            fine_selection_scores=fine_selection_scores,
        )
        fine_refinement = mtd["fine_weight"] * (
            mtd["local_transport_weight"] * fine_local
            + mtd["motion_residual_weight"]
            * mtd["fine_motion_weight"]
            * fine_motion
        )
        diagnostics.update(fine_diagnostics)

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
        "fine": mtd["total_weight"] * fine_refinement,
    }
    total = sum(weighted_components.values())
    if return_components and return_diagnostics:
        return total, weighted_components, diagnostics
    if return_components:
        return total, weighted_components
    if return_diagnostics:
        return total, diagnostics
    return total


# ---------------------------------------------------------------------------
# MTD-v2: schedule-consistent x0 transport distillation.
# ---------------------------------------------------------------------------


def normalize_mtd_v2_config(config=None):
    """Normalize the independent MTD-v2 calibration configuration."""
    config = _plain_dict(config)
    already_normalized = "enabled" in config and "coarse_size" in config
    method = config if "frame_axis" in config else _plain_dict(config.get("method"))
    mtd = config if already_normalized else _plain_dict(config.get("mtd_v2"))
    offsets = tuple(int(value) for value in mtd.get("temporal_offsets", (1, 2, 4)))
    normalized = {
        "enabled": bool(config["enabled"]) if already_normalized else (
            str(method.get("frame_axis", "BASELINE")).upper() == "MTD_V2"
        ),
        "noise_channels": int(config.get("noise_channels", 4)),
        # Controlled representation ablation. ``x0`` preserves the original
        # MTD-v2 behavior; ``epsilon`` changes only the coarse/fine transport
        # tensors. SNR weighting and the auxiliary global x0 term are kept.
        "feature_source": str(mtd.get("feature_source", "x0")).lower(),
        "total_weight": float(mtd.get("total_weight", 1.0)),
        "coarse_size": int(mtd.get("coarse_size", 16)),
        "fine_size": int(mtd.get("fine_size", 32)),
        "fine_radius": int(mtd.get("fine_radius", 2)),
        "coarse_topk": int(mtd.get("coarse_topk", 16)),
        "temporal_offsets": offsets,
        "temperature": float(mtd.get("temperature", 0.07)),
        "global_temperature": float(mtd.get("global_temperature", 1.0)),
        "confidence_min": float(mtd.get("confidence_min", 0.15)),
        "confidence_power": float(mtd.get("confidence_power", 1.0)),
        "cycle_enabled": bool(mtd.get("cycle_enabled", False)),
        "cycle_min": float(mtd.get("cycle_min", 0.05)),
        "cycle_power": float(mtd.get("cycle_power", 1.0)),
        "snr_full_weight": float(mtd.get("snr_full_weight", 1.0)),
        "correspondence_weight": float(mtd.get("correspondence_weight", 1.0)),
        "flow_weight": float(mtd.get("flow_weight", 0.5)),
        "global_relation_weight": float(mtd.get("global_relation_weight", 0.1)),
        "fine_scale_weight": float(mtd.get("fine_scale_weight", 0.5)),
        "flow_huber_beta": float(mtd.get("flow_huber_beta", 0.05)),
        "eps": float(mtd.get("eps", 1.0e-8)),
    }
    if normalized["noise_channels"] <= 0:
        raise ValueError("mtd_v2 noise_channels must be positive")
    if normalized["feature_source"] not in {"x0", "epsilon"}:
        raise ValueError("mtd_v2 feature_source must be x0 or epsilon")
    if normalized["coarse_size"] <= 1 or normalized["fine_size"] <= 1:
        raise ValueError("mtd_v2 coarse_size and fine_size must exceed one")
    if normalized["fine_size"] < normalized["coarse_size"]:
        raise ValueError("mtd_v2 fine_size must be at least coarse_size")
    if normalized["fine_radius"] < 1:
        raise ValueError("mtd_v2 fine_radius must be at least one")
    if not offsets or any(value <= 0 for value in offsets):
        raise ValueError("mtd_v2 temporal_offsets must contain positive offsets")
    if len(set(offsets)) != len(offsets):
        raise ValueError("mtd_v2 temporal_offsets must not contain duplicates")
    if normalized["coarse_topk"] <= 0:
        raise ValueError("mtd_v2 coarse_topk must be positive")
    if normalized["temperature"] <= 0 or normalized["global_temperature"] <= 0:
        raise ValueError("mtd_v2 temperatures must be positive")
    if not 0.0 <= normalized["confidence_min"] < 1.0:
        raise ValueError("mtd_v2 confidence_min must lie in [0, 1)")
    if normalized["confidence_power"] <= 0:
        raise ValueError("mtd_v2 confidence_power must be positive")
    if not 0.0 <= normalized["cycle_min"] < 1.0:
        raise ValueError("mtd_v2 cycle_min must lie in [0, 1)")
    if normalized["cycle_power"] <= 0:
        raise ValueError("mtd_v2 cycle_power must be positive")
    if normalized["snr_full_weight"] <= 0:
        raise ValueError("mtd_v2 snr_full_weight must be positive")
    if any(
        normalized[name] < 0
        for name in (
            "total_weight",
            "correspondence_weight",
            "flow_weight",
            "global_relation_weight",
            "fine_scale_weight",
        )
    ):
        raise ValueError("mtd_v2 loss weights must be non-negative")
    if normalized["flow_huber_beta"] <= 0 or normalized["eps"] <= 0:
        raise ValueError("mtd_v2 flow_huber_beta and eps must be positive")
    return normalized


def build_mtd_v2_schedule_context(scheduler):
    """Extract the exact epsilon-to-x0 coefficients from an IDDPM scheduler."""
    required = (
        "sqrt_recip_alphas_cumprod",
        "sqrt_recipm1_alphas_cumprod",
    )
    missing = [name for name in required if not hasattr(scheduler, name)]
    if missing:
        raise ValueError(
            "MTD-v2 requires an epsilon-parameterized IDDPM scheduler with "
            f"{', '.join(missing)}"
        )
    coeff_a = torch.as_tensor(
        getattr(scheduler, "sqrt_recip_alphas_cumprod"), dtype=torch.float32
    ).cpu().contiguous()
    coeff_b = torch.as_tensor(
        getattr(scheduler, "sqrt_recipm1_alphas_cumprod"), dtype=torch.float32
    ).cpu().contiguous()
    if coeff_a.ndim != 1 or coeff_b.ndim != 1 or coeff_a.shape != coeff_b.shape:
        raise ValueError("MTD-v2 scheduler coefficients must be equal-length vectors")
    digest = hashlib.sha256()
    digest.update(coeff_a.numpy().tobytes())
    digest.update(coeff_b.numpy().tobytes())
    context = {
        "sqrt_recip_alphas_cumprod": coeff_a,
        "sqrt_recipm1_alphas_cumprod": coeff_b,
    }
    # IDDPM uses SpacedDiffusion for DDIM sampling.  Its denoiser receives
    # original-diffusion timestep IDs while its coefficient vectors are indexed
    # by the respaced solver step.  Preserve the scheduler's authoritative map
    # so x0 reconstruction never guesses a conversion between the two.
    timestep_map = getattr(scheduler, "timestep_map", None)
    if timestep_map is not None:
        timestep_map = torch.as_tensor(timestep_map, dtype=torch.long).cpu().contiguous()
        if timestep_map.ndim != 1 or timestep_map.numel() != coeff_a.numel():
            raise ValueError(
                "MTD-v2 scheduler timestep_map must align with its coefficient vectors"
            )
        digest.update(timestep_map.numpy().tobytes())
        context["timestep_map"] = timestep_map
    context["signature"] = digest.hexdigest()
    return context


def _mtd_v2_tokens(features, size):
    batch, channels, frames, height, width = features.shape
    pooled = F.adaptive_avg_pool2d(
        features.permute(0, 2, 1, 3, 4).reshape(
            batch * frames, channels, height, width
        ),
        (size, size),
    ).reshape(batch, frames, channels, size, size)
    return F.normalize(
        pooled.permute(0, 1, 3, 4, 2).reshape(
            batch, frames, size * size, channels
        ),
        dim=-1,
        eps=1.0e-6,
    )


def _mtd_v2_coordinates(size, device, dtype):
    axis = torch.linspace(-1.0, 1.0, size, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack((grid_x, grid_y), dim=-1).reshape(size * size, 2)


def _mtd_v2_confidence(probabilities, valid, mtd):
    valid_count = valid.sum(dim=-1).clamp_min(2).to(probabilities.dtype)
    safe = probabilities.masked_fill(~valid, 0.0)
    entropy = -(safe * safe.clamp_min(mtd["eps"]).log()).sum(dim=-1)
    confidence = (1.0 - entropy / valid_count.log()).clamp(0.0, 1.0)
    weights = confidence.pow(mtd["confidence_power"])
    weights = torch.where(
        confidence >= mtd["confidence_min"], weights, torch.zeros_like(weights)
    )
    return weights.detach(), entropy.detach(), confidence.detach()


def _mtd_v2_weighted_mean(values, query_weights, sample_weights, eps):
    """Average valid queries per sample, then apply the per-timestep SNR weight.

    Normalizing by the sum of ``sample_weights`` would cancel SNR weighting for
    a batch that shares one timestep (the common calibration case).  Query
    confidence is a validity/ambiguity normalization; SNR is an intended loss
    scale and therefore remains outside that normalization.
    """
    per_sample_weight = query_weights.sum(dim=(1, 2))
    per_sample = (values * query_weights).sum(dim=(1, 2)) / per_sample_weight.clamp_min(eps)
    per_sample = torch.where(
        per_sample_weight > 0, per_sample, torch.zeros_like(per_sample)
    )
    return (per_sample * sample_weights).mean()


def _mtd_v2_soft_flow(probabilities, candidate_coordinates, query_coordinates, offset):
    matched = torch.einsum("bpqk,bpqkd->bpqd", probabilities, candidate_coordinates)
    return (matched - query_coordinates.view(1, 1, -1, 2)) / float(offset)


def _mtd_v2_soft_cycle_gate(forward_probs, backward_probs, mtd):
    """Return detached teacher forward-backward consistency weights.

    ``forward_probs`` maps query positions at frame t to keys at t+offset;
    ``backward_probs`` maps those keys back to the original query grid.  The
    diagonal of their soft composition is the probability of returning to the
    same position without imposing a brittle argmax match.
    """
    cycle_score = torch.einsum("bpqk,bpkq->bpq", forward_probs, backward_probs)
    if not mtd["cycle_enabled"]:
        return torch.ones_like(cycle_score), cycle_score.detach()
    cycle_gate = (
        (cycle_score - mtd["cycle_min"])
        / max(1.0 - mtd["cycle_min"], mtd["eps"])
    ).clamp(0.0, 1.0).pow(mtd["cycle_power"])
    return cycle_gate.detach(), cycle_score.detach()


def _mtd_v2_coarse_terms(pred_tokens, target_tokens, offset, mtd, sample_weights):
    current = pred_tokens[:, :-offset]
    following = pred_tokens[:, offset:]
    with torch.no_grad():
        target_current = target_tokens[:, :-offset]
        target_following = target_tokens[:, offset:]
        target_logits = torch.einsum(
            "bpqc,bpkc->bpqk", target_current, target_following
        ) / mtd["temperature"]
        target_probs = F.softmax(target_logits, dim=-1)
        target_backward_logits = torch.einsum(
            "bpkc,bpqc->bpkq", target_following, target_current
        ) / mtd["temperature"]
        target_backward_probs = F.softmax(target_backward_logits, dim=-1)
        cycle_gate, cycle_score = _mtd_v2_soft_cycle_gate(
            target_probs, target_backward_probs, mtd
        )
    pred_logits = torch.einsum("bpqc,bpkc->bpqk", current, following) / mtd[
        "temperature"
    ]
    pred_log_probs = F.log_softmax(pred_logits, dim=-1)
    pred_probs = pred_log_probs.exp()
    topk = min(mtd["coarse_topk"], target_probs.shape[-1])
    target_top_probs, target_top_indices = target_probs.topk(topk, dim=-1)
    target_top_mass = target_top_probs.sum(dim=-1)
    # Keep the teacher's top-k distribution conditional on its selected support,
    # then add one explicit ``outside`` bin.  This is a proper coarsened KL:
    # it is zero for identical teacher/student predictions while still making
    # probability mass outside the teacher candidate set observable.
    target_top_conditional = target_top_probs / target_top_mass.unsqueeze(-1).clamp_min(
        mtd["eps"]
    )
    target_log_probs = target_top_conditional.clamp_min(mtd["eps"]).log()
    pred_top_log_probs = pred_log_probs.gather(-1, target_top_indices)
    pred_top_mass = pred_top_log_probs.exp().sum(dim=-1).clamp(max=1.0)
    pred_top_conditional_log_probs = pred_top_log_probs - pred_top_mass.clamp_min(
        mtd["eps"]
    ).log().unsqueeze(-1)
    selected_kl = target_top_mass * (
        target_top_conditional
        * (target_log_probs - pred_top_conditional_log_probs)
    ).sum(dim=-1)
    # ``selected_kl`` compares only the distributions conditional on the
    # selected teacher support.  Include the KL of the support-bin masses as
    # well: omitting it permits a negative sum when the student's total mass
    # on the support differs from the teacher's.
    top_mass_kl = target_top_mass * (
        target_top_mass.clamp_min(mtd["eps"]).log()
        - pred_top_mass.clamp_min(mtd["eps"]).log()
    )
    target_outside_mass = (1.0 - target_top_mass).clamp_min(0.0)
    pred_outside_mass = (1.0 - pred_top_mass).clamp_min(mtd["eps"])
    outside_kl = target_outside_mass * (
        target_outside_mass.clamp_min(mtd["eps"]).log() - pred_outside_mass.log()
    )
    correspondence_per_query = selected_kl + top_mass_kl + outside_kl
    valid = torch.ones_like(target_probs, dtype=torch.bool)
    query_weights, entropy, confidence = _mtd_v2_confidence(
        target_probs, valid, mtd
    )
    query_weights = query_weights * cycle_gate
    size = int(round(target_probs.shape[-1] ** 0.5))
    coordinates = _mtd_v2_coordinates(size, pred_tokens.device, pred_tokens.dtype)
    target_coordinates = coordinates.view(1, 1, 1, -1, 2).expand(
        *target_probs.shape, 2
    )
    pred_flow = _mtd_v2_soft_flow(
        pred_probs,
        target_coordinates,
        coordinates,
        offset,
    )
    with torch.no_grad():
        target_flow = _mtd_v2_soft_flow(
            target_probs,
            target_coordinates,
            coordinates,
            offset,
        )
    flow_per_query = F.smooth_l1_loss(
        pred_flow,
        target_flow,
        beta=mtd["flow_huber_beta"],
        reduction="none",
    ).mean(dim=-1)
    diagnostics = {
        "entropy": entropy,
        "confidence": confidence,
        "topk_mass": target_top_mass.detach(),
        "flow_mae": (pred_flow - target_flow).abs().mean(dim=-1).detach(),
        "effective_queries": (query_weights > 0).sum().detach(),
        "query_weight_sum": query_weights.sum().detach(),
        "cycle_gate": cycle_gate,
        "cycle_score": cycle_score,
        "cycle_pass_ratio": (cycle_gate > 0).float().mean().detach(),
    }
    return (
        _mtd_v2_weighted_mean(
            correspondence_per_query, query_weights, sample_weights, mtd["eps"]
        ),
        _mtd_v2_weighted_mean(
            flow_per_query, query_weights, sample_weights, mtd["eps"]
        ),
        target_flow.detach(),
        diagnostics,
    )


def _mtd_v2_fine_candidates(centres, size, radius):
    batch, pairs, tokens, _ = centres.shape
    coordinates = _mtd_v2_coordinates(size, centres.device, centres.dtype)
    centre_xy = ((centres + 1.0) * 0.5 * (size - 1)).round().long()
    offsets_y, offsets_x = torch.meshgrid(
        torch.arange(-radius, radius + 1, device=centres.device),
        torch.arange(-radius, radius + 1, device=centres.device),
        indexing="ij",
    )
    offsets = torch.stack((offsets_x, offsets_y), dim=-1).reshape(-1, 2)
    candidate_xy = centre_xy.unsqueeze(-2) + offsets.view(1, 1, 1, -1, 2)
    valid = (
        (candidate_xy[..., 0] >= 0)
        & (candidate_xy[..., 0] < size)
        & (candidate_xy[..., 1] >= 0)
        & (candidate_xy[..., 1] < size)
    )
    safe_xy = candidate_xy.clamp(0, size - 1)
    candidate_indices = safe_xy[..., 1] * size + safe_xy[..., 0]
    candidate_coordinates = coordinates[candidate_indices]
    return candidate_indices, candidate_coordinates, valid


def _mtd_v2_fine_terms(
    pred_tokens,
    target_tokens,
    target_coarse_flow,
    target_coarse_cycle_gate,
    coarse_size,
    fine_size,
    offset,
    mtd,
    sample_weights,
):
    batch, pairs, _, _ = target_coarse_flow.shape
    upsampled_centres = F.interpolate(
        target_coarse_flow.reshape(batch * pairs, coarse_size, coarse_size, 2).permute(
            0, 3, 1, 2
        ),
        size=(fine_size, fine_size),
        mode="bilinear",
        align_corners=True,
    ).permute(0, 2, 3, 1).reshape(batch, pairs, fine_size * fine_size, 2)
    fine_coordinates = _mtd_v2_coordinates(
        fine_size, pred_tokens.device, pred_tokens.dtype
    )
    target_centres = fine_coordinates.view(1, 1, -1, 2) + upsampled_centres * float(
        offset
    )
    candidate_indices, candidate_coordinates, valid = _mtd_v2_fine_candidates(
        target_centres, fine_size, mtd["fine_radius"]
    )
    current = pred_tokens[:, :-offset]
    following = pred_tokens[:, offset:]
    batch_index = torch.arange(batch, device=pred_tokens.device)[:, None, None, None]
    pair_index = torch.arange(pairs, device=pred_tokens.device)[None, :, None, None]
    pred_neighbours = following[batch_index, pair_index, candidate_indices]
    pred_logits = (current.unsqueeze(-2) * pred_neighbours).sum(dim=-1) / mtd[
        "temperature"
    ]
    pred_logits = pred_logits.masked_fill(~valid, torch.finfo(pred_logits.dtype).min)
    pred_log_probs = F.log_softmax(pred_logits, dim=-1)
    pred_probs = pred_log_probs.exp().masked_fill(~valid, 0.0)
    with torch.no_grad():
        target_current = target_tokens[:, :-offset]
        target_following = target_tokens[:, offset:]
        target_neighbours = target_following[
            batch_index, pair_index, candidate_indices
        ]
        target_logits = (
            target_current.unsqueeze(-2) * target_neighbours
        ).sum(dim=-1) / mtd["temperature"]
        target_logits = target_logits.masked_fill(
            ~valid, torch.finfo(target_logits.dtype).min
        )
        target_probs = F.softmax(target_logits, dim=-1).masked_fill(~valid, 0.0)
    correspondence_per_query = (
        target_probs
        * (target_probs.clamp_min(mtd["eps"]).log() - pred_log_probs)
    ).sum(dim=-1)
    query_weights, entropy, confidence = _mtd_v2_confidence(
        target_probs, valid, mtd
    )
    upsampled_cycle_gate = F.interpolate(
        target_coarse_cycle_gate.reshape(
            batch * pairs, 1, coarse_size, coarse_size
        ),
        size=(fine_size, fine_size),
        mode="bilinear",
        align_corners=True,
    ).reshape(batch, pairs, fine_size * fine_size)
    query_weights = query_weights * upsampled_cycle_gate.detach()
    pred_flow = _mtd_v2_soft_flow(
        pred_probs, candidate_coordinates, fine_coordinates, offset
    )
    with torch.no_grad():
        target_flow = _mtd_v2_soft_flow(
            target_probs, candidate_coordinates, fine_coordinates, offset
        )
    flow_per_query = F.smooth_l1_loss(
        pred_flow,
        target_flow,
        beta=mtd["flow_huber_beta"],
        reduction="none",
    ).mean(dim=-1)
    return (
        _mtd_v2_weighted_mean(
            correspondence_per_query, query_weights, sample_weights, mtd["eps"]
        ),
        _mtd_v2_weighted_mean(
            flow_per_query, query_weights, sample_weights, mtd["eps"]
        ),
        {
            "entropy": entropy,
            "confidence": confidence,
            "flow_mae": (pred_flow - target_flow).abs().mean(dim=-1).detach(),
            "effective_queries": (query_weights > 0).sum().detach(),
            "query_weight_sum": query_weights.sum().detach(),
            "cycle_gate_mean": upsampled_cycle_gate.mean().detach(),
            "cycle_pass_ratio": (upsampled_cycle_gate > 0).float().mean().detach(),
        },
    )


def _mtd_v2_global_relation(pred, target, sample_weights, mtd):
    pred_summary = F.normalize(
        pred.mean(dim=(-1, -2)).transpose(1, 2), dim=-1, eps=1.0e-6
    )
    with torch.no_grad():
        target_summary = F.normalize(
            target.mean(dim=(-1, -2)).transpose(1, 2), dim=-1, eps=1.0e-6
        )
        target_relation = target_summary @ target_summary.transpose(1, 2)
    pred_relation = pred_summary @ pred_summary.transpose(1, 2)
    frames = pred_relation.shape[-1]
    diagonal = torch.eye(frames, device=pred.device, dtype=torch.bool).view(
        1, frames, frames
    )
    floor = torch.finfo(pred_relation.dtype).min
    pred_log_probs = F.log_softmax(
        (pred_relation / mtd["global_temperature"]).masked_fill(diagonal, floor),
        dim=-1,
    )
    with torch.no_grad():
        target_probs = F.softmax(
            (target_relation / mtd["global_temperature"]).masked_fill(
                diagonal, floor
            ),
            dim=-1,
        )
    per_row = (
        target_probs
        * (target_probs.clamp_min(mtd["eps"]).log() - pred_log_probs)
    ).sum(dim=-1)
    # First average rows within each video: every non-diagonal frame relation
    # contributes equally, independent of the number of video frames.  Keep
    # the SNR factor outside this normalization for the same reason as above.
    return (per_row.mean(dim=-1) * sample_weights).mean()


def _mtd_v2_predict_x0(pred, target, context, mtd):
    if context is None:
        raise ValueError("MTD-v2 requires x_t, timesteps, and scheduler context")
    try:
        x_t = context["x_t"]
        timesteps = context["timesteps"]
        schedule = context["schedule"]
        coeff_a = schedule["sqrt_recip_alphas_cumprod"]
        coeff_b = schedule["sqrt_recipm1_alphas_cumprod"]
    except (KeyError, TypeError) as error:
        raise ValueError("invalid MTD-v2 context") from error
    if x_t.ndim != 5 or x_t.shape[0] != pred.shape[0]:
        raise ValueError("MTD-v2 x_t must be a 5D batch-aligned latent tensor")
    if x_t.shape[1] < mtd["noise_channels"]:
        raise ValueError("MTD-v2 x_t has fewer channels than noise_channels")
    timesteps = timesteps.reshape(-1).to(device=pred.device, dtype=torch.long)
    if timesteps.numel() != pred.shape[0]:
        raise ValueError("MTD-v2 timesteps must contain one value per batch item")
    coeff_a = torch.as_tensor(coeff_a, device=pred.device, dtype=torch.float32)
    coeff_b = torch.as_tensor(coeff_b, device=pred.device, dtype=torch.float32)
    if coeff_a.ndim != 1 or coeff_a.shape != coeff_b.shape:
        raise ValueError("MTD-v2 scheduler coefficients must be equal-length vectors")
    coefficient_indices = timesteps
    timestep_map = schedule.get("timestep_map") if isinstance(schedule, Mapping) else None
    if timestep_map is not None:
        timestep_map = torch.as_tensor(
            timestep_map, device=pred.device, dtype=torch.long
        )
        if timestep_map.ndim != 1 or timestep_map.numel() != coeff_a.numel():
            raise ValueError(
                "MTD-v2 scheduler timestep_map must align with its coefficient vectors"
            )
        matches = timesteps[:, None].eq(timestep_map[None, :])
        if matches.any(dim=1).all():
            coefficient_indices = matches.to(torch.long).argmax(dim=1)
        elif timesteps.min().item() < 0 or timesteps.max().item() >= coeff_a.numel():
            raise ValueError(
                "MTD-v2 timestep is neither an IDDPM timestep_map value nor a solver-step index"
            )
    elif timesteps.min().item() < 0 or timesteps.max().item() >= coeff_a.numel():
        raise ValueError("MTD-v2 timestep is outside scheduler coefficients")
    a_t = coeff_a.index_select(0, coefficient_indices).view(-1, 1, 1, 1, 1)
    b_t = coeff_b.index_select(0, coefficient_indices).view(-1, 1, 1, 1, 1)
    x_t = x_t.float()[:, :mtd["noise_channels"]]
    pred_x0 = a_t * x_t - b_t * pred[:, :mtd["noise_channels"]]
    with torch.no_grad():
        target_x0 = a_t * x_t - b_t * target[:, :mtd["noise_channels"]]
    alpha_bar = a_t.flatten().reciprocal().square()
    snr = alpha_bar / (1.0 - alpha_bar).clamp_min(mtd["eps"])
    sample_weights = (snr / mtd["snr_full_weight"]).clamp(max=1.0).detach()
    return pred_x0, target_x0, sample_weights, snr.detach()


def motion_transport_distillation_v2(
    pred,
    target,
    context=None,
    config=None,
    return_components=False,
    return_diagnostics=False,
):
    """Compute multi-scale, trajectory-aware transport distillation on x0 latents."""
    mtd = normalize_mtd_v2_config(config)
    zero = pred.new_tensor(0.0)
    empty_components = {"correspondence": zero, "flow": zero, "global": zero}
    if not mtd["enabled"]:
        if return_components and return_diagnostics:
            return zero, empty_components, {}
        if return_components:
            return zero, empty_components
        if return_diagnostics:
            return zero, {}
        return zero
    if pred.ndim != 5 or target.ndim != 5 or pred.shape != target.shape:
        raise ValueError("MTD-v2 requires equal-shape 5D prediction and teacher tensors")
    if pred.shape[1] < mtd["noise_channels"]:
        raise ValueError("MTD-v2 prediction has fewer channels than noise_channels")
    if pred.shape[2] <= 4 or pred.shape[2] <= max(mtd["temporal_offsets"]):
        raise ValueError(
            "MTD-v2 video must contain more than four frames and exceed its largest temporal offset"
        )
    pred_x0, target_x0, sample_weights, snr = _mtd_v2_predict_x0(
        pred.float(), target.detach().float(), context, mtd
    )
    if mtd["feature_source"] == "epsilon":
        pred_transport = pred.float()[:, :mtd["noise_channels"]]
        target_transport = target.detach().float()[:, :mtd["noise_channels"]]
    else:
        pred_transport = pred_x0
        target_transport = target_x0
    coarse_size = min(
        mtd["coarse_size"], pred_transport.shape[-2], pred_transport.shape[-1]
    )
    fine_size = min(
        mtd["fine_size"], pred_transport.shape[-2], pred_transport.shape[-1]
    )
    if fine_size < coarse_size:
        fine_size = coarse_size
    pred_coarse = _mtd_v2_tokens(pred_transport, coarse_size)
    with torch.no_grad():
        target_coarse = _mtd_v2_tokens(target_transport, coarse_size)
    pred_fine = _mtd_v2_tokens(pred_transport, fine_size)
    with torch.no_grad():
        target_fine = _mtd_v2_tokens(target_transport, fine_size)
    coarse_correspondence = []
    coarse_flow = []
    fine_correspondence = []
    fine_flow = []
    diagnostics = {"snr_mean": snr.mean(), "snr_min": snr.min(), "snr_max": snr.max()}
    for offset in mtd["temporal_offsets"]:
        coarse_corr, coarse_motion, teacher_flow, coarse_info = _mtd_v2_coarse_terms(
            pred_coarse, target_coarse, offset, mtd, sample_weights
        )
        fine_corr, fine_motion, fine_info = _mtd_v2_fine_terms(
            pred_fine,
            target_fine,
            teacher_flow,
            coarse_info["cycle_gate"],
            coarse_size,
            fine_size,
            offset,
            mtd,
            sample_weights,
        )
        coarse_correspondence.append(coarse_corr)
        coarse_flow.append(coarse_motion)
        fine_correspondence.append(fine_corr)
        fine_flow.append(fine_motion)
        prefix = f"offset_{offset}"
        diagnostics.update(
            {
                f"{prefix}_coarse_entropy_mean": coarse_info["entropy"].mean(),
                f"{prefix}_coarse_confidence_mean": coarse_info["confidence"].mean(),
                f"{prefix}_coarse_topk_mass_mean": coarse_info["topk_mass"].mean(),
                f"{prefix}_coarse_flow_mae": coarse_info["flow_mae"].mean(),
                f"{prefix}_coarse_effective_queries": coarse_info["effective_queries"],
                f"{prefix}_coarse_query_weight_sum": coarse_info["query_weight_sum"],
                f"{prefix}_coarse_cycle_mean": coarse_info["cycle_score"].mean(),
                f"{prefix}_coarse_cycle_pass_ratio": coarse_info["cycle_pass_ratio"],
                f"{prefix}_fine_entropy_mean": fine_info["entropy"].mean(),
                f"{prefix}_fine_confidence_mean": fine_info["confidence"].mean(),
                f"{prefix}_fine_flow_mae": fine_info["flow_mae"].mean(),
                f"{prefix}_fine_effective_queries": fine_info["effective_queries"],
                f"{prefix}_fine_query_weight_sum": fine_info["query_weight_sum"],
                f"{prefix}_fine_cycle_gate_mean": fine_info["cycle_gate_mean"],
                f"{prefix}_fine_cycle_pass_ratio": fine_info["cycle_pass_ratio"],
            }
        )
    correspondence = torch.stack(coarse_correspondence).mean() + mtd[
        "fine_scale_weight"
    ] * torch.stack(fine_correspondence).mean()
    flow = torch.stack(coarse_flow).mean() + mtd["fine_scale_weight"] * torch.stack(
        fine_flow
    ).mean()
    global_relation = _mtd_v2_global_relation(
        pred_x0, target_x0, sample_weights, mtd
    )
    components = {
        "correspondence": mtd["total_weight"]
        * mtd["correspondence_weight"]
        * correspondence,
        "flow": mtd["total_weight"] * mtd["flow_weight"] * flow,
        "global": mtd["total_weight"]
        * mtd["global_relation_weight"]
        * global_relation,
    }
    total = sum(components.values())
    diagnostics.update(
        {
            "coarse_size": coarse_size,
            "fine_size": fine_size,
            "effective_sample_weight_mean": sample_weights.mean(),
        }
    )
    if return_components and return_diagnostics:
        return total, components, diagnostics
    if return_components:
        return total, components
    if return_diagnostics:
        return total, diagnostics
    return total
