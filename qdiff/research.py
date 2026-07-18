import math
from collections.abc import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


def _plain_dict(value):
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return {key: _plain_dict(item) if isinstance(item, Mapping) else item for key, item in value.items()}
    if hasattr(value, "items"):
        return {key: _plain_dict(item) if hasattr(item, "items") else item for key, item in value.items()}
    return dict(value)


# Layer families whose inputs are video-token sequences and can therefore
# carry a TARQ branch.  "cross_attn" covers the video-token side of cross
# attention (q_linear and the output proj); the text-token kv_linear can never
# use TARQ and always keeps the TQE compensation.
TARQ_SCOPES = ("spatial_attn", "temporal_attn", "cross_attn", "ffn")


def _normalize_tarq_scope(value):
    if value is None:
        return ("spatial_attn", "temporal_attn")
    if isinstance(value, str):
        value = [value]
    scope = tuple(str(item).lower() for item in value)
    unknown = [item for item in scope if item not in TARQ_SCOPES]
    if unknown:
        raise ValueError(f"Unknown tarq.apply_to entries {unknown}; valid: {list(TARQ_SCOPES)}")
    return scope


def normalize_research_config(config=None):
    config = _plain_dict(config)
    method = config if "token_axis" in config else _plain_dict(config.get("method"))
    tarq = _plain_dict(config.get("tarq"))
    mtd = _plain_dict(config.get("mtd"))
    taq = _plain_dict(config.get("taq"))
    return {
        "token_axis": str(method.get("token_axis", "TQE")).upper(),
        "frame_axis": str(method.get("frame_axis", "BASELINE")).upper(),
        "diffusion_axis": str(method.get("diffusion_axis", "NONE")).upper(),
        # STDiT predicts noise and variance stacked on the channel axis; only
        # the first `noise_channels` channels are the denoising signal that the
        # research losses should supervise.
        "noise_channels": int(config.get("noise_channels", 4)),
        "tarq": {
            "num_groups": int(tarq.get("num_groups", 4)),
            "rank_per_group": int(tarq.get("rank_per_group", 1)),
            "apply_to": _normalize_tarq_scope(tarq.get("apply_to")),
            "transport_size": int(tarq.get("transport_size", 16)),
            "descriptor_dim": int(tarq.get("descriptor_dim", 8)),
            "temperature": float(tarq.get("temperature", 0.07)),
            "rank_budget": float(tarq.get("rank_budget", 1.0)),
            "rank_budget_weight": float(tarq.get("rank_budget_weight", 1.0e-4)),
        },
        "mtd": {
            "transport_size": int(mtd.get("transport_size", 16)),
            "temperature": float(mtd.get("temperature", 0.07)),
            "local_transport_weight": float(mtd.get("local_transport_weight", 1.0)),
            "motion_residual_weight": float(mtd.get("motion_residual_weight", 1.0)),
            "global_relation_weight": float(mtd.get("global_relation_weight", 0.1)),
        },
        "taq": {
            "num_bins": int(taq.get("num_bins", 8)),
            "trajectory_weight": float(taq.get("trajectory_weight", 1.0)),
            "clip_min": float(taq.get("clip_min", 0.5)),
            "scale_min": float(taq.get("scale_min", 0.5)),
            "scale_max": float(taq.get("scale_max", 2.0)),
        },
    }


def trajectory_bin(step_index, num_steps, num_bins):
    if num_steps <= 1:
        return 0
    progress = max(0.0, min(1.0, float(step_index) / float(num_steps - 1)))
    return min(num_bins - 1, int(progress * num_bins))


def set_trajectory_position(module, step_index, num_steps):
    for child in module.modules():
        if hasattr(child, "set_trajectory_position"):
            child.set_trajectory_position(step_index, num_steps)


def sample_trajectory_pair_indices(total_size, n_steps, samples_per_step, iters, batch_size, device, num_bins=8):
    if batch_size % 2:
        raise ValueError("TAQ trajectory batches must have an even batch size")
    if total_size < n_steps * samples_per_step:
        raise ValueError("Calibration cache is smaller than the configured trajectory layout")
    valid_steps = []
    for step in range(n_steps - 1):
        if trajectory_bin(step, n_steps, num_bins) == trajectory_bin(step + 1, n_steps, num_bins):
            valid_steps.append(step)
    if not valid_steps:
        raise ValueError("No adjacent calibration steps fall inside the same TAQ bin")
    valid_steps = torch.tensor(valid_steps, device=device)
    step_ids = valid_steps[torch.randint(0, valid_steps.numel(), (iters,), device=device)]
    pair_count = batch_size // 2
    sample_ids = torch.randint(0, samples_per_step, (iters, pair_count), device=device)
    base = step_ids[:, None] * samples_per_step + sample_ids
    paired = base + samples_per_step
    return torch.stack((base, paired), dim=-1).reshape(iters, batch_size)


MOTION_FEATURE_DIM = 5  # magnitude, confidence, entropy, dx, dy


def _compact_transport_features(video, transport_size, descriptor_dim, temperature):
    # video: [B, T, H, W, C]
    batch, frames, height, width, channels = video.shape
    size = min(transport_size, height, width)
    descriptors = video[..., : min(descriptor_dim, channels)].permute(0, 1, 4, 2, 3)
    descriptors = descriptors.reshape(batch * frames, descriptors.shape[2], height, width)
    descriptors = F.adaptive_avg_pool2d(descriptors, (size, size))
    descriptors = descriptors.reshape(batch, frames, -1, size, size)
    descriptors = F.normalize(descriptors, dim=2, eps=1.0e-6)

    if frames == 1:
        zeros = video.new_zeros(batch, 1, size, size, MOTION_FEATURE_DIM)
        return zeros

    current = descriptors[:, :-1]
    following = descriptors[:, 1:]
    current_flat = current.permute(0, 1, 3, 4, 2).reshape(batch * (frames - 1), size * size, -1)
    following_flat = following.reshape(batch * (frames - 1), following.shape[2], size, size)
    neighbours = F.unfold(following_flat, kernel_size=3, padding=1)
    neighbours = neighbours.reshape(batch * (frames - 1), -1, 9, size * size).permute(0, 3, 2, 1)
    logits = (current_flat[:, :, None, :] * neighbours).sum(-1) / temperature
    probs = F.softmax(logits, dim=-1)
    offsets = video.new_tensor(
        [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 0], [0, 1], [1, -1], [1, 0], [1, 1]]
    )
    displacement = probs @ offsets
    # vector_norm has the same forward value as sqrt(dx^2 + dy^2), but defines
    # a zero gradient at zero displacement.  The explicit sqrt form produces
    # inf in backward and turns an upstream zero into NaN (0 * inf).
    magnitude = torch.linalg.vector_norm(displacement, dim=-1)
    confidence = probs.max(dim=-1).values
    entropy = -(probs.clamp_min(1.0e-8).log() * probs).sum(-1) / math.log(9.0)
    # Keep the expected displacement direction alongside its magnitude so the
    # gate can separate motion *patterns*, not only motion strength.
    features = torch.stack(
        (magnitude, confidence, entropy, displacement[..., 0], displacement[..., 1]), dim=-1
    )
    features = features.reshape(batch, frames - 1, size, size, MOTION_FEATURE_DIM)
    first = features[:, :1]
    return torch.cat((first, features), dim=1)


class BlockMotionContext:
    """Per-transformer-block cache for TARQ motion features.

    All TARQ branches inside one block describe the same token grid, so the
    first branch that runs in a forward pass computes the motion descriptor
    and the others reuse it.  A forward pre-hook on the owning block clears
    the cache, which also keeps gradient-checkpoint recomputation correct
    (the hook fires again on the recompute pass).
    """

    def __init__(self):
        self._features = None

    def clear(self, *_args, **_kwargs):
        self._features = None

    def get(self, batch, frames, spatial_tokens):
        cached = self._features
        if cached is None or cached.shape[:3] != (batch, frames, spatial_tokens):
            return None
        return cached

    def set(self, features):
        self._features = features


class TrackAwareResidual(nn.Module):
    def __init__(self, in_features, out_features, research_config):
        super().__init__()
        config = normalize_research_config(research_config)
        tarq = config["tarq"]
        self.num_groups = tarq["num_groups"]
        self.rank_per_group = tarq["rank_per_group"]
        self.transport_size = tarq["transport_size"]
        self.descriptor_dim = tarq["descriptor_dim"]
        self.base_temperature = tarq["temperature"]
        self.rank_budget = tarq["rank_budget"]
        self.taq_enabled = config["diffusion_axis"] == "TAQ"
        self.num_bins = config["taq"]["num_bins"]
        self.down = nn.Parameter(torch.empty(self.num_groups, self.rank_per_group, in_features))
        self.up = nn.Parameter(torch.zeros(self.num_groups, out_features, self.rank_per_group))
        # Learned projection for the transport descriptor.  Matching on an
        # arbitrary slice of the first channels has no semantic selectivity;
        # an orthogonally-initialized learned projection preserves distances at
        # init and lets calibration pick motion-discriminative directions.
        self.descriptor_proj = nn.Linear(in_features, min(self.descriptor_dim, in_features), bias=False)
        self.gate = nn.Linear(MOTION_FEATURE_DIM, self.num_groups)
        nn.init.kaiming_uniform_(self.down, a=math.sqrt(5))
        nn.init.orthogonal_(self.descriptor_proj.weight)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        if self.taq_enabled:
            self.taq_log_temperature = nn.Parameter(torch.zeros(self.num_bins))
            # Start from one active rank/group per timestep bin. Reconstruction
            # gradients can spend more rank in difficult denoising regions.
            self.taq_rank_logit = nn.Parameter(torch.full((self.num_bins,), -6.0))
        else:
            self.register_parameter("taq_log_temperature", None)
            self.register_parameter("taq_rank_logit", None)
        self.trajectory_step_index = 0
        self.trajectory_num_steps = 1
        self.last_budget_loss = None
        # Optional per-block shared motion cache (plain attribute, no params).
        self.motion_context = None

    def set_trajectory_position(self, step_index, num_steps):
        self.trajectory_step_index = int(step_index)
        self.trajectory_num_steps = int(num_steps)

    def _temperature_and_budget(self):
        if not self.taq_enabled:
            return self.base_temperature, self.rank_budget
        bin_id = trajectory_bin(self.trajectory_step_index, self.trajectory_num_steps, self.num_bins)
        temperature = self.base_temperature * self.taq_log_temperature[bin_id].exp().clamp(0.5, 2.0)
        rank_budget = 1.0 + (self.num_groups - 1.0) * torch.sigmoid(self.taq_rank_logit[bin_id])
        return temperature, rank_budget

    def _allocate_rank_groups(self, soft_gates, rank_budget):
        if self.num_groups == 1:
            return soft_gates, torch.ones_like(soft_gates)
        # Rank positions are non-differentiable, while the soft active mask is
        # differentiable with respect to both gate scores and the TAQ budget.
        order = soft_gates.argsort(dim=-1, descending=True)
        positions = torch.empty_like(soft_gates)
        rank_ids = torch.arange(1, self.num_groups + 1, device=soft_gates.device, dtype=soft_gates.dtype)
        rank_ids = rank_ids.view(*([1] * (soft_gates.ndim - 1)), self.num_groups).expand_as(soft_gates)
        positions.scatter_(-1, order, rank_ids)
        if self.training:
            active_mask = torch.sigmoid((rank_budget + 0.5 - positions) / 0.25)
        else:
            budget_tensor = torch.as_tensor(
                rank_budget, device=soft_gates.device, dtype=soft_gates.dtype
            )
            active_count = torch.round(budget_tensor.detach()).clamp(1, self.num_groups)
            active_mask = (positions <= active_count).to(soft_gates.dtype)
        allocated = soft_gates * active_mask
        allocated = allocated / allocated.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
        return allocated, active_mask

    def budget_regularization(self):
        if not self.taq_enabled:
            return self.down.new_tensor(0.0)
        all_bin_budgets = 1.0 + (self.num_groups - 1.0) * torch.sigmoid(self.taq_rank_logit)
        return (all_bin_budgets.mean() - self.rank_budget).square()

    @staticmethod
    def effective_group_count(gates):
        """Differentiable number of groups used by a normalized gate vector.

        This is the inverse Simpson concentration: it is exactly one for a
        one-hot allocation and equals the number of groups for a uniform
        allocation.  Measuring the post-allocation gates makes the TARQ rank
        budget constrain the residual that is actually applied.
        """
        return gates.square().sum(dim=-1).clamp_min(1.0e-8).reciprocal()

    def _allocation_budget_loss(self, motion):
        temperature, rank_budget = self._temperature_and_budget()
        gate_logits = self.gate(motion) / temperature
        soft_gates = F.softmax(gate_logits, dim=-1)
        gates, _ = self._allocate_rank_groups(soft_gates, rank_budget)
        effective_groups = self.effective_group_count(gates).mean()
        loss = (effective_groups - rank_budget).square() + self.budget_regularization()
        return gates, loss

    def forward(self, inputs, batch, frames, spatial_tokens, layout):
        side = int(math.sqrt(spatial_tokens))
        if side * side != spatial_tokens:
            raise ValueError(f"TARQ requires a square spatial token grid, got {spatial_tokens}")
        if layout in ("spatial", "sequence"):
            # [B*T, S, C] and [B, T*S, C] share the same frame-major memory
            # order, so both reshape directly into [B, T, S, C].
            video = inputs.reshape(batch, frames, spatial_tokens, inputs.shape[-1])
        elif layout == "temporal":
            video = inputs.reshape(batch, spatial_tokens, frames, inputs.shape[-1]).permute(0, 2, 1, 3)
        else:
            raise ValueError(f"Unknown TARQ layout: {layout}")
        motion = None
        if self.motion_context is not None:
            motion = self.motion_context.get(batch, frames, spatial_tokens)
        if motion is None:
            video_grid = video.reshape(batch, frames, side, side, inputs.shape[-1])
            # The transport softmax keeps the fixed base temperature: TAQ
            # modulates only gate sparsity, not the motion evidence itself.
            descriptor_grid = F.linear(video_grid, self.descriptor_proj.weight.to(inputs.dtype))
            motion = _compact_transport_features(
                descriptor_grid, self.transport_size, descriptor_grid.shape[-1], self.base_temperature
            )
            motion = motion.permute(0, 1, 4, 2, 3).reshape(
                batch * frames, MOTION_FEATURE_DIM, motion.shape[2], motion.shape[3]
            )
            motion = F.interpolate(motion, size=(side, side), mode="bilinear", align_corners=False)
            motion = motion.reshape(batch, frames, MOTION_FEATURE_DIM, spatial_tokens).permute(0, 1, 3, 2)
            if self.motion_context is not None:
                self.motion_context.set(motion)
        gates, self.last_budget_loss = self._allocation_budget_loss(motion)
        if not torch.is_grad_enabled():
            # Reentrant gradient checkpointing executes the original forward
            # under no_grad.  Rebuild only the tiny gate/budget graph so the
            # auxiliary rank loss still trains every TARQ gate without keeping
            # the full transformer activation graph resident.
            with torch.enable_grad():
                _, self.last_budget_loss = self._allocation_budget_loss(motion.detach())

        correction = grouped_low_rank_residual(
            video,
            gates.to(inputs.dtype),
            self.down.to(inputs.dtype),
            self.up.to(inputs.dtype),
        )
        if layout == "spatial":
            return correction.reshape(batch * frames, spatial_tokens, -1)
        if layout == "sequence":
            return correction.reshape(batch, frames * spatial_tokens, -1)
        return correction.permute(0, 2, 1, 3).reshape(batch * spatial_tokens, frames, -1)


def grouped_low_rank_residual(video, gates, down, up):
    """Apply all TARQ groups without materializing one full output per group."""
    low_rank = torch.einsum("...c,grc->...gr", video, down)
    low_rank = low_rank * gates.unsqueeze(-1)
    return torch.einsum("...gr,gor->...o", low_rank, up)


def collect_rank_budget_loss(module):
    losses = [
        child.last_budget_loss
        for child in module.modules()
        if isinstance(child, TrackAwareResidual) and child.last_budget_loss is not None
    ]
    if not losses:
        parameter = next(module.parameters(), None)
        return torch.tensor(0.0, device=parameter.device if parameter is not None else "cpu")
    return torch.stack(losses).mean()


def _local_transport_distribution(features, size, temperature):
    batch, channels, frames, height, width = features.shape
    pooled = features.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width)
    pooled = F.adaptive_avg_pool2d(pooled, (size, size)).reshape(batch, frames, channels, size, size)
    pooled = F.normalize(pooled, dim=2, eps=1.0e-6)
    current = pooled[:, :-1]
    following = pooled[:, 1:]
    current_flat = current.permute(0, 1, 3, 4, 2).reshape(batch * (frames - 1), size * size, channels)
    following_flat = following.reshape(batch * (frames - 1), channels, size, size)
    neighbours = F.unfold(following_flat, kernel_size=3, padding=1)
    neighbours = neighbours.reshape(batch * (frames - 1), channels, 9, size * size).permute(0, 3, 2, 1)
    logits = (current_flat[:, :, None, :] * neighbours).sum(-1) / temperature
    return F.softmax(logits, dim=-1), current_flat, neighbours


def motion_transport_distillation(pred, target, research_config):
    config = normalize_research_config(research_config)
    if config["frame_axis"] != "MTD" or pred.ndim != 5 or pred.shape[2] < 2:
        return pred.new_tensor(0.0)
    mtd = config["mtd"]
    # Match transport on the denoising-signal channels only; the stacked
    # variance channels are not features an object "moves" in.
    noise_channels = config["noise_channels"]
    pred = pred[:, :noise_channels]
    target = target[:, :noise_channels]
    size = min(mtd["transport_size"], pred.shape[-2], pred.shape[-1])
    pred_probs, pred_current, pred_neighbours = _local_transport_distribution(pred, size, mtd["temperature"])
    with torch.no_grad():
        target_probs, target_current, target_neighbours = _local_transport_distribution(target, size, mtd["temperature"])
    local_kl = F.kl_div(pred_probs.clamp_min(1.0e-8).log(), target_probs, reduction="batchmean")
    pred_transport = (pred_probs[..., None] * pred_neighbours).sum(-2) - pred_current
    target_transport = (target_probs[..., None] * target_neighbours).sum(-2) - target_current
    motion_residual = F.smooth_l1_loss(pred_transport, target_transport)

    pred_summary = F.normalize(pred.mean(dim=(-1, -2)).transpose(1, 2), dim=-1, eps=1.0e-6)
    target_summary = F.normalize(target.mean(dim=(-1, -2)).transpose(1, 2), dim=-1, eps=1.0e-6)
    pred_relation = pred_summary @ pred_summary.transpose(1, 2)
    target_relation = target_summary @ target_summary.transpose(1, 2)
    global_relation = F.kl_div(
        F.log_softmax(pred_relation, dim=-1), F.softmax(target_relation, dim=-1), reduction="batchmean"
    )
    return (
        mtd["local_transport_weight"] * local_kl
        + mtd["motion_residual_weight"] * motion_residual
        + mtd["global_relation_weight"] * global_relation
    )


def trajectory_consistency_loss(pred, target, research_config):
    config = normalize_research_config(research_config)
    if config["diffusion_axis"] != "TAQ" or pred.shape[0] < 2 or pred.shape[0] % 2:
        return pred.new_tensor(0.0)
    noise_channels = config["noise_channels"]
    pred_eps = pred[:, :noise_channels] if pred.ndim == 5 else pred
    target_eps = target[:, :noise_channels] if target.ndim == 5 else target
    pred_delta = pred_eps[1::2] - pred_eps[0::2]
    target_delta = target_eps[1::2] - target_eps[0::2]
    return F.mse_loss(pred_delta, target_delta)
