import torch
import torch.nn.functional as F

from qdiff.mtd import (
    _local_transport_distribution,
    motion_transport_distillation,
    normalize_mtd_config,
)


def config(enabled=True, noise_channels=4):
    return {
        "noise_channels": noise_channels,
        "method": {"frame_axis": "MTD" if enabled else "BASELINE"},
        "mtd": {
            "transport_size": 4,
            "temperature": 0.2,
            "local_transport_weight": 1.0,
            "motion_residual_weight": 1.0,
            "global_relation_weight": 0.1,
        },
    }


def test_config_is_disabled_by_default():
    assert not normalize_mtd_config()["enabled"]


def test_config_normalization_is_idempotent():
    normalized = normalize_mtd_config(config())
    assert normalize_mtd_config(normalized) == normalized


def test_mtd_is_zero_when_disabled_or_single_frame():
    video = torch.randn(2, 4, 3, 8, 8)
    assert motion_transport_distillation(video, video, config(False)).item() == 0.0
    assert motion_transport_distillation(video[:, :, :1], video[:, :, :1], config()).item() == 0.0


def test_mtd_is_zero_for_identical_features_and_positive_for_motion_error():
    target = torch.randn(2, 4, 3, 8, 8)
    identical = motion_transport_distillation(target, target, config())
    shifted = motion_transport_distillation(
        torch.roll(target, shifts=1, dims=-1), target, config()
    )
    assert identical.abs().item() < 1.0e-5
    assert shifted.item() > identical.item()


def test_local_kl_is_averaged_over_spatial_tokens():
    torch.manual_seed(11)
    target = torch.randn(2, 4, 3, 8, 8)
    prediction = target + 0.05 * torch.randn_like(target)
    cfg = config()
    cfg["mtd"].update(
        motion_residual_weight=0.0,
        global_relation_weight=0.0,
    )
    pred_probs, _, _ = _local_transport_distribution(prediction, 4, 0.2)
    with torch.no_grad():
        target_probs, _, _ = _local_transport_distribution(target, 4, 0.2)
    expected = F.kl_div(
        pred_probs.clamp_min(1.0e-8).log(), target_probs, reduction="none"
    ).sum(dim=-1).mean()
    actual = motion_transport_distillation(prediction, target, cfg)
    assert torch.allclose(actual, expected, atol=1.0e-7, rtol=1.0e-6)


def test_padded_neighbours_are_masked_at_frame_boundaries():
    features = torch.zeros(1, 4, 2, 4, 4)
    probabilities, _, _ = _local_transport_distribution(features, 4, 0.2)
    assert torch.count_nonzero(probabilities[0, 0]).item() == 4
    assert torch.count_nonzero(probabilities[0, 1]).item() == 6
    assert torch.count_nonzero(probabilities[0, 5]).item() == 9
    assert torch.allclose(
        probabilities.sum(dim=-1), torch.ones_like(probabilities[..., 0])
    )


def test_bfloat16_input_produces_fp32_loss_and_finite_gradient():
    target = torch.randn(1, 8, 3, 8, 8, dtype=torch.bfloat16)
    prediction = target.clone().requires_grad_(True)
    loss = motion_transport_distillation(prediction, target, config())
    assert loss.dtype == torch.float32
    loss.backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad.float()).all()


def test_only_noise_channels_are_supervised():
    target = torch.randn(1, 8, 3, 8, 8)
    prediction = target.clone()
    prediction[:, 4:] += 100.0
    assert motion_transport_distillation(
        prediction, target, config(noise_channels=4)
    ).abs().item() < 1.0e-5
