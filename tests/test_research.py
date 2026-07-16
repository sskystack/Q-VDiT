from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from qdiff.research import (
    TrackAwareResidual,
    motion_transport_distillation,
    normalize_research_config,
    sample_trajectory_pair_indices,
    trajectory_bin,
    trajectory_consistency_loss,
)


def config(token="TARQ", frame="BASELINE", diffusion="NONE", groups=4):
    return {
        "method": {"token_axis": token, "frame_axis": frame, "diffusion_axis": diffusion},
        "tarq": {
            "num_groups": groups,
            "rank_per_group": 1,
            "transport_size": 2,
            "descriptor_dim": 2,
            "temperature": 0.2,
            "rank_budget": 1.0,
        },
        "mtd": {"transport_size": 4, "temperature": 0.2},
        "taq": {"num_bins": 8},
    }


def test_four_configs_have_the_expected_single_variable_progression():
    root = Path(__file__).parents[1] / "t2v/configs/quant/opensora"
    expected = {
        "w4a6_baseline.yaml": ("TQE", "BASELINE", "NONE"),
        "w4a6_tarq.yaml": ("TARQ", "BASELINE", "NONE"),
        "w4a6_tarq_mtd.yaml": ("TARQ", "MTD", "NONE"),
        "w4a6_tarq_mtd_taq.yaml": ("TARQ", "MTD", "TAQ"),
    }
    for filename, methods in expected.items():
        data = yaml.safe_load((root / filename).read_text())
        assert data["calib_data"]["n_steps"] == 50
        assert data["calib_data"]["batch_size"] == 4
        assert tuple(data["method"].values()) == methods


def test_research_config_normalization_is_idempotent():
    normalized = normalize_research_config(config(frame="MTD", diffusion="TAQ"))
    assert normalize_research_config(normalized) == normalized


def test_trajectory_bins_match_normalized_50_and_100_step_progress():
    for progress in (0.0, 0.1, 0.5, 0.9, 1.0):
        step_50 = min(49, int(progress * 50))
        step_100 = min(99, int(progress * 100))
        assert trajectory_bin(step_50, 50, 8) == trajectory_bin(step_100, 100, 8)


def test_trajectory_pair_sampler_returns_adjacent_steps_in_the_same_bin():
    indices = sample_trajectory_pair_indices(1000, 50, 20, 32, 4, torch.device("cpu"), 8)
    assert indices.shape == (32, 4)
    for row in indices:
        for first, second in row.reshape(-1, 2):
            assert int(second - first) == 20
            first_step = int(first) // 20
            second_step = int(second) // 20
            assert trajectory_bin(first_step, 50, 8) == trajectory_bin(second_step, 50, 8)


def test_single_group_tarq_reduces_to_one_low_rank_residual():
    module = TrackAwareResidual(3, 2, config(groups=1))
    with torch.no_grad():
        module.down.copy_(torch.tensor([[[1.0, -2.0, 0.5]]]))
        module.up.copy_(torch.tensor([[[2.0], [-1.0]]]))
    inputs = torch.randn(2, 4, 3)  # B*T=2, S=4, C=3 with B=1,T=2
    actual = module(inputs, batch=1, frames=2, spatial_tokens=4, layout="spatial")
    expected = F.linear(F.linear(inputs.reshape(1, 2, 4, 3), module.down[0]), module.up[0]).reshape(2, 4, 2)
    assert torch.allclose(actual, expected, atol=1.0e-6)
    assert module.last_budget_loss.item() == 0.0


def test_taq_rank_budget_changes_the_number_of_active_groups():
    module = TrackAwareResidual(3, 2, config(diffusion="TAQ", groups=4))
    gates = torch.tensor([[[[0.4, 0.3, 0.2, 0.1]]]])
    with torch.no_grad():
        module.taq_rank_logit.fill_(-8.0)
        _, low_budget = module._temperature_and_budget()
        _, low_mask = module._allocate_rank_groups(gates, low_budget)
        module.taq_rank_logit.fill_(8.0)
        _, high_budget = module._temperature_and_budget()
        _, high_mask = module._allocate_rank_groups(gates, high_budget)
    assert low_budget.item() < 1.01
    assert high_budget.item() > 3.99
    assert low_mask.sum().item() < high_mask.sum().item()


def test_taq_rank_budget_receives_reconstruction_gradients():
    module = TrackAwareResidual(3, 2, config(diffusion="TAQ", groups=4))
    with torch.no_grad():
        module.up.fill_(0.1)
        module.gate.weight.normal_(0.0, 0.1)
    module.set_trajectory_position(25, 50)
    output = module(torch.randn(2, 4, 3), batch=1, frames=2, spatial_tokens=4, layout="spatial")
    (output.square().mean() + module.budget_regularization()).backward()
    assert module.taq_rank_logit.grad is not None
    assert module.taq_rank_logit.grad.abs().sum().item() > 0.0


def test_mtd_is_zero_for_identical_features_and_positive_for_motion_error():
    target = torch.randn(2, 4, 3, 8, 8)
    identical = motion_transport_distillation(target, target, config(frame="MTD"))
    shifted = motion_transport_distillation(torch.roll(target, shifts=1, dims=-1), target, config(frame="MTD"))
    assert identical.abs().item() < 1.0e-5
    assert shifted.item() > identical.item()


def test_trajectory_loss_uses_adjacent_pairs():
    target = torch.randn(4, 6, 2, 2, 2)
    assert trajectory_consistency_loss(target, target, config(diffusion="TAQ")).item() == 0.0
    prediction = target.clone()
    prediction[1, :3] += 1.0
    assert trajectory_consistency_loss(prediction, target, config(diffusion="TAQ")).item() > 0.0
