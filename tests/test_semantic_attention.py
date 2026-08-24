import pytest
import torch


pytest.importorskip("xformers")

from opensora.models.layers.blocks import temporal_attention_incoming_mass
from qdiff.utils import _interleaved_cfg_pair_indices


def test_interleaved_cfg_pair_indices_preserve_adjacent_branches():
    cond, uncond = _interleaved_cfg_pair_indices(
        step_start=20, pair_start=2, pair_count=3
    )
    assert cond.tolist() == [24, 26, 28]
    assert uncond.tolist() == [25, 27, 29]
    assert torch.equal(uncond, cond + 1)


def test_uniform_temporal_attention_has_unit_incoming_mass():
    q = torch.zeros(3, 4, 2, 5)
    k = torch.zeros_like(q)
    incoming = temporal_attention_incoming_mass(
        q, k, scale=5**-0.5, flash_layout=True, chunk_size=2
    )
    assert incoming.shape == (3, 4)
    assert torch.allclose(incoming, torch.ones_like(incoming))


def test_temporal_incoming_mass_matches_explicit_attention():
    torch.manual_seed(7)
    q = torch.randn(2, 3, 2, 4)
    k = torch.randn_like(q)
    scale = 4**-0.5
    logits = torch.einsum("bqhd,bkhd->bhqk", q.float(), k.float()) * scale
    expected = logits.softmax(dim=-1).sum(dim=-2).mean(dim=1)
    actual = temporal_attention_incoming_mass(
        q, k, scale=scale, flash_layout=True, chunk_size=1
    )
    assert torch.allclose(actual, expected, atol=1.0e-7, rtol=1.0e-6)
