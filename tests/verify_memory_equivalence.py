"""Numerical equivalence checks for PTQ memory optimizations.

Run in the qvdit environment with CUDA_VISIBLE_DEVICES fixed to the intended
physical GPU.  Every check compares the optimized path against the original
expression for both forward values and gradients.
"""

import copy

import torch

from opensora.acceleration.checkpoint import auto_grad_checkpoint, set_grad_checkpoint
from opensora.models.layers.blocks import Attention
from qdiff.optimization.block_recon import (
    _AsyncCudaBatchPrefetcher,
    _select_reconstruction_batch,
)
from qdiff.quantizer.dynamic_quantizer import _FusedDynamicQuantizeSTE


def _max_error(left, right):
    return float((left - right).abs().max())


def verify_token_reduction():
    torch.manual_seed(1)
    source = torch.randn(4, 129, 17)
    # Explicit ties exercise the original first-occurrence gradient rule.
    source[0, 3, 0] = source[1, 3, 1] = -7.0
    source[0, 5, 0] = source[2, 5, 2] = 7.0
    weights = torch.randn(129)

    old_x = source.clone().requires_grad_()
    flattened = old_x.permute(1, 0, 2).reshape(129, -1)
    old_min = flattened.min(dim=-1).values.clamp(max=0.0)
    old_max = flattened.max(dim=-1).values.clamp(min=0.0)
    ((old_min + old_max) * weights).sum().backward()

    new_x = source.clone().requires_grad_()
    new_min = new_x.min(dim=2).values.min(dim=0).values.clamp(max=0.0)
    new_max = new_x.max(dim=2).values.max(dim=0).values.clamp(min=0.0)
    ((new_min + new_max) * weights).sum().backward()

    assert torch.equal(old_min, new_min)
    assert torch.equal(old_max, new_max)
    assert torch.equal(old_x.grad, new_x.grad)
    print("token_reduction: exact forward and gradient match")


def _original_dynamic_quant(x, delta, zero_point, lower, upper):
    scaled = x / delta
    rounded_ste = (scaled.round() - scaled).detach() + scaled
    return (torch.clamp(rounded_ste + zero_point, lower, upper) - zero_point) * delta


def verify_fused_dynamic_quantizer():
    torch.manual_seed(2)
    source = torch.randn(3, 129, 17) * 4.0
    probe = torch.randn_like(source)
    for lower, upper, zero_value in ((0, 63, 8.0), (-32, 31, 0.0)):
        old_x = source.clone().requires_grad_()
        old_delta = (torch.rand(1, 129, 1) + 0.1).requires_grad_()
        zero_point = torch.full((1, 129, 1), zero_value)
        old_output = _original_dynamic_quant(old_x, old_delta, zero_point, lower, upper)
        (old_output * probe).sum().backward()

        new_x = source.clone().requires_grad_()
        new_delta = old_delta.detach().clone().requires_grad_()
        new_output = _FusedDynamicQuantizeSTE.apply(
            new_x, new_delta, zero_point, lower, upper
        )
        (new_output * probe).sum().backward()

        assert torch.equal(old_output.detach(), new_output.detach())
        assert torch.allclose(old_x.grad, new_x.grad, atol=1.0e-6, rtol=1.0e-6)
        assert torch.allclose(old_delta.grad, new_delta.grad, atol=2.0e-5, rtol=1.0e-6)
        print(
            "fused_dynamic_quantizer:",
            f"bounds=({lower},{upper})",
            f"forward={_max_error(old_output, new_output):.3e}",
            f"grad_x={_max_error(old_x.grad, new_x.grad):.3e}",
            f"grad_scale={_max_error(old_delta.grad, new_delta.grad):.3e}",
        )


def verify_selective_checkpoint():
    torch.manual_seed(3)
    reference_first = torch.nn.Sequential(
        torch.nn.Linear(12, 24), torch.nn.GELU(), torch.nn.Linear(24, 12)
    )
    reference_second = copy.deepcopy(reference_first)
    checked_first = copy.deepcopy(reference_first)
    checked_second = copy.deepcopy(reference_second)
    set_grad_checkpoint(checked_first, use_reentrant=False)
    set_grad_checkpoint(checked_second, use_reentrant=True)

    source = torch.randn(5, 12)
    probe = torch.randn(5, 12)
    reference_output = reference_second(reference_first(source))
    (reference_output * probe).sum().backward()
    checked_output = auto_grad_checkpoint(
        checked_second, auto_grad_checkpoint(checked_first, source)
    )
    (checked_output * probe).sum().backward()

    assert torch.equal(reference_output, checked_output)
    max_parameter_error = 0.0
    for reference, checked in zip(
        list(reference_first.parameters()) + list(reference_second.parameters()),
        list(checked_first.parameters()) + list(checked_second.parameters()),
    ):
        max_parameter_error = max(max_parameter_error, _max_error(reference.grad, checked.grad))
    assert max_parameter_error == 0.0
    print("selective_checkpoint: exact forward and parameter-gradient match")


def verify_async_cache(device):
    cached_inps = [
        torch.arange(8 * 5, dtype=torch.float32).reshape(8, 5),
        torch.arange(8 * 3, dtype=torch.float32).reshape(8, 3),
        torch.arange(8 * 2, dtype=torch.float32).reshape(8, 2),
    ]
    cached_outs = torch.arange(8 * 7, dtype=torch.float32).reshape(8, 7)
    index = torch.tensor([1, 4, 6, 7])
    expected = _select_reconstruction_batch(
        cached_inps, cached_outs, None, False, index
    )

    prefetcher = _AsyncCudaBatchPrefetcher(
        lambda _: (*_select_reconstruction_batch(
            cached_inps, cached_outs, None, False, index
        ), None),
        device,
        pin_memory=True,
    )
    prefetcher.start(0)
    actual, host_batch = prefetcher.next(None)
    torch.cuda.synchronize(device)
    prefetcher.close()
    actual_inps, actual_outs, actual_grad, metadata = actual
    for expected_tensor, actual_tensor in zip(expected[0], actual_inps):
        assert torch.equal(expected_tensor, actual_tensor.cpu())
    assert torch.equal(expected[1], actual_outs.cpu())
    assert actual_grad is None and metadata is None
    assert all(tensor.dtype == torch.float32 for tensor in host_batch[0])
    print("async_cpu_cache: exact FP32 batch match")


def verify_memory_efficient_attention(device):
    torch.manual_seed(4)
    reference = Attention(
        dim=64,
        num_heads=4,
        qkv_bias=True,
        enable_flashattn=False,
        enable_memory_efficient_attention=False,
    ).to(device)
    optimized = Attention(
        dim=64,
        num_heads=4,
        qkv_bias=True,
        enable_flashattn=False,
        enable_memory_efficient_attention=True,
    ).to(device)
    optimized.load_state_dict(reference.state_dict())
    reference.eval()
    optimized.eval()

    source = torch.randn(2, 257, 64, device=device)
    probe = torch.randn_like(source)
    reference_input = source.clone().requires_grad_()
    optimized_input = source.clone().requires_grad_()
    reference_output = reference(reference_input)
    optimized_output = optimized(optimized_input)
    (reference_output * probe).sum().backward()
    (optimized_output * probe).sum().backward()

    forward_error = _max_error(reference_output, optimized_output)
    input_gradient_error = _max_error(reference_input.grad, optimized_input.grad)
    parameter_gradient_error = max(
        _max_error(left.grad, right.grad)
        for left, right in zip(reference.parameters(), optimized.parameters())
    )
    assert forward_error < 2.0e-5
    assert input_gradient_error < 3.0e-5
    assert parameter_gradient_error < 3.0e-4
    print(
        "memory_efficient_attention:",
        f"forward={forward_error:.3e}",
        f"grad_input={input_gradient_error:.3e}",
        f"grad_parameter={parameter_gradient_error:.3e}",
    )


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the xFormers and async-cache checks")
    device = torch.device("cuda:0")
    verify_token_reduction()
    verify_fused_dynamic_quantizer()
    verify_selective_checkpoint()
    verify_async_cache(device)
    verify_memory_efficient_attention(device)
    print("all memory-optimization equivalence checks passed")


if __name__ == "__main__":
    main()
