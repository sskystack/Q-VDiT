import torch
from qdiff.quantizer.base_quantizer import BaseQuantizer, WeightQuantizer, ActQuantizer


'''
The Quantizer that dynamically calculate the quant_params online.
No clipping error online
'''


class _FusedDynamicQuantizeSTE(torch.autograd.Function):
    """Memory-efficient equivalent of round_ste + clamp + dequantize."""

    @staticmethod
    def forward(ctx, x, delta, zero_point, lower_bound, upper_bound):
        # Reuse a single output-sized buffer.  The unfused expression creates
        # several simultaneous 1.12 GiB temporaries for a batch-four STDiT MLP.
        output = x / delta
        output.round_()
        output.add_(zero_point)
        output.clamp_(lower_bound, upper_bound)
        output.sub_(zero_point)
        output.mul_(delta)
        ctx.save_for_backward(x, delta, zero_point, output)
        ctx.lower_bound = lower_bound
        ctx.upper_bound = upper_bound
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, delta, zero_point, output = ctx.saved_tensors
        scaled = x / delta
        rounded = scaled.round().add_(zero_point)
        inside = (rounded >= ctx.lower_bound) & (rounded <= ctx.upper_bound)
        inside = inside.to(grad_output.dtype)

        grad_x = grad_output * inside
        quantized = output / delta
        grad_delta_full = grad_output * (quantized - inside * scaled)
        grad_delta = grad_delta_full.sum_to_size(delta.shape)
        return grad_x, grad_delta, None, None, None


class DynamicActQuantizer(ActQuantizer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.runtime_scale_multiplier = None

    def forward(self, x: torch.Tensor):
        assert self.init_done is True   # for dynamic act quantizer, no init_quant_params stage
        assert self.running_stat is False
        assert self.bit_idx == 0

        # INFO: for dynaimc calculateing quant_params, no handling of mixed_precision/timestep_wise, calculating online
        self.init_quant_params(x, self.per_group, momentum=self.running_stat)
        if self.runtime_scale_multiplier is not None:
            scale = self.runtime_scale_multiplier.to(device=self.delta.device, dtype=self.delta.dtype)
            self.delta = self.delta * scale
            if not self.sym:
                self.zero_point = torch.round(self.zero_point / scale)
        
        # self.delta = self.delta_list[self.bit_idx, 0]
        # self.zero_point = self.zero_point_list[self.bit_idx, 0]

        # INFO: for dynamic quant, for text_embeds act, may have different input shape
        self.delta_list = None
        self.zero_point_list = None

        assert not torch.all(self.delta == -1) # check if not -1

        self.n_levels = 2 ** self.n_bits if not self.sym else 2 ** (self.n_bits - 1) - 1
        # start quantization
        # print(f"x shape {x.shape} delta shape {self.delta.shape} zero shape {self.zero_point.shape}")
        if self.sym:
            lower_bound, upper_bound = -self.n_levels - 1, self.n_levels
        else:
            lower_bound, upper_bound = 0, self.n_levels - 1
        x_dequant = _FusedDynamicQuantizeSTE.apply(
            x, self.delta, self.zero_point, lower_bound, upper_bound
        )
        # import ipdb; ipdb.set_trace()
        # x_quant_ = self.rounding(x)
        return x_dequant
