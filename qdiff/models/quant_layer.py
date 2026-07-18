import logging
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
import time # DEBUG_ONLY
from omegaconf import ListConfig

from qdiff.quantizer.base_quantizer import WeightQuantizer, ActQuantizer, StraightThrough
from qdiff.quantizer.dynamic_quantizer import DynamicActQuantizer
from qdiff.research import normalize_research_config, trajectory_bin
# import diffusers
import math
logger = logging.getLogger(__name__)

def find_interval(timerange, timestep_id):
    for index, interval in enumerate(timerange):
        if interval[0] <= timestep_id <= interval[1]:
            return index
    return None  # If timestep_id is not within any interval

linear_dtype = torch.float32

class QuantLayer(nn.Module):
    """
    Quantized Module that can perform quantized convolution or normal convolution.
    To activate quantization, please use set_quant_state function.
    """
    def __init__(self, org_module: Union[nn.Conv2d, nn.Linear, nn.Conv1d], weight_quant_params: dict = {},
                 act_quant_params: dict = {}, disable_act_quant: bool = False, act_quant_mode: str = 'qdiff',
                 research_config=None):
        super(QuantLayer, self).__init__()
        # self._orginal_module = org_module
        self.weight_quant_params = weight_quant_params
        self.act_quant_params = act_quant_params
        self.research_config = normalize_research_config(research_config)
        
        if isinstance(org_module, nn.Conv2d):
            self.fwd_kwargs = dict(stride=org_module.stride, padding=org_module.padding,
                                   dilation=org_module.dilation, groups=org_module.groups)
            self.fwd_func = F.conv2d
        elif isinstance(org_module, nn.Conv1d):
            self.fwd_kwargs = dict(stride=org_module.stride, padding=org_module.padding,
                                   dilation=org_module.dilation, groups=org_module.groups)
            self.fwd_func = F.conv1d
        else:
            self.in_features = org_module.in_features
            self.fwd_kwargs = dict()
            self.fwd_func = F.linear
        self.weight = org_module.weight
        if org_module.bias is not None:
            self.bias = org_module.bias
        else:
            self.bias = None
        self.org_module = org_module

        self.r = 32
        self.loraA = nn.Linear(org_module.in_features, self.r, bias=False, dtype=linear_dtype)
        self.loraB = nn.Linear(self.r, org_module.out_features, bias=False, dtype=linear_dtype)
        self.r_out = 1
        self.loraA_out = nn.Linear(org_module.in_features, self.r_out, bias=False, dtype=linear_dtype)
        self.loraB_out = nn.Linear(self.r_out, org_module.out_features, bias=False, dtype=linear_dtype)
        nn.init.kaiming_uniform_(self.loraA.weight, a=math.sqrt(5))
        nn.init.zeros_(self.loraB.weight)
        nn.init.kaiming_uniform_(self.loraA_out.weight, a=math.sqrt(5))
        nn.init.zeros_(self.loraB_out.weight)

        # set use_quant as False, use set_quant_state to set
        self.weight_quant = False
        self.act_quant = False
        self.act_quant_mode = act_quant_mode
        self.disable_act_quant = disable_act_quant

        # initialize quantizer
        if self.weight_quant_params is not None:
            self.weight_quantizer = WeightQuantizer(self.weight_quant_params)
        if self.act_quant_params is not None:
            if self.act_quant_params.get('dynamic',False):
                self.act_quantizer = DynamicActQuantizer(self.act_quant_params)
            else:
                self.act_quantizer = ActQuantizer(self.act_quant_params)
        self.split = 0

        self.activation_function = StraightThrough()
        self.ignore_reconstruction = False
        self.trajectory_step_index = 0
        self.trajectory_num_steps = 1
        # TARQ branch: created in the subclass constructors (spatial/temporal
        # attn) or attached by QuantModel for ffn / cross-attn video-token
        # linears, according to research_config["tarq"]["apply_to"].
        self.tarq = None
        if self.research_config["diffusion_axis"] == "TAQ":
            num_bins = self.research_config["taq"]["num_bins"]
            # sigmoid(3) = 0.953 -> clip ratio 0.976: near-identity but far
            # enough from saturation that clipped elements still produce a
            # usable gradient.  At 6.0 the sigmoid is saturated (grad ~2.5e-3)
            # and almost nothing is clipped, so the parameter never trains.
            self.taq_clip_logit = nn.Parameter(torch.full((num_bins,), 3.0))
            self.taq_log_scale = nn.Parameter(torch.zeros(num_bins))
        else:
            self.register_parameter("taq_clip_logit", None)
            self.register_parameter("taq_log_scale", None)

        self.extra_repr = org_module.extra_repr
        # for smooth quant
        smooth_quant_params = act_quant_params.get("smooth_quant", {})
        self.smooth_quant = smooth_quant_params.get("enable", False)
        if self.smooth_quant:
            cur_timerange_id = 0
            self.timerange = smooth_quant_params.get("timerange", [[0, 1000]])
            # check the time range
            pre_t = -1
            for r in self.timerange:
                assert r[0] == pre_t + 1
                pre_t = r[1]
            assert pre_t == 1000

            self.timerange_num = len(self.timerange)  # how many ranges (how many alphas)
            self.act_quantizer.register_buffer("act_scale", None)
            self.channel_wise_scale_type = smooth_quant_params.get("channel_wise_scale_type", "dynamic")
            self.smooth_quant_momentum = smooth_quant_params.get("momentum", 0)
            self.smooth_quant_alpha = smooth_quant_params.get("alpha", None)
            # assert self.timerange_num == len(self.smooth_quant_alpha)
            self.smooth_quant_running_stat = False

    def set_trajectory_position(self, step_index, num_steps):
        self.trajectory_step_index = int(step_index)
        self.trajectory_num_steps = int(num_steps)

    def apply_taq(self, inputs):
        if self.taq_clip_logit is None:
            return inputs
        taq = self.research_config["taq"]
        bin_id = trajectory_bin(self.trajectory_step_index, self.trajectory_num_steps, taq["num_bins"])
        clip_ratio = taq["clip_min"] + (1.0 - taq["clip_min"]) * torch.sigmoid(self.taq_clip_logit[bin_id])
        max_abs = inputs.detach().abs().amax(dim=-1, keepdim=True).clamp_min(1.0e-6)
        inputs = inputs.clamp(-max_abs * clip_ratio, max_abs * clip_ratio)
        if hasattr(self, "act_quantizer"):
            scale = self.taq_log_scale[bin_id].exp().clamp(taq["scale_min"], taq["scale_max"])
            if isinstance(self.act_quantizer, DynamicActQuantizer):
                self.act_quantizer.runtime_scale_multiplier = scale
            elif not getattr(self, "_taq_scale_warned", False):
                # Static ActQuantizer never reads runtime_scale_multiplier, so
                # the learned TAQ scale would be silently ignored.
                logger.warning(
                    "TAQ scale multiplier has no effect on %s: activation quantizer "
                    "%s is static; use quant.activation.quantizer.dynamic=true.",
                    self.__class__.__name__, self.act_quantizer.__class__.__name__,
                )
                self._taq_scale_warned = True
        return inputs

    def forward(self, input: torch.Tensor, scale: float = 1.0, split: int = 0, smooth_quant_enable: bool = False):
        # DEBUG_ONLY: test the time of init
        if split != 0 and self.split != 0:
            assert(split == self.split)
        elif split != 0:
            logger.info(f"split at {split}!")
            self.split = split
            self.set_split()

        if self.smooth_quant:

            cur_timerange_id = find_interval(self.timerange, self.cur_timestep_id)
            if isinstance(self.smooth_quant_alpha, (list, ListConfig)):
                alpha = self.smooth_quant_alpha[cur_timerange_id]
            else:
                alpha = self.smooth_quant_alpha

            if self.channel_wise_scale_type == "dynamic":
                channel_wise_scale = input.abs().max(dim=-2)[0].pow(alpha).mean(dim=0, keepdim=True) / self.weight.abs().max(dim=0)[0].pow(1 - alpha)
            elif "momentum" in self.channel_wise_scale_type:
                if self.smooth_quant_running_stat:
                    cur_act_scale = input.abs().max(dim=-2)[0].mean(dim=0, keepdim=True)
                    if self.act_quantizer.act_scale is None:
                        self.act_quantizer.act_scale = torch.zeros([self.timerange_num, *cur_act_scale.shape]).to(input)
                    if self.act_quantizer.act_scale[cur_timerange_id].abs().mean()==0:
                        self.act_quantizer.act_scale[cur_timerange_id] = cur_act_scale
                    else:
                        self.act_quantizer.act_scale[cur_timerange_id] = self.act_quantizer.act_scale[cur_timerange_id] * self.smooth_quant_momentum + cur_act_scale * (1 - self.smooth_quant_momentum)
                else:
                    assert self.act_quantizer.act_scale[cur_timerange_id] is not None
                    assert self.act_quantizer.act_scale[cur_timerange_id].mean() != 0
                    if (self.act_quantizer.act_scale[cur_timerange_id] == 0).sum() != 0:
                        zero_mask = self.act_quantizer.act_scale[cur_timerange_id] == 0
                        eps = 1.e-5
                        self.act_quantizer.act_scale[cur_timerange_id][zero_mask] = eps
                        logging.info('act_scale containing zeros, replacing with {}'.format(eps))

                channel_wise_scale = self.act_quantizer.act_scale[cur_timerange_id].pow(alpha) / self.weight.abs().max(dim=0)[0].pow(1 - alpha)
            else:
                raise NotImplementedError

            input = input / channel_wise_scale
        else:
            # for timeranges, update the act_scale for each timerange respectively
            if not hasattr(self, 'timerange'):
                cur_timerange_id = 0
            else:
                cur_timerange_id = find_interval(self.timerange, self.cur_timestep_id)
            if getattr(self, "smooth_quant_running_stat", False) and "momentum" in self.channel_wise_scale_type:
                cur_act_scale = input.abs().max(dim=-2)[0].mean(dim=0, keepdim=True)
                if self.act_quantizer.act_scale is None:
                    self.act_quantizer.act_scale = torch.zeros([self.timerange_num, *cur_act_scale.shape]).to(input)
                if self.act_quantizer.act_scale[cur_timerange_id].abs().mean()==0:
                    self.act_quantizer.act_scale[cur_timerange_id] = cur_act_scale
                else:
                    self.act_quantizer.act_scale[cur_timerange_id] = self.act_quantizer.act_scale[cur_timerange_id] * self.smooth_quant_momentum + cur_act_scale * (1 - self.smooth_quant_momentum)
        
        # print(cur_timerange_id) # debug only
        
        if not self.disable_act_quant and self.act_quant:
            input = self.apply_taq(input)
            if self.split != 0:
                if self.act_quant_mode == 'qdiff':
                    input_0 = self.act_quantizer(input[:, :self.split, :, :])
                    input_1 = self.act_quantizer_0(input[:, self.split:, :, :])
                input = torch.cat([input_0, input_1], dim=1)
            else:
                if self.act_quant_mode == 'qdiff':
                    input = self.act_quantizer(input)

        if self.weight_quant:
            if self.split != 0:
                weight_0 = self.weight_quantizer(self.weight[:, :self.split, ...])
                weight_1 = self.weight_quantizer_0(self.weight[:, self.split:, ...])
                weight = torch.cat([weight_0, weight_1], dim=1)
            else:
                lora_weight = self.loraB.weight @ self.loraA.weight
                lora_weight_out = self.loraB_out.weight @ self.loraA_out.weight
                if self.smooth_quant:
                    # during the weight init stage
                    if self.weight_quantizer.timestep_wise is None: # reinit the weight_quantizer
                        self.weight_quantizer.timestep_wise = True
                        self.weight_quantizer.n_timestep = len(self.timerange)
                    self.weight_quantizer.cur_timestep_id = cur_timerange_id
                    weight = self.weight_quantizer(self.weight * channel_wise_scale + lora_weight)
                else:
                    weight = self.weight_quantizer(self.weight + lora_weight)
                if self.tarq is None:
                    weight = weight + lora_weight_out
            bias = self.bias
        else:
            if self.smooth_quant:
                weight = self.weight * channel_wise_scale
            else:
                weight = self.weight
            bias = self.bias


        if weight.dtype == torch.float32 and input.dtype == torch.float16:
            weight = weight.to(torch.float16)


        out = self.fwd_func(input, weight, bias, **self.fwd_kwargs)  # 在输出的channel上进行channel_wise的量化
        if self.weight_quant and self.tarq is not None and self.split == 0:
            # ffn linears see the full frame-major token sequence [B, T*S, C]
            T = self.act_quant_params['n_temporal_token']
            S = self.act_quant_params['n_spatial_token']
            assert input.ndim == 3 and input.shape[1] == T * S, \
                f"TARQ sequence layout expects [B, T*S, C], got {tuple(input.shape)}"
            out = out + self.tarq(input, input.shape[0], T, S, layout="sequence")
        out = self.activation_function(out)

        if not torch.isfinite(out).all():
            raise FloatingPointError(
                f"Non-finite activation in {self.__class__.__name__}"
            )


        return out

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):  # 判断是否设置为量化模式！！！
        self.weight_quant = weight_quant
        self.act_quant = act_quant

    def get_quant_state(self):
        return self.weight_quant, self.act_quant

    def set_split(self):
        self.weight_quantizer_0 = WeightQuantizer(self.weight_quant_params)
        if self.act_quant_mode == 'qdiff':
            self.act_quantizer_0 = ActQuantizer(self.act_quant_params)
