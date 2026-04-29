import torch
import torch.nn as nn
from omegaconf import OmegaConf

from qdiff.models.quant_layer import QuantLayer
from qdiff.models.stdit_quant_layer import QuantTemporalAttnLinear


def make_weight_quant_config():
    return OmegaConf.create(
        {
            "n_bits": 3,
            "per_group": "channel",
            "scale_method": "grid_search_lp",
            "round_mode": "nearest_ste",
            "sym": False,
        }
    )


def make_act_quant_config():
    return OmegaConf.create(
        {
            "n_bits": 8,
            "per_group": "token",
            "scale_method": "min_max",
            "round_mode": "nearest_ste",
            "sym": False,
            "n_temporal_token": 4,
            "n_spatial_token": 2,
            "smooth_quant": {"enable": False},
            "dynamic": True,
        }
    )


def run_quant_layer_check(device: str):
    layer = QuantLayer(
        nn.Linear(8, 6, bias=True),
        weight_quant_params=make_weight_quant_config(),
        act_quant_params=make_act_quant_config(),
    ).to(device)
    layer.set_quant_state(weight_quant=True, act_quant=False)
    x = torch.randn(3, 8, device=device, dtype=torch.float16)
    y = layer(x)
    assert y.dtype == torch.float16, y.dtype


def run_temporal_layer_check(device: str):
    layer = QuantTemporalAttnLinear(
        nn.Linear(8, 6, bias=True),
        weight_quant_params=make_weight_quant_config(),
        act_quant_params=make_act_quant_config(),
    ).to(device)
    layer.set_quant_state(weight_quant=True, act_quant=False)
    layer.cur_timestep_id = 0
    # BS=3, S=2, T=4 -> input shape [BS*S, T, C]
    x = torch.randn(6, 4, 8, device=device, dtype=torch.float16)
    y = layer(x)
    assert y.dtype == torch.float16, y.dtype


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to reproduce the fp16/fp32 mismatch path.")
    device = "cuda"
    run_quant_layer_check(device)
    run_temporal_layer_check(device)
    print("dtype alignment checks passed")


if __name__ == "__main__":
    main()
