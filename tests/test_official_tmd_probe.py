import json

import torch
from torch import nn

from qdiff.optimization.block_recon import _run_official_tmd_gradient_probe
from qdiff.utils import LossFunction


class ProbeBlock(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.offset = nn.Parameter(torch.randn(shape) * 0.01)


def test_official_tmd_probe_records_raw_and_weighted_gradients(tmp_path):
    torch.manual_seed(0)
    shape = (2, 4, 4, 3, 3)
    block = ProbeBlock(shape)
    target = torch.randn(shape)
    prediction = torch.randn(shape) + block.offset
    loss = LossFunction(
        block,
        reconstruction_loss_type="relation",
        use_reconstruction_loss=True,
        mtd_config={"method": {"frame_axis": "BASELINE"}},
    )

    total = loss(prediction, target)
    components = loss.last_component_tensors
    expected = (
        components["reconstruction_base"]
        + components["official_tmd_weighted"]
    )
    assert torch.allclose(total, expected)
    assert torch.allclose(
        components["official_tmd_weighted"],
        100.0 * components["official_tmd_raw"],
    )

    output = tmp_path / "official_tmd_gradients.jsonl"
    assert _run_official_tmd_gradient_probe(
        block, loss, iteration=1, grad_scale=128.0, output_path=str(output)
    )
    record = json.loads(output.read_text().strip())
    assert record["coefficient"] == 100.0
    assert record["reconstruction"]["grad_norm"] > 0
    assert record["tmd_raw"]["grad_norm"] > 0
    assert record["tmd_weighted"]["grad_norm"] > 0
    assert abs(
        record["tmd_weighted"]["grad_norm"]
        / record["tmd_raw"]["grad_norm"]
        - 100.0
    ) < 1.0e-3
