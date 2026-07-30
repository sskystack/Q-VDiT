#!/usr/bin/env python3
"""Create position-disjoint, CFG-paired calibration subsets.

The OpenSora calibration file generated with ``batch_size=1`` is stored as
``[cond_last, uncond_last, ..., cond_first, uncond_first]`` because
``get_calib_data.py`` prepends each newly generated batch.  Therefore each
source sample is an adjacent pair, not one element from each half.
"""

import argparse
import json
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--subsets", default="0,1,2;3,4,5;6,7,8")
    return parser.parse_args()


def validate_adjacent_cfg_layout(source):
    """Validate the structural invariants needed for adjacent CFG pairs."""
    required = ("xs", "ts", "cond_emb", "mask")
    missing = [key for key in required if key not in source]
    if missing:
        raise KeyError(f"Missing required calibration tensors: {missing}")

    width = source["xs"].shape[1]
    if width % 2:
        raise ValueError(f"Expected an even CFG axis, got width={width}")
    for key in required:
        value = source[key]
        if not torch.is_tensor(value) or value.ndim < 2 or value.shape[1] != width:
            raise ValueError(
                f"Tensor {key!r} does not share the expected CFG axis width {width}: "
                f"shape={getattr(value, 'shape', None)}"
            )

    # Every adjacent pair must represent the same diffusion timestep.  This is
    # a safe structural check even though cond/uncond latents need not remain
    # numerically identical throughout the saved trajectory.
    ts = source["ts"]
    if not torch.equal(ts[:, 0::2], ts[:, 1::2]):
        raise ValueError("Calibration data is not adjacent CFG-paired: timestep pairs differ")
    return width // 2


def main():
    args = parse_args()
    source = torch.load(args.input, map_location="cpu")
    position_count = validate_adjacent_cfg_layout(source)
    subsets = [
        [int(x) for x in group.split(",") if x.strip()]
        for group in args.subsets.split(";")
        if group.strip()
    ]
    flattened = [position for subset in subsets for position in subset]
    if len(set(flattened)) != len(flattened):
        raise ValueError(f"Subsets are not position-disjoint: {subsets}")
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    manifest = []
    for subset_id, positions in enumerate(subsets):
        if len(set(positions)) != len(positions):
            raise ValueError(f"Subset {subset_id} contains duplicate positions: {positions}")
        if any(position < 0 or position >= position_count for position in positions):
            raise IndexError(
                f"Subset {subset_id} has a position outside [0, {position_count - 1}]: "
                f"{positions}"
            )
        # Source layout is [cond_position, uncond_position] for each adjacent
        # pair. Preserve this order because get_quant_calib_data slices the
        # first n_samples*2 entries.
        indices = [index for position in positions for index in (2 * position, 2 * position + 1)]
        subset = {}
        for key, value in source.items():
            if torch.is_tensor(value) and value.ndim >= 2 and value.shape[1] == 2 * position_count:
                subset[key] = value[:, indices].contiguous()
            else:
                subset[key] = value
        subset_dir = root / f"subset_{subset_id}"
        subset_dir.mkdir(parents=True, exist_ok=True)
        path = subset_dir / "calib_data.pt"
        torch.save(subset, path)
        manifest.append(
            {
                "subset_id": subset_id,
                "source_positions": positions,
                "cfg_indices": indices,
                "source_layout": "adjacent_cond_uncond_pairs_in_reverse_generation_batch_order",
                "path": str(path),
            }
        )
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
