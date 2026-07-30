#!/usr/bin/env python3
"""Create equal-size uniform, early-heavy, and late-heavy calibration sets."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-root", required=True)
    # Keep the prompt set identical to calibration-stability subset_0 so that
    # only the timestep sampling distribution changes across schemes.
    parser.add_argument("--source-positions", default="0,1,2")
    return parser.parse_args()


def evenly_resample(pool, count):
    positions = np.linspace(0, len(pool) - 1, count).round().astype(int)
    return [pool[index] for index in positions]


def main():
    args = parse_args()
    source = torch.load(args.input, map_location="cpu")
    positions = [int(value) for value in args.source_positions.split(",") if value.strip()]
    width = source["xs"].shape[1]
    if width % 2 or not torch.equal(source["ts"][:, 0::2], source["ts"][:, 1::2]):
        raise ValueError("Expected adjacent conditional/unconditional CFG pairs")
    cfg_indices = [index for position in positions for index in (2 * position, 2 * position + 1)]

    # The saved first axis is ascending original diffusion timestep.  The
    # stored sampling_step runs 49 -> 0, so generation progress is
    # sampling_step + 1: progress 1 is high-noise/early, progress 50 is late.
    progress = source["sampling_step"][:, 0].long() + 1
    early = torch.where(progress <= 15)[0].tolist()
    middle = torch.where((progress >= 16) & (progress <= 25))[0].tolist()
    late = torch.where(progress >= 26)[0].tolist()
    schemes = {
        "uniform": list(range(source["xs"].shape[0])),
        "early_heavy": evenly_resample(early, 30) + evenly_resample(middle, 10) + evenly_resample(late, 10),
        "late_heavy": evenly_resample(early, 10) + evenly_resample(middle, 10) + evenly_resample(late, 30),
    }

    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, timestep_indices in schemes.items():
        subset = {}
        for key, value in source.items():
            if torch.is_tensor(value) and value.ndim >= 2 and value.shape[:2] == source["xs"].shape[:2]:
                subset[key] = value[timestep_indices][:, cfg_indices].contiguous()
            elif torch.is_tensor(value) and value.ndim >= 2 and value.shape[0] == source["xs"].shape[0] and value.shape[1] == width:
                subset[key] = value[timestep_indices][:, cfg_indices].contiguous()
            else:
                subset[key] = value
        outdir = root / name
        outdir.mkdir(parents=True, exist_ok=True)
        path = outdir / "calib_data.pt"
        torch.save(subset, path)
        selected_progress = progress[timestep_indices].tolist()
        manifest[name] = {
            "path": str(path),
            "source_positions": positions,
            "cfg_indices": cfg_indices,
            "timestep_indices": timestep_indices,
            "selected_progress": selected_progress,
            "early_count": sum(value <= 15 for value in selected_progress),
            "middle_count": sum(16 <= value <= 25 for value in selected_progress),
            "late_count": sum(value >= 26 for value in selected_progress),
        }
        if subset["xs"].shape[:2] != (50, 6):
            raise AssertionError(f"Unexpected output shape for {name}: {subset['xs'].shape}")
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
