#!/usr/bin/env python3
"""Isolated W4A6 sensitivity for selected spatial-attention and FFN layers."""

import argparse
import json
import math
from functools import partial
from pathlib import Path

import torch
from mmengine.config import Config
from mmengine.runner import set_random_seed

from opensora.schedulers.iddpm import forward_with_cfg
from opensora.utils.misc import to_torch_dtype
from qdiff.models.quant_layer import QuantLayer
from tools.profile_cfg_stage import assert_full_precision_state
from tools.profile_equal_energy_propagation import run_step
from tools.profile_stage_numeric_mechanism import (
    build_quant_model,
    error_metrics,
    prepare_conditioning,
    profiled_forward_with_cfg,
    set_quant_mode,
)


OPERATORS = ("attn.q", "attn.k", "attn.v", "attn.proj", "mlp.fc1", "mlp.fc2")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--calib-config", required=True)
    parser.add_argument("--quant-ckpt", required=True)
    parser.add_argument("--text-embeds", required=True)
    parser.add_argument("--prompt-indices", default="0,2,6")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--selected-progress", default="5,15,25")
    parser.add_argument("--selected-blocks", default="0,4,9,13,17,21,25,27")
    parser.add_argument("--time-mp-config-weight", required=True)
    parser.add_argument("--time-mp-config-act", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def candidate_layers(qnn, blocks):
    suffixes = (
        "attn.q",
        "attn.k",
        "attn.v",
        "attn.proj",
        "mlp.fc1",
        "mlp.fc2",
    )
    available = {
        name
        for name, module in qnn.model.named_modules()
        if isinstance(module, QuantLayer)
    }
    requested = [f"blocks.{block}.{suffix}" for block in blocks for suffix in suffixes]
    missing = [name for name in requested if name not in available]
    if missing:
        raise KeyError(f"Selected QuantLayers do not exist: {missing}")
    return requested


def main():
    args = parse_args()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    prompts = sorted({int(value) for value in args.prompt_indices.split(",") if value.strip()})
    selected = sorted({int(value) for value in args.selected_progress.split(",") if value.strip()})
    blocks = sorted({int(value) for value in args.selected_blocks.split(",") if value.strip()})

    cfg = Config.fromfile(args.config)
    cfg.multi_resolution = cfg.get("multi_resolution", False)
    dtype = to_torch_dtype(cfg.dtype)
    device = torch.device("cuda")
    set_random_seed(args.seed)
    torch.set_grad_enabled(False)

    scheduler, latent_size, qnn = build_quant_model(args, cfg, device, dtype)
    layers = candidate_layers(qnn, blocks)
    result_path = output / "single_layer_sensitivity.jsonl"
    rows = []
    completed = set()
    if result_path.is_file():
        for line_number, line in enumerate(result_path.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    f"Invalid partial result at {result_path}:{line_number}"
                ) from error
            key = (int(row["prompt_index"]), int(row["progress"]), row["layer"])
            if key in completed:
                raise RuntimeError(f"Duplicate partial single-layer result: {key}")
            completed.add(key)
            rows.append(row)
        print(f"Resuming from {len(rows)} completed single-layer rows", flush=True)
    noise_generator = torch.Generator(device=device).manual_seed(args.seed)
    initial_noise = torch.randn(
        1, qnn.in_channels, *latent_size, device=device, generator=noise_generator
    )
    initial = torch.cat([initial_noise, initial_noise], dim=0)
    for prompt_index in prompts:
        conditioning = prepare_conditioning(args.text_embeds, prompt_index, device, dtype)
        set_quant_mode(qnn, False, False)
        assert_full_precision_state(qnn)
        fp_model = partial(
            forward_with_cfg,
            qnn,
            cfg_scale=cfg.scheduler.cfg_scale,
            return_trajectory=False,
        )
        snapshots = {}
        x = initial.clone()
        for internal_timestep in range(scheduler.num_timesteps - 1, -1, -1):
            progress = scheduler.num_timesteps - internal_timestep
            if progress in selected:
                snapshots[progress] = x.detach().clone()
            x = run_step(scheduler, fp_model, x, internal_timestep, conditioning)

        for progress in selected:
            internal_timestep = scheduler.num_timesteps - progress
            original_timestep = int(scheduler.timestep_map[internal_timestep])
            timestep = torch.tensor(
                [original_timestep] * snapshots[progress].shape[0],
                device=device,
                dtype=torch.long,
            )
            set_quant_mode(qnn, False, False)
            fp = profiled_forward_with_cfg(
                qnn,
                snapshots[progress],
                timestep,
                conditioning,
                cfg.scheduler.cfg_scale,
            )
            for layer in layers:
                key = (prompt_index, progress, layer)
                if key in completed:
                    continue
                set_quant_mode(qnn, False, False)
                qnn.set_layer_quant(
                    model=qnn,
                    module_name_list=[layer],
                    quant_level="per_layer",
                    weight_quant=True,
                    act_quant=True,
                    prefix="",
                )
                quant = profiled_forward_with_cfg(
                    qnn,
                    snapshots[progress],
                    timestep,
                    conditioning,
                    cfg.scheduler.cfg_scale,
                )
                set_quant_mode(qnn, False, False)
                assert_full_precision_state(qnn)
                block = int(layer.split(".")[1])
                operator = ".".join(layer.split(".")[2:])
                row = {
                    "prompt_index": prompt_index,
                    "seed": args.seed,
                    "progress": progress,
                    "original_timestep": original_timestep,
                    "layer": layer,
                    "block_index": block,
                    "operator": operator,
                }
                for branch in ("cfg", "conditional", "unconditional"):
                    for key, value in error_metrics(quant[branch], fp[branch]).items():
                        row[f"{branch}_{key}"] = value
                rows.append(row)
                completed.add(key)
                with result_path.open("a") as handle:
                    handle.write(json.dumps(row) + "\n")
                    handle.flush()
                print(json.dumps(row), flush=True)

    expected_rows = len(prompts) * len(selected) * len(layers)
    if len(rows) != expected_rows:
        raise RuntimeError(
            f"Incomplete single-layer profiling: got {len(rows)}, expected {expected_rows}"
        )
    rows.sort(key=lambda row: (
        int(row["prompt_index"]),
        int(row["progress"]),
        int(row["block_index"]),
        OPERATORS.index(row["operator"]),
    ))
    with result_path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    metadata = {
        "prompt_indices": prompts,
        "seed": args.seed,
        "selected_progress": selected,
        "selected_blocks": blocks,
        "layers": layers,
        "rows": len(rows),
        "expected_rows": expected_rows,
        "all_values_finite": all(
            math.isfinite(value)
            for row in rows
            for value in row.values()
            if isinstance(value, float)
        ),
        "fp_snapshots_generated_before_isolated_quantization": True,
        "fp_state_restored_after_each_layer": True,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
