#!/usr/bin/env python3
"""Aggregate and visualize isolated single-layer W4A6 sensitivity results."""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OPERATORS = ("attn.q", "attn.k", "attn.v", "attn.proj", "mlp.fc1", "mlp.fc2")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows, keys, metric="cfg_relative_l2"):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(float(row[metric]))
    result = []
    for key, values in sorted(groups.items()):
        array = np.asarray(values, dtype=np.float64)
        result.append({
            **dict(zip(keys, key)),
            "count": int(array.size),
            "mean": float(array.mean()),
            "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
            "median": float(np.median(array)),
            "min": float(array.min()),
            "max": float(array.max()),
        })
    return result


def main():
    args = parse_args()
    input_path = Path(args.input)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in input_path.read_text().splitlines() if line.strip()]
    if not rows:
        raise RuntimeError(f"No rows in {input_path}")

    layer_rows = aggregate(rows, ("progress", "block_index", "operator", "layer"))
    operator_rows = aggregate(rows, ("progress", "operator"))
    block_operator_rows = aggregate(rows, ("progress", "block_index", "operator"))
    write_csv(output / "single_layer_aggregate.csv", layer_rows)
    write_csv(output / "operator_aggregate.csv", operator_rows)
    write_csv(output / "block_operator_aggregate.csv", block_operator_rows)

    progresses = sorted({int(row["progress"]) for row in rows})
    blocks = sorted({int(row["block_index"]) for row in rows})
    layers = sorted({row["layer"] for row in rows}, key=lambda value: (
        int(value.split(".")[1]), OPERATORS.index(".".join(value.split(".")[2:])),
    ))
    layer_lookup = {(int(row["progress"]), row["layer"]): float(row["mean"]) for row in layer_rows}
    heat = np.asarray([[layer_lookup[(progress, layer)] for progress in progresses] for layer in layers])
    fig, axis = plt.subplots(figsize=(7.2, max(10.0, len(layers) * 0.25)))
    image = axis.imshow(heat, aspect="auto", cmap="magma")
    axis.set_xticks(range(len(progresses)), progresses)
    axis.set_yticks(range(len(layers)), layers, fontsize=7)
    axis.set_xlabel("Sampling progress")
    axis.set_ylabel("Isolated QuantLayer")
    axis.set_title("Single-layer W4A6 output sensitivity (mean across prompts)")
    fig.colorbar(image, ax=axis, label="CFG output relative L2")
    fig.tight_layout()
    fig.savefig(output / "single_layer_timestep_heatmap.png", dpi=220)
    plt.close(fig)

    lookup = {
        (int(row["progress"]), int(row["block_index"]), row["operator"]): float(row["mean"])
        for row in block_operator_rows
    }
    fig, axes = plt.subplots(1, len(progresses), figsize=(5.2 * len(progresses), 6.2), sharey=True)
    axes = np.atleast_1d(axes)
    vmax = max(lookup.values())
    for axis, progress in zip(axes, progresses):
        matrix = np.asarray([[lookup[(progress, block, op)] for op in OPERATORS] for block in blocks])
        image = axis.imshow(matrix, aspect="auto", cmap="viridis", vmin=0, vmax=vmax)
        axis.set_title(f"progress {progress}")
        axis.set_xticks(range(len(OPERATORS)), OPERATORS, rotation=55, ha="right")
        axis.set_yticks(range(len(blocks)), blocks)
        axis.set_xlabel("Operator")
    axes[0].set_ylabel("Block index")
    fig.colorbar(image, ax=axes.ravel().tolist(), label="CFG output relative L2", shrink=0.82)
    fig.suptitle("Block × operator isolated W4A6 sensitivity", y=1.01)
    fig.subplots_adjust(left=0.08, right=0.92, bottom=0.20, top=0.90, wspace=0.18)
    fig.savefig(output / "block_operator_timestep_heatmap.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    prompt_winners = []
    for prompt in sorted({int(row["prompt_index"]) for row in rows}):
        for progress in progresses:
            subset = [row for row in rows if int(row["prompt_index"]) == prompt and int(row["progress"]) == progress]
            winner = max(subset, key=lambda row: float(row["cfg_relative_l2"]))
            prompt_winners.append({
                "prompt_index": prompt,
                "progress": progress,
                "layer": winner["layer"],
                "block_index": int(winner["block_index"]),
                "operator": winner["operator"],
                "cfg_relative_l2": float(winner["cfg_relative_l2"]),
            })
    write_csv(output / "prompt_progress_winners.csv", prompt_winners)

    top_by_progress = {}
    for progress in progresses:
        candidates = [row for row in layer_rows if int(row["progress"]) == progress]
        top_by_progress[str(progress)] = sorted(candidates, key=lambda row: row["median"], reverse=True)[:10]
    summary = {
        "rows": len(rows),
        "expected_rows": len({row["prompt_index"] for row in rows}) * len(progresses) * len(layers),
        "all_values_finite": all(
            math.isfinite(float(value))
            for row in rows
            for value in row.values()
            if isinstance(value, (int, float))
        ),
        "prompts": sorted({int(row["prompt_index"]) for row in rows}),
        "progresses": progresses,
        "blocks": blocks,
        "operators": list(OPERATORS),
        "top_layers_by_progress_ranked_by_cross_prompt_median": top_by_progress,
    }
    (output / "single_layer_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({key: summary[key] for key in ("rows", "expected_rows", "all_values_finite")}, indent=2))


if __name__ == "__main__":
    main()
