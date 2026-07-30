#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch


MODES = ("w4", "a6", "w4a6")
WINDOWS = tuple((start, start + 9) for start in range(1, 100, 10))


def latent_metrics(reference, candidate):
    ref = reference.double().reshape(-1)
    cur = candidate.double().reshape(-1)
    diff = cur - ref
    ref_norm = torch.linalg.vector_norm(ref).clamp_min(1e-12)
    return {
        "relative_l2_error": float(torch.linalg.vector_norm(diff) / ref_norm),
        "rmse": float(torch.sqrt(torch.mean(diff.square()))),
        "cosine_similarity": float(torch.nn.functional.cosine_similarity(ref, cur, dim=0)),
        "max_abs_error": float(diff.abs().max()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    fp = torch.load(root / "fp/runtime/final_latents/final_latent_0000.pt", map_location="cpu")
    fp_noise = torch.load(root / "fp/runtime/init_noise/init_noise_0000.pt", map_location="cpu")

    rows = []
    for start, end in WINDOWS:
        tag = f"p{start:03d}_{end:03d}"
        for mode in MODES:
            base = root / mode / tag / "runtime"
            latent = torch.load(base / "final_latents/final_latent_0000.pt", map_location="cpu")
            noise = torch.load(base / "init_noise/init_noise_0000.pt", map_location="cpu")
            trace = json.loads((base / "quant_trace_batch_0000.json").read_text())
            active = [item for item in trace if item["in_quant_window"]]
            expected_state = {
                "w4": (True, False),
                "a6": (False, True),
                "w4a6": (True, True),
            }[mode]
            row = {
                "mode": mode,
                "progress_start": start,
                "progress_end": end,
                "progress_center": (start + end) / 2,
                "latent_finite": bool(torch.isfinite(latent).all()),
                "initial_noise_exact": bool(torch.equal(fp_noise, noise)),
                "trace_total_steps": len(trace),
                "trace_active_steps": len(active),
                "trace_progress_correct": [x["sampling_progress"] for x in active]
                == list(range(start, end + 1)),
                "trace_state_correct": all(
                    x["weight_quant"] == expected_state[0]
                    and x["act_quant"] == expected_state[1]
                    for x in active
                ),
                **latent_metrics(fp, latent),
            }
            rows.append(row)

    summary = {
        "reference": "B0 FP prompt0 seed42",
        "windows": [list(x) for x in WINDOWS],
        "rows": rows,
        "all_latents_finite": all(x["latent_finite"] for x in rows),
        "all_initial_noise_exact": all(x["initial_noise_exact"] for x in rows),
        "all_traces_valid": all(
            x["trace_total_steps"] == 100
            and x["trace_active_steps"] == 10
            and x["trace_progress_correct"]
            and x["trace_state_correct"]
            for x in rows
        ),
    }
    (root / "b1_summary.json").write_text(json.dumps(summary, indent=2))

    with (root / "b1_sensitivity.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    labels = {"w4": "W4-only", "a6": "A6-only", "w4a6": "W4A6"}
    colors = {"w4": "#377eb8", "a6": "#e41a1c", "w4a6": "#4daf4a"}
    for mode in MODES:
        selected = [x for x in rows if x["mode"] == mode]
        xs = [x["progress_center"] for x in selected]
        axes[0].plot(xs, [x["relative_l2_error"] for x in selected], marker="o", label=labels[mode], color=colors[mode])
        axes[1].plot(xs, [1.0 - x["cosine_similarity"] for x in selected], marker="o", label=labels[mode], color=colors[mode])
        axes[2].plot(xs, [x["rmse"] for x in selected], marker="o", label=labels[mode], color=colors[mode])
    axes[0].set_ylabel("Final latent relative L2")
    axes[1].set_ylabel("1 - cosine similarity")
    axes[2].set_ylabel("Final latent RMSE")
    for ax in axes:
        ax.set_xlabel("Sampling progress window center")
        ax.grid(alpha=0.25)
    axes[0].legend()
    fig.suptitle("Experiment B1: causal timestep-window quantization sensitivity")
    fig.tight_layout()
    fig.savefig(root / "b1_sensitivity_curves.png", dpi=180)
    plt.close(fig)

    print(json.dumps({
        "all_latents_finite": summary["all_latents_finite"],
        "all_initial_noise_exact": summary["all_initial_noise_exact"],
        "all_traces_valid": summary["all_traces_valid"],
        "rows": len(rows),
    }, indent=2))


if __name__ == "__main__":
    main()
