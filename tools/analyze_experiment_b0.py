#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch


def metrics(reference, candidate):
    ref = reference.double().reshape(-1)
    cur = candidate.double().reshape(-1)
    diff = cur - ref
    ref_norm = torch.linalg.vector_norm(ref)
    diff_norm = torch.linalg.vector_norm(diff)
    return {
        "l2_error": float(diff_norm),
        "relative_l2_error": float(diff_norm / ref_norm.clamp_min(1e-12)),
        "rmse": float(torch.sqrt(torch.mean(diff.square()))),
        "cosine_similarity": float(torch.nn.functional.cosine_similarity(ref, cur, dim=0)),
        "max_abs_error": float(diff.abs().max()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root)

    latents = {}
    for mode in ("fp", "w4", "a6", "w4a6"):
        path = root / mode / "runtime" / "final_latents" / "final_latent_0000.pt"
        latents[mode] = torch.load(path, map_location="cpu")

    report = {
        "reference": "fp",
        "latent_shape": list(latents["fp"].shape),
        "comparisons": {
            mode: metrics(latents["fp"], latents[mode])
            for mode in ("w4", "a6", "w4a6")
        },
        "trace_validation": {},
        "init_noise_validation": {},
    }

    fp_noise = torch.load(
        root / "fp" / "runtime" / "init_noise" / "init_noise_0000.pt",
        map_location="cpu",
    )
    for mode in ("w4", "a6", "w4a6"):
        mode_noise = torch.load(
            root / mode / "runtime" / "init_noise" / "init_noise_0000.pt",
            map_location="cpu",
        )
        report["init_noise_validation"][mode] = {
            "exactly_equal_to_fp": bool(torch.equal(fp_noise, mode_noise)),
            "max_abs_difference": float((fp_noise - mode_noise).abs().max()),
        }

    for mode, expected in {
        "w4": (True, False),
        "a6": (False, True),
        "w4a6": (True, True),
    }.items():
        trace_path = root / mode / "runtime" / "quant_trace_batch_0000.json"
        trace = json.loads(trace_path.read_text())
        active = [row for row in trace if row["in_quant_window"]]
        report["trace_validation"][mode] = {
            "total_steps": len(trace),
            "active_steps": len(active),
            "active_progress": [row["sampling_progress"] for row in active],
            "expected_weight_quant": expected[0],
            "expected_act_quant": expected[1],
            "state_matches": all(
                row["weight_quant"] == expected[0]
                and row["act_quant"] == expected[1]
                for row in active
            ),
        }

    output = root / "b0_summary.json"
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
