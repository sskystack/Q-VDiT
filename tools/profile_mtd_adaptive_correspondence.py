#!/usr/bin/env python3
"""Profile adaptive MTD candidates from saved paired FP/quant features.

The experiment is flow-free and does not depend on the repository's RAFT
diagnostic.  It consumes ``step_*.pt`` files produced by
``qdiff.mtd_feature_profiler`` and writes compact JSON/CSV evidence.
"""

import argparse
import csv
import json
from pathlib import Path

import torch

from qdiff.mtd_adaptive_diagnostics import (
    DiagnosticConfig,
    analyze_adaptive_correspondence,
    summarize_proxy,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--teacher-key", default="fp_cond_pooled")
    parser.add_argument("--student-key", default="quant_cond_pooled")
    parser.add_argument("--search-radius", type=int, default=2)
    parser.add_argument("--topk", type=int, default=9)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def tensor_mean(value):
    return float(value.detach().float().mean().cpu())


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    files = sorted(args.profile_dir.rglob("step_*.pt"))
    if not files:
        raise FileNotFoundError(f"no step_*.pt files found below {args.profile_dir}")
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    config = DiagnosticConfig(
        search_radius=args.search_radius,
        topk=args.topk,
        temperature=args.temperature,
        seed=args.seed,
    )

    capture_rows = []
    scheme_values = {}
    proxy_values = {}
    need_values = []
    outside_mass = []
    outside_top1 = []

    for path in files:
        payload = torch.load(path, map_location="cpu")
        if args.teacher_key not in payload or args.student_key not in payload:
            raise KeyError(
                f"{path} lacks {args.teacher_key!r} or {args.student_key!r}"
            )
        teacher = payload[args.teacher_key].to(device)
        student = payload[args.student_key].to(device)
        result = analyze_adaptive_correspondence(teacher, student, config)

        row = {
            "path": str(path),
            "prompt": payload.get("prompt", ""),
            "prompt_index": payload.get("prompt_index", ""),
            "sampling_progress": payload.get("sampling_progress", ""),
            "mass_outside_fixed_3x3": tensor_mean(
                result["teacher_mass_outside_fixed_3x3"]
            ),
            "top1_outside_fixed_3x3_rate": tensor_mean(
                result["teacher_top1_outside_fixed_3x3"]
            ),
        }
        outside_mass.append(result["teacher_mass_outside_fixed_3x3"].cpu())
        outside_top1.append(result["teacher_top1_outside_fixed_3x3"].cpu())
        need_values.append(result["need_score"].cpu())

        for scheme, metrics in result["schemes"].items():
            scheme_values.setdefault(scheme, {})
            for metric, value in metrics.items():
                if metric == "teacher_displacement":
                    continue
                scheme_values[scheme].setdefault(metric, []).append(value.cpu())
                row[f"{scheme}_{metric}"] = tensor_mean(value)
        for name, value in result["proxies"].items():
            proxy_values.setdefault(name, []).append(value.cpu())
        capture_rows.append(row)

    scheme_summary = {}
    scheme_rows = []
    for scheme, metrics in scheme_values.items():
        scheme_summary[scheme] = {}
        row = {"scheme": scheme}
        for metric, chunks in metrics.items():
            value = torch.cat([chunk.flatten() for chunk in chunks]).float().mean()
            scheme_summary[scheme][metric] = float(value)
            row[metric] = float(value)
        scheme_rows.append(row)

    need_score = torch.cat([value.flatten() for value in need_values])
    proxy_summary = {
        name: summarize_proxy(
            torch.cat([value.flatten() for value in values]), need_score
        )
        for name, values in proxy_values.items()
    }
    proxy_rows = [
        {"proxy": name, **metrics} for name, metrics in proxy_summary.items()
    ]
    summary = {
        "profile_dir": str(args.profile_dir),
        "captures": len(files),
        "teacher_key": args.teacher_key,
        "student_key": args.student_key,
        "config": vars(config),
        "mass_outside_fixed_3x3": float(
            torch.cat([value.flatten() for value in outside_mass]).float().mean()
        ),
        "top1_outside_fixed_3x3_rate": float(
            torch.cat([value.flatten() for value in outside_top1]).float().mean()
        ),
        "schemes": scheme_summary,
        "proxies": proxy_summary,
        "interpretation": {
            "candidate_test": (
                "The fixed 3x3 hypothesis is weakened when teacher mass/top1 often "
                "falls outside it and an equal-budget adaptive scheme reduces both "
                "student-teacher local KL and transport error."
            ),
            "importance_test": (
                "A proxy is useful only when its Spearman correlation and top-k loss "
                "mass lift exceed the random-area expectation."
            ),
        },
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    write_csv(args.output / "per_capture.csv", capture_rows)
    write_csv(args.output / "scheme_summary.csv", scheme_rows)
    write_csv(args.output / "proxy_summary.csv", proxy_rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

