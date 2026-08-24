#!/usr/bin/env python3
"""Aggregate implementation-aligned temporal TQE branch profiling."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


CANDIDATES = (
    "no_tqe",
    "win_only",
    "eq8_masked_wout_only",
    "wout_unmasked_plus_masked",
    "previous_profiler_approximation",
    "opensource_actual",
)
PHASES = {
    "early": {5, 15, 25},
    "middle": {45, 65},
    "late": {85, 95},
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def mean(values):
    return sum(values) / len(values) if values else float("nan")


def phase_of(progress):
    for phase, points in PHASES.items():
        if progress in points:
            return phase
    return "unassigned"


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in args.inputs:
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row.get("operator") != "temporal_tqe_branch_decomposition":
                continue
            row["source"] = str(path)
            row["phase"] = phase_of(int(row["progress"]))
            rows.append(row)
    if not rows:
        raise RuntimeError("No temporal_tqe_branch_decomposition rows found")

    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["phase"], int(row["progress"]))].append(row)
        grouped[(row["phase"], None)].append(row)
        grouped[("all", None)].append(row)

    summaries = []
    for (phase, progress), subset in sorted(
        grouped.items(), key=lambda item: (item[0][0], -1 if item[0][1] is None else item[0][1])
    ):
        candidate_errors = {
            candidate: mean(
                [row[f"relative_error_to_local_fp__{candidate}"] for row in subset]
            )
            for candidate in CANDIDATES
        }
        no_tqe = candidate_errors["no_tqe"]
        best = min(candidate_errors, key=candidate_errors.get)
        summaries.append(
            {
                "phase": phase,
                "progress": progress,
                "cell_count": len(subset),
                "best_candidate": best,
                "best_mean_relative_error": candidate_errors[best],
                "mean_omitted_masked_branch_over_actual": mean(
                    [row["omitted_masked_branch_over_actual"] for row in subset]
                ),
                "mean_previous_vs_actual_cosine": mean(
                    [row["previous_vs_actual_cosine"] for row in subset]
                ),
                "mean_mask_mean": mean([row["mask_mean"] for row in subset]),
                **{
                    f"mean_error__{candidate}": error
                    for candidate, error in candidate_errors.items()
                },
                **{
                    f"mean_improvement_over_no_tqe__{candidate}": no_tqe - error
                    for candidate, error in candidate_errors.items()
                },
                **{
                    f"win_fraction_over_no_tqe__{candidate}": mean(
                        [
                            float(
                                row[f"relative_error_to_local_fp__{candidate}"]
                                < row["relative_error_to_local_fp__no_tqe"]
                            )
                            for row in subset
                        ]
                    )
                    for candidate in CANDIDATES
                    if candidate != "no_tqe"
                },
            }
        )

    overall = next(
        row for row in summaries if row["phase"] == "all" and row["progress"] is None
    )
    decision = {
        "profiled_cells": len(rows),
        "prompt_sources": len(args.inputs),
        "opensource_actual_mean_improvement_over_no_tqe": overall[
            "mean_improvement_over_no_tqe__opensource_actual"
        ],
        "opensource_actual_win_fraction_over_no_tqe": overall[
            "win_fraction_over_no_tqe__opensource_actual"
        ],
        "eq8_only_mean_improvement_over_no_tqe": overall[
            "mean_improvement_over_no_tqe__eq8_masked_wout_only"
        ],
        "eq8_only_win_fraction_over_no_tqe": overall[
            "win_fraction_over_no_tqe__eq8_masked_wout_only"
        ],
        "mean_omitted_masked_branch_over_actual": overall[
            "mean_omitted_masked_branch_over_actual"
        ],
        "interpretation_limit": (
            "This is a local same-input operator reconstruction test. It diagnoses branch "
            "attribution and phase dependence, but final video quality still requires the "
            "FP16/W4A6/MTD trajectory and VBench comparison."
        ),
    }
    (args.output / "summary.json").write_text(
        json.dumps({"decision": decision, "groups": summaries}, indent=2),
        encoding="utf-8",
    )
    with (args.output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
