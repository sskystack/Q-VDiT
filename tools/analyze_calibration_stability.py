#!/usr/bin/env python3
"""Compare quantization checkpoints produced from disjoint calibration subsets."""

import argparse
import csv
import itertools
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--labels", nargs="+")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def flatten_tensors(value, prefix=""):
    rows = {}
    if torch.is_tensor(value):
        # Only learned quantization/calibration parameters are relevant to
        # calibration-set stability. Skipping unchanged base-model tensors
        # avoids tens of millions of needless FP64 operations.
        if family(prefix) != "other":
            rows[prefix] = value.detach().float().cpu()
    elif isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            rows.update(flatten_tensors(child, child_prefix))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            rows.update(flatten_tensors(child, f"{prefix}[{index}]"))
    return rows


def family(key):
    if ".weight_quantizer" in key and key.endswith(".delta"):
        return "weight_delta"
    if ".weight_quantizer" in key and key.endswith(".zero_point"):
        return "weight_zero_point"
    if ".loraA_out.weight" in key:
        return "loraA_out"
    if ".loraB_out.weight" in key:
        return "loraB_out"
    if ".loraA.weight" in key:
        return "loraA"
    if ".loraB.weight" in key:
        return "loraB"
    return "other"


def region(key):
    match = re.search(r"(?:^|\.)blocks\.(\d+)\.", key)
    if not match:
        return "non_block"
    block = int(match.group(1))
    if block <= 8:
        return "early_blocks"
    if block <= 18:
        return "middle_blocks"
    return "late_blocks"


def pair_metrics(a, b):
    a = a.reshape(-1).float()
    b = b.reshape(-1).float()
    diff = a - b
    a_norm = torch.linalg.vector_norm(a)
    b_norm = torch.linalg.vector_norm(b)
    diff_norm = torch.linalg.vector_norm(diff)
    denom = ((a_norm + b_norm) / 2).clamp_min(1e-30)
    cosine = torch.dot(a, b) / (a_norm * b_norm).clamp_min(1e-30)
    return {
        "numel": int(a.numel()),
        "a_l2": float(a_norm),
        "b_l2": float(b_norm),
        "difference_l2": float(diff_norm),
        "symmetric_relative_l2": float(diff_norm / denom),
        "cosine": float(cosine),
    }


def effective_lora_specs(states, labels):
    shared = set.intersection(*(set(states[label]) for label in labels))
    for a_key in sorted(shared):
        if a_key.endswith(".loraA.weight"):
            b_key = a_key[: -len(".loraA.weight")] + ".loraB.weight"
            fam = "lora_effective"
        elif a_key.endswith(".loraA_out.weight"):
            b_key = a_key[: -len(".loraA_out.weight")] + ".loraB_out.weight"
            fam = "lora_effective_out"
        else:
            continue
        if b_key not in shared:
            continue
        compatible = all(
            states[label][a_key].ndim == 2
            and states[label][b_key].ndim == 2
            and states[label][b_key].shape[1] == states[label][a_key].shape[0]
            for label in labels
        )
        if compatible:
            yield a_key, b_key, fam


def append_aggregate_rows(target, accumulators, pair_labels=None):
    for group, acc in sorted(accumulators.items()):
        if pair_labels is None:
            fam, reg = group
            prefix = {}
        else:
            label_a, label_b, fam, reg = group
            prefix = {"checkpoint_a": label_a, "checkpoint_b": label_b}
        a_norm = math.sqrt(acc.get("a2", 0.0))
        b_norm = math.sqrt(acc.get("b2", 0.0))
        diff_norm = math.sqrt(acc["diff2"])
        denom = max((a_norm + b_norm) / 2, 1e-30)
        cosine_denom = max(a_norm * b_norm, 1e-30)
        target.append(
            {
                **prefix,
                "family": fam,
                "region": reg,
                "tensor_keys": acc["keys"],
                "numel": acc["numel"],
                "symmetric_relative_l2": diff_norm / denom,
                "cosine": acc["dot"] / cosine_denom,
            }
        )


def main():
    args = parse_args()
    torch.set_num_threads(4)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    labels = args.labels or [Path(path).parent.name for path in args.checkpoints]
    if len(labels) != len(args.checkpoints):
        raise ValueError("--labels must have the same length as --checkpoints")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    states = {}
    for label, path in zip(labels, args.checkpoints):
        states[label] = flatten_tensors(torch.load(path, map_location="cpu"))

    detail_rows = []
    aggregate_rows = []
    for label_a, label_b in itertools.combinations(labels, 2):
        state_a, state_b = states[label_a], states[label_b]
        shared = sorted(set(state_a) & set(state_b))
        accumulators = defaultdict(lambda: {"diff2": 0.0, "a2": 0.0, "b2": 0.0, "dot": 0.0, "numel": 0, "keys": 0})
        for key in shared:
            if state_a[key].shape != state_b[key].shape:
                continue
            metrics = pair_metrics(state_a[key], state_b[key])
            fam = family(key)
            reg = region(key)
            detail_rows.append(
                {"checkpoint_a": label_a, "checkpoint_b": label_b, "key": key,
                 "family": fam, "region": reg, **metrics}
            )
            for group in ((fam, "all"), (fam, reg), ("all", "all")):
                acc = accumulators[group]
                acc["diff2"] += metrics["difference_l2"] ** 2
                acc["a2"] += metrics["a_l2"] ** 2
                acc["b2"] += metrics["b_l2"] ** 2
                acc["dot"] += float((state_a[key].reshape(-1).float() * state_b[key].reshape(-1).float()).sum())
                acc["numel"] += metrics["numel"]
                acc["keys"] += 1
        for (fam, reg), acc in sorted(accumulators.items()):
            a_norm, b_norm, diff_norm = math.sqrt(acc["a2"]), math.sqrt(acc["b2"]), math.sqrt(acc["diff2"])
            denom = max((a_norm + b_norm) / 2, 1e-30)
            cosine_denom = max(a_norm * b_norm, 1e-30)
            aggregate_rows.append(
                {
                    "checkpoint_a": label_a,
                    "checkpoint_b": label_b,
                    "family": fam,
                    "region": reg,
                    "tensor_keys": acc["keys"],
                    "numel": acc["numel"],
                    "symmetric_relative_l2": diff_norm / denom,
                    "cosine": acc["dot"] / cosine_denom,
                }
            )

    # Direct population variance across all calibration checkpoints. Pairwise
    # distance is useful diagnostically but is not itself Var_Dcalib[theta].
    variance_detail_rows = []
    variance_accumulators = defaultdict(
        lambda: {"deviation2": 0.0, "energy2": 0.0, "numel": 0, "keys": 0}
    )
    shared_all = sorted(set.intersection(*(set(state) for state in states.values())))
    for key in shared_all:
        tensors = [states[label][key] for label in labels]
        if not all(tensor.shape == tensors[0].shape for tensor in tensors[1:]):
            continue
        values = [tensor.reshape(-1).float() for tensor in tensors]
        mean = values[0].clone()
        for value in values[1:]:
            mean.add_(value)
        mean.div_(len(values))
        deviation2 = sum(float((value - mean).square().sum()) for value in values)
        energy2 = sum(float(value.square().sum()) for value in values)
        numel = int(values[0].numel())
        population_mean_element_variance = deviation2 / max(len(values) * numel, 1)
        mean_element_second_moment = energy2 / max(len(values) * numel, 1)
        relative_rms_std = math.sqrt(deviation2 / max(energy2, 1e-30))
        fam = family(key)
        reg = region(key)
        variance_detail_rows.append(
            {
                "key": key,
                "family": fam,
                "region": reg,
                "num_checkpoints": len(values),
                "numel": numel,
                "population_mean_element_variance": population_mean_element_variance,
                "mean_element_second_moment": mean_element_second_moment,
                "relative_rms_std": relative_rms_std,
            }
        )
        for group in ((fam, "all"), (fam, reg), ("all", "all")):
            acc = variance_accumulators[group]
            acc["deviation2"] += deviation2
            acc["energy2"] += energy2
            acc["numel"] += numel
            acc["keys"] += 1

    # A and B factors are not individually identifiable: A can be rescaled or
    # changed by an invertible rank-space basis while B changes inversely.
    # Compare the gauge-invariant effective low-rank matrix B @ A as the actual
    # parameterized compensation direction. Products are materialized one layer
    # at a time to keep peak memory bounded.
    effective_pair_accumulators = defaultdict(
        lambda: {"diff2": 0.0, "a2": 0.0, "b2": 0.0, "dot": 0.0, "numel": 0, "keys": 0}
    )
    effective_keys = 0
    for a_key, b_key, fam in effective_lora_specs(states, labels):
        products = {
            label: torch.matmul(states[label][b_key].float(), states[label][a_key].float())
            for label in labels
        }
        reg = region(a_key)
        synthetic_key = a_key.rsplit(".loraA", 1)[0] + f".{fam}.weight"
        effective_keys += 1
        for label_a, label_b in itertools.combinations(labels, 2):
            metrics = pair_metrics(products[label_a], products[label_b])
            detail_rows.append(
                {
                    "checkpoint_a": label_a,
                    "checkpoint_b": label_b,
                    "key": synthetic_key,
                    "family": fam,
                    "region": reg,
                    **metrics,
                }
            )
            for group in ((fam, "all"), (fam, reg)):
                acc = effective_pair_accumulators[(label_a, label_b, *group)]
                acc["diff2"] += metrics["difference_l2"] ** 2
                acc["a2"] += metrics["a_l2"] ** 2
                acc["b2"] += metrics["b_l2"] ** 2
                acc["dot"] += float(
                    (products[label_a].reshape(-1) * products[label_b].reshape(-1)).sum()
                )
                acc["numel"] += metrics["numel"]
                acc["keys"] += 1

        values = [products[label].reshape(-1) for label in labels]
        mean = torch.stack(values, dim=0).mean(dim=0)
        deviation2 = sum(float((value - mean).square().sum()) for value in values)
        energy2 = sum(float(value.square().sum()) for value in values)
        numel = int(values[0].numel())
        variance_detail_rows.append(
            {
                "key": synthetic_key,
                "family": fam,
                "region": reg,
                "num_checkpoints": len(values),
                "numel": numel,
                "population_mean_element_variance": deviation2 / max(len(values) * numel, 1),
                "mean_element_second_moment": energy2 / max(len(values) * numel, 1),
                "relative_rms_std": math.sqrt(deviation2 / max(energy2, 1e-30)),
            }
        )
        for group in ((fam, "all"), (fam, reg)):
            acc = variance_accumulators[group]
            acc["deviation2"] += deviation2
            acc["energy2"] += energy2
            acc["numel"] += numel
            acc["keys"] += 1
        del products, values, mean

    append_aggregate_rows(aggregate_rows, effective_pair_accumulators, pair_labels=True)

    variance_aggregate_rows = []
    for (fam, reg), acc in sorted(variance_accumulators.items()):
        denominator = max(len(labels) * acc["numel"], 1)
        variance_aggregate_rows.append(
            {
                "family": fam,
                "region": reg,
                "num_checkpoints": len(labels),
                "tensor_keys": acc["keys"],
                "numel_per_checkpoint": acc["numel"],
                "population_mean_element_variance": acc["deviation2"] / denominator,
                "mean_element_second_moment": acc["energy2"] / denominator,
                "relative_rms_std": math.sqrt(
                    acc["deviation2"] / max(acc["energy2"], 1e-30)
                ),
            }
        )

    def write_csv(path, rows):
        if not rows:
            return
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    write_csv(output / "pairwise_tensor_details.csv", detail_rows)
    write_csv(output / "pairwise_aggregate.csv", aggregate_rows)
    write_csv(output / "checkpoint_variance_tensor_details.csv", variance_detail_rows)
    write_csv(output / "checkpoint_variance_aggregate.csv", variance_aggregate_rows)
    summary = {
        "checkpoints": dict(zip(labels, args.checkpoints)),
        "tensor_counts": {label: len(state) for label, state in states.items()},
        "effective_lora_matrix_count": effective_keys,
        "all_values_finite": all(
            math.isfinite(float(row[field]))
            for row in aggregate_rows
            for field in ("symmetric_relative_l2", "cosine")
        ),
        "multi_checkpoint_variance_all_values_finite": all(
            math.isfinite(float(row[field]))
            for row in variance_aggregate_rows
            for field in (
                "population_mean_element_variance",
                "mean_element_second_moment",
                "relative_rms_std",
            )
        ),
        "variance_definition": (
            "relative_rms_std = sqrt(sum_i ||theta_i - mean(theta)||_2^2 / "
            "sum_i ||theta_i||_2^2), using population variance across calibration checkpoints"
        ),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
