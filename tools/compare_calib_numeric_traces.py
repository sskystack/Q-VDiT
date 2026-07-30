#!/usr/bin/env python3
import argparse
import json
import math


def load(path):
    records = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def nested(record, path):
    value = record
    for key in path.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


METRICS = {
    "activation_rmse": "activation_quantization.relative_rmse",
    "activation_saturation": "activation_quantization.saturation_ratio",
    "weight_rmse": "error_decomposition.weight_only_relative_rmse",
    "combined_rmse": "error_decomposition.combined_quant_relative_rmse",
    "runtime_compute_rmse": "error_decomposition.runtime_compute_relative_rmse",
    "weight_cast_rmse": "effective_weight_runtime_cast.relative_rmse",
    "weight_cast_zero_ratio": "effective_weight_runtime_cast.nonzero_to_zero_ratio",
    "lora_out_cast_zero_ratio": "lora_out_runtime_cast.nonzero_to_zero_ratio",
    "actual_output_abs_max": "actual_output.abs_max",
}


def index(records):
    result = {}
    for record in records:
        if record.get("kind") != "quant_layer":
            continue
        key = (
            record.get("phase"),
            tuple(record.get("timesteps") or []),
            record.get("module"),
        )
        result[key] = record
    return result


def fmt(value):
    if value is None:
        return "-"
    if not math.isfinite(float(value)):
        return "nonfinite"
    return f"{float(value):.6g}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("fp16_trace")
    parser.add_argument("bf16_trace")
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    fp16 = index(load(args.fp16_trace))
    bf16 = index(load(args.bf16_trace))
    common = sorted(set(fp16) & set(bf16))
    print(f"paired quant-layer records: {len(common)}")
    for label, path in METRICS.items():
        rows = []
        for key in common:
            f_value = nested(fp16[key], path)
            b_value = nested(bf16[key], path)
            if f_value is None or b_value is None:
                continue
            delta = float(f_value) - float(b_value)
            ratio = float(f_value) / max(abs(float(b_value)), 1.0e-12)
            rows.append((abs(delta), ratio, delta, f_value, b_value, key))
        rows.sort(reverse=True)
        print(f"\n[{label}] paired={len(rows)}")
        for _, ratio, delta, f_value, b_value, key in rows[: args.top]:
            phase, timesteps, module = key
            print(
                f"FP16={fmt(f_value)} BF16={fmt(b_value)} "
                f"delta={fmt(delta)} ratio={fmt(ratio)} "
                f"phase={phase} t={list(timesteps)} module={module}"
            )


if __name__ == "__main__":
    main()
