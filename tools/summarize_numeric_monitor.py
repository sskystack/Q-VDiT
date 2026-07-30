#!/usr/bin/env python3
import argparse
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="numeric_monitor.jsonl")
    parser.add_argument("--top", type=int, default=12)
    args = parser.parse_args()

    iterations = []
    parameters = []
    with open(args.path, "r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("kind") == "iteration":
                iterations.append(record)
            elif record.get("kind") == "parameter":
                parameters.append(record)

    print(f"iteration_records={len(iterations)} parameter_records={len(parameters)}")
    for record in iterations[-10:]:
        categories = record.get("trainable_categories", {})
        category_text = ", ".join(
            f"{name}:zero_grad={stats.get('gradient_zero_ratio')},"
            f"grad_l2={stats.get('gradient_l2_norm')},"
            f"nonfinite={stats.get('gradient_nonfinite_count')}"
            for name, stats in categories.items()
        )
        residual = record.get("residual", {})
        print(
            f"iter={record['iteration']} loss={record.get('loss')} "
            f"scale={record.get('grad_scaler_scale')} "
            f"residual_abs_max={residual.get('abs_max')} {category_text}"
        )

    def score(record):
        grad = record.get("gradient") or {}
        parameter = record.get("parameter") or {}
        return (
            grad.get("nonfinite_count", 0) > 0,
            grad.get("zero_ratio", 0.0),
            parameter.get("below_fp16_min_subnormal_ratio", 0.0),
        )

    print("\nTop suspicious parameter snapshots:")
    for record in sorted(parameters, key=score, reverse=True)[:args.top]:
        grad = record.get("gradient") or {}
        parameter = record.get("parameter") or {}
        print(
            f"iter={record['iteration']} {record['name']} "
            f"dtype={parameter.get('dtype')} "
            f"grad_zero={grad.get('zero_ratio')} "
            f"grad_nonfinite={grad.get('nonfinite_count')} "
            f"grad_abs_max={grad.get('abs_max')} "
            f"param_below_fp16={parameter.get('below_fp16_min_subnormal_ratio')} "
            f"param_abs_max={parameter.get('abs_max')}"
        )


if __name__ == "__main__":
    main()
