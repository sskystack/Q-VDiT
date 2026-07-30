#!/usr/bin/env python3
"""Print a compact, stable summary of a paired gradient probe JSON file."""

import argparse
import json


def ratio(count, total):
    return float(count) / max(int(total), 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("probe")
    args = parser.parse_args()

    with open(args.probe, "r", encoding="utf-8") as handle:
        probe = json.load(handle)

    print("scale", probe.get("scale"))
    print("ordinary_loss", probe.get("ordinary_loss"))
    print("scaled_probe_loss", probe.get("scaled_probe_loss"))
    print("output_relative_difference", probe.get("output_relative_difference"))
    print("CATEGORIES")
    for category, values in sorted(probe.get("categories", {}).items()):
        numel = int(values.get("numel", 0))
        print(
            category,
            "numel", numel,
            "ordinary_zero_ratio", ratio(values.get("ordinary_zero_count", 0), numel),
            "scaled_zero_ratio", ratio(values.get("scaled_recovered_zero_count", 0), numel),
            "recovered_from_zero", values.get("recovered_from_zero_count", 0),
            "lost_after_scaling", values.get("lost_after_scaling_count", 0),
            "ordinary_norm", values.get("ordinary_norm"),
            "scaled_recovered_norm", values.get("scaled_recovered_norm"),
            "relative_difference", values.get("relative_difference"),
        )

    print("TOP_RECOVERED_PARAMETERS")
    rows = sorted(
        probe.get("top_parameters", []),
        key=lambda row: int(row.get("recovered_from_zero_count", 0)),
        reverse=True,
    )
    for row in rows[:20]:
        print(
            row.get("name"),
            row.get("category"),
            "numel", row.get("numel"),
            "ordinary_zero", row.get("ordinary_zero_count"),
            "scaled_zero", row.get("scaled_recovered_zero_count"),
            "recovered", row.get("recovered_from_zero_count"),
            "ordinary_norm", row.get("ordinary_norm"),
            "scaled_recovered_norm", row.get("scaled_recovered_norm"),
            "relative_difference", row.get("relative_difference"),
            "cosine", row.get("cosine"),
        )


if __name__ == "__main__":
    main()
