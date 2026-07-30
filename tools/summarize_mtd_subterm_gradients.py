#!/usr/bin/env python3
import argparse
import json
import statistics
from pathlib import Path


def describe(values):
    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.input.read_text().splitlines()
        if line.strip()
    ]
    if not rows:
        raise RuntimeError(f"No probe records found in {args.input}")

    summary = {
        "iterations": [row["iteration"] for row in rows],
        "aggregate_mtd": {
            "ratio_to_rec": describe([
                row["gradient"]["mtd_to_rec_ratio"] for row in rows
            ]),
            "cosine_with_rec": describe([
                row["gradient"]["cosine"] for row in rows
            ]),
        },
        "subterms": {},
    }
    for name in ("local", "motion", "global"):
        summary["subterms"][name] = {
            "ratio_to_rec": describe([
                row["subterms"][name]["ratio_to_rec"] for row in rows
            ]),
            "cosine_with_rec": describe([
                row["subterms"][name]["cosine_with_rec"] for row in rows
            ]),
            "loss": describe([
                row["subterms"][name]["loss"] for row in rows
            ]),
        }

    json_path = args.output_prefix.with_suffix(".json")
    text_path = args.output_prefix.with_suffix(".txt")
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    lines = [
        "MTD subterm gradient summary",
        f"iterations: {summary['iterations']}",
        (
            "aggregate MTD: median ratio={:.6f}, median cosine={:.6f}"
        ).format(
            summary["aggregate_mtd"]["ratio_to_rec"]["median"],
            summary["aggregate_mtd"]["cosine_with_rec"]["median"],
        ),
    ]
    for name in ("local", "motion", "global"):
        item = summary["subterms"][name]
        lines.append(
            "{}: median ratio={:.6f}, median cosine={:.6f}, median loss={:.6g}".format(
                name,
                item["ratio_to_rec"]["median"],
                item["cosine_with_rec"]["median"],
                item["loss"]["median"],
            )
        )
    text_path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
