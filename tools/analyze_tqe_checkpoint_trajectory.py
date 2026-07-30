#!/usr/bin/env python3
"""Compare paired FP16/BF16 Q-VDiT TQE reconstruction checkpoints."""

import argparse
import csv
import math
import os
import re

import torch
import yaml


def product_norm(a, b):
    # ||B A||_F^2 = tr((B^T B)(A A^T)); only rank-sized matrices are formed.
    gram_b = b.T.float() @ b.float()
    gram_a = a.float() @ a.T.float()
    return math.sqrt(max(float((gram_b * gram_a.T).sum()), 0.0))


def product_inner(a1, b1, a2, b2):
    return float(torch.trace((b1.T.float() @ b2.float()) @ (a2.float() @ a1.T.float())))


def product_relative_difference(a1, b1, a2, b2):
    n1 = product_norm(a1, b1)
    n2 = product_norm(a2, b2)
    inner = product_inner(a1, b1, a2, b2)
    diff = math.sqrt(max(n1 * n1 + n2 * n2 - 2.0 * inner, 0.0))
    return diff / max(n2, 1.0e-12)


def rank1_product_norm(a, b):
    return float(torch.linalg.vector_norm(a.float()) * torch.linalg.vector_norm(b.float()))


def rank1_product_relative_difference(a1, b1, a2, b2):
    n1 = rank1_product_norm(a1, b1)
    n2 = rank1_product_norm(a2, b2)
    inner = float((a1.float().flatten() @ a2.float().flatten()) *
                  (b1.float().flatten() @ b2.float().flatten()))
    diff = math.sqrt(max(n1 * n1 + n2 * n2 - 2.0 * inner, 0.0))
    return diff / max(n2, 1.0e-12)


def quant_state(checkpoint, module):
    state = checkpoint[module + ".weight_quantizer"]
    buffers = state[0]
    return buffers["delta"].float(), buffers["zero_point"].float()


def module_names(checkpoint):
    suffix = ".loraA.weight"
    return sorted(key[:-len(suffix)] for key in checkpoint if key.endswith(suffix))


def block_index(module):
    match = re.match(r"blocks\.(\d+)\.", module)
    return int(match.group(1)) if match else -1


def quantize(weight, delta, zero_point, n_bits):
    levels = 2 ** int(n_bits)
    code = torch.clamp(torch.round(weight / delta) + zero_point, 0, levels - 1)
    return (code - zero_point) * delta, code


def effective_metrics(base_weight, checkpoint, module, n_bits):
    a = checkpoint[module + ".loraA.weight"].float()
    b = checkpoint[module + ".loraB.weight"].float()
    ao = checkpoint[module + ".loraA_out.weight"].float()
    bo = checkpoint[module + ".loraB_out.weight"].float()
    delta, zero = quant_state(checkpoint, module)
    pre = b @ a
    post = bo @ ao
    plain, plain_code = quantize(base_weight, delta, zero, n_bits)
    with_tqe, tqe_code = quantize(base_weight + pre, delta, zero, n_bits)
    effective_weight = with_tqe + post
    correction = effective_weight - plain
    ideal = base_weight - plain
    residual_plain = torch.linalg.vector_norm(ideal)
    residual_tqe = torch.linalg.vector_norm(base_weight - effective_weight)
    correction_norm = torch.linalg.vector_norm(correction)
    alignment = float((correction.flatten() @ ideal.flatten()) /
                      (correction_norm * residual_plain + 1.0e-12))
    return {
        "effective_correction_norm": float(correction_norm),
        "ideal_error_norm": float(residual_plain),
        "correction_to_error": float(correction_norm / (residual_plain + 1.0e-12)),
        "alignment": alignment,
        "residual_ratio": float(residual_tqe / (residual_plain + 1.0e-12)),
        "code_flip_ratio": float((plain_code != tqe_code).float().mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp16-dir", required=True)
    parser.add_argument("--bf16-dir", required=True)
    parser.add_argument("--base-checkpoint")
    parser.add_argument("--bit-config", required=True)
    parser.add_argument("--max-iteration", type=int, default=7000)
    parser.add_argument("--csv")
    parser.add_argument(
        "--effective-csv",
        help=(
            "Optional CSV containing effective-weight correction, alignment, "
            "residual, and code-flip metrics for every quantized module at "
            "the selected checkpoint iterations."
        ),
    )
    args = parser.parse_args()

    with open(args.bit_config, "r", encoding="utf-8") as handle:
        raw_bits = yaml.safe_load(handle)
    bit_config = {key.removeprefix("model."): value for key, value in raw_bits.items()}

    iterations = list(range(500, args.max_iteration + 1, 500))
    rows = []
    for iteration in iterations:
        filename = f"ckpt_iter_{iteration:08d}.pth"
        fp = torch.load(os.path.join(args.fp16_dir, filename), map_location="cpu")
        bf = torch.load(os.path.join(args.bf16_dir, filename), map_location="cpu")
        for module in module_names(fp):
            if module not in bit_config:
                continue
            af, bf_factor = fp[module + ".loraA.weight"], fp[module + ".loraB.weight"]
            ab, bb_factor = bf[module + ".loraA.weight"], bf[module + ".loraB.weight"]
            aof, bof = fp[module + ".loraA_out.weight"], fp[module + ".loraB_out.weight"]
            aob, bob = bf[module + ".loraA_out.weight"], bf[module + ".loraB_out.weight"]
            df, _ = quant_state(fp, module)
            db, _ = quant_state(bf, module)
            pre_fp = product_norm(af, bf_factor)
            pre_bf = product_norm(ab, bb_factor)
            post_fp = rank1_product_norm(aof, bof)
            post_bf = rank1_product_norm(aob, bob)
            rows.append({
                "iteration": iteration,
                "module": module,
                "block": block_index(module),
                "bits": bit_config[module],
                "delta_ratio": float(torch.linalg.vector_norm(df) /
                                     (torch.linalg.vector_norm(db) + 1.0e-12)),
                "delta_relative_difference": float(torch.linalg.vector_norm(df - db) /
                                                    (torch.linalg.vector_norm(db) + 1.0e-12)),
                "lora_b_ratio": float(torch.linalg.vector_norm(bf_factor) /
                                      (torch.linalg.vector_norm(bb_factor) + 1.0e-12)),
                "pre_norm_fp16": pre_fp,
                "pre_norm_bf16": pre_bf,
                "pre_norm_ratio": pre_fp / max(pre_bf, 1.0e-12),
                "pre_relative_difference": product_relative_difference(
                    af, bf_factor, ab, bb_factor),
                "post_norm_fp16": post_fp,
                "post_norm_bf16": post_bf,
                "post_norm_ratio": post_fp / max(post_bf, 1.0e-12),
                "post_relative_difference": rank1_product_relative_difference(
                    aof, bof, aob, bob),
            })
        del fp, bf

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    print("TOP FINAL PRE-COMPENSATION RATIOS")
    final_rows = [row for row in rows if row["iteration"] == iterations[-1]]
    for row in sorted(final_rows, key=lambda item: item["pre_norm_ratio"], reverse=True)[:20]:
        print(row["module"], "ratio=%.4f" % row["pre_norm_ratio"],
              "fp=%.6g" % row["pre_norm_fp16"], "bf=%.6g" % row["pre_norm_bf16"],
              "rel_diff=%.4f" % row["pre_relative_difference"])

    print("TOP FINAL POST-COMPENSATION RATIOS")
    for row in sorted(final_rows, key=lambda item: item["post_norm_ratio"], reverse=True)[:20]:
        print(row["module"], "ratio=%.4f" % row["post_norm_ratio"],
              "fp=%.6g" % row["post_norm_fp16"], "bf=%.6g" % row["post_norm_bf16"],
              "rel_diff=%.4f" % row["post_relative_difference"])

    tracked = {
        "blocks.27.attn.q", "blocks.26.mlp.fc2", "blocks.27.attn.k",
        "blocks.26.attn.q", "blocks.26.mlp.fc1",
    }
    print("TRACKED TRAJECTORIES")
    for row in rows:
        if row["module"] in tracked:
            print(row["iteration"], row["module"],
                  "delta_diff=%.5f" % row["delta_relative_difference"],
                  "pre_ratio=%.4f" % row["pre_norm_ratio"],
                  "post_ratio=%.4f" % row["post_norm_ratio"])

    if not args.base_checkpoint:
        return
    base = torch.load(args.base_checkpoint, map_location="cpu")
    print("EFFECTIVE CORRECTION AT SELECTED ITERATIONS")
    effective_rows = []
    selected_iterations = []
    for iteration in (500, 1500, 3000, args.max_iteration):
        filename = f"ckpt_iter_{iteration:08d}.pth"
        if (
            iteration <= args.max_iteration
            and iteration not in selected_iterations
            and os.path.isfile(os.path.join(args.fp16_dir, filename))
            and os.path.isfile(os.path.join(args.bf16_dir, filename))
        ):
            selected_iterations.append(iteration)
    for iteration in selected_iterations:
        filename = f"ckpt_iter_{iteration:08d}.pth"
        fp = torch.load(os.path.join(args.fp16_dir, filename), map_location="cpu")
        bf = torch.load(os.path.join(args.bf16_dir, filename), map_location="cpu")
        candidates = [row for row in rows if row["iteration"] == iteration]
        candidates = sorted(candidates, key=lambda item: max(item["pre_norm_ratio"], item["post_norm_ratio"]), reverse=True)
        selected = list(dict.fromkeys([row["module"] for row in candidates[:8]] + sorted(tracked)))
        modules_to_compute = (
            [row["module"] for row in candidates]
            if args.effective_csv
            else selected
        )
        metrics = {}
        for module in modules_to_compute:
            if module + ".weight" not in base:
                continue
            weight = base[module + ".weight"].float()
            mf = effective_metrics(weight, fp, module, bit_config[module])
            mb = effective_metrics(weight, bf, module, bit_config[module])
            metrics[module] = (mf, mb)
            effective_rows.append({
                "iteration": iteration,
                "module": module,
                "block": block_index(module),
                "bits": bit_config[module],
                **{"first_" + key: value for key, value in mf.items()},
                **{"second_" + key: value for key, value in mb.items()},
            })
        for module in selected:
            if module not in metrics:
                continue
            mf, mb = metrics[module]
            print(iteration, module,
                  "corr_ratio=%.4f" % (mf["effective_correction_norm"] / max(mb["effective_correction_norm"], 1.0e-12)),
                  "fp_corr/error=%.4f" % mf["correction_to_error"],
                  "bf_corr/error=%.4f" % mb["correction_to_error"],
                  "fp_align=%.4f" % mf["alignment"],
                  "bf_align=%.4f" % mb["alignment"],
                  "fp_residual_ratio=%.4f" % mf["residual_ratio"],
                  "bf_residual_ratio=%.4f" % mb["residual_ratio"],
                  "fp_flip=%.5f" % mf["code_flip_ratio"],
                  "bf_flip=%.5f" % mb["code_flip_ratio"])
        del fp, bf

    if args.effective_csv and effective_rows:
        with open(args.effective_csv, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=effective_rows[0].keys()
            )
            writer.writeheader()
            writer.writerows(effective_rows)


if __name__ == "__main__":
    main()
