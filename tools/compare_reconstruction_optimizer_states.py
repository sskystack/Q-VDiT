#!/usr/bin/env python3
"""Compare paired AdamW states from Q-VDiT reconstruction checkpoints."""

import argparse
import math

import torch


def mapped_state(checkpoint):
    optimizer = checkpoint["optimizer"]
    result = {}
    for names, group in zip(checkpoint["optimizer_parameter_names"], optimizer["param_groups"]):
        lr = float(group["lr"])
        beta1, beta2 = group["betas"]
        eps = float(group["eps"])
        weight_decay = float(group["weight_decay"])
        for name, parameter_id in zip(names, group["params"]):
            state = optimizer["state"].get(parameter_id, {})
            parameter = checkpoint["trainable_parameters"][name].float()
            if not state:
                continue
            step = float(state["step"])
            exp_avg = state["exp_avg"].float()
            exp_avg_sq = state["exp_avg_sq"].float()
            m_hat = exp_avg / (1.0 - beta1 ** step)
            v_hat = exp_avg_sq / (1.0 - beta2 ** step)
            direction = m_hat / (torch.sqrt(v_hat) + eps) + weight_decay * parameter
            update = lr * direction
            result[name] = {
                "parameter_norm": float(torch.linalg.vector_norm(parameter)),
                "exp_avg_norm": float(torch.linalg.vector_norm(exp_avg)),
                "exp_avg_sq_norm": float(torch.linalg.vector_norm(exp_avg_sq)),
                "update_norm": float(torch.linalg.vector_norm(update)),
                "relative_update": float(torch.linalg.vector_norm(update) /
                                         (torch.linalg.vector_norm(parameter) + 1.0e-12)),
                "lr": lr,
            }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("fp16")
    parser.add_argument("bf16")
    parser.add_argument("--top", type=int, default=30)
    args = parser.parse_args()
    fp = mapped_state(torch.load(args.fp16, map_location="cpu"))
    bf = mapped_state(torch.load(args.bf16, map_location="cpu"))
    rows = []
    for name in sorted(fp.keys() & bf.keys()):
        if ".lora" not in name and ".delta" not in name:
            continue
        rows.append({
            "name": name,
            "parameter_ratio": fp[name]["parameter_norm"] / max(bf[name]["parameter_norm"], 1.0e-12),
            "momentum_ratio": fp[name]["exp_avg_norm"] / max(bf[name]["exp_avg_norm"], 1.0e-12),
            "variance_ratio": fp[name]["exp_avg_sq_norm"] / max(bf[name]["exp_avg_sq_norm"], 1.0e-12),
            "update_ratio": fp[name]["update_norm"] / max(bf[name]["update_norm"], 1.0e-12),
            "relative_update_fp": fp[name]["relative_update"],
            "relative_update_bf": bf[name]["relative_update"],
            "lr": fp[name]["lr"],
        })
    for key in ("update_ratio", "momentum_ratio", "parameter_ratio"):
        print("TOP", key.upper())
        for row in sorted(rows, key=lambda item: item[key], reverse=True)[:args.top]:
            print(row["name"],
                  "param_ratio=%.4f" % row["parameter_ratio"],
                  "m_ratio=%.4f" % row["momentum_ratio"],
                  "v_ratio=%.4f" % row["variance_ratio"],
                  "update_ratio=%.4f" % row["update_ratio"],
                  "rel_update_fp=%.6g" % row["relative_update_fp"],
                  "rel_update_bf=%.6g" % row["relative_update_bf"])
    print("TRACKED")
    tracked = ("blocks.26.mlp.fc2", "blocks.27.attn.q", "blocks.27.attn.k",
               "blocks.26.attn.q", "blocks.26.mlp.fc1")
    for row in rows:
        if any(token in row["name"] for token in tracked):
            print(row["name"],
                  "param_ratio=%.4f" % row["parameter_ratio"],
                  "m_ratio=%.4f" % row["momentum_ratio"],
                  "v_ratio=%.4f" % row["variance_ratio"],
                  "update_ratio=%.4f" % row["update_ratio"],
                  "rel_update_fp=%.6g" % row["relative_update_fp"],
                  "rel_update_bf=%.6g" % row["relative_update_bf"])


if __name__ == "__main__":
    main()
