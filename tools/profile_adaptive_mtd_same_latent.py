#!/usr/bin/env python3
"""Diagnose adaptive-MTD checkpoints on identical saved latents."""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from mmengine.config import Config
from mmengine.runner import set_random_seed

from opensora.utils.misc import to_torch_dtype
from qdiff.mtd_adaptive_diagnostics import DiagnosticConfig, analyze_adaptive_correspondence
from tools.profile_stage_numeric_mechanism import build_quant_model, error_metrics, prepare_conditioning, set_quant_mode


VIEWS = ("conditional", "guided")
SCHEMES = ("fixed_3x3", "teacher_centered_3x3")
DIAGNOSTIC_METRICS = ("local_kl", "motion_error")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--baseline-calib-config", required=True)
    parser.add_argument("--candidate-calib-config", required=True)
    parser.add_argument("--baseline-ckpt", required=True)
    parser.add_argument("--candidate-ckpt", required=True)
    parser.add_argument("--baseline-features", type=Path, required=True)
    parser.add_argument("--candidate-features", type=Path, required=True)
    parser.add_argument("--text-embeds", required=True)
    parser.add_argument("--time-mp-config-weight", required=True)
    parser.add_argument("--time-mp-config-act", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--transport-size", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.07)
    return parser.parse_args()


def extract_tensor(output):
    return output["x"] if isinstance(output, dict) else output


@torch.no_grad()
def forward_views(qnn, latent, model_timestep, conditioning, cfg_scale):
    y = conditioning["y"]
    mask = conditioning["mask"]
    y_shape = y.shape
    y = y.reshape([2, y_shape[0] // 2] + list(y_shape[1:]))
    y_cond, y_uncond = y.unbind(0)
    timestep = torch.tensor([model_timestep, model_timestep], device=latent.device, dtype=torch.long)
    t_cond, t_uncond = timestep.reshape(2, -1).unbind(0)
    cond = extract_tensor(qnn.forward(latent, t_cond, y_cond, mask=mask))
    uncond = extract_tensor(qnn.forward(latent, t_uncond, y_uncond, mask=mask))
    cond_eps, cond_rest = cond[:, :3], cond[:, 3:]
    uncond_eps = uncond[:, :3]
    guided = torch.cat([uncond_eps + cfg_scale * (cond_eps - uncond_eps), cond_rest], dim=1)
    return {"conditional": cond, "guided": guided}


def pooled_first_four(tensor, size):
    feature = tensor[:, :4].float()
    pooled = F.adaptive_avg_pool2d(
        feature.permute(0, 2, 1, 3, 4).reshape(
            feature.shape[0] * feature.shape[2], feature.shape[1], feature.shape[3], feature.shape[4]
        ),
        (size, size),
    )
    return pooled.reshape(
        feature.shape[0], feature.shape[2], feature.shape[1], size, size
    ).permute(0, 2, 1, 3, 4)


def diagnostic_metrics(teacher, student, config):
    result = analyze_adaptive_correspondence(teacher, student, config)
    return {
        scheme: {
            metric: float(result["schemes"][scheme][metric].float().mean())
            for metric in DIAGNOSTIC_METRICS
        }
        for scheme in SCHEMES
    }


def load_capture_specs(label, root):
    specs = []
    for path in sorted(root.rglob("step_*.pt")):
        payload = torch.load(path, map_location="cpu")
        specs.append(
            {
                "trajectory_source": label,
                "path": str(path),
                "prompt_index": int(payload["prompt_index"]),
                "sampling_progress": int(payload["sampling_progress"]),
                "model_timestep": int(payload["model_timestep"]),
                "latent": payload["latent"],
            }
        )
    if not specs:
        raise FileNotFoundError(f"no captures below {root}")
    return specs


def relative_improvement(baseline, candidate):
    return (baseline - candidate) / baseline if baseline != 0 else math.nan


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["trajectory_source"], row["view"])].append(row)
    result = {}
    all_gates = True
    for (source, view), values in sorted(groups.items()):
        target = result.setdefault(source, {})[view] = {"captures": len(values)}
        for scheme in SCHEMES:
            target[scheme] = {}
            for metric in DIAGNOSTIC_METRICS:
                baseline = sum(row["baseline_quant"][scheme][metric] for row in values) / len(values)
                candidate = sum(row["candidate_quant"][scheme][metric] for row in values) / len(values)
                gain = relative_improvement(baseline, candidate)
                target[scheme][metric] = {
                    "baseline": baseline,
                    "candidate": candidate,
                    "relative_improvement": gain,
                }
                if scheme == "teacher_centered_3x3":
                    all_gates &= gain > 0
        for metric in ("relative_l2", "nmse", "rmse", "max_abs_error"):
            baseline = sum(row["baseline_output_error"][metric] for row in values) / len(values)
            candidate = sum(row["candidate_output_error"][metric] for row in values) / len(values)
            target.setdefault("output_error", {})[metric] = {
                "baseline": baseline,
                "candidate": candidate,
                "relative_improvement": relative_improvement(baseline, candidate),
            }
        target["candidate_fp_vs_baseline_fp"] = {
            metric: sum(row["candidate_fp_vs_baseline_fp"][metric] for row in values) / len(values)
            for metric in ("relative_l2", "nmse", "rmse", "max_abs_error")
        }
    result["candidate_improves_centered_kl_and_motion_on_all_source_views"] = all_gates
    return result


def main():
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cfg = Config.fromfile(args.config)
    cfg.multi_resolution = cfg.get("multi_resolution", False)
    dtype = to_torch_dtype(cfg.dtype)
    device = torch.device("cuda")
    set_random_seed(args.seed)
    torch.set_grad_enabled(False)

    common = {
        "time_mp_config_weight": args.time_mp_config_weight,
        "time_mp_config_act": args.time_mp_config_act,
    }
    baseline_args = SimpleNamespace(
        calib_config=args.baseline_calib_config,
        quant_ckpt=args.baseline_ckpt,
        **common,
    )
    candidate_args = SimpleNamespace(
        calib_config=args.candidate_calib_config,
        quant_ckpt=args.candidate_ckpt,
        **common,
    )
    _, _, baseline_qnn = build_quant_model(baseline_args, cfg, device, dtype)
    _, _, candidate_qnn = build_quant_model(candidate_args, cfg, device, dtype)
    diagnostic_config = DiagnosticConfig(
        search_radius=2,
        topk=9,
        temperature=args.temperature,
        seed=args.seed,
    )
    specs = load_capture_specs("baseline_trajectory", args.baseline_features)
    specs += load_capture_specs("candidate_trajectory", args.candidate_features)
    conditioning_cache = {}
    rows = []

    for spec in specs:
        prompt_index = spec["prompt_index"]
        conditioning = conditioning_cache.setdefault(
            prompt_index,
            prepare_conditioning(args.text_embeds, prompt_index, device, dtype),
        )
        latent = spec["latent"].to(device=device, dtype=dtype)

        set_quant_mode(baseline_qnn, False, False)
        baseline_fp = forward_views(
            baseline_qnn, latent, spec["model_timestep"], conditioning, float(cfg.scheduler.cfg_scale)
        )
        set_quant_mode(baseline_qnn, True, True)
        baseline_quant = forward_views(
            baseline_qnn, latent, spec["model_timestep"], conditioning, float(cfg.scheduler.cfg_scale)
        )
        set_quant_mode(candidate_qnn, False, False)
        candidate_fp = forward_views(
            candidate_qnn, latent, spec["model_timestep"], conditioning, float(cfg.scheduler.cfg_scale)
        )
        set_quant_mode(candidate_qnn, True, True)
        candidate_quant = forward_views(
            candidate_qnn, latent, spec["model_timestep"], conditioning, float(cfg.scheduler.cfg_scale)
        )

        for view in VIEWS:
            teacher = pooled_first_four(baseline_fp[view], args.transport_size)
            baseline_student = pooled_first_four(baseline_quant[view], args.transport_size)
            candidate_student = pooled_first_four(candidate_quant[view], args.transport_size)
            rows.append(
                {
                    "trajectory_source": spec["trajectory_source"],
                    "prompt_index": prompt_index,
                    "sampling_progress": spec["sampling_progress"],
                    "model_timestep": spec["model_timestep"],
                    "view": view,
                    "baseline_output_error": error_metrics(
                        baseline_quant[view][:, :3], baseline_fp[view][:, :3]
                    ),
                    "candidate_output_error": error_metrics(
                        candidate_quant[view][:, :3], baseline_fp[view][:, :3]
                    ),
                    "candidate_fp_vs_baseline_fp": error_metrics(
                        candidate_fp[view][:, :3], baseline_fp[view][:, :3]
                    ),
                    "baseline_quant": diagnostic_metrics(teacher, baseline_student, diagnostic_config),
                    "candidate_quant": diagnostic_metrics(teacher, candidate_student, diagnostic_config),
                }
            )

    summary = summarize(rows)
    summary["metadata"] = {
        "baseline_checkpoint": str(Path(args.baseline_ckpt).resolve()),
        "candidate_checkpoint": str(Path(args.candidate_ckpt).resolve()),
        "baseline_features": str(args.baseline_features.resolve()),
        "candidate_features": str(args.candidate_features.resolve()),
        "captures": len(specs),
        "rows": len(rows),
        "common_input_design": True,
        "common_teacher": "baseline checkpoint with quantization disabled",
        "temperature": args.temperature,
        "transport_size": args.transport_size,
        "no_raft": True,
    }
    finite = all(
        math.isfinite(value)
        for row in rows
        for section in ("baseline_output_error", "candidate_output_error", "candidate_fp_vs_baseline_fp")
        for value in row[section].values()
    )
    summary["all_output_metrics_finite"] = finite
    with (output / "per_capture.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
