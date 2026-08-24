import torch

from qdiff.mtd_feature_profiler import _safe_name
from qdiff.mtd_adaptive_diagnostics import (
    DiagnosticConfig,
    analyze_adaptive_correspondence,
    summarize_proxy,
)


def shifted_video(size=8, channels=16, shift=2):
    torch.manual_seed(7)
    first = torch.randn(1, channels, size, size)
    second = torch.zeros_like(first)
    second[..., shift:] = first[..., :-shift]
    return torch.stack([first, second], dim=2)


def test_wide_teacher_search_detects_matches_outside_fixed_3x3():
    teacher = shifted_video()
    student = teacher.clone()
    result = analyze_adaptive_correspondence(
        teacher,
        student,
        DiagnosticConfig(search_radius=2, topk=9, temperature=0.03),
    )
    assert result["teacher_top1_outside_fixed_3x3"].mean() > 0.4
    assert result["teacher_mass_outside_fixed_3x3"].mean() > 0.25
    assert (
        result["schemes"]["teacher_topk"]["teacher_mass_covered"].mean()
        > result["schemes"]["fixed_3x3"]["teacher_mass_covered"].mean()
    )


def test_identical_teacher_student_has_zero_scheme_errors():
    teacher = shifted_video()
    result = analyze_adaptive_correspondence(
        teacher,
        teacher.clone(),
        DiagnosticConfig(search_radius=2, topk=9, temperature=0.07),
    )
    for metrics in result["schemes"].values():
        assert metrics["local_kl"].abs().max() < 1.0e-6
        assert metrics["motion_error"].abs().max() < 1.0e-6
        assert metrics["displacement_error"].abs().max() < 1.0e-6


def test_coarse_centered_window_is_not_truncated_by_coarse_search_bank():
    teacher = shifted_video(size=12)
    result = analyze_adaptive_correspondence(
        teacher,
        teacher.clone(),
        DiagnosticConfig(search_radius=2, topk=9, temperature=0.03),
    )
    fixed_count = result["schemes"]["fixed_3x3"]["candidate_count"]
    centred_count = result["schemes"]["teacher_centered_3x3"]["candidate_count"]
    # Away from actual image boundaries, both schemes must carry all nine
    # candidates.  A previous implementation incorrectly clipped centred
    # windows to the coarse 5x5 bank and averaged only six to seven candidates.
    assert (centred_count == 9).sum() >= (fixed_count == 9).sum() * 0.6


def test_oracle_proxy_captures_more_loss_than_its_area_fraction():
    need = torch.arange(1, 101, dtype=torch.float32)
    summary = summarize_proxy(need, need, fractions=(0.25, 0.5))
    assert summary["spearman"] > 0.99
    assert summary["top_25_lift"] > 1.0
    assert summary["top_50_lift"] > 1.0


def test_invalid_shape_is_rejected():
    teacher = torch.randn(1, 4, 8, 8)
    try:
        analyze_adaptive_correspondence(teacher, teacher)
    except ValueError as error:
        assert "[batch, channels, frames, height, width]" in str(error)
    else:
        raise AssertionError("invalid feature shape was accepted")


def test_feature_capture_safe_name_bounds_long_prompt_component():
    prompt = "A very long prompt " * 100
    first = _safe_name(prompt)
    second = _safe_name(prompt)
    assert first == second
    assert len(first) <= 120
    assert first[-11] == "_"
    assert _safe_name("short prompt") == "short_prompt"
