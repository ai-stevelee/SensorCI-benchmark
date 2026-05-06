"""Composite score functions per draft_paper_v1.tex §7.3."""

from __future__ import annotations


def composite_harmonic_mean(s_p1: float, s_p2: float, s_p3_ci: float, s_p4: float) -> float:
    """HM = 4 / sum(1/S_Pi). Penalizes any axis near 0."""
    eps = 1e-9
    vals = [max(s_p1, eps), max(s_p2, eps), max(s_p3_ci, eps), max(s_p4, eps)]
    if any(v <= eps * 10 for v in vals):
        # If any axis is essentially 0, HM tends to 0
        return 0.0
    return 4.0 / sum(1.0 / v for v in vals)


def composite_worst_axis(s_p1: float, s_p2: float, s_p3_ci: float, s_p4: float) -> float:
    """WA = min over the 4 axes."""
    return min(s_p1, s_p2, s_p3_ci, s_p4)


def composite_worst_role(per_role_sci: dict[str, float]) -> float:
    """WR = min over roles of S_P3-CI(role)."""
    if not per_role_sci:
        return 0.0
    return min(per_role_sci.values())
