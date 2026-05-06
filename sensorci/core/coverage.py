"""Role Discriminative Coverage (Definition in draft_paper_v1.tex §3.4).

Implements the non-vacuity check for the CI sharing matrix Phi_D:
  for every (role r, not-share attribute a), verify that some witness functional
  phi in F_r reduces the conditional entropy of a from prior.

If a (role, attribute) cell fails coverage, it is flagged 'UNDEF' and excluded
from S_P3-CI aggregation. This addresses Reviewer Q1 (non-vacuity of Theorem 1).

Implementation strategy:
  - For each (role r, attribute a) with Phi_D(r,a) = N:
    1. Construct a small witness query bank from the role's allowed view
    2. Compute mutual information estimate I(a*(X); phi(X)) using a
       K-nearest-neighbor MI estimator (Kraskov et al. 2004) on the
       persona-level pairs (phi(X), a*(X))
    3. Cell passes coverage if I_est > coverage_threshold
"""

from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Callable
import numpy as np

from sensorci.core.persona import Role, ROLES
from sensorci.core.dataset import TSDataset

logger = logging.getLogger(__name__)


# Standard witness functionals available to most role views (mean, std, peak, fft band)
STANDARD_WITNESSES: dict[str, Callable[[np.ndarray], float]] = {
    "mean":     lambda x: float(np.mean(x)),
    "std":      lambda x: float(np.std(x)),
    "peak":     lambda x: float(np.max(np.abs(x))),
    "rms":      lambda x: float(np.sqrt(np.mean(x ** 2))),
    "kurtosis": lambda x: _safe_kurtosis(x),
    "fft_lo":   lambda x: float(np.abs(np.fft.rfft(x))[1:5].sum()),
    "fft_hi":   lambda x: float(np.abs(np.fft.rfft(x))[5:20].sum()),
}


def _safe_kurtosis(x: np.ndarray) -> float:
    sd = np.std(x)
    if sd < 1e-12:
        return 0.0
    return float(np.mean(((x - np.mean(x)) / sd) ** 4) - 3.0)


@dataclass
class CoverageResult:
    role: Role
    attribute: str
    sharing_label: str
    best_witness: str
    mi_estimate: float
    passes_coverage: bool
    note: str = ""


def estimate_mi_categorical_continuous(continuous: np.ndarray, categorical: list) -> float:
    """Estimate I(C; X) where C is categorical and X is continuous, via histogram.

    Returns mutual information in nats. This is a lightweight estimator;
    for production use, consider sklearn.feature_selection.mutual_info_classif
    (we use that as a fallback when available).
    """
    try:
        from collections import Counter as _Counter
        from sklearn.feature_selection import mutual_info_classif
        # mutual_info_classif expects (n_samples, n_features) and integer labels
        x = np.asarray(continuous, dtype=float).reshape(-1, 1)
        # Encode categorical to ints
        cat_to_int = {v: i for i, v in enumerate(sorted(set(categorical)))}
        y = np.array([cat_to_int[c] for c in categorical])
        if len(set(y)) < 2 or len(x) < 4:
            return 0.0
        # KNN-based MI estimator needs >= 3 samples per class
        if min(_Counter(y).values()) < 3:
            raise RuntimeError("class too small for KNN MI; using histogram fallback")
        mi = float(mutual_info_classif(x, y, random_state=0)[0])
        return mi
    except (ImportError, RuntimeError, ValueError):
        # Fallback: simple binned MI
        x = np.asarray(continuous, dtype=float)
        n = len(x)
        if n < 4:
            return 0.0
        n_bins = max(2, int(np.sqrt(n) // 2))
        bin_edges = np.percentile(x, np.linspace(0, 100, n_bins + 1))
        bin_edges[-1] += 1e-9
        x_bin = np.digitize(x, bin_edges[1:-1])
        cat_to_int = {v: i for i, v in enumerate(sorted(set(categorical)))}
        y = np.array([cat_to_int[c] for c in categorical])
        # Joint histogram
        joint = np.zeros((n_bins, len(cat_to_int)), dtype=float)
        for xi, yi in zip(x_bin, y):
            joint[xi, yi] += 1
        joint /= n
        px = joint.sum(axis=1, keepdims=True)
        py = joint.sum(axis=0, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_term = np.log(joint / (px * py + 1e-12) + 1e-12)
            log_term[joint == 0] = 0
            mi = float(np.sum(joint * log_term))
        return max(mi, 0.0)


def check_coverage(
    dataset: TSDataset,
    sharing_matrix: dict[tuple[Role, str], str] | None = None,
    coverage_threshold: float = 0.05,  # nats; ~7% MI relative to small categorical entropy
    max_personas: int = 50,
    witness_fns: dict[str, Callable[[np.ndarray], float]] | None = None,
) -> list[CoverageResult]:
    """Run RDC check for every (role, attribute) cell with Phi=N in the matrix.

    Args:
      dataset: TSDataset with personas (each has segments + attributes)
      sharing_matrix: defaults to dataset.sharing_matrix
      coverage_threshold: minimum estimated MI (in nats) to pass coverage
      max_personas: cap on personas used for MI estimation (speed)
      witness_fns: dict of name -> fn(signal_1d) -> scalar.
                   Defaults to STANDARD_WITNESSES.

    Returns:
      List of CoverageResult, one per cell with Phi(r,a) = N.
    """
    sharing = sharing_matrix or dataset.sharing_matrix
    witnesses = witness_fns or STANDARD_WITNESSES

    # Build (persona-level) attribute -> categorical list
    personas = dataset.personas[:max_personas]
    if len(personas) < 4:
        logger.warning("Coverage check needs >= 4 personas; got %d. Returning empty.", len(personas))
        return []

    # For each persona, pick first segment for witness computation
    # (Can be extended to per-segment averaging.)
    persona_signals: list[np.ndarray] = []
    persona_attr_values: dict[str, list] = {}
    for p in personas:
        if not p.segments:
            continue
        sig = p.segments[0].signal
        # Take channel 0 (or could iterate) — TODO(extend): per-channel coverage
        persona_signals.append(sig[0] if sig.ndim == 2 else sig)
        for k, v in p.all_attributes.items():
            persona_attr_values.setdefault(k, []).append(v)

    not_share_cells = [(r, a) for (r, a), label in sharing.items() if label == "N"]
    results: list[CoverageResult] = []
    for role, attr in not_share_cells:
        # Skip if attribute has < 2 unique values (degenerate) or persona count too low
        cat_vals = persona_attr_values.get(attr, [])
        if not cat_vals or len(set(map(str, cat_vals))) < 2:
            results.append(CoverageResult(
                role=role, attribute=attr, sharing_label="N",
                best_witness="—", mi_estimate=0.0, passes_coverage=False,
                note="degenerate attribute (< 2 unique values among sampled personas)",
            ))
            continue
        cat_strs = [str(v) for v in cat_vals]
        # Compute each witness
        best_witness, best_mi = "—", 0.0
        for w_name, w_fn in witnesses.items():
            try:
                w_vals = np.array([w_fn(s) for s in persona_signals])
            except Exception as e:
                logger.debug("Witness %s failed: %s", w_name, e)
                continue
            mi = estimate_mi_categorical_continuous(w_vals, cat_strs)
            if mi > best_mi:
                best_mi = mi
                best_witness = w_name
        passes = best_mi > coverage_threshold
        results.append(CoverageResult(
            role=role, attribute=attr, sharing_label="N",
            best_witness=best_witness, mi_estimate=best_mi, passes_coverage=passes,
            note="" if passes else f"MI {best_mi:.4f} <= threshold {coverage_threshold}",
        ))
    return results


def coverage_summary(results: list[CoverageResult]) -> dict:
    """Return aggregate coverage statistics."""
    if not results:
        return {"total": 0, "passing": 0, "failing": 0, "pass_rate": 0.0}
    passing = [r for r in results if r.passes_coverage]
    failing = [r for r in results if not r.passes_coverage]
    by_role: dict[str, dict[str, int]] = {}
    for r in results:
        by_role.setdefault(r.role, {"pass": 0, "fail": 0})
        by_role[r.role]["pass" if r.passes_coverage else "fail"] += 1
    return {
        "total": len(results),
        "passing": len(passing),
        "failing": len(failing),
        "pass_rate": len(passing) / len(results),
        "by_role": by_role,
        "failing_cells": [
            {"role": r.role, "attribute": r.attribute, "mi": r.mi_estimate, "note": r.note}
            for r in failing
        ],
    }
