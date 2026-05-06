"""Aggregation utilities: bootstrap CIs, Wilcoxon, Friedman, per-cell mapping."""

from __future__ import annotations
from typing import Any
import numpy as np


def bootstrap_ci(values: list[float], n_resamples: int = 1000,
                 ci_level: float = 0.95, rng_seed: int = 42) -> tuple[float, float]:
    """Bootstrap confidence interval (percentile method)."""
    if not values:
        return (0.0, 0.0)
    rng = np.random.default_rng(rng_seed)
    arr = np.array(values, dtype=float)
    n = len(arr)
    if n < 2:
        return (float(arr[0]), float(arr[0]))
    means = np.empty(n_resamples)
    for i in range(n_resamples):
        means[i] = arr[rng.integers(0, n, size=n)].mean()
    alpha = (1 - ci_level) / 2
    lo = float(np.quantile(means, alpha))
    hi = float(np.quantile(means, 1 - alpha))
    return (lo, hi)


def aggregate_seeds(per_seed_metric: list[float]) -> dict[str, float]:
    """Mean / std / 95% CI over seeds."""
    if not per_seed_metric:
        return {"mean": 0.0, "std": 0.0, "ci_low": 0.0, "ci_high": 0.0, "n": 0}
    arr = np.array(per_seed_metric, dtype=float)
    ci_lo, ci_hi = bootstrap_ci(per_seed_metric)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1) if len(arr) > 1 else 0.0),
        "ci_low": ci_lo,
        "ci_high": ci_hi,
        "n": int(len(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def pairwise_wilcoxon(
    defense_results: dict[str, list[float]],
    fdr_correction: bool = True,
) -> dict[tuple[str, str], dict[str, float]]:
    """Pairwise Wilcoxon signed-rank between defenses with optional BH FDR.

    defense_results: dict defense_name -> list of metric values across cells.
    """
    from scipy.stats import wilcoxon
    names = list(defense_results.keys())
    pairs = []
    pvals = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a = defense_results[names[i]]
            b = defense_results[names[j]]
            if len(a) != len(b) or len(a) < 2:
                continue
            try:
                stat, p = wilcoxon(a, b)
            except Exception:
                stat, p = 0.0, 1.0
            pairs.append((names[i], names[j]))
            pvals.append(float(p))

    # Benjamini-Hochberg FDR
    if fdr_correction and pvals:
        m = len(pvals)
        order = np.argsort(pvals)
        adj = np.empty(m)
        prev = 1.0
        for rank_pos in range(m - 1, -1, -1):
            idx = order[rank_pos]
            adjusted = pvals[idx] * m / (rank_pos + 1)
            adjusted = min(adjusted, prev)
            adj[idx] = adjusted
            prev = adjusted
        pvals_adj = adj.tolist()
    else:
        pvals_adj = pvals

    out = {}
    for (a, b), p_raw, p_adj in zip(pairs, pvals, pvals_adj):
        out[(a, b)] = {"p_raw": p_raw, "p_adj": float(p_adj)}
    return out


def friedman_test(defense_results: dict[str, list[float]]) -> dict[str, float]:
    """Friedman test for repeated measures across defenses."""
    from scipy.stats import friedmanchisquare
    arrays = list(defense_results.values())
    if len(arrays) < 2 or any(len(a) < 2 for a in arrays):
        return {"chi2": 0.0, "p": 1.0, "n_defenses": len(arrays)}
    try:
        # Truncate to common length
        L = min(len(a) for a in arrays)
        truncated = [a[:L] for a in arrays]
        stat, p = friedmanchisquare(*truncated)
        return {"chi2": float(stat), "p": float(p), "n_defenses": len(arrays)}
    except Exception:
        return {"chi2": 0.0, "p": 1.0, "n_defenses": len(arrays)}


def cliffs_delta(a: list[float], b: list[float]) -> float:
    """Cliff's delta effect size in [-1, 1]."""
    if not a or not b:
        return 0.0
    a_arr, b_arr = np.array(a), np.array(b)
    n_a, n_b = len(a_arr), len(b_arr)
    greater = np.sum(a_arr[:, None] > b_arr[None, :])
    less = np.sum(a_arr[:, None] < b_arr[None, :])
    return float((greater - less) / (n_a * n_b))


def collect_per_attack_scores(results: list, dataset: str | None = None) -> dict:
    """Collect AttackResult list -> nested dict {defense: {attack: [vals]}}."""
    out: dict[str, dict[str, list[float]]] = {}
    for r in results:
        if dataset is not None and r.dataset != dataset:
            continue
        out.setdefault(r.defense, {}).setdefault(r.attack, []).append(r.metric_value)
    return out
