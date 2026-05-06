"""Inter-LLM agreement metrics for the LLM-as-Judge protocol.

Addresses Reviewer Q1 (LLM-as-judge validity beyond single-expert):
- Fleiss' kappa: agreement among >=3 raters on categorical labels
- Leave-one-out Kendall's tau: how much does removing one judge change ranking?
- Cross-judge agreement matrix: pairwise per-cell agreement
"""

from __future__ import annotations
from collections import Counter
import logging
import numpy as np

logger = logging.getLogger(__name__)


def fleiss_kappa(ratings: list[list[str]]) -> float:
    """Compute Fleiss' kappa for inter-rater agreement on categorical labels.

    Args:
      ratings: list of N items; each item is a list of K judges' categorical labels.
               All items must have the same number of judges K.

    Returns:
      Kappa in [-1, 1]. Conventional thresholds:
        > 0.81: almost perfect
        0.61-0.80: substantial
        0.41-0.60: moderate
        < 0.40: weak
    """
    if not ratings:
        return 0.0
    K = len(ratings[0])
    if any(len(r) != K for r in ratings):
        raise ValueError("All items must have the same number of judges")
    if K < 2:
        return 1.0

    # Collect category set
    categories = sorted(set(c for r in ratings for c in r))
    cat_to_idx = {c: i for i, c in enumerate(categories)}
    C = len(categories)
    N = len(ratings)
    if C < 2:
        return 1.0

    # Build N x C matrix of counts per item
    M = np.zeros((N, C), dtype=int)
    for i, r in enumerate(ratings):
        for c in r:
            M[i, cat_to_idx[c]] += 1

    # P_i: per-item agreement rate
    P_i = (np.sum(M ** 2, axis=1) - K) / (K * (K - 1))
    P_bar = float(np.mean(P_i))

    # Pe: expected agreement under chance
    p_j = M.sum(axis=0) / (N * K)
    Pe = float(np.sum(p_j ** 2))

    if abs(1 - Pe) < 1e-12:
        return 1.0
    return (P_bar - Pe) / (1 - Pe)


def leave_one_out_kendall(
    judge_scores: dict[str, list[float]],
) -> dict[str, float]:
    """Compute leave-one-out Kendall's tau for ranking stability.

    Args:
      judge_scores: dict judge_name -> list of scores (one per item).
                    All judges must rate the same items in the same order.

    Returns:
      dict judge_name -> Kendall tau between (full mean ranking) and
                         (mean ranking with that judge removed).
      Lower tau means that judge has more influence; tau near 1 means
      protocol is robust to removing that judge.
    """
    from scipy.stats import kendalltau
    judges = list(judge_scores.keys())
    if len(judges) < 3:
        logger.warning("Leave-one-out requires >=3 judges; got %d", len(judges))
        return {}
    scores_array = np.array([judge_scores[j] for j in judges])  # (n_judges, n_items)
    full_mean = scores_array.mean(axis=0)
    out: dict[str, float] = {}
    for i, j in enumerate(judges):
        loo_mean = np.delete(scores_array, i, axis=0).mean(axis=0)
        tau, _ = kendalltau(full_mean, loo_mean)
        out[j] = float(tau if tau == tau else 1.0)  # nan -> 1.0 (tied)
    return out


def cross_judge_agreement(
    judge_labels: dict[str, list[str]],
) -> dict[str, dict[str, float]]:
    """Pairwise per-cell agreement (Cohen's kappa-style) between every pair of judges.

    Args:
      judge_labels: dict judge_name -> list of categorical labels (same length).

    Returns:
      Nested dict: pairwise_agreement[judge_a][judge_b] = Cohen's kappa
    """
    from sklearn.metrics import cohen_kappa_score
    judges = list(judge_labels.keys())
    out: dict[str, dict[str, float]] = {j: {} for j in judges}
    for i, ja in enumerate(judges):
        for jb in judges[i:]:
            if ja == jb:
                k = 1.0
            else:
                try:
                    k = float(cohen_kappa_score(judge_labels[ja], judge_labels[jb]))
                except Exception:
                    k = 0.0
            out[ja][jb] = k
            out[jb][ja] = k
    return out
