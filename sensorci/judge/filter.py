"""LLM-as-Judge Stage 3-4: consensus filtering + numerical sanity."""

from __future__ import annotations
import logging
from typing import Any
import numpy as np

logger = logging.getLogger(__name__)


def consensus_filter(
    voted_candidates: list[dict],
    min_score: float = 4.0,
    max_std: float = 0.8,
) -> list[dict]:
    """Stage 3: keep only candidates with high mean and low std among judges."""
    out = []
    for c in voted_candidates:
        if c.get("mean_score", 0) < min_score:
            continue
        if c.get("score_std", 99) > max_std:
            continue
        out.append(c)
    logger.info(f"Consensus filter: {len(out)} of {len(voted_candidates)} candidates kept")
    return out


def numerical_sanity_check(
    candidates: list[dict],
    sample_signals: list[dict],
) -> list[dict]:
    """Stage 4: execute each query on sample signals; remove queries that
    produce NaN/Inf/constant outputs.

    sample_signals: list of dicts {channel_name -> 1D array} for evaluation.
    """
    if not sample_signals:
        return candidates
    out = []
    for c in candidates:
        formula = c.get("formula_python", "")
        if not formula:
            continue
        results = []
        ok = True
        for sig_dict in sample_signals[:20]:
            try:
                # Provide np in eval namespace
                local = {"np": np, "signal": sig_dict}
                val = eval(formula, {"__builtins__": {}}, local)
                if val is None or np.isnan(val) or np.isinf(val):
                    ok = False
                    break
                results.append(float(val))
            except Exception:
                ok = False
                break
        if not ok:
            continue
        # Check for trivially constant
        if results and (np.std(results) < 1e-9):
            continue
        out.append(c)
    logger.info(f"Numerical sanity: {len(out)} candidates pass")
    return out
