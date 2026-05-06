"""LLM-as-Judge Stage 5: difficulty stratification."""

from __future__ import annotations
import numpy as np


def stratify_by_difficulty(
    candidates: list[dict],
    target_per_difficulty: dict[str, int] | None = None,
    rng_seed: int = 42,
) -> list[dict]:
    """Stage 5: assign difficulty stratum tags and sample to a balanced bank.

    target_per_difficulty: e.g., {"easy": 15, "medium": 25, "hard": 10}
    """
    target_per_difficulty = target_per_difficulty or {"easy": 15, "medium": 25, "hard": 10}
    rng = np.random.default_rng(rng_seed)

    # Without actually running each query through an LLM (too costly here),
    # use mean_score as a proxy: high mean → easy (LLM judges agree), low → hard
    for c in candidates:
        ms = c.get("mean_score", 3.0)
        if ms >= 4.5:
            c["difficulty_stratum"] = "easy"
        elif ms >= 4.0:
            c["difficulty_stratum"] = "medium"
        else:
            c["difficulty_stratum"] = "hard"

    out = []
    for stratum, target in target_per_difficulty.items():
        pool = [c for c in candidates if c.get("difficulty_stratum") == stratum]
        rng.shuffle(pool)
        out.extend(pool[:target])
    return out
