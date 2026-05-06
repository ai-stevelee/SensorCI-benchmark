"""LLM-as-Judge robustness analysis — addresses Reviewer Q1 (methodology).

Demonstrates the protocol with synthetic ratings (until full Phase 3 LLM-judge
pipeline is wired). Computes:
  - Fleiss' kappa across simulated 5-judge ratings
  - Leave-one-out Kendall's tau
  - Pairwise Cohen's kappa between judges

This script will be re-run on real LLM-judge outputs in Phase 3 (P3.5-P3.6).

Usage:
  python scripts/14_judge_robustness.py --simulate  # synthetic demo
"""

import argparse
import sys
import logging
import json
from pathlib import Path
import random

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")


def simulate_judge_outputs(n_items: int = 100, n_judges: int = 5,
                            agreement_strength: float = 0.7, seed: int = 42):
    """Simulate ratings from K judges with controllable agreement.

    Higher agreement_strength → judges more correlated.
    """
    rng = random.Random(seed)
    categories = ["S", "N", "P"]  # Share / Not-share / Partial
    # Ground truth label per item
    truth = [rng.choice(categories) for _ in range(n_items)]
    # Each judge agrees with truth with prob = agreement_strength
    judges = {}
    for j_idx in range(n_judges):
        labels = []
        for t in truth:
            if rng.random() < agreement_strength:
                labels.append(t)
            else:
                # Wrong: pick a different label
                labels.append(rng.choice([c for c in categories if c != t]))
        judges[f"judge_{j_idx}"] = labels
    return judges, truth


def simulate_score_outputs(n_items: int = 100, n_judges: int = 5,
                            corr_strength: float = 0.6, seed: int = 42):
    """Simulate continuous Likert scores from K judges (1-5 scale)."""
    import numpy as np
    rng = np.random.default_rng(seed)
    base = rng.uniform(1, 5, size=n_items)
    judges = {}
    for j_idx in range(n_judges):
        noise = rng.normal(0, 1.0 - corr_strength, size=n_items)
        scores = np.clip(base + noise, 1, 5).tolist()
        judges[f"judge_{j_idx}"] = scores
    return judges


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--simulate", action="store_true",
                    help="Run on simulated judge outputs (default; real-judge mode in Phase 3)")
    ap.add_argument("--n_items", type=int, default=100)
    ap.add_argument("--n_judges", type=int, default=5)
    ap.add_argument("--agreement", type=float, default=0.7)
    ap.add_argument("--input_json", default=None,
                    help="Path to JSON with real judge outputs (overrides --simulate)")
    args = ap.parse_args()

    from sensorci.judge.agreement import (
        fleiss_kappa,
        leave_one_out_kendall,
        cross_judge_agreement,
    )

    if args.input_json:
        data = json.loads(Path(args.input_json).read_text())
        # Expected schema: {"categorical": {judge_name: [labels...]}, "continuous": {judge_name: [scores...]}}
        cat = data.get("categorical", {})
        cont = data.get("continuous", {})
    elif args.simulate:
        cat, _truth = simulate_judge_outputs(args.n_items, args.n_judges, args.agreement)
        cont = simulate_score_outputs(args.n_items, args.n_judges, args.agreement)
    else:
        ap.error("Must pass --simulate or --input_json")
        return 1

    print("=== LLM-as-Judge Robustness Report ===\n")

    # 1. Fleiss' kappa on categorical
    if cat:
        # Build per-item rating list
        judges = list(cat.keys())
        n = len(cat[judges[0]])
        ratings_per_item = [[cat[j][i] for j in judges] for i in range(n)]
        kappa = fleiss_kappa(ratings_per_item)
        print(f"Fleiss' kappa across {len(judges)} judges on {n} items: {kappa:.3f}")
        print(f"  (>= 0.61 substantial; >= 0.41 moderate; < 0.41 weak)\n")

    # 2. Pairwise Cohen's kappa
    if cat:
        pw = cross_judge_agreement(cat)
        print(f"Pairwise Cohen's kappa matrix (5 judges):")
        names = list(pw.keys())
        print(f"  {'':<12}", *(f"{n:<10}" for n in names))
        for i, j_a in enumerate(names):
            row = [f"{pw[j_a][j_b]:.3f}" if i != idx else "  -" for idx, j_b in enumerate(names)]
            print(f"  {j_a:<12}", *(f"{r:<10}" for r in row))
        print()

    # 3. Leave-one-out Kendall's tau on continuous scores
    if cont:
        loo = leave_one_out_kendall(cont)
        print(f"Leave-one-out Kendall's tau (continuous scores):")
        for j, tau in loo.items():
            interp = "stable" if tau > 0.85 else ("influential" if tau < 0.7 else "moderate")
            print(f"  remove {j:<12}: tau = {tau:.3f}  [{interp}]")
        print(f"  (closer to 1.0 = removing this judge has little effect on ranking)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
