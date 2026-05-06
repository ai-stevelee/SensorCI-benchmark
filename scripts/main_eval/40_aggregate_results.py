"""Aggregate results from main evaluation: per-defense composite scores + statistics.

Usage:
  python scripts/40_aggregate_results.py --input results/main_eval_*.jsonl
"""

import argparse
import sys
import os
import json
import glob
import logging
from pathlib import Path
import numpy as np

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="results/main_eval_*.jsonl",
                    help="Glob pattern for input JSONL files")
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    from sensorci.metrics import (
        composite_harmonic_mean, composite_worst_axis,
        aggregate_seeds, friedman_test,
    )
    from sensorci.core.result import load_results

    out_dir = Path(args.out_dir or os.environ.get("SENSORCI_RESULTS_DIR", "./results"))
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(glob.glob(args.input))
    if not files:
        logger.error(f"No files matching {args.input}")
        return 1

    all_results = []
    for f in files:
        all_results.extend(load_results(Path(f)))
    logger.info(f"Loaded {len(all_results)} results from {len(files)} files")

    # Map attack name -> property axis
    ATTACK_TO_AXIS = {
        "a0_frontier": "P3-CI",  # also touches P4
        "a1_distinguish": "P2",
        "a2_linear_recon": "P4",
        "a3_neural_recon": "P4",
        "a4_ci_linkage": "P3-CI",
        "a5_stat_tests": "P2",
        "a6_algebra": "P1",
        "a7_mia_ts": "P4",
    }

    # For each (defense, dataset), collect per-axis scores from attacks
    # NOTE: attack metric_value semantics differ:
    #   A1, A4-CI, A7 → adv (lower better) → S = 1 - adv
    #   A2, A3 → NRMSE (higher better) → S = NRMSE (already in [0,1])
    #   A5 → MMD_norm (lower better) → S = 1 - MMD
    #   A6 → S_P1 (already a score)
    def to_score(r):
        a = r["attack"]
        v = r["metric_value"]
        if a in ("a1_distinguish", "a4_ci_linkage", "a7_mia_ts", "a0_frontier"):
            return max(0.0, 1.0 - v)
        if a in ("a2_linear_recon", "a3_neural_recon"):
            return float(v)
        if a == "a5_stat_tests":
            return max(0.0, 1.0 - v)
        if a == "a6_algebra":
            return float(v)
        return 0.5

    # Group by (defense, dataset)
    grouped = {}
    for r in all_results:
        key = (r["defense"], r["dataset"])
        grouped.setdefault(key, []).append(r)

    composite = {}
    for (defense, dataset), rs in grouped.items():
        per_axis_scores = {"P1": [], "P2": [], "P3-CI": [], "P4": []}
        for r in rs:
            ax = ATTACK_TO_AXIS.get(r["attack"])
            if ax:
                per_axis_scores[ax].append(to_score(r))
        # Mean per axis (across attacks of same property)
        mean_axes = {}
        for ax in per_axis_scores:
            if per_axis_scores[ax]:
                mean_axes[f"S_{ax}"] = float(np.mean(per_axis_scores[ax]))
            else:
                mean_axes[f"S_{ax}"] = 0.5  # missing → neutral
        # Composite metrics
        s1 = mean_axes["S_P1"]
        s2 = mean_axes["S_P2"]
        s3 = mean_axes["S_P3-CI"]
        s4 = mean_axes["S_P4"]
        hm = composite_harmonic_mean(s1, s2, s3, s4)
        wa = composite_worst_axis(s1, s2, s3, s4)
        composite[f"{defense}_{dataset}"] = {
            "defense": defense, "dataset": dataset,
            **mean_axes, "HM": hm, "WA": wa,
            "n_attack_results": len(rs),
        }

    # Cross-dataset aggregation per defense
    cross_dataset = {}
    by_defense = {}
    for k, v in composite.items():
        by_defense.setdefault(v["defense"], []).append(v)
    for d, lst in by_defense.items():
        cross_dataset[d] = {
            "S_P1": float(np.mean([x["S_P1"] for x in lst])),
            "S_P2": float(np.mean([x["S_P2"] for x in lst])),
            "S_P3-CI": float(np.mean([x["S_P3-CI"] for x in lst])),
            "S_P4": float(np.mean([x["S_P4"] for x in lst])),
            "HM": float(np.mean([x["HM"] for x in lst])),
            "WA": float(np.mean([x["WA"] for x in lst])),
            "n_datasets": len(lst),
        }

    # Friedman across defenses on HM
    hm_per_defense = {d: [x["HM"] for x in lst] for d, lst in by_defense.items()}
    friedman = friedman_test(hm_per_defense)

    summary = {
        "n_results": len(all_results),
        "n_defense_dataset_cells": len(composite),
        "per_defense_dataset": composite,
        "cross_dataset_per_defense": cross_dataset,
        "friedman_hm": friedman,
    }
    out_path = out_dir / "aggregated_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))

    # Print summary
    print(f"\n{'Defense':<22} {'S_P1':>6} {'S_P2':>6} {'S_P3-CI':>8} {'S_P4':>6} {'HM':>6} {'WA':>6}")
    print("-" * 65)
    for d, sc in sorted(cross_dataset.items(), key=lambda x: -x[1]["HM"]):
        print(f"{d:<22} {sc['S_P1']:>6.3f} {sc['S_P2']:>6.3f} {sc['S_P3-CI']:>8.3f} "
              f"{sc['S_P4']:>6.3f} {sc['HM']:>6.3f} {sc['WA']:>6.3f}")
    print(f"\nFriedman test on HM: chi2={friedman['chi2']:.3f}, p={friedman['p']:.4f}")
    print(f"\nSummary saved to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
