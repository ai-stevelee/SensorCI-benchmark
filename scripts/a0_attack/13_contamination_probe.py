"""Contamination probe — addresses Reviewer Q4.

Runs A0 with two prompt regimes:
  (a) Original: full dataset name + channel labels + class labels (current default)
  (b) Counterfactual: dataset/channels/classes anonymized via stable aliases
                      (sensorci.judge.contamination.counterfactual_relabel)

For each (LLM, attribute), the contamination delta = orig_acc - blinded_acc:
  - delta > 0.15: HIGH contamination — LLM relies heavily on prior memorization
  - 0.05 < delta < 0.15: MODERATE — some prior knowledge effect
  - delta <= 0.05: LOW — leakage attributable to signal content

Output: results/contamination_<timestamp>.jsonl + summary table.

Usage:
  python scripts/13_contamination_probe.py --dataset tep --defenses d1_raw --models gpt-4o-mini
"""

import argparse
import sys
import os
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="tep")
    ap.add_argument("--defenses", nargs="+", default=["d1_raw"])
    ap.add_argument("--models", nargs="+", default=["gpt-4o-mini"])
    ap.add_argument("--n_segments", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_workers", type=int, default=16)
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    from sensorci.datasets import load_dataset
    from sensorci.defenses import get_defense
    from sensorci.attacks import get_attack
    from sensorci.api_pool import get_default_pool
    from sensorci.judge.contamination import counterfactual_relabel, contamination_delta
    from sensorci.core.result import save_results

    out_dir = Path(args.out_dir or os.environ.get("SENSORCI_RESULTS_DIR", "./results"))
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading dataset {args.dataset}...")
    ds_orig = load_dataset(args.dataset)
    logger.info("Constructing counterfactual (relabeled) dataset...")
    ds_cf, relabel_map = counterfactual_relabel(ds_orig)
    logger.info(f"  original name: {ds_orig.name} -> alias: {ds_cf.name}")
    logger.info(f"  channel aliases: {len(relabel_map.channel_aliases)}")
    logger.info(f"  class aliases: {len(relabel_map.class_aliases)}")

    pool = get_default_pool()
    a0_orig = get_attack("a0_frontier", models=args.models,
                         encodings=["raw_stats"], prompt_variant="v0_zeroshot",
                         max_workers=args.max_workers)
    a0_blind = get_attack("a0_frontier", models=args.models,
                          encodings=["raw_stats"], prompt_variant="v3_blinded",
                          max_workers=args.max_workers)

    all_results = []
    for d_name in args.defenses:
        defense = get_defense(d_name)
        defense.fit(ds_orig)

        logger.info(f"\n=== Defense: {d_name} ===")
        logger.info("Running A0 on ORIGINAL dataset (with metadata)...")
        r_orig = a0_orig.run(defense, ds_orig, n_samples=args.n_segments,
                             seed=args.seed, pool=pool)
        all_results.append(r_orig)

        # Refit defense on counterfactual (signal data is identical, but new metadata)
        defense2 = get_defense(d_name)
        defense2.fit(ds_cf)
        logger.info("Running A0 on COUNTERFACTUAL dataset (metadata-blinded)...")
        r_cf = a0_blind.run(defense2, ds_cf, n_samples=args.n_segments,
                            seed=args.seed, pool=pool)
        all_results.append(r_cf)

        # Compute per-attribute delta for first model
        model = args.models[0]
        orig_scores = r_orig.extra["per_model_scores"].get(model, {})
        cf_scores = r_cf.extra["per_model_scores"].get(model, {})
        orig_acc = {a: s["acc"] for a, s in orig_scores.items()}
        cf_acc = {a: s["acc"] for a, s in cf_scores.items()}
        delta_analysis = contamination_delta(orig_acc, cf_acc)

        print(f"\nContamination probe — model={model}, defense={d_name}:")
        print(f"  {'attribute':<30} {'orig_acc':>8} {'blind_acc':>9} {'delta':>7} {'flag':>10}")
        print("  " + "-" * 70)
        for attr, info in delta_analysis.items():
            print(f"  {attr:<30} {info['original_acc']:>8.3f} {info['blinded_acc']:>9.3f} "
                  f"{info['delta']:>+7.3f} {info['contamination_flag']:>10}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"contamination_{timestamp}.jsonl"
    save_results(all_results, out_path)
    logger.info(f"\nResults saved to {out_path}")
    logger.info(f"Total API cost: ${pool.total_cost_usd:.3f}")


if __name__ == "__main__":
    sys.exit(main() or 0)
