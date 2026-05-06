"""Pilot script: run A6 algebraic probe on TEP × {defenses}.

This is the cheap baseline (no LLM calls; pure CPU). Useful to confirm
defenses are wired correctly.

Usage:
  python scripts/11_pilot_algebra.py --dataset tep --defenses d1_raw d5_affine
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
    ap.add_argument("--defenses", nargs="+", default=["d1_raw", "d5_affine"])
    ap.add_argument("--n_samples", type=int, default=50)
    ap.add_argument("--n_queries", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    from sensorci.datasets import load_dataset
    from sensorci.defenses import get_defense
    from sensorci.attacks import get_attack
    from sensorci.core.result import save_results

    out_dir = Path(args.out_dir or os.environ.get("SENSORCI_RESULTS_DIR", "./results"))
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading dataset {args.dataset}...")
    ds = load_dataset(args.dataset)
    a6 = get_attack("a6_algebra", n_stub_queries=args.n_queries)

    results = []
    print(f"\n{'Defense':<20} {'S_P1':>8} {'median_RE':>12}")
    print("-" * 42)
    for defense_name in args.defenses:
        defense = get_defense(defense_name)
        defense.fit(ds)
        r = a6.run(defense, ds, n_samples=args.n_samples, seed=args.seed)
        results.append(r)
        print(f"{defense_name:<20} {r.metric_value:>8.4f} {r.subscores['median_re']:>12.4f}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"pilot_algebra_{timestamp}.jsonl"
    save_results(results, out_path)
    logger.info(f"\nResults saved to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
