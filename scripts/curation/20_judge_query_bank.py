"""Run full LLM-as-Judge 5-stage pipeline to generate cured query bank for a dataset.

Stages:
  1. Generation (K LLMs propose N queries each)
  2. Voting (each candidate scored by other LLMs)
  3. Consensus filter (mean score >= 4.0, std <= 0.8)
  4. Numerical sanity (execute on real signals)
  5. Difficulty stratification (sample to balanced bank)

Usage:
  python scripts/20_judge_query_bank.py --dataset tep --n_per_generator 30
"""

import argparse
import sys
import os
import json
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
    ap.add_argument("--n_per_generator", type=int, default=20)
    ap.add_argument("--generators", nargs="+",
                    default=["gpt-4o-mini", "claude-haiku-4-5", "gemini-3-flash"])
    ap.add_argument("--out_dir", default="./query_bank")
    args = ap.parse_args()

    from sensorci.datasets import load_dataset
    from sensorci.judge import (
        generate_queries, vote_on_queries,
        consensus_filter, numerical_sanity_check, stratify_by_difficulty,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading {args.dataset}...")
    ds = load_dataset(args.dataset)

    # Build sample signals as dict {channel_name -> 1D array}
    sample_signals = []
    for p in ds.personas[:8]:
        for s in p.segments[:2]:
            sig_dict = {ch: s.signal[i] for i, ch in enumerate(s.channels)}
            sample_signals.append(sig_dict)

    # Stage 1
    logger.info("Stage 1: Generation...")
    candidates = generate_queries(ds, generators=args.generators,
                                   n_per_generator=args.n_per_generator)
    logger.info(f"  {len(candidates)} candidates from {len(args.generators)} generators")

    # Stage 2
    logger.info("Stage 2: Voting...")
    voted = vote_on_queries(candidates, judges=args.generators)

    # Stage 3
    logger.info("Stage 3: Consensus filtering...")
    filtered = consensus_filter(voted, min_score=4.0, max_std=0.8)

    # Stage 4
    logger.info("Stage 4: Numerical sanity check...")
    sane = numerical_sanity_check(filtered, sample_signals)

    # Stage 5
    logger.info("Stage 5: Difficulty stratification...")
    bank = stratify_by_difficulty(sane,
                                   target_per_difficulty={"easy": 15, "medium": 25, "hard": 10})

    # Write outputs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bank_path = out_dir / f"{args.dataset}_query_bank_{timestamp}.json"
    bank_path.write_text(json.dumps({
        "dataset": args.dataset,
        "version": timestamp,
        "n_candidates_initial": len(candidates),
        "n_after_voting": len(voted),
        "n_after_consensus": len(filtered),
        "n_after_sanity": len(sane),
        "n_final": len(bank),
        "queries": bank,
    }, indent=2))
    logger.info(f"\nQuery bank saved to {bank_path} ({len(bank)} queries)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
