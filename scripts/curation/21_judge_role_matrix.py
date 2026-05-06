"""Curate role × attribute sharing matrix for a dataset via multi-LLM voting.

Usage:
  python scripts/21_judge_role_matrix.py --dataset tep
"""

import argparse
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from collections import Counter

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="tep")
    ap.add_argument("--generators", nargs="+",
                    default=["gpt-4o-mini", "claude-haiku-4-5", "gemini-3-flash"])
    ap.add_argument("--out_dir", default="./query_bank")
    args = ap.parse_args()

    from sensorci.datasets import load_dataset
    from sensorci.judge import generate_sharing_matrix
    from sensorci.core.persona import ROLES

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(args.dataset)
    proposals = generate_sharing_matrix(ds, generators=args.generators)
    logger.info(f"Got {len(proposals)} sharing matrix proposals")

    # Build voted matrix: for each (role, attribute), majority vote across proposals
    attrs = ds.attribute_set()
    voted_matrix: dict[str, str] = {}
    per_cell_votes: dict[str, dict[str, int]] = {}
    for role in ROLES:
        for attr in attrs:
            key = f"{attr}_{role}"
            votes = []
            for prop in proposals:
                m = prop.get("sharing_matrix", {})
                v = m.get(key)
                if v in ("S", "N", "P"):
                    votes.append(v)
            if not votes:
                # Default for unanimous absence: N (conservative)
                voted_matrix[key] = "N" if role != "Operator" else "S"
                per_cell_votes[key] = {}
                continue
            counter = Counter(votes)
            top = counter.most_common(1)[0][0]
            voted_matrix[key] = top
            per_cell_votes[key] = dict(counter)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"{args.dataset}_sharing_matrix_{timestamp}.json"
    out_path.write_text(json.dumps({
        "dataset": args.dataset,
        "version": timestamp,
        "n_proposals": len(proposals),
        "voted_matrix": voted_matrix,
        "per_cell_votes": per_cell_votes,
        "raw_proposals": proposals,
    }, indent=2))
    logger.info(f"Sharing matrix saved to {out_path} ({len(voted_matrix)} cells)")
    # Quick summary
    counter_total = Counter(voted_matrix.values())
    print(f"\nSharing label distribution: {dict(counter_total)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
