"""Role Discriminative Coverage check — addresses Reviewer Q1.

Verifies that for every (role, not-share attribute) cell in Phi_D, there exists
a witness functional in the role's view that has non-trivial mutual information
with the attribute. Cells failing coverage are excluded from S_P3-CI aggregation
(per draft_paper_v1.tex §3.4 Definition 1).

Reports:
  - Total cells / passing / failing
  - Per-role pass rate
  - List of failing cells with MI estimates

Usage:
  python scripts/15_coverage_check.py --dataset tep
"""

import argparse
import sys
import logging
import json
from pathlib import Path

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="tep")
    ap.add_argument("--coverage_threshold", type=float, default=0.05,
                    help="Min MI in nats to pass coverage")
    ap.add_argument("--max_personas", type=int, default=50)
    ap.add_argument("--out", default=None,
                    help="Optional JSON output file with failing cells")
    args = ap.parse_args()

    from sensorci.datasets import load_dataset
    from sensorci.core.coverage import check_coverage, coverage_summary

    print(f"Loading {args.dataset}...")
    ds = load_dataset(args.dataset)
    print(f"  {len(ds.personas)} personas")
    print(f"  {len(ds.sharing_matrix)} (role, attribute) cells in sharing matrix")
    n_not_share = sum(1 for v in ds.sharing_matrix.values() if v == "N")
    print(f"  {n_not_share} cells with label 'N' (not-share)\n")

    print(f"Running RDC check (threshold MI >= {args.coverage_threshold} nats)...")
    results = check_coverage(
        ds,
        coverage_threshold=args.coverage_threshold,
        max_personas=args.max_personas,
    )
    summary = coverage_summary(results)

    print(f"\n=== Coverage Summary ===")
    print(f"Total not-share cells checked: {summary['total']}")
    print(f"Passing coverage:              {summary['passing']} ({summary['pass_rate']:.1%})")
    print(f"Failing coverage:              {summary['failing']}")

    print(f"\n=== Per-role pass rate ===")
    for role, counts in summary.get("by_role", {}).items():
        total = counts["pass"] + counts["fail"]
        rate = counts["pass"] / max(total, 1)
        print(f"  {role:<12}: {counts['pass']:>3} pass / {total:>3} total  ({rate:.0%})")

    failing = summary.get("failing_cells", [])
    if failing:
        print(f"\n=== Failing cells (excluded from S_P3-CI aggregation) ===")
        print(f"  {'role':<12} {'attribute':<25} {'MI':>8} note")
        print("  " + "-" * 70)
        for cell in failing[:20]:  # show first 20
            print(f"  {cell['role']:<12} {cell['attribute']:<25} {cell['mi']:>8.4f}  {cell['note']}")
        if len(failing) > 20:
            print(f"  ... and {len(failing) - 20} more")
    else:
        print(f"\nAll cells pass coverage. CI matrix is non-vacuously enforceable.")

    if args.out:
        out_path = Path(args.out)
        out_path.write_text(json.dumps(summary, indent=2, default=str))
        print(f"\nFull report saved to {out_path}")

    return 0 if summary["failing"] <= int(0.1 * max(summary["total"], 1)) else 1


if __name__ == "__main__":
    sys.exit(main())
