"""Generate paper figures + LaTeX tables from aggregated results.

Usage:
  python scripts/41_make_paper_artifacts.py --input results/aggregated_summary.json
"""

import argparse
import sys
import os
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="results/aggregated_summary.json")
    ap.add_argument("--out_dir", default="./paper_artifacts_out")
    args = ap.parse_args()

    from sensorci.viz import save_all_figures
    from sensorci.paper_artifacts import (
        make_headline_table, make_per_role_table,
        make_composite_table, write_latex_table,
    )

    out_dir = Path(args.out_dir)
    fig_dir = out_dir / "figures"
    tab_dir = out_dir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    in_path = Path(args.input)
    if not in_path.exists():
        logger.error(f"Input not found: {in_path}")
        return 1
    summary = json.loads(in_path.read_text())

    cross = summary.get("cross_dataset_per_defense", {})
    if not cross:
        logger.error("No cross-dataset summary in input")
        return 1

    # 1) Pareto frontier figure
    defense_scores_for_plot = {
        d: {k: sc.get(k, 0.0) for k in ["S_P1", "S_P2", "S_P3-CI", "S_P4"]}
        for d, sc in cross.items()
    }
    paths = save_all_figures(
        defense_scores=defense_scores_for_plot,
        per_role_scores=None,  # populate from t3 results in future
        t2_curves=None,
        t4_curves=None,
        out_dir=fig_dir,
    )
    logger.info(f"Figures: {list(paths.keys())} -> {fig_dir}")

    # 2) Composite table (LaTeX)
    composite_for_table = {
        d: {**sc, "Rank": rank + 1}
        for rank, (d, sc) in enumerate(
            sorted(cross.items(), key=lambda x: -x[1]["HM"])
        )
    }
    composite_tex = make_composite_table(composite_for_table)
    write_latex_table(composite_tex, tab_dir / "table_composite.tex")
    logger.info(f"Composite table -> {tab_dir / 'table_composite.tex'}")

    # 3) Per-role table (placeholder if no role-stratified data yet)
    # TODO: extract from t3 results when available

    print(f"\nArtifacts written to {out_dir}/")
    print("  figures/   - Pareto frontier, radar, heatmaps")
    print("  tables/    - LaTeX tables")
    return 0


if __name__ == "__main__":
    sys.exit(main())
