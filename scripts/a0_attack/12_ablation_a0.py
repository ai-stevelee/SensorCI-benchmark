"""A0 Ablation script — addresses Reviewer Q2.

Sweeps:
  - encodings: raw_stats, sax, compact_numeric (and combinations)
  - prompt variants: v0_zeroshot, v1_cot, v2_minimal, v3_blinded
  - LLM tracks: cheap (pilot), proprietary (frontier), open (Llama+Mistral+Qwen)

Output: results/ablation_a0_<timestamp>.jsonl with one row per (encoding, prompt, track, defense, dataset, seed) cell.

Usage:
  # Quick (cheap models, all encodings, all prompts, 1 defense):
  python scripts/12_ablation_a0.py --quick

  # Full (proprietary + open tracks):
  python scripts/12_ablation_a0.py --tracks proprietary open --n_segments 50
"""

import argparse
import sys
import os
import logging
import itertools
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
    ap.add_argument("--encodings", nargs="+",
                    default=["raw_stats", "sax", "compact_numeric"])
    ap.add_argument("--prompt_variants", nargs="+",
                    default=["v0_zeroshot", "v1_cot", "v2_minimal", "v3_blinded"])
    ap.add_argument("--tracks", nargs="+", default=["cheap"],
                    help="cheap | proprietary | open (multi-select)")
    ap.add_argument("--n_segments", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_workers", type=int, default=16)
    ap.add_argument("--quick", action="store_true",
                    help="Quick mode: 5 segments, cheap track only, 1 defense")
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    if args.quick:
        args.n_segments = 5
        args.tracks = ["cheap"]
        args.defenses = ["d1_raw"]

    from sensorci.datasets import load_dataset
    from sensorci.defenses import get_defense
    from sensorci.attacks import get_attack
    from sensorci.api_pool import get_default_pool
    from sensorci.core.result import save_results

    out_dir = Path(args.out_dir or os.environ.get("SENSORCI_RESULTS_DIR", "./results"))
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading dataset {args.dataset}...")
    ds = load_dataset(args.dataset)
    pool = get_default_pool()

    # Pre-fit defenses (each only fitted once)
    fitted_defenses = {}
    for d_name in args.defenses:
        d = get_defense(d_name)
        d.fit(ds)
        fitted_defenses[d_name] = d

    cells = list(itertools.product(args.encodings, args.prompt_variants, args.tracks, args.defenses))
    logger.info(f"Sweep: {len(cells)} cells = "
                f"{len(args.encodings)} enc × {len(args.prompt_variants)} prompt × "
                f"{len(args.tracks)} track × {len(args.defenses)} defense")

    results = []
    for enc, prompt, track, d_name in cells:
        logger.info(f"\n=== enc={enc} prompt={prompt} track={track} defense={d_name} ===")
        a0 = get_attack(
            "a0_frontier",
            encodings=[enc],
            prompt_variant=prompt,
            track=track,
            max_workers=args.max_workers,
        )
        defense = fitted_defenses[d_name]
        try:
            r = a0.run(defense, ds, n_samples=args.n_segments, seed=args.seed, pool=pool)
        except Exception as e:
            logger.error(f"  cell failed: {type(e).__name__}: {e}")
            continue
        # Annotate with ablation params
        r.extra["ablation"] = {
            "encoding": enc, "prompt_variant": prompt,
            "track": track, "defense": d_name,
        }
        results.append(r)
        logger.info(f"  Adv (max LLM,attr): {r.metric_value:.4f}  "
                    f"max_cell={r.extra.get('max_cell_label')}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"ablation_a0_{timestamp}.jsonl"
    save_results(results, out_path)

    # Summary table by encoding × prompt
    print(f"\n=== Summary table ===")
    print(f"{'encoding':<18} {'prompt':<14} {'track':<12} {'defense':<12} {'Adv':>8}")
    print("-" * 68)
    for r in sorted(results, key=lambda x: x.extra["ablation"]["encoding"]):
        a = r.extra["ablation"]
        print(f"{a['encoding']:<18} {a['prompt_variant']:<14} {a['track']:<12} "
              f"{a['defense']:<12} {r.metric_value:>8.4f}")

    logger.info(f"\nResults saved to {out_path}")
    logger.info(f"Total API cost: ${pool.total_cost_usd:.3f}")


if __name__ == "__main__":
    sys.exit(main() or 0)
