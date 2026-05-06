"""Pilot script: run A0 frontier-LLM attack on TEP × {defenses}.

Outputs results/pilot_a0_{timestamp}.jsonl with per-cell AttackResult.

Usage:
  python scripts/10_pilot_a0.py --dataset tep --defenses d1_raw d5_affine --n_segments 20
"""

import argparse
import sys
import os
import logging
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv(override=False)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def _env_models() -> list[str]:
    """Build default model list from .env AZURE_OPENAI_MODEL / CLAUDE_MODEL / GEMINI_MODEL."""
    models = []
    for var in ("AZURE_OPENAI_MODEL", "CLAUDE_MODEL", "GEMINI_MODEL"):
        val = os.environ.get(var, "").strip()
        if val:
            models += [m.strip() for m in val.split(",") if m.strip()]
    return models or ["gpt-5.4", "claude-sonnet-4-5", "gemini-2.5-flash"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="tep", help="dataset name (default: tep)")
    ap.add_argument("--defenses", nargs="+", default=["d1_raw", "d5_affine"],
                    help="defense names")
    ap.add_argument("--models", nargs="+",
                    default=None,
                    help="frontier LLM names (default: reads AZURE_OPENAI_MODEL/CLAUDE_MODEL/GEMINI_MODEL from .env)")
    ap.add_argument("--encodings", nargs="+", default=["raw_stats"],
                    help="signal encodings (raw_stats / sax / compact_numeric)")
    ap.add_argument("--n_segments", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_workers", type=int, default=100)
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()
    if args.models is None:
        args.models = _env_models()
        logger.info(f"Using models from .env: {args.models}")

    from sensorci.datasets import load_dataset
    from sensorci.defenses import get_defense
    from sensorci.attacks import get_attack
    from sensorci.api_pool import get_default_pool
    from sensorci.core.result import save_results

    out_dir = Path(args.out_dir or os.environ.get("SENSORCI_RESULTS_DIR", "./results"))
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading dataset {args.dataset}...")
    ds = load_dataset(args.dataset)
    logger.info(f"  {len(ds.personas)} personas, {sum(len(p.segments) for p in ds.personas)} segments")

    pool = get_default_pool()
    logger.info(f"API pool: {pool}")

    a0 = get_attack(
        "a0_frontier",
        models=args.models,
        encodings=args.encodings,
        max_workers=args.max_workers,
    )

    results = []
    for defense_name in args.defenses:
        logger.info(f"\n=== Defense: {defense_name} ===")
        defense = get_defense(defense_name)
        defense.fit(ds)
        r = a0.run(defense, ds, n_samples=args.n_segments, seed=args.seed, pool=pool)
        results.append(r)

        logger.info(f"  Adv (max over LLM,attr): {r.metric_value:.4f}")
        logger.info(f"  Max cell: {r.extra.get('max_cell_label')}")
        # Print per-model per-attribute table
        for model, scores in r.extra["per_model_scores"].items():
            print(f"\n  {model}:")
            for attr, s in scores.items():
                print(f"    {attr:30s} acc={s['acc']:.3f} prior={s['prior']:.3f} "
                      f"adv={s['adv']:.3f} parsed={s['parsed_rate']:.0%}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"pilot_a0_{timestamp}.jsonl"
    save_results(results, out_path)
    logger.info(f"\nResults saved to {out_path}")
    logger.info(f"Total API cost: ${pool.total_cost_usd:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
