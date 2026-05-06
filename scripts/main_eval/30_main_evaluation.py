"""Main evaluation: run the full attack × defense × tier matrix on selected datasets.

Usage:
  python scripts/30_main_evaluation.py --config configs/full_main.yaml
  python scripts/30_main_evaluation.py --quick   # Tiny matrix for verification
"""

import argparse
import sys
import os
import logging
import yaml
import time
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv(override=False)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def _env_models() -> list[str]:
    models = []
    for var in ("AZURE_OPENAI_MODEL", "CLAUDE_MODEL", "GEMINI_MODEL"):
        val = os.environ.get(var, "").strip()
        if val:
            models += [m.strip() for m in val.split(",") if m.strip()]
    return models or ["gpt-5.4", "claude-sonnet-4-5", "gemini-2.5-flash"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None,
                    help="YAML config (defaults to configs/full_main.yaml or --quick)")
    ap.add_argument("--quick", action="store_true",
                    help="Quick mode: 1 dataset, 3 defenses, A1+A2+A6, T1 only")
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--defenses", nargs="+", default=None)
    ap.add_argument("--attacks", nargs="+", default=None)
    ap.add_argument("--models", nargs="+", default=None,
                    help="LLM models for A0 (default: reads from .env)")
    ap.add_argument("--tiers", nargs="+", default=None,
                    help="Subset of T1 T2 T3 T4")
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--n_samples", type=int, default=None)
    ap.add_argument("--max_workers", type=int, default=64,
                    help="Concurrent workers for LLM API calls (default 64)")
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()
    if args.models is None:
        args.models = _env_models()
        logger.info(f"A0 models from .env: {args.models}")

    from sensorci.datasets import load_dataset, LOADER_REGISTRY
    from sensorci.defenses import get_defense, DEFENSE_REGISTRY
    from sensorci.attacks import get_attack
    from sensorci.core.tier_protocol import (
        run_t1_single_segment, run_t2_cross_segment,
        run_t3_multi_recipient, run_t4_longitudinal,
    )
    from sensorci.core.result import save_results

    # Load config
    cfg = {}
    if args.quick:
        cfg = {
            "datasets": ["tep"],
            "defenses": ["d1_raw", "d5_affine", "d6_per_seg_affine"],
            "attacks": ["a1_distinguish", "a2_linear_recon", "a6_algebra"],
            "tiers": ["T1"],
            "seeds": [42],
            "n_samples": 20,
        }
    elif args.config:
        cfg = yaml.safe_load(Path(args.config).read_text())

    datasets = args.datasets or cfg.get("datasets", ["tep"])
    defenses = args.defenses or cfg.get("defenses", ["d1_raw", "d5_affine"])
    attacks = args.attacks or cfg.get("attacks", ["a1_distinguish", "a6_algebra"])
    tiers = args.tiers or cfg.get("tiers", ["T1"])
    seeds = args.seeds or cfg.get("seeds", [42])
    n_samples = args.n_samples or cfg.get("n_samples", 30)

    out_dir = Path(args.out_dir or os.environ.get("SENSORCI_RESULTS_DIR", "./results"))
    out_dir.mkdir(parents=True, exist_ok=True)
    ds_tag = "_".join(datasets) if datasets else "all"
    timestamp = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{ds_tag}"

    logger.info("=== SensorCI Main Evaluation ===")
    logger.info(f"Datasets: {datasets}")
    logger.info(f"Defenses: {defenses}")
    logger.info(f"Attacks: {attacks}")
    logger.info(f"Tiers: {tiers}, Seeds: {seeds}, N samples: {n_samples}")

    all_results = []
    t_start = time.time()

    for ds_name in datasets:
        if ds_name not in LOADER_REGISTRY:
            logger.warning(f"Skip unknown dataset {ds_name}")
            continue
        logger.info(f"\n>>> Loading dataset: {ds_name}")
        ds = load_dataset(ds_name)

        for defense_name in defenses:
            if defense_name not in DEFENSE_REGISTRY:
                logger.warning(f"Skip unknown defense {defense_name}")
                continue
            for seed in seeds:
                logger.info(f"\n--- {ds_name} × {defense_name} (seed={seed}) ---")
                defense = get_defense(defense_name)
                try:
                    defense.fit(ds)
                except Exception as e:
                    logger.warning(f"defense.fit failed: {e}")
                    continue

                # T1
                if "T1" in tiers:
                    attack_objs = []
                    for atk_name in attacks:
                        try:
                            kw = {}
                            if atk_name == "a0_frontier":
                                kw = {"models": args.models, "max_workers": args.max_workers}
                            attack_objs.append(get_attack(atk_name, **kw))
                        except Exception as e:
                            logger.warning(f"Skip attack {atk_name}: {e}")
                    t1_results = run_t1_single_segment(defense, ds, attack_objs,
                                                       n_samples=n_samples, seed=seed)
                    all_results.extend(t1_results)
                    for r in t1_results:
                        logger.info(f"  T1 {r.attack:25s}: {r.metric_name}={r.metric_value:.4f}")

                # T2
                if "T2" in tiers:
                    try:
                        t2 = run_t2_cross_segment(defense, ds, seed=seed)
                        all_results.append(t2)
                        logger.info(f"  T2 cross-segment: AUC max={t2.metric_value:.4f}")
                    except Exception as e:
                        logger.warning(f"T2 failed: {e}")

                # T3
                if "T3" in tiers:
                    try:
                        t3 = run_t3_multi_recipient(defense, ds, n_samples=n_samples, seed=seed)
                        all_results.append(t3)
                        logger.info(f"  T3 collusion expansion: {t3.metric_value:.4f}")
                    except Exception as e:
                        logger.warning(f"T3 failed: {e}")

                # T4
                if "T4" in tiers:
                    try:
                        t4 = run_t4_longitudinal(defense, ds, seed=seed)
                        all_results.append(t4)
                        logger.info(f"  T4 longitudinal max leak: {t4.metric_value:.4f}")
                    except Exception as e:
                        logger.warning(f"T4 failed: {e}")

    elapsed = time.time() - t_start
    out_path = out_dir / f"main_eval_{timestamp}.jsonl"
    save_results(all_results, out_path)
    logger.info(f"\n=== Done in {elapsed:.1f}s ===")
    logger.info(f"Total cells: {len(all_results)}")
    logger.info(f"Saved to: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
