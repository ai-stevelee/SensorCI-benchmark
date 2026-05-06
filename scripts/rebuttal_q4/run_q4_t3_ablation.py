"""Q4 (T3 collusion sensitivity): three ablations on PAMAP2 only.

Reviewer Q4: "PAMAP2 yields near-zero per-role advantage; is this a
measurement artifact?" -- we test by running three orthogonal ablations.

Ablations:
  1. stronger_probe : GradientBoostingClassifier (200 estimators, depth 5).
                     Tests: "is the siamese probe too weak?"
  2. all_attributes : enumerate every attribute (Share + Not-share + Partial),
                     not only Not-share. Tests: "does the probe work at all?"
  3. no_downsample  : force role downsample_factor = 1 (full signal for every
                     role). Tests: "is the role view too lossy?"

Cells: 12 defenses x 3 ablations x 5 seeds = 180 cells, PAMAP2 only.
Compute: ~1-2 GPU-hours (uses _features + sklearn classifiers, no torch GPU
                          training); CUDA_VISIBLE_DEVICES=0 still respected
                          for any neural defense (e.g. d11_vfae) it loads.

Output:
  results/q4_t3_ablation_<TIMESTAMP>.jsonl   (one row per cell)
  results/q4_t3_ablation_summary.json        (per-ablation, per-defense aggregation)
"""
from __future__ import annotations
import sys
from pathlib import Path
_HERE = Path(__file__).resolve().parent
_CODE_ROOT = _HERE.parent
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

import json
import os
import time
import warnings
from collections import defaultdict
from datetime import datetime, timezone
import statistics
import numpy as np

warnings.filterwarnings("ignore")
os.environ.setdefault("SENSORCI_DATA_DIR", str(_CODE_ROOT / "data"))


from sensorci.datasets import load_dataset
from sensorci.defenses import get_defense
from sensorci.attacks.a4_ci_t3_ablation import T3AblationProbe


DATASETS = ["pamap2"]   # Q4 is PAMAP2-only by design
ABLATIONS = ["stronger_probe", "all_attributes", "no_downsample"]
DEFENSES = [
    "d1_raw", "d2_gaussian_dp", "d3_laplace_dp", "d4_dp_ae",
    "d5_affine", "d6_per_seg_affine", "d7_ope", "d8_doppelganger",
    "d9_kanon", "d10_fe_ip", "d11_vfae", "d12_ib",
]
SEEDS = [42, 43, 44, 45, 46]
N_SAMPLES = 50


def _result_to_row(r) -> dict:
    return {
        "cell_id": r.cell_id,
        "dataset": r.dataset,
        "defense": r.defense,
        "attack": r.attack,
        "seed": r.seed,
        "metric_name": r.metric_name,
        "metric_value": r.metric_value,
        "subscores": dict(r.subscores) if r.subscores else {},
        "n_samples": r.n_samples,
        "wall_seconds": getattr(r, "wall_seconds", None),
        "extra": {k: v for k, v in (r.extra or {}).items()
                  if k != "per_role_scores"},
        # per-role drilldown is large; keep separately
        "per_role_scores": (r.extra or {}).get("per_role_scores", {}),
    }


def main() -> None:
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    jsonl_path = out_dir / f"q4_t3_ablation_{ts}.jsonl"

    print(f"=== Q4 T3 ablation runner ({ts}) ===")
    print(f"Datasets:  {DATASETS}")
    print(f"Defenses:  {len(DEFENSES)} ({DEFENSES})")
    print(f"Ablations: {ABLATIONS}")
    print(f"Seeds:     {SEEDS}, n_samples={N_SAMPLES}")
    print(f"Out:       {jsonl_path}")
    print()

    rows: list[dict] = []
    t_global = time.time()

    for ds_name in DATASETS:
        print(f"--- loading: {ds_name} ---")
        ds = load_dataset(ds_name)
        sig_std = float(ds.personas[0].segments[0].signal.std())
        if sig_std < 1e-3:
            raise RuntimeError(
                f"{ds_name} signal std={sig_std:.2e}; possible synthetic fallback. "
                f"Aborting (real data required)."
            )
        print(f"  loaded: {len(ds.personas)} personas, "
              f"signal[0]={ds.personas[0].segments[0].signal.shape}, std={sig_std:.3f}")

        for ablation in ABLATIONS:
            attack = T3AblationProbe(ablation_kind=ablation)
            for def_name in DEFENSES:
                for seed in SEEDS:
                    t1 = time.time()
                    try:
                        defense = get_defense(def_name, key_seed=seed) \
                                  if hasattr(get_defense, "__call__") \
                                  else get_defense(def_name)
                        if hasattr(defense, "fit"):
                            defense.fit(ds)
                        result = attack.run(defense, ds, n_samples=N_SAMPLES, seed=seed)
                    except TypeError:
                        # some defenses don't take key_seed; try plain ctor
                        defense = get_defense(def_name)
                        if hasattr(defense, "fit"):
                            defense.fit(ds)
                        result = attack.run(defense, ds, n_samples=N_SAMPLES, seed=seed)
                    except Exception as e:
                        print(f"  {ds_name:<8} {def_name:<20} {ablation:<16} "
                              f"seed={seed} -> FAILED: {e}")
                        continue

                    row = _result_to_row(result)
                    row["ablation_kind"] = ablation
                    rows.append(row)
                    with open(jsonl_path, "a") as f:
                        # Drop per_role_scores to keep the jsonl line compact
                        compact = {k: v for k, v in row.items() if k != "per_role_scores"}
                        f.write(json.dumps(compact) + "\n")

                    t3_exp = result.subscores.get("t3_expansion_mean", float("nan"))
                    print(f"  {ds_name:<8} {def_name:<20} {ablation:<16} "
                          f"seed={seed} -> max_adv={result.metric_value:.4f}, "
                          f"T3_exp={t3_exp:.3f}  ({time.time()-t1:.1f}s)")

    print()
    print(f"=== Done: {len(rows)} cells in {time.time()-t_global:.1f}s ===")

    # Aggregate
    summary: dict = {
        "timestamp_utc": ts,
        "n_cells": len(rows),
        "per_ablation_per_defense": {},
        "per_ablation_summary": {},
    }
    per_ad = defaultdict(list)   # (ablation, defense) -> [max_adv, ...]
    per_a = defaultdict(list)    # ablation -> [max_adv, ...]
    per_a_t3 = defaultdict(list) # ablation -> [t3_expansion_mean, ...]
    for r in rows:
        ab = r["ablation_kind"]
        de = r["defense"]
        per_ad[(ab, de)].append(r["metric_value"])
        per_a[ab].append(r["metric_value"])
        t3_v = r["subscores"].get("t3_expansion_mean")
        if t3_v is not None:
            per_a_t3[ab].append(t3_v)

    for (ab, de), vs in per_ad.items():
        summary["per_ablation_per_defense"].setdefault(ab, {})[de] = {
            "max_adv_mean": statistics.mean(vs),
            "max_adv_std": statistics.stdev(vs) if len(vs) > 1 else 0.0,
            "n_seeds": len(vs),
        }
    for ab in per_a:
        summary["per_ablation_summary"][ab] = {
            "max_adv_mean": statistics.mean(per_a[ab]),
            "max_adv_max": max(per_a[ab]),
            "max_adv_p90": float(np.quantile(per_a[ab], 0.9)) if per_a[ab] else 0.0,
            "t3_expansion_mean": (statistics.mean(per_a_t3[ab])
                                  if per_a_t3[ab] else float("nan")),
            "n_cells": len(per_a[ab]),
        }

    summary_path = out_dir / "q4_t3_ablation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary: {summary_path}")

    print()
    print("=== Per-ablation overview (PAMAP2 only) ===")
    for ab in ABLATIONS:
        s = summary["per_ablation_summary"].get(ab, {})
        print(f"  {ab:<16}: mean max_adv={s.get('max_adv_mean', 0):.4f}, "
              f"p90={s.get('max_adv_p90', 0):.4f}, "
              f"max={s.get('max_adv_max', 0):.4f}, "
              f"T3 expansion mean={s.get('t3_expansion_mean', float('nan')):.3f} "
              f"(n={s.get('n_cells', 0)})")


if __name__ == "__main__":
    main()
