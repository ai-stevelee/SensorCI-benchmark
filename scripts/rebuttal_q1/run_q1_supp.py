"""Q1 (SP1 fairness): compute S_P1^supp for D7 OPE and D10 FE-IP across 5 datasets.

This script runs the AlgebraSupportedScopeProbe attack:
  - For D7 OPE: rank-domain query bank (rank queries are preserved by OPE bucket).
  - For D10 FE-IP: inner-product query bank with mode="key_sharing_analyst".

Output:
  results/q1_supp_<timestamp>.jsonl   (one row per cell)
  results/q1_supp_summary.json        (per-defense, per-dataset aggregation)

Cells: 2 defenses x 5 datasets x 5 seeds = 50 cells, all CPU, ~25 min.
Real data only (synthetic fallback raises if dataset files absent).
"""
from __future__ import annotations
import sys
from pathlib import Path
# Ensure local sensorci package (Downloads/SensorCI/code/) is preferred over any
# pip-editable install elsewhere on the system.
_HERE = Path(__file__).resolve().parent
_CODE_ROOT = _HERE.parent  # Downloads/SensorCI/code/
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

import json
import os
import time
import warnings
from datetime import datetime, timezone

warnings.filterwarnings("ignore")
os.environ.setdefault("SENSORCI_DATA_DIR", str(_CODE_ROOT / "data"))

from sensorci.datasets import load_dataset
from sensorci.defenses.d7_ope import OPEDefense
from sensorci.defenses.d10_fe_ip import FunctionalEncryptionIPDefense
from sensorci.attacks.a6_algebra import AlgebraSupportedScopeProbe


DATASETS = ["tep", "cmapss", "ampds2", "pamap2", "mitbih"]
SEEDS = [42, 43, 44, 45, 46]
N_SAMPLES = 50
N_QUERIES = 20


def main() -> None:
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    jsonl_path = out_dir / f"q1_supp_{ts}.jsonl"

    print(f"=== Q1 SP1^supp runner ({ts}) ===")
    print(f"Datasets: {DATASETS}")
    print(f"Defenses: d7_ope, d10_fe_ip (mode=key_sharing_analyst)")
    print(f"Seeds: {SEEDS}, n_samples={N_SAMPLES}, n_queries={N_QUERIES}")
    print(f"Out: {jsonl_path}")
    print()

    attack = AlgebraSupportedScopeProbe(n_queries=N_QUERIES)
    rows: list[dict] = []
    t_global = time.time()

    for ds_name in DATASETS:
        print(f"--- loading dataset: {ds_name} ---")
        t0 = time.time()
        ds = load_dataset(ds_name)
        n_p = len(ds.personas)
        sig_shape = ds.personas[0].segments[0].signal.shape
        sig_std = float(ds.personas[0].segments[0].signal.std())
        if sig_std < 1e-3:
            raise RuntimeError(
                f"{ds_name} signal std={sig_std:.2e} is suspiciously small; "
                f"possible synthetic fallback. Aborting (real data only)."
            )
        print(f"  loaded in {time.time()-t0:.1f}s: personas={n_p}, "
              f"signal[0]={sig_shape}, std={sig_std:.3f}")

        for defense_factory in [
            ("d7_ope", lambda seed: OPEDefense(n_bits=16, key_seed=seed)),
            ("d10_fe_ip",
             lambda seed: FunctionalEncryptionIPDefense(
                 mode="key_sharing_analyst", key_seed=seed)),
        ]:
            def_name, factory = defense_factory
            for seed in SEEDS:
                t1 = time.time()
                defense = factory(seed)
                defense.fit(ds)
                result = attack.run(defense, ds, n_samples=N_SAMPLES, seed=seed)

                row = result.to_dict() if hasattr(result, "to_dict") else {
                    "cell_id": result.cell_id,
                    "dataset": result.dataset,
                    "defense": result.defense,
                    "attack": result.attack,
                    "seed": result.seed,
                    "metric_name": result.metric_name,
                    "metric_value": result.metric_value,
                    "subscores": dict(result.subscores) if result.subscores else {},
                    "n_samples": result.n_samples,
                    "wall_seconds": getattr(result, "wall_seconds", time.time()-t1),
                    "extra": dict(result.extra) if result.extra else {},
                }
                # Trim heavy fields
                if "extra" in row and "per_query_median_re" in row["extra"]:
                    row["extra"] = {
                        k: v for k, v in row["extra"].items()
                        if k != "per_query_median_re"
                    }
                rows.append(row)

                with open(jsonl_path, "a") as f:
                    f.write(json.dumps(row) + "\n")
                print(f"  {ds_name:<8} {def_name:<12} seed={seed} -> "
                      f"S_P1^supp={result.metric_value:.4f} "
                      f"({time.time()-t1:.1f}s)")

    print()
    print(f"=== Done: {len(rows)} cells in {time.time()-t_global:.1f}s ===")

    # Aggregate
    summary: dict = {
        "timestamp_utc": ts,
        "n_cells": len(rows),
        "per_defense_dataset": {},
        "per_defense_cross_dataset": {},
    }
    from collections import defaultdict
    import statistics

    per_dd = defaultdict(list)
    per_d = defaultdict(list)
    for r in rows:
        key_dd = (r["defense"], r["dataset"])
        per_dd[key_dd].append(r["metric_value"])
        per_d[r["defense"]].append(r["metric_value"])

    for (de, ds), vs in per_dd.items():
        summary["per_defense_dataset"].setdefault(de, {})[ds] = {
            "S_P1_supp_mean": statistics.mean(vs),
            "S_P1_supp_std": statistics.stdev(vs) if len(vs) > 1 else 0.0,
            "n_seeds": len(vs),
        }
    for de, vs in per_d.items():
        summary["per_defense_cross_dataset"][de] = {
            "S_P1_supp_mean": statistics.mean(vs),
            "S_P1_supp_std": statistics.stdev(vs) if len(vs) > 1 else 0.0,
            "n_cells": len(vs),
        }

    summary_path = out_dir / "q1_supp_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary: {summary_path}")

    # Print summary table
    print()
    print("=== Cross-dataset S_P1^supp ===")
    for de in sorted(per_d):
        m = statistics.mean(per_d[de])
        s = statistics.stdev(per_d[de]) if len(per_d[de]) > 1 else 0.0
        print(f"  {de:<12}: mean={m:.4f} ± {s:.4f} (n={len(per_d[de])})")

    print()
    print("=== Per-dataset S_P1^supp ===")
    print(f"  {'defense':<12} | " + " | ".join(f"{ds:<8}" for ds in DATASETS))
    for de in sorted(per_d):
        cells = []
        for ds in DATASETS:
            vs = per_dd.get((de, ds), [])
            cells.append(f"{statistics.mean(vs):.3f}" if vs else "n/a")
        print(f"  {de:<12} | " + " | ".join(f"{c:<8}" for c in cells))


if __name__ == "__main__":
    main()
