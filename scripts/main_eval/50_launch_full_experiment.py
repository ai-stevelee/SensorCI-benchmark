"""Parallel SensorCI full experiment launcher.

Spawns up to 6 parallel processes simultaneously:
  gpu0 : tep, paderborn  → a1 a3 a4 a7  (CUDA_VISIBLE_DEVICES=0)
  gpu1 : cmapss, lbnl_fdd → a1 a3 a4 a7 (CUDA_VISIBLE_DEVICES=1)
  gpu2 : ampds2, pamap2  → a1 a3 a4 a7  (CUDA_VISIBLE_DEVICES=2)
  gpu3 : mitbih          → a1 a3 a4 a7  (CUDA_VISIBLE_DEVICES=3)
  cpu  : all datasets    → a2 a5 a6     (no GPU, --max_workers 48)
  api  : all datasets    → a0_frontier  (no GPU, --max_workers 100)
  dl   : download_all.sh (background, fire-and-forget)

Usage:
  python scripts/50_launch_full_experiment.py
  python scripts/50_launch_full_experiment.py --phase gpu
  python scripts/50_launch_full_experiment.py --phase api
  python scripts/50_launch_full_experiment.py --phase cpu
  python scripts/50_launch_full_experiment.py --defenses d1_raw d5_affine
  python scripts/50_launch_full_experiment.py --seeds 42 43 --n_samples 30 --no_download
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=False)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent          # code/
MAIN_SCRIPT = ROOT / "scripts" / "30_main_evaluation.py"
DOWNLOAD_SH = ROOT.parent / "data_downloads" / "scripts" / "download_all.sh"

# ---------------------------------------------------------------------------
# Experiment layout
# ---------------------------------------------------------------------------

GPU_DATASETS: dict[int, list[str]] = {
    0: ["tep"],
    1: ["cmapss"],
    2: ["ampds2", "pamap2"],
    3: ["mitbih"],
}
ALL_DATASETS: list[str] = [ds for dss in GPU_DATASETS.values() for ds in dss]

GPU_ATTACKS = ["a1_distinguish", "a3_neural_recon", "a4_ci_linkage", "a7_mia_ts"]
CPU_ATTACKS = ["a2_linear_recon", "a5_stat_tests", "a6_algebra"]
API_ATTACKS = ["a0_frontier"]

ALL_DEFENSES = [
    "d1_raw", "d2_gaussian_dp", "d3_laplace_dp", "d4_dp_ae",
    "d5_affine", "d6_per_seg_affine", "d7_ope",
    "d8_doppelganger", "d9_kanon", "d10_fe_ip",
    "d11_vfae", "d12_ib",
]
DEFAULT_SEEDS = [42, 43, 44, 45, 46]
DEFAULT_TIERS = ["T1", "T2", "T3", "T4"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_env(gpu_id: int | None) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id) if gpu_id is not None else ""
    return env


def _build_cmd(
    datasets: list[str],
    attacks: list[str],
    defenses: list[str],
    tiers: list[str],
    seeds: list[int],
    n_samples: int,
    max_workers: int | None = None,
    out_dir: Path | None = None,
) -> list[str]:
    cmd = [
        sys.executable, str(MAIN_SCRIPT),
        "--datasets", *datasets,
        "--attacks", *attacks,
        "--defenses", *defenses,
        "--tiers", *tiers,
        "--seeds", *[str(s) for s in seeds],
        "--n_samples", str(n_samples),
    ]
    if max_workers is not None:
        cmd += ["--max_workers", str(max_workers)]
    if out_dir is not None:
        cmd += ["--out_dir", str(out_dir)]
    return cmd


def _spawn(
    name: str,
    cmd: list[str],
    env: dict[str, str],
    log_dir: Path,
) -> tuple[subprocess.Popen, object, Path]:
    log_path = log_dir / f"{name}.log"
    log_f = open(log_path, "w", encoding="utf-8")
    extra: dict = {}
    if sys.platform == "win32":
        extra["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    p = subprocess.Popen(
        cmd,
        env=env,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        **extra,
    )
    logger.info(f"  [{name}] PID={p.pid}  log → {log_path.name}")
    return p, log_f, log_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description="SensorCI parallel experiment launcher")
    ap.add_argument("--phase",
                    choices=["all", "gpu", "cpu", "api", "download"],
                    default="all",
                    help="Which process group(s) to start (default: all)")
    ap.add_argument("--defenses", nargs="+", default=None,
                    help="Subset of defenses (default: all 12)")
    ap.add_argument("--tiers", nargs="+", default=None,
                    help="Subset of tiers T1..T4 (default: all)")
    ap.add_argument("--seeds", nargs="+", type=int, default=None,
                    help="Seeds (default: 42 43 44 45 46)")
    ap.add_argument("--n_samples", type=int, default=50,
                    help="Samples per cell (default: 50)")
    ap.add_argument("--out_dir", default=None,
                    help="Results output directory")
    ap.add_argument("--no_download", action="store_true",
                    help="Skip data download subprocess")
    ap.add_argument("--gpus", nargs="+", type=int, default=None,
                    help="Subset of GPU IDs to use (default: 0 1 2 3)")
    args = ap.parse_args()

    defenses = args.defenses or ALL_DEFENSES
    tiers = args.tiers or DEFAULT_TIERS
    seeds = args.seeds or DEFAULT_SEEDS
    out_dir = Path(args.out_dir or os.environ.get("SENSORCI_RESULTS_DIR", "./results"))
    out_dir.mkdir(parents=True, exist_ok=True)
    gpu_ids = args.gpus if args.gpus is not None else list(GPU_DATASETS.keys())

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = out_dir / f"logs_{ts}"
    log_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("SensorCI Full Experiment Launcher")
    logger.info(f"Phase     : {args.phase}")
    logger.info(f"GPUs      : {gpu_ids}")
    logger.info(f"Defenses  : {len(defenses)}")
    logger.info(f"Tiers     : {tiers}")
    logger.info(f"Seeds     : {seeds}")
    logger.info(f"N samples : {args.n_samples}")
    logger.info(f"Logs      : {log_dir}")
    logger.info("=" * 60)

    procs: list[tuple[str, subprocess.Popen, object, Path]] = []

    # -----------------------------------------------------------------------
    # Data download (fire-and-forget background)
    # -----------------------------------------------------------------------
    if not args.no_download and args.phase in ("all", "download"):
        if DOWNLOAD_SH.exists():
            bash = shutil.which("bash")
            if bash is None:
                # Common Windows Git Bash locations
                for candidate in [
                    r"C:\Program Files\Git\bin\bash.exe",
                    r"C:\Program Files (x86)\Git\bin\bash.exe",
                    r"C:\Windows\System32\bash.exe",
                ]:
                    if Path(candidate).exists():
                        bash = candidate
                        break
            if bash:
                dl_cmd = [bash, str(DOWNLOAD_SH)]
                p, lf, lp = _spawn("download_all", dl_cmd, os.environ.copy(), log_dir)
                procs.append(("download_all", p, lf, lp))
            else:
                logger.warning("bash not found — skip auto-download.")
                logger.warning(f"Run manually: bash {DOWNLOAD_SH}")
        else:
            logger.warning(f"Download script not found: {DOWNLOAD_SH} — skipping")

    # -----------------------------------------------------------------------
    # GPU workers — neural attacks, one process per GPU
    # -----------------------------------------------------------------------
    if args.phase in ("all", "gpu"):
        for gpu_id in gpu_ids:
            datasets = GPU_DATASETS[gpu_id]
            name = f"gpu{gpu_id}_neural"
            cmd = _build_cmd(
                datasets=datasets,
                attacks=GPU_ATTACKS,
                defenses=defenses,
                tiers=tiers,
                seeds=seeds,
                n_samples=args.n_samples,
                out_dir=out_dir,
            )
            env = _make_env(gpu_id)
            p, lf, lp = _spawn(name, cmd, env, log_dir)
            procs.append((name, p, lf, lp))
            logger.info(f"    datasets={datasets} attacks={GPU_ATTACKS}")

    # -----------------------------------------------------------------------
    # CPU worker — math / statistical attacks (no GPU)
    # -----------------------------------------------------------------------
    if args.phase in ("all", "cpu"):
        name = "cpu_math"
        cmd = _build_cmd(
            datasets=ALL_DATASETS,
            attacks=CPU_ATTACKS,
            defenses=defenses,
            tiers=tiers,
            seeds=seeds,
            n_samples=args.n_samples,
            max_workers=48,
            out_dir=out_dir,
        )
        env = _make_env(None)
        p, lf, lp = _spawn(name, cmd, env, log_dir)
        procs.append((name, p, lf, lp))
        logger.info(f"    datasets={ALL_DATASETS} attacks={CPU_ATTACKS} workers=48")

    # -----------------------------------------------------------------------
    # API worker — A0 frontier LLM (T1 only, 100 concurrent API keys)
    # -----------------------------------------------------------------------
    if args.phase in ("all", "api"):
        name = "api_a0"
        cmd = _build_cmd(
            datasets=ALL_DATASETS,
            attacks=API_ATTACKS,
            defenses=defenses,
            tiers=["T1"],   # A0 is the headline T1 attack
            seeds=seeds,
            n_samples=args.n_samples,
            max_workers=100,
            out_dir=out_dir,
        )
        env = _make_env(None)
        p, lf, lp = _spawn(name, cmd, env, log_dir)
        procs.append((name, p, lf, lp))
        logger.info(f"    datasets={ALL_DATASETS} A0 frontier workers=100")

    if not procs:
        logger.error("No processes started — nothing to do.")
        return 1

    logger.info(f"\n{len(procs)} process(es) running.")
    logger.info("Monitor with:  tail -f " + str(log_dir) + "/*.log")
    logger.info("Per-process :  tail -f " + str(log_dir) + "/gpu0_neural.log")
    logger.info("Press Ctrl-C to terminate all.\n")

    # -----------------------------------------------------------------------
    # Monitor until all done
    # -----------------------------------------------------------------------
    finished: set[str] = set()
    try:
        while True:
            time.sleep(30)
            still_running = []
            for name, p, lf, lp in procs:
                rc = p.poll()
                if rc is not None and name not in finished:
                    finished.add(name)
                    status = "OK" if rc == 0 else f"FAILED(rc={rc})"
                    logger.info(f"  [{status}] {name} (PID {p.pid})")
                elif rc is None:
                    still_running.append(name)
            if not still_running:
                break
            logger.info(f"  Running ({len(still_running)}): {still_running}")

    except KeyboardInterrupt:
        logger.warning("\nCtrl-C — terminating all processes...")
        for name, p, _lf, _lp in procs:
            if p.poll() is None:
                p.terminate()
                logger.info(f"  Terminated {name} (PID {p.pid})")

    # Close log file handles
    for _name, _p, lf, _lp in procs:
        try:
            lf.close()
        except Exception:
            pass

    failed = [(n, p.returncode) for n, p, _, _ in procs
              if p.returncode is not None and p.returncode != 0]
    if failed:
        logger.error(f"{len(failed)} process(es) failed: {failed}")
        return 1

    logger.info("All processes completed successfully.")
    logger.info(f"Results → {out_dir}")
    logger.info(f"Logs    → {log_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
