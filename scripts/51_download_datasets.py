"""Download all auto-downloadable SensorCI datasets without wget (Windows-safe).

Auto-downloads: TEP, C-MAPSS, AMPDs2, PAMAP2, MIT-BIH
Manual required: Paderborn (registration), LBNL FDD (email request)

Usage:
  python scripts/51_download_datasets.py
  python scripts/51_download_datasets.py --datasets tep cmapss mitbih
"""

from __future__ import annotations

import argparse
import os
import sys
import zipfile
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import ssl
import urllib3

import requests
from tqdm import tqdm
from dotenv import load_dotenv

# Corporate network uses SSL inspection with self-signed certs — disable verification
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
ssl._create_default_https_context = ssl._create_unverified_context
# Also patch for requests-based libraries (wfdb, etc.)
os.environ.setdefault("CURL_CA_BUNDLE", "")
os.environ.setdefault("REQUESTS_CA_BUNDLE", "")
os.environ.setdefault("PYTHONHTTPSVERIFY", "0")

load_dotenv(override=False)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

DATA_ROOT = Path(os.environ.get("SENSORCI_DATA_DIR", "./data"))


# ---------------------------------------------------------------------------
# Core HTTP downloader
# ---------------------------------------------------------------------------

def _download_file(url: str, dest: Path, desc: str = "", chunk_mb: int = 4) -> bool:
    """Stream-download url → dest with tqdm progress. Returns True on success."""
    if dest.exists() and dest.stat().st_size > 0:
        logger.info(f"  [skip] {dest.name} already exists ({dest.stat().st_size // 1024**2} MB)")
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        r = requests.get(url, stream=True, timeout=120, verify=False,
                         headers={"User-Agent": "SensorCI-downloader/1.0"})
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        chunk = chunk_mb * 1024 * 1024
        with open(dest, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True,
            desc=desc or dest.name, leave=False,
        ) as bar:
            for data in r.iter_content(chunk_size=chunk):
                f.write(data)
                bar.update(len(data))
        return True
    except Exception as e:
        logger.error(f"  [fail] {dest.name}: {e}")
        if dest.exists():
            dest.unlink()
        return False


def _unzip(zip_path: Path, dest_dir: Path) -> None:
    logger.info(f"  Extracting {zip_path.name} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)
    zip_path.unlink()


# ---------------------------------------------------------------------------
# Dataset-specific downloaders
# ---------------------------------------------------------------------------

def download_tep(data_dir: Path) -> bool:
    """TEP (Rieth extended) from Harvard Dataverse. ~500 MB."""
    data_dir.mkdir(parents=True, exist_ok=True)
    existing = list(data_dir.glob("*.RData"))
    if len(existing) >= 4:
        logger.info(f"  [skip] TEP already downloaded ({len(existing)} RData files)")
        return True
    # Download full dataset ZIP (more robust than per-file persistent IDs)
    doi = "doi:10.7910/DVN/6C3JR1"
    url = f"https://dataverse.harvard.edu/api/access/dataset/:persistentId/?persistentId={doi}"
    zip_path = data_dir / "tep.zip"
    if _download_file(url, zip_path, desc="TEP"):
        _unzip(zip_path, data_dir)
        logger.info(f"  TEP ready in {data_dir}")
        logger.info("  NOTE: requires pyreadr to parse .RData → pip install pyreadr")
        return True
    logger.error("  TEP failed. Manual: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/6C3JR1")
    return False


def download_cmapss(data_dir: Path) -> bool:
    """C-MAPSS from NASA. ~25 MB."""
    data_dir.mkdir(parents=True, exist_ok=True)
    if (data_dir / "train_FD001.txt").exists():
        logger.info("  [skip] C-MAPSS already extracted")
        return True
    urls = [
        "https://data.nasa.gov/download/ff5v-kuh6/application%2Fzip",
        "https://www.nasa.gov/sites/default/files/atoms/files/cmapssdata.zip",
    ]
    zip_path = data_dir / "cmapss.zip"
    for url in urls:
        logger.info(f"  Trying {url[:60]}...")
        if _download_file(url, zip_path, desc="C-MAPSS"):
            _unzip(zip_path, data_dir)
            logger.info(f"  C-MAPSS ready in {data_dir}")
            return True
    logger.error("  C-MAPSS: all URLs failed. Download manually:")
    logger.error("  https://www.kaggle.com/datasets/behrad3d/nasa-cmaps")
    return False


def download_ampds2(data_dir: Path) -> bool:
    """AMPDs2 from Harvard Dataverse. ~150 MB."""
    data_dir.mkdir(parents=True, exist_ok=True)
    if (data_dir / "Electricity_WHE.csv").exists():
        logger.info("  [skip] AMPDs2 already extracted")
        return True
    # Download full dataset ZIP
    doi = "doi:10.7910/DVN/FIE0S4"
    url = f"https://dataverse.harvard.edu/api/access/dataset/:persistentId/?persistentId={doi}"
    zip_path = data_dir / "ampds2.zip"
    if _download_file(url, zip_path, desc="AMPDs2"):
        _unzip(zip_path, data_dir)
        logger.info(f"  AMPDs2 ready in {data_dir}")
        return True
    logger.error("  AMPDs2 failed. Manual: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/FIE0S4")
    return False


def download_pamap2(data_dir: Path) -> bool:
    """PAMAP2 from UCI. ~700 MB."""
    data_dir.mkdir(parents=True, exist_ok=True)
    if (data_dir / "Protocol" / "subject101.dat").exists():
        logger.info("  [skip] PAMAP2 already extracted")
        return True
    url = "https://archive.ics.uci.edu/static/public/231/pamap2+physical+activity+monitoring.zip"
    zip_path = data_dir / "pamap2.zip"
    if _download_file(url, zip_path, desc="PAMAP2"):
        _unzip(zip_path, data_dir)
        # Handle nested zip
        nested = data_dir / "PAMAP2_Dataset.zip"
        if nested.exists():
            _unzip(nested, data_dir)
        logger.info(f"  PAMAP2 ready in {data_dir}")
        return True
    logger.error("  PAMAP2 failed. Manual: https://archive.ics.uci.edu/dataset/231")
    return False


def download_mitbih(data_dir: Path) -> bool:
    """MIT-BIH from PhysioNet via wfdb. ~75 MB."""
    data_dir.mkdir(parents=True, exist_ok=True)
    existing = list(data_dir.glob("*.dat"))
    if len(existing) >= 40:
        logger.info(f"  [skip] MIT-BIH already downloaded ({len(existing)} records)")
        return True
    try:
        import wfdb
    except ImportError:
        logger.info("  Installing wfdb...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "wfdb", "-q"])
        import wfdb

    records = [
        "100","101","102","103","104","105","106","107","108","109",
        "111","112","113","114","115","116","117","118","119","121",
        "122","123","124","200","201","202","203","205","207","208",
        "209","210","212","213","214","215","217","219","220","221",
        "222","223","228","230","231","232","233","234",
    ]
    logger.info(f"  Downloading {len(records)} MIT-BIH records via wfdb...")
    failed = []
    for r in tqdm(records, desc="MIT-BIH", leave=False):
        if (data_dir / f"{r}.dat").exists():
            continue
        try:
            wfdb.dl_database("mitdb", dl_dir=str(data_dir), records=[r])
        except Exception as e:
            failed.append(r)
            logger.warning(f"  failed {r}: {e}")
    if failed:
        logger.warning(f"  MIT-BIH: {len(failed)} records failed: {failed}")
    logger.info(f"  MIT-BIH ready in {data_dir}")
    return len(failed) == 0


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

DOWNLOADERS = {
    "tep":    (download_tep,    "~500 MB", "Harvard Dataverse (open)"),
    "cmapss": (download_cmapss, "~25 MB",  "NASA public domain"),
    "ampds2": (download_ampds2, "~150 MB", "Harvard Dataverse CC-BY"),
    "pamap2": (download_pamap2, "~700 MB", "UCI public"),
    "mitbih": (download_mitbih, "~75 MB",  "PhysioNet ODC-By"),
}

MANUAL_DATASETS = {
    "paderborn": "Registration required → https://mb.uni-paderborn.de/kat/forschung/kat-datacenter/bearing-datacenter/data-sets-and-download/",
    "lbnl_fdd":  "Email request required → see data_downloads/04_lbnl_fdd.md",
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Download SensorCI datasets (Windows-safe, no wget)")
    ap.add_argument("--datasets", nargs="+", default=list(DOWNLOADERS),
                    help=f"Datasets to download (default: all auto). Choices: {list(DOWNLOADERS)}")
    ap.add_argument("--parallel", action="store_true",
                    help="Download all datasets in parallel (faster but noisy logs)")
    ap.add_argument("--data_dir", default=None,
                    help="Override SENSORCI_DATA_DIR")
    args = ap.parse_args()

    root = Path(args.data_dir) if args.data_dir else DATA_ROOT
    root.mkdir(parents=True, exist_ok=True)

    targets = [d for d in args.datasets if d in DOWNLOADERS]
    skipped = [d for d in args.datasets if d not in DOWNLOADERS]

    logger.info("=== SensorCI Dataset Downloader ===")
    logger.info(f"Data root : {root.resolve()}")
    logger.info(f"Datasets  : {targets}")
    if skipped:
        logger.warning(f"Skipped (not auto-downloadable): {skipped}")
    logger.info("")

    for ds, reason in MANUAL_DATASETS.items():
        if ds in (args.datasets or []):
            logger.warning(f"  {ds}: {reason}")

    results: dict[str, bool] = {}

    if args.parallel and len(targets) > 1:
        with ThreadPoolExecutor(max_workers=len(targets)) as ex:
            futures = {
                ex.submit(DOWNLOADERS[ds][0], root / ds): ds
                for ds in targets
            }
            for fut in as_completed(futures):
                ds = futures[fut]
                try:
                    results[ds] = fut.result()
                except Exception as e:
                    logger.error(f"  {ds} error: {e}")
                    results[ds] = False
    else:
        for ds in targets:
            fn, size, source = DOWNLOADERS[ds]
            logger.info(f"\n--- {ds} ({size}, {source}) ---")
            try:
                results[ds] = fn(root / ds)
            except Exception as e:
                logger.error(f"  {ds} error: {e}")
                results[ds] = False

    logger.info("\n=== Summary ===")
    all_ok = True
    for ds in targets:
        ok = results.get(ds, False)
        status = "OK" if ok else "FAILED"
        logger.info(f"  [{status}] {ds}")
        all_ok &= ok

    logger.info("\nManual downloads still needed:")
    for ds, reason in MANUAL_DATASETS.items():
        logger.info(f"  {ds}: {reason}")

    if all_ok:
        logger.info("\nAll auto-downloadable datasets ready.")
        logger.info("Now run: python scripts/50_launch_full_experiment.py")
    else:
        logger.warning("\nSome downloads failed — check logs above.")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
