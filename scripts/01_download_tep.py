"""Download or generate TEP dataset.

Default: synthetic mode (offline, deterministic, fast).
Pass --real to download Rieth et al. extended TEP from Harvard Dataverse (~500MB).
"""

import argparse
import sys
from sensorci.datasets.tep import TEPDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true",
                    help="Download real TEP RData from Harvard Dataverse instead of synthetic")
    ap.add_argument("--n_simulations", type=int, default=12)
    ap.add_argument("--segments_per_sim", type=int, default=8)
    ap.add_argument("--segment_length", type=int, default=256)
    args = ap.parse_args()

    mode = "real" if args.real else "synthetic"
    print(f"Loading TEP in {mode} mode...")
    loader = TEPDataset(
        mode=mode,
        n_simulations=args.n_simulations,
        segments_per_sim=args.segments_per_sim,
        segment_length=args.segment_length,
    )
    ds = loader.load()
    print(f"\nLoaded {ds.name}:")
    print(f"  personas: {len(ds.personas)}")
    print(f"  total segments: {sum(len(p.segments) for p in ds.personas)}")
    print(f"  channels: {ds.dataset_metadata['channels']}")
    print(f"  classes: {len(ds.class_list)} (e.g., {ds.class_list[:5]}...)")
    print(f"  sharing labels: {len(ds.sharing_matrix)}")

    # Print first persona for sanity
    p = ds.personas[0]
    print(f"\nFirst persona ({p.persona_id}):")
    print(f"  identity: {p.identity_attributes}")
    print(f"  operating: {p.operating_attributes}")
    print(f"  sensitive: {p.sensitive_attributes}")
    print(f"  n_segments: {len(p.segments)}")
    print(f"  segment[0] shape: {p.segments[0].signal.shape}")

    # Save metadata
    out = loader.data_dir / "metadata.json"
    ds.save_metadata(out)
    print(f"\nMetadata saved to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
