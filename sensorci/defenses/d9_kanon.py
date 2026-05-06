"""D9: k-Anonymity for time series (microaggregation).

Cluster personas into groups of size k, replace each persona's signal with
the cluster centroid signal. Provides identity protection up to k-membership.
"""

from __future__ import annotations
import numpy as np

from sensorci.defenses.base import BaseDefense
from sensorci.core.dataset import TSDataset


class KAnonymityTSDefense(BaseDefense):
    name = "d9_kanon"
    has_decoder = False

    def __init__(self, k: int = 5, key_seed: int = 42):
        super().__init__()
        self.k = k
        self.key_seed = key_seed
        # source_id -> cluster_id; cluster_id -> centroid signal (per-channel mean)
        self._cluster: dict[str, int] = {}
        self._centroid: dict[int, np.ndarray] = {}

    def fit(self, dataset: TSDataset) -> None:
        if not dataset.personas:
            return
        # Compute per-persona signature: mean per channel of first segment
        sigs = []
        ids = []
        for p in dataset.personas:
            if not p.segments:
                continue
            s = p.segments[0].signal
            sigs.append(s.mean(axis=1))  # (n_channels,)
            ids.append(p.persona_id)
        if not sigs:
            return
        X = np.array(sigs)
        # Simple greedy clustering: sort by first-channel mean, group every k
        order = np.argsort(X[:, 0])
        clusters: dict[int, list[int]] = {}
        for cid_offset, idx in enumerate(order):
            cid = cid_offset // self.k
            clusters.setdefault(cid, []).append(idx)
            self._cluster[ids[idx]] = cid

        # Compute centroid signal per cluster (concatenate all segments)
        rng = np.random.default_rng(self.key_seed)
        for cid, member_idxs in clusters.items():
            members = [dataset.personas[i] for i in member_idxs]
            # Build a representative signal: mean over members' first segment
            arrays = [m.segments[0].signal for m in members if m.segments]
            if not arrays:
                continue
            min_len = min(a.shape[1] for a in arrays)
            stacked = np.stack([a[:, :min_len] for a in arrays], axis=0)
            centroid = stacked.mean(axis=0)
            # Add small noise to break exact equality across same cluster
            centroid += rng.normal(0, 0.05 * centroid.std(), centroid.shape)
            self._centroid[cid] = centroid

    def transform(self, signal: np.ndarray, source_id: str | None = None,
                  segment_id: str | None = None) -> np.ndarray:
        if source_id is None or source_id not in self._cluster:
            return signal.copy()
        cid = self._cluster[source_id]
        centroid = self._centroid.get(cid)
        if centroid is None:
            return signal.copy()
        # Resize centroid to match input shape
        out = np.zeros_like(signal)
        n_ch_centroid, len_centroid = centroid.shape
        n_ch_in, len_in = signal.shape
        for c in range(n_ch_in):
            cc = c % n_ch_centroid
            if len_centroid >= len_in:
                out[c] = centroid[cc, :len_in]
            else:
                # Tile/pad
                reps = (len_in // len_centroid) + 1
                tiled = np.tile(centroid[cc], reps)
                out[c] = tiled[:len_in]
        return out

    @property
    def hyperparameters(self) -> dict:
        return {"name": self.name, "k": self.k, "key_seed": self.key_seed}
