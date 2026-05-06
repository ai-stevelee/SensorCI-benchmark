"""D7: Order-Preserving Encryption (OPE) — quantization-based stand-in.

Real OPE (Boldyreva 2009) is rank-preserving by construction; we simulate via
per-channel quantile bucket assignment using a shared seed.

This produces output that:
  - Preserves rank ordering (P1 partial: order-aware queries OK)
  - Looks NOT like a natural signal (P2 fail: integer buckets)
  - Has known leakage (Grubbs et al. S&P 2017)
"""

from __future__ import annotations
import numpy as np

from sensorci.defenses.base import BaseDefense
from sensorci.core.dataset import TSDataset


class OPEDefense(BaseDefense):
    name = "d7_ope"
    has_decoder = True  # rank decoder

    def __init__(self, n_bits: int = 16, key_seed: int = 42):
        super().__init__()
        self.n_bits = n_bits
        self.n_buckets = 2 ** n_bits
        self.key_seed = key_seed
        # Per-channel quantile boundaries (key)
        self._quantiles: dict[int, np.ndarray] = {}

    def fit(self, dataset: TSDataset) -> None:
        # Aggregate per-channel values across personas
        per_channel: dict[int, list[np.ndarray]] = {}
        for p in dataset.personas[:16]:
            for s in p.segments[:4]:
                for c in range(s.signal.shape[0]):
                    per_channel.setdefault(c, []).append(s.signal[c].flatten())
        for c, arrs in per_channel.items():
            cat = np.concatenate(arrs)
            qs = np.linspace(0, 100, self.n_buckets + 1)
            self._quantiles[c] = np.percentile(cat, qs)

    def transform(self, signal: np.ndarray, source_id: str | None = None,
                  segment_id: str | None = None) -> np.ndarray:
        out = np.zeros_like(signal, dtype=float)
        for c in range(signal.shape[0]):
            qs = self._quantiles.get(c)
            if qs is None:
                out[c] = signal[c]
                continue
            # Map each value to bucket index (preserves order)
            indices = np.searchsorted(qs, signal[c]) - 1
            indices = np.clip(indices, 0, self.n_buckets - 1)
            out[c] = indices.astype(float)
        return out

    def decode_query(self, y_query: float, query_meta: dict, source_id: str | None = None) -> float:
        # Order-preserving but does not recover values; identity decoder
        return y_query

    @property
    def hyperparameters(self) -> dict:
        return {"name": self.name, "n_bits": self.n_bits, "n_buckets": self.n_buckets,
                "key_seed": self.key_seed,
                "implementation": "quantile-bucket OPE (rank-preserving)"}
