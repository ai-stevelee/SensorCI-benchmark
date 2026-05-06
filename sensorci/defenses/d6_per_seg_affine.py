"""D6: Per-segment random affine masking.

Each segment gets its own (a, b) key. Cross-segment linkage protection improves
vs D5 (per-source affine), but cross-segment algebraic queries break.
"""

from __future__ import annotations
import numpy as np

from sensorci.defenses.base import BaseDefense
from sensorci.core.dataset import TSDataset


class PerSegmentAffineDefense(BaseDefense):
    name = "d6_per_seg_affine"
    has_decoder = True

    def __init__(self, a_mean: float = 1.0, a_std: float = 0.5,
                 b_std_relative: float = 1.0, key_seed: int = 42):
        super().__init__()
        self.a_mean = a_mean
        self.a_std = a_std
        self.b_std_relative = b_std_relative
        self.key_seed = key_seed
        self._signal_std_estimate: float = 1.0

    def fit(self, dataset: TSDataset) -> None:
        stds = []
        for p in dataset.personas[:8]:
            for s in p.segments[:2]:
                stds.append(float(s.signal.std()))
        self._signal_std_estimate = float(np.mean(stds)) if stds else 1.0

    def _get_key(self, source_id: str | None, segment_id: str | None) -> tuple[float, float]:
        seed_str = f"{self.key_seed}_{source_id}_{segment_id}"
        rng = np.random.default_rng(abs(hash(seed_str)) % (2**31))
        a = float(rng.normal(self.a_mean, self.a_std))
        while abs(a) < 0.1:
            a = float(rng.normal(self.a_mean, self.a_std))
        b = float(rng.normal(0.0, self.b_std_relative * self._signal_std_estimate))
        return a, b

    def transform(self, signal: np.ndarray, source_id: str | None = None,
                  segment_id: str | None = None) -> np.ndarray:
        a, b = self._get_key(source_id, segment_id)
        return a * signal + b

    def decode_query(self, y_query: float, query_meta: dict, source_id: str | None = None) -> float:
        # Without segment_id we cannot recover; fall back to identity
        seg_id = query_meta.get("segment_id")
        if seg_id is None:
            return y_query
        a, b = self._get_key(source_id, seg_id)
        sum_w = query_meta.get("sum_w", 0.0)
        if a == 0:
            return y_query
        return (y_query - b * sum_w) / a

    @property
    def hyperparameters(self) -> dict:
        return {"name": self.name, "a_mean": self.a_mean, "a_std": self.a_std,
                "b_std_relative": self.b_std_relative, "key_seed": self.key_seed}
