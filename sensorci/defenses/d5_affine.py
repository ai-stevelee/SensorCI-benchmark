"""D5: Per-source random affine masking T(x) = a*x + b.

Per Binfet et al. (Automatica 2024). Each source gets a unique (a, b) key.
Affine queries f(x) = c*x + d satisfy:
  f(T(x)) = c*(a*x+b) + d = (c*a)*x + (c*b+d)
so D_theta(y; key) recovers f(x) = (y - c*b - d)/a + (c*b+d)/a etc.
For pure linear functionals f(x) = <w, x>, decoder = (y - <w, b·1>) / a.

Limitations:
  - Per-source key shared across all segments → cross-segment linkage trivial (Theorem 3 demo)
  - Single segment statistics shifted/scaled → A0 detection moderate
"""

from __future__ import annotations
import numpy as np

from sensorci.defenses.base import BaseDefense
from sensorci.core.dataset import TSDataset


class PerSourceAffineDefense(BaseDefense):
    name = "d5_affine"
    has_decoder = True

    def __init__(self, a_mean: float = 1.0, a_std: float = 0.5,
                 b_std_relative: float = 1.0, key_seed: int = 42):
        super().__init__()
        self.a_mean = a_mean
        self.a_std = a_std
        self.b_std_relative = b_std_relative
        self.key_seed = key_seed
        self.keys: dict[str, tuple[float, float]] = {}

    def fit(self, dataset: TSDataset) -> None:
        rng = np.random.default_rng(self.key_seed)
        for p in dataset.personas:
            # Estimate per-source signal magnitude for b prior
            sig_std_estimates = [s.signal.std() for s in p.segments[:3]] if p.segments else [1.0]
            sig_std = float(np.mean(sig_std_estimates))
            a = float(rng.normal(self.a_mean, self.a_std))
            # Avoid near-zero a (would destroy utility)
            while abs(a) < 0.1:
                a = float(rng.normal(self.a_mean, self.a_std))
            b = float(rng.normal(0.0, self.b_std_relative * sig_std))
            self.keys[p.persona_id] = (a, b)

    def transform(self, signal: np.ndarray, source_id: str | None = None,
                  segment_id: str | None = None) -> np.ndarray:
        if source_id is None or source_id not in self.keys:
            # Fallback: random fresh key (less safe but always works)
            rng = np.random.default_rng(hash(source_id or "_") % (2**31))
            a = float(rng.normal(self.a_mean, self.a_std)) or 1.0
            b = float(rng.normal(0.0, signal.std()))
        else:
            a, b = self.keys[source_id]
        return a * signal + b

    def decode_query(self, y_query: float, query_meta: dict, source_id: str | None = None) -> float:
        """For an affine functional f(x) = sum(w_i * x_i) + d:
            f(T(x)) = a * sum(w_i * x_i) + b * sum(w_i) + d
            so f(x) = (f(T(x)) - b*sum(w) - d)/a + (b*sum(w) + d)/a effectively
            simplest: f(x) = (y - b * sum_w) / a, assuming d = 0.
        """
        if source_id is None or source_id not in self.keys:
            return y_query
        a, b = self.keys[source_id]
        sum_w = query_meta.get("sum_w", 0.0)
        return (y_query - b * sum_w) / a

    @property
    def hyperparameters(self) -> dict:
        return {
            "name": self.name,
            "a_mean": self.a_mean,
            "a_std": self.a_std,
            "b_std_relative": self.b_std_relative,
            "key_seed": self.key_seed,
            "n_keys": len(self.keys),
        }
