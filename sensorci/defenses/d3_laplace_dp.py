"""D3: Laplace Differential Privacy mechanism."""

from __future__ import annotations
import numpy as np

from sensorci.defenses.base import BaseDefense
from sensorci.core.dataset import TSDataset


class LaplaceDPDefense(BaseDefense):
    name = "d3_laplace_dp"
    has_decoder = False

    def __init__(self, b: float = 0.1, key_seed: int = 42):
        super().__init__()
        self.b = b  # scale parameter (Laplace noise)
        self.key_seed = key_seed
        self._signal_std_estimate: float = 1.0

    def fit(self, dataset: TSDataset) -> None:
        stds = []
        for p in dataset.personas[:8]:
            for s in p.segments[:2]:
                stds.append(float(s.signal.std()))
        self._signal_std_estimate = float(np.mean(stds)) if stds else 1.0

    def transform(self, signal: np.ndarray, source_id: str | None = None,
                  segment_id: str | None = None) -> np.ndarray:
        rng = np.random.default_rng(hash(f"{source_id}_{segment_id}") % (2**31))
        noise_scale = self.b * self._signal_std_estimate
        noise = rng.laplace(0, noise_scale, signal.shape)
        return signal + noise

    @property
    def hyperparameters(self) -> dict:
        return {"name": self.name, "b": self.b, "key_seed": self.key_seed}
