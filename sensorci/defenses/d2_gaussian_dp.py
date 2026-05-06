"""D2: Gaussian Differential Privacy mechanism."""

from __future__ import annotations
import numpy as np

from sensorci.defenses.base import BaseDefense
from sensorci.core.dataset import TSDataset


class GaussianDPDefense(BaseDefense):
    name = "d2_gaussian_dp"
    has_decoder = False  # noise is irrecoverable

    def __init__(self, sigma: float = 0.1, key_seed: int = 42):
        super().__init__()
        self.sigma = sigma
        self.key_seed = key_seed
        self._signal_std_estimate: float = 1.0

    def fit(self, dataset: TSDataset) -> None:
        # Estimate per-signal std for noise calibration
        stds = []
        for p in dataset.personas[:8]:
            for s in p.segments[:2]:
                stds.append(float(s.signal.std()))
        self._signal_std_estimate = float(np.mean(stds)) if stds else 1.0

    def transform(self, signal: np.ndarray, source_id: str | None = None,
                  segment_id: str | None = None) -> np.ndarray:
        rng = np.random.default_rng(hash(f"{source_id}_{segment_id}") % (2**31))
        noise_scale = self.sigma * self._signal_std_estimate
        return signal + rng.normal(0, noise_scale, signal.shape)

    @property
    def hyperparameters(self) -> dict:
        return {"name": self.name, "sigma": self.sigma, "key_seed": self.key_seed}
