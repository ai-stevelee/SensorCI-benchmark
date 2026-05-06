"""D8: DoppelGANger synthetic replacement (stand-in).

Real DoppelGANger (Lin et al. IMC 2020) requires GAN training on the dataset.
This stand-in samples from a per-class Gaussian mixture fit to training data,
preserving marginal statistics but losing pointwise correspondence with X
(satisfies P2 partially, breaks P1 entirely as documented).

For full implementation, install gretel-synthetics or DoppelGANger reference.
"""

from __future__ import annotations
import numpy as np

from sensorci.defenses.base import BaseDefense
from sensorci.core.dataset import TSDataset


class DoppelGANgerDefense(BaseDefense):
    name = "d8_doppelganger"
    has_decoder = False  # synthetic data has no pointwise decoder

    def __init__(self, key_seed: int = 42):
        super().__init__()
        self.key_seed = key_seed
        # Per-class per-channel Gaussian parameters
        self._class_stats: dict[str, dict[int, tuple[float, float]]] = {}
        self._fallback_stats: dict[int, tuple[float, float]] = {}

    def fit(self, dataset: TSDataset) -> None:
        # Collect signals by class
        by_class: dict[str, dict[int, list[float]]] = {}
        all_per_channel: dict[int, list[float]] = {}
        for p in dataset.personas:
            for s in p.segments:
                cls = s.class_label or "unknown"
                by_class.setdefault(cls, {})
                for c in range(s.signal.shape[0]):
                    by_class[cls].setdefault(c, []).extend(s.signal[c].tolist())
                    all_per_channel.setdefault(c, []).extend(s.signal[c].tolist())
        # Compute class+channel stats
        for cls, ch_dict in by_class.items():
            self._class_stats[cls] = {}
            for c, vals in ch_dict.items():
                arr = np.array(vals)
                self._class_stats[cls][c] = (float(arr.mean()), float(arr.std()))
        # Fallback per-channel stats
        for c, vals in all_per_channel.items():
            arr = np.array(vals)
            self._fallback_stats[c] = (float(arr.mean()), float(arr.std()))

    def transform(self, signal: np.ndarray, source_id: str | None = None,
                  segment_id: str | None = None,
                  class_label: str | None = None) -> np.ndarray:
        rng = np.random.default_rng(hash(f"{source_id}_{segment_id}") % (2**31))
        out = np.zeros_like(signal)
        n_ch, length = signal.shape
        cls = class_label  # if not provided, use fallback
        stats_for_cls = self._class_stats.get(cls or "", self._fallback_stats)
        for c in range(n_ch):
            mu, sd = stats_for_cls.get(c, self._fallback_stats.get(c, (0.0, 1.0)))
            # Generate AR(1) for some temporal structure
            phi = 0.85
            x = np.zeros(length)
            x[0] = rng.normal(mu, sd)
            for t in range(1, length):
                x[t] = mu + phi * (x[t - 1] - mu) + rng.normal(0, sd * np.sqrt(1 - phi ** 2))
            out[c] = x
        return out

    @property
    def hyperparameters(self) -> dict:
        return {"name": self.name, "key_seed": self.key_seed,
                "implementation": "stand-in (per-class Gaussian AR(1); full GAN is TODO)"}
