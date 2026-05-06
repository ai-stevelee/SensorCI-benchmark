"""D4: DP-SGD trained autoencoder.

Stand-in implementation: PCA + per-component DP noise (calibrated to epsilon).
Full Opacus + neural autoencoder is heavier; this stand-in captures the
'compress + DP-noise + reconstruct' behavior of DP-AE without GPU training.

For full neural implementation, see TODO(P4.5-full).
"""

from __future__ import annotations
import numpy as np

from sensorci.defenses.base import BaseDefense
from sensorci.core.dataset import TSDataset


class DPSGDAutoencoderDefense(BaseDefense):
    name = "d4_dp_ae"
    has_decoder = False

    def __init__(self, epsilon: float = 8.0, latent_ratio: float = 0.5, key_seed: int = 42):
        super().__init__()
        self.epsilon = epsilon
        self.latent_ratio = latent_ratio
        self.key_seed = key_seed
        # Per-channel projection (PCA stand-in)
        self._proj: dict[str, np.ndarray] = {}
        self._latent_dim: int = 1
        self._noise_scale: float = 0.0

    def fit(self, dataset: TSDataset) -> None:
        if not dataset.personas or not dataset.personas[0].segments:
            return
        first = dataset.personas[0].segments[0].signal
        n_ch, length = first.shape
        self._latent_dim = max(1, int(length * self.latent_ratio))
        rng = np.random.default_rng(self.key_seed)
        for c in range(n_ch):
            # Random orthogonal projection per channel (PCA stand-in)
            P = rng.standard_normal((length, self._latent_dim))
            Q, _ = np.linalg.qr(P)
            self._proj[f"ch_{c}"] = Q[:, :self._latent_dim]
        # DP noise scale: smaller epsilon → more noise (Gaussian DP analog)
        sample_std = float(np.mean([s.signal.std() for s in dataset.personas[0].segments]))
        self._noise_scale = sample_std * (1.0 / max(self.epsilon, 0.1))

    def transform(self, signal: np.ndarray, source_id: str | None = None,
                  segment_id: str | None = None) -> np.ndarray:
        rng = np.random.default_rng(hash(f"{source_id}_{segment_id}") % (2**31))
        out = np.zeros_like(signal)
        for c in range(signal.shape[0]):
            key = f"ch_{c}"
            if key not in self._proj:
                out[c] = signal[c]
                continue
            Q = self._proj[key]
            z = Q.T @ signal[c]
            z_noisy = z + rng.normal(0, self._noise_scale, z.shape)
            out[c] = Q @ z_noisy
        return out

    @property
    def hyperparameters(self) -> dict:
        return {"name": self.name, "epsilon": self.epsilon,
                "latent_ratio": self.latent_ratio,
                "implementation": "stand-in (PCA + DP-noise; full Opacus AE is TODO)"}
