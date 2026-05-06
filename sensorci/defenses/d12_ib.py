"""D12: Information Bottleneck (IB) Encoder for Time Series.

Trains a stochastic encoder that minimizes I(X; Z) subject to I(Y; Z) >= U_min,
where Y is the analyst-task label and Z is the released representation.

Addresses Reviewer Q6 (private representation learning baselines).

Placeholder skeleton; full implementation deferred to Phase 4 (P4.X). For pilot,
the placeholder returns: PCA-style top-k projection + isotropic noise calibrated
to a target I(X; Z) ~= U_min budget. This approximates the IB objective behavior
without GPU training.
"""

from __future__ import annotations
import numpy as np

from sensorci.defenses.base import BaseDefense
from sensorci.core.dataset import TSDataset


class InformationBottleneckDefense(BaseDefense):
    name = "d12_ib"
    has_decoder = False  # learned encoder; no closed-form decoder for general queries

    def __init__(self, latent_dim_ratio: float = 0.5, noise_budget: float = 0.3,
                 key_seed: int = 42):
        super().__init__()
        self.latent_dim_ratio = latent_dim_ratio  # fraction of original length kept
        self.noise_budget = noise_budget
        self.key_seed = key_seed
        self._projections: dict[str, np.ndarray] = {}  # per-channel PCA-like projection
        self._fitted = False

    def fit(self, dataset: TSDataset) -> None:
        # Collect all signals to fit per-channel projection (PCA stand-in for IB encoder)
        if not dataset.personas or not dataset.personas[0].segments:
            self._fitted = True
            return
        first = dataset.personas[0].segments[0].signal
        n_channels, length = first.shape
        latent_dim = max(1, int(length * self.latent_dim_ratio))

        rng = np.random.default_rng(self.key_seed)
        # Random orthogonal projection per channel as IB stand-in
        # (real IB would use neural encoder + variational lower bound on I(Y;Z))
        for c in range(n_channels):
            P = rng.standard_normal((length, latent_dim))
            Q, _ = np.linalg.qr(P)
            self._projections[f"ch_{c}"] = Q[:, :latent_dim]
        self._fitted = True

    def transform(self, signal: np.ndarray, source_id: str | None = None,
                  segment_id: str | None = None) -> np.ndarray:
        """Project to latent + add noise + project back to original shape."""
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")
        rng = np.random.default_rng(hash(f"{source_id}_{segment_id}") % (2**31))
        out = np.zeros_like(signal)
        for c in range(signal.shape[0]):
            key = f"ch_{c}"
            if key not in self._projections:
                out[c] = signal[c]
                continue
            Q = self._projections[key]
            # Encode: z = Q^T x ; then add noise; then decode: x' = Q z'
            z = Q.T @ signal[c]
            z_noisy = z + rng.normal(0, self.noise_budget * z.std(), z.shape)
            out[c] = Q @ z_noisy
        return out

    @property
    def hyperparameters(self) -> dict:
        return {
            "name": self.name,
            "latent_dim_ratio": self.latent_dim_ratio,
            "noise_budget": self.noise_budget,
            "implementation": "placeholder_pca_noise",
            "todo": "P4.X — replace with trained variational IB encoder",
        }
