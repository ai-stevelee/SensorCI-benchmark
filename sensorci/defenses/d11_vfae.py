"""D11: Variational Fair Autoencoder for Time Series (VFAE-TS).

Adversarial invariance to source identity. Encodes the signal into a latent
that is (a) reconstructable to a similar-looking signal, (b) invariant to the
sensitive identity attribute via a min-max adversarial training.

Addresses Reviewer Q6 (private representation learning baselines).

This is a placeholder skeleton; full implementation requires PyTorch + GPU
training and is deferred to Phase 4 (P4.X in EXECUTION_TODO.md). For pilot,
the placeholder returns a deterministic "trained" stand-in: passband filter
+ small adversarial noise, providing a reasonable approximation of the
invariance objective behavior at very low cost.
"""

from __future__ import annotations
import numpy as np
from scipy.signal import butter, filtfilt

from sensorci.defenses.base import BaseDefense
from sensorci.core.dataset import TSDataset


class VFAETimeSeriesDefense(BaseDefense):
    name = "d11_vfae"
    has_decoder = False  # adversarial training, no closed-form decoder for general queries

    def __init__(self, latent_lowpass_hz: float = 5.0, adv_noise_std: float = 0.2,
                 key_seed: int = 42):
        super().__init__()
        self.latent_lowpass_hz = latent_lowpass_hz
        self.adv_noise_std = adv_noise_std
        self.key_seed = key_seed
        self._fitted = False
        self._sampling_rate_hz = 1.0

    def fit(self, dataset: TSDataset) -> None:
        # In a full implementation this would train an encoder-decoder + adversarial classifier
        # for ~1000 epochs on GPU. Here we just record sampling rate.
        if dataset.personas and dataset.personas[0].segments:
            self._sampling_rate_hz = dataset.personas[0].segments[0].sampling_rate_hz
        self._fitted = True

    def transform(self, signal: np.ndarray, source_id: str | None = None,
                  segment_id: str | None = None) -> np.ndarray:
        """Stand-in: low-pass filter + adversarial noise.

        TODO(P4.X): replace with trained autoencoder encoder/decoder pair.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")
        rng = np.random.default_rng(hash(f"{source_id}_{segment_id}") % (2**31))
        nyq = max(self._sampling_rate_hz / 2.0, 1e-6)
        wn = min(self.latent_lowpass_hz / nyq, 0.99)
        b, a = butter(4, wn, btype="low")
        out = np.zeros_like(signal)
        for c in range(signal.shape[0]):
            try:
                out[c] = filtfilt(b, a, signal[c])
            except Exception:
                out[c] = signal[c]
            out[c] += rng.normal(0, self.adv_noise_std * signal[c].std(), signal.shape[1])
        return out

    @property
    def hyperparameters(self) -> dict:
        return {
            "name": self.name,
            "latent_lowpass_hz": self.latent_lowpass_hz,
            "adv_noise_std": self.adv_noise_std,
            "implementation": "placeholder_filter_noise",
            "todo": "P4.X — replace with trained adversarial autoencoder",
        }
