"""Shared synthetic-signal generators for dataset loaders that need offline fallback.

Used when the real dataset isn't available on the target machine. Each generator
produces signals with the right shape, channel count, sampling rate, and class
structure of the target dataset, enabling pipeline tests without downloads.
"""

from __future__ import annotations
import numpy as np


def synth_multichannel(
    n_channels: int,
    length: int,
    sampling_rate_hz: float,
    rng: np.random.Generator,
    base_freqs_hz: list[float] | None = None,
    noise_std: float = 0.5,
    fault_offset: float = 0.0,
    fault_channel: int = 0,
) -> np.ndarray:
    """Generate (n_channels, length) signal with sinusoid + noise + optional fault."""
    t = np.arange(length) / max(sampling_rate_hz, 1e-6)
    base_freqs_hz = base_freqs_hz or [0.5 + 0.3 * c for c in range(n_channels)]
    signal = np.zeros((n_channels, length))
    for c in range(n_channels):
        f = base_freqs_hz[c % len(base_freqs_hz)]
        signal[c] = (
            np.sin(2 * np.pi * f * t)
            + 0.3 * np.sin(2 * np.pi * f * 3 * t)
            + rng.normal(0, noise_std, length)
        )
    if fault_offset != 0.0 and 0 <= fault_channel < n_channels:
        signal[fault_channel] += fault_offset * np.tanh((t - length / (2 * sampling_rate_hz)) / 5)
    return signal
