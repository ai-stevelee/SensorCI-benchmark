"""raw_stats encoding: time + frequency-domain summary statistics as JSON-like text."""

from __future__ import annotations
import json
import numpy as np


def encode_raw_stats(signal: np.ndarray, sampling_rate_hz: float = 1.0,
                     channels: list[str] | None = None) -> str:
    """Encode signal as a compact JSON-like dict of summary statistics.

    Args:
      signal: shape (n_channels, length)
      sampling_rate_hz: for FFT frequency labels
      channels: optional channel names

    Returns:
      Multi-line text with per-channel stats + spectral peaks.
    """
    if signal.ndim != 2:
        raise ValueError(f"signal must be 2D, got {signal.shape}")
    n_ch, length = signal.shape
    channels = channels or [f"ch{i}" for i in range(n_ch)]

    out: dict[str, dict] = {}
    for c in range(n_ch):
        x = signal[c]
        x_mean = float(np.mean(x))
        x_std = float(np.std(x))
        x_norm = (x - x_mean) / (x_std + 1e-9)
        # FFT magnitude top 5 frequencies
        mag = np.abs(np.fft.rfft(x_norm))
        freqs = np.fft.rfftfreq(length, d=1.0 / sampling_rate_hz)
        top_idx = np.argsort(mag)[-5:][::-1]
        top = [
            (round(float(freqs[i]), 3), round(float(mag[i]), 3))
            for i in top_idx
        ]
        out[channels[c]] = {
            "mean":     round(x_mean, 4),
            "std":      round(x_std, 4),
            "min":      round(float(np.min(x)), 4),
            "max":      round(float(np.max(x)), 4),
            "peak":     round(float(np.max(np.abs(x))), 4),
            "rms":      round(float(np.sqrt(np.mean(x ** 2))), 4),
            "kurtosis": round(_kurtosis(x), 4),
            "skewness": round(_skewness(x), 4),
            "fft_top5_freq_mag": top,
        }
    return json.dumps(out, indent=2)


def _kurtosis(x: np.ndarray) -> float:
    mu = np.mean(x)
    sd = np.std(x)
    if sd < 1e-12:
        return 0.0
    return float(np.mean(((x - mu) / sd) ** 4) - 3.0)


def _skewness(x: np.ndarray) -> float:
    mu = np.mean(x)
    sd = np.std(x)
    if sd < 1e-12:
        return 0.0
    return float(np.mean(((x - mu) / sd) ** 3))
