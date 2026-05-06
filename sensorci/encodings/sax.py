"""SAX encoding: Symbolic Aggregate approXimation (Lin et al. 2003).

Converts a continuous signal into a string of letters from a small alphabet.
"""

from __future__ import annotations
import numpy as np
from scipy.stats import norm


def encode_sax(signal: np.ndarray, n_segments: int = 64, alphabet_size: int = 8,
               channels: list[str] | None = None, **_kw) -> str:
    """SAX representation for each channel of `signal`.

    Args:
      signal: shape (n_channels, length)
      n_segments: number of PAA segments (output length per channel)
      alphabet_size: number of letters (uses Gaussian breakpoints)

    Returns:
      Multi-line text: "<channel>: <symbol_string>" per channel.
    """
    if signal.ndim != 2:
        raise ValueError(f"signal must be 2D, got {signal.shape}")
    n_ch, length = signal.shape
    channels = channels or [f"ch{i}" for i in range(n_ch)]

    breakpoints = _gaussian_breakpoints(alphabet_size)
    alphabet = "abcdefghijklmnopqrstuvwxyz"[:alphabet_size]

    lines = []
    for c in range(n_ch):
        x = signal[c]
        # z-normalize
        mu = np.mean(x)
        sd = np.std(x)
        x_norm = (x - mu) / (sd + 1e-9)
        # PAA
        paa = _paa(x_norm, n_segments)
        # Symbolize
        symbols = "".join(alphabet[_symbolize(v, breakpoints)] for v in paa)
        lines.append(f"{channels[c]}: {symbols}")
    return "\n".join(lines)


def _paa(x: np.ndarray, n_segments: int) -> np.ndarray:
    """Piecewise Aggregate Approximation."""
    length = len(x)
    if length == n_segments:
        return x
    if length % n_segments == 0:
        return x.reshape(n_segments, -1).mean(axis=1)
    # General case via interpolation onto n_segments equal-length bins
    bin_edges = np.linspace(0, length, n_segments + 1)
    out = np.empty(n_segments)
    for i in range(n_segments):
        lo = int(np.floor(bin_edges[i]))
        hi = int(np.ceil(bin_edges[i + 1]))
        out[i] = x[lo:hi].mean() if hi > lo else x[lo]
    return out


def _gaussian_breakpoints(alphabet_size: int) -> np.ndarray:
    """Equiprobable breakpoints under N(0,1)."""
    return norm.ppf(np.linspace(0, 1, alphabet_size + 1))[1:-1]


def _symbolize(v: float, breakpoints: np.ndarray) -> int:
    """Return symbol index in [0, len(breakpoints)]."""
    for i, b in enumerate(breakpoints):
        if v < b:
            return i
    return len(breakpoints)
