"""compact_numeric encoding: z-normalized floats, comma-separated."""

from __future__ import annotations
import numpy as np


def encode_compact_numeric(signal: np.ndarray, n_points: int = 128, decimals: int = 3,
                           channels: list[str] | None = None, **_kw) -> str:
    """Downsample to n_points per channel, z-normalize, format as CSV.

    Args:
      signal: shape (n_channels, length)
      n_points: target points per channel after downsample
      decimals: float precision in output
    """
    if signal.ndim != 2:
        raise ValueError(f"signal must be 2D, got {signal.shape}")
    n_ch, length = signal.shape
    channels = channels or [f"ch{i}" for i in range(n_ch)]

    lines = []
    for c in range(n_ch):
        x = signal[c]
        # Downsample with linear resample
        if length != n_points:
            xs = np.linspace(0, length - 1, n_points)
            x_ds = np.interp(xs, np.arange(length), x)
        else:
            x_ds = x
        # z-normalize
        mu = np.mean(x_ds)
        sd = np.std(x_ds)
        x_norm = (x_ds - mu) / (sd + 1e-9)
        line = f"{channels[c]}: " + ",".join(f"{v:.{decimals}f}" for v in x_norm)
        lines.append(line)
    return "\n".join(lines)
