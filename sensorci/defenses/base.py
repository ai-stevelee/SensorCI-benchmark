"""BaseDefense ABC."""

from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np

from sensorci.core.dataset import TSDataset


class BaseDefense(ABC):
    """Abstract base for privacy-preserving transformations T_theta : R^T -> R^T."""

    name: str = "base"
    has_decoder: bool = False  # whether D_theta is implementable

    @abstractmethod
    def fit(self, dataset: TSDataset) -> None:
        """Initialize defense state given the full dataset (e.g., per-source keys)."""
        ...

    @abstractmethod
    def transform(self, signal: np.ndarray, source_id: str | None = None,
                  segment_id: str | None = None) -> np.ndarray:
        """Apply T_theta to a single signal (n_channels, length)."""
        ...

    def decode_query(self, y_query: float, query_meta: dict, source_id: str | None = None) -> float:
        """Inverse-transform a query result f(T(X)) back to f(X) when possible.

        Default: identity (P1 score will be conservatively penalized).
        Override in subclasses with implementable decoders.
        """
        return y_query

    @property
    def hyperparameters(self) -> dict:
        return {"name": self.name}

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name}>"
