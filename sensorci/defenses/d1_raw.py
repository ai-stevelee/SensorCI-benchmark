"""D1: No-op defense (identity). Reference upper bound for utility, lower for privacy."""

from __future__ import annotations
import numpy as np

from sensorci.defenses.base import BaseDefense
from sensorci.core.dataset import TSDataset


class RawDefense(BaseDefense):
    name = "d1_raw"
    has_decoder = True

    def fit(self, dataset: TSDataset) -> None:
        pass

    def transform(self, signal: np.ndarray, source_id: str | None = None,
                  segment_id: str | None = None) -> np.ndarray:
        return signal.copy()

    def decode_query(self, y_query: float, query_meta: dict, source_id: str | None = None) -> float:
        return y_query
