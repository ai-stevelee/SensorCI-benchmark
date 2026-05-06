"""BaseDatasetLoader ABC."""

from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
import os

from sensorci.core.dataset import TSDataset


class BaseDatasetLoader(ABC):
    """Abstract loader. Subclasses implement download + parse + persona construction."""

    name: str = "base"

    def __init__(self, data_dir: Path | str | None = None, **kw):
        if data_dir is None:
            data_dir = os.environ.get("SENSORCI_DATA_DIR", "./data")
        self.data_dir = Path(data_dir) / self.name
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.kwargs = kw

    @abstractmethod
    def load(self) -> TSDataset:
        """Return a fully constructed TSDataset."""
        ...
