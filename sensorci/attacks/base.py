"""BaseAttack ABC."""

from __future__ import annotations
from abc import ABC, abstractmethod
from sensorci.core.dataset import TSDataset
from sensorci.defenses.base import BaseDefense
from sensorci.core.result import AttackResult


class BaseAttack(ABC):
    """Abstract base for attacks. Each attack measures one or more properties (P1-P4)."""

    name: str = "base"
    target_property: str = ""  # one of "P1", "P2", "P3-CI", "P4"

    @abstractmethod
    def run(
        self,
        defense: BaseDefense,
        dataset: TSDataset,
        n_samples: int = 100,
        seed: int = 42,
        **kwargs,
    ) -> AttackResult:
        """Run the attack against `defense` on `dataset`.

        Returns AttackResult with .metric_value populated.
        """
        ...

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name} target={self.target_property}>"
