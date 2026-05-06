"""Hyperparameter sweep utility.

Each attack implements an `evaluate_hyperparameters(hp_dict, val_split)` method
that returns a scalar. HyperparameterSearch picks the best on validation set,
then reports test set performance with the chosen HPs.

Critical for paper-quality: hyperparameters must be tuned on val, NOT test.
"""

from __future__ import annotations
import itertools
import logging
from typing import Callable, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class HPSweepResult:
    best_hp: dict[str, Any]
    best_val_score: float
    all_scores: list[tuple[dict, float]]
    n_configs_tried: int


class HyperparameterSweep:
    """Grid search hyperparameter sweep on validation set.

    Usage:
      sweep = HyperparameterSweep(grid={"alpha": [0.1, 1.0, 10.0],
                                        "n_estimators": [50, 100]})
      result = sweep.run(train_fn, eval_fn, train_data, val_data)
    """

    def __init__(self, grid: dict[str, list], maximize: bool = True):
        self.grid = grid
        self.maximize = maximize

    def expand(self) -> list[dict]:
        """Expand grid into list of HP dicts."""
        keys = list(self.grid.keys())
        values_lists = [self.grid[k] for k in keys]
        configs = []
        for combo in itertools.product(*values_lists):
            configs.append(dict(zip(keys, combo)))
        return configs

    def run(
        self,
        train_eval_fn: Callable[[dict, Any, Any], float],
        train_data: Any,
        val_data: Any,
    ) -> HPSweepResult:
        """Run grid search.

        train_eval_fn(hp_dict, train_data, val_data) -> validation score.
        """
        configs = self.expand()
        logger.info(f"HP sweep: {len(configs)} configs over {list(self.grid.keys())}")
        scored = []
        for hp in configs:
            try:
                score = train_eval_fn(hp, train_data, val_data)
            except Exception as e:
                logger.warning(f"HP config {hp} failed: {e}")
                score = float("-inf") if self.maximize else float("inf")
            scored.append((hp, score))

        if self.maximize:
            best_hp, best_score = max(scored, key=lambda x: x[1])
        else:
            best_hp, best_score = min(scored, key=lambda x: x[1])

        return HPSweepResult(
            best_hp=best_hp,
            best_val_score=float(best_score),
            all_scores=scored,
            n_configs_tried=len(scored),
        )
