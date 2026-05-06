"""A2: Linear (Ridge) reconstruction attack — paper-quality.

Persona-stratified train/val/test split + alpha tuning on validation set.
"""

from __future__ import annotations
import logging
import time
import numpy as np

from sensorci.attacks.base import BaseAttack
from sensorci.core.dataset import TSDataset
from sensorci.core.result import AttackResult
from sensorci.core.splits import PersonaStratifiedSplitter
from sensorci.core.hp_search import HyperparameterSweep
from sensorci.defenses.base import BaseDefense

logger = logging.getLogger(__name__)


class LinearReconstructionAttack(BaseAttack):
    name = "a2_linear_recon"
    target_property = "P4"

    def __init__(
        self,
        n_known_pairs: int = 50,
        ridge_alpha: float | None = None,
        alpha_grid: list[float] | None = None,
    ):
        super().__init__()
        self.n_known_pairs = n_known_pairs
        self.ridge_alpha = ridge_alpha
        self.alpha_grid = alpha_grid or [0.01, 0.1, 1.0, 10.0, 100.0]

    def run(
        self,
        defense: BaseDefense,
        dataset: TSDataset,
        n_samples: int = 100,
        seed: int = 42,
        **kwargs,
    ) -> AttackResult:
        from sklearn.linear_model import Ridge

        try:
            splitter = PersonaStratifiedSplitter(0.6, 0.2, 0.2, seed=seed)
            split = splitter.split(dataset)
        except ValueError:
            return self._empty(dataset, defense, seed)

        train_pairs = split.train_segments[: self.n_known_pairs]
        val_pairs = split.val_segments[: max(2, n_samples // 4)]
        test_pairs = split.test_segments[: n_samples]
        if not train_pairs or not test_pairs:
            return self._empty(dataset, defense, seed)

        L = min(s.signal.size for _p, s in train_pairs + val_pairs + test_pairs)

        def to_arrays(pairs):
            TX_list = []
            X_list = []
            for p, s in pairs:
                tx = defense.transform(s.signal, source_id=p.persona_id,
                                        segment_id=s.segment_id).flatten()[:L]
                x = s.signal.flatten()[:L]
                TX_list.append(tx)
                X_list.append(x)
            return np.array(TX_list), np.array(X_list)

        TX_train, X_train = to_arrays(train_pairs)
        TX_val, X_val = to_arrays(val_pairs)
        TX_test, X_test = to_arrays(test_pairs)

        if len(TX_train) < 2 or TX_train.shape[1] == 0:
            return self._empty(dataset, defense, seed)

        t0 = time.time()
        used_hp_search = False

        if self.ridge_alpha is None and len(TX_val) >= 2:
            def train_eval(hp, train_data, val_data):
                tx_tr, x_tr = train_data
                tx_va, x_va = val_data
                try:
                    r = Ridge(alpha=hp["alpha"])
                    r.fit(tx_tr, x_tr)
                    x_pred = r.predict(tx_va)
                    nrmse = float(np.mean([
                        np.linalg.norm(x_pred[i] - x_va[i]) /
                        (np.linalg.norm(x_va[i]) + 1e-9)
                        for i in range(len(x_va))
                    ]))
                    return -nrmse  # maximize -NRMSE = attacker minimizes NRMSE
                except Exception:
                    return -float("inf")

            sweep = HyperparameterSweep(
                grid={"alpha": self.alpha_grid}, maximize=True,
            )
            result = sweep.run(train_eval, (TX_train, X_train), (TX_val, X_val))
            best_alpha = result.best_hp["alpha"]
            used_hp_search = True
        else:
            best_alpha = self.ridge_alpha if self.ridge_alpha is not None else 1.0

        try:
            ridge = Ridge(alpha=best_alpha)
            ridge.fit(TX_train, X_train)
        except Exception:
            return self._empty(dataset, defense, seed)

        nrmse_list = []
        for i in range(len(TX_test)):
            x_hat = ridge.predict(TX_test[i].reshape(1, -1))[0]
            num = float(np.linalg.norm(x_hat - X_test[i]))
            den = float(np.linalg.norm(X_test[i])) + 1e-9
            nrmse_list.append(num / den)
        nrmse = float(np.mean(nrmse_list)) if nrmse_list else 1.0
        nrmse_capped = min(1.0, nrmse)
        elapsed = time.time() - t0

        return AttackResult(
            cell_id=f"{dataset.name}_{defense.name}_{self.name}_seed{seed}",
            dataset=dataset.name, defense=defense.name,
            attack=self.name, seed=seed,
            metric_name="nrmse_a2", metric_value=nrmse_capped,
            subscores={"raw_nrmse": float(nrmse),
                       "n_train": len(train_pairs),
                       "best_alpha": float(best_alpha)},
            n_samples=len(test_pairs),
            wall_seconds=elapsed,
            extra={"split": "persona_stratified",
                   "hp_search": used_hp_search,
                   "alpha_grid": self.alpha_grid},
        )

    def _empty(self, dataset, defense, seed):
        return AttackResult(
            cell_id=f"{dataset.name}_{defense.name}_{self.name}_seed{seed}",
            dataset=dataset.name, defense=defense.name,
            attack=self.name, seed=seed,
            metric_name="nrmse_a2", metric_value=1.0,
            subscores={"best_alpha": 1.0, "n_train": 0, "raw_nrmse": 1.0},
            n_samples=0,
        )
