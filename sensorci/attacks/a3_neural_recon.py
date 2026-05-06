"""A3: Neural reconstruction attack — paper-quality.

Two modes:
  - "neural" (default if torch available): 1D-UNet trained on persona-stratified
    train split, NRMSE reported on test.
  - "kernel" (fallback): kernel ridge with RBF.
"""

from __future__ import annotations
import logging
import time
import numpy as np

from sensorci.attacks.base import BaseAttack
from sensorci.core.dataset import TSDataset
from sensorci.core.result import AttackResult
from sensorci.core.splits import PersonaStratifiedSplitter
from sensorci.defenses.base import BaseDefense

logger = logging.getLogger(__name__)


class NeuralReconstructionAttack(BaseAttack):
    name = "a3_neural_recon"
    target_property = "P4"

    def __init__(
        self,
        arch: str = "auto",
        epochs: int = 30,
        base_filters: int = 32,
        depth: int = 4,
        n_known_pairs: int = 200,
        kernel_alpha: float = 0.1,
        kernel_gamma: float = 0.01,
    ):
        super().__init__()
        self.arch = arch
        self.epochs = epochs
        self.base_filters = base_filters
        self.depth = depth
        self.n_known_pairs = n_known_pairs
        self.kernel_alpha = kernel_alpha
        self.kernel_gamma = kernel_gamma

    def run(
        self,
        defense: BaseDefense,
        dataset: TSDataset,
        n_samples: int = 100,
        seed: int = 42,
        **kwargs,
    ) -> AttackResult:
        arch = self.arch
        if arch == "auto":
            try:
                import torch  # noqa
                arch = "neural"
            except ImportError:
                arch = "kernel"

        if arch == "neural":
            return self._run_neural(defense, dataset, n_samples, seed)
        return self._run_kernel(defense, dataset, n_samples, seed)

    def _run_neural(self, defense, dataset, n_samples, seed):
        from sensorci.attacks._neural_models import make_unet1d, train_reconstruction
        try:
            import torch
        except ImportError:
            return self._run_kernel(defense, dataset, n_samples, seed)

        try:
            split = PersonaStratifiedSplitter(0.6, 0.2, 0.2, seed=seed).split(dataset)
        except ValueError:
            return self._empty(dataset, defense, seed)

        train_pairs = split.train_segments[: self.n_known_pairs]
        test_pairs = split.test_segments[: n_samples]
        if not train_pairs or not test_pairs:
            return self._empty(dataset, defense, seed)

        L_raw = min(s.signal.shape[1] for _p, s in train_pairs + test_pairs)
        L = (L_raw // (2 ** self.depth)) * (2 ** self.depth)
        if L < 16:
            return self._run_kernel(defense, dataset, n_samples, seed)
        n_ch = train_pairs[0][1].signal.shape[0]

        def to_arrays(pairs):
            TX_list = []
            X_list = []
            for p, s in pairs:
                tx = defense.transform(s.signal, source_id=p.persona_id,
                                        segment_id=s.segment_id)[:, :L]
                x = s.signal[:, :L]
                TX_list.append(tx)
                X_list.append(x)
            return np.array(TX_list), np.array(X_list)

        TX_train, X_train = to_arrays(train_pairs)
        TX_test, X_test = to_arrays(test_pairs)

        t0 = time.time()
        model = make_unet1d(n_channels=n_ch, base_filters=self.base_filters,
                             depth=self.depth)
        if model is None:
            return self._run_kernel(defense, dataset, n_samples, seed)

        try:
            model = train_reconstruction(
                model, TX_train, X_train,
                epochs=self.epochs, batch_size=4, lr=1e-3,
            )
            device = next(model.parameters()).device
            model.eval()
            nrmse_list = []
            with torch.no_grad():
                for i in range(len(TX_test)):
                    tx_t = torch.from_numpy(TX_test[i:i+1]).float().to(device)
                    x_pred = model(tx_t).cpu().numpy()[0]
                    num = float(np.linalg.norm(x_pred - X_test[i]))
                    den = float(np.linalg.norm(X_test[i])) + 1e-9
                    nrmse_list.append(num / den)
        except Exception as e:
            logger.warning(f"Neural A3 failed: {e}; fallback to kernel")
            return self._run_kernel(defense, dataset, n_samples, seed)

        nrmse = float(np.mean(nrmse_list)) if nrmse_list else 1.0
        nrmse_capped = min(1.0, nrmse)
        elapsed = time.time() - t0

        return AttackResult(
            cell_id=f"{dataset.name}_{defense.name}_{self.name}_seed{seed}",
            dataset=dataset.name, defense=defense.name,
            attack=self.name, seed=seed,
            metric_name="nrmse_a3", metric_value=nrmse_capped,
            subscores={"raw_nrmse": float(nrmse), "n_train": len(train_pairs)},
            n_samples=len(test_pairs),
            wall_seconds=elapsed,
            extra={"arch": "neural_unet1d", "epochs": self.epochs,
                   "depth": self.depth, "split": "persona_stratified"},
        )

    def _run_kernel(self, defense, dataset, n_samples, seed):
        from sklearn.kernel_ridge import KernelRidge
        from sklearn.preprocessing import StandardScaler

        try:
            split = PersonaStratifiedSplitter(0.6, 0.2, 0.2, seed=seed).split(dataset)
        except ValueError:
            return self._empty(dataset, defense, seed)

        train_pairs = split.train_segments[: self.n_known_pairs]
        test_pairs = split.test_segments[: n_samples]
        if not train_pairs or not test_pairs:
            return self._empty(dataset, defense, seed)

        L = min(p[1].signal.size for p in train_pairs + test_pairs)

        def flat(s):
            return s.signal.flatten()[:L]

        TX_train = np.array([
            defense.transform(s.signal, source_id=p.persona_id,
                              segment_id=s.segment_id).flatten()[:L]
            for p, s in train_pairs
        ])
        X_train = np.array([flat(s) for _p, s in train_pairs])

        try:
            scaler = StandardScaler().fit(TX_train)
            TX_train_s = scaler.transform(TX_train)
            kr = KernelRidge(alpha=self.kernel_alpha, kernel="rbf",
                              gamma=self.kernel_gamma)
            kr.fit(TX_train_s, X_train)
        except Exception:
            return self._empty(dataset, defense, seed)

        nrmse_list = []
        for p, s in test_pairs:
            x = flat(s)
            tx = defense.transform(s.signal, source_id=p.persona_id,
                                    segment_id=s.segment_id).flatten()[:L]
            tx_s = scaler.transform(tx.reshape(1, -1))
            x_hat = kr.predict(tx_s)[0]
            num = np.linalg.norm(x_hat - x)
            den = np.linalg.norm(x) + 1e-9
            nrmse_list.append(num / den)
        nrmse = float(np.mean(nrmse_list)) if nrmse_list else 1.0
        nrmse_capped = min(1.0, nrmse)

        return AttackResult(
            cell_id=f"{dataset.name}_{defense.name}_{self.name}_seed{seed}",
            dataset=dataset.name, defense=defense.name,
            attack=self.name, seed=seed,
            metric_name="nrmse_a3", metric_value=nrmse_capped,
            subscores={"raw_nrmse": float(nrmse), "n_train": len(train_pairs)},
            n_samples=len(test_pairs),
            extra={"arch": "kernel_ridge_rbf", "split": "persona_stratified"},
        )

    def _empty(self, dataset, defense, seed):
        return AttackResult(
            cell_id=f"{dataset.name}_{defense.name}_{self.name}_seed{seed}",
            dataset=dataset.name, defense=defense.name,
            attack=self.name, seed=seed,
            metric_name="nrmse_a3", metric_value=1.0, n_samples=0,
        )
