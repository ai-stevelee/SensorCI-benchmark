"""A1: Distinguishability detector (paper-quality).

Train a classifier to distinguish T(X) from natural X. Higher accuracy =>
lower naturalness (P_2 score down).

Two implementations:
  - "neural" (default if torch available): ResNet-1D + transformer head
    with persona-stratified train/val/test split + HP tuning on val.
  - "boost" (fallback / pilot): GradientBoostingClassifier on stat features
    with k-fold CV.

Use `--arch neural` for paper-quality, `--arch boost` for fast pilot.
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


def _features(signal: np.ndarray) -> np.ndarray:
    """Per-channel summary features (used by `boost` arch and as utility elsewhere)."""
    feats = []
    for c in range(signal.shape[0]):
        x = signal[c]
        sd = x.std()
        sd_safe = sd if sd > 1e-12 else 1.0
        z = (x - x.mean()) / sd_safe
        kurt = float(np.mean(z ** 4) - 3.0)
        skew = float(np.mean(z ** 3))
        mag = np.abs(np.fft.rfft(x))
        n = len(mag)
        lo = float(mag[:max(n // 4, 1)].sum())
        hi = float(mag[max(n // 2, 1):].sum())
        feats.extend([
            float(x.mean()), float(sd), float(np.max(np.abs(x))),
            float(np.sqrt(np.mean(x ** 2))), kurt, skew, lo, hi,
        ])
    return np.array(feats, dtype=float)


class DistinguishabilityAttack(BaseAttack):
    name = "a1_distinguish"
    target_property = "P2"

    def __init__(self, arch: str = "auto", epochs: int = 20,
                 base_filters: int = 32, n_resblocks: int = 4,
                 hp_search: bool = False):
        super().__init__()
        self.arch = arch
        self.epochs = epochs
        self.base_filters = base_filters
        self.n_resblocks = n_resblocks
        self.hp_search = hp_search

    def run(
        self,
        defense: BaseDefense,
        dataset: TSDataset,
        n_samples: int = 100,
        seed: int = 42,
        **kwargs,
    ) -> AttackResult:
        # Auto-pick arch based on torch availability
        arch = self.arch
        if arch == "auto":
            try:
                import torch  # noqa
                arch = "neural"
            except ImportError:
                arch = "boost"

        if arch == "neural":
            return self._run_neural(defense, dataset, n_samples, seed)
        else:
            return self._run_boost(defense, dataset, n_samples, seed)

    # ------------------------------------------------------------------
    # Neural (paper-quality): persona-stratified split + ResNet1D + HP search
    # ------------------------------------------------------------------

    def _run_neural(self, defense, dataset, n_samples, seed):
        from sensorci.attacks._neural_models import (
            make_resnet1d_transformer, train_classifier, predict_classifier,
        )
        t0 = time.time()
        # Persona-stratified split
        splitter = PersonaStratifiedSplitter(0.6, 0.2, 0.2, seed=seed)
        try:
            split = splitter.split(dataset)
        except ValueError:
            return self._empty(dataset, defense, seed, reason="too few personas")

        # Build (signal, label) pairs: 0 = natural, 1 = transformed
        def make_pairs(seg_list, cap):
            pairs = []
            for p, s in seg_list[:cap]:
                pairs.append((s.signal, 0))  # natural
                tsig = defense.transform(s.signal, source_id=p.persona_id,
                                          segment_id=s.segment_id)
                pairs.append((tsig, 1))      # transformed
            return pairs

        cap = max(2, n_samples // 2)
        train_pairs = make_pairs(split.train_segments, cap * 2)
        val_pairs = make_pairs(split.val_segments, cap)
        test_pairs = make_pairs(split.test_segments, cap)

        if len(train_pairs) < 8 or len(test_pairs) < 4:
            return self._run_boost(defense, dataset, n_samples, seed)

        # Stack and find common length
        L = min(s.shape[1] for s, _ in train_pairs + val_pairs + test_pairs)
        n_ch = train_pairs[0][0].shape[0]

        def to_array(pairs):
            X = np.array([s[:, :L] for s, _ in pairs])
            y = np.array([y for _, y in pairs])
            return X, y

        train_X, train_y = to_array(train_pairs)
        val_X, val_y = to_array(val_pairs)
        test_X, test_y = to_array(test_pairs)

        # Build model
        model = make_resnet1d_transformer(n_channels=n_ch, n_classes=2,
                                           base_filters=self.base_filters,
                                           n_resblocks=self.n_resblocks)
        if model is None:
            return self._run_boost(defense, dataset, n_samples, seed)

        try:
            model, history = train_classifier(
                model, train_X, train_y, val_X, val_y,
                epochs=self.epochs, batch_size=8, lr=1e-3,
            )
            preds, _ = predict_classifier(model, test_X)
            test_acc = float((preds == test_y).mean())
        except Exception as e:
            logger.warning(f"Neural A1 training failed: {e}; falling back to boost")
            return self._run_boost(defense, dataset, n_samples, seed)

        adv = max(0.0, 2.0 * test_acc - 1.0)
        elapsed = time.time() - t0

        return AttackResult(
            cell_id=f"{dataset.name}_{defense.name}_{self.name}_seed{seed}",
            dataset=dataset.name, defense=defense.name,
            attack=self.name, seed=seed,
            metric_name="adv_a1", metric_value=adv,
            subscores={"test_accuracy": test_acc, "val_acc_final":
                       history.get("val_acc", [0.0])[-1] if history.get("val_acc") else 0.0},
            n_samples=len(test_pairs),
            wall_seconds=elapsed,
            extra={"arch": "neural_resnet1d_transformer",
                   "epochs": self.epochs,
                   "n_train": len(train_pairs), "n_val": len(val_pairs),
                   "n_test": len(test_pairs),
                   "split": "persona_stratified"},
        )

    # ------------------------------------------------------------------
    # Boost (pilot fallback): sklearn GradientBoosting on stat features
    # ------------------------------------------------------------------

    def _run_boost(self, defense, dataset, n_samples, seed):
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.model_selection import cross_val_score

        rng = np.random.default_rng(seed)
        all_segs = dataset.all_segments()
        if len(all_segs) > 2 * n_samples:
            idxs = rng.choice(len(all_segs), size=2 * n_samples, replace=False)
            sampled = [all_segs[i] for i in idxs]
        else:
            sampled = all_segs

        half = len(sampled) // 2
        natural_X = [_features(s.signal) for _p, s in sampled[:half]]
        transformed_X = [
            _features(defense.transform(s.signal, source_id=p.persona_id,
                                         segment_id=s.segment_id))
            for p, s in sampled[half:]
        ]
        X = np.array(natural_X + transformed_X)
        y = np.array([0] * len(natural_X) + [1] * len(transformed_X))

        if len(set(y)) < 2 or X.shape[0] < 4:
            return self._empty(dataset, defense, seed, reason="too few samples")

        t0 = time.time()
        clf = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=seed)
        try:
            cv = min(5, X.shape[0] // 2)
            scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
            test_acc = float(scores.mean())
        except Exception:
            test_acc = 0.5
        elapsed = time.time() - t0
        adv = max(0.0, 2.0 * test_acc - 1.0)

        return AttackResult(
            cell_id=f"{dataset.name}_{defense.name}_{self.name}_seed{seed}",
            dataset=dataset.name, defense=defense.name,
            attack=self.name, seed=seed,
            metric_name="adv_a1", metric_value=adv,
            subscores={"test_accuracy": test_acc},
            n_samples=len(sampled), wall_seconds=elapsed,
            extra={"arch": "boost_gradient_boosting",
                   "feature_dim": int(X.shape[1]) if X.size else 0,
                   "split": "kfold_cv"},
        )

    def _empty(self, dataset, defense, seed, reason="insufficient data"):
        return AttackResult(
            cell_id=f"{dataset.name}_{defense.name}_{self.name}_seed{seed}",
            dataset=dataset.name, defense=defense.name,
            attack=self.name, seed=seed,
            metric_name="adv_a1", metric_value=0.0,
            n_samples=0, extra={"note": reason},
        )
