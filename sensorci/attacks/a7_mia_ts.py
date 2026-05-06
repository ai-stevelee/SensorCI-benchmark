"""A7: LiRA-TS Membership Inference Attack adapted for time-series.

Based on:
  - Shokri et al. "Membership Inference Attacks against Machine Learning Models" (S&P 2017)
  - Carlini et al. "Membership Inference Attacks From First Principles" (S&P 2022, LiRA)
  - Liu et al. arXiv:2407.02870 "Membership Inference Attacks Against Time-Series Models"

LiRA approach (paper-quality):
  1. Train K shadow models on disjoint random subsets of train segments.
  2. For each test segment x, compute target classifier likelihood vs shadow
     classifier likelihoods.
  3. Likelihood ratio test: TPR @ low FPR.
"""

from __future__ import annotations
import logging
import time
import numpy as np

from sensorci.attacks.base import BaseAttack
from sensorci.attacks.a1_distinguish import _features
from sensorci.core.dataset import TSDataset
from sensorci.core.result import AttackResult
from sensorci.core.splits import PersonaStratifiedSplitter
from sensorci.defenses.base import BaseDefense

logger = logging.getLogger(__name__)


class MIATimeSeriesAttack(BaseAttack):
    """LiRA-style MIA against a target classifier trained on transformed signals."""

    name = "a7_mia_ts"
    target_property = "P4"

    def __init__(
        self,
        n_shadow_models: int = 8,
        target_fpr: float = 0.1,
        shadow_subsample_frac: float = 0.5,
    ):
        super().__init__()
        self.n_shadow_models = n_shadow_models
        self.target_fpr = target_fpr
        self.shadow_subsample_frac = shadow_subsample_frac

    def run(
        self,
        defense: BaseDefense,
        dataset: TSDataset,
        n_samples: int = 100,
        seed: int = 42,
        **kwargs,
    ) -> AttackResult:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_curve

        rng = np.random.default_rng(seed)

        # Persona-stratified split: members vs non-members
        try:
            splitter = PersonaStratifiedSplitter(
                train_frac=0.5, val_frac=0.0, test_frac=0.5, seed=seed,
            )
            split = splitter.split(dataset)
        except ValueError:
            return self._empty_result(dataset, defense, seed)

        member_segs = split.train_segments[: max(2, n_samples // 2)]
        non_member_segs = split.test_segments[: max(2, n_samples // 2)]
        if not member_segs or not non_member_segs:
            return self._empty_result(dataset, defense, seed)

        t0 = time.time()

        def to_features(p, s):
            tsig = defense.transform(s.signal, source_id=p.persona_id,
                                     segment_id=s.segment_id)
            return _features(tsig)

        member_X = np.array([to_features(p, s) for p, s in member_segs])
        non_member_X = np.array([to_features(p, s) for p, s in non_member_segs])

        all_train_segs = split.train_segments
        if len(all_train_segs) < 8:
            return self._empty_result(dataset, defense, seed)

        train_X = np.array([to_features(p, s) for p, s in all_train_segs])
        train_y = np.array([p.persona_id for p, s in all_train_segs])
        id_to_int = {pid: i for i, pid in enumerate(sorted(set(train_y)))}
        train_y_int = np.array([id_to_int[y] for y in train_y])

        if len(set(train_y_int)) < 2:
            return self._empty_result(dataset, defense, seed)

        # Train target model
        try:
            target_model = LogisticRegression(max_iter=200, random_state=seed)
            target_model.fit(train_X, train_y_int)
        except Exception as e:
            logger.warning(f"Target model training failed: {e}")
            return self._empty_result(dataset, defense, seed)

        # Train K shadow models on disjoint subsets
        n_train = len(all_train_segs)
        sub_n = max(4, int(n_train * self.shadow_subsample_frac))
        shadow_lk_m_list = []
        shadow_lk_nm_list = []

        for k in range(self.n_shadow_models):
            shadow_idx = rng.choice(n_train, size=sub_n, replace=False)
            shadow_X = train_X[shadow_idx]
            shadow_y = train_y_int[shadow_idx]
            if len(set(shadow_y)) < 2:
                continue
            try:
                shadow_model = LogisticRegression(max_iter=200, random_state=seed + k + 1)
                shadow_model.fit(shadow_X, shadow_y)
            except Exception:
                continue
            member_pred = shadow_model.predict_proba(member_X)
            non_member_pred = shadow_model.predict_proba(non_member_X)
            shadow_lk_m_list.append(np.max(member_pred, axis=1))
            shadow_lk_nm_list.append(np.max(non_member_pred, axis=1))

        if not shadow_lk_m_list:
            return self._empty_result(dataset, defense, seed)

        shadow_lk_m = np.array(shadow_lk_m_list)
        shadow_lk_nm = np.array(shadow_lk_nm_list)

        # Target likelihoods
        target_member_pred = target_model.predict_proba(member_X)
        target_non_member_pred = target_model.predict_proba(non_member_X)
        target_lk_m = np.max(target_member_pred, axis=1)
        target_lk_nm = np.max(target_non_member_pred, axis=1)

        # LiRA score: target_lk - mean(shadow_lk)
        member_scores = target_lk_m - shadow_lk_m.mean(axis=0)
        non_member_scores = target_lk_nm - shadow_lk_nm.mean(axis=0)

        scores = np.concatenate([member_scores, non_member_scores])
        labels = np.concatenate([
            np.ones(len(member_scores)),
            np.zeros(len(non_member_scores)),
        ])

        try:
            fpr, tpr, _ = roc_curve(labels, scores)
            target_tpr = float(np.interp(self.target_fpr, fpr, tpr))
        except Exception:
            target_tpr = 0.0

        adv = max(0.0, target_tpr - self.target_fpr)
        elapsed = time.time() - t0

        return AttackResult(
            cell_id=f"{dataset.name}_{defense.name}_{self.name}_seed{seed}",
            dataset=dataset.name, defense=defense.name,
            attack=self.name, seed=seed,
            metric_name="adv_a7", metric_value=adv,
            subscores={"tpr_at_fpr": target_tpr, "fpr_target": self.target_fpr,
                       "n_shadow_trained": len(shadow_lk_m_list)},
            n_samples=len(member_segs) + len(non_member_segs),
            wall_seconds=elapsed,
            extra={"implementation": "LiRA-TS with logistic regression shadow models",
                   "n_shadow_models_requested": self.n_shadow_models},
        )

    def _empty_result(self, dataset, defense, seed):
        return AttackResult(
            cell_id=f"{dataset.name}_{defense.name}_{self.name}_seed{seed}",
            dataset=dataset.name, defense=defense.name,
            attack=self.name, seed=seed,
            metric_name="adv_a7", metric_value=0.0,
            n_samples=0,
            extra={"note": "insufficient data for MIA evaluation"},
        )
