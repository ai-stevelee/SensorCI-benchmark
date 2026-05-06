"""A5: Statistical 2-sample tests.

MMD, KS, Wasserstein-1 between natural and transformed distributions.
"""

from __future__ import annotations
import numpy as np

from sensorci.attacks.base import BaseAttack
from sensorci.core.dataset import TSDataset
from sensorci.core.result import AttackResult
from sensorci.defenses.base import BaseDefense


def _mmd_rbf(X: np.ndarray, Y: np.ndarray, gamma: float = 1.0) -> float:
    """Maximum Mean Discrepancy (squared) with RBF kernel.

    X, Y: (n_samples, n_features)
    """
    from sklearn.metrics.pairwise import rbf_kernel
    XX = rbf_kernel(X, X, gamma=gamma)
    YY = rbf_kernel(Y, Y, gamma=gamma)
    XY = rbf_kernel(X, Y, gamma=gamma)
    return float(XX.mean() + YY.mean() - 2 * XY.mean())


class StatisticalTestsAttack(BaseAttack):
    name = "a5_stat_tests"
    target_property = "P2"

    def __init__(self, calibration_split: float = 0.5):
        super().__init__()
        self.calibration_split = calibration_split

    def run(
        self,
        defense: BaseDefense,
        dataset: TSDataset,
        n_samples: int = 100,
        seed: int = 42,
        **kwargs,
    ) -> AttackResult:
        from scipy.stats import ks_2samp, wasserstein_distance

        rng = np.random.default_rng(seed)
        all_segs = dataset.all_segments()
        if len(all_segs) > 2 * n_samples:
            idxs = rng.choice(len(all_segs), size=2 * n_samples, replace=False)
            sampled = [all_segs[i] for i in idxs]
        else:
            sampled = all_segs

        half = len(sampled) // 2
        nat_signals = [s.signal for _p, s in sampled[:half]]
        trans_signals = [
            defense.transform(s.signal, source_id=p.persona_id, segment_id=s.segment_id)
            for p, s in sampled[half:]
        ]
        if not nat_signals or not trans_signals:
            return AttackResult(
                cell_id=f"{dataset.name}_{defense.name}_{self.name}_seed{seed}",
                dataset=dataset.name, defense=defense.name,
                attack=self.name, seed=seed,
                metric_name="mmd_norm", metric_value=0.0,
                n_samples=len(sampled),
            )

        # Concatenate per-channel for univariate KS / Wasserstein
        nat_flat = np.concatenate([s.flatten() for s in nat_signals])
        trans_flat = np.concatenate([s.flatten() for s in trans_signals])

        # Subsample for KS (it gets slow on huge arrays)
        sub = 5000
        if nat_flat.size > sub:
            sel = rng.choice(nat_flat.size, sub, replace=False)
            nat_sub = nat_flat[sel]
        else:
            nat_sub = nat_flat
        if trans_flat.size > sub:
            sel = rng.choice(trans_flat.size, sub, replace=False)
            trans_sub = trans_flat[sel]
        else:
            trans_sub = trans_flat

        try:
            ks_stat, ks_p = ks_2samp(nat_sub, trans_sub)
            ks_stat = float(ks_stat)
        except Exception:
            ks_stat, ks_p = 0.5, 1.0
        try:
            w1 = float(wasserstein_distance(nat_sub, trans_sub))
            # Normalize w1 by std of natural for scale-invariance
            w1_norm = w1 / (nat_sub.std() + 1e-9)
            w1_capped = float(min(w1_norm, 1.0))
        except Exception:
            w1_capped = 0.5

        # MMD on summary stat features (same as A1)
        from sensorci.attacks.a1_distinguish import _features
        nat_feats = np.array([_features(s) for s in nat_signals])
        trans_feats = np.array([_features(s) for s in trans_signals])
        try:
            mmd2 = _mmd_rbf(nat_feats, trans_feats, gamma=1.0 / max(nat_feats.shape[1], 1))
            mmd_norm = float(min(max(mmd2, 0.0), 1.0))
        except Exception:
            mmd_norm = 0.5

        return AttackResult(
            cell_id=f"{dataset.name}_{defense.name}_{self.name}_seed{seed}",
            dataset=dataset.name, defense=defense.name,
            attack=self.name, seed=seed,
            metric_name="mmd_norm", metric_value=mmd_norm,
            subscores={"ks_stat": ks_stat, "ks_p": float(ks_p),
                       "wasserstein1_norm": w1_capped},
            n_samples=len(sampled),
        )
