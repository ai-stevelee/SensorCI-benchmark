"""A4-CI: Role-stratified linkage / attribute inference attack — paper-quality.

Two modes:
  - "siamese" (default if torch available): Siamese encoder trained with
    contrastive loss (SimCLR-style) on persona-stratified split.
  - "rf" (fallback): RandomForestClassifier on stat features with k-fold CV.
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
from sensorci.core.persona import ROLES, Role
from sensorci.attacks.a1_distinguish import _features

logger = logging.getLogger(__name__)


def _apply_role_view(signal: np.ndarray, role: Role,
                     downsample_factor: int = 1) -> np.ndarray:
    if role == "Operator":
        return signal
    if downsample_factor > 1 and signal.shape[1] > downsample_factor:
        return signal[:, ::downsample_factor]
    return signal


class RoleStratifiedLinkageAttack(BaseAttack):
    name = "a4_ci_linkage"
    target_property = "P3-CI"

    def __init__(self, arch: str = "auto", n_estimators: int = 50,
                 siamese_epochs: int = 15, embed_dim: int = 64):
        super().__init__()
        self.arch = arch
        self.n_estimators = n_estimators
        self.siamese_epochs = siamese_epochs
        self.embed_dim = embed_dim

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
                arch = "siamese"
            except ImportError:
                arch = "rf"

        if arch == "siamese":
            return self._run_siamese(defense, dataset, n_samples, seed)
        return self._run_rf(defense, dataset, n_samples, seed)

    def _run_siamese(self, defense, dataset, n_samples, seed):
        from sensorci.attacks._neural_models import make_siamese_encoder, train_siamese
        try:
            import torch
        except ImportError:
            return self._run_rf(defense, dataset, n_samples, seed)

        try:
            split = PersonaStratifiedSplitter(0.6, 0.2, 0.2, seed=seed).split(dataset)
        except ValueError:
            return self._empty(dataset, defense, seed)

        if len(split.train_segments) < 8 or len(split.test_segments) < 4:
            return self._run_rf(defense, dataset, n_samples, seed)

        all_segs = split.train_segments + split.val_segments + split.test_segments
        L = min(s.signal.shape[1] for _p, s in all_segs)
        n_ch = all_segs[0][1].signal.shape[0]

        all_pids = sorted({p.persona_id for p, _s in all_segs})
        pid_to_int = {pid: i for i, pid in enumerate(all_pids)}

        def make_train_data(segs):
            return [
                (defense.transform(s.signal, source_id=p.persona_id,
                                    segment_id=s.segment_id)[:, :L],
                 pid_to_int[p.persona_id])
                for p, s in segs
            ]

        train_data = make_train_data(split.train_segments[: 2 * n_samples])

        t0 = time.time()
        model = make_siamese_encoder(n_channels=n_ch, embed_dim=self.embed_dim)
        if model is None:
            return self._run_rf(defense, dataset, n_samples, seed)
        try:
            model = train_siamese(
                model, train_data,
                n_epochs=self.siamese_epochs, batch_size=8, lr=1e-3,
            )
        except Exception as e:
            logger.warning(f"Siamese training failed: {e}; fallback to rf")
            return self._run_rf(defense, dataset, n_samples, seed)

        per_role_results = {}
        max_adv = 0.0
        device = next(model.parameters()).device
        model.eval()

        for role in ROLES:
            cap = dataset.role_capabilities.get(role)
            ds_factor = cap.downsample_factor if cap else 1

            def embed_segs(segs):
                arrs = []
                for p, s in segs:
                    tsig = defense.transform(s.signal, source_id=p.persona_id,
                                              segment_id=s.segment_id)
                    view = _apply_role_view(tsig, role, ds_factor)
                    if view.shape[1] != L:
                        from scipy.signal import resample
                        view = resample(view, L, axis=-1)
                    arrs.append(view[:, :L])
                if not arrs:
                    return np.zeros((0, self.embed_dim))
                X_t = torch.from_numpy(np.array(arrs)).float().to(device)
                with torch.no_grad():
                    emb = model(X_t).cpu().numpy()
                return emb

            train_emb = embed_segs(split.train_segments[: n_samples])
            test_emb = embed_segs(split.test_segments[: n_samples])
            if len(train_emb) < 4 or len(test_emb) < 2:
                continue

            train_attrs = {}
            for k in (split.train_segments[0][0].all_attributes if split.train_segments else {}):
                train_attrs[k] = [str(p.all_attributes.get(k, ""))
                                  for p, _s in split.train_segments[: n_samples]]
            test_attrs = {}
            for k in train_attrs:
                test_attrs[k] = [str(p.all_attributes.get(k, ""))
                                 for p, _s in split.test_segments[: n_samples]]

            not_share = dataset.not_share_attributes(role)
            role_scores = {}
            for attr in not_share:
                if attr not in train_attrs:
                    continue
                from sklearn.neighbors import KNeighborsClassifier
                tr_y = train_attrs[attr]
                te_y = test_attrs[attr]
                if len(set(tr_y)) < 2:
                    continue
                try:
                    knn = KNeighborsClassifier(n_neighbors=min(5, len(train_emb)))
                    knn.fit(train_emb, tr_y)
                    pred = knn.predict(test_emb)
                    acc = float((pred == np.array(te_y)).mean())
                except Exception:
                    acc = 1.0 / max(len(set(tr_y)), 2)
                prior = 1.0 / max(len(set(tr_y)), 2)
                adv = max(0.0, acc - prior)
                role_scores[attr] = {"acc": acc, "prior": prior, "adv": adv}
                if adv > max_adv:
                    max_adv = adv
            per_role_results[role] = role_scores

        elapsed = time.time() - t0
        return AttackResult(
            cell_id=f"{dataset.name}_{defense.name}_{self.name}_seed{seed}",
            dataset=dataset.name, defense=defense.name,
            attack=self.name, seed=seed,
            metric_name="adv_a4_ci_max", metric_value=max_adv,
            n_samples=n_samples,
            wall_seconds=elapsed,
            extra={"per_role_scores": per_role_results,
                   "arch": "siamese_simclr",
                   "embed_dim": self.embed_dim,
                   "split": "persona_stratified"},
        )

    def _run_rf(self, defense, dataset, n_samples, seed):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score

        rng = np.random.default_rng(seed)
        all_segs = dataset.all_segments()
        if len(all_segs) > n_samples:
            idxs = rng.choice(len(all_segs), size=n_samples, replace=False)
            sampled = [all_segs[i] for i in idxs]
        else:
            sampled = all_segs
        if len(sampled) < 4:
            return self._empty(dataset, defense, seed)

        per_role_results = {}
        max_adv = 0.0

        for role in ROLES:
            cap = dataset.role_capabilities.get(role)
            ds_factor = cap.downsample_factor if cap else 1
            X_features = []
            attr_values = {}
            for p, s in sampled:
                tsig = defense.transform(s.signal, source_id=p.persona_id,
                                          segment_id=s.segment_id)
                view = _apply_role_view(tsig, role, ds_factor)
                X_features.append(_features(view))
                for k, v in p.all_attributes.items():
                    attr_values.setdefault(k, []).append(str(v))
            X = np.array(X_features)

            not_share_attrs = dataset.not_share_attributes(role)
            role_scores = {}
            for attr in not_share_attrs:
                vals = attr_values.get(attr, [])
                if len(set(vals)) < 2:
                    continue
                y = np.array(vals)
                try:
                    cv = min(3, max(2, X.shape[0] // 3))
                    clf = RandomForestClassifier(
                        n_estimators=self.n_estimators, random_state=seed, n_jobs=1,
                    )
                    scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
                    acc = float(scores.mean())
                except Exception:
                    acc = 1.0 / max(len(set(vals)), 2)
                prior = 1.0 / max(len(set(vals)), 2)
                adv = max(0.0, acc - prior)
                role_scores[attr] = {"acc": acc, "prior": prior, "adv": adv}
                if adv > max_adv:
                    max_adv = adv
            per_role_results[role] = role_scores

        return AttackResult(
            cell_id=f"{dataset.name}_{defense.name}_{self.name}_seed{seed}",
            dataset=dataset.name, defense=defense.name,
            attack=self.name, seed=seed,
            metric_name="adv_a4_ci_max", metric_value=max_adv,
            n_samples=len(sampled),
            extra={"per_role_scores": per_role_results,
                   "arch": "random_forest", "split": "kfold_cv"},
        )

    def _empty(self, dataset, defense, seed):
        return AttackResult(
            cell_id=f"{dataset.name}_{defense.name}_{self.name}_seed{seed}",
            dataset=dataset.name, defense=defense.name,
            attack=self.name, seed=seed,
            metric_name="adv_a4_ci_max", metric_value=0.0, n_samples=0,
        )
