"""A4-CI T3 sensitivity ablations (Q4 rebuttal).

The standard A4-CI siamese probe yields near-zero per-role advantage on
PAMAP2 (paper §8.5). Reviewer Q4 asks whether this null result is a
measurement artifact (probe weakness, attribute curation, role view too lossy)
or an intrinsic property of the curated sharing matrix on PAMAP2.

This file provides three ablation probes that share the same A4-CI advantage
formula (acc - prior) but vary one experimental knob each:

  1. ``stronger_probe`` — replace siamese encoder + KNN head with a
     GradientBoostingClassifier (200 estimators, depth 5) directly on the
     same per-role statistical features used by the RF fallback. This isolates
     "is the probe too weak?".

  2. ``all_attributes`` — enumerate EVERY attribute available on the dataset,
     not only those marked Not-share by the curated sharing matrix. If the
     probe finds no advantage even on Share-labelled attributes (which the
     adversary IS allowed to recover), the probe is genuinely under-powered.
     Conversely, if Share-labelled attributes ARE predictable but Not-share
     ones are not, the curated sharing matrix is the discriminative line.

  3. ``no_downsample`` — override the role's ``downsample_factor`` to 1 for
     every non-Operator role, effectively granting every role the full raw
     signal. Tests whether the null is caused by the role view being too
     lossy.

Each ablation uses the standard A4-CI metric:
    advantage = max(0, accuracy - prior)
and reports max-advantage per role per attribute, so downstream T3 collusion
expansion can be computed by ``run_t3_multi_recipient`` (or the
``T3CollusionFromAblation`` helper below) without further changes.
"""

from __future__ import annotations
import logging
import time
from itertools import combinations

import numpy as np

from sensorci.attacks.base import BaseAttack
from sensorci.core.dataset import TSDataset
from sensorci.core.result import AttackResult
from sensorci.core.persona import ROLES
from sensorci.core.splits import PersonaStratifiedSplitter
from sensorci.defenses.base import BaseDefense
from sensorci.attacks.a1_distinguish import _features
from sensorci.attacks.a4_ci_linkage import _apply_role_view

logger = logging.getLogger(__name__)


class T3AblationProbe(BaseAttack):
    """A4-CI advantage with one experimental knob varied (see module doc)."""
    name = "a4_ci_t3_ablation"
    target_property = "P3-CI"

    VALID_KINDS = ("stronger_probe", "all_attributes", "no_downsample")

    def __init__(self, ablation_kind: str = "stronger_probe",
                 n_estimators: int = 200, max_depth: int = 5):
        super().__init__()
        if ablation_kind not in self.VALID_KINDS:
            raise ValueError(
                f"ablation_kind must be one of {self.VALID_KINDS}, "
                f"got {ablation_kind!r}"
            )
        self.ablation_kind = ablation_kind
        self.n_estimators = n_estimators
        self.max_depth = max_depth

    # ------------------------------------------------------------------
    def _attribute_set_for(self, dataset: TSDataset, role: str) -> tuple[list[str], str]:
        """Pick the attribute list for the given (dataset, role) under the ablation."""
        if self.ablation_kind == "all_attributes":
            return sorted(dataset.attribute_set()), "all"
        return dataset.not_share_attributes(role), "not_share"

    def _downsample_factor_for(self, dataset: TSDataset, role: str) -> int:
        if self.ablation_kind == "no_downsample":
            return 1
        cap = dataset.role_capabilities.get(role)
        return cap.downsample_factor if cap else 1

    def _make_classifier(self, n_classes: int):
        """Pick a classifier per the ablation knob."""
        if self.ablation_kind == "stronger_probe":
            from sklearn.ensemble import GradientBoostingClassifier
            return GradientBoostingClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=0.05,
                random_state=0,
            )
        # "all_attributes" and "no_downsample" share the standard RF baseline
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(
            n_estimators=200, max_depth=8, random_state=0, n_jobs=1,
        )

    # ------------------------------------------------------------------
    def run(
        self,
        defense: BaseDefense,
        dataset: TSDataset,
        n_samples: int = 100,
        seed: int = 42,
        **kwargs,
    ) -> AttackResult:
        # Use persona-stratified split (no persona appears in both train and test)
        # to match the protocol of the original A4-CI siamese probe and avoid
        # the same-subject memorisation that k-fold CV would allow.
        try:
            split = PersonaStratifiedSplitter(0.6, 0.2, 0.2, seed=seed).split(dataset)
        except ValueError:
            return self._empty(dataset, defense, seed, "persona-stratified split failed")

        train_segs = split.train_segments[: n_samples]
        test_segs = split.test_segments[: max(1, n_samples // 3)]
        if len(train_segs) < 4 or len(test_segs) < 2:
            return self._empty(dataset, defense, seed, "split too small")

        per_role_results: dict[str, dict[str, dict]] = {}
        max_adv = 0.0
        t0 = time.time()

        def role_features(segs, role, ds_factor):
            X = []
            attrs: dict[str, list[str]] = {}
            for p, s in segs:
                tsig = defense.transform(s.signal, source_id=p.persona_id,
                                          segment_id=s.segment_id)
                view = _apply_role_view(tsig, role, ds_factor)
                X.append(_features(view))
                for k, v in p.all_attributes.items():
                    attrs.setdefault(k, []).append(str(v))
            return np.asarray(X), attrs

        for role in ROLES:
            if role == "Operator":
                continue   # Operator has full access by definition
            ds_factor = self._downsample_factor_for(dataset, role)

            X_train, attr_train = role_features(train_segs, role, ds_factor)
            X_test, attr_test = role_features(test_segs, role, ds_factor)
            if X_train.size == 0 or X_test.size == 0:
                continue

            attrs, scope = self._attribute_set_for(dataset, role)
            role_scores: dict[str, dict] = {}
            for attr in attrs:
                tr_y = attr_train.get(attr, [])
                te_y = attr_test.get(attr, [])
                if len(set(tr_y)) < 2 or not te_y:
                    continue
                # Class-set on TRAIN only; use prior over union so adversary
                # is not rewarded for picking a label class never seen at train.
                y_tr = np.asarray(tr_y)
                y_te = np.asarray(te_y)
                try:
                    clf = self._make_classifier(len(set(tr_y)))
                    clf.fit(X_train, y_tr)
                    pred = clf.predict(X_test)
                    acc = float((pred == y_te).mean())
                except Exception as e:
                    logger.debug(f"Classifier failed for {role}/{attr}: {e}")
                    acc = 1.0 / max(len(set(tr_y)), 2)
                # Prior = chance = 1/n_classes_in_train_union_test
                n_cls = max(len(set(list(tr_y) + list(te_y))), 2)
                prior = 1.0 / n_cls
                adv = max(0.0, acc - prior)
                role_scores[attr] = {
                    "acc": acc, "prior": prior, "adv": adv,
                    "n_classes": n_cls,
                    "scope": scope,
                }
                if adv > max_adv:
                    max_adv = adv
            per_role_results[role] = role_scores

        # Compute T3 collusion expansion (same formula as run_t3_multi_recipient,
        # but using the ablation per-role advantages)
        per_role_max_adv: dict[str, float] = {}
        for role, attrs in per_role_results.items():
            if not attrs:
                continue
            per_role_max_adv[role] = max(s["adv"] for s in attrs.values())

        expansion_factors: list[float] = []
        for size in range(2, min(4, len(per_role_max_adv) + 1)):
            for combo in combinations(per_role_max_adv.keys(), size):
                col_adv = min(1.0, sum(per_role_max_adv[r] for r in combo))
                max_solo = max(per_role_max_adv[r] for r in combo)
                if max_solo > 0:
                    expansion_factors.append(col_adv / max_solo)
        mean_expansion = float(np.mean(expansion_factors)) if expansion_factors else 1.0

        elapsed = time.time() - t0
        cell_id = (
            f"{dataset.name}_{defense.name}_{self.name}_"
            f"{self.ablation_kind}_seed{seed}"
        )
        return AttackResult(
            cell_id=cell_id,
            dataset=dataset.name,
            defense=defense.name,
            attack=self.name,
            seed=seed,
            metric_name="adv_a4_ci_max",
            metric_value=max_adv,
            subscores={
                "ablation_kind": self.ablation_kind,
                "t3_expansion_mean": mean_expansion,
                "n_collusion_combos": len(expansion_factors),
                **{f"max_role_adv_{r}": v for r, v in per_role_max_adv.items()},
            },
            n_samples=len(train_segs) + len(test_segs),
            wall_seconds=elapsed,
            extra={
                "ablation_kind": self.ablation_kind,
                "per_role_scores": per_role_results,
                "per_role_max_adv": per_role_max_adv,
                "t3_expansion_mean": mean_expansion,
                "tier": "T3_ablation",
            },
        )

    def _empty(self, dataset, defense, seed, reason: str = ""):
        return AttackResult(
            cell_id=f"{dataset.name}_{defense.name}_{self.name}_{self.ablation_kind}_seed{seed}",
            dataset=dataset.name,
            defense=defense.name,
            attack=self.name,
            seed=seed,
            metric_name="adv_a4_ci_max",
            metric_value=0.0,
            n_samples=0,
            extra={"note": reason, "ablation_kind": self.ablation_kind},
        )
