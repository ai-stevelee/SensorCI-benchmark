"""4-Tier Compositional Evaluation protocol (T1, T2, T3, T4).

Each tier is a different evaluation regime defined in draft_paper_v1.tex §3.5.
"""

from __future__ import annotations
import logging
import time
from itertools import combinations
import numpy as np

from sensorci.core.dataset import TSDataset
from sensorci.core.persona import ROLES, Role
from sensorci.core.result import AttackResult
from sensorci.defenses.base import BaseDefense
from sensorci.attacks.base import BaseAttack
from sensorci.attacks.a4_ci_linkage import _apply_role_view
from sensorci.attacks.a1_distinguish import _features

logger = logging.getLogger(__name__)


def run_t1_single_segment(
    defense: BaseDefense,
    dataset: TSDataset,
    attacks: list[BaseAttack],
    n_samples: int = 50,
    seed: int = 42,
) -> list[AttackResult]:
    """T1: Standard single-segment evaluation. Each attack runs once per defense."""
    results = []
    for atk in attacks:
        try:
            r = atk.run(defense, dataset, n_samples=n_samples, seed=seed)
            r.extra["tier"] = "T1"
            results.append(r)
        except Exception as e:
            logger.warning(f"T1 attack {atk.name} failed: {e}")
    return results


def run_t2_cross_segment(
    defense: BaseDefense,
    dataset: TSDataset,
    n_values: list[int] | None = None,
    seed: int = 42,
) -> AttackResult:
    """T2: Cross-segment linkage AUC vs N segments per source.

    For each N, attacker has N segments from the same source; computes pairwise
    "same-source vs different-source" classification AUC.
    """
    from sklearn.metrics import roc_auc_score
    n_values = n_values or [1, 5, 10, 30, 100]
    rng = np.random.default_rng(seed)

    auc_per_n: dict[int, float] = {}
    for N in n_values:
        # For each persona, take up to N segments and compute mean feature
        per_source_features = []
        per_source_ids = []
        for p in dataset.personas:
            if not p.segments:
                continue
            segs = p.segments[:N]
            feats = []
            for s in segs:
                tsig = defense.transform(s.signal, source_id=p.persona_id, segment_id=s.segment_id)
                feats.append(_features(tsig))
            if not feats:
                continue
            per_source_features.append(np.mean(np.array(feats), axis=0))
            per_source_ids.append(p.persona_id)

        if len(per_source_features) < 4:
            auc_per_n[N] = 0.5
            continue

        # Build pairwise dataset: same-source vs different-source
        labels = []
        scores = []
        feat_arr = np.array(per_source_features)
        n_src = len(per_source_features)
        # Sample pairs
        for i in range(n_src):
            for j in range(i + 1, n_src):
                same = int(per_source_ids[i] == per_source_ids[j])
                # Score: negative distance = more likely same source
                dist = float(np.linalg.norm(feat_arr[i] - feat_arr[j]))
                labels.append(same)
                scores.append(-dist)
        if len(set(labels)) < 2:
            # All distinct sources (the typical case for one-segment-per-persona)
            # Use within-persona pairs from segments
            within_scores, within_labels = [], []
            for p in dataset.personas:
                if len(p.segments) < 2:
                    continue
                segs = p.segments[:N]
                feats = [_features(defense.transform(s.signal, source_id=p.persona_id,
                                                     segment_id=s.segment_id)) for s in segs]
                # Pairs within
                for i in range(len(feats)):
                    for j in range(i + 1, len(feats)):
                        d = float(np.linalg.norm(feats[i] - feats[j]))
                        within_scores.append(-d)
                        within_labels.append(1)
                # Pairs across (with first persona's first segment)
                if dataset.personas:
                    other = dataset.personas[0]
                    if other.persona_id != p.persona_id and other.segments:
                        os = other.segments[0]
                        of = _features(defense.transform(os.signal, source_id=other.persona_id,
                                                          segment_id=os.segment_id))
                        for f in feats:
                            d = float(np.linalg.norm(f - of))
                            within_scores.append(-d)
                            within_labels.append(0)
            labels = within_labels
            scores = within_scores

        if len(set(labels)) >= 2 and len(labels) > 4:
            try:
                auc = float(roc_auc_score(labels, scores))
            except Exception:
                auc = 0.5
        else:
            auc = 0.5
        auc_per_n[N] = auc

    return AttackResult(
        cell_id=f"{dataset.name}_{defense.name}_t2_seed{seed}",
        dataset=dataset.name, defense=defense.name,
        attack="t2_cross_segment", seed=seed,
        metric_name="t2_auc_max", metric_value=float(max(auc_per_n.values())),
        subscores={f"auc_N{N}": v for N, v in auc_per_n.items()},
        n_samples=len(dataset.personas),
        extra={"tier": "T2", "auc_curve": [{"N": N, "auc": v} for N, v in auc_per_n.items()]},
    )


def run_t3_multi_recipient(
    defense: BaseDefense,
    dataset: TSDataset,
    n_samples: int = 50,
    seed: int = 42,
    max_collusion_size: int = 3,
) -> AttackResult:
    """T3: Per-role + collusion P3-CI advantage. Returns expansion factor."""
    from sensorci.attacks.a4_ci_linkage import RoleStratifiedLinkageAttack
    a4 = RoleStratifiedLinkageAttack()
    base_result = a4.run(defense, dataset, n_samples=n_samples, seed=seed)
    per_role_scores = base_result.extra.get("per_role_scores", {})
    # Per-role max advantage
    per_role_max_adv: dict[str, float] = {}
    for role, attrs in per_role_scores.items():
        if not attrs:
            continue
        per_role_max_adv[role] = max(s["adv"] for s in attrs.values())

    # Collusion expansion: take pairs/triples of roles and union their adv
    expansion_factors = []
    if len(per_role_max_adv) >= 2:
        for size in range(2, min(max_collusion_size + 1, len(per_role_max_adv) + 1)):
            for combo in combinations(per_role_max_adv.keys(), size):
                # Approximation: collusion adv = max over (sum of role adv, capped at 1)
                col_adv = min(1.0, sum(per_role_max_adv[r] for r in combo))
                max_solo = max(per_role_max_adv[r] for r in combo)
                if max_solo > 0:
                    expansion_factors.append(col_adv / max_solo)

    mean_expansion = float(np.mean(expansion_factors)) if expansion_factors else 1.0

    return AttackResult(
        cell_id=f"{dataset.name}_{defense.name}_t3_seed{seed}",
        dataset=dataset.name, defense=defense.name,
        attack="t3_multi_recipient", seed=seed,
        metric_name="t3_collusion_expansion_mean", metric_value=mean_expansion,
        subscores={"max_role_adv_" + r: v for r, v in per_role_max_adv.items()},
        n_samples=n_samples,
        extra={"tier": "T3", "per_role_max_adv": per_role_max_adv,
               "n_collusion_combos_tested": len(expansion_factors)},
    )


def run_t4_longitudinal(
    defense: BaseDefense,
    dataset: TSDataset,
    t_values: list[int] | None = None,
    seed: int = 42,
) -> AttackResult:
    """T4: Cumulative leakage as function of release count.

    For each persona, simulate K=t releases and measure cumulative information
    leakage as the linkage AUC of the K-aggregated representation.
    """
    t_values = t_values or [1, 30, 90, 365]
    rng = np.random.default_rng(seed)

    # We approximate longitudinal release with repeat sampling of segments
    # and compute MI proxy (negative entropy of mean feature distance)
    leak_per_t: dict[int, float] = {}
    for t in t_values:
        # Pick a subset of personas
        n_segs_per_persona = min(t, 100)  # cap for compute
        all_features = []
        all_ids = []
        for p in dataset.personas[:16]:
            if not p.segments:
                continue
            # Sample with replacement if not enough segments
            seg_sample = rng.choice(p.segments, size=n_segs_per_persona, replace=True)
            for s in seg_sample:
                tsig = defense.transform(s.signal, source_id=p.persona_id, segment_id=s.segment_id)
                all_features.append(_features(tsig))
                all_ids.append(p.persona_id)
        if len(set(all_ids)) < 2:
            leak_per_t[t] = 0.0
            continue
        feat_arr = np.array(all_features)
        # Cumulative leakage proxy: 1 - (within-source mean distance / between-source mean distance)
        within_dists = []
        between_dists = []
        for i in range(len(all_features)):
            for j in range(i + 1, len(all_features)):
                d = float(np.linalg.norm(feat_arr[i] - feat_arr[j]))
                if all_ids[i] == all_ids[j]:
                    within_dists.append(d)
                else:
                    between_dists.append(d)
        if not within_dists or not between_dists:
            leak_per_t[t] = 0.0
            continue
        ratio = float(np.mean(within_dists)) / (float(np.mean(between_dists)) + 1e-9)
        leak = max(0.0, min(1.0, 1.0 - ratio))
        leak_per_t[t] = leak

    return AttackResult(
        cell_id=f"{dataset.name}_{defense.name}_t4_seed{seed}",
        dataset=dataset.name, defense=defense.name,
        attack="t4_longitudinal", seed=seed,
        metric_name="t4_leak_max", metric_value=float(max(leak_per_t.values())),
        subscores={f"leak_t{t}": v for t, v in leak_per_t.items()},
        n_samples=len(dataset.personas),
        extra={"tier": "T4",
               "leak_curve": [{"t": t, "leak": v} for t, v in leak_per_t.items()]},
    )
