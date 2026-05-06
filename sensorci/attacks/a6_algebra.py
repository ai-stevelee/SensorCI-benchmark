"""A6: Algebraic Correctness Probe (P1 evaluation).

For each cured query f in F_D, compare f(T(X)) decoded vs. f(X).
Median Relative Error across queries gives S_P1.

For pilot, uses a stub query bank (linear sums over channel subsets).
Phase 3 LLM-as-Judge curation will replace with ~50 cured queries per dataset.
"""

from __future__ import annotations
import time
import json
from pathlib import Path
import numpy as np

from sensorci.attacks.base import BaseAttack
from sensorci.core.dataset import TSDataset
from sensorci.core.result import AttackResult
from sensorci.defenses.base import BaseDefense


def make_stub_query_bank(n_channels: int, n_queries: int = 20, rng_seed: int = 0) -> list[dict]:
    """Generate a generic linear-functional query bank (placeholder for cured queries).

    Each query: f(x) = sum over (channel, sample) of weight * value.
    """
    rng = np.random.default_rng(rng_seed)
    queries = []
    for i in range(n_queries):
        # Simple per-channel mean / std / max / range queries
        ch = int(rng.integers(0, n_channels))
        op = rng.choice(["mean", "std", "max_abs", "rms", "range"])
        queries.append({
            "query_id": f"stub_q{i:03d}",
            "kind": "channel_stat",
            "channel": ch,
            "op": op,
            "linear_class": "stat",
        })
    return queries


def make_rank_query_bank(n_channels: int, n_queries: int = 20,
                          rng_seed: int = 0) -> list[dict]:
    """Rank-domain query bank for Order-Preserving Encryption (OPE) evaluation.

    All queries here are invariant under any rank-preserving (monotone) transform
    of the input signal, so OPE bucket-quantization preserves them by construction.
    This is the natural query class for OPE's design scope.

    Operations:
      - arg_max / arg_min : index of extreme value
      - argsort_top3_sum  : sum of indices of the top-3 values
      - rank_of_x0        : rank of the first sample within its channel
      - is_first_gt_last  : Boolean (0/1) -- is x[ch][0] > x[ch][-1]?
    """
    rng = np.random.default_rng(rng_seed)
    queries = []
    ops = ["arg_max", "arg_min", "argsort_top3_sum", "rank_of_x0", "is_first_gt_last"]
    for i in range(n_queries):
        ch = int(rng.integers(0, n_channels))
        op = ops[int(rng.integers(0, len(ops)))]
        queries.append({
            "query_id": f"rank_q{i:03d}",
            "kind": "rank_query",
            "channel": ch,
            "op": op,
            "support_class": "rank",
        })
    return queries


def make_ip_query_bank(n_channels: int, signal_length: int,
                       n_queries: int = 20, rng_seed: int = 0,
                       sparsity: float = 0.3) -> list[dict]:
    """Inner-product query bank for Functional Encryption (FE-IP) evaluation.

    Each query: f(x) = <w, x[ch]>, where w is a fixed random sparse vector of
    the same length as x[ch]. These queries are supported natively by FE-IP
    with key sharing; the decoder recovers <w, x_raw> from <w, T(x)>.

    Args:
        sparsity: fraction of weights that are nonzero (random Gaussian).
    """
    rng = np.random.default_rng(rng_seed)
    queries = []
    for i in range(n_queries):
        ch = int(rng.integers(0, n_channels))
        # Random sparse weights
        mask = (rng.random(signal_length) < sparsity).astype(float)
        w = rng.standard_normal(signal_length) * mask
        queries.append({
            "query_id": f"ip_q{i:03d}",
            "kind": "inner_product",
            "channel": ch,
            "weights": w.tolist(),  # serializable
            "sum_w": float(w.sum()),
            "support_class": "inner_product",
        })
    return queries


def evaluate_query(query: dict, signal: np.ndarray) -> float:
    """Compute query value on a signal (n_channels, length)."""
    if query["kind"] == "channel_stat":
        x = signal[query["channel"]]
        op = query["op"]
        if op == "mean":
            return float(np.mean(x))
        if op == "std":
            return float(np.std(x))
        if op == "max_abs":
            return float(np.max(np.abs(x)))
        if op == "rms":
            return float(np.sqrt(np.mean(x ** 2)))
        if op == "range":
            return float(np.max(x) - np.min(x))
    if query["kind"] == "rank_query":
        x = signal[query["channel"]]
        op = query["op"]
        if op == "arg_max":
            return float(np.argmax(x))
        if op == "arg_min":
            return float(np.argmin(x))
        if op == "argsort_top3_sum":
            # rank-preserving: indices of top-3 values
            return float(np.argsort(x)[-3:].sum())
        if op == "rank_of_x0":
            # rank of first sample within channel
            return float((x < x[0]).sum())
        if op == "is_first_gt_last":
            return float(x[0] > x[-1])
    if query["kind"] == "inner_product":
        x = signal[query["channel"]]
        w = np.asarray(query["weights"], dtype=float)
        if w.shape[0] != x.shape[0]:
            # Length mismatch: tile / truncate w to match
            if w.shape[0] > x.shape[0]:
                w = w[:x.shape[0]]
            else:
                w = np.pad(w, (0, x.shape[0] - w.shape[0]))
        return float(np.dot(w, x))
    raise ValueError(f"Unknown query: {query}")


class AlgebraProbeAttack(BaseAttack):
    name = "a6_algebra"
    target_property = "P1"

    def __init__(self, query_bank_path: Path | str | None = None, n_stub_queries: int = 20):
        super().__init__()
        self.query_bank_path = Path(query_bank_path) if query_bank_path else None
        self.n_stub_queries = n_stub_queries
        self.eps = 1e-6

    def _load_query_bank(self, dataset: TSDataset) -> list[dict]:
        if self.query_bank_path and self.query_bank_path.exists():
            return json.loads(self.query_bank_path.read_text())
        # Stub: generic queries
        first_seg = dataset.personas[0].segments[0]
        n_ch = first_seg.signal.shape[0]
        return make_stub_query_bank(n_ch, n_queries=self.n_stub_queries)

    def run(
        self,
        defense: BaseDefense,
        dataset: TSDataset,
        n_samples: int = 100,
        seed: int = 42,
        **kwargs,
    ) -> AttackResult:
        rng = np.random.default_rng(seed)
        bank = self._load_query_bank(dataset)
        all_segs = dataset.all_segments()
        if len(all_segs) > n_samples:
            idxs = rng.choice(len(all_segs), size=n_samples, replace=False)
            sampled = [all_segs[i] for i in idxs]
        else:
            sampled = all_segs

        per_query_re: dict[str, list[float]] = {q["query_id"]: [] for q in bank}
        t0 = time.time()
        for persona, seg in sampled:
            transformed = defense.transform(seg.signal, source_id=persona.persona_id,
                                            segment_id=seg.segment_id)
            for q in bank:
                f_orig = evaluate_query(q, seg.signal)
                f_trans = evaluate_query(q, transformed)
                # Apply decoder
                f_dec = defense.decode_query(f_trans, q, source_id=persona.persona_id)
                # Relative error
                re = abs(f_dec - f_orig) / (abs(f_orig) + self.eps)
                per_query_re[q["query_id"]].append(re)
        elapsed = time.time() - t0

        # Median RE per query, then median across queries
        median_re_per_query = {q: float(np.median(v)) if v else 1.0
                               for q, v in per_query_re.items()}
        overall_median_re = float(np.median(list(median_re_per_query.values()))) \
            if median_re_per_query else 1.0
        s_p1 = max(0.0, 1.0 - overall_median_re)

        cell_id = f"{dataset.name}_{defense.name}_{self.name}_seed{seed}"
        return AttackResult(
            cell_id=cell_id,
            dataset=dataset.name,
            defense=defense.name,
            attack=self.name,
            seed=seed,
            metric_name="s_p1",
            metric_value=s_p1,
            subscores={"median_re": overall_median_re},
            n_samples=len(sampled),
            wall_seconds=elapsed,
            extra={"per_query_median_re": median_re_per_query, "n_queries": len(bank)},
        )


class AlgebraSupportedScopeProbe(BaseAttack):
    """A6-supp: Algebraic correctness probe restricted to a defense's supported query scope.

    Computes S_P1^supp using the natural query class for each defense:
      - rank-domain queries for OPE (D7)
      - inner-product queries for FE-IP (D10) with key sharing
      - the standard channel-stat bank for all other defenses

    The metric is reported alongside (not replacing) the universal S_P1 of
    AlgebraProbeAttack, addressing the cross-paradigm fairness concern that
    function-limited cryptographic schemes are penalized by an evaluation against
    a query class outside their design scope.
    """
    name = "a6_algebra_supp"
    target_property = "P1"

    def __init__(self, n_queries: int = 20, ip_sparsity: float = 0.3):
        super().__init__()
        self.n_queries = n_queries
        self.ip_sparsity = ip_sparsity
        self.eps = 1e-6

    def _select_bank(self, defense_name: str, n_channels: int,
                     signal_length: int, seed: int) -> tuple[list[dict], str]:
        """Pick the appropriate query bank for the given defense."""
        if defense_name == "d7_ope":
            bank = make_rank_query_bank(n_channels, n_queries=self.n_queries,
                                         rng_seed=seed)
            return bank, "rank"
        if defense_name == "d10_fe_ip":
            bank = make_ip_query_bank(n_channels, signal_length,
                                       n_queries=self.n_queries,
                                       rng_seed=seed,
                                       sparsity=self.ip_sparsity)
            return bank, "inner_product"
        # Default: universal channel-stat bank (same as AlgebraProbeAttack)
        bank = make_stub_query_bank(n_channels, n_queries=self.n_queries,
                                     rng_seed=seed)
        return bank, "channel_stat"

    def run(self, defense: BaseDefense, dataset: TSDataset,
            n_samples: int = 100, seed: int = 42, **kwargs) -> AttackResult:
        rng = np.random.default_rng(seed)
        first_seg = dataset.personas[0].segments[0]
        n_ch, sig_len = first_seg.signal.shape
        bank, support_class = self._select_bank(defense.name, n_ch, sig_len, seed)

        all_segs = dataset.all_segments()
        if len(all_segs) > n_samples:
            idxs = rng.choice(len(all_segs), size=n_samples, replace=False)
            sampled = [all_segs[i] for i in idxs]
        else:
            sampled = all_segs

        per_query_re: dict[str, list[float]] = {q["query_id"]: [] for q in bank}
        t0 = time.time()
        for persona, seg in sampled:
            transformed = defense.transform(seg.signal, source_id=persona.persona_id,
                                            segment_id=seg.segment_id)
            for q in bank:
                f_orig = evaluate_query(q, seg.signal)
                f_trans = evaluate_query(q, transformed)
                f_dec = defense.decode_query(f_trans, q, source_id=persona.persona_id)
                re = abs(f_dec - f_orig) / (abs(f_orig) + self.eps)
                per_query_re[q["query_id"]].append(re)
        elapsed = time.time() - t0

        median_re_per_query = {q: float(np.median(v)) if v else 1.0
                               for q, v in per_query_re.items()}
        overall_median_re = float(np.median(list(median_re_per_query.values()))) \
            if median_re_per_query else 1.0
        s_p1_supp = max(0.0, 1.0 - overall_median_re)

        cell_id = f"{dataset.name}_{defense.name}_{self.name}_seed{seed}"
        return AttackResult(
            cell_id=cell_id,
            dataset=dataset.name,
            defense=defense.name,
            attack=self.name,
            seed=seed,
            metric_name="s_p1_supp",
            metric_value=s_p1_supp,
            subscores={"median_re": overall_median_re,
                       "support_class": support_class,
                       "n_queries": len(bank)},
            n_samples=len(sampled),
            wall_seconds=elapsed,
            extra={"per_query_median_re": median_re_per_query,
                   "support_class": support_class,
                   "n_queries": len(bank)},
        )
