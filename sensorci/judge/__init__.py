"""LLM-as-Judge robustness module.

Implements:
- Cross-LLM agreement metrics (Fleiss' kappa, leave-one-out Kendall's tau)
- Contamination control (counterfactual relabeling, metadata blinding)
"""

from sensorci.judge.agreement import (
    fleiss_kappa,
    leave_one_out_kendall,
    cross_judge_agreement,
)
from sensorci.judge.contamination import (
    counterfactual_relabel,
    metadata_blind,
    contamination_delta,
)
from sensorci.judge.generator import generate_queries, generate_sharing_matrix
from sensorci.judge.voter import vote_on_queries
from sensorci.judge.filter import consensus_filter, numerical_sanity_check
from sensorci.judge.stratify import stratify_by_difficulty

__all__ = [
    "fleiss_kappa", "leave_one_out_kendall", "cross_judge_agreement",
    "counterfactual_relabel", "metadata_blind", "contamination_delta",
    "generate_queries", "generate_sharing_matrix",
    "vote_on_queries", "consensus_filter", "numerical_sanity_check",
    "stratify_by_difficulty",
]
