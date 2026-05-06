"""LLM-as-Judge Stage 2: cross-validation voting.

Each candidate is rated by the K-1 LLMs (excluding its own generator).
"""

from __future__ import annotations
import json
import re
import logging
import numpy as np

from sensorci.api_pool import get_default_pool, APIKeyPool
from sensorci.judge.generator import _extract_json

logger = logging.getLogger(__name__)


VOTE_PROMPT = """\
You are reviewing a proposed analyst query.

Query:
{query_json}

Rate on three axes (1-5 scale):
- formula_correctness: does the Python formula produce a sensible scalar?
- physical_meaningfulness: is the physical interpretation correct?
- canonicalness: would a working domain engineer compute this?

Output JSON: {{"formula_correctness": int, "physical_meaningfulness": int,
              "canonicalness": int}}

Respond with valid JSON only:
"""


def vote_on_queries(
    candidates: list[dict],
    judges: list[str] | None = None,
    pool: APIKeyPool | None = None,
) -> list[dict]:
    """For each candidate, ask each non-generator judge for a 3-axis Likert score.

    Returns candidates annotated with .judge_scores dict.
    """
    pool = pool or get_default_pool()
    judges = judges or ["gpt-4o-mini", "claude-haiku-4-5", "gemini-3-flash"]

    voted = []
    for q in candidates:
        gen = q.get("generator", "")
        relevant_judges = [j for j in judges if j != gen]
        prompt = VOTE_PROMPT.format(query_json=json.dumps(
            {k: v for k, v in q.items() if k not in ("generator", "query_id_global")},
            indent=2))
        scores = {}
        for j in relevant_judges:
            resp = pool.call(j, [{"role": "user", "content": prompt}],
                             max_tokens=256, temperature=0.0)
            if resp.error:
                continue
            parsed = _extract_json(resp.text)
            if parsed and all(k in parsed for k in
                              ("formula_correctness", "physical_meaningfulness", "canonicalness")):
                scores[j] = {
                    "correctness": float(parsed["formula_correctness"]),
                    "meaning": float(parsed["physical_meaningfulness"]),
                    "canonical": float(parsed["canonicalness"]),
                }
        # Aggregate
        if scores:
            mean_per_judge = [(s["correctness"] + s["meaning"] + s["canonical"]) / 3.0
                              for s in scores.values()]
            mean_score = float(np.mean(mean_per_judge))
            std_score = float(np.std(mean_per_judge))
        else:
            mean_score = 0.0
            std_score = 0.0
        out = dict(q)
        out["judge_scores"] = scores
        out["mean_score"] = mean_score
        out["score_std"] = std_score
        voted.append(out)
    logger.info(f"Voting complete: {len(voted)} candidates scored")
    return voted
