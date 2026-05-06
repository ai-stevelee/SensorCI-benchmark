"""LLM-as-Judge Stage 1: candidate generation.

Generates algebraic queries OR role-attribute sharing labels for a dataset
by calling K frontier LLMs with structured prompts.
"""

from __future__ import annotations
import json
import re
import logging
from typing import Any
import numpy as np

from sensorci.api_pool import get_default_pool, APIKeyPool
from sensorci.core.dataset import TSDataset

logger = logging.getLogger(__name__)


QUERY_GEN_PROMPT = """\
You are a senior {domain_expert} reviewing data from the {dataset_name} dataset.

Channels: {channel_list}
Sampling rate: {sampling_rate_hz} Hz
Number of fault classes / labels: {n_classes}

Task: Propose {n_queries} canonical "analyst queries" that an expert would
compute on this data to characterize system behavior. Each query is a
deterministic function f: signal -> scalar.

Constraints:
- Use ONLY linear, affine, or spectral operations on raw channels
- Each query must have physical meaning (cite the underlying principle)
- Provide expected value range for sanity checking

Output JSON list of {n_queries} items, each:
{{
  "query_id": "Q01",
  "formula_python": "np.mean(signal['{example_channel}'])",
  "physical_meaning": "...",
  "expected_range": [low, high],
  "linear_class": "linear" | "affine" | "spectral",
  "physical_law": "..."
}}

Respond with valid JSON only (no markdown, no extra text):
"""


SHARING_GEN_PROMPT = """\
You are a regulatory compliance expert reviewing the {dataset_name} dataset
({domain}). The data is shared with 6 stakeholder roles per EU Data Act 2024:

  Operator   — full access
  OEM        — equipment manufacturer; sees own product fault stats
  Vendor     — analytics software vendor; aggregate fault counts
  Insurer    — fleet-level annual statistics only
  Regulator  — safety incident statistics
  Public     — published industry aggregates

Attributes available:
{attribute_list}

For each (role, attribute) pair, classify the appropriate sharing label:
  S = Share (role may learn this attribute exactly)
  N = Not-share (role MUST NOT learn this attribute, even probabilistically)
  P = Partial (role may learn a generalized / coarse-grained version)

Apply Nissenbaum's contextual integrity: information flow is appropriate
when (sender, recipient, attribute, transmission principle, context) align.

Output JSON: {{"sharing_matrix": {{"<attribute>_<role>": "<S|N|P>", ...}},
              "rationale": "<one-sentence justification>"}}

Respond with valid JSON only:
"""


def _extract_json(text: str) -> dict | None:
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None


def generate_queries(
    dataset: TSDataset,
    generators: list[str] | None = None,
    n_per_generator: int = 30,
    pool: APIKeyPool | None = None,
    max_workers: int = 8,
) -> list[dict]:
    """Stage 1 query generation: K LLMs each propose N queries.

    Returns list of candidate query dicts (no filtering yet).
    """
    pool = pool or get_default_pool()
    generators = generators or ["gpt-4o-mini", "claude-haiku-4-5", "gemini-3-flash"]

    sample_seg = dataset.personas[0].segments[0]
    domain = dataset.dataset_metadata.get("domain", "industrial sensor")

    prompt = QUERY_GEN_PROMPT.format(
        domain_expert=f"{domain.split('(')[0].strip()} engineer",
        dataset_name=dataset.name,
        channel_list=", ".join(sample_seg.channels),
        sampling_rate_hz=sample_seg.sampling_rate_hz,
        n_classes=len(dataset.class_list),
        n_queries=n_per_generator,
        example_channel=sample_seg.channels[0],
    )

    candidates = []
    for gen_model in generators:
        logger.info(f"Generating {n_per_generator} candidates with {gen_model} ...")
        resp = pool.call(gen_model, [{"role": "user", "content": prompt}],
                         max_tokens=4096, temperature=0.7)
        if resp.error:
            logger.warning(f"  {gen_model} failed: {resp.error}")
            continue
        parsed = _extract_json(resp.text)
        if isinstance(parsed, list):
            queries = parsed
        elif isinstance(parsed, dict) and "queries" in parsed:
            queries = parsed["queries"]
        else:
            queries = []
        for i, q in enumerate(queries):
            if isinstance(q, dict) and "formula_python" in q:
                q["generator"] = gen_model
                q["query_id_global"] = f"{gen_model}_q{i:03d}"
                candidates.append(q)

    logger.info(f"Total candidates: {len(candidates)} from {len(generators)} generators")
    return candidates


def generate_sharing_matrix(
    dataset: TSDataset,
    generators: list[str] | None = None,
    pool: APIKeyPool | None = None,
) -> list[dict]:
    """Stage 1 sharing-matrix generation. Each LLM proposes one full matrix."""
    pool = pool or get_default_pool()
    generators = generators or ["gpt-4o-mini", "claude-haiku-4-5", "gemini-3-flash"]

    attrs = dataset.attribute_set()
    attr_list = "\n".join(f"  - {a}" for a in sorted(attrs))
    prompt = SHARING_GEN_PROMPT.format(
        dataset_name=dataset.name,
        domain=dataset.dataset_metadata.get("domain", "industrial"),
        attribute_list=attr_list,
    )

    proposals = []
    for gen_model in generators:
        logger.info(f"Generating sharing matrix with {gen_model} ...")
        resp = pool.call(gen_model, [{"role": "user", "content": prompt}],
                         max_tokens=4096, temperature=0.3)
        if resp.error:
            logger.warning(f"  {gen_model} failed: {resp.error}")
            continue
        parsed = _extract_json(resp.text)
        if parsed and "sharing_matrix" in parsed:
            proposals.append({
                "generator": gen_model,
                "sharing_matrix": parsed["sharing_matrix"],
                "rationale": parsed.get("rationale", ""),
            })
    return proposals
